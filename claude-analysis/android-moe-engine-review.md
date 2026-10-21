# android-moe-engine: code review and optimization plan for the Snapdragon 665

Reviewed commit `a1d4cc7` (main, October 2026), the llama.cpp patches in `patches/`, the pinned llama.cpp ARM kernels at `ff5a2337`, and the retained run records under `evidence/`. Every number below is either read from your evidence files or derived from them with the arithmetic shown. Estimates are labelled as estimates; none of them replaces a controlled measurement on the phone.

---

## TL;DR

1. **Decode is roughly 85–90% synchronous storage I/O.** Your own runs put decode at about 2.3 s/token, of which about 2.0 s is `pread` and about 0.3 s is everything else. Kernel work cannot move the needle until the I/O is parallel and overlapped.
2. **The project cache cannot hit by construction.** One 64-slot LRU is shared by all 120 routed tensors, while one decode token touches 960 distinct expert slices. Every retained 64-slot run reports `hits: 0`. The kernel page cache is doing whatever caching happens, and the arena duplicates it.
3. **Reads are queue-depth 1 and run inside the operator while the other three worker threads spin at a barrier.** No read overlaps compute, and no two reads overlap each other.
4. **Three correctness/robustness bugs** deserve fixing before any performance work: a sticky error latch in the ggml patch that wedges the process after one materializer failure, an unsynchronized `/health` read of streaming state, and a prefill admission bound (256) that exceeds the lease capacity (128) as soon as slots exceed 128.
5. **Speculative decoding does not help on this configuration.** Verifying k draft tokens loads the union of their experts, and your prompt traces show that union grows almost linearly. Break-even needs roughly 60–80% acceptance before counting draft cost. Your GGUF has no MTP head, but an MTP variant of the same model exists, which makes drafting nearly free; the verify-side union is still the problem (section 6.2 and 8.4). Revisit after the cache works. *Speculative prefetch* (router stays authoritative) is the speculation that pays here.
6. **Kernel work is worth doing second.** The CPU reports only `fp asimd … sha2 crc32`: no dot-product, no i8mm, no fp16 arithmetic. llama.cpp's fallback for `vdotq_s32` spends about 13 NEON ops where 5 suffice for Q4_K, and the result stays bit-identical. Prefill benefits most, especially once the tiled MUL_MAT_ID path is re-enabled for staged tensors.

Section 8 surveys current papers and repositories, including an open llama.cpp PR (#25294) that already implements much of sections 4.2–4.6.

A back-of-envelope target after the I/O work in sections 4.1–4.4: about 0.6–0.8 s/token (roughly 1.2–1.7 tok/s) versus about 2.3 s/token today. That is an estimate built on assumed storage queue-depth scaling and the 35% hit rate from your 1024-slot run. Measure before trusting it.

---

## 1. What this phone actually offers

| Resource | What the repo records | What it means for the engine |
|---|---|---|
| CPU | SM6125 (Kryo 260): 4× Cortex-A73 + 4× Cortex-A53. Probe flags: `fp asimd evtstrm aes pmull sha1 sha2 crc32 cpuid` (`docs/DESIGN_DECISION.md`) | ARMv8.0 baseline. No `asimddp` (SDOT), no `i8mm`, no `asimdhp`. llama.cpp's repacked Q4_0 4×4/4×8 paths are compiled out because they require dotprod or i8mm. |
| RAM | 7.8 GB total, about 5.3 GB available before load (`configs/device-vivo-s1-pro.json`); about 2.0 GiB server RSS, 84 MB already swapped (`evidence/api/qwen36-memory.txt`) | Roughly 2.5–3 GB is usable for caching experts, but it is currently left to the page cache. Swap activity means memory pressure is already present. |
| Storage | Not measured (`storage_throughput_bytes_per_second: null`). Observed effective expert-read throughput is 150–355 MB/s at QD1 across your 35B runs | Unknown headroom. A queue-depth sweep is the first experiment (4.0). |
| Kernel/OS | Android 12, locked bootloader, no root, ADB shell UID 2000 | You cannot change the I/O scheduler, `read_ahead_kb`, cpufreq governor, or `RLIMIT_MEMLOCK`. You can use `posix_fadvise`, `madvise`, `sched_setaffinity`, thread priorities, and `/proc/self/io`. An SD665 Android 12 build most likely runs a 4.14/4.19 kernel, which predates `io_uring` (5.1); check `uname -r`. |
| Model | `qwen35moe`: 40 layers, `full_attention_interval 4` (10 attention + 30 recurrent layers), 256 experts, top-8, one shared expert, `n_embd 2048`, expert FFN 512. Every routed slice is Q4_K, 589,824 bytes (`evidence/phase4/qwen36-35b-a3b-q4km-index.json`) | Per decode token: 40 layers × 3 tensors × 8 experts × 589,824 B = **566 MB of routed expert reads** with zero cache hits. About 2.29 GB of non-expert weights stay mmapped. |

---

## 2. Where the time goes (from your evidence)

`phase4/qwen36-stream-07.json` and `-08.json` are the cleanest pair. They share the same prompt (5 tokens), the same output (8 tokens), and the same build, and differ only in slots.

| Run | Slots | Misses | Hits | Read bytes | `read_seconds` | Effective MB/s | Prefill s | Generate s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stream-07 | 64 | 9,999 | 0 | 5.90 GB | 21.4 | 276 | 12.3 | 28.7 |
| stream-08 | 1,024 | 7,650 | 2,349 | 4.51 GB | 21.6 | 209 | 12.8 | 28.0 |

Derivation for stream-07:

- Decode makes 7 graph evaluations × 120 routed tensors × 8 experts = 6,720 lookups, all misses. The remaining 3,279 misses are prefill: 27.3 unique experts per tensor for a 5-token prompt (max union recorded: 37 of a possible 40).
- Decode I/O: 6,720 × 589,824 B = 3.96 GB. At 276 MB/s that is about 14.4 s of the 16.4 s decode phase, which leaves **≈0.29 s/token of non-I/O work**.
- Prefill I/O: 1.93 GB at 276 MB/s is about 7.0 s, leaving about 5.3 s of compute for 5 tokens (≈1 s/token). Prefill is compute-heavy for reasons covered in 3.8 and section 5.

What stream-08 tells you: 1,024 slots produced a **35% decode hit rate** (2,349 of 6,720 decode lookups) and 23% fewer bytes, yet wall time did not improve and effective MB/s dropped. The most likely explanation is that at 64 slots the kernel page cache already served many of the "misses" at memcpy speed, and a 604 MB arena took that memory away from the page cache. This is a hypothesis; `/proc/self/io` settles it (4.0).

Run-to-run spread is large. stream-01..09 are identical requests ranging from 18.9 to 36.8 s, which points to thermal or background-load variance. Record `scaling_cur_freq` for cpu4–7 alongside every run.

---

## 3. Code review findings

### 3.1 High: the error latch in the ggml patch is never cleared (process wedges after one failure)

`patches/0001-ggml-moe-materializer.patch` adds `g_moe_error_latched`. Thread 0 sets it inside `ggml_compute_forward_mul_mat_id`, and `ggml_graph_compute_thread` checks it **after every node**:

```c
if (state->ith == 0 && (g_moe_error_latched ||
        (cplan->abort_callback && cplan->abort_callback(cplan->abort_callback_data)))) {
    atomic_store_explicit(&tp->abort, node_n + 1, memory_order_relaxed);
```

The only place that writes 0 is the next `MUL_MAT_ID`. After a FAILED or CANCELLED outcome, the next graph aborts after its first node, before any MoE op can reset the latch. Every later request in the process fails with "decode failed", and `g_stats.failed` stays true as well. A restart costs a full-artifact rehash (174–404 s).

Fix: export `void ggml_moe_reset(void)` from the patch that zeroes the latch and outcome, call it from `release_moe_stream_leases()` (and reset `g_stats.failed/cancelled/error` there). Add a server test that injects an I/O failure on request N (a fault hook that can be switched off between requests, since `AME_IO_FAIL_AFTER_BYTES` is cumulative per source) and asserts request N+1 succeeds. Your fresh-after-failure evidence uses a new process, so it does not cover this path.

### 3.2 High: `/health` reads streaming state without synchronization

`apps/server/main.cpp` `/health` calls `moe_stream_stats()` from an httplib worker thread. That function copies `g_stats` (which contains several `std::string` fields) and iterates `g_caches` (a `std::map`) while the generation thread may be inside `materialize()`, which writes `g_stats.error` and can `emplace` into `g_caches` on first use. Your queue evidence (`qwen36-queue-health-during.json`) polls `/health` mid-generation, so this path runs in practice. It is undefined behaviour even if it has not crashed yet.

Fix: make the counters `std::atomic<uint64_t>`, and have the generation thread publish a snapshot under a mutex at the end of each `materialize()` call (or each token). `/health` should read only the snapshot.

### 3.3 High: prefill admission can exceed lease capacity once slots > 128

`kMaxLeaseExperts = 128` (`src/residency/store.hpp:18`) bounds one lease, but the server passes `lease_limit = 256` into `plan_model_prefill` (`apps/server/main.cpp:326`), which overwrites the bound captured from `kMaxLeaseExperts`. Today slots=64 caps the unit at 8 tokens, so the mismatch is latent. With `AME_SLOTS=256` or more, the plan admits 32-token units whose worst-case union (256) exceeds 128, and `materialize()` fails with "expert union exceeds lease capacity". Combined with 3.1, this takes the server down.

Fix: pass `ame::residency::kMaxLeaseExperts` (or a `min` of the two) and add a unit test that asserts `unit_tokens × expert_used ≤ kMaxLeaseExperts`. If you want larger prefill units, raise `kMaxLeaseExperts` to 256, since it is only a fixed-size array of pins.

### 3.4 Medium: deadlines and cancellation are only checked between tokens or chunks

The 300 s execution deadline is checked at `apps/server/main.cpp:358` and `:396`, between prefill chunks and between decode steps. Nothing calls `request_moe_stream_cancel()` from the server, so a single decode step (about 2–4 s) or prefill chunk (about 10 s) runs to completion regardless of deadline or client disconnect. This matches the near-context request blowing through 300 s. Once 3.1 is fixed, wire a watchdog that calls `request_moe_stream_cancel()` on deadline or disconnect; the store already checks cancellation between expert reads.

### 3.5 Performance: one LRU for 120 tensors guarantees zero hits at 64 slots

`cache_for()` keys caches by `expert_bytes`. Every routed slice is 589,824 B, so all 120 tensors share **one** store with `g_slots` = 64 entries. A decode token touches 960 distinct (tensor, expert) slices in a fixed cyclic order (layer 0 gate/up/down → layer 39). An LRU smaller than the cycle evicts every entry before it is reused, which is why every 64-slot run shows `hits: 0` and the 36 MiB arena behaves as a staging buffer only.

Section 4.3 covers the policy choice. The short version: either let the page cache be the cache and keep the arena small, or make the arena the cache with capacity above ~1,000 slots and stop the page cache from holding a second copy.

### 3.6 Performance: reads are serial (QD1) and block the whole thread pool

`Store::acquire` (`src/residency/store.cpp:293–315`) reads misses one at a time with `pread`. It runs on ggml thread 0 inside `ggml_compute_forward_mul_mat_id`, while threads 1–3 wait at `ggml_barrier`, which spins. Three consequences follow:

- Flash never sees more than one outstanding request, so UFS/eMMC command queueing is unused.
- No read overlaps any compute.
- Three worker threads (unpinned, so usually but not always on the A73s) spin for about 2 s of every 2.3 s token. That burns power and heats the SoC, which plausibly contributes to the run-to-run spread and will matter for the two-hour campaign you have not run yet.

### 3.7 Performance: the page cache and the arena hold the same bytes twice

`FileSource::open*` uses buffered `pread` with no `posix_fadvise`. Each expert read goes flash → page cache → `memcpy` → arena. With default readahead, each 576 KiB random read may also pull in neighbouring pages that are never used. The page cache and the arena compete for the same ~3 GB of free RAM, which is the most plausible reason stream-08 did not get faster.

### 3.8 Performance: the tiled MUL_MAT_ID path is disabled for every staged tensor

The patch skips `ggml_compute_forward_mul_mat_id_tiled` whenever `g_moe_outcome == GGML_MOE_STAGED`, because the tiled function reads `dst->src[0]` directly. Every routed matmul therefore uses the per-row `vec_dot` path. For decode (one token) this costs little; for prefill it is likely a large part of the ≈1 s/token compute. The fix is mechanical: pass `src0_eff` into the tiled function (add a parameter or a staged-tensor override), then re-run `moe_kernel_gate` and `moe_quantized_gate` for bitwise parity.

### 3.9 Performance: a new `llama_context` per request; no prefix reuse

`generate()` calls `llama_init_from_model` per request (`apps/server/main.cpp:339`), so a multi-turn chat re-prefills the entire history every turn. At roughly 1–2 s per prompt token, a 10-turn conversation spends most of its time re-reading history. Hybrid recurrent layers mean you cannot truncate state cheaply, but you can keep one live context and, when the new request's token prefix equals the previous conversation, decode only the new suffix. Fall back to a fresh context on any mismatch.

### 3.10 Minor

- `getenv()` runs in hot paths: three times per chunk in `FileSource::read_exact`, and once per `materialize()` for `AME_MOE_TRACE_ONLY`. Read them once at install, or compile the fault-injection hooks out of release builds.
- `g_artifact->recheck()` calls `fstat` on every MUL_MAT_ID (120 times per token). It is cheap, but checking once per graph is enough.
- The routing trace calls `flush()` per call. That is fine for diagnostics, but buffer it if you leave tracing on during timing runs.
- `Store::acquire` scans every slot to find each victim (`store.cpp:229`). At 64 slots this is fine; at 2,000–4,000 slots use an intrusive LRU list or a CLOCK hand.
- Arena alignment is 64 bytes. If you ever try `O_DIRECT`, buffers and file ranges must be 4 KiB-aligned (and GGUF expert offsets are only 32-byte aligned, so reads would need padding).
- `Hasher::update` copies every 64-byte block into an internal buffer and calls `compress` once per block, and the ARM path reloads and stores the state vectors each call. Read and hash are also serialized. This likely explains why the accelerated hash only reaches ~117 MB/s (20.4 GB in 174 s); see 4.8.

### 3.11 What is good

- The verification discipline is unusually strong: artifact identity via retained FD + `fstat` recheck, a layout-hashed manifest, source-derived consumer contracts, and a fail-closed materializer. Most hobby inference engines have none of this.
- The patch to llama.cpp is small and keeps routing, arithmetic, and accumulation untouched, which keeps parity questions tractable.
- Disabling `MAP_POPULATE` in streaming mode (patch 0002) was the right call (RSS 5.2–6.1 GiB → about 1.2 GiB on the 9.5 GB model).
- Evidence files carry enough counters (`read_seconds`, `misses`, `max_union`) to do the analysis in this document without access to the phone. Keep that habit.
- Fault-injection hooks (`AME_IO_FAIL_AFTER_BYTES`, EINTR injection, cancel-after-chunks) and the ASan/UBSan presets exist; 3.1 just needs a test that uses them across two requests.

---

## 4. Optimization plan for this hardware

Ordered by expected gain per unit of effort. Every item keeps the router authoritative and changes no arithmetic, so your parity gates still apply unchanged.

### 4.0 Measure first (no code changes to the engine)

These four measurements decide between the options in 4.1–4.3.

1. **Page-cache hit ratio.** Sample `/proc/self/io` before and after each token. `rchar` counts bytes your process asked `pread` for; `read_bytes` counts bytes that actually came from the block device. `1 − read_bytes/rchar` is the page-cache hit ratio the project counters cannot see. Add both fields to the run record.
2. **Storage queue-depth sweep.** Add a small `bench/io_qd.cpp` that issues random 589,824-byte `pread`s at real expert offsets from 1, 2, 4, and 8 threads and reports MB/s. You cannot drop caches without root, but `posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` on your own fd evicts that file's clean pages, which is enough for a cold-read benchmark. If QD8 is not clearly faster than QD1, skip 4.2 and concentrate on caching.
3. **Where compute goes.** `simpleperf` from the NDK works for shell-owned processes without root: `simpleperf record -g -p $(pidof ame-server) --duration 30`, then `simpleperf report`. You want the split between `ggml_vec_dot_q4_K_q8_K`, the Q6_K/Q8_0 dots (non-expert weights and output head), the recurrent-layer ops, and barrier spinning.
4. **Clocks and faults.** Log `/sys/devices/system/cpu/cpu{4..7}/cpufreq/scaling_cur_freq` and `majflt` from `/proc/self/stat` every second. Major faults during decode mean non-expert mmapped weights are being evicted and re-read from flash, which `read_seconds` does not count.

Then run the replay script in the appendix over an `AME_MOE_TRACE` capture of a few hundred **generated** tokens. It gives the hit-rate curve for cache sizes and policies, the same-layer temporal locality, and the union growth that section 6 depends on, all offline.

### 4.1 Cheapest overlap: `posix_fadvise` hints (a few dozen lines)

Gate, up, and down for one layer use the **same** `selected_experts` ids. When `materialize()` sees the first routed tensor of layer L, it already knows exactly which 16 slices the other two tensors will need. Ask the kernel to start reading them before you `pread` the current tensor:

```cpp
// In materialize(), after computing `unique` for spec (layer L, role R):
for (const TensorSpec* sibling : g_layer_siblings[spec.layer]) {   // the other two roles
  if (sibling == &spec || g_hinted[spec.layer][sibling->table_id]) continue;
  for (uint32_t i = 0; i < unique_count; ++i) {
    const uint64_t off = sibling->offset + uint64_t(unique[i].expert) * sibling->stride;
    ::posix_fadvise(g_hint_fd, off, sibling->expert_bytes, POSIX_FADV_WILLNEED);
  }
  g_hinted[spec.layer][sibling->table_id] = true;   // reset per graph
}
```

This gives the device several outstanding requests and lets later reads overlap the current tensor's compute, without threads. Also try `POSIX_FADV_RANDOM` on the **read** descriptor to stop speculative readahead around each 576 KiB read. `FADV_RANDOM` sets per-open-file readahead to zero, and on some kernels that also limits `WILLNEED`, so open a **separate** descriptor (`g_hint_fd`) for hints rather than a `dup`. Measure both settings; kernel behaviour here varies by version.

### 4.2 Parallel demand reads (queue depth > 1)

If the 4.0 sweep shows QD scaling, replace the serial loop in `Store::acquire` with a submit-and-wait over a small reader pool. `pread` on a shared descriptor is thread-safe, and slots are already reserved and pinned before any read starts, so the existing rollback logic still works.

```cpp
// Store::acquire, replacing the loop at store.cpp:293
ReadBatch batch;                                   // atomic remaining + first error
for (uint32_t i = 0; i < loading_count; ++i)
  g_readers.submit({source_.fd(), entry_offset(i), slot_span(i), &batch, &stop});
batch.wait();                                      // futex/condvar, not a spin
if (batch.error()) { rollback(); return batch.error(); }
```

Pin two to four reader threads to the A53 cores (cpu0–3 on most SM6125 builds; confirm with `cpuinfo_max_freq`). The `memcpy` out of the page cache then happens off the big cores. Keep the cancellation check per read.

### 4.3 Decide who owns the cache

Today both the page cache and the arena hold expert bytes, and the 64-slot arena never hits. Pick one owner.

**Option A: the page cache is the cache (recommended first).** Keep a small arena as a staging buffer (64–128 slots), add the 4.1 hints, and leave the rest of RAM to the kernel. Page-cache pages are clean and file-backed, so under memory pressure Android simply drops them. Anonymous arena pages are instead compressed into zram (you already have 84 MB swapped), and Q4 weights compress poorly, so zram costs CPU and saves little. The downside is that the kernel's active/inactive LRU knows nothing about routing.

**Option B: the arena is the cache.** Raise slots to about 1,500–4,000 (0.9–2.4 GB), call `POSIX_FADV_DONTNEED` on each extent right after copying it into a slot so the page cache does not keep a second copy, and use a routing-aware policy:

- A per-layer budget (or segmented LRU) so the 40-layer cycle cannot flush everything.
- Frequency-based pinning: from traces over a calibration set, permanently pin the most frequently routed experts per layer. MoE routers are usually skewed, but measure the skew for this model before relying on it.
- An intrusive LRU list or CLOCK instead of the O(slots) victim scan.

Your stream-08 run already shows a 35% decode hit rate at 1,024 slots with plain LRU. Under Option B that should translate into real time saved, once the page cache stops competing for the same RAM. Use the replay script to compare A and B before writing B. Watch `lmkd`: a multi-GB anonymous allocation in a shell process can be killed or pushed into zram under pressure.

### 4.4 Take I/O out of the barrier

Even with 4.1–4.3, reads that happen inside `ggml_compute_forward_mul_mat_id` leave three worker threads spinning at `ggml_barrier`. Two changes fix that:

1. **Observe routing at `ffn_moe_topk` with the eval callback.** Set `cb_eval` in `llama_context_params` and ask for the `ffn_moe_topk-<il>` node. The scheduler then splits the graph there, runs your callback with the ids computed and the worker pool parked, and you submit reads for all three tensors of that layer. Your MUL_MAT_ID hook then only *waits* for its own tensor, so gate compute overlaps the up/down reads. This is the structure BigMoeOnEdge uses (your `research.md`, "RouterHook"). It costs about 40 extra graph splits per token; measure that overhead.
2. **Own the threadpool.** Create it with `ggml_threadpool_params` (cpumask for the four A73 cores, `strict_cpu`, a low `poll` value) and attach it with `llama_attach_threadpool`. That pins compute to the big cluster and stops idle workers from spinning between graph splits.

### 4.5 Cross-layer prefetch (exact; the next phase after parity)

`AGENTS.md` defers speculative prefetch until CPU inference is solid. That stage now looks reached, and your timing shows prefetch is the largest remaining lever. It stays exact because prefetch only issues hints, the real router still selects every expert, and a wrong guess falls back to demand loading.

Two predictors, in order of effort:

- **Same-layer temporal:** prefetch the experts this layer used for the previous token. You can measure its precision from existing traces with no new instrumentation (the appendix prints it).
- **Next-layer pre-gating:** apply layer L+1's router (`ffn_gate_inp`, 2048×256, tiny) to layer L's hidden state and prefetch its top-8 (or top-12, to buy recall at the cost of extra bytes). The residual stream changes slowly between layers, which is why this works in the pre-gated MoE literature, but its precision for this model is unknown. Measure it before building it.

On this model, 30 of 40 layers are recurrent (linear-attention) layers whose compute is short, so there is less compute per layer to hide reads behind than in a full-attention model. That makes the prediction's lead time (one layer ahead versus one token ahead) more important.

### 4.6 Prefill: bigger units and the tiled kernel

- Fix 3.3, then set slots to 128. The arena is 75.5 MB and the unit grows from 8 to 16 tokens with the current lease size. Raising `kMaxLeaseExperts` to 256 with 256 slots (151 MB, still inside your 256 MiB allowance) allows 32-token units. Every expert read during prefill is then shared by more tokens; your 5-token trace already averages 5.5 slices per token per tensor versus 8 for decode, and larger units amortize further. Quantify with the replay script.
- Fix 3.8 (tiled path for staged tensors). Prefill is where batch matmul kernels matter.
- Implement 3.9 (prefix reuse). For chat, the cheapest prefill token is the one you never recompute.

### 4.7 Repack experts so one read fetches a whole expert

GGUF stores `ffn_gate_exps`, `ffn_up_exps`, and `ffn_down_exps` as three separate 151 MB tensors per layer, so one expert means three 576 KiB reads at distant offsets. A repacked artifact that stores `[gate|up|down]` contiguously per (layer, expert) turns that into one 1.77 MB read. Staging stays simple, because each role becomes a view into the same slot with `nb[2] = 1,769,472` and a role offset. Your manifest already carries per-tensor stride, so most of the plumbing exists. The bytes of every tensor are unchanged, but the file is new, so it needs its own enrollment digest. Do this only if the 4.0 sweep shows large reads beating several small concurrent ones; on queued flash they may be close. Note the distinction from co-activation *reordering* (permuting experts so co-firing ones sit together): a llama.cpp community experiment found that only pays when slices are in the 16–128 KiB range, and your slices are 576 KiB. Per-expert contiguous packing is what Swiftlet's iOS container does (section 8.1).

### 4.8 Startup: from 3–7 minutes to seconds

1. Fix the hasher. Process many 64-byte blocks per `compress` call with the state held in NEON registers, hash directly from the read buffer when no partial block is pending, and double-buffer so the next 4 MB `pread` runs while the current one is hashed. Hardware SHA-256 on an A73 core should run well above the ~117 MB/s you measured; confirm with a no-I/O microbenchmark.
2. Replace the single whole-file digest with a **per-extent digest table** in a manifest v3 (one SHA-256 per expert slice and per non-expert tensor), and pin the manifest's own digest. At startup, verify only the manifest. Verify each expert slice the first time it is read and keep a "verified" bitmap, so steady-state cost is zero. Let a low-priority verifier thread on the A53s cover the rest in the background, and report progress in `/health`. This also closes part of your "later writes not authenticated per extent" gap.

### 4.9 Threads, affinity, and heat

- Confirm which CPU IDs are the A73s (`cpuinfo_max_freq`), and pin compute there (4.4). Put readers and the verifier on the A53s.
- With compute at about 0.3 s/token, try 3 compute threads; leaving one big core free can help if the page-cache `memcpy` competes with compute.
- Always log cluster frequencies. A passively cooled phone throttles the big cluster within minutes of sustained load, and the 2× spread across your identical runs suggests this is already happening.

### 4.10 Keep the non-expert weights resident

About 2.29 GB of non-expert weights are file-backed mmap pages. Under pressure the kernel drops them, and the next token re-faults them from flash in small random reads that never show up in `read_seconds`. If 4.0 shows major faults during decode, issue `madvise(MADV_WILLNEED)` on those ranges periodically, or shrink other memory use. `mlock` is not an option because a shell process has a tiny `RLIMIT_MEMLOCK`.

---

## 5. Compute kernels on an ARMv8.0 CPU without dot-product

### 5.1 What runs today

Without `__ARM_FEATURE_DOTPROD`, the pinned llama.cpp emulates `vdotq_s32` (`ggml/src/ggml-cpu/ggml-cpu-impl.h:310`):

```c
inline static int32x4_t ggml_vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    const int16x8_t p0 = vmull_s8(vget_low_s8 (a), vget_low_s8 (b));
    const int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    return vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
}
```

The Q4_K inner loop (`arch/arm/quants.c`, `ggml_vec_dot_q4_K_q8_K`, NEON branch) calls it twice per 32 products and then reduces with `vaddvq_s32`. That is about 13 NEON operations for 32 multiply-accumulates, before unpacking and scales. The generic macro has to work this way because two full-range `int8×int8` products (up to 16,384 each) can overflow an `int16` lane when added.

### 5.2 A Q4_K-specific inner loop (bit-identical)

Q4_K weights are nibbles in 0..15 and Q8_K activations are in −128..127, so each product is at most 1,920 in magnitude. Four products per `int16` lane stay under 7,680, far from overflow, so the whole 32-product dot fits in one `int16x8` accumulator:

```c
// replaces: p1 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, q4[0], q8[0]), q4[1], q8[1]);
//           sumi1 += vaddvq_s32(p1) * scales[2*j+0];
int16x8_t acc = vmull_s8(vget_low_s8 (q4bytes.val[0]), vget_low_s8 (q8bytes.val[0]));
acc = vmlal_s8(acc,      vget_high_s8(q4bytes.val[0]), vget_high_s8(q8bytes.val[0]));
acc = vmlal_s8(acc,      vget_low_s8 (q4bytes.val[1]), vget_low_s8 (q8bytes.val[1]));
acc = vmlal_s8(acc,      vget_high_s8(q4bytes.val[1]), vget_high_s8(q8bytes.val[1]));
sumi1 += vaddlvq_s16(acc) * scales[2*j+0];   // |lane| <= 4 * 15 * 128 = 7680
```

That is 5 operations instead of about 13 for the same integer sum. Integer addition is exact in any order, so the output is **bit-identical**, and your `moe_kernel_gate` / `moe_quantized_gate` bitwise checks remain the right acceptance test. The same idea applies to Q6_K (values −32..31, products up to 4,096, so keep it to at most 7 products per lane; 4 is safe) and to Q5_K. Guard the new path with `#if defined(__ARM_NEON) && !defined(__ARM_FEATURE_DOTPROD)` so dot-product builds are untouched. Because it relies on Q4_K's value range rather than general int8, it is a reasonable upstream patch for all ARMv8.0 devices.

Unpacking nibbles, loading scales, and memory traffic are unchanged, so the whole-kernel gain is smaller than the instruction count suggests. Measure it with a single-thread microbenchmark before and after, on an A73 core at a fixed frequency.

### 5.3 Where kernel work pays, and where it does not

- **Prefill benefits most.** It is compute-heavy (≈1 s/token in your runs), currently runs without the tiled path (3.8), and benefits directly from the cheaper inner loop.
- **Decode benefits less.** Its ≈0.3 s/token of non-I/O work includes streaming most of the ~2.3 GB of non-expert weights through DRAM every token. SD665's memory system (LPDDR4X, dual 16-bit channels, ~15 GB/s theoretical) puts a floor under that which no ALU change removes. Faster kernels matter for decode only after sections 4.1–4.4 have hidden the I/O.
- **Profile the output head.** A vocabulary in the ~250k range at 2048 hidden is hundreds of MB per decode token. `simpleperf` (4.0) will show whether it is one of the top items; if it is, it needs the same treatment as Q4_K.
- **Fuse gate and up.** Both MUL_MAT_IDs share the same input row and expert ids. A fused op quantizes the input and groups rows once, and reads gate and up weights in one pass. This matters most after 4.7 places them next to each other.

### 5.4 What will not help on this chip

- llama.cpp's repacked Q4_0 4×4 / 4×8 layouts are compiled only with dot-product or i8mm, so they are unavailable here.
- FP16 arithmetic (`asimdhp`) is absent, so there are no fp16 compute shortcuts.
- Lower-bit expert quantization (for example Q3_K experts) would cut bytes by about a quarter, but it produces a different model with different outputs. Treat it as a separate, explicitly approximate configuration, not an optimization.

---

## 6. Does speculative decoding help here?

**Short answer: not in the current configuration. It roughly breaks even at best and loses at typical acceptance rates. It becomes interesting only after the expert cache works.**

### 6.1 Why the usual argument fails for streamed MoE

Speculative decoding pays off when verifying k+1 tokens costs about the same as decoding one, because a dense model reads the same weights for every token. Here the dominant cost is reading the *experts each token selects*, and different tokens select different experts. Verifying a window of k+1 tokens means loading the **union** of their experts.

Your 5-token prompt traces give 27.3 unique experts per tensor versus 8 for one token: 3.4× the bytes for 5× the tokens, so the union grows almost linearly. Interpolating linearly between U(1)=8 and U(5)=27.3:

| Draft length k | Tokens verified | Est. union per tensor | Expert bytes vs. 1 token | Break-even acceptance α |
|---:|---:|---:|---:|---:|
| 1 | 2 | ≈12.8 | ≈1.6× | ≈0.60 |
| 2 | 3 | ≈17.7 | ≈2.2× | ≈0.71 |
| 3 | 4 | ≈22.5 | ≈2.8× | ≈0.77 |

Speedup ≈ (expected tokens per verify) ÷ (bytes ratio), where expected tokens = 1 + α + … + α^k for greedy verification:

| k | α = 0.5 | α = 0.7 | α = 0.9 |
|---:|---:|---:|---:|
| 1 | 0.94× | 1.06× | 1.19× |
| 2 | 0.79× | 0.99× | 1.23× |
| 3 | 0.67× | 0.90× | 1.22× |

These are upper bounds, because they ignore draft cost, rejected-branch work, and state rollback. They also rest on prompt-token traces; generated tokens may show more locality, so confirm with the appendix script on a decode trace.

### 6.2 Extra obstacles specific to this setup

- **No MTP head in *this* artifact, but one exists.** The GGUF metadata in your index has no `nextn`/MTP keys. However, Qwen3.6 ships an MTP head, llama.cpp merged MTP support in May 2026 (PR #22673), and ggml-org publishes `Qwen3.6-35B-A3B-MTP-GGUF` (40 trunk layers + 1 MTP layer). With it, a draft token costs one extra MoE layer (about 8 × 3 × 589,824 B ≈ 14 MB of expert reads, roughly 2.5% of a full token), so draft cost is not the obstacle. Reported acceptance on resident setups is around 75% at 3 drafts; by the table above that is roughly 1.1× at k=1 and break-even at k=3 on this phone. Check whether your pinned llama.cpp commit predates #22673 before trying it; switching artifacts also means re-enrolling the digest.
- **Recurrent state rollback.** 30 of 40 layers carry linear-attention state (by the metadata, about 32 × 128 × 128 floats ≈ 2 MiB per layer, roughly 60 MiB total). Rejected drafts require restoring it. Copying 60 MiB is cheap next to a 2 s token, but llama.cpp cannot partially remove positions from recurrent memory, so you need save/restore around every verify step. Your `research.md` already lists this as unproven.
- **Prefill compute grows.** A k+1 window also adds MoE compute, which is not free at ≈1 s/token for batched work today.

### 6.3 When it could still win

- **Opportunistic prompt lookup (n-gram drafts).** No draft weights; draft only when a long suffix match exists (quoting the prompt, editing code, repeating JSON keys), with k = 1–2. When it fires, acceptance is often high, which is the only regime in the table above that wins. Keep it greedy-only and report target bytes per accepted token.
- **After the cache works.** Once most experts in a verify window are already resident (4.3, 4.5), the bytes ratio drops toward 1 and the classic argument returns. Rerun the analysis then.

### 6.4 The speculation that does pay: speculative prefetch

Predicting *which experts* the next layer or token will need (4.5) costs nothing when wrong except some wasted bandwidth, cannot change outputs because the router still decides, and attacks the 85–90% I/O share directly.

---

## 7. Suggested order and expected effect

| # | Step | Effort | Expected effect | Output-exact? |
|---:|---|---|---|---|
| 1 | Fix 3.1 latch, 3.2 `/health` race, 3.3 admission bound | Small | Server survives failures; slots can be raised safely | Yes |
| 2 | Measurements in 4.0 + trace replay | Small | Decides between 4.2 and 4.3 options | n/a |
| 3 | `fadvise` hints, separate hint fd (4.1) | Small | More outstanding I/O and some overlap; size depends on QD sweep | Yes |
| 4 | Pick cache owner (4.3) | Medium | ~25–35% fewer flash bytes per decode token (stream-08: 35% hits at 1,024 slots) | Yes |
| 5 | Reader pool on A53s (4.2) | Medium | Up to the device's QD scaling | Yes |
| 6 | Router hook at `ffn_moe_topk` + pinned threadpool (4.4) | Medium | Hides ≈0.3 s/token compute; stops barrier spinning; less heat | Yes |
| 7 | Prefill: slots 128–256, tiled path, prefix reuse (4.6, 3.8, 3.9) | Small–medium | Fewer bytes and less compute per prompt token; multi-turn turns stop re-prefilling | Yes (tiled path: verify with gates) |
| 8 | Q4_K/Q6_K no-dotprod inner loop (5.2) | Small | Lower prefill compute; modest decode gain | Bit-identical |
| 9 | Cross-layer / temporal prefetch (4.5) | Larger | Depends on measured predictor precision | Yes |
| 10 | Hasher fix + per-extent verification (4.8) | Medium | Startup from minutes to seconds | Yes |
| — | Speculative decoding | Defer | ≈0.7–1.2× now (section 6) | Greedy only |

**Rough end state after steps 1–7:** at 35% hits, decode reads about 368 MB/token. If queue depth doubles effective throughput to ~550 MB/s, that is about 0.67 s of I/O, with the 0.3 s of compute hidden behind it, so roughly 0.6–0.8 s/token. If queue depth does not scale, expect about 1.3–1.4 s/token, still about 1.7× better than today. Both figures are estimates from your recorded numbers and assumptions stated above.

---

## 8. Research and implementation landscape (October 2026)

This section covers work your `research.md` does not already review (it covers EdgeMoE, LLM in a flash, PowerInfer-2, FlashMoE, MobileMoE, BigMoMo, Paging the Experts/Routide, BigMoeOnEdge, mllm, MNN, LiteRT-LM, and AirLLM). I read abstracts, PR threads, and model cards, not full papers or code, so treat every number here as the authors' claim on their hardware. Almost none of it was measured on a phone-class CPU without dot-product.

### 8.1 Implementations to study

| Project | What it does | What to take from it |
|---|---|---|
| **llama.cpp PR #25294** "stream MoE routed experts from disk" (open since July 2026, awaiting maintainer review) | Per-layer cache of expert slots; a CPU id-remap op after router top-k maps expert ids to slots; misses load through a `pread` worker pool; optional `O_DIRECT` with a buffered fallback; eviction by decaying "route hotness" with an LRU tie-break; wave-partitioned prefill so each touched expert is loaded once per ubatch. Author reports bit-exact output under matched kernels and ubatch. | Almost exactly the structure sections 4.2–4.6 recommend, already written against current llama.cpp. Community findings in the thread matter: one tester found `O_DIRECT` *slower* than buffered reads when the page cache could hold the working set; the same tester later retracted a parity claim, finding output divergence that grows with generation length in the multi-pass wave GEMMs; and auto cache sizing aborts when it gives fewer than 3 × top-k slots. Consider rebasing your verification/contract layer onto this design instead of maintaining a hook inside `mul_mat_id`, and post your SD665 numbers in the thread; phone-class CPU data is missing there. |
| **BigMoeOnEdge** (already in your research) | Android, UFS, `O_DIRECT`, out-of-tree via the eval callback | Its author's note in the #25294 thread: the `O_DIRECT`-versus-page-cache decision is sharper on phones because the kernel reclaims aggressively and dense weights cannot be pinned. That supports measuring 4.3 Option A against Option B rather than assuming either. |
| **Swiftlet** (Swift + Metal, iOS) | Runs Qwen3.6-35B-A3B on iPhone; its `.qpack` container keeps dense weights resident and stores each layer's experts as fixed-stride blobs, one `pread` per expert. Reports 7–11 tok/s for the 4-bit model with 2.6 GB RAM. | Evidence that per-expert contiguous packing (4.7) is a workable format. The hardware gap (Apple flash and GPU versus UFS-class storage and four A73 cores) is enormous, so its speed says nothing about yours. |
| **Edge0** (Qwen3.6-35B-A3B derivative) | Streams experts from SSD, but runs **4 of 256 experts per token instead of 8** | Halves expert bytes, but it is a different, approximate model with its own benchmark results. A useful upper-bound reference, not a drop-in optimization. |
| **FreeToken** (Berkeley/MIT, Aug 2026) | GPU-centric edge MoE serving; combines double-buffered full-layer prefill, semantic recurrent-state checkpoints, a shared LRU expert cache, and bandwidth-adaptive CPU/GPU miss execution | Its recurrent-state checkpointing for hybrid models is the relevant part for you: it is the mechanism 3.9 (prefix reuse across turns) needs for the 30 linear-attention layers. |
| **T-MAC** (Microsoft) | Replaces multiply-accumulate with bit-wise table lookups for low-bit weights on CPU | Table lookup (`TBL`) exists on ARMv8.0, so this sidesteps the missing dot-product entirely. But it needs its own weight formats (GPTQ-style), not Q4_K, and its biggest wins are at 1–2 bits. A research spike, not a near-term change. |
| **llama.cpp MTP (PR #22673, merged May 2026)** + `ggml-org/Qwen3.6-35B-A3B-MTP-GGUF` | Self-speculative decoding with the model's own MTP head | See 6.2 and 8.4. |

### 8.2 Caching and routing locality

- **"Not All Models Suit Expert Offloading" (ICLR 2026).** Across 20 MoE LLMs, models that apply MoE on every layer and do not use shared experts have the highest local routing consistency, while shared experts tend to lower it. It also finds that a cache of about twice the active experts is the usual balance point. For your model that is about 2 × 960 ≈ 1,900 slots (≈1.1 GB), a good first target for Option B in 4.3. Qwen3 scored well in that study, but Qwen3.6 adds a shared expert, so measure with the appendix script rather than inheriting the result.
- **"In-Depth Analysis on Caching and Pre-fetching in MoE Offloading" (Nov 2025).** Reports that the chance of a token reusing its predecessor's expert is above random (12.5% for top-8) and sometimes near 30%, and observes that LRU caches tend to repeat history rather than predict the next selection. Consistent with your 35% hit rate at 1,024 slots.
- **Approximate alternatives (change outputs):** Mixture of Cache-Conditional Experts makes routing cache-aware on mobile; ReMoE fine-tunes the router to boost reuse; BuddyMoE substitutes a cached "buddy" expert when a prefetch misses. All trade accuracy for hit rate. Keep them out of the exact path.

### 8.3 Expert prediction and prefetch

- **Pre-attention expert prediction (Nov 2025).** Predicts a layer's experts from that layer's *pre-attention* activations with two small trained linear layers, reporting 94.69% accuracy on Qwen3-30B. This beats the previous-layer predictor sketched in 4.5 and also covers layer 0. The cost is training a small predictor per layer, which is fine offline. On your hybrid model the lead time is short (one attention or DeltaNet block), so pair it with the same-layer temporal predictor for longer lead time.
- **Speculating Experts (Mar 2026)** is a parameter-free alternative that uses internal representations to predict future-layer experts.
- **Also relevant:** PreScope (prediction-driven cross-layer scheduling), "A Spatio-Temporal Expert Prefetching Framework" (2606.15453), and "Patterns behind Chaos: Forecasting Data Movement" (ISCA 2026). All use the real router as the final authority, so they fit your exactness rules.

### 8.4 Speculative decoding meets expert offloading

The 2025–2026 literature has converged on the problem section 6 describes. DraftExpert frames it as *expert-set expansion*: multi-token verification activates a growing union of experts, so parallel verification stops being nearly free when experts come from CPU memory or flash. SpecFit makes the same point and notes speculative decoding is sometimes slower than plain decoding under offloading. The approaches split by exactness:

| Approach | Changes outputs? | Idea |
|---|---|---|
| **SpecFit** (Oct 2026) | No | Chooses the verification subtree and draft depth by draft-probability reward divided by expert-union cost, solved exactly, without changing routing |
| **EcoSpec** (Jul 2026) | No (verification rule unchanged) | Prefers draft paths that reuse experts already in the verification set; reports up to 1.62× |
| **EVICT** (May 2026) | No (described as lossless) | Adaptive verification that verifies fewer draft nodes, shrinking the expert union |
| **AcceptMoE** (Aug 2026) | Yes | Restricts which experts verification may use |
| **SpecMoEOff**, **Specter**, **MoE-SpeQ** | Mostly GPU/PCIe setups | Use drafts to enlarge per-expert work or to prefetch verification experts; SpecMoEOff reports up to 2.5× over offloading baselines on GPU+CPU |
| **"The Limits of Speculation"** (Sep 2026) | Analysis | Models speculation-budget choice as balancing marginal expert-loading cost against expected accepted tokens |

The takeaway for this phone: if you try speculation, use the MTP GGUF, keep k at 1–2, and make the draft-length decision cost-aware in the SpecFit/EcoSpec style, verifying a draft only when its predicted experts are mostly resident. That stays exact and is the only shape likely to beat 1.0× here. Most published speedups come from GPU/PCIe offload, where compute is plentiful and the ratios differ from flash-bound CPU decoding.

### 8.5 Mobile NPU and flash-specific work

- **EStream (Sep 2026)** runs MoE prefill on a mobile NPU by keeping experts in UFS and paging them through a fixed-size NPU-addressable arena, hiding loading behind computation; evaluated on a commercial Snapdragon phone with 7B–16B MoEs, scaling to 46.7B. It targets a modern Hexagon NPU, which your SD665 cannot use from an unprivileged shell process, but its "fixed arena + group-by-group paging" structure is the same contract your arena already enforces.
- **MoE-APEX (ASPLOS 2026)** offloads experts at adaptive precision. That is approximate, in the same family as Edge0's K=4 and lower-bit experts in 5.4.

### 8.6 How this changes the plan

1. **Read PR #25294 before writing 4.2–4.6 yourself.** Its I/O pool, per-layer slots, hotness-decay eviction, and wave prefill overlap heavily with this plan; reusing them leaves you to focus on what is unique to your project (verification, contracts, Android evidence). Do not adopt wave prefill without your own parity gate, given the divergence reported in that thread.
2. **Size Option B around 2× active experts** (~1,900 slots) first, then sweep.
3. **Prefetch:** start with the temporal predictor (free, measurable today), then evaluate a trained pre-attention predictor.
4. **Speculation:** only after the cache works, MTP GGUF, k ≤ 2, cost-aware draft length.
5. **Kernels:** the bit-identical Q4_K inner loop (5.2) stays the practical step; T-MAC is the longer-term option if you ever accept a different weight format.

---

## Appendix A: trace replay script

Save as `tools/replay_trace.py` and run it on an `AME_MOE_TRACE` file captured during a few hundred generated tokens (`AME_MOE_TRACE=/data/local/tmp/trace.txt`). It treats calls with exactly `top_k` experts as decode calls.

```python
#!/usr/bin/env python3
"""Offline analysis of an AME_MOE_TRACE routing trace.

Usage: python3 replay_trace.py trace.txt [--slots 64,256,1024,2048,4096]
"""
import argparse
import collections
import statistics

def load(path, top_k):
    rows = []                       # (tensor, [experts]) for decode calls, in order
    with open(path) as handle:
        for line in handle:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            tensor, count = parts[1], int(parts[2])
            experts = [int(x) for x in parts[3:3 + count]]
            if count == top_k:      # single-token call; prefill calls have larger unions
                rows.append((tensor, experts))
    return rows

def per_tensor_sequences(rows):
    seqs = collections.defaultdict(list)
    for tensor, experts in rows:
        seqs[tensor].append(frozenset(experts))
    return seqs

def union_growth(seqs, max_n=6):
    out = {}
    for n in range(1, max_n + 1):
        sizes = [len(frozenset().union(*seq[i:i + n]))
                 for seq in seqs.values() for i in range(len(seq) - n + 1)]
        if sizes:
            out[n] = statistics.mean(sizes)
    return out

def temporal_reuse(seqs):
    reused = total = 0
    for seq in seqs.values():
        for prev, cur in zip(seq, seq[1:]):
            reused += len(prev & cur)
            total += len(cur)
    return reused / total if total else 0.0

def lru_hit_rate(rows, slots):
    cache = collections.OrderedDict()
    hits = lookups = 0
    for tensor, experts in rows:
        for expert in experts:
            key = (tensor, expert)
            lookups += 1
            if key in cache:
                hits += 1
                cache.move_to_end(key)
            else:
                cache[key] = None
                if len(cache) > slots:
                    cache.popitem(last=False)
    return hits / lookups if lookups else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trace')
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--slots', default='64,256,1024,2048,4096')
    args = parser.parse_args()
    rows = load(args.trace, args.top_k)
    seqs = per_tensor_sequences(rows)
    print(f'decode calls: {len(rows)}  tensors: {len(seqs)}')
    for n, size in union_growth(seqs).items():
        print(f'U({n}) = {size:6.2f}  bytes ratio vs 1 token = {size / args.top_k:4.2f}')
    print(f'same-layer reuse from previous token: {temporal_reuse(seqs):.1%}')
    for slots in (int(x) for x in args.slots.split(',')):
        print(f'LRU {slots:5d} slots ({slots * 589824 / 2**20:7.1f} MiB): '
              f'hit rate {lru_hit_rate(rows, slots):.1%}')

if __name__ == '__main__':
    main()
```

`U(n)` feeds section 6 directly; "same-layer reuse" is the precision ceiling of the temporal predictor in 4.5; the LRU curve tells you how many slots Option B in 4.3 needs. Extending it with a per-layer budget or frequency pinning is a few more lines.

## Appendix B: latch reset for the ggml patch (3.1)

```c
// ggml/include/ggml-moe.h
void ggml_moe_reset(void);

// ggml/src/ggml-cpu/ggml-cpu.c, next to the other hook state
void ggml_moe_reset(void) {
    g_moe_outcome = GGML_MOE_PASS_THROUGH;
    g_moe_error_latched = 0;
}
```

```cpp
// src/backend/moe_stream.cpp
void release_moe_stream_leases() {
  for (auto& [key, cache] : g_caches) { (void)key; cache.lease = Lease{}; }
  ggml_moe_reset();
  g_cancellation.reset();
  g_stats.failed = false;      // keep a separate lifetime failure counter for /health
  g_stats.cancelled = false;
  g_stats.error.clear();
}
```

Call it only after all backend workers have drained (`llama_decode` has returned), which is already where the server calls `release_moe_stream_leases()`. Add a test that fails request N through a switchable fault hook and asserts request N+1 completes in the same process.
