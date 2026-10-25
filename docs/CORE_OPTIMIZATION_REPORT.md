# Core optimization report (2026-10-24 to 2026-10-25)

Executes `prompts/SONNET_CORE_OPTIMIZATION.md` on branch `perf/core-optimization`, which sits on
`fix/audit-a1-a7` (pull request 3). Nothing is pushed. Every number below is read from a retained record under
`evidence/core-opt/`; each unit's directory has its own README with tables, method and limits. The device is the
vivo 1920 (Snapdragon 665, 8 GB, Android 12, charger connected), the artifact is the enrolled Qwen3.6-35B-A3B
Q4_K_M (SHA-256 `671e47e0...5140c7`), the backend is llama.cpp `ff5a233`.

## Outcome

Emitted token IDs never changed: they are identical in every phone A/B trial (70 in Units 4 to 6, plus the diagnostic
series in `evidence/core-opt/unit6b/`), and identical on the host
across five cache/reader/overlap variants of the same artifact. A/B medians, five alternating fresh-process trials
per arm, request wall times from the server's own records:

| Configuration | Short request (24-token prompt) | 32-token request | Decode step | Decode tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Shipped: 64-slot cache, serial reads | 55.4 s | 122.3 s | 1.74 s | 0.51 |
| + layered cache, 32 slots per layer (Unit 4) | 40.0 s (-27.7%) | 74.0 s (-39.5%) | 1.19 s | 0.79 |
| + 4 reader threads (Unit 5) | 25.9 s (-39.3% vs previous) | 51.2 s (-31.5%) | 0.92 s | 1.04 |
| + top-k overlap and pinned compute pool (Unit 6) | 22.1 s (-5.2%) | 46.2 s (-7.8%) | 0.86 s | 1.13 |

Each row's percentage is within its own paired series. The rows come from separate series on different days with
different page-cache state, so the end-to-end comparison (55.4 to 22.1 s short, 122.3 to 46.2 s 32-token, 2.2x decode
throughput) is indicative, not a paired result. Clean restart is 170.6 s, 99% of it hashing the artifact.

**Defaults that changed:** the phone launcher `deploy/phone/start-35b.sh` now starts the layered cache (32 slots per
layer, 2.2 GB), four readers, the top-k overlap and a threadpool pinned to the big cores. The server binary's own
defaults did not change (shipped cache, one reader, overlap and pinning off), because the new settings spend 2.2 GB of an
8 GB device. `deploy/phone/stop.sh` now matches the process name; it used to miss a server started as `./ame-server`.

**Negative or neutral results:** the 1.13 GB layered cache (16 slots) is faster in prefill and slower in decode and was
not promoted; keeping the kernel's page-cache copy is 8.4% slower than dropping it; pinning the readers to the little
cores made no measurable difference; the pinned threadpool alone changes nothing measurable; the top-k overlap alone is 30%
slower; the tiled kernel change is inert on the phone; LFU and per-layer partitioning give at most 3.1 points over the
shipped LRU at equal bytes.

## Unit 1: lifecycle defects (`535438d`)

Files: `patches/0001`, `src/backend/moe_stream.{cpp,hpp}`, `apps/server/{main.cpp,request_scope.hpp}`,
`bench/{moe_recovery_gate,complete_runner}.cpp`, `tests/request_scope_test.cpp`, `tools/{verify_complete,verify_inference}.py`,
docs.

- `ggml_moe_reset()` clears the process-global error latch; installing or removing a materializer calls it; the project
  calls `begin_moe_stream_request()` per request under exclusive admission, after the previous context was destroyed and
  its leases released. The recovery gate runs: failed or cancelled MoE graph, an unrecovered dense graph (it aborts: the
  latch is real), reset, a dense graph (succeeds), the same MoE graph (bit-identical to the resident reference), with 1, 2
  and 4 workers. With the reset call disabled the gate fails on the dense graph.
- `RequestScope` releases leases, publishes the terminal snapshot, then releases admission last, on every exit including
  failed SSE writes (`request_scope_test` checks the order, idempotence, shared ownership and throwing hooks). The snapshot
  carries a bounded error string.
- `complete_runner --recover-after-fault` fails one generation in the materializer, resets and runs a fresh context in the
  same process; its logits and token IDs equal an uninterrupted run byte for byte.
- Commands: `tools/verify.py` 15/15 in Debug, Release, ASan/UBSan; `tools/verify_inference.py` passed in all three.
- Limit: real-35B recovery parity is not established; the `RequestScope` test has no pre-change equivalent because the class
  is new.

## Unit 2: per-request measurement (`b208ac7`)

Files: `apps/server/{main.cpp,request_metrics.hpp}`, `src/residency/file_source.*`, `src/residency/store.*`,
`src/backend/moe_stream.*`, `tools/{run_record,request_series,summarize_series}.py`, `docs/REQUEST_RECORDS.md`.

- Wall time for context creation, prefill, decode, teardown and total; deltas of `/proc/self/io` `rchar` and `read_bytes`
  (separate), `majflt`, `pswpin`; expert hits, misses, slice reads and bytes, actual `pread` calls and bytes, read seconds;
  big-core `scaling_cur_freq` at phase boundaries with big cores found from `cpuinfo_max_freq`; startup stages; sampled token
  IDs, decode step times and TTFT. Unreadable counters are null with a reason. `getenv` is read when a source opens and on
  `reload_fault_injection()`, not per chunk or operator.
- Commands: `tools/verify.py` 16/16; `tools/verify_inference.py` passed (recovery gate now also checks per-request expert
  deltas across a read failure and a cancellation).
- Phone: clean restart 170.6 s, 168.7 s of it the artifact hash (20.43 GB read at about 119 MB/s). Three identical short
  requests 50.1-58.4 s with token IDs `[9419, 0, 248046]`; one 32-token request 114.6 s. Prefill is 90% of a 24-token request;
  expert read wall time is 76-81% of prefill and 71-73% of decode; block-layer `read_bytes` was 1.9-2.75x `rchar` in prefill.
  The historical 1,732 s startup was not reproduced and cannot be decomposed (no stage records existed then).
- Limits: parity with the pre-change binary was not run separately; one series.

## Unit 3: routing trace and exact replay (`a7716a8`)

Files: `src/backend/moe_stream.*`, `tools/replay_routing.py`, `tests/{replay_routing_test.py,store_replay_driver.cpp}`,
`docs/ROUTING_TRACE.md`.

- Trace format 2 labels request, graph, phase and ubatch, buffered and flushed per graph. The replay's baseline reproduces
  `Store::acquire`; the test runs random scenarios against the real Store and the case that separates it from a naive
  replay (capacity two, `[A, B]`, then `[C, A]`: the real Store keeps A, evicts B, records one hit; a naive replay records
  zero).
- A 330-token trace from the phone (34-token prompt, 40,080 rows, 334 graphs). Decode bytes read per token: 566 MB
  (shipped, no hits) to 361, 280, 224 and 184 MB at 0.57, 1.13, 1.70 and 2.26 GB. Capacity is the lever; per-layer
  partitioning adds at most 3.1 points; LRU vs LFU differ by under 1 point except 0.5 in LFU's favour at 1.13 GB.
- Commands: `tools/verify.py` 17/17; `tools/verify_inference.py` passed.
- Limits: one prompt; the replay cannot re-split prefill into smaller units.

## Unit 4: layered, layer-partitioned cache (`63bf98e`, A/B `8ee2fb0`)

Files: `src/residency/layered_store.*`, `src/residency/file_source.*`, `src/backend/moe_stream.*`, `apps/server/main.cpp`,
`bench/{moe_layered_gate,llama_baseline}.cpp`, `tests/layered_store_test.cpp`, `tools/ab_trials.py`.

- One pool per layer; one slot holds the gate, up and down slices; staged views `data = pool + role_offset`,
  `nb[2] = slot_bytes`, `ne[2] = slots`, which patch 0001 already honours. Fused gate/up, unequal expert counts, a pool
  below top-k and an allowance below one slot per layer are refused with specific errors. LRU via an intrusive list; LFU
  with halving behind `--eviction`. `--fadvise-random` and `--fadvise-dontneed`. The shipped cache stays in the same binary.
  All bookkeeping is counted in `cache_metadata_bytes` and held under `--arena-mib`.
- Host gates: `moe_layered_gate` bit-exact for F32, Q4_K and Q6_K against the resident reference with evictions, decode and
  prefill, counters equal to a predictor; harness token and logit parity with the shipped cache under ASan/UBSan.
- Phone: fifteen trials (base, 16-slot, 32-slot) plus ten (32-slot with and without dropping the kernel copy), token IDs
  identical in all 25. 32 slots: short request -27.7%, 32-token request -39.5%, decode step -31.8%; prefill block-layer
  reads 15.8 GB to 5.0 GB. Peak RSS 2.14 to 4.37 GB, `MemAvailable` after 3.6 GB; no `lmkd` line mentions the server.
- Limits: no memory-pressure campaign; zram statistics are not readable by the shell uid.

## Unit 5: reader threads (`ef2af62`, A/B `96f783f`)

Files: `src/residency/reader_pool.*`, `src/residency/{file_source,store,layered_store}.*`, `CMakePresets.json`, `tools/verify.py`,
`tests/{reader_pool_test.cpp,layered_store_test.cpp,residency_test.cpp}`.

- A batch ends only after every read has finished or been skipped; a failed batch publishes nothing; `FileSource` is
  safe for concurrent reads. Tests cover a failure with other reads in flight, cancellation mid-batch, byte equality for
  1, 2 and 4 readers, and a frozen destination after each call returns, under ASan/UBSan and ThreadSanitizer (no reports;
  a deliberate race in a scratch program is reported in this environment). TSan runs under `setarch -R` because it
  cannot map its shadow memory under full ASLR on this kernel.
- Phone: r1, r2a, r4a, 15 trials; r4a vs r1: short request -39.3%, 32-token request -31.5%, decode step -21.9%; bytes
  read and peak RSS unchanged; token IDs identical. Pinning readers to cpu0-3: no effect (10 trials).
- Limits: `O_DIRECT` not tried (needs an alignment self-test); no standalone storage queue-depth benchmark.

## Unit 6: top-k overlap and pinned threadpool (`345fc09`, A/B `e221c0a`)

Files: `src/backend/moe_stream.*`, `src/residency/{layered_store,reader_pool}.*`, `apps/server/main.cpp`,
`bench/{moe_layered_gate,llama_baseline}.cpp`, `tests/layered_store_test.cpp`.

- Option A (the scheduler eval callback) was chosen after reading the pinned source (`docs/DESIGN_DECISION.md`): the
  callback receives `ffn_moe_topk-<layer>` between graph segments, starts the layer's reads in the background, and each
  operator waits only for its own role; patch 0001 is unchanged. `--threadpool big --poll 0` pins one `ggml_threadpool`
  to the big cores.
- Host gates: the layered gate with `--overlap` for 1, 2, 4 workers x 1, 2, 4 readers (bit-exact, counters, ignored
  tensors, an armed layer never consumed, a read failure and cancellation); the harness through the real scheduler gives
  identical tokens and logits with every operator finding its layer started.
- Phone: twenty trials; overlap plus pinned pool -7.8% (32-token) and -5.2% (short); pool alone no effect; overlap alone
  +29.9% (reads hidden but compute inflates from 18 s to 44 s while readers run).
- Diagnostic (`evidence/core-opt/unit6b/`, 15 trials, identical tokens): pinning only the readers does not repair the
  overlap (32-token median 64.5 s vs 46.2 s with the pinned pool); pinning both gains nothing over the pool alone (45.9 s).
- Limits: a `--poll` sweep and next-layer prefetch were not run; one reader-pinned-only trial ran much faster than its four
  siblings and is unexplained.

## Unit 7: tiled kernel for staged tensors (`d5170cb`)

Files: `patches/0001` (adds the tiled files), `tests/runner_boundaries_test.py`, `tools/verify_inference.py`,
`bench/moe_layered_gate.cpp`.

- The tiled K-quant kernel and its dispatch take the effective expert tensor. With `GGML_CPU_TILED_MM_FORCE=1` the layered
  gate is bit-identical to the resident reference; with the old skip restored it mismatches on 24 of 24 Q4_K and 24 of 24
  Q6_K operators. The harness gives identical tokens and logits for staged and resident runs at the same prefill unit on a
  14-token prompt.
- Phone prefill A/B: **not run, because it cannot differ.** The tiled kernel is compiled in only for x86 AVX targets
  (`ggml_tiled_supported` is a stub returning false otherwise) and needs eight rows per expert; the phone's prefill units are
  2 to 8 tokens.

## Verification of the final tree

`.venv/bin/python tools/verify.py`: 19/19 in Debug, Release and ASan/UBSan; 4/4 in the ThreadSanitizer preset.
`tools/verify_inference.py` (qwen35moe and qwen2moe fixtures, rebuilt patched backend): debug, release and sanitizers
passed (`evidence/core-opt/final-integration.json`). Real 35B on the host (`tools/variant_parity.py`): five variants, identical
IDs, equal to the phone's (`evidence/core-opt/host-parity-35b/`). Backend patch set: `0001` (now including the tiled
files) and `0002`; the applied diff equals their concatenation (checked by the verifier).

## Next candidates (not implemented)

- **Context and recurrent-state reuse across turns.** Evidence: a fresh context costs 0.11 s but prefill is 90% of a short
  request and about half of a 32-token one (Unit 2 and Unit 4 tables); a multi-turn chat re-prefills its history.
- **Next-layer expert prefetch.** Evidence: with the overlap and pinned pool the generation thread is still blocked 8.85 s of
  27.4 s of decode (Unit 6), and the router of layer L+1 can be applied to layer L's residual stream. Precision for this
  hybrid model is unmeasured.
- **fs-verity or per-extent integrity instead of the startup hash.** Evidence: the hash is 99% of a 170.6 s restart. `ame probe`
  now reports that `/data` is f2fs and that the verity feature directory is not readable by the shell uid, so whether this
  kernel supports it is unknown.
- **Q4_K inner loop for CPUs without dot-product.** Evidence: decode compute is now about 18 s of 27 s for 31 tokens
  (0.58 s per token), the largest remaining share.
- **`O_DIRECT`.** Evidence: dropping the kernel copy was 8% faster than keeping it; direct reads need an alignment
  self-test because GGUF expert offsets are 32-byte aligned.
- **Speculative decoding, MTP, GPU or NPU, lower-bit expert artifacts, a second backend:** none implemented; no evidence in
  this series argues for them over the items above.

## Reproduce

```sh
.venv/bin/python tools/verify.py
AME_LLAMA_SOURCE_DIR=<patched llama.cpp> AME_LLAMA_HOST_LIB_DIR=<host build> \
  .venv/bin/python tools/verify_inference.py --fixture-model <qwen35moe.gguf> --layered-fixture-model <qwen2moe.gguf>
AME_TOOLS=<tools dir> sh bench/build_server.sh       # host and Android servers
.venv/bin/python tools/ab_trials.py run --variant 'NAME=server args' ... --trials 5 --output-dir DIR   # phone A/B
.venv/bin/python tools/ab_trials.py summarize DIR
```
