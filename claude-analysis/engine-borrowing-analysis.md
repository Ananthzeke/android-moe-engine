# What android-moe-engine can borrow from current inference engines

A companion to `android-moe-engine-review.md`. That review found the engine spends about 85–90% of each decode token on synchronous expert reads (≈566 MB per token at a 0% project-cache hit rate, about 2.3 s/token on the vivo S1 Pro). This document asks which ideas from ten other engines would move that number on a Snapdragon 665, and which would make the engine general enough to run other models on other phones.

## How this was done

I shallow-cloned and read the code and documentation of eight repositories on 21 October 2026:

| Repository | Commit read | License |
|---|---|---|
| andrei-borcea/liyab | `c2e07f9` (2026-10-20) | Apache-2.0 |
| danveloper/flash-moe | `3601d41` (2026-03-19) | **No license file** (ideas only; do not copy code) |
| deepseek-ai/DeepInfer | `9bd9cf2` (2026-10-21) | Apache-2.0 |
| microsoft/BitNet | `0b341e5` (2026-07-27) | MIT |
| PrismML-Eng/Bonsai-demo | `294035e` (2026-10-20) | see repo |
| UbiquitousLearning/mllm | `1d3c3d8` (2026-09-30) | MIT |
| google-ai-edge/LiteRT-LM | `13c208f` (2026-10-20) | Apache-2.0 |
| ollama/ollama | `2cd63f7` (2026-10-20) | MIT |

vLLM and SGLang are too large to review meaningfully this way; for them I used their published design documents and my knowledge of their architecture, and I say so where it matters. Numbers quoted from other projects are their authors' measurements on their hardware. None were measured on an ARMv8.0 CPU without dot-product, and several were measured on flagship phones with storage roughly an order of magnitude faster than yours.

---

## TL;DR: the twelve borrowings worth doing, in order

| # | Borrow | From | Why it matters on the SD665 | Effort | Output-exact? |
|---:|---|---|---|---|---|
| 1 | **Expert store with its own RAM cache, LFU with periodic halving, direct I/O with a self-test fallback** | Liyab `ExpertStore`, `DirectFile` | Replaces the 64-slot LRU that cannot hit. Liyab measured the page cache as a *second* expert cache and found it slower in every A/B run | Medium | Yes |
| 2 | **Split one expert's three matrix reads across idle I/O threads; 2–4 large requests in flight** | Liyab | Gives queue depth without tiny requests; a waiting forward pass gets its expert after one matrix's latency, not three | Small–medium | Yes |
| 3 | **Predict the next block's experts by applying its router to the current residual; settle wrong guesses immediately** | Liyab (79–83% of guesses used on this exact model) | The only predictor in this set measured to work on Qwen3.6-35B-A3B. flash-moe's temporal predictor *lost* 18% | Medium | Yes (router stays authoritative) |
| 4 | **Protect freshly loaded experts until the block they were loaded for has used them** | Liyab (+71% decode in one step) | Without it, prefetching evicts its own work. Becomes critical the day #3 lands | Small | Yes |
| 5 | **Run cached experts first and the shared expert while routed ones load; sum outputs in router order** | Liyab | Overlaps compute with I/O *and* keeps results bit-identical regardless of arrival order | Small | Yes, by construction |
| 6 | **Work-claiming thread pool with short low-power spin (WFE) then sleep** | Liyab `thread_pool` | Big.LITTLE (4×A73 + 4×A53) benefits from dynamic chunk claiming; avoids the barrier spin that heats your phone during I/O | Medium | Yes |
| 7 | **Context reuse across turns + recurrent-state snapshots + content-addressed prefix store on flash** | Liyab, SGLang RadixAttention/HiCache, vLLM prefix caching, llama.cpp `--ctx-checkpoints` | At ≈1 s per prompt token, re-prefilling a conversation each turn is the single largest waste in chat | Medium–large | Yes |
| 8 | **Chunked, cancellable, resumable prefill** | vLLM chunked prefill, LiteRT-LM `RunPrefillAsync`, Liyab | Fixes the uninterruptible 300 s deadline problem | Small–medium | Yes |
| 9 | **Jump-forward (compressed-FSM) structured decoding** | SGLang | Forced tokens in JSON/tool calls can be processed in one batched pass; on a 2 s/token engine every forced token skipped is seconds saved | Medium | Yes |
| 10 | **Thermal headroom forecast + pacing** | Liyab power manager | Your identical runs vary 2×; the A73 cluster throttles. `AThermal_getThermalHeadroom` exists on Android 12 (API 31) | Small–medium | Yes |
| 11 | **Batch-invariant kernels and golden-logit tests per kernel set** | DeepInfer, Liyab | Makes chunked prefill, waved prefill and any future speculation bit-identical to serial decode, which matches your parity philosophy | Medium | This *is* exactness |
| 12 | **Device-measured profile and memory plan** (cores, storage QD curve, RAM bandwidth, budget) instead of fixed flags | Liyab, LiteRT-LM, Ollama | The prerequisite for generalizing beyond one phone | Medium | n/a |

Things **not** to borrow for this phone, with evidence, are in section 10.

A rough expectation if items 1–8 land: on the order of 0.8–1.3 s/token (≈0.8–1.2 tok/s) versus ≈2.3 s/token today. That is an estimate assembled from your evidence and Liyab's per-phase numbers scaled to slower hardware; the storage queue-depth sweep from the review (section 4.0) decides where in that range you land.

---

## 1. Liyab: the closest relative, and the most useful repository on this list

Liyab is a C++20 engine for GGUF models on phones. Its headline is the same model you run: **Qwen3.6-35B-A3B at ~9–10.5 tok/s from the command line on a Snapdragon 8 Elite**, streaming experts from UFS 4 into the 6 GiB an Android app may use. It is Apache-2.0, so you can adapt code with attribution and its NOTICE.

### 1.1 What its MoE path does

From `src/core/expert_store.h` and the README's "Models larger than RAM":

- **Direct I/O with a trust check.** `DirectFile` opens with `O_DIRECT` and falls back to buffered reads where direct I/O is refused *or untrustworthy*: on Android's FUSE paths (`/storage/emulated`) direct reads were seen to return success without the file's bytes, and it compares a direct and a buffered read of the first and middle block before trusting any file. Your model lives in `/data/local/tmp`, which is not FUSE, but the self-test is cheap insurance and essential if you ever load from shared storage.
- **Request shape.** On UFS 4 it measured ~4.4 GB/s with 2–4 concurrent requests of ≥256 KiB, against under 1.2 GB/s with 16 KiB requests. Eight I/O threads or splitting a matrix into 2–4 reads were *slower*. Your 576 KiB expert slices are already a good request size; what you lack is concurrency.
- **Cache entry = (block, expert)** holding gate, up and down; size classes per matrix layout; **LFU with periodic halving** (`age()` once per token); entries in use are never evicted; a freshly loaded entry is not evicted before its block uses it.
- **One-block-ahead prediction.** After a block's mixer, the *next* block's router is applied to the residual stream and its picks are prefetched, ordered by rank and only while that rank's guesses are paying off. About 79–83% of guesses were used on Qwen3.6-35B-A3B. When the real router decides, queued wrong guesses are dropped and already-read ones become first to evict.
- **Matrix-level parallelism.** The I/O thread that starts an expert reads gate and hands up and down to idle threads.
- **Compute/I/O overlap.** Experts already in RAM run first in one batched pass; the shared expert runs while routed ones wait.
- **Prefill.** Passes of up to 1,024 tokens read the union of their tokens' experts once, and such passes give experts **no LFU credit**, so a long prompt does not flush the experts a conversation keeps using.
- **Determinism.** Expert outputs are summed in router order whatever order they arrived in, so streamed and resident experts give bit-identical logits (tested).
- **Observability.** `Stats` counts hits, *late* (predicted but still in flight), misses, loads, *unused* (loaded and evicted unused), *dropped* guesses, bytes and stall milliseconds. Those counters are exactly what you need to tune prediction; your `MoeStreamStats` has none of the prediction-quality fields yet.
- **Warm restarts.** `trim()` records a hot list; `warm()` refills the cache in the background with large reads when the model is used again. Your server starts every process with a cold cache.

### 1.2 Its measured progression on this model

Liyab's README records how decode improved on the 8 Elite (CPU backend, 5,500 MiB budget, cold expert cache):

| Change | Decode | Expert reads/token |
|---|---:|---:|
| Condition-variable thread pool (start) | 3.06 tok/s | 596 MiB |
| + spinning thread pool | 3.24 | 480 MiB |
| + freshly loaded experts kept until used | 5.54 | 309 MiB |
| + a block's experts in one batched pass | 5.93 | 309 MiB |
| + Q4_K/Q5_K/Q6_K × Q8_K kernels | 6.15 | 308 MiB |
| + next block's experts predicted after the mixer | 6.44 | 283 MiB |
| + 7 threads, chunks claimed on demand | 7.83 | 283 MiB |
| + an expert's three matrices read in parallel | 8.89 | 280–311 MiB |

Two numbers matter for you. Its starting point, 596 MiB of expert reads per token, is essentially your 566 MB: the same model with no effective cache. And with a ~5.5 GB budget it settles near 280 MiB/token, roughly a 50% reduction; raising the budget to 7 GB took it to 208 MiB. Your phone has perhaps 2.5–3 GB to spare for experts, so expect less, but the ~2× working-set rule from the routing-consistency literature (≈1,900 slots ≈1.1 GB) suggests a meaningful fraction of that reduction is reachable.

Its per-token breakdown on a cool phone (~105 ms): routed experts ~55 ms (a third of which waits on flash), DeltaNet ~25, LM head ~8, attention ~7, routers and prediction ~6, shared expert ~4. It also notes ~2 GB of resident weights are read per token, so RAM bandwidth caps a single token near 20 tok/s there. On the SD665's LPDDR4X (≈15 GB/s theoretical, less in practice), the same 2 GB puts a floor of roughly 0.15–0.25 s/token under you before any expert is read. That floor is why the realistic target in the TL;DR is around 1 tok/s, not Liyab's 9.

### 1.3 Its negative results (save yourself the experiments)

From "What did not pay off":

- Predicting two or three blocks ahead: fewer late prefetches but ~11% more bytes; tied with one block ahead.
- More or smaller I/O requests: slower.
- **The kernel page cache as a second expert cache: slower in every A/B run**, because buffered misses hold the I/O threads that urgent reads wait for. This resolves the open question in the review's 4.3: on a phone with a tight memory budget, own the cache and bypass the page cache. (flash-moe found the opposite on a Mac with ~35 GB of page cache; see section 2.)
- **Speculation on streamed MoE:** verification reads the union of its tokens' experts, measured at 1.55× one token's cost for 2 tokens and 2.5× for 3. This matches the review's estimate (≈1.6× and ≈2.2×) closely. Liyab ships speculative decoding off by default for this reason; it pays on dense models with i8mm.
- 2:4 sparsity on phones: −54% decode.
- Binary low-rank experts: Qwen3.6's experts are nearly full-rank.
- Repacking every streamed expert on read: −6% decode; repack only for batched passes.
- `io_uring`: Android denies `io_uring_setup` to apps.

### 1.4 Where Liyab does not fit your phone

Liyab builds with `-march=armv8.2-a+dotprod+fp16` by default (`LIYAB_ARM_DOTPROD`, on for arm64), which will not run on the SD665's ARMv8.0 cores. The option can be turned off, and its code has plain-NEON paths, but its README does not claim testing on a CPU without dot-product. Its thermal and performance-hint integration partly depends on newer Android: thermal status needs API 30 and headroom API 31 (your Android 12 has both), while ADPF performance hints need Android 13, which your phone lacks.

This is also your differentiation. Liyab targets flagships with SDOT/i8mm and UFS 4 as an app; your engine targets the large installed base of ARMv8.0 phones, runs from an unprivileged shell, stays on upstream llama.cpp kernels, and carries a verification discipline (artifact identity, layout contracts, run records) that Liyab does not emphasize. Borrow its I/O and cache architecture; keep your verification layer as the thing that is yours.

---

## 2. flash-moe: what a 397B model on a laptop teaches, and where it disagrees

flash-moe runs Qwen3.5-397B-A17B on a 48 GB M3 Max at ~4.4 tok/s from a 209 GB file, in C/Objective-C and Metal. Its README documents 58+ experiments with keep/discard verdicts. It has no license file, so treat it as a source of ideas and measurements only.

**What it kept.** One packed file per layer with each expert's nine components (gate/up/down weights, scales, biases) contiguous at a fixed stride, so an expert is one `pread`; parallel `pread` on four threads; "trust the OS" page cache (~71% hit rate with ~35 GB of page cache); deferred GPU work so the CPU prepares the next layer while the GPU runs; BLAS for the DeltaNet recurrence (+64% on that op); a C tokenizer that cut startup from 3.5 s to 180 ms.

**What it discarded, and how it applies to you:**

| Experiment | Result there | Relevance to the SD665 |
|---|---|---|
| LZ4 expert compression | −13% | Your CPU is far weaker; decompression would cost more. Skip |
| Temporal expert prediction | −18% (25% hit rate) | Supports using Liyab's router-on-residual predictor instead of "same as last token" |
| MLP routing predictor | 31% accuracy | Same conclusion |
| Expert file clustering | 0% | Consistent with the llama.cpp community result for slices ≥ ~1.5 MB; your 576 KiB slices are in between, so low priority |
| mmap for experts | −5× | You already avoid it |
| Spin-poll GPU wait | −23% (CPU heat competes with GPU) | Supports replacing barrier spinning during I/O |
| MTP speculative decoding | Break-even | Same mechanism as Liyab's and the review's analysis |

**The disagreement with Liyab is informative, not contradictory.** flash-moe had ~35 GB of free page cache and Apple's 17.5 GB/s SSD; Liyab has a 6 GiB budget on a phone. With abundant RAM the kernel's cache is good enough and free; with scarce RAM, buffered misses and page-cache churn hurt. Your phone is the scarce-RAM case, so follow Liyab, but measure both with the `/proc/self/io` method from the review.

**One more lesson.** flash-moe found that on unified memory, SSD DMA and compute contend for the same memory controller, so overlapping them helped less than expected. A phone SoC also shares one LPDDR controller between the storage DMA and the CPU. Overlap will still help you (your compute is a small share of the token), but do not assume the I/O and compute times simply hide each other; measure the combined step time.

---

## 3. vLLM and SGLang: datacenter engines, with three ideas that transfer

Most of what makes vLLM and SGLang fast (continuous batching across many users, PagedAttention for multi-tenant KV memory, CUDA graphs, tensor/expert parallelism, prefill/decode disaggregation) solves problems a single-user phone does not have. Three ideas do transfer directly.

**Prefix caching.** vLLM hashes fixed-size KV blocks along the token sequence so identical prefixes are found and reused; SGLang's RadixAttention keeps finished requests' KV in a radix tree, and HiCache extends it across GPU memory, host memory and storage tiers. For you the equivalent is a content-addressed store of KV chunks keyed by a hash chained over every preceding token, plus recurrent-state images for the 30 DeltaNet layers, kept in RAM and on flash. Liyab implements exactly this on a phone, including surviving process restarts, and reports follow-up turns on the 35B starting in ~1.1–1.6 s instead of reprocessing the conversation. Upstream llama.cpp's server already has `--ctx-checkpoints` and `--cache-ram` for hybrid models (Bonsai's demo documents them), so check how much your pinned llama.cpp gives you before writing your own.

**Chunked, preemptible prefill.** vLLM splits long prefills into chunks so other work can interleave; for you the point is that a deadline or disconnect can stop work between chunks, and resumed work continues where it stopped. Your server already prefills in chunks; the missing piece is cancellation inside a chunk and keeping the processed prefix when a request is abandoned.

**Jump-forward structured decoding.** SGLang compresses the output grammar's finite-state machine so that runs of tokens with only one legal continuation are emitted together instead of one forward pass each. On a GPU this was worth a reported 1.6× on JSON. On your engine it is worth more, because each skipped decode step saves roughly two seconds. The forced tokens still need one forward pass to enter the KV/recurrent state, but processing k forced tokens in one pass reads the union of their experts, which is always fewer bytes than k separate decode steps. Tool calls and JSON responses are where this pays.

**Lesser ideas.** vLLM's and SGLang's speculative-decoding frameworks (n-gram, EAGLE, MTP) are useful as references for the verification loop, but see section 10. Their structured-output backends (XGrammar and similar) are a good model for a grammar engine if you add tool calling.

---

## 4. DeepInfer: DeepSeek's production engine

DeepInfer serves DeepSeek-V4.1-Flash on NVIDIA SM100 GPUs and Ascend NPUs. Almost none of its kernels or parallelism applies to a phone. Three ideas do:

1. **Confidence-scheduled verification.** In `schedule_mtp_varlen.py`, each draft position's confidence (from a small sigmoid head over the hidden state) is multiplied cumulatively along the draft, and a fixed verification budget is handed out to the positions with the highest expected acceptance. If you ever enable speculation, combine this with an expert-cost term: verify a draft position only when its cumulative confidence outweighs the marginal bytes of experts it would add to the verification union. That is the same shape as the SpecFit/EcoSpec work in the review's section 8.4, and it stays exact. Liyab's suffix drafter does a device-calibrated version of the same thing (expected tokens per millisecond).
2. **Batch-invariant inference.** DeepInfer can keep results identical across batch compositions (`DEEPINFER_BATCH_INVARIANT`). Liyab tests that each format gives "the same bits for every batch size and layout". For you this property makes chunked prefill, larger prefill units, speculative verification and waved prefill bit-identical to serial decoding. The llama.cpp PR #25294 thread reported growing divergence from multi-pass expert GEMMs precisely because this property was missing. Your parity gates are the right tool to enforce it.
3. **Hot-expert placement (EPLB).** DeepInfer rebalances and replicates hot experts across devices. The single-device analogue is pinning the most frequently routed experts permanently and letting the rest churn, which Liyab's LFU with halving approximates automatically.

---

## 5. LiteRT-LM: Google's production edge runtime

Your `docs/UPSTREAM_DESIGNS.md` already reviewed LiteRT-LM's streaming resources and session lifetimes. What else is worth taking from the current code:

- **Session `Clone()`** (`runtime/engine/engine.h`): copy a session's full context up to now. It is the API shape for branching a conversation, retrying a reply, or running a draft without disturbing the main context.
- **`RunPrefillAsync` returning a `TaskController`**: an asynchronous, cancellable prefill handle. That is the interface your server needs for deadlines and disconnects.
- **CPU affinity per SoC** (`cpu_affinity_utils.cc`): it hard-codes core sets per Pixel Tensor generation. The lesson is that pinning matters enough for Google to special-case it; the better implementation for many phones is to derive clusters from `cpuinfo_max_freq`, as the review suggested.
- **Built-in benchmark info and model introspection** (`describe`): per-session timing and a pre-load description of a model's requirements. Useful for a general engine that must tell a user whether a model will run.
- **MTP drafters** for Gemma 4 on GPU: confirms that MTP heads are becoming standard. On dense models and GPUs it pays; on your streamed MoE it does not (section 10).

---

## 6. Ollama: the product layer

Ollama is mostly a Go product around llama.cpp's server (`llm/llama_server.go`) and its own ggml-based engine. Its kernels add nothing for you; its product mechanics do:

- **Inference in a supervised child process.** The server survives a runner crash or abort. Your fail-closed design (abort on I/O failure) plus the sticky error latch from the review makes this attractive: a failed generation should kill a worker, not the API. The catch is your startup verification cost, which section 8 addresses.
- **A scheduler with keep-alive and load/unload by estimated memory.** On a phone, unloading after idle and warming on the next request (with Liyab's hot list) beats holding gigabytes forever.
- **Content-addressed model storage.** Models are stored by digest and verified at pull time, not on every start. Combined with per-extent or fs-verity verification (section 8), this removes your 3–7 minute startup hash.

---

## 7. mllm: mostly for newer phones

mllm v2's ARM CPU path uses Arm KleidiAI microkernels (`*_neon_dotprod`, `*_neon_i8mm`), and its top-level CMake defaults to `-march=native+fp16+fp16fml+dotprod+i8mm+sme` in one configuration. Its NPU path uses Qualcomm QNN ahead-of-time graphs for SM8650-class Hexagon. None of that runs on the SD665. Two ideas are still relevant: the **in-app server architecture** (a Go server packaged as an AAR inside the Android app, decoupling UI from inference), which is the right shape if you ship an app; and its EdgeMoE heritage (expert-wise bit-width adaptation), which your `research.md` already covers. For generalization to newer phones, KleidiAI is the obvious kernel source for dotprod and i8mm tiers.

---

## 8. BitNet and T-MAC: kernels that do not need dot-product

Two findings from `microsoft/BitNet` matter directly:

1. **Its no-dotprod fallback already uses the accumulation trick the review proposed.** In `src/ggml-bitnet-mad.cpp`, the `#else` branch of `__ARM_FEATURE_DOTPROD` accumulates eight `vmlal_s8` products into an `int16x8_t` before widening once with `vmovl_s16`. That is the same idea as the review's section 5.2 for Q4_K (value-range-aware accumulation in 16-bit lanes), independently used in a production kernel. It strengthens the case for writing that Q4_K/Q6_K path and upstreaming it.
2. **Its TL1 lookup-table kernels use `vqtbl1q_s8`,** which is part of ARMv8.0 NEON. Table lookup instead of multiply-accumulate is the one family of fast low-bit kernels that does not depend on SDOT. Its preset kernels are generated per model with tuned tile sizes (`kernel_config_tl1.ini`), and `include/gemm-config.h` exposes row/column block sizes and parallelism for per-CPU tuning.

The catch is that LUT kernels need their own weight formats (ternary I2_S/TL1/TL2, or T-MAC's GPTQ-style low-bit layouts), not Q4_K. They are a reason to support ternary and 1–2-bit models in a general engine, not a way to speed up your current GGUF.

**Borrowable now:** BitNet's practice of exposing tile and parallelism parameters, and a one-time on-device autotune of them, applies to your Q4_K kernel work as well.

---

## 9. Bonsai: a different bet on the same phone

PrismML's Bonsai models are dense models quantized to ~1–2 bits with Hadamard/sign-flip transforms. Bonsai 2 27B is 5.9 GB in `PTQ1_0` (1.75 bits/weight) and 7.2 GB in `PQ2_0`; the earlier 1-bit Bonsai 27B in `Q1_0` is about 3.5 GB. Official `Q2_0` Bonsai 2 runs on current upstream llama.cpp; the other packings need PrismML's fork.

**Why it matters for you.** A dense model reads *all* its weights every token, so it must fit in RAM to be fast. A ~3.5 GB 1-bit 27B fits in your phone's free memory; streaming it from flash would not be viable. Resident, its ~3.5 GB per token through DRAM would put a ceiling of roughly 3–4 tok/s at best on bandwidth alone, before compute on a CPU without dot-product (an estimate; the fork's CPU kernels on ARMv8.0 are untested here). That is plausibly faster than any streamed 35B MoE on this phone. A general engine should therefore treat "small low-bit dense, resident" and "large MoE, streamed" as two first-class plans and pick per device, rather than assuming MoE streaming is always the answer.

**Also worth borrowing:** the demo's community-benchmark template (one file per device with exact build, flags and results) is a cheap way to collect reproducible measurements from other phones, which also strengthens a sponsorship pitch.

---

## 10. What not to borrow on this phone, and why

| Idea | Who tried it | Result | Verdict for the SD665 |
|---|---|---|---|
| Speculative decoding (draft model, MTP, n-gram) on streamed MoE | Liyab, flash-moe, the review's analysis, 2025–26 papers | Verification reads the union of experts: 1.55× for 2 tokens, 2.5× for 3 (Liyab); MTP break-even (flash-moe) | Defer until the cache hit rate is high. If tried, MTP GGUF, k ≤ 2, cost-aware verification |
| Kernel page cache as the expert cache | Liyab (phone), flash-moe (Mac) | Slower on the phone; good on a Mac with 35 GB free | Own the cache with direct I/O; keep a buffered fallback |
| Temporal ("same as last token") prediction alone | flash-moe | −18% | Use router-on-residual prediction |
| Predicting 2–3 blocks ahead | Liyab | ~11% more bytes, tie | One block ahead |
| Many small or many concurrent requests | Liyab | Slower | 2–4 requests of ≥256 KiB |
| LZ4 / compressed experts | flash-moe | −13% on a fast CPU | Worse on yours |
| 2:4 sparsity | Liyab | −54% decode | No sparse units on phones |
| `io_uring` | Liyab | Denied to Android apps; your kernel likely predates it anyway | Threads + `pread` |
| `mmap` for experts | flash-moe | −5× | Already avoided |
| Repacking experts on every read | Liyab | −6% decode | Only for batched passes |
| Binary low-rank experts | Liyab | Qwen3.6 experts are nearly full rank | Skip |
| Spinning while waiting on I/O | flash-moe (GPU wait), your own barrier | −23% from heat on a Mac | Short WFE spin, then sleep |
| KleidiAI, i8mm repacks, QNN AOT, SME | mllm, Liyab | Require newer ISAs/NPUs | Only for higher device tiers |
| Lossy speedups (fewer experts, requantization, skip slow experts) | Liyab, Edge0 | Real speed, measurable KL cost | Only as opt-in, labeled lossy, measured |

Liyab's lossy table is a good template for the last row if you ever offer such modes: on the 8 Elite, `--expert-mass 0.9` (~6.5 experts) gave 8.19 tok/s at mean KL 0.039 and 92.0% top-1 agreement versus 7.41–8.08 tok/s lossless; requantizing resident weights to 4 bits gave 8.61 tok/s at mean KL 0.082. Measure the same way (KL and top-1 agreement against the lossless path over a fixed mixed text) before shipping any of them.

---

## 11. Generalizing the engine: a target architecture

Today the engine is one patch, one model, one phone and many global variables. The engines above converge on a layered structure. Here is one that keeps what is uniquely yours (verification and evidence) and borrows the rest.

```
┌──────────────────────────────────────────────────────────────┐
│ API / product: OpenAI-compatible server, CLI, (app later)   │  Ollama, LiteRT-LM
├──────────────────────────────────────────────────────────────┤
│ Session layer: context reuse, recurrent snapshots,          │  Liyab, SGLang, vLLM,
│ prefix store (RAM + flash), cancellable prefill,             │  LiteRT-LM, llama.cpp
│ structured output with jump-forward, optional speculation    │  --ctx-checkpoints
├──────────────────────────────────────────────────────────────┤
│ Planner: memory plan (resident vs streamed, cache size,      │  Liyab memory_plan,
│ prefill unit, threads) from the device profile + model       │  Ollama estimates
├───────────────────────────────┬──────────────────────────────┤
│ Expert/weight store:          │ Execution: llama.cpp graph   │  Liyab ExpertStore,
│ direct I/O + self-test,       │ via eval callback / id-remap │  PR #25294,
│ LFU-halving cache, prefetch,  │ op; kernels per ISA tier     │  BitNet/T-MAC (LUT),
│ hot list, stats               │ (v8.0 / dotprod / i8mm)      │  KleidiAI (higher tiers)
├───────────────────────────────┴──────────────────────────────┤
│ Device profile: clusters from cpufreq, HWCAP ISA flags,      │  Liyab device_detect,
│ storage QD curve, RAM bandwidth, budget, thermal APIs        │  LiteRT-LM affinity
├──────────────────────────────────────────────────────────────┤
│ Power manager: headroom forecast, pacing, core shedding      │  Liyab power_manager
├──────────────────────────────────────────────────────────────┤
│ Verification & evidence: artifact identity, per-extent or    │  Yours (keep and extend);
│ fs-verity integrity, layout contracts, golden logits per     │  DeepInfer/Liyab
│ kernel set, batch-invariance tests, run records              │  batch-invariance tests
└──────────────────────────────────────────────────────────────┘
```

### 11.1 Device profile and tiers

Measure once per device and cache the result:

- **CPU clusters** from `/sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq`, not hard-coded tables.
- **ISA tier** from `getauxval(AT_HWCAP/AT_HWCAP2)`: ARMv8.0 NEON (your SD665), +dotprod (SD 7-series and 8 Gen 1 era), +i8mm (8 Gen 2 and later). You already dispatch SHA-256 this way; do the same for matmul kernels so one binary serves every tier.
- **Storage curve:** MB/s at queue depth 1/2/4/8 for 256 KiB–4 MiB reads, direct and buffered, with the direct/buffered byte-equality self-test.
- **RAM bandwidth** with a simple streaming read, which gives the resident-weight floor per token.
- **Budget:** free memory now, swap/zram state, and for apps the per-app limit.
- **Thermal APIs available:** `AThermal_*` status (API 30) and headroom (API 31), ADPF (API 33).

### 11.2 Planner

Given the model's tensor index (your manifest already has it) and the profile, decide: which tensors are resident, the expert cache size (start at ~2× the per-token active set, cap by budget), I/O threads (2–4), prefill unit, compute threads, and whether the model should run at all. Report the plan, as Liyab's `memory_plan()` does, so a front end can tell the user "this model needs X GB to be usable here". Bonsai suggests the planner should also be able to say "a smaller resident model will be faster on this device".

### 11.3 Model generalization

Most architecture work is already done by llama.cpp; what is specific to you is the expert layer. To generalize it:

- Replace the `_exps` name heuristics with the consumer contracts you already bind from `llama_layer`, and handle fused `ffn_gate_up_exps` (one contiguous read per expert, as Liyab does) and per-expert scale tensors.
- Support larger-than-RAM *dense* models with a block pipeline (Liyab uses three slots: fetch → prepare → execute, with resident blocks spread through the stack).
- Keep architecture coverage evidence per model family, as your `ARCHITECTURE_COVERAGE.md` does.

### 11.4 Integrity without a startup hash

Two options, from cheapest:

1. **fs-verity.** Android uses fs-verity on ext4/f2fs; if your kernel and `/data` filesystem support it, enabling verity on the model file makes the kernel verify every page on read against a Merkle tree and makes the file immutable. Startup verification becomes reading a 32-byte digest. Check support first (it is not guaranteed on a phone that launched before Android 11), and note that direct I/O on verity files falls back to buffered reads on many kernels, which interacts with item 1 of the TL;DR.
2. **Per-extent digests in your manifest**, verified on first read and recorded in a bitmap, as the review's section 4.8 describes. Portable, and compatible with direct I/O.

Either way, verify at import (as Ollama verifies at pull) and stop rehashing 20 GB on every restart; that also makes process isolation (section 6) affordable.

---

## 12. Roadmap: what to build, in what order

| Phase | Work | Borrowed from | Expected effect (estimate) | Evidence to record |
|---|---|---|---|---|
| 0 | Fix the review's three bugs (latch, `/health` race, admission bound) | — | Safe to experiment | Two-request failure test |
| 0 | Measurements: storage QD sweep (direct vs buffered), `/proc/self/io`, RAM bandwidth, routing trace replay | Liyab's flash chart method | Picks parameters below | Device profile JSON |
| 1 | Expert store: (block, expert) entries, LFU with halving, direct I/O + self-test, 2–4 I/O threads, matrix-level split, stats incl. late/unused/stall | Liyab, PR #25294 | Expert bytes/token down substantially with ~1–2 GB cache; QD gains depend on the sweep | hits/late/misses/unused per token |
| 1 | Hook at `ffn_moe_topk` (eval callback or id-remap op) instead of inside `mul_mat_id`; cached experts first, shared expert during loads, router-order sum | Liyab, PR #25294, BigMoeOnEdge | Overlap; no barrier spinning | Bit-identical logits vs resident on a small model |
| 1 | Work-claiming thread pool, WFE spin then sleep, compute pinned to A73s, I/O on A53s | Liyab, LiteRT-LM | Better big.LITTLE use, less heat | Per-cluster frequency log |
| 2 | One-block-ahead prediction with immediate settlement; protect fresh loads | Liyab | Fewer stalls; protecting fresh loads was Liyab's single largest step (+71%) | Guess precision, late/unused counts |
| 2 | Context reuse + DeltaNet snapshots; then content-addressed prefix store on flash | Liyab, SGLang, vLLM, llama.cpp | Follow-up turns in seconds instead of re-prefilling | TTFT per turn |
| 2 | Cancellable, resumable chunked prefill | vLLM, LiteRT-LM | Deadlines and disconnects work | Cancel latency |
| 3 | Q4_K/Q6_K ARMv8.0 inner loop (vmlal accumulation), autotuned tiles | Review 5.2, BitNet | Lower compute share, mostly prefill | Bit-identical kernel gate |
| 3 | Thermal headroom pacing | Liyab | Stable sustained speed | 30-minute run |
| 3 | Jump-forward structured output for tool calls | SGLang | Fewer decode steps on JSON | Steps per tool call |
| 3 | Integrity at import (fs-verity or per-extent digests); hot-list warm start | Ollama, Liyab | Startup from minutes to seconds | Startup time |
| 4 | Device tiers (dotprod, i8mm kernels), dense larger-than-RAM pipeline, low-bit resident models | Liyab, KleidiAI, BitNet, Bonsai | Generalization | Per-device benchmark files |
| Later | Speculation with MTP, k ≤ 2, confidence × expert-cost scheduling | DeepInfer, Liyab, SpecFit/EcoSpec | Only once hit rate is high | Bytes per accepted token |

### A strategic note

Liyab already does most of phases 1–3 on flagship phones, and llama.cpp PR #25294 does much of phase 1 upstream. Rebuilding all of it in isolation would put you a year behind two active projects. The defensible position is the one the earlier review suggested: **the verified, exact engine for low-end ARMv8.0 Android**, built on upstream llama.cpp. Concretely, that means adapting Liyab's expert-store design (Apache-2.0, with attribution) or aligning with PR #25294's structure, contributing the ARMv8.0 kernel path and your SD665 measurements upstream, and keeping your verification and evidence layer as the part nobody else has.

---

## Appendix: where each borrowing lands in your repository

| Borrowing | Replace or extend in android-moe-engine |
|---|---|
| Expert store, LFU, direct I/O, stats | `src/residency/store.*`, `src/residency/file_source.*`, `MoeStreamStats` in `src/backend/moe_stream.hpp` |
| Router hook / id-remap, overlap, router-order sum | `patches/0001-ggml-moe-materializer.patch`, `materialize()` in `src/backend/moe_stream.cpp` |
| Thread pool and affinity | llama.cpp threadpool params from `apps/server/main.cpp` (`generate()`) |
| Prediction and fresh-load protection | New predictor beside `materialize()`; eviction policy in `Store` |
| Context reuse, snapshots, prefix store | `generate()` in `apps/server/main.cpp` (stop creating a context per request) |
| Cancellable prefill | `generate()` prefill loop; `request_moe_stream_cancel()` wiring |
| ARMv8.0 Q4_K/Q6_K kernels | A new patch against `ggml/src/ggml-cpu/arch/arm/quants.c`; `bench/moe_kernel_gate.cpp` for bit-equality |
| Thermal pacing | New module; sample only while generating |
| Integrity at import | `src/backend/artifact.*`, `src/common/sha256*`, manifest v3 in `tools/gguf_index.py` |
| Device profile and planner | `src/probe/probe.*` (already collects some of this), `configs/device-*.json` |
