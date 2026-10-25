# Open problems: android-moe-engine

Date: 2026-10-19  
Branch: `codex/hardening`  
Audited base: `7303320` plus hardening commits through `af6b6fb`  
Target: vivo S1 Pro, Snapdragon 665, 8 GB RAM, Android 12, rootless

This file is a research handoff. It lists the concrete problems blocking a
validated release, the evidence for each, and the questions that need an
answer. It is not a request to relax correctness or to replace the working
inference path.

## Current system

- Backend: pinned llama.cpp `ff5a23376725e5a889fa0bfb987b7541dc304da0`.
- Custom CPU `MUL_MAT_ID` materializer installs a compact expert tensor and
  remapped expert IDs at the real kernel boundary.
- Project owns the arena, leases, demand reads, LRU, cancellation, telemetry,
  and lifecycle records.
- One active CPU text generation. No GPU, concurrent sessions, second backend,
  UI, or OpenAI server.

### 35B measured result

Artifact: `Qwen3.6-35B-A3B-Q4_K_M.gguf`, 20,419,565,568 bytes,
SHA-256 `671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7`.
Architecture: qwen35moe, 40 layers, 256 experts, top-8, 34.66B parameters.

- Five runs at contexts 512, 1024, and 2048 produced identical tokens:
  `11751, 11, 264, 3177, 34756, 364, 1141, 8807`.
- Peak VmHWM: 2,159,888–2,190,576 KiB.
- Project arena: 28,311,552–37,748,736 bytes.
- Decode: 0.37–0.69 tok/s; TTFT: 8.8–17.8 s.
- Application reads: 9,999 calls, 5,897,650,176 bytes.
- Read time: 16.6–21.4 s of a 22–28 s generation.
- Store hits: 0.

The hardened Android binary still reproduces `11751, 11` with the existing
version 1 phone manifest. That manifest reports `manifest_version: 1` and
`identity_checked: false`.

### Phone memory after “Extended RAM”

```text
MemTotal       7,834,596 KiB
MemAvailable   4,849,928 KiB
Cached         3,172,380 KiB
SwapTotal      6,291,452 KiB (6.0 GiB)
SwapFree       5,962,292 KiB
```

The 35B process peak is about 2.19 GiB. Swap is almost entirely unused, so the
extra 2 GiB does not improve current inference throughput. It may improve
headroom for other apps, but it is flash-backed and can hurt an already I/O-bound
run if swapping starts.

## Gate status

| Gate | Status | Summary |
| --- | --- | --- |
| 1. Reproducible inference integration | **Passed locally** | CMake integration targets, stream/kernel/manifest/broken-adapter gates pass in Debug, Release, ASan+UBSan; Android ELF inspected. Hosted CI workflow added but not executed. |
| 2. Artifact-bound expert storage | **Failed** | v2 layout/count/duplicate/overflow/truncation checks pass, but artifact SHA is only declared; same-size wrong content and wrong in-bounds offsets are not detected. |
| 3. Structured failure and cancellation | **Failed** | Cancellation reaches reads and supervisor classifies signals, but materializer/store failures still call `std::abort()`. |
| 4. Prefill plans and admission | **Failed** | No safe prefill refusal or microbatch plan; union above `--moe-slots` aborts. |
| 5. Numerical correctness | **Blocked** | Comparator is fixed; no complete quantized MoE fixture and no host with >20.4 GB usable memory for a 35B reference. |
| 6. Practical phone campaign | **Failed** | Existing runs are short (5 input/8 output tokens); no corpus, 128–256-token outputs, PSS, sustained load, or controlled memory pressure. |

Full gate record: [RELEASE_READINESS.md](RELEASE_READINESS.md).

## P0 problems

### P0-1. No full-resident 35B reference

**Evidence**

- 20.4 GB of tensor bytes.
- Phone has ~7.8 GB RAM + 6.0 GB swap = ~13.8 GB total.
- No identified host has more than 20.4 GB usable memory with the same pinned
  CPU backend.
- Correctness currently rests on bitwise F32 kernel/runtime gates plus
  deterministic repeated tokens. That is repeatability and kernel parity, not
  full-model reference parity.
- A 14.3B Q4 artifact is 9.49 GB; it could theoretically fit in RAM+swap but
  would thrash and has not been tested as a reference.

**Questions for research**

1. What is a defensible reference strategy when no host can hold the full
   artifact in RAM?
2. Is a block-wise or layer-wise reference on a larger host with the same
   quantized artifact and pinned backend acceptable if it exercises the full
   graph and produces final logits? What must be recorded to make it credible?
3. Can a reference be produced from a read-only block device or memory-mapped
   host file with controlled page-cache state, and how should that be labeled?
4. What is the minimum useful reference: full logits, teacher-forced token
   likelihoods, router IDs, per-expert outputs, or final generated tokens?
5. How should cross-ISA arithmetic differences be bounded before inspecting
   candidate results?
6. If no reference is possible, what exact claim may the 35B result support
   without overclaiming?

**Acceptance needed**

A reproducible command that runs the unmodified pinned backend with the same
artifact, prompt tokens, sampler, and context on an identified host, plus a
declared comparison method and tolerance.

### P0-2. Artifact identity is declared, not verified

**Current behavior**

- `tools/gguf_index.py` writes manifest v2 headers:
  `artifact_bytes`, `artifact_sha256`, `tensor_count`, `layout_fnv1a64`.
- The engine verifies file size, row count, layout hash, duplicates, overflow,
  alignment, and extents.
- The engine does **not** recompute SHA-256. A same-size wrong artifact and a
  wrong but in-bounds tensor offset are accepted.
- Existing phone manifests are v1 and carry no identity; the hardened 35B
  regression explicitly reports `identity_checked: false`.

**Why it is difficult**

- Hashing 20.4 GB on the phone adds startup time and flash reads.
- The model is on writable `/data/local/tmp`; file replacement/truncation
  between validation and use must be handled.
- Rootless Android has no easy global-cache control.
- A caller-supplied hash string is not proof.

**Questions for research**

1. What is the cheapest sound artifact-identity scheme on Android for a 20 GB
   file: full SHA-256, chunked Merkle tree, fs-verity, dm-verity, or a
   read-only bind mount plus inode/size/mtime binding?
2. Can fs-verity be enabled on a rootless app-private file? What Android
   versions/filesystems support it?
3. Is hashing only the expert extents sufficient, or can non-expert and
   tensor-layout tampering still alter results?
4. How should the engine bind the manifest to the exact file across
   validation and later reads: open file descriptor + `fstat` + generation
   counter, or reopen per read?
5. What is the measured SHA-256 throughput on this phone and the resulting
   startup cost?
6. How can a wrong in-bounds offset be detected without hashing the whole
   file? Is a per-tensor range hash or Merkle leaf enough?
7. What is the right fallback when identity cannot be verified: refuse
   streaming, or run with an explicit `identity_unverified` status?

**Acceptance needed**

A same-size wrong artifact, a wrong in-bounds offset, a truncated file, and a
replaced file are all refused before any affected weight reaches a kernel.

### P0-3. Expected failures still abort

**Current behavior**

- `install_moe_stream` now returns a structured failure for malformed
  manifests.
- Cancellation is wired to `request_moe_stream_cancel()`; the store checks it
  between expert slots and read chunks.
- The supervisor classifies child exit, SIGABRT, and timeout.
- But `fail()` in [src/backend/moe_stream.cpp](../src/backend/moe_stream.cpp)
  still calls `std::abort()` for store, arena, metadata, and unindexed-tensor
  failures. The CPU kernel has no error return from `MUL_MAT_ID`.

**Questions for research**

1. What is the smallest safe patch to propagate a materializer failure through
   the pinned llama.cpp CPU scheduler without leaving worker threads waiting?
2. Can `ggml_compute_forward_mul_mat_id` set a graph-level error or abort
   status that `ggml_backend_sched_graph_compute_async` returns as
   `GGML_STATUS_FAILED`? Where are the barriers and joins?
3. Is returning a sentinel tensor or zero output ever safe? The project rule is
   no zero-substitution.
4. How should the hook distinguish “not my tensor” from “my tensor failed”?
5. What drain order is required: stop workers, join, release lease, then free
   arena? Which state can be reused after cancellation?
6. Is a supervised child process with a structured “aborted” record an
   acceptable alternative until kernel propagation exists? What must be
   recorded to distinguish inference failure from OOM without OS evidence?

**Acceptance needed**

Refusal, cancellation before/during materialization/prefill/decode, short read,
EINTR, permanent read failure, and backend failure all produce a bounded
structured outcome, no hung workers, no partial success label, and a correct
fresh request afterward.

### P0-4. No safe prefill plan

**Current behavior**

- Prefill is chunked by `--batch-size` only.
- The materializer computes the routed-expert union for the current tensor and
  aborts if it exceeds `--moe-slots`.
- The 35B run used a 5-token prompt, 8 output tokens, batch 64, slots 64,
  observed union 37.
- Top-8 routing gives a conservative union bound of
  `min(n_expert, prompt_tokens * top_k)`; at 9+ prompt tokens with slots 64
  this bound exceeds capacity even if the actual union might fit.
- No 63/64/65-token boundary test, slot capacity minus-one/at/plus-one test,
  or microbatch parity test exists.

**Questions for research**

1. What is the correct supported prefill plan for a top-8, 256-expert model
   with 64 slots? Is a conservative refusal based on
   `tokens * top_k > slots` acceptable, or is a microbatch schedule required?
2. If microbatches are used, how are attention state, token positions,
   recurrent state, and sampler state preserved across chunks?
3. What is the maximum microbatch size that guarantees the union fits, given
   routing is data-dependent?
4. Can routing be computed cheaply before expert materialization to choose a
   safe chunk, without changing top-k or accumulation?
5. How should the engine report “resource refusal” without producing partial
   output that looks successful?
6. What parity evidence is required to promote a microbatch plan for a
   specific architecture?

### P0-5. Storage-read latency dominates

**Evidence**

- 5.90 GB application reads over 9,999 calls.
- 16.6–21.4 s in `pread` for 22–28 s of generation.
- A 1,024-slot global LRU cache reduced bytes 23.5% but increased read time
  5.0 s, reduced decode 18%, and added 539 MiB RSS; it was rejected.
- 64-slot demand reads remain the default. Store hits are zero.
- Physical flash traffic was not measured; page-cache state is uncontrolled.

**Update (core optimization series):** question 1 is partly answered. Queue depth matters on this storage (four readers cut the 32-token request 31.5%), and capacity partitioned per layer cut bytes and time (layered cache, 32 slots per layer, -39.5%); the 1,024-slot global cache above remains a measured loss. Dropping the kernel page-cache copy was 8.4% faster. Question 5 remains open: only the current layer's reads are overlapped. See [the report](CORE_OPTIMIZATION_REPORT.md) and `evidence/core-opt/`.

**Questions for research**

1. Why did fewer bytes not reduce read time? Is the bottleneck per-read
   latency, page-cache eviction, storage queue depth, or CPU copy?
2. What is the best rootless measurement of physical I/O and page-cache
   behavior on this phone?
3. Is per-tensor caching, read-ahead, `posix_fadvise`, `preadv2`, or a
   different I/O scheduler likely to help an expert set that changes each
   token?
4. How should a cache be sized and keyed to exploit the measured 237–612-key
   consecutive overlap without the 604 MB RSS cost?
5. Would a small prefetch of the next layer’s expert extents help, or does it
   compete with the current forward’s reads?
6. What is the expected benefit ceiling if compute alone were free?

### P1 problems

#### P1-1. No complete quantized MoE fixture

The existing runtime gate uses F32 expert tensors. It does not exercise
quantized projection blocks, packed strides, or a complete model graph. Research
needed: the smallest valid GGUF qwen2moe/qwen35moe fixture with a tokenizer and
quantized experts that llama.cpp can load on this host, plus a parity plan for
router IDs, expert outputs, and logits.

#### P1-2. Phone campaign is not practical

Existing runs use five prompt tokens and eight output tokens. Missing:
ordinary instruction prompts, code, multilingual text, long input, 128–256
generated tokens, PSS sampling, thermal-controlled alternating trials, and a
30-minute sustained run. Rootless PSS and physical-I/O measurement methods are
needed.

#### P1-3. Release integration gaps

- Hosted CI workflow exists but has not run.
- Upstream llama.cpp libraries are not sanitizer-instrumented; only project
  adapter code is.
- Existing phone manifests are v1.
- `ame run` is a lifecycle supervisor, not a model-option frontend.
- No OpenAI-compatible API or concurrent sessions.

## Constraints for any proposed solution

- Keep one pinned production backend and one active CPU text generation.
- Do not skip experts, change top-k routing, truncate layers, zero missing
  weights, or silently alter the model.
- Do not add GPU execution, UI, concurrent generation, speculative prefetch, or
  a second backend.
- Do not purchase compute or upload weights to an external service without
  owner authorization.
- Do not claim a hard process memory cap while backend workspace, KV/recurrent
  state, page cache, and load peaks remain uncovered.
- Preserve token IDs and evidence; keep failed runs and null unsupported
  measurements.
- Prefer a structured refusal over a partial success.

## What to return from the research

For each P0/P1, please provide:

1. Recommended approach and why it is sound on rootless Android.
2. Exact code locations or patch points in the pinned backend.
3. Memory, time, and flash-I/O cost.
4. Failure modes and concurrency/worker-drain implications.
5. A reproducible experiment or test.
6. Acceptance criteria and what would make the approach unacceptable.
7. If blocked, the exact missing resource or API.

Prioritize P0-2 (identity), P0-3 (structured failure), and P0-4 (prefill).
P0-1 may be impossible on the available hardware; an honest blocked conclusion
is acceptable if it states the exact missing resource and a ready-to-run
reference command.

## Reference paths

- [Release gates](RELEASE_READINESS.md)
- [Backend feasibility](backend-feasibility.md)
- [Architecture](ARCHITECTURE.md)
- [Validation contract](VALIDATION.md)
- [Materializer](../src/backend/moe_stream.cpp)
- [Residency store](../src/residency/store.cpp)
- [Runtime patch](../patches/0001-ggml-moe-materializer.patch)
- [Baseline harness](../bench/llama_baseline.cpp)
- [GGUF index](../tools/gguf_index.py)
- [35B aggregate](../evidence/phase4/qwen36-stream-aggregate.md)
- [Hardening regression](../evidence/hardening/qwen36-hardened-regression.json)
