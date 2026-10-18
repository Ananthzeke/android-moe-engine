# Implementation plan

This plan delivers a small working engine first, then a falsifiable storage-offloading experiment. Large-model completion is a research outcome, not a promised milestone result. Use [the architecture](ARCHITECTURE.md) as the ownership contract and [the validation contract](VALIDATION.md) for every measurement.

## Phase 0: Reproduce the target and pin the build

Deliver a host CMake build, an Android ARM64 cross-build, a minimal `ame probe`, and a machine-readable dependency lock. Probe before choosing ISA flags. Do not compile the whole program for instructions unsupported by this phone.

Start by evaluating these inspected revisions. A newer revision requires a recorded reason and the same gates:

| Dependency | Inspected commit |
| --- | --- |
| MNN | `dbf9ba61b08106a3b43426e295d1a29cb41cc8fe` |
| llama.cpp | `ff5a23376725e5a889fa0bfb987b7541dc304da0` |

The first implemented build pins its NDK, compiler, converter, build flags, and backend SHA. Host sanitizers and warnings cover project code. Upstream warnings are recorded separately. Target Android API 31 initially and use 64-bit file offsets for large weights. Exercise files larger than 4 GiB without allocating their full contents.

Exit checks:

- A host test executable runs, and the produced Android ELF has the expected architecture and linkage.
- `ame probe` runs on the physical phone and reports allowed memory, CPU, storage, and OS observations without serials or private files.
- Running without ADB produces an explicit unavailable-target result. It does not fabricate a device profile or treat an emulator as this phone.
- The dependency lock and build commands reproduce the selected sources.

## Phase 1: Establish baseline and prove the backend boundary

First run one verified small model candidate at context 512. Prefer the catalog's 2B baseline in an actually supported low-bit artifact. Preserve tokenization, chat template, and generation settings. If that artifact fails, use a smaller real supported artifact and record the substitution.

Then inspect routing, expert construction, packing, kernel access, and release in the selected backend. Build a tiny synthetic or reduced MoE fixture whose complete reference fits host and phone memory. Record whether the fixture was trained, converted, or synthetic. It establishes integration mechanics, not large-model quality.

Deliver `docs/backend-feasibility.md`, a small prototype patch, fixture provenance, and reference evidence. Populate this matrix with source locations and executed results:

| Question | Required proof |
| --- | --- |
| When are selected expert IDs available? | Trace before the consuming operator, for decode and prefill. |
| What owns every expert's weights? | Trace retained modules, constants, packed buffers, and copies through load and execution. |
| Can one expert's backing be replaced or released? | Redirect a valid expert extent and prove reference parity without materializing all experts. |
| How are expert bytes laid out? | Validate all projections, scales, blocks, strides, fused tensors, and shared weights. |
| When may the backing be reused? | Trace completion of all worker tasks that reference the bytes. |
| What is the smallest live compute unit? | Measure simultaneous extents and workspace for decode and bounded prefill. |
| Are all backend allocations accounted for? | Show allocator coverage, transient peaks, and any exclusions. |
| Does the unchanged small model still work? | Run deterministic reference comparison after the patch. |

MNN is provisional. If it cannot pass with a small maintainable change, repeat the gate in llama.cpp and record the backend decision before building residency. If both fail, finish the small engine and record the MoE blocker. Do not implement a fake callback or duplicate backend stack to make the gate appear green.

## Phase 2: Implement bounded residency

Implement the validated artifact index, checked file extents, one reservation owner, aligned arena, byte-based LRU, move-only leases, and synchronous demand reads. Start with one quantized layout established by phase 1. Use the same layout and fixture in the resident reference and candidate.

Exit checks:

- A forced-churn fixture produces reference-equivalent outputs while covered allocations remain within the declared cap.
- Zero experts, duplicate IDs, missing extents, out-of-range IDs, overflow, misalignment, short reads, and I/O errors fail cleanly.
- A lease prevents eviction until every consuming task completes. Cancellation and partial acquisition release all pins.
- A working set one byte over budget returns `ResourceLimit` without over-allocation or self-deadlock.
- Prefill uses a proven bounded scheduling unit. It cannot request the union of an unbounded prompt's experts.
- Dense execution remains functional. A streaming-disabled run follows the reference behavior.

A hard overall owned-memory cap remains conditional on complete backend allocation coverage. Keep the arena guarantee and coverage report separate until that proof passes.

## Phase 3: Make physical-device comparison reproducible

Implement `ame run`, `ame compare`, per-run records, a bounded token-output log, and a report generator. Run controlled reference and candidate comparisons on the phone using the same artifact, prompts, and generation settings.

The first campaign uses one warm-up and three measured trials per variant per cache condition, alternating order. Use a fixed declared prompt set and token count. Increase repetitions only to resolve observed variability. Record failed trials rather than replacing them silently.

Exit checks:

- Correctness, provenance, latency phases, memory scopes, I/O semantics, and run termination satisfy [validation](VALIDATION.md).
- Reports are generated from raw records and include trial count, spread, failed runs, and missing counters.
- Best-effort cold-process runs disclose uncontrolled OS storage cache state.
- Cancellation, repeated open/close, and process interruption leave interpretable records and no persistent corrupted model artifact.
- A negative result remains a completed experiment. Keep the simpler reference as default unless a stated memory benefit justifies a measured latency cost.

## Phase 4: Attempt one large MoE artifact

Choose the first large candidate from [the catalog](../configs/model-candidates.json) using actual converter support, numerical reference, artifact footprint, and minimum live set. Gemma 4 26B-A4B is the first hypothesis because its total weight count is smaller. Do not force that choice if source evidence favors Qwen3.6.

Preflight source, conversion, transfer, disk-cache, and temporary storage. Verify immutable hashes before timing. Begin CPU text-only at context 512 and a short fixed output. Attempt 1,024 and 2,048 only after admission and the previous run pass. Use a bounded declared trial duration and record timeout if exceeded.

Exit checks:

- At least one attempted real large-model run has a complete outcome: success, resource refusal, unsupported conversion, timeout, backend failure, or interrupted/unknown termination.
- A successful run has reference-correct token or logit evidence. If no full reference fits the phone, produce it on a larger identified host with the same artifact and pinned CPU backend. Label cross-device numerical differences explicitly.
- The report names what limits execution and distinguishes storage wait, compute, memory pressure, and unavailable evidence.
- Model support is promoted only for the exact tested artifact and runtime combination. A family name alone stays a candidate.

## Phase 5: Improve only the measured limiter

Choose one intervention from phase 4 evidence. Possibilities include expert layout, cache allocation, prefill microbatch size, kernel packing reuse, or thread count. Test one change at a time with correctness and resource gates.

Only after CPU evidence earns the cost, consider an Android host app, GPU execution, prefetch, or KV offload. Revisit architecture when the same workaround recurs across modules. Do not accumulate two production engines or introduce a generic plugin framework in the MVP.

## Delivery units

Commit phases as reviewable units. Each unit includes what changed, the commands actually run, evidence locations, limitations, and the next unit. Do not commit weights or private raw device logs. A build success establishes buildability, not target performance.

The initial repository preparation is complete when this plan, the architecture, source evidence, and DeepSeek prompt are present and the scaffold check passes. It does not imply any engine phase has passed.
