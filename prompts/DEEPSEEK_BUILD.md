# Build android-moe-engine

You are implementing this repository as a systems engineer. Build a working CPU inference engine for memory-constrained Android, then establish whether exact storage-backed expert execution makes a larger MoE model feasible. Act on the repository and available physical device. Deliver implementation and reproducible evidence, not another architecture proposal.

## Read the contract

Read these files before editing:

1. `AGENTS.md` and `README.md`.
2. `docs/ARCHITECTURE.md` and `docs/DESIGN_DECISION.md`.
3. `docs/IMPLEMENTATION_PLAN.md` and `docs/VALIDATION.md`.
4. `docs/RESEARCH.md`, `docs/MODELS.md`, and the three JSON files in `configs/`.

The repository is currently a design scaffold. It has no inference implementation or benchmark results. Commands and C++ interfaces in the architecture are proposed project APIs, not existing backend APIs. Inspect source before writing integrations.

## Target and objective

The first target is the owner's vivo S1 Pro, reported as vivo 1920, with Snapdragon 665/SM6125, ARM64, Android 12/API31, and an 8 GB RAM configuration. ADB debugging works in the originating environment. Termux is installed. A prior snapshot found about 5.1 GiB MemAvailable and 95 GiB rounded available data storage. Refresh these values. They are not guaranteed resources.

The ADB shell had UID 2000, no available `su`, a locked bootloader, and green verified boot. Build without root. Keep device serials and private phone contents out of repository artifacts. Do not root the phone, unlock its bootloader, flash firmware, disable thermal protection, or change unrelated device settings.

The objective is an engine with explicit memory admission and expert residency. A script that enables mmap is a useful baseline but does not complete the expert-streaming objective. Large-model success and useful speed are unproven. Preserve a useful small-model path if large-model offload fails.

## Model candidates

Use `configs/model-candidates.json` as the candidate registry. Begin with a verified quantized Qwen3.5 2B artifact, then 4B if feasible. Gemma 4 E2B is an intermediate option, but its embedding-inclusive footprint is larger than its effective-compute name.

The main large experiments are Gemma 4 26B-A4B and Qwen3.6-35B-A3B. Choose order from actual conversion support and measured live memory. Qwen3.8-27B is dense and deferred. Resolve the intended instruction-tuned artifact variant and exact source revision. Do not infer conversion compatibility, quantization correctness, or Android support from a model-family announcement.

## Backend decision

Evaluate MNN first at `dbf9ba61b08106a3b43426e295d1a29cb41cc8fe`. The initial alternative is llama.cpp at `ff5a23376725e5a889fa0bfb987b7541dc304da0`. These revisions were inspected, not built or verified. Record a justified newer pin if an identified defect or missing capability requires it.

Reuse one backend's tokenizer, graph execution, routing, attention, recurrent-state semantics, numerical kernels, and native artifact format. Our contribution is validated indexing, memory admission, expert residency, I/O, lifecycle, and trustworthy evaluation. Keep a small identifiable backend patch, not two production adapters or a standalone replacement kernel stack.

In inspected MNN source, `express/module/MoEModule.cpp` obtains selected expert IDs before expert submodule execution. Its constructor also retains all expert modules. Prove that module/tensor backing can be constructed lazily, redirected, and released with bounded peak memory. Merely observing routing is not that proof. Trace packing and duplicate retained weights.

In inspected llama.cpp source, `ggml/src/ggml-cpu/ggml-cpu.c` contains the CPU expert matmul path, and `src/llama-mmap.cpp` has conditional eager population and lazy ranges. Do not code against stale file locations or assume all mmap loading eagerly faults every weight.

## Execute the phases

Create a concise task checklist. Implement phases 0 through 4 in order, completing each phase's checks before moving on. Proceed autonomously on routine reversible work. Do not stop after writing a plan or ask whether to continue each phase. If a prerequisite is genuinely unavailable, complete unaffected work, record the exact blocker, and report the smallest missing input.

### Phase 0: Build and inspect

Create a host-testable C++20 core, CMake build, dependency lock, and native ARM64 CLI. Resolve and pin the actual Android NDK and compiler. Probe supported CPU instructions before setting ISA flags. Inspect the resulting ELF and linkage. Implement a read-only `ame probe` and run it on the connected phone. If no phone is accessible, label all work host-only and preserve the physical-target checks as incomplete.

### Phase 1: Baseline and expert-lifetime experiment

Run a real supported small model at context 512 with recorded prompt, token IDs, artifact hashes, and build provenance. Then build a tiny MoE fixture whose complete reference fits available memory.

Populate the feasibility matrix from `docs/IMPLEMENTATION_PLAN.md`. Demonstrate router visibility, expert tensor layout, bounded backing lifetime, kernel completion, decode/prefill working sets, and backend allocation coverage. Redirect a valid expert's storage and compare numerical results to the unchanged resident reference. Record source locations and actual peak allocations.

If MNN fails, evaluate the same gate in llama.cpp before implementing residency. Select one production backend and update the decision record. If both fail, retain the verified small CPU engine and report expert streaming blocked with evidence. Never implement a mock hook or silently omit experts.

### Phase 2: Bounded residency

Implement overflow-safe artifact validation, a model-owned immutable expert index, one allocation/reservation owner, bounded aligned storage, byte-based LRU, and move-only leases. Expert entries include all needed projections, scales, strides, and packing metadata. Shared weights have separate explicit ownership.

Acquire a proven finite compute unit. Deduplicate expert IDs, reserve before reading, handle short reads/EINTR/errors, and evict only unleased entries. Keep backing valid until every consuming task completes. Return a resource error when the live set cannot fit. Avoid waiting for memory held by that same unit.

Start with one synchronous I/O operation and no prefetch. Cancellation checks occur between bounded reads and compute units. A read's byte limit is not a wall-clock interruption guarantee. Join all work before buffer destruction. Keep routing, arithmetic, expert count, and accumulation semantics unchanged.

Prefill's selected-expert union can exceed decode requirements. Prove bounded scheduling with parity, or reject that configuration. Do not pin the entire union and call the cache bounded.

### Phase 3: Evaluation and lifecycle

Implement the validation contract and a report generator. Record actual token IDs, correctness evidence, separate load/prefill/TTFT/decode phases, covered allocation peaks, RSS/PSS scope, logical expert bytes, application-read bytes, available physical I/O counters, thermal state, and failures.

Compare the same artifact and settings in reference and candidate. Start with one warm-up and three measured trials per condition, alternating order. Do not benchmark conversion or hashing inside inference timing. Distinguish fresh-process runs from proven cold OS file-cache runs. Rootless Android normally cannot establish the latter.

Write a start record before allocation and atomic final results afterward. A supervisor classifies missing final results as interrupted/unknown unless stronger OS evidence exists. Test cancellation, memory pressure where accessible, repeated sessions, partial read failure, and process interruption.

### Phase 4: One large-model experiment

Preflight source, converted artifact, transfer, temporary, repack, and disk-cache space before downloading a large model. Validate exact file hashes and conversion reference. Start CPU text-only at context 512 with short output and an explicit trial timeout. Increase context only when admission and prior checks pass.

If a fully resident reference cannot fit the phone, obtain it on a larger identified host using the same quantized artifact. Do not equate host success with device performance. Record cross-device numerical limitations. Report successful completion, resource refusal, conversion failure, timeout, backend error, or interrupted outcome honestly.

Only after these phases, choose one measured limiter to optimize. Keep unsupported optional GPU, QNN, multimodal, UI, KV-offload, and multi-session work out of the MVP.

## Hard contracts

- No expert skipping, routing changes, random/zero substitution, layer truncation, or unannounced model changes.
- Project cache, staging, metadata, alignment, backend packing, and transient load copies must be accounted for. A cache cap alone is not an overall engine cap.
- Claim a hard covered allocation limit only after proving its accounting scope. Treat RSS, PSS, OS page cache, and system availability separately.
- Bound metadata counts and arithmetic before allocating parser structures. Validate large file offsets, quant blocks, split shards, overlapping/aliased layout rules, and immutable file identity.
- Reset request-local KV, recurrent state, RNG, and sampler state for independent requests. Permit only one active generation in the MVP.
- Use deterministic exact comparisons where possible. Declare floating-point tolerances before observing candidate results, and never loosen them to hide a failure.
- Keep source commits, model revisions, converter and artifact hashes, binary hashes, toolchain identity, commands, raw evidence, and unavailable-counter reasons.
- Do not manufacture speed, quality, support, successful builds, completed tests, or device runs.

## Working style and delivery

Keep ownership close to the architecture. Avoid a generic plugin system, multiple overlapping configuration layers, per-token heap churn, or a second universal tensor format. Build focused behavioral tests for implemented failure and lifetime paths. Leave GPU and app UX until CPU evidence justifies them.

Commit small reviewable implementation units. Update affected architecture contracts and `docs/DESIGN_DECISION.md` when a source experiment proves a necessary deviation. Save negative results. Keep weights, secrets, and raw private device logs outside Git.

Finish with a self-contained report containing:

1. What now runs and which engine phase actually passed.
2. The chosen backend and exact model artifact identity.
3. Build/test commands executed and evidence paths.
4. Physical-phone results with trial count and spread, or an explicit statement that no physical run occurred.
5. Memory-accounting coverage, measured tradeoffs, failures, and remaining blockers.
6. The next concrete implementation unit.

Do not describe this repository as a completed 30B phone engine unless a real target run establishes that claim.
