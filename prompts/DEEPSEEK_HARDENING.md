# Harden android-moe-engine for a validated release

You are continuing an existing C++20 Android CPU inference project. Implement and test the hardening work below. Deliver working code and reproducible evidence. Do not stop after producing a plan or updating documentation.

The goal is a reliable single-generation engine for the owner's vivo S1 Pro, followed by an honest assessment of whether the 35B path is useful. A release candidate needs correctness, bounded resource behavior, safe failure, and realistic physical-device tests. No finite test campaign proves universal reliability. State the exact tested configurations and remaining limits.

## Establish the starting point

Read `AGENTS.md`, `README.md`, `CONTRIBUTING.md`, `docs/CODING.md`, `docs/ARCHITECTURE.md`, `docs/IMPLEMENTATION_PLAN.md`, `docs/VALIDATION.md`, `docs/DESIGN_DECISION.md`, and `docs/FINAL_REPORT.md`. Inspect the model catalog, dependency lock, backend patches, actual code, and raw evidence. Several documents contain stale statements. Resolve contradictions against source and evidence instead of accepting the most optimistic claim.

The audited local revision was `7303320e2e210004509319edd8b4d8fd3a6a421e`. At audit time, published GitHub main was `e5b4a0404b773eefc84b85ca5f619114b282494f`, 20 commits behind. Inspect the current worktree, branches, and remotes before editing. Preserve all existing work. If you only have the older remote revision, identify the missing implementation instead of rebuilding an imagined version. Keep new work on a `codex/` branch and commit reviewable units. Do not force-push, overwrite another agent's work, or publish to main as part of this prompt.

The implementation uses pinned llama.cpp at `ff5a23376725e5a889fa0bfb987b7541dc304da0`. It has a real CPU `MUL_MAT_ID` materializer, expert leases, demand reads, an arena, and a separate inference harness. Retain this approach. Do not add a second backend, GPU execution, UI, speculative prefetch, concurrent generation, or a new universal model format.

The recorded 35B results use five prompt tokens and eight output tokens. Context settings of 512, 1024, and 2048 describe capacity. They do not establish long-prompt reliability. Repeated tokens establish repeatability, not reference correctness. The existing host verifier passes nine tests per configuration but does not compile or execute the streaming integration.

Create a checklist with acceptance gates and evidence paths. Run the existing verifier first. Complete each gate before dependent work. Continue independent fixes when an external prerequisite is missing. Never mark a blocked gate passed.

## Gate 1. Make the real inference path reproducible and testable

Integrate the streaming adapter, pinned backend patches, harness, and numerical gates into explicit supported build targets. Preserve the existing offline verification command. Provide a separate documented dependency bootstrap that verifies the source revision, patch identities, toolchain, and build options. Verification must not silently download dependencies or use unverified prebuilt libraries.

Add CI coverage for the actual adapter and numerical gates. Run project integration code under Debug, Release, ASan, and UBSan on the host. Record upstream sanitizer coverage and exclusions explicitly. Build the Android ARM64 artifact with the pinned NDK and inspect its ELF and ISA requirements. Avoid `-march=native` and unsupported target instructions.

Acceptance requires a clean build directory, exact reproduction commands, passing integration tests, and evidence that a broken adapter causes the integration check to fail. The standard foundation tests alone cannot satisfy this gate. Downloaded model weights are not required for small generated fixtures.

## Gate 2. Bind expert storage to the actual model

Audit `src/backend/moe_stream.cpp`, `tools/gguf_index.py`, and model loading. The current manifest parser accepts offsets without proving they match authoritative GGUF tensor locations. It ignores comment-only identity data. Missing tensor names fall back to mapped weights.

Make the GGUF metadata and verified artifact identity authoritative. Bind every index to the exact artifact or shard set and its immutable tensor layout. Validate complete dimensions, quantization blocks, strides, packing, extents, alignment, and supported aliases. Check arithmetic before multiplication or allocation. Bound parser counts and metadata sizes.

Validate every routed expert tensor before generation. A streaming request must refuse an incomplete, duplicate, stale, or incompatible index. Never silently switch an unindexed routed tensor to resident execution. Preserve explicitly declared resident shared tensors and non-expert weights.

Use stable backing identity across validation, mappings, and reads. Address file replacement and truncation between checks and use. Hashing must occur outside inference timing and have recorded provenance. Do not claim that a caller-supplied hash string verifies a file.

Acceptance tests must include a valid artifact, a same-size wrong artifact, a wrong but in-bounds tensor offset, missing and duplicate entries, unsupported layouts, malformed counts, overflow, truncated backing, and offsets beyond 4 GiB. Invalid input must refuse before affected weights reach a kernel.

## Gate 3. Handle resource limits, cancellation, and failures safely

Replace expected resource and I/O failure paths that call `std::abort()` with structured outcomes. Trace the patched CPU worker barriers before changing error propagation. No worker may return while peers wait forever or consume incomplete weights. Drain consumers before releasing leases or storage.

Connect request cancellation to storage acquisition and bounded reads. The current materializer creates a fresh cancellation token that never receives the request's stop signal. Define cancellation observation points and measure their behavior. A synchronous read is not a hard wall-clock cancellation guarantee. Retain an independent supervisor timeout for an unresponsive process.

Preserve one active generation. After cancellation or backend failure, require reopen unless an explicit reset path is proven equivalent to a fresh request. Process isolation is acceptable if documented and tested. Do not introduce a reusable SDK merely to claim lifecycle completion.

Test cancellation before reads, during materialization, during prefill, and during decode. Inject short reads, EINTR, permanent read failure, backend failure, and termination around lease boundaries. Prove pin rollback, safe teardown, bounded timeout behavior, and a correct fresh request afterward. Same-process reset needs separate proof if offered.

The supervisor must retain bounded logs and complete structured outcomes for refusal, cancellation, timeout, signal, spawn failure, and backend failure. Distinguish runner exit from inference success. Reconcile a start record with no result as interrupted or unknown. Do not label a signal as a low-memory kill without OS evidence. Check process-tree cleanup where subprocesses are supported. Make record durability and exclusive run-directory ownership match the validation contract.

## Gate 4. Prove supported prefill plans and memory admission

The current harness accepts a token batch size, then aborts if its routed expert union exceeds the slots. Implement a supported prefill plan or reject the request safely. Use a proven conservative bound when useful. If an execution unit still exceeds capacity, return a resource refusal without partial output being labelled successful.

Preserve routing, top-k, all experts, accumulation semantics, attention, recurrent state, and token positions. Never drop experts, substitute zeros, truncate layers, or silently trim the prompt. Rebatching requires parity evidence for the affected model architecture.

Test real prompt lengths around batch boundaries, including 63, 64, and 65 tokens where batch 64 is supported. Test prompt plus output at context minus one, context, and context plus one. Force expert unions at slot capacity minus one, capacity, and capacity plus one. Cover duplicates, hits, misses, evictions, and multiple projections.

Implement fresh memory and storage observations before admission. Reserve project-owned allocations before materialization. Account for alignment, index metadata, remapped IDs, staging, cache payload, packing copies, and overlapping old and new state. Describe backend workspace, shared weights, KV or recurrent state, and load peaks. Define a conservative admission policy where exact backend accounting is unavailable.

Keep separate measurements for covered owned bytes, process RSS, sampled PSS, system availability, application reads, and physical I/O. An arena cap is not an engine or process cap. Unsupported measurements are null with a reason. Do not promise a total hard cap while exclusions remain.

## Gate 5. Establish numerical correctness and trustworthy comparisons

Fix `src/compare/compare.cpp` before using it as an acceptance oracle. It currently casts JSON numbers to integers, allowing fractional token IDs such as 1.1 and 1.9 to compare equal. Reject fractional and out-of-range IDs without undefined conversion. Validate required fields, supported schemas, and ambiguous duplicate keys. Prevent incompatible run provenance or settings from being presented as a controlled performance comparison. Distinguish equality between two records from correctness against a trusted reference.

Create a small complete quantized MoE fixture whose fully resident reference fits this host. Exercise actual model loading, graph execution, prefill, decode, routing, quantized projections, and state. An F32 matmul-only fixture cannot replace this test. Compare the unmodified pinned backend against streaming with the same artifact, input tokens, sampling, and supported execution plan. Compare router IDs, intermediate expert outputs, and logits where practical, alongside generated token IDs.

Declare exact comparisons or justified tolerances before examining candidate differences. Account explicitly for different arithmetic order or ISA. Record top-1 margins and divergences. Never widen tolerances after a failure merely to pass.

For the real 35B artifact, produce an end-to-end reference on an available identified host with sufficient resources using the same artifact and pinned backend. Plan its memory and storage first. Do not purchase compute or transfer weights to a new external service without owner authorization. If no suitable host exists, finish all independent work and mark this gate blocked with the exact missing resource and a ready-to-run reference command. A small-model parity pass does not promote 35B to reference-verified support.

## Gate 6. Test practical workloads on the physical phone

Refresh device availability, memory, free storage, battery, and thermal observations. Keep all device protections enabled. Do not change root, bootloader, ROM, or unrelated settings. Use bounded trials and stop safely on failure or unacceptable pressure. Preserve every failed or excluded trial with its reason.

Commit a reproducible workload corpus and configuration matrix before the campaign. Cover ordinary instruction prompts, code, multilingual text, and long input. Select the actual artifact's template deliberately. Include actual prompt lengths near supported batch and context boundaries. Attempt 128 to 256 generated tokens where context permits, with explicit trial timeouts. A timeout is a valid recorded result.

For baseline and candidate, use the same artifact and declared settings. Start with one warm-up and three measured trials per feasible condition. Alternate order when possible. Record fresh-process cache state honestly. Do not call an uncontrolled rootless run cold-cache.

Run repeated fresh requests and a bounded sustained-load campaign of at least 30 minutes when safe. Sample resource and thermal behavior throughout. Test controlled, reversible memory pressure on the target only with a bounded helper and conservative stop conditions. Do not try to crash Android. If pressure tests are unsafe or unavailable, preserve that release limitation.

Record real prompt and output lengths, EOS behavior, token IDs, correctness results, load time, prefill, TTFT, decode rate, RSS, PSS when available, covered allocation peaks, reads, wait time, thermal state, and outcomes. Keep units and timing boundaries consistent. Preserve per-run engine, backend, patch, binary, model, tokenizer, template, compiler, NDK, and configuration identities. Keep weights and private device logs untracked.

Generate aggregate results from raw records. Report count, median, minimum, maximum, failures, and timeout count. Explain the measured limiter. Do not infer physical flash traffic from application `pread` bytes.

## Deliver a reviewable release decision

Update stale architecture, validation, dependency, model-support, contribution, and handoff statements as implementation changes. Preserve historical evidence with dates instead of rewriting unsuccessful experiments as successes. Keep support scoped to exact tested artifacts and configurations.

Create `docs/RELEASE_READINESS.md` with each gate marked passed, failed, or blocked. Link commands, code, raw evidence, and remaining risks. Include the smallest next action for each blocker. Do not use a completion percentage as a substitute for gates.

Provide one reproducible build command and one user-facing command to run a validated model, with explicit settings, cancellation, timeout, and error behavior. Keep a functional small-model path if the 35B route remains slow or unverified. Report actual latency so the owner can decide whether it is useful.

Finish with the commits, changes, tests actually executed, physical-device results, accounting coverage, open blockers, and readiness decision. Run `.venv/bin/python tools/verify.py` and the new inference integration checks before handoff. Do not describe the engine as battle tested merely because it compiles, emits plausible text, survives one restart, or has a small arena.
