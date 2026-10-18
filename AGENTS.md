# Implementation instructions

Read `README.md`, `docs/ARCHITECTURE.md`, `docs/IMPLEMENTATION_PLAN.md`, and `docs/VALIDATION.md` before implementation. `prompts/DEEPSEEK_BUILD.md` is the complete handoff. `configs/model-candidates.json` owns candidate identities and support status.

## Preserve the research question

Build a real CPU inference path and prove whether bounded expert streaming is feasible. A launcher that enables mmap is a baseline. Do not label it a completed expert-streaming engine.

- Use one pinned production backend after the phase 1 gate. Keep upstream modifications small and separately identifiable.
- Keep graph execution, tokenizer semantics, routing, attention, recurrent state, and numerical kernels in the backend unless evidence requires a change.
- Implement memory reservations, expert storage, leases, residency, cancellation, and telemetry in this project.
- Never skip experts, change top-k routing, zero missing weights, truncate layers, or silently alter a model to satisfy a resource limit.
- Make CPU text inference work before adding GPU, UI, multi-session execution, or speculative prefetch.

## Evidence and resource contracts

- Use the actual artifact and model architecture. Active parameter counts are not resident-memory estimates.
- Validate tensor extents, strides, quantization blocks, packing, hashes, and file bounds before execution.
- Count alignment, staging, cache metadata, backend packing, and retained module weights. A hard cap applies only to allocations covered by accounting.
- Report project-owned bytes, RSS, PSS, system memory, application-read bytes, and physical I/O as different quantities.
- Keep token IDs and correctness evidence. A plausible response is not a parity check.
- Record failed runs and unsupported counters with a reason. Use null for unavailable measurements.
- Never invent backend APIs, benchmark values, device results, or model support.
- Refresh memory and storage observations before admission. Preflight model transfer and conversion space before large downloads.

## Work and handoff

Implement phases in order. Complete a phase's acceptance checks before expanding scope. Keep a functional small-model baseline if the MoE gate fails. Update the architecture when a proven integration changes a contract, and record the evidence for that change in `docs/DESIGN_DECISION.md`.

Commit reviewable units with the commands and evidence needed to reproduce them. Keep large weights and private logs untracked. Do not change the phone's root, bootloader, ROM, or thermal protection as part of this project.
