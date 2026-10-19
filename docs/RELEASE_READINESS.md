# Release readiness

Date: 2026-10-19. Scope: hardening audit of revision `7303320` on branch
`codex/hardening`. Status is per gate; a gate is never marked passed because a
different gate passed.

## Decision

**Not release-ready.** Unit 2 verifies startup artifact identity and layout, but full-hash startup cost and runtime mutation coverage remain blockers. Unit 1 closed the safe-failure blocker and added a conservative
prefill plan: gate 3 passes for the tested release scope. Gate 4's bounded-prefill
part is implemented, but its memory-admission and changed-microbatch parity portions
remain open. Gate 2 is passed for startup identity/layout and blocked for runtime mutation; gate 6
(practical campaign) remains failed, and gate 5 (reference parity) remains blocked. Do not describe the engine as battle-tested or as having a
hard process memory cap.

## Gate 1 — reproducible inference integration: PASSED (local), CI pending

- Added CMake targets behind `AME_INFERENCE_INTEGRATION=ON`: `ame_backend`,
  `moe_stream_gate`, `moe_kernel_gate`, `manifest_gate`, `artifact_gate`, and
  `llama_baseline_stream`.
- `tools/verify_inference.py` verifies the pinned llama.cpp HEAD and that the applied
  git diff hash equals `patches/0001` + `patches/0002`, then builds clean Debug,
  Release, and ASan+UBSan directories and runs the gates.
- The deliberately broken materializer target exits nonzero and reports
  `identical: false`.
- The Android harness was rebuilt with NDK r27c. `tools/check_elf.py` records ELF64
  AArch64, `/system/bin/linker64`, only liblog/libm/libdl/libc dependencies, and no
  dot-product, I8MM, BF16, or SVE instructions.
- Evidence: [integration JSON](../evidence/hardening/gate1-integration.json),
  [Android ELF JSON](../evidence/hardening/gate1-android-elf.json).
- Reproduction:

```sh
AME_LLAMA_SOURCE_DIR="$AME_TOOLS/src/llama.cpp" \
AME_LLAMA_HOST_LIB_DIR="$AME_TOOLS/build/llama-host" \
.venv/bin/python tools/verify_inference.py --configs debug,release,sanitizers

AME_TOOLS="$AME_TOOLS" sh bench/build_harness.sh
.venv/bin/python tools/check_elf.py --binary "$AME_TOOLS/build/harness/llama_baseline_android" \
  --readelf "$AME_TOOLS/android-ndk-r27c/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-readelf" \
  --objdump "$AME_TOOLS/android-ndk-r27c/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-objdump" \
  --output evidence/hardening/gate1-android-elf.json
```

Limitation: the GitHub workflow `.github/workflows/inference.yml` was added but was
not executed in this environment. Logs prove the local matrix, not hosted CI. The
sanitizer builds instrument project adapter/gate code; the pinned llama.cpp static
libraries were built without ASan/UBSan because the upstream library was distributed
as a Release build. Upstream kernel sanitizer coverage is therefore excluded.

## Gate 2 — artifact-bound storage: PASSED (startup identity/layout), runtime mutation BLOCKED

- Added `src/backend/artifact`: one read-only FD, full SHA-256 enrollment check,
  GGUF tensor-table parse, and fstat recheck. The backend model loader uses
  `/proc/self/fd/<fd>`; every expert size-class source is `FileSource::open_fd`.
  No verified path reopens the artifact by pathname.
- The manifest is compared to the actual GGUF table: name, type, complete dimensions,
  absolute offset, byte extent, stride, quantization row arithmetic, and complete
  `_exps` coverage. Missing, duplicate, swapped, stale, or non-routed entries fail
  before materialization.
- A full artifact SHA-256 must match an independently enrolled digest. The manifest's
  declared SHA must agree. A matching sidecar for a wrong file does not override the
  enrolled digest.
- Tests cover same-size bit flip, wrong file with matching sidecar, swapped in-bounds
  offsets, missing/duplicate rows, pathname replacement, rename, hard link,
  truncation, and in-place mutation before install. Replacement/rename/hard-link
  tests continue from the retained FD.
- The backend loader binding was exercised by replacing the model pathname with a
  17-byte file after identity verification; the 14.3B run still produced token
  `32671` from the retained FD.
- Every materialization rechecks the retained-inode metadata before acquiring slots.
  This catches truncation/in-place metadata changes before a later slot publication.

Still blocked or limited:

- Full SHA-256 costs **393.93–412.67 s** for the 20.4 GB phone artifact (Release), and
  **120.55 s** for the 35B host artifact; the phone startup is not practically fast.
- The guarantee is **startup identity + metadata recheck**, not per-extent runtime
  integrity. A byte write after a recheck and before the bounded read can still be
  consumed; no authenticated digest table or per-slot byte verification exists.
- The enrolled expected digest is the trust root. A caller who supplies a wrong
  expected digest can enroll a wrong file; release commands must take it from the
  pinned catalog, not from the manifest.
- Version 1 manifests are accepted when the artifact itself is verified; they lack a
  layout checksum but the authoritative GGUF table still validates every row.
- The 14.3B replacement test used a hardlink on the same filesystem. The 35B phone
  test used the v2 manifest.

Evidence: [35B phone verified run](../evidence/unit2/qwen36-unit2-verified.json),
[35B host verified run](../evidence/unit2/host-qwen36-v2.json),
[backend replacement test](../evidence/unit2/host-replacement-after-artifact.json),
[artifact tests](../tests/artifact_gate_test.py),
[integration matrix](../evidence/hardening/gate1-integration.json).

## Gate 3 — resource limits, cancellation, failures: PASSED (Unit 1 scope)

- The materializer returns explicit resident/staged/cancelled/failed outcomes.
  Thread 0 publishes the outcome before the existing worker barrier; every worker
  takes the same branch. A latched error causes the CPU graph loop to stop before a
  later node consumes incomplete output, and `llama_decode` observes the aborted
  graph (returns 2).
- Expected manifest, arena, store, capacity, metadata, and unindexed-expert failures
  no longer call `std::abort()`. The lease is released before the failed operator
  returns; arenas and stores are retained until teardown.
- The real patched graph was tested with 1, 2, and 4 workers for: permanent read
  failure, unexpected EOF, cancellation before materialization, and capacity
  overflow. Every case returned an aborted graph with `failed` or `cancelled`
  telemetry, never `identical: true` from resident fallback. A fresh process
  repeated each successful case after the fault.
- The store also has a deterministic EINTR-retry test; the retry is not reported as
  inference failure.
- A 14.3B streaming read failure produced status `failed`,
  `finish_reason: materialization_failure`, the specific tensor name, no output, and
  exit 1 without SIGABRT. A fresh request returned `32671`.
- Streaming cancellation before prefill and during decode produced `cancelled`
  records; the decode case retained two partial tokens but did not label the run
  successful.

Remaining limitations: no compute-failure injection through the backend, no
deterministic cancellation exactly between two read chunks, and no same-process
reset. Reopen is required after failure or cancellation; that is the release
contract for this unit.

Evidence: [worker/fault matrix](../evidence/hardening/gate1-integration.json),
[read failure](../evidence/unit1/unit1-read-failure.json),
[fresh request](../evidence/unit1/unit1-fresh-after-failure.json),
[decode cancellation](../evidence/unit1/unit1-cancel-decode-late.json),
[EINTR test](../tests/residency_test.cpp).

## Gate 4 — bounded prefill passed; memory admission and parity still open

- Added `src/runtime/prefill_plan` and tests. It derives expert count/top-k from
  model metadata, computes
  `min(expert_count, tokens_per_unit * expert_used)`, caps the unit by batch,
  slot capacity, the 128-expert lease limit, and checked arithmetic.
- The harness sets `llama_context_params.n_ubatch` before context construction,
  refuses slots below top-k or a zero-token plan, and checks the backend's reported
  effective `n_ubatch` after construction. Runtime union validation remains.
- For 256 experts/top-8/64 slots the plan is **8 tokens per compute unit**.
- A 14.3B streaming run executed exact 63-, 64-, and 65-token inputs with a 15-token
  unit and did not abort. The 35B regression executed with an 8-token unit.
- The fault gate exercised slot counts 7/8/9 with top-8 and unions C−1/C/C+1.
  Below-top-k configurations refused; at/above capacity staged correctly.
- Changed-microbatch numerical parity is not established. The 63/64/65 evidence
  uses one model and does not prove every architecture's recurrent-state parity.

Memory admission is still not implemented: there is no fresh pre-open observation of
project-owned bytes, backend workspace, KV/recurrent state, page cache, or load
peaks, and the arena is not a process cap.

Evidence: [prefill tests](../tests/prefill_plan_test.cpp),
[63 tokens](../evidence/unit1/prefill-63.json),
[64 tokens](../evidence/unit1/prefill-64.json),
[65 tokens](../evidence/unit1/prefill-65.json),
[35B unit-8 regression](../evidence/unit1/qwen36-unit1-regression.json).

## Gate 5 — numerical correctness and comparisons: BLOCKED

- `ame compare` no longer casts fractional JSON numbers to integers. It rejects
  fractional and out-of-range token IDs, duplicate keys, unsupported schemas, and
  missing required fields; it compares artifact bytes, settings, tokens, and finish
  reason. Unit tests cover each rejection.
- There is no small complete **quantized** MoE fixture (the existing runtime gate uses
  F32 experts).
- The 35B full-resident reference is blocked: no available phone or host has more than
  20.4 GB usable memory with the same pinned backend. Repeatability and bitwise kernel
  gates are not reference logits.

Evidence: [compare tests](../src/compare/compare.cpp),
[phase3 comparisons](../evidence/phase3/compare-same.json).

Smallest next action: produce a complete quantized MoE GGUF that fits this host, run
unmodified vs streaming with matched inputs, and declare tolerances before inspection.

## Gate 6 — practical phone workloads: FAILED

- Existing physical evidence: three Qwen3.5-2B trials, seven Qwen1.5-MoE trials, and
  five 35B trials at five prompt tokens / eight output tokens.
- No committed workload corpus, no 128-256-token generation campaign, no controlled
  memory-pressure test, no PSS sampling, and no 30-minute sustained run.
- The 35B result is 0.37-0.69 tok/s at 9-18 s TTFT; it is not interactive.

Smallest next action: commit a four-workload corpus and run one warm-up plus three
measured 2B trials per workload with 128 output tokens and recorded thermal state.

## Remaining release blockers

1. Runtime mutation/per-extent content verification and practical full-hash startup cost.
2. Quantized small-model parity and, if a host exists, 35B reference logits.
3. Changed-microbatch numerical parity for the conservative prefill plan.
4. A practical phone workload/sustained campaign.
5. Complete backend allocation accounting; the arena is still not a process cap.

## Reproduce the current state

```sh
.venv/bin/python tools/verify.py
AME_LLAMA_SOURCE_DIR="$AME_TOOLS/src/llama.cpp" \
AME_LLAMA_HOST_LIB_DIR="$AME_TOOLS/build/llama-host" \
.venv/bin/python tools/verify_inference.py --configs debug,release,sanitizers
```

The standard verifier passes nine tests in Debug, Release, and ASan/UBSan. The
inference matrix passes locally in all three configurations.