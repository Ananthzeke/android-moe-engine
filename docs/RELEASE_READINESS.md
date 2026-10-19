# Release readiness

Date: 2026-10-19. Scope: hardening audit of revision `7303320` on branch
`codex/hardening`. Status is per gate; a gate is never marked passed because a
different gate passed.

## Decision

**Not release-ready.** The validated small-model path and the 35B streaming
experiment remain usable for research, but gates 2, 3, 4, and 6 are failed and gate 5
is blocked. Do not describe the engine as battle-tested or as having a hard process
memory cap.

## Gate 1 — reproducible inference integration: PASSED (local), CI pending

- Added CMake targets behind `AME_INFERENCE_INTEGRATION=ON`: `ame_backend`,
  `moe_stream_gate`, `moe_kernel_gate`, `manifest_gate`, and
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

## Gate 2 — expert storage bound to the artifact: FAILED

Implemented and tested:

- Version 2 manifests carry artifact bytes, declared SHA-256, tensor count, and a
  layout hash; the materializer rejects version 2 manifests whose file size, row
  count, or layout hash disagree.
- Duplicate tensor names, byte-count overflow, size mismatch, unaligned extents,
  truncated extents, unsupported versions, and offsets above 4 GiB are rejected before
  any tensor reaches the kernel.
- An unindexed tensor whose name contains `_exps` or `expert` now aborts instead of
  silently falling back to resident weights.

Still failed:

- The `artifact_sha256` header is **declared**, not recomputed. A same-size wrong
  artifact and a wrong but in-bounds offset are not detected. There is no SHA-256
  implementation in the engine.
- The manifest does not carry `ne0`, `ne1`, or the quantization row size, so block
  arithmetic and packed strides are not independently re-derived.
- The unindexed-expert abort is not covered by an integration test.
- Existing phone manifests are version 1 and therefore have no layout identity. The
  hardened Android binary reproduced the 35B prefix `11751, 11` with the existing
  version 1 manifest and reported `manifest_version: 1`,
  `identity_checked: false`; this is direct evidence of the gap, not a pass.

Evidence: [35B version 1 regression](../evidence/hardening/qwen36-hardened-regression.json),
[manifest test output](../evidence/hardening/gate1-integration.json),
[tests/manifest_gate_test.py](../tests/manifest_gate_test.py),
[src/backend/moe_stream.cpp](../src/backend/moe_stream.cpp).

Smallest next action: add a streaming SHA-256 implementation and require a verified
full-artifact digest before installation, then add a same-size-wrong-artifact test.

## Gate 3 — resource limits, cancellation, failures: FAILED

Working parts:

- `install_moe_stream` returns a structured failure for invalid manifests.
- The harness cancellation flag is now wired into `request_moe_stream_cancel`; the
  store observes it between expert slots and read chunks.
- `ame run` classifies exited/signaled/timeout; a SIGABRT test records `signaled` and
  exit 1.
- Phone cancellation and injected 100 MB read failure each left the model SHA
  unchanged and a fresh request reproduced the baseline prefix.

Still failed:

- Expected materializer, arena, store, and unindexed-tensor failures still call
  `std::abort()`; the backend cannot return a structured inference error through the
  CPU kernel. The abort path writes no result record.
- No test injects cancellation before reads, during materialization, during prefill,
  and during decode separately.
- Same-process reset is not offered; reopen is required and documented.
- After wiring request cancellation into the store, a host 2B decode cancellation
  still produced a `cancelled` record with four partial tokens and no error; the
  backend error text remains in stderr.

Evidence: [qwen36-cancel-01.json](../evidence/phase4/qwen36-cancel-01.json),
[qwen36-iofail-01.err](../evidence/phase4/qwen36-iofail-01.err),
[tests/run_supervisor_test.cpp](../tests/run_supervisor_test.cpp).

Smallest next action: change the materializer hook to carry an error state and make
`MUL_MAT_ID` return a backend error, or move generation behind a fully supervised
process boundary and record the abort as a structured refusal.

## Gate 4 — prefill plans and memory admission: FAILED

- Routing, top-k, layers, and tokens are unchanged; the engine does not drop experts.
- There is no supported prefill plan. A prefill routed-expert union above
  `--moe-slots` still reaches `fail("expert union ... exceeds ... slots")` and aborts.
- No 63/64/65-token boundary test, no slot capacity minus-one/at/plus-one test, and no
  fresh memory/storage admission observation exists.
- The arena is not a process cap; backend workspace, KV/recurrent state, page cache,
  and load peaks remain uncovered.

Smallest next action: add a conservative prefill refusal
(`min(n_expert, tokens * top_k) > slots`) before decode, then a bounded microbatch
plan with parity evidence.

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

1. Full artifact identity and same-size wrong-artifact refusal.
2. Structured backend failure instead of `std::abort()`.
3. Safe prefill refusal or microbatch plan with boundary tests.
4. Quantized small-model parity and, if a host exists, 35B reference logits.
5. A practical phone workload/sustained campaign.

## Reproduce the current state

```sh
.venv/bin/python tools/verify.py
AME_LLAMA_SOURCE_DIR="$AME_TOOLS/src/llama.cpp" \
AME_LLAMA_HOST_LIB_DIR="$AME_TOOLS/build/llama-host" \
.venv/bin/python tools/verify_inference.py --configs debug,release,sanitizers
```

The standard verifier passes nine tests in Debug, Release, and ASan/UBSan. The
inference matrix passes locally in all three configurations.