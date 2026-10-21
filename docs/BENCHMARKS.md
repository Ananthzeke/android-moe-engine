# Recorded phone measurements

Public claims reviewed on 21 October 2026 against repository revision `22f6a25` and its retained evidence. This document audits existing observations; it is not a new device benchmark or a controlled speedup campaign.

## Artifact and execution

The target is Qwen3.6-35B-A3B Q4_K_M, 20,419,565,568 bytes, SHA-256 `671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7`. The [catalog and lock](../configs/dependencies.lock.json) identify the artifact and pinned llama.cpp backend `ff5a23376725e5a889fa0bfb987b7541dc304da0`.

The [phone snapshot](../configs/device-vivo-s1-pro.json) records vivo S1 Pro / vivo 1920, Snapdragon 665, Android 12, and 7,834,636 KiB MemTotal, marketed as an 8 GB configuration. It is a historical snapshot, not a live memory-admission observation.

The accelerated API release uses four threads, context 512, 64 expert slots, and a 256 MiB arena allowance. It executes one request at a time. Later queue changes add bounded waiting, not continuous batching or guaranteed FIFO ordering.

## Accelerated API observations

| Measurement | Recorded value | Evidence and interpretation |
| --- | ---: | --- |
| Short answer | 46.603491851 s | `requests.timed_short` in [API summary](../evidence/api/qwen36-api-summary.json), 19 prompt and 2 completion tokens, `Hello!`, stop. Request wall time. |
| Longer output | 105 s | `qwen36.generation_32_seconds` in [release record](RELEASE_RECORD.json). The [raw response](../evidence/api/qwen36-chat-32.json) confirms 22 prompt and 32 completion tokens and finish reason `length`. No per-stage timer record accompanies this duration. |
| Effective longer-output rate | About 0.30 completion tokens/s | Derived as 32 / 105. Includes prompt processing and request overhead. It is not isolated decode throughput or a measured TTFT. |
| RSS / VmHWM / PSS | 2,000,584 / 2,109,232 / 2,083,974 KiB | Captured text in [API summary](../evidence/api/qwen36-api-summary.json). RSS and PSS are snapshots; VmHWM is the process lifetime peak RSS. VmHWM is about 2.01 GiB. |
| Swap snapshot | 84,104 KiB VmSwap | Same captured text. A low RSS is not proof of zero swap or a complete memory budget. |
| Arena commitment | 37,748,736 bytes, 36 MiB | [Health after longer output](../evidence/api/qwen36-health-32.json). Expert staging only, not total model or process memory. |
| Expert application reads | 31,032,999,936 bytes | Same health snapshot. Cumulative process counter, not proven request-local bytes. It includes prior materialization; no before/after delta establishes a 32-token-only total. |

The [multi-turn record](../evidence/api/qwen36-multiturn.json) and [SSE transcript](../evidence/api/qwen36-sse.txt) retain functional examples. The summary also records a manual OpenAI client call. These are bounded functional checks, not a full compatibility suite.

## Earlier harness and separate API records

The [five-run 35B harness aggregate](../evidence/phase4/qwen36-stream-aggregate.json) records 0.368-0.688 decode tokens/s, rounded to 0.37-0.69 in the README. Each run generated eight tokens from a short prompt. Context 512 has three trials, while contexts 1024 and 2048 have one each. Slot settings differ across some runs. These are observations across configurations, not a controlled fixed-configuration performance estimate. Identical generated token IDs across these runs do not establish pristine-reference correctness.

The [separate direct API record](../evidence/api/phone-35b-direct/run.json) reports 246.767 s for 24 prompt and 2 completion tokens and 284.022 s for 26 prompt and 1 completion token. It reports process VmHWM 2,191,548 KiB, about 2.09 GiB, and the same 36 MiB arena. The record binds different source/build identities and uncontrolled cache state. Do not divide these request times by the harness throughput or present them as a controlled regression against the accelerated API pair.

That direct API monitor first observed readiness 1,732.275 s after monitoring began. This is not an isolated hash timer or a complete launch-to-ready measurement. Its process-lifetime kernel `read_bytes`, application `rchar`, and run-specific expert-read counters cover different work and cannot establish flash bandwidth or read amplification by subtraction.

## Startup hashing

[Hash evidence](../evidence/api/sha256-acceleration.json) records 403.728 s scalar versus 173.759 s ARMv8 SHA2, with the same artifact digest. Their arithmetic ratio is 2.32. This is one observed pair, not a repeated controlled startup speedup. A [later queued-phone run](../evidence/api/qwen36-queue-summary.json) records 268.139 s for hashing. Total startup includes other work.

The source has runtime HWCAP dispatch and a scalar fallback. Hashing is not skipped. Metadata rechecks after startup do not authenticate subsequent in-place byte mutations.

## Queue and correctness evidence

[The host queue test](../evidence/api/queue-host.json) records five concurrent requests with four accepted and one rejected. [The phone summary](../evidence/api/qwen36-queue-summary.json) records two simultaneous 35B requests, both HTTP 200 and one completion token each, with health showing two outstanding and one active. This does not establish four-request phone behavior, FIFO fairness, or concurrent decoding.

Tiny complete-model and quantized operator parity are recorded in [complete verification](../evidence/unit3/complete-verification.json) and [production verification](../evidence/unit3/production-verification.json). The real 35B artifact has no full pristine reference comparison of tokens, routers, and teacher-forced logits. API text and usage are not such a comparison.

## Failed and unqualified cases

- [Near-context 35B request](../evidence/api/qwen36-near-context.json) timed out after 300.102 s. A larger allocated context does not prove long-context generation.
- The [sustained summary](../evidence/api/phone-sustained-summary.txt) covers 216 dense 2B one-token requests over 1,805 seconds. It is not a sustained 35B or long-answer campaign.
- No two-hour mixed-request 35B campaign, reliable request-stage TTFT dataset, or complete memory-pressure qualification is recorded.
- Client-disconnect cancellation was not re-qualified on the 35B release. Queue waiting and streaming bounds lack exhaustive tests.
- Project allocations, mapped weights, backend workspace, OS cache, swap, RSS, and PSS have different scopes. No hard process memory cap is established.

## Reproduction and future measurements

Follow [the 35B source-build and launch guide](RUN_35B.md) and record the exact engine/backend/binary/artifact identities and command. Use the same prompts, template, output limits, thread count, slots, and context for comparisons. Capture request timers and counter deltas, thermal state, system memory, and repeated alternating baseline/candidate trials. Keep failures and unavailable counters. Publish correctness and memory outcomes beside speed results.

The [validation contract](VALIDATION.md) and [research experiment order](../research.md) define the next measurements. No new phone throughput was measured while preparing this public repository.
