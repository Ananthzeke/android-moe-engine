# Troubleshooting and limitations

## Build failures

**Backend mismatch:** `build_server.sh` rejects a llama.cpp HEAD that differs
from `ff5a23376725e5a889fa0bfb987b7541dc304da0`. Check the tree is clean and
both patches are applied.

**Undefined `sha256_arm_*` symbols:** `sha256.cpp` dispatches ARM SHA2 on
AArch64. Compile `src/common/sha256_arm.cpp` with
`-march=armv8-a+crypto` and link `sha256_arm.o` into every Android binary that
includes `sha256.cpp`. This repository's `bench/build_harness.sh` and
`bench/build_server.sh` now do so.

**Host/Android library mismatch:** after changing a patch, rebuild both
`$AME_TOOLS/build/llama-host` and `$AME_TOOLS/build/llama-android`. Do not
mix objects.

**NDK not found:** set `AME_TOOLS` to the documented layout and use NDK r27c
(`27.2.12479018`), API 31. The Android build is not a host-toolchain fallback.

## Hash mismatch

A hash mismatch means the artifact is not the enrolled artifact. Re-download the
exact revision and filename from `evidence/small-moe/20261021/artifacts.json`
or the 35B release record. Do not edit the manifest to make a mismatch pass.
The server verifies the full file at startup. Startup hashing took 173–340 s for
20.4 GB on the test phone; a slow start is expected and is not a hang.

## Insufficient storage

The 35B file alone is 20.4 GB. Downloads may temporarily require additional
space. Check:

```sh
adb shell 'df -h /data/local/tmp'
```

Do not delete the verified artifact while the server retains its descriptor.
The current project verifies at startup and rechecks metadata; it does **not**
claim runtime content authentication against writes through another descriptor.

## Unsupported layouts

The manifest/index builder rejects unsupported quantization types, malformed
tensor strides, out-of-file extents, invalid expert counts, and top-k larger than
the native selected experts. Do not hand-edit these fields. A model that is not
covered by the architecture matrix is unsupported, even if llama.cpp can read
it.

## Queue overflow and deadlines

- More than four outstanding requests: HTTP **429** with `Retry-After`.
- Queued longer than 300 s: HTTP **504**.
- Active longer than 300 s: generation failure; the deadline is checked between
  backend calls and cannot interrupt a blocked kernel read.
- Streaming client blocked on writes: 10 s HTTP write timeout.

The queue is bounded, but strict FIFO is not guaranteed by the current
condition-variable implementation. One generation is active.

## Cancellation and recovery

A stream disconnect marks the request cancelled on the next write attempt.
Cancellation during 35B prefill was not prompt in testing; the request remained
active after 30 s. A fresh request succeeded after the prior request completed.
Do not describe this as immediate prefill cancellation.

## Memory pressure

`--arena-mib` caps the project's expert arena, not total process memory. It does
not cap:
- backend weights and packing;
- KV/recurrent state;
- context and batch buffers;
- worker stacks, metadata, page tables;
- mapped file-backed pages.

35B observed RSS was about 2.0 GiB, HWM 2.11 GiB, with a 37.7 MB arena. Resident
models can use much more. On an 8 GB phone, other applications and the OS still
compete. No OOM killer, thermal, or governor changes are part of this project.

## What remains unqualified

- Real-35B pristine reference parity (tokens, router IDs, teacher-forced logits).
- Continuous/two-sequence batching; there is one active generation.
- Two-hour 35B mixed-request campaign.
- Near-context 35B prompts beyond 300 s.
- Prompt client-disconnect cancellation.
- Request-stage telemetry/TTFT distributions.
- Runtime writable-file mutation integrity.

## Not the JCOW404 router

This documentation covers **Ananthzeke/android-moe-engine** only. JCOW404 is a
separate router project. Do not mix ports, model paths, APIs, or deployment
instructions.
