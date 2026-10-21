# 35B CPU release report

Date: 2026-10-19. Branch: `codex/hardening`. Backend:
`ff5a23376725e5a889fa0bfb987b7541dc304da0`.

## Status matrix

| Area | Status | Evidence |
| --- | --- | --- |
| Real 35B phone API | Passed before queue change | qwen36 API evidence |
| ARMv8 SHA2 startup | Passed, 2.32× hash speedup | sha256-acceleration.json |
| Bounded FIFO queue | Passed on host | queue-host.json |
| Queued 35B phone deployment | Passed at 192.168.0.4 | qwen36-queue-summary.json |
| Real 35B pristine reference | Blocked | no host reference run |
| Continuous two-sequence batching | Not implemented | no claim |
| Two-hour 35B campaign | Not run | no claim |
| Near-context 35B | Failed 300 s client deadline | qwen36-near-context.json |
| 30-minute dense campaign | Passed, one-token requests | phone-sustained-summary.txt |

## 35B baseline and final

| Metric | Value |
| --- | ---: |
| Artifact bytes | 20,419,565,568 |
| Digest | `671e47e0…5140c7` |
| Scalar full SHA-256 | 403.728 s |
| ARMv8 SHA2 full SHA-256 | 173.759 s / 268.139 s later run |
| Short request, 2 tokens | 46.603 s |
| 32-token request | 105 s |
| Effective 32-token throughput | ~0.30 completion tokens/s |
| Expert reads, 32-token request | 31,032,999,936 bytes |
| Arena | 37,748,736 bytes |
| RSS / HWM / PSS | 2,000,584 / 2,109,232 / 2,083,974 KiB |

These are not interactive speeds.

## Implemented

- Native OpenAI-compatible server: health, models, JSON chat, SSE, usage,
  multi-turn, stop strings, validation, errors.
- Verified artifact and retained FD; 64-slot expert streaming.
- ARMv8 SHA2 acceleration with scalar fallback and runtime dispatch.
- Four-outstanding-request FIFO admission.
- HTTP 429 with `Retry-After` when four requests are outstanding.
- 300 s queue waiting and 300 s execution deadlines.
- Request-owned cancellation/deadline state.
- Lease release after each request.
- 10 s HTTP write timeout and larger worker pool for health responsiveness.
- Fresh llama context per request; no poisoned context reuse.

## Queue evidence

Five simultaneous requests at the 2B host API:

- 4 accepted
- 1 rejected with HTTP 429 `server_busy`
- health remained responsive

```json
{"accepted":4,"rejected":1,"outstanding_limit":4}
```

## Queue on phone

Wireless debugging was restored at 192.168.0.4:5555. The queued binary was
deployed and two simultaneous 35B requests completed. During execution health
reported outstanding: 2, active: 1; both returned HTTP 200 with one completion
token each. Startup hash on this run was 268.139 s.


```text
adb connect 192.168.0.4:5555
# device
```

## Reproduction

Build:

```sh
AME_TOOLS=/media/ananth/17AB9D0651C6CC1F/ame-tools sh bench/build_server.sh
```

Host API and queue tests:

```sh
.venv/bin/python tools/verify_api.py --base-url http://127.0.0.1:8112 \
  --model Qwen3.5-2B --output evidence/api/host-api-queue-verification.json
```

35B launch after ADB is restored:

```sh
adb forward tcp:8080 tcp:8080
adb shell 'cd /data/local/tmp/ame-api && nohup ./ame-server \
  --model /data/local/tmp/ame-engine/models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --model-id Qwen3.6-35B-A3B --stream \
  --manifest /data/local/tmp/ame-api/qwen36-35b.manifest \
  --artifact-sha 671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7 \
  --slots 64 --arena-mib 256 --context 512 --threads 4 --port 8080 \
  >qwen36.log 2>&1 </dev/null &'
```

## Remaining blockers

1. Real 35B pristine reference comparison (tokens, router IDs, teacher-forced logits).
3. Genuine two-sequence continuous batching; current server remains one active generation.
4. Near-context prefill beyond 300 s.
5. Two-hour mixed-request 35B campaign and graphs.
6. Request-scoped stage timings and diagnostics.
7. Writable-file mutation limitation remains.
