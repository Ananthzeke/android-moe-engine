# Playground implementation plan, API gap, and telemetry schema

Status: Phase 1 implemented; Phase 2 browser metrics implemented, backend
request telemetry is an explicit gap; Phases 3–5 partial.

## Architecture

```text
Browser (SvelteKit + Tailwind, IndexedDB)
  │ same-origin /api/backend
  ▼
Laptop SvelteKit Node server (allowlisted proxy)
  │ http://127.0.0.1:8080 after adb forward
  ▼
Android phone native C++ server
```

The proxy only permits `health`, `v1/models`, and `v1/chat/completions`.
Destinations come from `AME_ALLOWED_BACKENDS` and `AME_BACKEND_URL`; the
default is `http://127.0.0.1:8080`. The browser never receives an arbitrary
backend URL. No model weights or inference cross the laptop.

## API gap assessment

Current native API:

| Endpoint | Status | Gap |
| --- | --- | --- |
| `GET /health` | present | status/model/queue/last streaming snapshot; no process RSS, no request view |
| `GET /v1/models` | present | one loaded model; no resident/streamed metadata |
| `POST /v1/chat/completions` | present | non-stream JSON and SSE; usage present when requested/streamed |
| Request telemetry | **gap** | no request-scoped monotonic timestamps, queue wait, token timestamps, or memory sample |
| Model configuration | **gap** | context/threads/mode/slots/arena are not returned by health |

The UI therefore labels browser TTFT and total duration as browser-observed.
Decode throughput is calculated only when final usage supplies
`completion_tokens` and at least two tokens. SSE chunk count is never treated as
a token count. Token-latency distribution and backend memory are unavailable
until the native schema below exists.

## Proposed request telemetry schema

```json
{
  "schema_version": 1,
  "request_id": "chatcmpl-...",
  "model": "granite-1b",
  "admission": {"queue_wait_ms": 0, "outcome": "admitted"},
  "execution": {
    "started_ms": 0,
    "prefill_start_ms": 0,
    "prefill_end_ms": 0,
    "first_token_ms": 0,
    "last_token_ms": 0,
    "completed_ms": 0
  },
  "counts": {"prompt_tokens": 20, "generated_tokens": 8},
  "tokens": {"ids": [1, 2], "latencies_ms": [null, 3.1]},
  "runtime": {"mode": "streamed", "context": 512, "threads": 4, "slots": 64, "arena_bytes": 268435456},
  "memory": {"rss_bytes": null, "peak_rss_bytes": null, "scope": "unavailable"},
  "cache": {"hits": null, "misses": null, "evictions": null, "read_bytes": null, "scope": "unavailable"},
  "outcome": "completed",
  "reason": null
}
```

All durations use the phone's monotonic clock. Token timestamps are bounded to
the most recent 4096 generated tokens; records expire after 128 requests or 30
minutes. A separate endpoint (or an optional response extension) will expose
this schema without changing OpenAI fields. No browser timestamp is subtracted
from a phone timestamp.

## Implemented

- connection health/models; SSE split/multiple event parser; Markdown/code;
  system prompt, temperature, max tokens, seed; stop/retry/copy/clear/export;
  IndexedDB history; error mapping; TTFT/total/decode/usage strip; optional
  ECharts chart; expert-streaming health counters.
- proxy allowlist and loopback default.
- unit tests for split SSE, multiple events, UTF-8, role-only events, missing
  usage, fewer than two tokens, malformed JSON, and HTTP 429.

## Not implemented / no claim

- Native request telemetry (above) and memory samples.
- Token-latency distribution; backend token timestamps.
- Controlled resident/streamed comparison runner (requires safe model restart).
- Two-hour campaign, browser E2E against a real phone, monitoring-overhead
  measurement.
- Strict FIFO, continuous batching, or automatic model loading.
