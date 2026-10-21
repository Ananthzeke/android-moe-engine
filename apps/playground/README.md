# Android MoE Playground

Local SvelteKit chat UI for the native Android MoE server. Inference stays on
the phone. The laptop proxies requests and stores conversations in IndexedDB.

## Status

Experimental. Phase 1 chat MVP and browser-observed metrics are implemented.
Backend request telemetry is not yet exposed by the C++ server; missing values
are shown as unavailable, never zero.

## Install

```sh
cd apps/playground
npm ci
npm run check
npm test
npm run build
```

## Run

Terminal 1 — forward the already-running phone server (do not start another
model and do not stop the existing server):

```sh
adb forward tcp:8080 tcp:8080
curl -s http://127.0.0.1:8080/health
```

Terminal 2:

```sh
cd apps/playground
AME_BACKEND_URL=http://127.0.0.1:8080 \
AME_ALLOWED_BACKENDS=http://127.0.0.1:8080 \
npm run dev -- --host 127.0.0.1 --port 5173
```

Open `http://127.0.0.1:5173`.

`AME_ALLOWED_BACKENDS` is a comma-separated server-side allowlist. The proxy
rejects any path other than health, models, and Chat Completions. Keep both
services loopback-bound; do not expose the phone API to a LAN.

## Production preview

```sh
npm run build
AME_BACKEND_URL=http://127.0.0.1:8080 \
AME_ALLOWED_BACKENDS=http://127.0.0.1:8080 \
HOST=127.0.0.1 node build/index.js
```

## Metrics and privacy

- TTFT ends at the first nonempty visible text delta, not the role-only event.
- Decode throughput is `(N-1)/(t_last-t_first)`, only when usage has
  `completion_tokens >= 2` and the interval is positive.
- SSE chunks are not tokens. Token-latency distribution is unavailable.
- Conversations and exports stay in the browser's IndexedDB/downloads. No
  analytics or external provider is used.
- Expert reads are cumulative process counters, not request attribution and not
  physical flash I/O.

## Tests

```sh
npm run check
npm test
```

The tests use deterministic mocked SSE responses; they do not claim a phone
run. Browser E2E against the real phone remains a limitation.