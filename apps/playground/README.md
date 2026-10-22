# Android MoE Playground

A local chat interface for the native Android MoE server. Stream replies, adjust generation settings, inspect browser timings, and export conversations. The phone runs the model; the laptop hosts the SvelteKit UI and its same-origin proxy.

Experimental: backend request telemetry and phone-side cancellation qualification remain open. This UI adds no batching, scheduling, or model-loading capability.

## Start the playground

Requirements: Node.js 22, npm, an authorized ADB connection, and an already-running [native phone server](../../docs/RUN_PHONE.md). Verification used Node.js 22.23.2 and npm 10.9.8.

1. Confirm that ADB sees your phone. For wireless debugging, connect using the phone's displayed address and port; complete pairing if required. Do not publish those identifiers.

   ```sh
   adb devices
   adb forward tcp:8080 tcp:8080
   curl --fail http://127.0.0.1:8080/health
   curl --fail http://127.0.0.1:8080/v1/models
   ```

   Health must report `ready`; models must list the loaded model. The UI does not start the phone server, download weights, or switch models.

2. From the repository root, install and run the UI:

   ```sh
   cd apps/playground
   npm ci
   AME_BACKEND_URL=http://127.0.0.1:8080 \
   AME_ALLOWED_BACKENDS=http://127.0.0.1:8080 \
   npm run dev -- --host 127.0.0.1 --port 5173
   ```

3. Open [localhost:5173](http://127.0.0.1:5173). Wait for **Connected**, set **Max tokens** to a small value such as 8, then send a short message. Large streamed MoE models can take minutes before producing text.

The proxy permits only `GET /health`, `GET /v1/models`, and `POST /v1/chat/completions`. `AME_ALLOWED_BACKENDS` is a comma-separated server-side destination allowlist. Keep the UI and API bound to loopback.

If laptop port 8080 is occupied, use `adb forward tcp:18080 tcp:8080` and change **both** backend environment variables to `http://127.0.0.1:18080`. A forward maps the laptop port to the phone port; it does not start a server.

## Run the built UI

Use the built server for longer verification sessions so development reloads do not interrupt chat:

```sh
npm run check
npm test
npm run build
HOST=127.0.0.1 PORT=5173 ORIGIN=http://127.0.0.1:5173 \
AME_BACKEND_URL=http://127.0.0.1:8080 \
AME_ALLOWED_BACKENDS=http://127.0.0.1:8080 \
node build/index.js
```

These commands run from `apps/playground`. Stop an existing UI on port 5173 first. Rebuild only between chat requests.

## Read the metrics

| Field | Meaning and limitations |
| --- | --- |
| Browser TTFT | Request start to first nonempty visible text delta. Includes browser, proxy, network, queue, and model work. A role-only SSE event does not count. |
| Request total | Request start to stream closure, including prompt processing and transport. |
| Text delivery rate | `(completion_tokens - 1) / seconds_between_first_and_last_visible_text`, using final backend usage. Only available with at least two tokens and a positive interval. SSE buffering and hidden/non-visible tokens can skew this estimate; it is not native engine decode speed. |
| Generated | Backend-reported completion token count, not the number of SSE chunks. |
| Token latency | Unavailable: the native API does not expose per-token timestamps. |
| Expert cache snapshot | Process-cumulative counters from the last health fetch. Use **Reconnect** after a request to refresh. Application read bytes are not physical flash I/O or request-attributed reads. |

Missing values remain `unavailable`. Context size, threads, and runtime mode are not inferred from the model name or fixed labels. The optional chart plots browser TTFT and request duration on a seconds axis (exports retain milliseconds) and loads ECharts only when opened. Historical exports retain the `decode_tps` key for the browser delivery estimate.

## Conversation controls and privacy

- **Stop** aborts the browser request. Immediate phone-side cancellation and queue draining still require separate qualification.
- **Retry** resends the last user turn with the preceding conversation. **Clear** starts an empty on-screen conversation.
- **Copy**, **JSON**, and **CSV** export the current response or conversation.
- History is stored in this browser's IndexedDB. If storage fails, chat continues with a warning. The history list is a record of saved sessions; reopening a saved conversation is not implemented.
- Prompts and responses pass through the local proxy to the configured backend. No analytics or external inference provider is configured. Exports can contain your prompts; inspect them before sharing.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Disconnected / API unreachable (502) | Check `adb devices`, the port forward, and `/health` directly. When healthy, select **Reconnect**. |
| Phone appears offline or unauthorized | Unlock the phone and restore ADB authorization or wireless pairing. |
| Loading health or empty model list | Wait for the server to finish loading and verifying the artifact. The UI cannot accelerate startup. |
| Destination/path forbidden (403) | Match `AME_BACKEND_URL` exactly to an entry in `AME_ALLOWED_BACKENDS`; restart the UI after changing these variables. |
| Server busy (429) | The native server's admission limit is full. Wait before retrying; this UI adds no queue. |
| Long wait / execution timeout | Inspect the phone server and reduce the prompt/output length. A responsive health endpoint alone does not prove generation progress. |
| Stream ends without completion | The request is shown as failed with retained partial text; retry after the server is ready. |

## Verification

```sh
npm run check
npm test
npm run build
```

**Real-phone smoke test:** `Qwen3.6-35B-A3B` returned `Hi!` through the built UI (2 completion tokens). Browser TTFT was 238.45 s and total duration was 254.60 s, including possible queue time. The synthetic [JSON export](../../evidence/playground/phone-chat.json) and [screenshot](../../evidence/playground/phone-chat.jpg) retain this single observation; it is not a controlled decode benchmark. Offline failure feedback was also exercised in the browser.

Eleven deterministic client/proxy tests cover split SSE reads, role-only events, UTF-8 text, usage gaps, short outputs, malformed and interrupted streams, connection failures, path/method rejection, and overload propagation. They are not device performance tests.

The dependency audit on 2026-10-21 reported 17 advisories (3 low, 6 moderate, 6 high, 2 critical), including development tools. `npm audit fix` did not resolve them within the declared ranges. A dependency migration and another verification pass are required before deployment beyond this experimental loopback setup; no forced major-version upgrade was applied.

See [the current connectivity verification record](../../evidence/playground/VERIFICATION.md) and [the telemetry implementation plan](PLAN.md).
