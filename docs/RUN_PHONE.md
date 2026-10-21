# Run the phone inference server

The package is deployed at `/data/local/tmp/ame-api`. It is a native ARM64
executable; no root, Termux, or Python runtime is required.

## Files

- `ame-server` — native HTTP server.
- `start-dense.sh` — default Qwen3.5-2B server on port 8080.
- `start-moe.sh` — real Qwen1.5-MoE streamed server on port 8080.
- `stop.sh` — stop the server.
- `qwen15.manifest` — verified streaming manifest.
- Models remain in `/data/local/tmp/ame-engine/models`.

## Start the default server

```sh
adb shell 'cd /data/local/tmp/ame-api && nohup ./start-dense.sh >dense.log 2>&1 </dev/null &'
adb forward tcp:8080 tcp:8080
curl http://127.0.0.1:8080/health
```

Expected: `{"model":"Qwen3.5-2B","status":"ready"}`.

## Chat

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen3.5-2B","messages":[{"role":"user","content":"Say hello."}],"max_tokens":16,"temperature":0}'
```

Streaming:

```sh
curl -sN http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen3.5-2B","messages":[{"role":"user","content":"Say hello."}],"max_tokens":8,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}'
```

Python OpenAI client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
print(client.chat.completions.create(
    model="Qwen3.5-2B",
    messages=[{"role": "user", "content": "Say hello."}],
    max_tokens=16, temperature=0).choices[0].message.content)
```

## Start the real streamed MoE

Stop the dense server first, then:

```sh
adb shell 'cd /data/local/tmp/ame-api && \
  AME_ARTIFACT_SHA=09555491f85f40208e3999f3ced97b064f7c7c0b88bb3fedcbd5df51aad7d1ed \
  nohup ./start-moe.sh >moe.log 2>&1 </dev/null &'
adb forward tcp:8080 tcp:8080
```

The MoE startup performs the full 9.49 GB SHA-256 check; on the tested phone this
took roughly twenty minutes. During that interval `/health` is served by a loading
endpoint after it binds. `/health` reports streaming reads, misses, evictions,
materialize calls, contract counts, and arena bytes once ready.

## Stop

```sh
adb shell /data/local/tmp/ame-api/stop.sh
adb forward --remove tcp:8080
```

## API scope

OpenAI-compatible text Chat Completions:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions` (JSON and SSE)

Supported: text system/user/assistant messages, `model`, `stream`, `max_tokens`,
`temperature`, `top_p`, `stop`, `seed`, and `stream_options.include_usage`.

Rejected: tools, images/audio, `response_format`, logprobs, and `n != 1`.
One generation is active at a time; concurrent requests receive 429 `server_busy`.
Each request uses a fresh llama context; the model and verified artifact FD remain
resident.

## Known limits

- The server is bound to 127.0.0.1; use `adb forward`.
- No authentication.
- SSE UTF-8 pieces are emitted on complete UTF-8 boundaries.
- Client disconnect is checked between emitted tokens; a disconnected generation may
  remain busy until the next token boundary or completion.
- Real 35B API serving was not qualified in this release.
- Runtime writable-file mutation integrity remains the Unit 2 limit.

## 35B server (experimental)

```sh
adb shell 'kill $(pgrep -f /data/local/tmp/ame-api/ame-server) 2>/dev/null || true'
adb shell 'cd /data/local/tmp/ame-api && \
  nohup ./ame-server --model /data/local/tmp/ame-engine/models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --model-id Qwen3.6-35B-A3B --stream \
  --manifest /data/local/tmp/ame-api/qwen36-35b.manifest \
  --artifact-sha 671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7 \
  --slots 64 --arena-mib 256 --context 512 --threads 4 --port 8080 >qwen36.log 2>&1 </dev/null &'
adb forward tcp:8080 tcp:8080
```

Measured on this phone:

- full digest: `671e47e0…5140c7`, 20,419,565,568 bytes
- scalar hash: 403.728 s; ARMv8 SHA2 hash: 173.759 s (2.32×)
- short answer: `Hello!`, 2 completion tokens, 46.6 s total
- 32-token deterministic answer: 105 s, 22 prompt / 32 completion tokens
- multi-turn "What is my name?" answered "Your name is Sam."
- SSE completed with `[DONE]`
- RSS 2.00 GB, HWM 2.11 GB, swap 84 MB; arena 37,748,736 bytes
- near-context 450-token prefill exceeded a 300 s client deadline; retain as a failed case

This is experimental: 35B decode is sub-token-per-second, and real-35B pristine
reference parity was not established.

## Queue behavior

The server admits four outstanding requests, with one active generation. Excess
requests receive HTTP 429 and `Retry-After: 1`. Queued requests wait up to 300 s;
active execution has a separate 300 s deadline. Health exposes `queue` counters.

The queued binary was host-tested but not phone-tested after wireless ADB disconnected.
