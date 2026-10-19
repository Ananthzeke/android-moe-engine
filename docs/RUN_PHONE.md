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
