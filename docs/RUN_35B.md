# Run the 35B model on the phone

Use the existing native server with `Qwen3.6-35B-A3B-Q4_K_M.gguf` and expert
streaming. The launcher uses 64 expert slots, a 256 MiB arena allowance, a
512-token context, and four CPU threads. The arena allowance is not a process
memory cap.

## Separate direct API observation (2026-10-20)

Other accelerated API records report 46.60 s for a two-token answer and 105 s for 32 completion tokens. They are different runs, not a controlled before/after comparison. See [recorded measurements and their scopes](BENCHMARKS.md).

The enrolled 35B artifact completed both requests through this server on the
physical vivo 1920, with thinking disabled by default:

| Request | Answer | Finish | Request wall time |
| --- | --- | --- | ---: |
| JSON greeting | `Hello!` | `stop` | 246.77 s |
| SSE arithmetic | `4` | `stop`, then `[DONE]` | 284.02 s |

Both returned HTTP 200 and usage counts. Readiness was first observed 1,732.28 s
after monitoring started, including full artifact verification and model loading.
These request times include context creation, prompt processing, and generation;
they are not decode throughput measurements. This is a working bounded text API
smoke, and its latency is several minutes per short request on this phone.

Peak process RSS was 2,191,548 KiB (about 2.09 GiB). The project expert arena was
37,748,736 bytes (36 MiB). Expert materialization recorded 27,648 reads and
16,307,453,952 application-read bytes across the two requests, with no streaming
failure. RSS, PSS, arena bytes, and kernel I/O counters have separate scopes in
[the run record](../evidence/api/phone-35b-direct/run.json). Raw
[JSON](../evidence/api/phone-35b-direct/chat.json) and
[SSE](../evidence/api/phone-35b-direct/sse.sse) responses are retained.

## Build and deploy in an existing tool environment

These commands assume the server package already exists at
`/data/local/tmp/ame-api` and the model already exists at
`/data/local/tmp/ame-engine/models/Qwen3.6-35B-A3B-Q4_K_M.gguf`.

Run from the repository root:

```sh
AME_TOOLS=/path/to/existing/ame-tools sh bench/build_server.sh
adb shell /data/local/tmp/ame-api/stop.sh
adb push /path/to/existing/ame-tools/build/harness/ame-server-android \
  /data/local/tmp/ame-api/ame-server
adb push deploy/phone/start-35b.sh /data/local/tmp/ame-api/start-35b.sh
adb push evidence/phase4/qwen36-35b-a3b-q4km-v2.manifest \
  /data/local/tmp/ame-api/qwen36-35b-v2.manifest
adb shell chmod 755 /data/local/tmp/ame-api/ame-server /data/local/tmp/ame-api/start-35b.sh
```

Use the existing pinned tool directory, including its patched llama.cpp builds
and Android NDK. The server build checks the backend revision and runs the actual
35B chat-template fixture before compiling the host and Android servers.

The launcher pins the expected artifact SHA-256:

```text
671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7
```

The enrolled artifact has 20,419,565,568 bytes. Startup verifies the whole file
before loading the model. Keep the process running to retain the verified model
across requests. Restarting repeats verification. A historical direct API monitor observed readiness after roughly half an hour. Separate isolated hash records range from 173.759 to 403.728 s. These have different timing boundaries. Allow for loading rather than treating a loading response as an error.

## Start the server

First check available memory and storage and stop any other inference server
using the same port. The following stop script stops this package's servers:

```sh
adb shell /data/local/tmp/ame-api/stop.sh
adb shell 'cd /data/local/tmp/ame-api; nohup ./start-35b.sh >35b-start.log 2>&1 </dev/null & echo $! >35b.pid'
adb forward tcp:8081 tcp:8081
curl --max-time 10 http://127.0.0.1:8081/health
```

Wait for `status` to become `ready` and confirm the model is
`Qwen3.6-35B-A3B`. A loading response is not a generation result. Inspect the
startup log if the process exits:

```sh
adb shell tail -n 30 /data/local/tmp/ame-api/35b-start.log
```

## Generate an answer

The total context must fit the formatted prompt and generated tokens. Begin with
a short prompt and a bounded output. The server renders the model's stored Jinja
template with thinking disabled by default, using the pinned upstream chat
helpers. This prevents a short answer budget from being consumed by thinking
text. The optional top-level `enable_thinking` boolean can explicitly enable it;
arbitrary `chat_template_kwargs` are rejected.

```sh
curl --max-time 900 -sS http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"Say hello in one short sentence. Answer directly without reasoning."}],"max_tokens":48,"temperature":0}'
```

For streaming:

```sh
curl --max-time 900 -sSN http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"What is 2 + 2? Answer with the number only."}],"max_tokens":24,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}'
```

Use the OpenAI client with `base_url="http://127.0.0.1:8081/v1"`, a local
placeholder key, model `Qwen3.6-35B-A3B`, and a request timeout of at least 900
seconds. The desktop connection uses the ADB forward above. A client running on
the phone connects directly to the phone's localhost port.

The current server creates a fresh context per request. Include earlier messages
in each request when continuing a conversation. Only one generation runs at a
time.

## Stop the server

```sh
adb shell /data/local/tmp/ame-api/stop.sh
adb forward --remove tcp:8081
```

## Qualification limits

Earlier harness runs establish short 35B execution. They do not establish
complete reference-logit parity. New API startup and generation evidence is kept
separately from those historical harness results. A successful bounded API smoke
does not establish long-context correctness, sustained throughput, or a hard
process memory limit. The current server includes stop/UTF-8 buffering and request deadlines, but client-disconnect cancellation was not re-qualified on the 35B release. One generation is active at a time, with at most four requests outstanding. FIFO waiting is not guaranteed. See [current qualification limits](BENCHMARKS.md).
