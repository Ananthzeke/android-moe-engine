# API examples

The server is an **OpenAI-compatible text subset**, not a complete OpenAI
implementation. It binds loopback by default.

```sh
adb forward tcp:8080 tcp:8080
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/models
```

## JSON Chat Completions

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "granite-1b",
    "messages": [
      {"role": "system", "content": "Be concise."},
      {"role": "user", "content": "Why is the sky blue?"}
    ],
    "max_tokens": 32,
    "temperature": 0,
    "top_p": 1,
    "seed": 1234
  }'
```

Response shape:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "granite-1b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}
}
```

## SSE

```sh
curl -sN http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "granite-1b",
    "messages": [{"role": "user", "content": "Say hi."}],
    "max_tokens": 8,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

The stream ends with `data: [DONE]`. After streaming starts, an error is sent as
a documented error event and the stream closes; a successful finish event is not
emitted for a failed stream. UTF-8 is emitted only on complete code-point
boundaries, and stop strings may span emitted pieces.

## Python client

```sh
.venv/bin/python -m pip install 'openai==1.109.1'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
response = client.chat.completions.create(
    model="granite-1b",
    messages=[{"role": "user", "content": "Say hello."}],
    max_tokens=16,
    temperature=0,
)
print(response.choices[0].message.content)
print(response.usage)
```

## Supported text parameters

- `model`
- `messages` with `system`, `user`, and `assistant`
- `max_tokens`
- `temperature`, `top_p`, `seed`
- `stop` strings
- `stream`
- `stream_options.include_usage`

Unsupported capabilities are rejected rather than silently ignored: tools/function
calls, multimodal content, audio, `n > 1`, logprobs, and arbitrary OpenAI
parameters. This is documented text-only behavior.

## Errors, overflow, and timeouts

| Status | Meaning |
| ---: | --- |
| 400 | malformed JSON, invalid messages, unsupported capability |
| 404 | unknown model/path |
| 429 | four outstanding requests; `Retry-After: 1` |
| 500 | generation/backend failure |
| 504 | queued request waited 300 seconds |

A prompt plus completion that exceeds the context limit is rejected or fails
without truncating the model architecture. Do not assume a completion is
guaranteed to fit after tokenization; the server checks the rendered prompt and
reserved completion space.

The active request has a 300-second execution deadline including context setup,
prefill, and decode. This deadline is checked between backend calls; it does not
interrupt a blocked kernel read. Client disconnect cancels at the next emitted
piece; it was not prompt in the tested 35B prefill case. A fresh request recovered
after the earlier request completed.
