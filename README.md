# Android MoE Engine

[![Host foundation](https://github.com/Ananthzeke/android-moe-engine/actions/workflows/verify.yml/badge.svg?branch=main)](https://github.com/Ananthzeke/android-moe-engine/actions/workflows/verify.yml)
[![MIT license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Run quantized language models on Android CPUs with storage-backed expert streaming and a native OpenAI-compatible text API. This C++20 research engine targets developers exploring large mixture-of-experts models on memory-limited phones.

**A 20.4 GB Qwen3.6-35B-A3B Q4_K_M artifact has generated real API responses on a vivo S1 Pro / vivo 1920 with Snapdragon 665 and an 8 GB RAM configuration.** The engine reads the selected experts from storage into a bounded project arena while the pinned llama.cpp backend executes the model.

The release is **experimental**. Generation is slow, real-35B pristine-reference parity remains unverified, and the arena is not a hard process memory cap. Root is not required. No model weights or prebuilt inference distribution are included.

## Recorded demonstration

This is an excerpt from a [retained 35B phone response](evidence/api/qwen36-chat-short.json), not a live demo or a newly measured run:

```json
{
  "model": "Qwen3.6-35B-A3B",
  "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Hello!"}}],
  "usage": {"prompt_tokens": 19, "completion_tokens": 2, "total_tokens": 21}
}
```

The [timed request record](evidence/api/qwen36-api-summary.json) reports 46.60 seconds for this short-answer case. This demonstrates execution, not interactive latency or numerical reference correctness.

## Recorded measurements

The following are historical observations on the physical Snapdragon 665 phone, not forecasts for other devices or one controlled benchmark suite.

| Measurement | Recorded result | Scope and evidence |
| --- | ---: | --- |
| 35B artifact | 20,419,565,568 bytes | Q4_K_M GGUF, [identity and configuration](configs/dependencies.lock.json) |
| Short API answer | 46.60 s, 2 completion tokens | 19 prompt tokens, total request time; [record](evidence/api/qwen36-api-summary.json) |
| Longer API output | 105 s, 32 completion tokens | Reported total request time, about 0.30 completion tokens/s including prompt processing; [release record](docs/RELEASE_RECORD.json) |
| Earlier 35B harness decode | 0.37-0.69 tokens/s | Five short runs across allocated contexts 512/1024/2048, eight generated tokens each; [aggregate](evidence/phase4/qwen36-stream-aggregate.json) |
| Process peak RSS, accelerated API record | 2,109,232 KiB, about 2.01 GiB | Process VmHWM; [memory snapshot](evidence/api/qwen36-api-summary.json) |
| Expert arena | 37,748,736 bytes, 36 MiB | Project staging allocation only, with a 256 MiB allowance; [health snapshot](evidence/api/qwen36-health-32.json) |
| Full-artifact hashing | 403.728 s scalar, 173.759 s ARMv8 SHA2 | Matching digest, one recorded pair; a later accelerated run took 268.139 s; [hash evidence](evidence/api/sha256-acceleration.json), [later run](evidence/api/qwen36-queue-summary.json) |

Other API runs took 246.77 s for two completion tokens and 284.02 s for one. They have different prompts, builds, and timing conditions. See [benchmark definitions, variation, raw evidence, and failed cases](docs/BENCHMARKS.md). Short prompts in a larger allocated context do not establish long-context performance. The near-context API request exceeded a 300 s client deadline.

## What is implemented

- Native ARM64 HTTP server with health, model listing, JSON Chat Completions, SSE, usage, and multi-turn message formatting.
- One active generation, at most four outstanding requests, HTTP 429 overflow handling, and separate queue-wait and execution deadlines. This is not continuous batching or a guaranteed FIFO scheduler.
- Verified-artifact loading through a retained file descriptor, GGUF layout validation, and source-derived expert-consumer contracts.
- Bounded expert arenas, demand reads, move-only leases, eviction, cancellation, and conservative prefill admission. Every selected expert remains present.
- Runtime-dispatched ARMv8 SHA2 hashing with scalar fallback.
- Tiny complete-model and quantized operator comparisons, structured failure tests, and separate host foundation and inference verification.

The engine keeps graph execution, routing, attention, recurrent state, tokenization, and numerical kernels in the [pinned llama.cpp backend](configs/dependencies.lock.json). This project owns expert storage, leases, residency, admission, lifecycle handling, and measurement. [Architecture coverage](docs/ARCHITECTURE_COVERAGE.md) records the exact fixtures tested; arbitrary MoE architectures are not claimed.

## Build the host foundation

You need Git, Python 3.10+ with venv support, and GCC or Clang with C++20 support on Linux. The pinned Python development requirements provide CMake, Ninja, and clang-format.

```sh
git clone https://github.com/Ananthzeke/android-moe-engine.git
cd android-moe-engine
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python tools/verify.py
./build/host-release/ame --help
```

This builds and tests the foundation CLI. It does not build an inference server or download weights. See [CONTRIBUTING.md](CONTRIBUTING.md) for the Debug, Release, and ASan/UBSan checks.

## Run the 35B phone API

Inference requires additional setup: the locked and patched llama.cpp host/Android builds, Android NDK r27c, ADB, enough phone storage for the 20.4 GB artifact, and the independently enrolled model digest. There is no one-command clean inference installer yet.

Follow [the existing-environment build and deployment guide](docs/RUN_35B.md). It builds the server with `bench/build_server.sh`, deploys the manifest and launcher, and checks startup. Keep the verified server running between requests; a restart repeats full-artifact hashing.

Once the documented package is deployed and the 35B server is ready on phone port 8081:

```sh
adb forward tcp:8081 tcp:8081
curl --max-time 10 http://127.0.0.1:8081/health
curl --max-time 900 -sS http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"Say hello in one short sentence."}],"max_tokens":8,"temperature":0}'
```

The health response must name the model and report `ready`. A successful chat response has `choices` and `usage`, as in the recorded demonstration. The client timeout does not override the server's 300 s execution deadline.

For the OpenAI Python client, install the `openai` package in your client environment:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8081/v1",
    api_key="local",
    timeout=900.0,
    max_retries=0,
)
response = client.chat.completions.create(
    model="Qwen3.6-35B-A3B",
    messages=[{"role": "user", "content": "Say hello in one short sentence."}],
    max_tokens=8,
    temperature=0,
)
print(response.choices[0].message.content)
```

Compatibility covers the documented text Chat Completions subset, not the entire OpenAI API. Tools, images, audio, logprobs, and structured-output formats are unsupported. The server binds to localhost and has no authentication. See [API options and limits](docs/RUN_PHONE.md).

## Limits and next work

- Real-35B full-resident reference logits, router IDs, and token parity remain unverified. Tiny-fixture parity does not establish real-model correctness.
- No hard process memory cap, complete memory-pressure qualification, or two-hour long-form 35B campaign is established.
- Startup hashing is verified, but later writes to the backing file are not authenticated per extent.
- Stage-level request timing and controlled repeated performance measurements are incomplete. Client-disconnect cancellation has not been re-qualified on the 35B artifact.
- GPU/NPU execution and continuous multi-sequence batching are not implemented. The current path is CPU text inference with demand-loaded experts.

The immediate work is trustworthy timing, kernel/materializer efficiency, measured cache and read scheduling, complete reference comparisons, and sustained phone testing. See [release status](docs/RELEASE_REPORT_35B.md), [research and experiment order](research.md), and [hardening gates](docs/RELEASE_READINESS.md).

## Sponsorship and contributions

**I am looking for sponsors to continue developing this engine.** Funding, Android test devices, and access to a reference machine would help improve startup and generation speed, verify the real 35B model, and run reproducible device campaigns.

[Discuss sponsorship](https://github.com/Ananthzeke/android-moe-engine/issues/new?template=sponsorship.md) or read [the sponsorship priorities](SPONSORSHIP.md). No payment page is configured yet. Contributions, reproducible bug reports, and independent measurements are also welcome through [the contribution guide](CONTRIBUTING.md).

## Documentation and license

- [Published documentation site](https://ananthzeke.github.io/android-moe-engine/) and [site source](site-docs/index.md).

- [Architecture](docs/ARCHITECTURE.md), [implementation plan](docs/IMPLEMENTATION_PLAN.md), and [validation contract](docs/VALIDATION.md).
- [Model catalog](configs/model-candidates.json), [dependency lock](configs/dependencies.lock.json), and [coding rules](docs/CODING.md).
- [Research references](docs/RESEARCH.md), [MoE source review](research.md), and [LiteRT-LM/AirLLM lessons](docs/UPSTREAM_DESIGNS.md).
- [Production-path tests](docs/PRODUCTION_PATH.md) and [complete-fixture coverage](docs/UNIT3_REPORT.md).

Project-owned code and documentation are available under [MIT](LICENSE). Third-party source, upstream patch context, and the extracted Qwen template keep their own terms in [third-party notices](THIRD_PARTY_NOTICES.md). Model weights are separate downloads governed by their model licenses.