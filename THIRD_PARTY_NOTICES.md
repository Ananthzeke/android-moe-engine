# Third-party notices

The root [MIT license](LICENSE) covers project-owned code and documentation. It does not replace third-party licenses, notices, model terms, or the license of extracted fixtures.

## llama.cpp and ggml

The production backend is [ggml-org/llama.cpp at ff5a23376725e5a889fa0bfb987b7541dc304da0](https://github.com/ggml-org/llama.cpp/tree/ff5a23376725e5a889fa0bfb987b7541dc304da0), under MIT, copyright 2023-2026 The ggml authors. Its [full license](LICENSES/llama.cpp-MIT.txt) is retained here.

The backend is fetched and built separately. Upstream context in `patches/` and the generator patches in `evidence/unit3/` retains this license. Local modifications are identified by the patch files and Git history. Preserve the notices when redistributing patched source or linked binaries.

## Qwen chat-template fixture

`tests/fixtures/qwen36-chat-template.jinja` is an unchanged template extracted from the independently enrolled Qwen3.6-35B-A3B Q4_K_M artifact. It is third-party model material, excluded from the project's MIT grant.

- Source model: [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B).
- Copyright 2026 Alibaba Cloud. License: Apache License 2.0.
- [Full upstream license](LICENSES/Qwen-Apache-2.0.txt), retrieved from [revision 995ad96eacd98c81ed38be0c5b274b04031597b0](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/resolve/995ad96eacd98c81ed38be0c5b274b04031597b0/LICENSE).
- [Extraction provenance](tests/fixtures/qwen36-chat-template.provenance.json) binds the fixture to its GGUF and template hashes. The license-source revision above identifies the retrieved license, not a verified conversion revision for the GGUF.

No model weights are included. Obtain models separately and follow each model's applicable terms.

## Libraries supplied by the external backend

The server build uses headers and libraries from the pinned backend checkout. Their licenses remain separate. This source repository does not vendor their implementations.

| Component | Notice and retained license |
| --- | --- |
| cpp-httplib | MIT, copyright 2017 yhirose; the pinned header also identifies copyright 2026 Yuji Hirose. [License](LICENSES/cpp-httplib-MIT.txt), [source](https://github.com/ggml-org/llama.cpp/tree/ff5a23376725e5a889fa0bfb987b7541dc304da0/vendor/cpp-httplib). |
| nlohmann JSON 3.12.0 | MIT, copyright 2013-2025 Niels Lohmann. [License](LICENSES/nlohmann-json-MIT.txt), [source header](https://github.com/ggml-org/llama.cpp/blob/ff5a23376725e5a889fa0bfb987b7541dc304da0/vendor/nlohmann/json.hpp). |
| xxHash | BSD 2-Clause, copyright 2012-2021 Yann Collet. [License](LICENSES/xxHash-BSD-2-Clause.txt), [source](https://github.com/ggml-org/llama.cpp/tree/ff5a23376725e5a889fa0bfb987b7541dc304da0/vendor/hash/xxhash). |
| rotate-bits | MIT, copyright 2021 William Casarin. [License](LICENSES/rotate-bits-MIT.txt), [source](https://github.com/ggml-org/llama.cpp/tree/ff5a23376725e5a889fa0bfb987b7541dc304da0/vendor/hash/rotate-bits). |
| Backend SHA256 | Public-domain implementation credited to Igor Pavlov. [Notice](LICENSES/SHA256-public-domain.txt), [source](https://github.com/ggml-org/llama.cpp/tree/ff5a23376725e5a889fa0bfb987b7541dc304da0/vendor/hash/sha256). |

When distributing binaries, retain the licenses and notices for the components actually included, plus the relevant backend and toolchain runtime notices. This list does not claim that every optional vendor backend is linked into this engine.

MNN, LiteRT-LM, AirLLM, BigMoeOnEdge, and other research references retain their own terms. Source inspection or citation does not mean those implementations are bundled or relicensed here.
