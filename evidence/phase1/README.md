# Phase 1 evidence

These records are machine-written outputs from the pinned tools and the physical
phone. They contain no serial number, account, or private application file.

## Artifact

- Model: `unsloth/Qwen3.5-2B-GGUF`, file `Qwen3.5-2B-Q4_K_M.gguf`
- SHA-256: `aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223`
- Bytes: 1,280,835,840
- Base model: `Qwen/Qwen3.5-2B`

## Binaries

- llama.cpp commit: `ff5a23376725e5a889fa0bfb987b7541dc304da0`
- Baseline harness (Android), SHA-256: `c28e365417cd58fc32dac77723011de9e59c8116978570688fc57a3fc838bbe8`
- Gate harness (host), SHA-256: `4059cd72c542bbb9a2bf0e125f032d490709efd8e78c5cde97e4067428adb35a`
- Gate harness (Android), SHA-256: `874e345fa14bd1231c5774ef2e645aa90dfd12f9c76090c636166938cd13e983`

## Runs

- Baseline host: context 512, batch 512, 4 threads, greedy, seed 1234, chat template, 16 new tokens.
- Baseline phone: same settings. Four runs: the first with verbose stderr, then three with stderr retained.
- Gate host and phone: 32 F32 experts of 64x128, top-k 4, 1 decode token and 8 prefill tokens, 17-slot arena, 4 threads.
- Probe records are the raw device observations before and after the measured runs.

Phone deployment paths were `/data/local/tmp/ame-engine` and
`/data/local/tmp/ame-probe`. The model was transferred with `adb push`; the
device's own storage was not otherwise modified.
