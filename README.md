# android-moe-engine

An Android inference-engine research project for running quantized language models under limited memory, including storage-backed execution of larger mixture-of-experts models.

The first physical target is a vivo S1 Pro with a Snapdragon 665, Android 12, and an 8 GB RAM configuration. Root is not required. The large-model question is whether exact expert streaming can complete useful work on this hardware within a declared memory budget. Throughput and feasibility remain unmeasured.

## Current state

This repository contains a buildable C++20 host coding foundation and a read-only device probe:

- a help/version/probe CLI, checked file extents, a validated expert-table layout, and boundary tests;
- fixture-tested observation of memory, CPU, storage, OS, thermal, and power data;
- a pinned Android NDK r27c cross-build whose AArch64 ELF was executed on the connected phone;
- a bounded expert residency store with move-only leases, byte-based LRU, checked demand reads, and cancellation;
- a llama.cpp baseline on host and phone for Qwen3.5-2B Q4_K_M, plus an expert-ID remap gate against the real CPU `MUL_MAT_ID` kernel that is bitwise identical on both;
- a bounded run recorder and aggregator with atomic start/final records and hash checks;
- a metadata-first GGUF index that computes expert extents and refuses an over-budget large artifact;
- an offline local/CI verifier for Debug, Release, and host ASan/UBSan.

The probe ran on the physical vivo 1920 and its snapshot is retained at [evidence/phase0/probe-vivo-1920.json](evidence/phase0/probe-vivo-1920.json). The Qwen3.5-2B baseline reached 4.36-4.50 decode tokens/s at context 512 with peak process RSS around 1.37 GiB; the host reference and the kernel-boundary gate are recorded in [evidence/phase1](evidence/phase1). A recorded three-trial phone campaign is in [evidence/phase3](evidence/phase3). A metadata-only preflight of Qwen3.6-35B-A3B Q4_K_M in [evidence/phase4](evidence/phase4) computes a 2.30 GB streaming estimate but refuses full-resident execution; it is not a run. Expert streaming is **not** integrated into the backend runtime, and the overall engine memory cap remains unproven. Commands beyond `probe` still describe planned interfaces.

## Start implementation

Give DeepSeek access to this repository and paste [the build prompt](prompts/DEEPSEEK_BUILD.md). The prompt directs it through a working small-model baseline, an expert-lifetime feasibility experiment, bounded residency, and physical-device evaluation.

The proposed engine reuses a pinned backend's graph execution, tokenizer, and numerical kernels. This project owns expert storage, memory reservations, residency, cancellation, and evidence. MNN is the first integration candidate; llama.cpp is the next candidate. The inspected pinned revisions both implement the selected model families, but neither has been built or executed here. A failed source-level feasibility gate can select llama.cpp before residency implementation. The MVP has one production backend.

The design also draws from source inspection of LiteRT-LM and current AirLLM. It adopts explicit resource/request lifetimes, supported prefill plans, selective weight ownership, and accounting for materialization peaks. These references do not establish phone compatibility or change the production backend.

## Design documents

- [Architecture and API sketch](docs/ARCHITECTURE.md) defines ownership, execution, and memory contracts.
- [Implementation plan](docs/IMPLEMENTATION_PLAN.md) defines phases and acceptance gates.
- [Validation contract](docs/VALIDATION.md) defines correctness and performance evidence.
- [Model selection](docs/MODELS.md) explains the current candidates and support policy.
- [Sources and inspected backend code](docs/RESEARCH.md) distinguishes observations from hypotheses.
- [LiteRT-LM and AirLLM research](docs/UPSTREAM_DESIGNS.md) records useful mechanisms, limits, and adopted changes.
- [Design decision](docs/DESIGN_DECISION.md) records alternatives and independent review.
- [DeepSeek build prompt](prompts/DEEPSEEK_BUILD.md) is the implementation handoff.
- [Repository preparation checklist](docs/WORKPLAN.md) tracks this initial handoff.

## Configuration examples

[The device profile](configs/device-vivo-s1-pro.json) records a snapshot, not permanent memory availability. [The model catalog](configs/model-candidates.json) records candidates, not verified support. [The dependency lock](configs/dependencies.lock.json) pins the NDK and inspected backend revisions. [The MVP configuration](configs/mvp.example.json) proposes conservative settings that implementation must admit or reject using fresh observations.

Follow [contribution instructions](CONTRIBUTING.md) to bootstrap pinned tools, then run `.venv/bin/python tools/verify.py`. This checks formatting, documentation/configuration, and Debug, Release, and ASan/UBSan host builds and tests. See [coding rules](docs/CODING.md). It does not test inference.

## Scope

The first release is CPU-first, text-only, one session, and a command-line executable that runs through ADB or Termux. Start with a 512-token total context. Advance toward 2,048 only after memory admission succeeds. GPU execution, an Android UI, predictive prefetch, and concurrent sessions follow measured CPU results.

Weights, device serials, credentials, and raw private device reports stay outside Git. Select a project license before distributing engine code. Preserve upstream notices and review each model's license before distributing converted artifacts.