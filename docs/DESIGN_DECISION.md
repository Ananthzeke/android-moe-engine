# Design decision

Date: 18 October 2026. Scope: the initial design and DeepSeek handoff, not implementation approval of an untested engine.

## Chosen direction

Build a CPU engine around one pinned backend fork with a private routed-expert integration. Keep tokenization, graph mathematics, and kernels in that backend. Own expert storage, accounting, residency, leases, cancellation, and evidence here. Evaluate MNN first because inspected code exposes an expert-module boundary. Select the production backend only after the phase 1 source experiment.

The small public surface lets a caller inspect, open, and generate. The caller does not coordinate file loading, expert cache slots, or backend tensors.

## Alternatives considered

| Shape | Decision and cost |
| --- | --- |
| Standalone runtime and new universal model format | Rejected for MVP. Kernel, tokenizer, conversion, hybrid-state, and numerical work would obscure the storage experiment. |
| Unmodified runtime with mmap | Retained as baseline. It delegates residency to the OS and does not establish our bounded expert-memory ownership. |
| Multiple production backend adapters | Deferred. Integration experiments may compare two, but the MVP maintains one. |
| Backend fork with explicit expert lifetime | Selected provisionally. It provides the needed control at the cost of a small upstream patch and source-level feasibility risk. |

## Independent designs and synthesis

Three independent candidates used the configured architect models: gpt-6.1-sol, gpt-6-sol, and gpt-6-luna. Each compared whole architecture shapes and recommended a gated backend fork. A separate gpt-6-luna cross-judge evaluated the completed designs. The primary author read every candidate and agreed with its selected base.

Scores are design-review judgments from 1 to 5, not performance measurements:

| Candidate | Integration gate | Memory and I/O | Model correctness | Evidence | Interface and scope |
| --- | ---: | ---: | ---: | ---: | ---: |
| gpt-6.1-sol | 5 | 4 | 4 | 5 | 5 |
| gpt-6-sol | 4 | 4 | 4 | 5 | 3 |
| gpt-6-luna | 4 | 3 | 3 | 4 | 2 |

Use the gpt-6.1-sol candidate as the base for its single owned budget and narrow public interface. Adapt the gpt-6-sol candidate's explicit source-experiment matrix and interruption records. Retain the consensus outcome that a failed streaming gate still leaves a verified small CPU engine. Reject broader installation, Android UI, and runtime-plugin APIs for the MVP.

Review identified four corrections now included in the architecture:

- Seeing routed expert IDs does not prove that retained expert-module weights can be bounded.
- A decode-sized cache cannot be assumed to fit prefill's union of experts. Prove bounded scheduling or refuse.
- A regular-file read has no guaranteed wall-clock cancellation deadline. Bound chunks and measure latency.
- An arena cap is not a process-memory cap. Backend allocation coverage and OS page-cache behavior remain separate.

## Accepted tradeoffs

We accept upstream patch maintenance to reuse numerical kernels. We accept demand-read latency to keep ownership and I/O interpretable. We accept one-session CPU text scope to establish correctness before optional accelerators and an app interface.

## Unresolved facts and resolution owners

The phase 1 implementer resolves expert backing lifetime, exact hook placement, allocation coverage, minimum live extents, and conversion support. The physical-target evaluator resolves storage wait, compute cost, thermal behavior, and achievable latency. These are experimental questions, not preference questions for the repository owner.

## Implementation reconciliation

No engine changes have been implemented. Future accepted deviations must update the affected architecture, API, phase gates, and this section together. Record the source experiment or reviewer decision that supports each deviation. Do not alter a contract merely to relabel a failing test as success.

The next implementation step is phase 0 followed by one real small-model baseline and the source-level MoE fixture.

## LiteRT-LM and AirLLM revision, 19 October 2026

The owner requested source research and useful architecture updates. [The research report](UPSTREAM_DESIGNS.md) records the pinned implementations. Two source explorers examined the runtime and weight-lifetime paths. Three independent design candidates compared retaining the current backend gate with switching to LiteRT-LM or adopting an AirLLM-style runtime. A separate cross-judge reviewed all three completed candidates.

Keep MNN provisional and retain the existing llama.cpp fallback. All candidates agreed that neither upstream's inspected streaming path closes our native expert-lifetime gate. The synthesized revision uses the gpt-6-luna candidate's source distinctions and resource/request boundary, the gpt-6.1-sol candidate's private capability/request-plan records, and the gpt-6-sol candidate's concrete lifetime and phase proof gates. The public API and one-production-backend scope remain intact.

The accepted changes are:

- Separate model-resource and request-state lifetimes internally without implying safe concurrent executor or cache sharing.
- Derive supported prefill/decode schedules from the exact artifact and retain backend token/state semantics.
- Index canonical backing storage, aliases, companions, and materialization requirements. Release only owned transient weights.
- Admit load, prefill, decode, state-growth, and packed/expanded transition peaks. Distinguish chunked delivery from bounded residency.
- Drain execution after cancellation or failure, then destroy damaged request state. Reuse requires proven reset or backend recreation.
- Reuse verified native extents and publish derived manifests atomically. Completion markers alone do not establish identity or integrity.

We rejected an immediate LiteRT-LM backend switch, a Python/CUDA AirLLM port, a new universal container, and default prefetch. Whole-layer streaming and next-unit overlap remain later measured experiments. LiteRT-LM may be an external small-model characterization, with conversion and build differences explicit.

The decision follows source inspection and design review only. No native hook, allocation ceiling, model compatibility, Android build, inference result, or speed has been verified by this revision. The phase 1 implementation still owns those proofs. This update reconciles the existing architecture, plan, validation contract, and DeepSeek prompt before handoff.

## Host coding foundation (2026-10-19)

Choose CLI plus one internal checked-file-extent primitive over CLI-only scaffolding or a full engine skeleton. CLI-only offers little domain validation; empty runtime libraries and public engine declarations would freeze backend-dependent interfaces before the source gate. The private constructor and explicit extent errors prove containment without a generic result framework. Empty extents including EOF are valid positions; future required tensors add nonempty/layout checks.

Use C++20, target-scoped strict warnings, pinned development tools, and one offline verifier for Debug, Release, and host ASan/UBSan. No backend or model is downloaded by verification. Planned modules retain ownership READMEs only. Android ARM64/API31 has a conditional preset, but NDK identity, cross-build, ELF inspection, and phone execution remain unverified; this does not complete Phase 0. See [contribution instructions](../CONTRIBUTING.md) for commands.
## Device-local probe and pinned Android build (2026-10-19)

Choose a device-local, read-only probe over a host-side ADB orchestrator. The executable reads `/proc`, `/sys`, and `statvfs` itself, which matches the eventual on-phone engine, works from Termux without a host, and keeps observation parsing out of shell pipelines. ADB is a deployment transport, not the observation source. A host invocation reports `execution_tag: host`; it does not fabricate a phone profile. This refines the earlier implicit assumption that `ame probe` would always query through ADB. The physical run is retained in [the phase 0 evidence](../evidence/phase0/probe-vivo-1920.json).

Pin NDK r27c (27.2.12479018) because it is a stable LTS release with API 31 support and archives reproducibly. The probe reports only `fp asimd evtstrm aes pmull sha1 sha2 crc32 cpuid`, so the Android build uses the NDK's ARMv8-A baseline and does not enable dot-product, i8mm, or bf16 instructions. The produced ELF was inspected for architecture, interpreter, linkage, and unsupported instructions before deployment.

Treat probe values as raw observations. Thermal zones include sentinel and implausible values, and power-supply attributes are unreadable from the ADB shell uid; the probe reports them raw or null with reasons instead of guessing. A snapshot is not a reservation. No performance, model support, or streaming claim follows from this build.
## Backend selection and kernel-boundary gate (2026-10-19)

Select llama.cpp (`ff5a23376725e5a889fa0bfb987b7541dc304da0`) provisionally as the production backend. It builds for host and Android ARM64, supports the qwen35 architecture in source, runs the exact Qwen3.5-2B Q4_K_M artifact on the physical phone, and exposes a shape-preserving expert-ID boundary at `GGML_OP_MUL_MAT_ID`. MNN exposes routing but constructs every expert module in `MoEModule::create` and retains them, so it fails the bounded-lifetime gate without a substantial patch. The detailed matrix and evidence are in [the feasibility report](backend-feasibility.md).

The fixture gate is a real result, not a mock. A fully resident expert table and a 17-slot arena backing a compact ggml tensor produced bitwise-identical decode and prefill outputs, with zero mismatched floats on both host and phone. A 17-expert union against 16 slots returned `ResourceLimit`. The arena directly backs the staged tensor, so no second staging copy is hidden in this fixture. The gate does **not** patch llama.cpp's runtime: materializing the selected union inside a real model graph, remapping IDs, and restoring graph state remains the next implementation unit. No streaming or speed claim follows.

Two portability findings are recorded. First, the pinned NDK's libc++ does not expose `std::stop_token`; the internal boundary uses a project-owned atomic `Cancellation` instead, and the public sketch is updated accordingly. Second, greedy token IDs diverge between host and phone after the first generated token while all phone trials agree. This is a cross-architecture floating-point difference, not a model substitution; the project does not claim cross-device numerical parity and does not loosen the exact-comparison rule.

Keep the overall memory cap conditional. Project code owns the arena, slot metadata, and read bytes, but llama.cpp owns a 505.53 MiB compute buffer at context 512/batch 512, KV/recurrent state, and mmap-backed pages. The baseline and gate establish a working path and a testable boundary, not backend allocation coverage.
