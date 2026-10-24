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

The engine implementation has since been reconciled against this decision: the pinned llama.cpp backend, runtime materializer, residency store, harness, and physical-device runs are recorded in [the final report](FINAL_REPORT.md) and [release readiness](RELEASE_READINESS.md). The unresolved release gates are listed there; this section is not a claim that they passed. Future accepted deviations must update the architecture, API, phase gates, and this section together.

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

Use C++20, target-scoped strict warnings, pinned development tools, and one offline foundation verifier for Debug, Release, and host ASan/UBSan. No backend or model is downloaded by verification. Android ARM64/API31 has a conditional preset; NDK r27c cross-build, ELF inspection, and phone execution have since passed. The 35B materializer and streaming path are covered by the separate integration matrix, whose current gate status is in [release readiness](RELEASE_READINESS.md).
## Device-local probe and pinned Android build (2026-10-19)

Choose a device-local, read-only probe over a host-side ADB orchestrator. The executable reads `/proc`, `/sys`, and `statvfs` itself, which matches the eventual on-phone engine, works from Termux without a host, and keeps observation parsing out of shell pipelines. ADB is a deployment transport, not the observation source. A host invocation reports `execution_tag: host`; it does not fabricate a phone profile. This refines the earlier implicit assumption that `ame probe` would always query through ADB. The physical run is retained in [the phase 0 evidence](../evidence/phase0/probe-vivo-1920.json).

Pin NDK r27c (27.2.12479018) because it is a stable LTS release with API 31 support and archives reproducibly. The probe reports only `fp asimd evtstrm aes pmull sha1 sha2 crc32 cpuid`, so the Android build uses the NDK's ARMv8-A baseline and does not enable dot-product, i8mm, or bf16 instructions. The produced ELF was inspected for architecture, interpreter, linkage, and unsupported instructions before deployment.

Treat probe values as raw observations. Thermal zones include sentinel and implausible values, and power-supply attributes are unreadable from the ADB shell uid; the probe reports them raw or null with reasons instead of guessing. A snapshot is not a reservation. No performance, model support, or streaming claim follows from this build.
## Backend selection and kernel-boundary gate (2026-10-19)

Select llama.cpp (`ff5a23376725e5a889fa0bfb987b7541dc304da0`) provisionally as the production backend. It builds for host and Android ARM64, supports the qwen35 architecture in source, runs the exact Qwen3.5-2B Q4_K_M artifact on the physical phone, and exposes a shape-preserving expert-ID boundary at `GGML_OP_MUL_MAT_ID`. MNN exposes routing but constructs every expert module in `MoEModule::create` and retains them, so it fails the bounded-lifetime gate without a substantial patch. The detailed matrix and evidence are in [the feasibility report](backend-feasibility.md).

The fixture gate is a real result, not a mock. A fully resident expert table and a 17-slot arena backing a compact ggml tensor produced bitwise-identical decode and prefill outputs, with zero mismatched floats on both host and phone. A 17-expert union against 16 slots returned `ResourceLimit`. The arena directly backs the staged tensor, so no second staging copy is hidden in this fixture. The gate does **not** patch llama.cpp's runtime: materializing the selected union inside a real model graph, remapping IDs, and restoring graph state remains the next implementation unit. No streaming or speed claim follows.

Two portability findings are recorded. First, the pinned NDK's libc++ does not expose `std::stop_token`; the internal boundary uses a project-owned atomic `Cancellation` instead, and the public sketch is updated accordingly. Second, greedy token IDs diverge between host and phone after the first generated token while all phone trials agree. This is a cross-architecture floating-point difference, not a model substitution; the project does not claim cross-device numerical parity and does not loosen the exact-comparison rule.

Keep the overall memory cap conditional. Project code owns the arena, slot metadata, and read bytes, but llama.cpp owns a 505.53 MiB compute buffer at context 512/batch 512, KV/recurrent state, and mmap-backed pages. The baseline and gate establish a working path and a testable boundary, not backend allocation coverage.
## Runtime expert materialization (2026-10-19)

Add two small patches to pinned llama.cpp. [`0001`](../patches/0001-ggml-moe-materializer.patch) lets the project supply a compact expert tensor and remapped IDs at the real CPU `MUL_MAT_ID` boundary; the tiled fast path is skipped when staging is active because it reads the original tensor. [`0002`](../patches/0002-llama-no-eager-mmap-prefetch.patch) disables eager `MAP_POPULATE`/prefetch for the streaming profile. Neither changes routing, arithmetic, accumulation, or expert selection.

The measured effect is large and recorded: with prefetch, a 9.48 GB Qwen1.5-MoE fixture peaked at 5.2-6.1 GiB process RSS and decoded at about 1.0 token/s; without it, the same seven-run token sequence peaked at about 1.20 GiB and decoded at about 2.0 token/s. TTFT fell from 4.9-8.4 s to about 2.1 s. This makes the loader's mapping policy part of the residency contract, not an implementation detail.

The no-prefetch change is scoped to the streaming profile. On the dense Qwen3.5-2B it reproduced token IDs exactly but reduced decode from 9.69 to 0.50 token/s and moved TTFT from 0.42 to 7.01 s, while peak RSS fell from 1.82 to 1.31 GiB. Do not make it the default for dense execution; make the loader policy part of the declared execution profile.

Keep the claims narrow. The project arena was 120-160 MB, not a process cap. The LRU had zero hits because each tensor switch recreated its store; 5.96 GB was read for 13 prompt/generated tokens. No full-resident reference parity exists for this model; bitwise equivalence is established at the synthetic kernel/runtime gate. The target 35B artifact has since run: three context-512 trials produced identical tokens with a 28-38 MB arena and about 2.06 GiB peak RSS at 0.37-0.61 token/s. No full-resident reference parity exists, so the bitwise synthetic gate and run-to-run equality carry the correctness claim. The process peak is not an enforced cap. The next units are one shared arena with per-tensor slot maps, bounded prefill microbatches, cancellation/failure cleanup, 1,024/2,048-context admission, and allocation coverage.
## Cache capacity decision (2026-10-19)

The 35B default path reads 5.90 GB from storage over nine forwards and spends 16.6-21.4 s in application reads. The routing trace shows a 960-key decode working set and only 237-612 keys of consecutive overlap. A 1,024-slot multi-table cache (604 MB) was implemented and measured against the 64-slot path: it cut bytes by 23.5% but raised read time by 5.0 s, reduced decode from 0.561 to 0.461 tok/s, and raised peak RSS by 539 MiB. Keep demand reads and the small arena as the default. Do not adopt a global LRU without a policy that reduces actual read latency; the next cache unit must measure per-tensor reuse and I/O latency, not only byte counts.

## Model-owned chat formatting (2026-10-20)

Use the pinned upstream common Jinja helpers for the HTTP server's model template.
The previous heuristic C template helper did not supply the actual template's
thinking flag; a bounded 35B greeting consumed its 48-token budget on thinking
text. Set `enable_thinking=false` by default through upstream generation inputs,
with a strict boolean HTTP opt-in. Preserve marked input parts and upstream
BOS/EOS policy when producing owned request tokens. Keep rendering in a private
server helper shared with the actual-model template fixture. Do not handwrite a
Qwen prompt format or change inference mathematics to produce shorter answers.

Scope-own loading-listener, model, streaming-store, and artifact lifetimes so
startup errors return cleanly. Host API checks, a Release-active template fixture,
and a forced-thinking mutation test passed. The final Android binary completed
`Hello!` through JSON and `4` through SSE on the enrolled 35B artifact, with normal
stops and no emitted thinking text. [The phone record](../evidence/api/phone-35b-direct/run.json)
retains the monitor-only PID guard failure and the successful continuation without
a server restart. [The runbook](RUN_35B.md) gives reproduction commands and timing
limits. This establishes a short 35B text API path; reference-logit parity,
long-form qualification, runtime mutation integrity, stop buffering, UTF-8
buffering, and bounded disconnect cancellation remain separate open work.

## Same-process recovery after a failed request (2026-10-24)

Patch 0001 latched `g_moe_error_latched` on a failed or cancelled `MUL_MAT_ID` and never
cleared it, so every later graph in the process aborted after its first node. The
`moe_recovery_gate` reproduces it with the real patched backend: after an injected read
failure or a cancellation, a dense `ggml_add` graph returns a non-success status. The patch
now exports `ggml_moe_reset()`, and `ggml_moe_set_materializer` calls it, so installing or
removing a materializer starts clean. The project calls `begin_moe_stream_request()` under
exclusive admission, after the previous context is destroyed and its leases released.

Evidence: the recovery gate passes with 1, 2 and 4 workers in Debug, Release and ASan/UBSan
(`tools/verify_inference.py`); with the reset call commented out the gate fails on the dense
graph. `complete_runner --recover-after-fault` fails one generation in the materializer, resets,
and runs a fresh context in the same process; its logits and token IDs equal an uninterrupted
streamed run byte for byte on the tiny qwen35moe fixture. This satisfies the stricter
condition in `prompts/DEEPSEEK_STABILITY_AND_IO.md` (drain, lease release, fresh-reference
parity) for the fixture only. A reset is never issued while another request holds admission.

The server publishes the terminal snapshot inside `RequestScope`, after lease release and
before admission release, on every exit including failed SSE writes. `/health` reads only the
published value. Limits: real-35B recovery parity, a blocked `pread`, and backend abort on
deadline or disconnect remain open.

## One cache owner: a layer-keyed, layer-partitioned arena (2026-10-24)

The routing replay of a 330-token phone trace (`evidence/core-opt/unit3/README.md`) shows
that capacity, not policy, sets the reads per token: 566 MB with the shipped 64-slot design,
then 361, 280, 224 and 184 MB at 0.57, 1.13, 1.70 and 2.26 GB. Per-layer partitioning
and LFU change the hit rate by at most 3.1 points over a (tensor, expert) LRU of the same bytes.
The layered cache is built for what partitioning does buy: one reservation and one read batch per
layer visit (three slices per expert), no cross-layer eviction, a pool that can be sized per layer, and
an arena that can be the only cache when the kernel is told to drop its copy. Per-layer LRU is the
default policy; LFU stays behind a flag. The phone A/B that decides the default layout is in
`evidence/core-opt/unit4/`: with 32 slots per layer (2.26 GB), LRU, and the kernel told to drop its
copy, requests were 28 to 40% faster than the shipped cache with identical token IDs in 25 trials, outside
the run-to-run spread; with 16 slots per layer the gain was mixed (faster prefill, slower decode) and
keeping the kernel's copy was 8% slower. The phone launcher defaults to the 32-slot layered cache; the
server binary keeps the shipped cache as its default because 2.2 GB is not a safe default for an unknown
device.

## Overlap through the eval callback, not a graph-level remap (2026-10-25)

Two ways to start reads before the expert operator were considered. Option B, porting the id-remap
structure of llama.cpp PR 25294 (an extra op after top-k that makes the expert matmuls index cache
slots directly), needs a new ggml op and graph surgery, a larger patch than 0001, and tracks an open
PR; its parity reports show divergence that grows with generation length for its wave-partitioned
prefill, which this project does not adopt. Option A uses an API that exists at the pinned commit
(`llama_context_params.cb_eval`, the `ffn_moe_topk-<layer>` name, `ggml_backend_sched_eval_callback`,
`llama_attach_threadpool`, `ggml_threadpool_new`), needs no patch change, and keeps the per-operator
hook as the authority on which experts are staged. The cost is one extra graph segment boundary per
routed layer per graph, which the phone A/B measures. Evidence and the measured effect are in
`evidence/core-opt/unit6/`.

## Tiled MUL_MAT_ID for staged experts (2026-10-25)

Patch 0001 skipped the tiled K-quant `MUL_MAT_ID` kernel whenever the materializer staged a compact
expert tensor, because that kernel read `dst->src[0]`. A staged run therefore used `vec_dot` where the
resident run used the tiled kernel, so the two could differ in the last bits on hosts that have it. The
effective expert tensor is now passed through. At the pinned commit the tiled kernel exists only for
x86 AVX builds (`ggml_tiled_supported` is a stub returning false otherwise) and runs only for experts
that receive at least eight rows in one graph (`ggml_tiled_min_batch`), so it is inert on the
Snapdragon 665, whose prefill units are 2 to 8 tokens. No phone prefill A/B was run, because the
source shows the kernel cannot run there. Host evidence: forcing it with `GGML_CPU_TILED_MM_FORCE=1`,
`moe_layered_gate` is bit-identical to the resident reference with the fix and mismatches on all 48
Q4_K/Q6_K operators without it; the harness gives identical token IDs and logits for staged and resident
runs at the same prefill unit on a 14-token prompt.

## Four reader threads are the phone default (2026-10-25)

Fifteen alternating fresh-process trials on the phone (evidence/core-opt/unit5) show that reading a layer's
missing experts with 2 or 4 threads instead of one cuts the short request by 30% and 39% and the 32-token
request by 24% and 32%, with the same token IDs and the same bytes read; the ranges do not overlap. Pinning
the readers to the cores with the lower `cpuinfo_max_freq` made no measurable difference (ten more trials), so
it stays off. The phone launcher starts four readers; the server binary defaults to one. The exactness argument
is structural: a batch ends only after every read has finished or been skipped, a failed batch publishes
nothing, and the same bytes land in the same slots in the same lease order as the serial path; it is checked by
the store tests under AddressSanitizer and ThreadSanitizer and by token and logit parity through the real
graph.
