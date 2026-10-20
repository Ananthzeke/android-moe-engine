# Sources and inspected evidence

Reviewed on 18 October 2026. Source inspection establishes code structure, not working streaming or performance. Model-card identities are recorded in [the catalog](../configs/model-candidates.json).

The [19 October LiteRT-LM and AirLLM review](UPSTREAM_DESIGNS.md) adds pinned primary-source evidence and the resulting architecture changes. It distinguishes delivery streaming from runtime residency and current conditional expert hooks from generic model support.

The [20 October MoE research review](../research.md) covers the supplied EdgeMoE,
FlashMoE, BigMoMo, PowerInfer-2, Paging the Experts, LLM in a flash, and MobileMoE
studies, their available implementations, and optimizations relevant to the
working 35B phone path. Its source inventory distinguishes full reads, partial
reads, unreleased code, and gated source.

## MNN integration evidence

Inspected commit: `dbf9ba61b08106a3b43426e295d1a29cb41cc8fe`.

[MoEModule.cpp at the inspected revision](https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/express/module/MoEModule.cpp) exposes selected expert indices in `MoEModule::onForward`. Decode calls expert submodules, and prefill groups work by expert. Construction retains the expert modules. This is a candidate integration boundary, not proof that their weights can be released or replaced within a budget. Trace module loading, packing, and ownership before selecting a hook.

[MNN runtime configuration](https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/transformers/README.md) documents mmap-related weight and KV options. These provide an off-the-shelf baseline. They do not establish an expert-aware fixed cache.

[MNN release notes](https://github.com/alibaba/MNN/releases) document Gemma4, Qwen3.5, and low-bit execution work. They do not establish successful execution of our selected artifact on this phone. Qwen3.6 conversion remains unverified here.

## llama.cpp alternative evidence

Inspected commit: `ff5a23376725e5a889fa0bfb987b7541dc304da0`.

[The CPU implementation](https://github.com/ggml-org/llama.cpp/blob/ff5a23376725e5a889fa0bfb987b7541dc304da0/ggml/src/ggml-cpu/ggml-cpu.c) includes `ggml_compute_forward_mul_mat_id`, selected-ID processing, and expert-strided addressing. Its current location matters: old links to `ops.cpp` are insufficient. A bounded slot integration still needs layout, scheduler, packing, and lifetime proof.

[The mapping implementation](https://github.com/ggml-org/llama.cpp/blob/ff5a23376725e5a889fa0bfb987b7541dc304da0/src/llama-mmap.cpp) accepts lazy ranges and conditionally avoids eager `MAP_POPULATE` when those ranges exist. Do not assume every current build eagerly faults all experts. Inspect the actual model loader's configuration.

[Android build guidance](https://github.com/ggml-org/llama.cpp/blob/ff5a23376725e5a889fa0bfb987b7541dc304da0/docs/android.md) provides a reference for native ARM64 builds. A documented backend does not guarantee its optional accelerators work on SM6125.

## Device and model evidence

[Android memory management](https://developer.android.com/topic/performance/memory-management) describes shared CPU/GPU physical RAM. GPU offload is not an extra memory bank on this target.

The project device JSON contains selected read-only ADB observations from this chat. The model is vivo 1920, SM6125, ARM64, Android 12. Root was unavailable in the ADB shell. Memory availability and rounded free storage were snapshots. No storage-speed or inference benchmark was performed.

The primary model cards linked from the catalog distinguish dense models, active MoE parameters, and effective versus total embedding-inclusive parameters. Read the exact artifact configuration during implementation rather than copying family-level arithmetic assumptions.

## Reuse published experiments critically

[A llama.cpp disk-offload discussion](https://github.com/ggml-org/llama.cpp/discussions/23324) and [an expert-pool discussion](https://github.com/ggml-org/llama.cpp/discussions/28248) contain candidate approaches and implementation experience. They are author-reported experiments on other hardware, not evidence of phone performance or upstream integration. Evaluate their code, current status, and licenses before reuse. Do not copy their reported speed into this project's goals or results.
