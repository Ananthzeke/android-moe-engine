# LiteRT-LM and AirLLM design research

Reviewed on 19 October 2026. This is primary-source inspection, not a build, Android compatibility test, or benchmark. The inspected revisions are:

| Project | Inspected commit | Role here |
| --- | --- | --- |
| [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM/tree/9a5f0ae124ae952e080105e6d45bd3c6df213e3c) | `9a5f0ae124ae952e080105e6d45bd3c6df213e3c` | Resource, request-state, execution-plan, and artifact design reference. |
| [AirLLM](https://github.com/lyogavin/airllm/tree/421b183d990bb69800200b98562baa0c2d612689) | `421b183d990bb69800200b98562baa0c2d612689` | Layer/expert weight lifetime and selective storage-read reference. |

Neither becomes an additional production backend in this update. The existing MNN-first, llama.cpp-fallback experiment remains open. Borrowing an architecture does not require adopting its framework or model format.

## LiteRT-LM separates model resources from request execution

[EngineAdvancedImpl](https://github.com/google-ai-edge/LiteRT-LM/blob/9a5f0ae124ae952e080105e6d45bd3c6df213e3c/runtime/core/engine_advanced_impl.cc) owns the model resources, tokenizer, environment, and execution manager. Its destruction order keeps backing resources alive until execution management is torn down. [SessionAdvanced](https://github.com/google-ai-edge/LiteRT-LM/blob/9a5f0ae124ae952e080105e6d45bd3c6df213e3c/runtime/core/session_advanced.h) tracks request phase and borrows the manager and tokenizer rather than independently owning the entire model.

Adopt this ownership distinction internally. Immutable artifact metadata and backing files outlive backend consumers; request-local state has a shorter lifetime. Keep one active generation and one session-owned cache. This is not a reason to add a public session manager or shared mutable global caches.

The newer [executor interface](https://github.com/google-ai-edge/LiteRT-LM/blob/9a5f0ae124ae952e080105e6d45bd3c6df213e3c/runtime/executor/llm_executor_interface.h) passes explicit state to prefill and decode operations. The [concrete compiled executor](https://github.com/google-ai-edge/LiteRT-LM/blob/9a5f0ae124ae952e080105e6d45bd3c6df213e3c/runtime/executor/llm_litert_compiled_model_executor.cc) also retains internal state. Therefore, separating project types does not establish a stateless backend or remove retained bytes from accounting.

The concrete CPU path supports configurable prefill chunks for dynamically exported graphs. Static artifacts have signature/shape constraints. [Executor settings](https://github.com/google-ai-edge/LiteRT-LM/blob/9a5f0ae124ae952e080105e6d45bd3c6df213e3c/runtime/executor/llm_executor_settings.h) describe that restriction. Adopt a capability-checked execution plan, not arbitrary chunk sizes. Preserve backend pending-token, position, and state-update conventions.

[State buffers](https://github.com/google-ai-edge/LiteRT-LM/blob/9a5f0ae124ae952e080105e6d45bd3c6df213e3c/runtime/executor/litert/state.cc) can grow and use different banking arrangements. Budget the maximum overlapping old/new allocation during growth, plus any staging or copies. An eventual steady-state size alone is insufficient.

## LiteRT-LM delivery streaming is not an expert cache

The [.litertlm schema](https://github.com/google-ai-edge/LiteRT-LM/blob/9a5f0ae124ae952e080105e6d45bd3c6df213e3c/schema/core/litertlm_header_schema.fbs) separates typed model, metadata, tokenizer, and external-weight sections with 64-bit ranges. This is useful inspiration for a checked sidecar index over the selected backend's native files. Its format block alignment is not the phone's page size or a universal tensor alignment.

Crucially, [ModelResourcesStreaming::SetWeightsFromStream](https://github.com/google-ai-edge/LiteRT-LM/blob/9a5f0ae124ae952e080105e6d45bd3c6df213e3c/runtime/components/model_resources_streaming.cc) allocates the whole requested weight section plus alignment before filling it in chunks. Its section release does not prove that compiled backend copies have disappeared. External sections, file-backed regions, chunked delivery, and bounded expert residency are different mechanisms.

Adopt load-strategy provenance and validate ranges before materialization. Do not describe chunked transfer as a small resident-memory limit, treat `.litertlm` as an MNN input, or build a new universal container.

The [session API](https://github.com/google-ai-edge/LiteRT-LM/blob/9a5f0ae124ae952e080105e6d45bd3c6df213e3c/runtime/engine/engine.h) warns that cancellation leaves a session unsuitable for reuse. Our engine must drain consumers, release leases, and discard damaged request state. Reuse requires a proven reset or a new backend execution instance.

[Accelerator memory monitoring](https://github.com/google-ai-edge/LiteRT-LM/blob/9a5f0ae124ae952e080105e6d45bd3c6df213e3c/runtime/engine/accelerator_memory_monitor.h) is observation rather than enforcement. Its Android DMA-BUF view excludes driver-owned allocations and can be unavailable. Keep subset observations and hard allocation coverage separate.

## Current AirLLM has conditional expert streaming

The inspected [AirLLM base](https://github.com/lyogavin/airllm/blob/421b183d990bb69800200b98562baa0c2d612689/air_llm/airllm/airllm_base.py) constructs the actual Transformers model on meta tensors and delegates forward/generation to it. Module hooks load weights before execution and release afterward. The same source contains `_setup_expert_streaming`, `_expert_pre_hook`, and `_expert_post_hook`; it is inaccurate to describe this revision as only whole-layer streaming.

That expert path requires an expert-prefix mapping, separately addressable numeric expert tensors, safetensor storage, and model execution that invokes selected experts separately. [The Kimi mapping](https://github.com/lyogavin/airllm/blob/421b183d990bb69800200b98562baa0c2d612689/air_llm/airllm/airllm_kimi_k3.py) supplies the prefix. [The inspected Qwen3.5 mapping](https://github.com/lyogavin/airllm/blob/421b183d990bb69800200b98562baa0c2d612689/air_llm/airllm/airllm_qwen3_5.py) does not establish that path for our MoE candidates. Fused expert tensors do not automatically become independent slices. Model naming checks alone do not prove compatibility.

The useful mechanism is selective materialization immediately around a real consumer. Keep routing and model mathematics in the chosen native backend. Require a complete expert layout and lifetime proof before installing a similar boundary there.

AirLLM tracks moved tensor names when selective unloading is necessary. It preserves resident children and tied embeddings instead of recursively removing every descendant. Adapt that idea into canonical storage ownership, alias-aware views, and leases. A blanket parent unload can free shared data or preserve hidden copies.

The hook path does not establish our required cross-token LRU, fixed byte arena, complete allocator coverage, or failure/cancellation cleanup. Successful post-hooks are not a substitute for RAII unwinding when load or compute fails.

## Read exact units and account for materialization

[AirLLM utilities](https://github.com/lyogavin/airllm/blob/421b183d990bb69800200b98562baa0c2d612689/air_llm/airllm/utils.py) expose tensor-name inspection and subset reads, split checkpoints by module ownership, and reuse some existing shards. Adopt metadata-first indexing and avoid unnecessary artifact copies when identity and immutability are proven. Its existence-based completion markers are not our integrity contract; require source/converter identity, hashes, and atomic publication.

Optional AirLLM compression can require decompression during loading and CUDA-dependent operations. The base also preserves packed payloads and companions in applicable quantizer paths. Neither means every selected artifact executes directly in its stored bit width. Record stored bytes, materialized dtype/size, quantization companions, backend packing, and simultaneous temporary copies. Do not transplant Python, CUDA, or allocator-trimming assumptions into Android/Bionic.

Its next-module prefetch uses one worker, not a predictor for future routed experts. A pinned-copy threshold is not a total-memory cap. Keep demand reads for the MVP. Any later overlap needs a reservation for current leases, next-unit loading, packed/expanded copies, state, and workspace at once.

Whole-layer streaming is a possible later comparison for dense models. It still requires all selected experts in a streamed MoE layer unless a finer consumer boundary is proven. Re-reading weights each decode step can dominate latency. This update does not add a whole-layer runtime or claim a phone speed.

## Resulting design changes

| Borrowed idea | Change to this repository | Proof required |
| --- | --- | --- |
| Resource/request separation | Explicit internal owners and destruction order; public API stays small. | Request reset and cancellation cannot leave stale state or invalidate backing files. |
| Prefill/decode distinction | Model-specific phase plan and state-growth admission. | Supported shapes, token positions, router outputs, and numerical parity. |
| Selective expert loading | Complete tensor/storage groups and precise release. | Addressable extents, companions, aliases, and consumer completion. |
| Metadata-first native sections | Versioned sidecar, native-file identity, and materialization provenance. | No full-model load during inspection or unchecked stale shard reuse. |
| Overlap and expansion awareness | Transition-peak accounting and future prefetch gate. | Covered peak fits before allocation; failures unwind every reservation. |

No backend is switched and no candidate model is promoted to supported. LiteRT-LM may serve as an optional external small-model characterization using its documented build and verified artifact. Different conversions must be reported as different artifacts, not a causal residency comparison. Both inspected projects use Apache-2.0 licenses; retain applicable notices if code is later reused and inspect dependency licenses separately.
