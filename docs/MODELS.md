# Model selection

The [model catalog](../configs/model-candidates.json) is the source of truth for candidate IDs and support status. The review date is 18 October 2026. None has verified inference evidence from this repository yet.

## Begin with a real small baseline

Start with the catalog's Qwen3.5 2B candidate, then test its 4B candidate if admission succeeds. Choose an actual supported 4-bit conversion and record the source revision, converter, quantization details, tokenizer, template, shard hashes, and resulting byte footprint. The candidate bit depth is not a promise of a specific artifact size or quality.

Small baselines exercise hybrid state handling before large MoE streaming. Keep generation short and context at 512 initially. Do not substitute advertised maximum context for a feasible phone configuration.

Gemma 4 E2B is an intermediate option. Its effective-compute name excludes substantial embedding storage. Index and account for those tables according to actual access patterns. Treat embedding lookup residency as a separate future optimization, not an MoE expert by another name.

## Select a large target using evidence

The first hypothesis is Gemma 4 26B-A4B because fewer total weights can reduce storage footprint. Qwen3.6-35B-A3B is the other primary MoE candidate. Actual converter support, live expert footprint, tensor layout, and the backend feasibility gate decide order. Resolve instruction-tuned artifact variants explicitly.

These models select a subset of experts for each token. The full model remains necessary. An expert chosen on the next token may be absent from the cache, and prefill can touch a much larger union of experts.

The theoretical payload formula is `parameters * bits_per_weight / 8`. This omits quantization metadata, nonquantized tensors, vision components, alignment, and runtime memory. Use it only as a lower-bound calculation. Admission must inspect the artifact and real allocation trace.

The newer Qwen3.8-27B candidate is dense. Its feed-forward weights are not a sparse expert pool, so defer it until there is evidence for a separate dense streaming experiment.

## Promote support precisely

Add an artifact record only after validation. Record exact model ID and revision, backend commit, converter commit, quantization, file sizes/hashes, tested context, execution mode, correctness evidence, and run IDs.

Keep these statuses distinct:

1. Candidate identified from a model card.
2. Conversion validated against a reference.
3. Small or synthetic integration tested.
4. Exact artifact runs on a host.
5. Exact artifact runs on the physical phone under recorded settings.

A family-support announcement establishes none of the later artifact-specific statuses by itself. Review model and converter licenses before redistribution.
## Verified baseline artifact (2026-10-19)

`unsloth/Qwen3.5-2B-GGUF`, file `Qwen3.5-2B-Q4_K_M.gguf`, 1,280,835,840 bytes, SHA-256 `aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223`, is a derived GGUF of `Qwen/Qwen3.5-2B`. It reached status 5 for the text-only baseline: it ran on the physical phone at context 512 with recorded settings and token IDs. The base safetensors artifact was not converted by this project, so the base-model entry remains a candidate.

This verification is narrow. It establishes a working CPU text baseline and provenance, not MoE streaming, long context, quantized expert extents, or cross-architecture numerical parity. Host and phone tokenization matched; generated token IDs diverged after the first token. Four phone trials agreed with each other. The artifact is not promoted for large-model or multimodal use.
## Verified large artifact (2026-10-19)

`ggml-org/Qwen3.6-35B-A3B-GGUF`, file `Qwen3.6-35B-A3B-Q4_K_M.gguf`, 20,419,565,568 bytes, SHA-256 `671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7`, reached status 5 for a text-only storage-backed run on the physical phone. Three runs at context 512 produced identical token IDs with a 28-38 MB project arena and about 2.06 GiB peak process RSS. Decode was 0.37-0.61 token/s; TTFT was 9.6-17.8 s.

The claim is deliberately narrow. No full-resident reference logits or tokens exist for this artifact, so parity rests on the bitwise synthetic runtime gate and run-to-run token equality. The process peak is not an enforced cap, the store did not cache across tensor switches, and contexts 1,024 and 2,048 have not run. The base `Qwen/Qwen3.6-35B-A3B` entry remains a candidate; the exact tested GGUF is the promoted artifact.
