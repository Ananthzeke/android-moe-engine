# Backend feasibility

Date: 2026-10-19. Pinned revisions: llama.cpp
`ff5a23376725e5a889fa0bfb987b7541dc304da0`, MNN
`dbf9ba61b08106a3b43426e295d1a29cb41cc8fe`. This document distinguishes source
inspection, a fixture gate, and a real model baseline. It does not claim that
storage-backed expert streaming is integrated.

Decision: **llama.cpp is the provisional production backend**. MNN remains an
inspected alternative. The reasons and limits are below.

## Method

1. Inspected routing, expert layout, kernel access, weight ownership, and release in
   both pinned sources.
2. Built llama.cpp for the host and Android ARM64 with the pinned NDK.
3. Ran a project-owned fixture against llama.cpp's actual CPU `MUL_MAT_ID` kernel:
   resident reference versus a compact arena with remapped expert IDs.
4. Ran the exact Qwen3.5-2B Q4_K_M artifact on the host and on the physical phone.

Evidence:

- [expert-remap gate, host](../evidence/phase1/moe-gate-host.json)
- [expert-remap gate, phone](../evidence/phase1/moe-gate-phone.json)
- [baseline, host](../evidence/phase1/baseline-host.json)
- [baseline, phone](../evidence/phase1/baseline-phone.json) and trials
  [02](../evidence/phase1/baseline-phone-02.json),
  [03](../evidence/phase1/baseline-phone-03.json),
  [04](../evidence/phase1/baseline-phone-04.json)
- [phone probe before a measured run](../evidence/phase1/probe-before-02.json) and
  [after](../evidence/phase1/probe-after-02.json)

## llama.cpp source answers

| Question | Evidence and result |
| --- | --- |
| When are selected expert IDs available? | [`build_moe_ffn`](https://github.com/ggml-org/llama.cpp/blob/ff5a23376725e5a889fa0bfb987b7541dc304da0/src/llama-graph.cpp#L2152) computes `selected_experts` as `[n_expert_used, n_tokens]` before `build_lora_mm_id`; the same IDs drive decode and prefill. The gate used that layout directly. |
| What owns every expert's weights? | The model loader owns GGUF tensors; an expert is a matrix plane of a 3-D tensor with stride `nb02`. [`llama_moe_cache`](https://github.com/ggml-org/llama.cpp/blob/ff5a23376725e5a889fa0bfb987b7541dc304da0/src/llama-moe-cache.cpp#L1) keeps an LRU of host-resident experts in a **device** buffer for MoE offload; it does not evict the host-resident source or stream from storage. Project code owned no bytes before the gate. |
| Can one expert's backing be replaced or released? | Not through a public runtime API. The kernel reads `src0->data + cur_a*nb02` per selected expert in [`ggml_compute_forward_mul_mat_id`](https://github.com/ggml-org/llama.cpp/blob/ff5a23376725e5a889fa0bfb987b7541dc304da0/ggml/src/ggml-cpu/ggml-cpu.c#L1554). The result shape in [`ggml_mul_mat_id`](https://github.com/ggml-org/llama.cpp/blob/ff5a23376725e5a889fa0bfb987b7541dc304da0/ggml/src/ggml.c#L3413) depends on `n_ff`, `n_expert_used`, and tokens, not on the number of experts in `src0`. Remapping IDs to a compact staged tensor is therefore shape-preserving. The gate proved bitwise equality; releasing the original model mapping is not implemented. |
| How are expert bytes laid out? | Expert-major 3-D GGUF tensor: `[n_embd, n_ff, n_expert]`; an expert payload is `n_embd * n_ff * stored_element_bytes` with a checked stride. The gate validated this layout for F32. Real quantized extents, scales, and fused projections remain unvalidated for the selected artifact. |
| When may the backing be reused? | CPU kernels are synchronous; the gate releases the lease after `ggml_graph_compute_with_ctx` returns. An asynchronous backend would need a completion fence. Cancellation before and during leases is tested. |
| What is the smallest live compute unit? | Decode: one expert union of `top_k` (4 in the fixture). Prefill: the union over the token microbatch (17 in the fixture). The gate refused a 17-expert unit with 16 slots instead of widening the arena. The real model's decode/prefill unions are still unmeasured. |
| Are all backend allocations accounted for? | No. The store covers arena, slot metadata, and application-read bytes. llama.cpp owns the 505.53 MiB CPU compute buffer at context 512/batch 512, KV/recurrent state, and mmap-backed model pages. These are not project-owned reservations and no hard engine cap is claimed. |

The existing `llama_moe_cache` is useful prior art for slot maps and LRU, but its
source is host RAM and its destination is a device buffer. It is not a storage
residency mechanism and was not used as the gate's implementation.

## Executed expert-remap gate

The fixture writes 32 deterministic F32 experts of `64 x 128` values. The reference
keeps all 1,048,576 expert bytes resident in one ggml tensor. The candidate creates a
17-slot arena of 557,056 bytes, acquires the selected union, points a compact ggml
tensor directly at the arena (no staging copy), remaps IDs to slots, and runs the same
`GGML_OP_MUL_MAT_ID` kernel.

| Round | Union | Result | Mismatched floats | Host ms (ref/cand) | Phone ms (ref/cand) |
| --- | ---: | --- | ---: | --- | --- |
| Decode, 1 token | 4 | bitwise identical | 0 | 2.067 / 0.123 | 9.184 / 1.624 |
| Prefill, 8 tokens | 17 | bitwise identical | 0 | 0.167 / 0.144 | 2.271 / 1.568 |

Both runs also returned `ResourceLimit` when the same 17-expert union was offered a
16-slot store. The store reported 4 reads/131,072 bytes for decode and 19 cumulative
reads/622,592 bytes through prefill, including 2 hits and 2 evictions. Store-measured
application read time was 0.221 ms / 0.883 ms for host decode/prefill and 0.098 ms /
0.413 ms on the phone. These are page-cached reads, not physical flash reads, and the
fixture timings are not a speed claim.

The gate proves the numerical and lifetime boundary, not a llama.cpp runtime
integration. A runtime patch still has to intercept the graph's expert tensors,
materialize the union before the compute unit, remap IDs, and restore the original
graph state.

## MNN source answers

MNN exposes selected expert IDs in
[`MoEModule::onForward`](https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/express/module/MoEModule.cpp#L18),
including separate decode and prefill paths. Its
[`create`](https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/express/module/MoEModule.cpp#L117)
constructs every expert module up front:

```cpp
for (int i = 0; i < numExperts; ++i) {
    auto& expertG = subGraph.find(expertName)->second;
    module->mExperts.push_back(expertG.m);
}
```

`mExperts` is retained for the module lifetime. The inspected code has no lazy
construction, storage redirection, or release path for an individual expert. Seeing
routing is therefore not a bounded backing lifetime. MNN fails this project's phase 1
lifetime gate without a substantial patch, and no MNN conversion of the selected
artifacts was built or validated. It is not selected.

## Real model baseline

Artifact: `unsloth/Qwen3.5-2B-GGUF`, file `Qwen3.5-2B-Q4_K_M.gguf`, 1,280,835,840
bytes, SHA-256 `aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223`.
GGUF reports qwen35, 1,881,825,088 parameters, 1,269,873,920 tensor bytes, 24 layers,
`n_embd` 2048. This is a derived GGUF of `Qwen/Qwen3.5-2B`; the base safetensors
artifact was not converted by this project.

Settings: context 512, batch 512, 4 threads, greedy sampling (temperature 0), seed
1234, one user message through the model chat template, 16 generated tokens.

| Quantity | Host (x86-64) | Phone (vivo 1920, SM6125) |
| --- | ---: | ---: |
| Prompt tokens | 15 | 15 (same IDs) |
| Load seconds | 3.504 | 1.631 / 1.58-1.63 |
| TTFT seconds | 0.422 | 1.9998, 2.037, 2.032, 2.004 |
| Prefill seconds | 0.382 | 1.996-2.04 |
| Decode tokens/s | 9.69 (second run 10.78) | 4.50, 4.39, 4.37, 4.36 |
| Peak VmHWM | 1,818,592 KiB | 1,370,064 / 1,369,928 / 1,370,132 / 1,370,072 KiB |
| System MemAvailable after load | 1,515,360 KiB | 5,482,980 KiB |

The phone runs produced identical token IDs across all four trials. The host and
phone match the 15 prompt IDs and the first generated token, then diverge. For
example, the host continues `271, 248069, ...` while the phone continues
`198, 90700, ...`. This is a cross-architecture floating-point difference, not a
silently changed model; no cross-architecture token or logit parity is claimed.
Greedy sampling turns a small logit difference into a different token, so the
contract's exact-comparison rule was met as a **failure report**, not relaxed.

Memory is reported as process RSS/VmHWM and system availability only.
`project_owned_bytes` is null because llama.cpp allocations are not instrumented.
The phone's CPU thermal zones rose from about 30.3 C before a run to 44.5-45.8 C
afterward; the `skin-therm-adc` zone reports an implausible raw value around 107.8 C
and is not interpreted. Power-supply nodes were unreadable from the ADB shell.

## Consequences and next unit

- llama.cpp is provisionally selected because it supports qwen35 in source, builds
  for the target ABI, runs the exact artifact on the phone, and exposes a
  shape-preserving expert-ID boundary that passed a bitwise gate.
- MNN's eager expert construction fails the bounded-lifetime gate for this project.
- The next implementation unit is a small llama.cpp runtime patch that materializes
  only the selected expert union into the project arena before `MUL_MAT_ID`, remaps
  IDs, and proves parity and bounded live bytes on the real 2B artifact. Until then,
  the engine has a verified baseline and a kernel-boundary proof, not integrated
  expert streaming.
- Allocation coverage, per-layer expert extents from the real GGUF, quantized staging,
  prefill microbatch limits, and eviction policy under a real model remain open.
## Runtime materializer prototype (2026-10-19)

[`patches/0001-ggml-moe-materializer.patch`](../patches/0001-ggml-moe-materializer.patch)
adds an optional hook at the start of `ggml_compute_forward_mul_mat_id`. The project
hook receives the original expert tensor and selected-ID tensor, acquires the union
from a fixed arena, fills a compact staged tensor with the same type and row strides,
remaps IDs to arena slots, and returns the descriptors. All worker threads use them
after a barrier. The tiled fast path is skipped when staging is active because it
reads `dst->src[0]` directly. The patch does not change routing, arithmetic, or
accumulation.

The [runtime gate](../evidence/runtime/moe-stream-gate-host.json) runs a real
`GGML_OP_MUL_MAT_ID` graph twice: once with the complete resident expert tensor and
once with the materializer installed. The selected union is 4 of 8 experts; 4 reads
(8,192 bytes) fill an 8,192-byte arena. Outputs are bitwise identical on host and
phone, with zero mismatched floats. The [dense regression](../evidence/runtime/dense-regression-post-patch.json)
reproduces the pre-patch Qwen3.5-2B token IDs exactly.

Limits: the materializer and staged descriptors are process-global and assume one
active generation. Capacity or I/O failure aborts instead of falling back. A real MoE
run, prefill microbatch proof, cancellation cleanup, and allocation coverage are still
required before describing the engine as a bounded streaming implementation.
## Real MoE storage-backed run (2026-10-19)

This is the first exact storage-backed expert decode on the physical phone. It uses
the [runtime materializer](../patches/0001-ggml-moe-materializer.patch) and the
[`no MAP_POPULATE` profile](../patches/0002-llama-no-eager-mmap-prefetch.patch).
The artifact is a smaller real MoE substituted for the catalog's 26B/35B targets, as
the phase 1/4 plan permits:

- `gdax/Qwen1.5-MoE-A2.7B_gguf`, `Qwen1.5-MoE-A2.7B_q4_k_m.gguf`, 9,490,338,464
  bytes, SHA-256 `09555491f85f40208e3999f3ced97b064f7c7c0b88bb3fedcbd5df51aad7d1ed`.
- qwen2moe, 24 layers, 60 experts, top-4 routing, 14.3B parameters, 9,484,384,256
  tensor bytes. 8,304,721,920 of those are routed expert bytes.
- Prompt `The capital of France is` (5 tokens), 8 greedy tokens, context 512, batch
  512, 4 threads, seed 1234.

Seven measured runs are recorded in [the aggregate](../evidence/runtime/qwen15-stream-aggregate.json).
A later run with the final conditional-prefetch patch reproduced the same tokens and a
1,254,620 KiB peak but was slower; it is retained separately because the page-cache and
thermal state were not controlled.
The first three used the original eager-mmap-prefetch loader; the last four used
`init_mappings(false)`. All seven produced the **same** token IDs:
`32671, 624, 59604, 198, 39572, 198, 94409, 198`.

| Condition | Runs | TTFT s | Decode tok/s | Peak VmHWM KiB | Arena bytes |
| --- | ---: | --- | --- | ---: | ---: |
| mmap prefetch | 3 | 8.39 / 4.88 / 6.35 | 0.85 / 1.10 / 0.97 | 5,245,404 / 6,032,660 / 6,085,276 | 400,097,280 / 160,038,912 |
| no prefetch | 4 | 2.54 / 2.10 / 2.10 / 2.10 | 1.96 / 2.00 / 2.01 / 2.01 | 1,254,524 / 1,254,636 / 1,254,452 / 1,254,740 | 160,038,912 / 120,029,184 |

In every run: 576 materialize calls (24 layers x 3 projections x 8 decode steps plus
prefill), 3,108 application reads, 5,963,972,608 application-read bytes, zero cache
hits, and a maximum expert union of 18. The arena was 160 MB for 24 slots and 120 MB
for 18 slots. Process peak RSS fell to about 1.20 GiB after prefetch was disabled;
the compute buffer was 317.27 MiB. CPU thermal zones rose from about 35.8 C to 49.3 C
during a measured run.

What this establishes:

- Selected expert planes can be replaced by arena-resident copies at the real CPU
  `MUL_MAT_ID` boundary without changing routing, arithmetic, or accumulation order.
  The synthetic runtime gate is bitwise identical on host and phone, and seven real
  runs with different arena sizes are token-identical.
- A 9.48 GB model can complete a short generation with a 120-160 MB project arena and
  about 1.2 GiB peak process RSS on the phone.
- Eager `MAP_POPULATE` of the 9.5 GB mapping was a dominant cost: disabling it cut
  load from 17.9 s to 0.85 s, TTFT from 4.9-8.4 s to about 2.1 s, and peak RSS from
  5.2-6.1 GiB to about 1.2 GiB. Decode improved from about 1.0 to 2.0 tok/s.

What this does **not** establish:

- No full-resident reference logits or tokens were obtained for this model; the
  parity evidence is the bitwise kernel/runtime gate, not a same-artifact reference.
- The process peak is not a hard cap. The arena is project-owned, but mmap pages, OS
  page cache, the backend compute buffer, shared experts, and allocator behavior are
  not fully covered.
- The LRU cache did not help: each tensor switch recreated the store, so hits were
  zero and 5.96 GB was read for 13 prompt/generated tokens. A shared per-tensor cache
  over one arena is the next unit.
- The target 26B/35B artifacts have not run. Their preflight estimates are not bounded
  process memory.
- The no-prefetch profile is a streaming tradeoff, not a general default. On the
  dense Qwen3.5-2B, `0002` kept tokens identical but moved load from 3.50 to 5.26 s,
  TTFT from 0.42 to 7.01 s, and decode from 9.69 to 0.50 tok/s while peak RSS fell
  from 1.82 to 1.31 GiB. Apply it only when expert pages should stay unfaulted.
- One prompt, one seed, no cancellation or failure injection, and no long-context or
  prefill-union stress test were run.