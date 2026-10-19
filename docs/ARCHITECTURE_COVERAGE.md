# Architecture coverage matrix

Date: 2026-10-19. Scope: tiny upstream-generated fixtures, exact tested layouts.

| Fixture | Family/type | Experts / top-k | Float | Q4_K_M | Q6_K | Sparse selection | Same storage core |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen35moe sparse | hybrid recurrent + full attention | 8 / 2, slots 4 | exact | exact | exact | yes | yes |
| qwen2moe sparse | ordinary transformer MoE | 8 / 2, slots 4 | exact | exact | exact | yes | yes |
| qwen2moe | ordinary transformer MoE | 2 / 2 | exact | exact | exact | no (all selected) | yes |
| granite-moe | non-Qwen transformer MoE | 2 / 2 | exact | exact | exact | no (all selected) | yes |

All matched comparisons produced zero mismatched logits and identical tokens. Sparse
qwen35moe used a 4-slot arena, 56 misses, 48 evictions, 56 reads, 33,030,144 read
bytes, 4,718,592 arena bytes, and 12 distinct two-expert selected sets over 32 calls.
Capacity 1 refused before logits for qwen2moe and granite; sparse qwen35moe refused
at union 2 > 1 slot.

A deliberately corrupted staged expert changed 1,024 logits and all generated tokens.

## Consumer contract

A small pinned-backend extraction function now reports the loaded model's per-layer
expert tensor fields: separate gate/up/down, fused gate-up, and shared-expert tensors.
The project binds every routed consumer to the verified manifest after model load:

- duplicate source consumers, missing source consumers, and manifest routed tensors
  without a source consumer fail closed;
- source tensor type, expert cardinality, stride, and byte extent must match the
  manifest;
- companion tensors are enumerated by exact prefix; only `.scale` and
  `.input_scale` are currently accepted and owned as companion metadata;
- the materializer refuses a tensor not present in the bound routed set;
- contract counts are exposed in complete-runner telemetry.

This contract is exercised on qwen35moe, qwen2moe, and granite-moe. The automated
comparison command is `tools/verify_complete.py`.

Still not unified:

- Harness prefill admission now consumes the bound contract's per-consumer expert
  cardinality and source `n_expert_used` bound. `complete_runner` still uses explicit
  matched schedules rather than contract admission.
- The contract is not serialized or shared with the Python indexer; indexing still
  produces the manifest from GGUF names, and the C++ side validates against it.
- No variable-size experts, shards, arbitrary expert-axis layouts, or unsupported
  companion semantics are accepted.
- Only the tested contiguous-plane CPU consumer is supported.