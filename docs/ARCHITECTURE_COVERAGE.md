# Architecture coverage matrix

Date: 2026-10-19. Scope: tiny upstream-generated fixtures, exact tested layouts.

| Fixture | Family/type | Experts / top-k | Float | Q4_K_M | Q6_K | Sparse selection | Same storage core |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen35moe sparse | hybrid recurrent + full attention | 8 / 2, slots 4 | exact | exact | exact | yes | yes |
| qwen2moe | ordinary transformer MoE | 2 / 2 | exact | exact | exact | no (all selected) | yes |
| granite-moe | non-Qwen transformer MoE | 2 / 2 | exact | exact | exact | no (all selected) | yes |

All matched comparisons produced zero mismatched logits and identical tokens. Sparse
qwen35moe used a 4-slot arena, 56 misses, 48 evictions, 56 reads, 33,030,144 read
bytes, 4,718,592 arena bytes, and 12 distinct two-expert selected sets over 32 calls.
Capacity 1 refused before logits for qwen2moe and granite; sparse qwen35moe refused
at union 2 > 1 slot.

A deliberately corrupted staged expert changed 1,024 logits and all generated tokens.

## What is not generalized

Discovery is still the existing manifest/GGUF name-based path (`_exps`, `.weight`,
`blk.*`). There is no source-derived `llama_layer` consumer extractor, no single
contract object consumed by indexing, validation, materialization, and prefill
planning, and no companion-tensor ownership model beyond ignoring scale companions in
the indexer. The matrix demonstrates that the existing materializer already accepts
these three tested layouts; it does not establish architecture-general discovery.

Unsupported: arbitrary expert-axis layouts, variable-size planes within a table,
fused packing variants not present in these fixtures, grouped/attention MoE consumers
not exercised, and any 35B claim.
