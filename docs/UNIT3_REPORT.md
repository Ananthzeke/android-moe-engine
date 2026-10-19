# Unit 3 report: quantized operator parity and remaining reference gaps

Date: 2026-10-19  
Branch: `codex/hardening`  
Base: `0dfd1fc`; backend `ff5a23376725e5a889fa0bfb987b7541dc304da0`

## Result

**Unit 3 partially passed.**

- Quantized operator parity passes for **Q4_K and Q6_K** across gate, up, and down
  layouts, 1/2/4 workers, prefill/decode shapes, duplicate/noncontiguous IDs, cache
  hits/misses/evictions, and deliberate corruption.
- A pristine unpatched pinned backend produces identical output fingerprints to the
  patched backend for the same quantized operator inputs.
- A complete tiny hybrid qwen35moe fixture was **not produced**. Full model
  reference-vs-streamed parity, changed-microbatch parity, and real 35B reference
  parity remain blocked.

Operator parity is not full-model correctness.

## Quantized operator gate

`bench/moe_quantized_gate`:

- Deterministic expert planes are quantized with the pinned backend
  `ggml_quantize_chunk`.
- Reference keeps all experts resident in one quantized tensor.
- Candidate stores the identical bytes in a bounded arena, acquires selected experts,
  backs a compact tensor with the arena, and remaps IDs to slots.
- Exact bitwise float equality is required.
- Rounds cover decode, repeated decode (hits), alternate decode (misses/evictions),
  duplicate IDs, noncontiguous IDs, and prefill.
- Gate, up, and down layouts are exercised. With `embd=512`, `ff=256`, gate/up
  planes are 73,728 bytes (Q4_K) and 107,520 bytes (Q6_K); down has the transposed
  layout.
- `--corrupt` overwrites the selected staged plane and must produce a divergence.
- Tested at 1, 2, and 4 workers.

Host results: 36 checks per sanitizer/configuration (18 normal, 18 corruption),
all passing. Cache statistics for a 4-slot store: 4 hits, 4 misses, 3 evictions.

Phone results at 4 workers and 512/256 dimensions:

- Q4_K: gate, up, down pass; reference and candidate fingerprints equal within each.
- Q6_K: gate, up, down pass; reference and candidate fingerprints equal within each.
- Phone and host fingerprints differ, as expected from different ISA/arithmetic; the
  gate compares reference and candidate on the same machine, not across devices.

## Pristine backend comparison

The pinned commit was extracted into `/tmp/llama-pristine`, built Release without the
patches, and linked with the same gate source. For Q4_K and Q6_K gate/up/down:

- patched and pristine decode reference fingerprints are equal;
- patched and pristine staged candidate fingerprints are equal;
- prefill fingerprints are equal;
- both binaries report pass.

Evidence: [quantized-pristine-vs-patched.json](../evidence/unit3/quantized-pristine-vs-patched.json).

## Complete hybrid fixture: blocked

No valid tiny qwen35moe GGUF was generated. The pinned loader requires, from
`src/models/qwen35moe.cpp` and its hparams:

- expert feed-forward and shared-feed-forward lengths;
- attention RMS epsilon and four RoPE sections;
- SSM convolution kernel, inner size, state size, time-step rank, and group count;
- recurrent-layer flags or a full-attention interval;
- token embedding, output norm/head;
- per-layer attention norms; QKV/out/Q-norm/K-norm for full-attention layers;
- `wqkv`, `wqkv_gate`, `ssm_conv1d`, `ssm_dt`, `ssm_a`, `ssm_beta`,
  `ssm_alpha`, `ssm_norm`, `ssm_out` for gated-delta-net layers;
- routed `ffn_gate_inp`, `ffn_down_exps`, gate/up expert tensors;
- shared-expert gate/up/down and gate-input tensors;
- tokenizer/vocabulary metadata compatible with the qwen35 pre-tokenizer.

There is no fixture generator in the repository, and no small real qwen35moe
artifact was available. This blocks complete reference-vs-streamed parity,
changed-microbatch parity, router/logit comparisons, and teacher-forced generation.
It is recorded as blocked, not passed.

## Hash diagnostic

`bench/hash_diagnostic` separates a bounded file read from resident-buffer SHA-256
and combined read+hash. Host Release, 1 GiB prefix, 64 MiB resident buffer:

| Measurement | Result |
| --- | ---: |
| Sequential read | 498.08 MiB/s (2.06 s/GiB) |
| Resident scalar SHA-256 | 235.99 MiB/s |
| Combined read + SHA-256 | 194.83 MiB/s |
| CPU | AMD Ryzen 5 4600H, `sha_ni` present |
| Hash implementation | portable scalar SHA-256 |

The phone total for the 20.4 GB artifact remains **393.93–412.67 s**, which is
consistent with a slower CPU/storage combination. The diagnostic does not prove the
phone cost is entirely storage-bound or entirely compute-bound. The CPU advertises
`sha_ni`; the implementation does not use it. The phone probe reported `sha2`.
Recommended next optimization: a reviewed runtime-dispatched SHA-NI/ARMv8-SHA2
implementation with a portable fallback, measured with the same diagnostic. Do not
remove hashing or treat mtime as a digest cache in this unit.

## Cheap Unit 1 tests

- Deterministic cancellation between two bounded read chunks was added to
  `FileSource` tests. The first chunk completes, cancellation is requested, the next
  chunk observes it, and the result is `Cancelled`.
- Missing-model-metadata refusal is still not exercised by a real integration test.
  It remains open.

## Evidence

- [Unit 3 hash diagnostic](../evidence/unit3-hash-host.json)
- [Pristine vs patched operator comparison](../evidence/unit3/quantized-pristine-vs-patched.json)
- Phone Q4_K/Q6_K gate/up/down records in `evidence/unit3/qgate-*-512.json`
- [Quantized gate](../bench/moe_quantized_gate.cpp)
- [Integration matrix](../evidence/hardening/gate1-integration.json)

## Remaining blockers

1. Complete qwen35moe fixture and full-model reference/streamed comparison.
2. Changed-microbatch parity.
3. Real 35B reference logits/tokens; identity verification does not provide them.
4. Missing-model-metadata integration test (Unit 1).
5. Compute-failure injection.
6. Phone full-hash startup cost and runtime mutation coverage (Unit 2).
7. Practical phone workload campaign.

No result here promotes the real 35B artifact to reference-verified support.
