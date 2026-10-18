# Runtime streaming evidence

These records cover the runtime expert materializer and the first real storage-backed
MoE decode on the physical phone.

## Synthetic runtime gate

- [Host](moe-stream-gate-host.json) and [phone](moe-stream-gate-phone.json):
  `GGML_OP_MUL_MAT_ID` over 8 F32 experts, top-4 union, 4-slot arena. Both are
  bitwise identical, with 4 application reads and 8,192 bytes read.
- [Dense regression](dense-regression-post-patch.json): the patched build reproduces
  the pre-patch Qwen3.5-2B token IDs.

## Qwen1.5-MoE-A2.7B Q4_K_M

Artifact: `gdax/Qwen1.5-MoE-A2.7B_gguf`, file `Qwen1.5-MoE-A2.7B_q4_k_m.gguf`,
9,490,338,464 bytes, SHA-256
`09555491f85f40208e3999f3ced97b064f7c7c0b88bb3fedcbd5df51aad7d1ed`, qwen2moe,
24 layers, 60 experts, top-4 routing, 14.3B parameters.

Seven measured runs used the patched runtime materializer. Runs 01-03 used eager mmap
prefetch; runs 04-07 ran with prefetch disabled. All seven produced the same token
IDs. Run 08 re-ran the final conditional-prefetch patch: the same token IDs and a
1,254,620 KiB peak, but slower TTFT and decode, consistent with an uncontrolled page
cache/thermal state. It is kept separately rather than folded into the campaign
medians. The
[aggregate](qwen15-stream-aggregate.md) records medians, arena sizes, reads, peak RSS,
and the prefetch comparison. Raw harness records and before/after probe snapshots are
kept beside it.

Limits: no full-resident reference parity was obtained for this model; the bitwise
claim is the synthetic gate. Store hits were zero because each tensor switch recreated
the store. The process peak is not a hard cap, and the 35B target was not run here.