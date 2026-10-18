# Phase 4 preflight evidence

No large-model inference ran. These records are metadata-first indexes:

- [Qwen3.6-35B-A3B Q4_K_M](qwen36-35b-a3b-q4km-index.json): 20,419,565,568-byte
  artifact, indexed from a 64 MiB prefix. Full-resident execution refused.
- [Qwen3.5-2B Q4_K_M](qwen35-2b-q4km-index.json): 1,280,835,840-byte artifact, dense,
  indexed from a 16 MiB prefix.

The index reads GGUF metadata and tensor information, computes routed and shared
expert byte extents, checks declared file bounds, and applies a conservative
memory preflight. It does not load weights, infer, or reserve memory.

The streaming estimate (2,303,339,008 bytes) is not an admission: backend workspace,
KV/recurrent state, shared-expert residency, page cache, and the measured compute
buffer are not covered. The next experiment is the runtime integration patch.
