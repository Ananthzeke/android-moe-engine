# Unit 1 report: safe failures and bounded prefill

Date: 2026-10-19  
Branch: `codex/hardening`  
Base: `d5961db`; backend `ff5a23376725e5a889fa0bfb987b7541dc304da0`

## Result

**Unit 1 passed for its tested scope.** It is not a release declaration.

- Expected materializer, storage, arena, capacity, metadata, and unindexed-expert
  failures now return structured outcomes. The engine no longer calls
  `std::abort()` from the materializer path.
- The outcome is published before the existing worker barrier, so 1/2/4 workers
  take the same branch. A latched error stops the CPU graph before later nodes
  consume incomplete output. `llama_decode` returns 2 for the aborted graph.
- The conservative prefill plan is active before context construction. For 256
  experts, top-8, and 64 slots, the unit is **8 tokens**.
- Exact 63/64/65-token inputs executed on the 14.3B streaming model. Capacity
  boundary, read failure, EOF, cancellation, and fresh-process replay tests pass.

## Changed

- `patches/0001-ggml-moe-materializer.patch`: explicit
  `GGML_MOE_PASS_THROUGH / STAGED / CANCELLED / FAILED` outcomes; shared outcome at
  the existing barrier; graph abort on latched error.
- `src/backend/moe_stream.cpp/.hpp`: structured failure state, cancellation
  outcome, specific error retention, no abort path, no resident fallback for
  indexed failures.
- `src/runtime/prefill_plan.{hpp,cpp}`: checked conservative union plan.
- `bench/llama_baseline.cpp`: reads model metadata, sets `n_ubatch` before context,
  refuses unsupported configurations, emits failed/cancelled status, and keeps
  partial output out of the success label.
- `bench/moe_stream_gate.cpp`: 1/2/4-worker success and fault gates for read, EOF,
  cancellation, and capacity.
- `tools/verify_inference.py`: runs the fault matrix and a fresh-process replay after
  every failure.
- `src/residency/file_source.*` and `tests/residency_test.cpp`: deterministic EINTR
  retry test.
- `docs/RELEASE_READINESS.md`: gates 3 and 4 updated.

## Tests executed

```sh
.venv/bin/python tools/verify.py
# 10/10 tests passed in Debug, Release, and ASan+UBSan

AME_LLAMA_SOURCE_DIR=/media/ananth/17AB9D0651C6CC1F/ame-tools/src/llama.cpp \
AME_LLAMA_HOST_LIB_DIR=/media/ananth/17AB9D0651C6CC1F/ame-tools/build/llama-host \
.venv/bin/python tools/verify_inference.py --configs debug,release,sanitizers
# stream, kernel, 1/2/4-worker faults, fresh replay, manifest, broken adapter:
# passed in all three configurations
```

Observed failure outcomes:

| Case | Candidate status | Telemetry |
| --- | ---: | --- |
| success, 1/2/4 workers | success | identical output |
| permanent read failure | aborted | `failed: true`, no fallback |
| unexpected EOF | aborted | `failed: true`, no fallback |
| cancellation before materialization | aborted | `cancelled: true` |
| union capacity +1 | aborted | `failed: true` |
| after each failure | success | fresh process replay |

Physical-device checks:

- 35B regression: `[11751, 11]`, prefill unit 8, arena 37,748,736 bytes, no failure.
- 14.3B read failure: `status: failed`, specific tensor error, no output, exit 1.
- 14.3B fresh request after failure: `status: ok`, token 32671.
- 14.3B decode cancellation: `status: cancelled`, two partial tokens, no success
  label.
- 14.3B exact 63/64/65 input tokens: `status: ok`, unit 15.

Evidence paths:

- [integration matrix](../evidence/hardening/gate1-integration.json)
- [35B regression](../evidence/unit1/qwen36-unit1-regression.json)
- [read failure](../evidence/unit1/unit1-read-failure.json)
- [fresh after failure](../evidence/unit1/unit1-fresh-after-failure.json)
- [decode cancellation](../evidence/unit1/unit1-cancel-decode-late.json)
- [63 tokens](../evidence/unit1/prefill-63.json)
- [64 tokens](../evidence/unit1/prefill-64.json)
- [65 tokens](../evidence/unit1/prefill-65.json)

## Remaining blockers

1. **Artifact identity** is still declared, not verified. Gate 2 remains failed.
2. **Changed-microbatch numerical parity** is not established. The conservative
   plan executes, but the 35B 63/64/65-token case was not run on-device.
3. **Quantized MoE fixture and 35B reference logits** remain absent/blocked.
4. **Cancellation exactly between two read chunks** is not deterministically tested.
   The store observes cancellation before/after chunks; a blocked synchronous read is
   not forcibly interrupted.
5. **Compute-failure injection** through the backend is not covered; read, EOF,
   cancellation, and capacity failures are.
6. **Same-process reset** is not offered. Reopen after failure is the contract.
7. **Practical phone workload campaign** remains failed (gate 6).

No statement here implies the engine is battle-tested or has a hard process memory
cap.
