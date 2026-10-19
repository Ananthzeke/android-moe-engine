# Unit 3 report: complete tiny qwen35moe parity

Date: 2026-10-19  
Branch: `codex/hardening`  
Backend: `ff5a23376725e5a889fa0bfb987b7541dc304da0`

## Result

**Complete float-model parity passed. Complete quantized-model parity passed at
matched microbatches. Changed microbatches were measured separately and are not
bitwise identical. Missing expert metadata is refused before graph execution.
Real-35B reference parity remains blocked.**

This is synthetic-fixture correctness, not language quality and not real-35B
correctness.

## Fixture generation

The previous blocker was concrete: no generator was wired. Upstream already ships
one in `tests/test-llama-archs.cpp`; it builds every required qwen35moe tensor and
tokenizer metadata through `llama_model_saver`.

Reproducible command:

```sh
AME_LLAMA_SOURCE=/path/to/pinned/llama.cpp \
AME_LLAMA_BUILD=/path/to/build \
tools/generate_qwen35_fixture.sh /tmp/qwen35-fixture 1234 0.1
```

Generated artifact and configuration:

- 8,841,120 bytes, SHA-256 `66db7b7b2bb206ae1b1f479c3f9e048eebc5c26617fe7370d7a8374906f3c006`.
- 73 tensors, 157 metadata entries, architecture `qwen35moe`.
- 2 layers, n_embd 256, n_ff 384, n_ff_exp 384, n_ff_shexp 192.
- 2 routed experts, top-2, shared experts.
- SSM conv kernel 4, inner size 256, state size 128, dt rank 2, group count 2.
- Layer 0 is recurrent Gated DeltaNet; layer 1 is full attention, using the
  loader's default full-attention interval 4.
- Tokenizer model `test`, 128-token vocabulary; the runner uses explicit token IDs.
- Router/attention/SSM/shared-expert/normalization/embedding/output tensors are
  present; the graph executes without operator substitution.
- The fixture contains quantization companion `.scale`/`.input_scale` tensors from
  the saver. Manifest indexing ignores those companions and uses only `.weight`
  expert planes.

The fixture itself is generated and untracked. Metadata, hashes, and type inventory:
[fixture-provenance.json](../evidence/unit3/fixture-provenance.json).

## Float complete-model execution

Pristine backend runner: `/tmp/complete_runner_pristine`, built from a clean
`git archive` of the pinned commit with no project patches.

Streamed runner: `build/inference-release/complete_runner`, linked to the patched
backend and Unit 2 artifact/materializer path.

Matched settings: prompt IDs `1,2,3,4,5`, context 256, batch 8, ubatch 8, greedy
sampling, 2 generated tokens, threads 1/2/4.

| Worker count | Logit rows × width | Mismatched floats | Tokens |
| --- | --- | ---: | --- |
| 1 | 6 × 128 | 0 | `[53, 98]` |
| 2 | 6 × 128 | 0 | `[53, 98]` |
| 4 | 6 × 128 | 0 | `[53, 98]` |

Prefill logits at all five prompt positions and the decode-step logits are included.
The router trace (9 materializer calls) is identical between trace-only resident and
staged execution for the float fixture. Evidence:
[complete-fixture-comparison.json](../evidence/unit3/complete-fixture-comparison.json),
[router-comparison.json](../evidence/unit3/router-comparison.json).

## Quantized complete-model execution

Quantized with the pinned `llama-quantize`:

- Q4_K_M: 1,469,760 bytes, SHA-256
  `a70984d805e29317c6f4c12ae7f4c9a079ac4e286eb8f96407906ea3a512f510`.
  Actual expert types: layer 0 down `q5_0`, gate_up `q4_K`; layer 1 down
  `q8_0`, gate_up `q4_K`. Four tensors required fallback quantization.
- Q6_K: 1,976,192 bytes, SHA-256
  `fc8b63117e5a232503f813e1623a894ad80d18d26a2b3efa84bbe7b1469d66a1`.
  Actual expert types: down `q8_0`; gate_up `q6_K`. Four tensors required
  fallback quantization.

The preset names do not imply every expert tensor has that type; the recorded types
above are the actual GGUF types. Manifest generation and artifact identity use the
Unit 2 path.

Matched settings were the same as the float run. Both quantized artifacts produced
**zero mismatched logits and identical tokens** `[53, 98]` between pristine and
streamed execution. Evidence:
[complete-q4-comparison.json](../evidence/unit3/complete-q4-comparison.json),
[complete-q6-comparison.json](../evidence/unit3/complete-q6-comparison.json).

Trace-only resident versus staged router traces are equal for both quantized models
(5 lines for Q4_K_M, 9 for Q6_K).

## Changed microbatch schedules

Comparison rule: **exact bitwise equality is required only at the same microbatch
schedule.** Changed schedules are reported separately; no tolerance was widened.

Prompt lengths 5, 63, 64, and 65; context 256; ubatch 1, 4, and 8.

- Pristine versus streamed at each **matched** ubatch: zero mismatched floats and
  identical tokens for every length and every ubatch.
- Pristine ubatch 8 versus pristine ubatch 1: 489 mismatched floats at length 5;
  6,197 at 63; 6,295 at 64/65; maximum absolute difference `1.1921e-7`.
- Pristine ubatch 8 versus pristine ubatch 4: exact at lengths 63/64/65; 98
  mismatched floats at length 5.
- Streamed ubatch 1 matches pristine ubatch 1 exactly. Against the ubatch-8
  reference it shows the same rounding-scale differences as pristine ubatch 1.
- Generated tokens remained equal in the changed-schedule comparisons.

This establishes that the backend's own chunking changes floating-point results;
streaming did not introduce a separate difference beyond the chosen schedule.

Evidence: [microbatch-comparison.json](../evidence/unit3/microbatch-comparison.json).

## Missing metadata refusal

Removed `qwen35moe.expert_count` and `qwen35moe.expert_used_count` from the fixture
with [bench/remove_metadata.cpp](../bench/remove_metadata.cpp). The pristine loader
refused before any graph execution:

```text
check_tensor_dims: tensor 'blk.0.ffn_gate_inp.weight' has wrong shape;
expected 256, 0, got 256, 2, 1, 1
```

The runner wrote `{"status":"refused","stage":"model_load",
"reason":"missing_or_invalid_metadata"}` and produced no logits:
[missing-metadata-refusal.json](../evidence/unit3/missing-metadata-refusal.json).

## Telemetry limits

- Router IDs are captured through the trace-only resident hook. Gating values
  (pre-top-k router logits) are not exposed by the unmodified graph and are not
  claimed.
- Routed projection outputs are covered by the quantized operator gate; complete
  model comparison uses final logits and router IDs.
- No tolerance was applied to matched schedules.

## Operator parity and hash diagnostic

The earlier Unit 3 operator work remains valid: Q4_K/Q6_K gate/up/down at 1/2/4
workers, cache hits/misses/evictions, duplicate/noncontiguous IDs, corruption
detection, and pristine-vs-patched fingerprints. Phone Q4_K/Q6_K runs pass.

Hash diagnostic on host Release: read 498.08 MiB/s, resident scalar SHA-256
235.99 MiB/s, combined 194.83 MiB/s. Phone 20.4 GB startup remains 393.93–412.67 s.
The implementation is scalar despite `sha_ni`; a dispatched SHA-NI/ARMv8-SHA2 path
is the measured next optimization. No hashing was removed.

## Phone complete fixture

The complete float/Q4_K_M/Q6_K fixture ran on the phone. Streamed and trace-only
resident logits and router traces were byte-identical within each model; all produced
tokens `[53, 98]`. This is target validation of streaming, not a phone pristine
backend build. Evidence:
[phone-comparison.json](../evidence/unit3/phone-comparison.json).

## Remaining blockers

1. Gating values are not captured; only router IDs.
2. Real 35B reference logits/tokens remain blocked by memory; tiny-fixture success
   does not promote the 35B artifact.
3. Unit 2 runtime per-extent mutation integrity and phone hash cost remain open.
4. Compute-failure injection and the practical phone campaign remain open.
## Reproduction commands

Generate the float fixture and index it:

```sh
AME_LLAMA_SOURCE=/path/to/pinned/llama.cpp \
AME_LLAMA_BUILD=/path/to/build \
tools/generate_qwen35_fixture.sh /tmp/qwen35-fixture 1234 0.1

.venv/bin/python tools/gguf_index.py \
  --input /tmp/qwen35-fixture/qwen35moe-moe.gguf --file-size 8841120 \
  --artifact-sha256 <sha256> \
  --manifest-out evidence/unit3/qwen35-fixture.manifest \
  --output evidence/unit3/qwen35-fixture-index.json
```

Quantize and index:

```sh
llama-quantize /tmp/qwen35-fixture/qwen35moe-moe.gguf \
  /tmp/qwen35-fixture/qwen35moe-q4.gguf Q4_K_M
llama-quantize /tmp/qwen35-fixture/qwen35moe-moe.gguf \
  /tmp/qwen35-fixture/qwen35moe-q6.gguf Q6_K
.venv/bin/python tools/gguf_index.py --input <quantized> --file-size <bytes> \
  --artifact-sha256 <sha256> --manifest-out evidence/unit3/qwen35-<type>.manifest \
  --output evidence/unit3/qwen35-<type>-index.json
```

Pristine reference runner (clean pinned tree, no project patches):

```sh
g++ -std=c++20 -O2 -DNDEBUG bench/complete_runner.cpp \
  -I<pristine>/include -I<pristine>/ggml/include \
  -L<pristine-build>/bin -Wl,-rpath,<pristine-build>/bin \
  -lllama -lggml -lggml-base -lggml-cpu -lpthread -ldl -lm \
  -o /tmp/complete_runner_pristine
```

Comparisons:

```sh
/tmp/complete_runner_pristine --model <fixture> --prompt-ids 1,2,3,4,5 \
  --context 256 --threads 1 --batch 8 --ubatch 8 --predict 2 \
  --output pristine.bin --tokens-output pristine.tokens

build/inference-release/complete_runner --model <fixture> \
  --manifest <manifest> --sha <sha256> --prompt-ids 1,2,3,4,5 \
  --context 256 --threads 1 --batch 8 --ubatch 8 --predict 2 \
  --output streamed.bin --tokens-output streamed.tokens
```

Missing metadata:

```sh
g++ -std=c++20 -O2 -DNDEBUG bench/remove_metadata.cpp \
  -I<ggml>/include -L<ggml-build>/bin -Wl,-rpath,<ggml-build>/bin \
  -lggml -lggml-base -lpthread -ldl -lm -o /tmp/remove_metadata
/tmp/remove_metadata <fixture> /tmp/missing.gguf \
  qwen35moe.expert_count qwen35moe.expert_used_count
/tmp/complete_runner_pristine --model /tmp/missing.gguf --prompt-ids 1,2 \
  --output /tmp/missing.logits --refusal-output <refusal.json>
```
