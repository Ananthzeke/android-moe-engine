# Phase 3 evidence

Three recorded phone trials of the Qwen3.5-2B Q4_K_M baseline at context 512, batch
512, 4 threads, greedy sampling, seed 1234, and 16 new tokens. Each run directory has
`started.json`, `result.json`, logs, the harness measurement, and before/after probe
records.

- [Aggregate Markdown](aggregate.md)
- [Aggregate JSON](aggregate.json)

The aggregate reports only completed runs whose measurement hashes match. It does not
claim a cold OS cache, controlled thermal state, or allocation coverage. Host results
are not included in the phone group.
## `ame compare` checks

- [Equal runs](compare-same.json): 35B run 01 against run 02; status, artifact,
  prompt tokens, output tokens, and finish reason match.
- [Different runs](compare-diff.json): 35B run 01 against the cancelled run; status,
  output tokens, and finish reason differ.

The comparator parses the measurement JSON without executing a model. Exit 0 means
all compared fields match, 1 means a difference, and 3 means a file/JSON failure.
## Supervised run

[`supervised-qwen35`](supervised-qwen35) is a Qwen3.5-2B host run launched through
`ame run`. It has atomic `started.json`/`result.json`, captured logs, and the
backend measurement. Host load average was 7.13, so its 0.36 tok/s is a
lifecycle smoke observation, not a performance result.
