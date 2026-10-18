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
