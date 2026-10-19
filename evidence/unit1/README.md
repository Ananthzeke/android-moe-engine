# Unit 1 evidence

- [Integration matrix](../hardening/gate1-integration.json): 1/2/4-worker success,
  read failure, unexpected EOF, cancellation, capacity, fresh replay, and broken
  adapter, in Debug/Release/ASan+UBSan.
- [35B regression](qwen36-unit1-regression.json): tokens `[11751, 11]`, prefill unit 8.
- [Read failure](unit1-read-failure.json): failed status, specific tensor error, no output.
- [Fresh after failure](unit1-fresh-after-failure.json): token 32671.
- [Decode cancellation](unit1-cancel-decode-late.json): cancelled, two partial tokens.
- [63](prefill-63.json), [64](prefill-64.json), and [65](prefill-65.json) input tokens
  on the 14.3B streaming model.

These records do not establish changed-microbatch parity or 35B 63/64/65 execution.
