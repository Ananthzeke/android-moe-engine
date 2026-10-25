# Unit 3: long routing trace and replay

Captured on the phone (vivo 1920, the enrolled Qwen3.6-35B-A3B Q4_K_M artifact) with the Unit 3
harness `llama_baseline_android --stream` (binary SHA-256 in `capture-run.json`), legacy cache
(64 slots, 256 MiB), context 512, 4 threads, greedy, `--chat`. The prompt asks for a long story;
the model started with a thinking block, so the routing is that of free generation, not of
one fixed phrase. **330 generated tokens** after a 34-token prompt (8-token prefill units),
status `ok`, finish reason `limit`. Prefill 73.6 s, decode 742.7 s (0.443 tokens/s).
333,462 expert reads, 196.7 GB of application reads, zero cache hits (the shipped design).
Trace: `routing-35b-330tok.trace` (40,080 rows, 334 graphs: 5 prefill, 329 decode, 40 layers; SHA-256
`25d044c6...f9cc`). Replay: `replay.json`, `replay.md`. Command:

```sh
.venv/bin/python tools/replay_routing.py routing-35b-330tok.trace --json replay.json --markdown replay.md
```

## Decode, 329 graphs (slice hit rate / application bytes read per token)

| budget | slots per layer | store_lru (shipped design, bigger arena) | shared_layer_lru | per_layer_lru | per_layer_lfu |
| --- | ---: | --- | --- | --- | --- |
| 0.57 GB | 8 | 36.3% / 361 MB | 36.3% / 361 MB | 36.3% / 361 MB | 36.3% / 361 MB |
| 1.13 GB | 16 | 47.5% / 297 MB | 47.5% / 297 MB | 50.6% / 280 MB | 51.1% / 277 MB |
| 1.70 GB | 24 | 58.8% / 233 MB | 58.8% / 233 MB | 60.4% / 224 MB | 59.6% / 229 MB |
| 2.26 GB | 32 | 67.0% / 187 MB | 67.0% / 187 MB | 67.5% / 184 MB | 66.9% / 187 MB |

The shipped 64-slot design reads 566 MB per token with no hits. Protect-until-consumed variants
are identical to the plain policies, as expected when gate, up and down share IDs and a slot.

## What the curve says

- **Capacity is the lever, not the policy.** Bytes per token fall from 566 MB to 361, 280, 224 and
  184 MB as the budget grows from 0 to 0.57, 1.13, 1.70 and 2.26 GB. There is no knee. The shipped
  (tensor, expert) LRU already reaches 47.5% at 1.13 GB; per-layer partitioning adds at most
  3.1 points there and 0.5 points at 2.26 GB.
- **LRU vs LFU:** LFU is 0.5 points better at 1.13 GB and 0.7-0.8 points worse at 1.70 and 2.26 GB. These
  differences are within what one trace of one prompt can support, and the halving interval (32) was not
  tuned. Per-layer LRU is chosen as the default policy because it is best or tied at three of the four
  budgets and needs no tuning; LFU stays behind a flag.
- **Size is a memory decision.** The curve does not pick 1.13 over 2.26 GB; the phone's headroom does.
  A/B trials in Unit 4 test 1.13 GB (16 slots per layer) and 2.26 GB (32 per layer).
- **Prefill gets almost nothing from caching** (0.1-16% slice hit rate at every budget) because each
  prefill unit selects about 31 experts per layer. A smaller per-layer pool also shrinks the prefill unit
  (`pool / top-k` tokens: 2 at 16 slots, 4 at 32), which the replay cannot model because it replays
  the captured 8-token units. Unit 4 measures that on the phone.

## Limits

One prompt, one run, a thinking-mode continuation, 5 prefill graphs. Hit rates depend on the workload.
Application bytes are not physical reads. The harness does not record `/proc/self/io`, so this capture
gives routing only, not I/O deltas.
