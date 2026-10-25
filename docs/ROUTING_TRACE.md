# Routing trace and replay

`AME_MOE_TRACE=PATH` makes the adapter write one row per routed `MUL_MAT_ID` call. Format 2:

```text
# ame-moe-trace 2
# columns: call request graph phase ubatch tensor n_experts experts...
0 0 1 prefill 8 blk.0.ffn_gate_exps.weight 31 112 106 ...
```

`call` counts calls since install. `request` counts `begin_moe_stream_request()` calls.
`graph` counts `label_moe_graph()` calls within the request; the server and the harness
label each `llama_decode` with phase `prefill` or `decode` and the number of tokens in the
ubatch. Without labels the phase is `unknown` and the ubatch is 0. The listed experts are
the distinct IDs the router selected for that tensor, in first-seen order. Writes are
buffered (1 MiB) and flushed at each graph label, at request start, and at uninstall.
Format 1 (`call tensor n experts...`) is still read; graph boundaries are then inferred from
the layer index falling, and every row has phase `unknown`.

`tools/replay_routing.py TRACE` replays a trace against cache policies at a byte budget:

- `store_lru` reproduces `Store::acquire` exactly, with (tensor, expert) keys and one slice
  per slot: hits are pinned before victims are chosen, free slots are used first, the victim
  is the unpinned slot with the oldest tick, and a lease is released just before the next
  acquire. `tests/replay_routing_test.py` checks it against the real `Store` and
  `LayeredStore` on random scenarios and on the case that distinguishes it from a naive
  replay: capacity two, resident `[A, B]`, acquisition `[C, A]` keeps A, evicts B, records one
  hit.
- `shared_layer_lru`, `per_layer_lru`, `per_layer_lfu` key on (layer, expert): the gate, up
  and down slices of an expert share one slot. LFU halves a pool's counts every
  `--halve-every` acquisitions and breaks ties by recency. The `_protect` variants state that
  a slot loaded for a layer stays pinned until that layer finishes; because the three roles use
  the same IDs, they equal the plain variants.

Counts are per phase. `slice hit rate` is the share of needed slice bytes served from memory,
so policies with different slot sizes compare. `bytes_read` is application bytes, not flash
reads. A replay reproduces decisions already made; it does not change what is computed. It
cannot re-split a prefill into smaller units, so prefill rows are replayed at the captured
ubatch size.
