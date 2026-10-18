# Final build report

## Decision

The production backend is pinned llama.cpp at
`ff5a23376725e5a889fa0bfb987b7541dc304da0`. MNN was rejected because it constructed
every expert module eagerly and failed the bounded-lifetime gate. The project keeps
llama.cpp graph execution, tokenization, routing, attention, and kernels, and owns
expert storage, residency, leases, cancellation, telemetry, and lifecycle records.

## Exact artifacts

| Artifact | Bytes | SHA-256 | Result |
| --- | ---: | --- | --- |
| Qwen3.5-2B Q4_K_M (dense) | 1,269,873,920 tensor bytes | `aaf42c8b…699223` | phone baseline, 4.36-4.50 tok/s |
| Qwen1.5-MoE-A2.7B Q4_K_M | 9,490,338,464 | `09555491…d7d1ed` | 7 streaming runs, ~2.0 tok/s |
| Qwen3.6-35B-A3B Q4_K_M | 20,419,565,568 | `671e47e0…5140c7` | 5 streaming runs, 0.37-0.69 tok/s |

The 35B artifact is qwen35moe: 40 layers, 256 experts, top-8, 34,660,610,688
parameters, 20,408,576,512 tensor bytes, 18,119,393,280 routed expert bytes.

## Commands

```sh
.venv/bin/python tools/verify.py                 # 9 tests x Debug/Release/ASan+UBSan
ame probe                                        # read-only device observation
ame compare --left run-a.json --right run-b.json # measurement equality
ame run --runner llama_baseline_stream --run-dir runs/x --timeout 600 -- … 
```

The phone harness is `llama_baseline_stream`; streaming requires the manifest and
`--stream --moe-slots N --moe-arena-mib M`. The two llama.cpp patches are
`patches/0001-ggml-moe-materializer.patch` and
`patches/0002-llama-no-eager-mmap-prefetch.patch`.

## Large-model result

Five runs, contexts 512, 1024, and 2048, batch 64, four threads, greedy, seed 1234:

| Context | Runs | TTFT | Decode | Peak VmHWM | Arena |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 3 | 9.65-17.82 s | 0.37-0.61 tok/s | 2,160,700 KiB | 28-38 MB |
| 1024 | 1 | 10.33 s | 0.60 tok/s | 2,169,872 KiB | 38 MB |
| 2048 | 1 | 8.77 s | 0.69 tok/s | 2,190,576 KiB | 38 MB |

All five runs produced `11751, 11, 264, 3177, 34756, 364, 1141, 8807`
(" Paris, a city renowned for its rich"). Application reads were 9,999 calls and
5,897,650,176 bytes; read time was 16.6-21.4 s of a 22-28 s generation. Store hits
were zero. Peak RSS is not an enforced cap: it excludes full backend allocation
coverage and OS page-cache behavior.

## Correctness evidence

- The CPU `MUL_MAT_ID` kernel gate is bitwise identical between resident and
  arena-backed experts on host and phone.
- The runtime materializer gate is bitwise identical.
- The 35B runs are deterministic across five runs, two arena sizes, and three
  contexts.
- No full-resident 35B reference logits exist: the 20.4 GB artifact does not fit the
  phone or the available host. This is the main open parity limitation.

## Lifecycle evidence

- Decode cancellation: 7 partial tokens, `status: cancelled`, no error, exit 0; a
  fresh run reproduced `11751, 11`; model SHA unchanged.
- Injected 100 MB read failure: abort at `blk.1.ffn_up_exps.weight`, exit 134, no
  JSON; a fresh run reproduced the baseline prefix; model SHA unchanged.
- `ame run` owns `started.json`, `result.json`, logs, timeout enforcement, and
  outcome classification. It supervises the pinned backend runner; it does not parse
  model options.
- `ame compare` reports status/artifact/prompt/output/finish equality and exits
  0/1/3.

## Rejected and negative results

- A 1,024-slot global LRU cache cut application bytes 23.5% but increased read time
  by 5.0 s, reduced decode 18%, and added 539 MiB RSS. Demand reads and the small
  arena remain the default.
- Eager mmap prefetch raised the 14.3B peak to 5.2-6.1 GiB. The conditional
  no-prefetch profile lowered it to about 1.2 GiB, at the cost of slower load/TTFT.
- Gemma 4 26B-A4B was not converted or run; Qwen3.6-35B-A3B was selected first.

## Open blockers and next units

1. Full-resident reference parity needs a host with substantially more than 20.4 GB
   usable memory and the same pinned CPU backend. None is available here.
2. Complete backend allocation coverage is not proven; the process peak is not a hard
   cap.
3. Memory-pressure testing and a controlled cold-cache campaign remain.
4. A better read-latency policy (per-tensor reuse, prefetch, or scheduling) must beat
   the measured demand-read path before replacing it.
