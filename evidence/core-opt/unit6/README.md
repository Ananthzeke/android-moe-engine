# Unit 6: top-k overlap and a pinned compute threadpool, phone A/B

Same device, artifact, method and trial structure as [Unit 4](../unit4/README.md): fresh process per trial, a short
request then a 32-token request, alternating arms, one binary (the committed final build, `ame-server` SHA-256
`67213f672c39899efad8c80f679c9e9f388eaa3fb3078e0caff2885bbdda34ee`). Every arm uses the promoted cache and readers:
`--cache layered --eviction lru --layer-slots 32 --arena-mib 2200 --fadvise-random --fadvise-dontneed --readers 4`.

- `o0`: nothing added (the Unit 5 configuration)
- `o1`: `--overlap topk` (the scheduler callback starts a layer's reads from the router's top-k result)
- `o1tp`: `--overlap topk --threadpool big --poll 0` (also one compute threadpool pinned to cpu4-7, no polling)
- `tp`: `--threadpool big --poll 0` without the overlap

cpu4-7 are the cores with the highest `cpuinfo_max_freq` (2,016,000 kHz), found at startup. Raw records: `ab5/`.

## Twenty trials, five per arm, alternating (no failures)

Median [min, max]. Emitted token IDs are **identical in all 20 trials** and equal to the earlier units.

| metric | o0 | o1 | o1tp | tp |
| --- | --- | --- | --- | --- |
| short request, total s | 23.27 [22.95, 23.40] | 23.86 [23.63, 24.38] | **22.05 [21.64, 22.29]** | 23.33 [23.00, 24.79] |
| 32-token request, total s | 50.09 [49.58, 50.58] | 65.07 [64.10, 66.06] | **46.21 [45.81, 46.33]** | 49.64 [49.26, 50.01] |
| 32-token decode s | 29.59 [29.33, 30.00] | 45.03 [44.22, 45.38] | 27.43 [27.19, 27.99] | 29.19 [28.89, 29.40] |
| decode median step s | 0.914 [0.902, 0.925] | 1.449 [1.420, 1.476] | **0.855 [0.837, 0.874]** | 0.900 [0.893, 0.903] |
| decode tokens/s | 1.048 | 0.689 | 1.130 | 1.062 |
| application bytes read, 32-token decode | 3.85 GB | 3.86 GB | 3.87 GB | 3.86 GB |
| process peak RSS (`VmHWM`) | 4.37 GB | 4.37 GB | 4.37 GB | 4.37 GB |

Relative to `o0`: **`o1tp` -7.8% on the 32-token request and -5.2% on the short request** (decode step -6.5%), ranges
disjoint; **`tp` alone -0.9%**, ranges overlap (no measurable effect); **`o1` alone +29.9%** (decode step +58.5%), a
clear regression.

## Where the time goes (first 32-token trial of each arm, from the per-request records)

The overlap works on the read side and breaks the compute side unless compute is pinned. Decode seconds split into
the time the generation thread was blocked waiting for reads (`read_wait_seconds`) and everything else:

| arm | decode s | blocked on reads | everything else (compute, barriers, sampling) |
| --- | ---: | ---: | ---: |
| o0 | 30.00 | 12.05 | 17.95 |
| o1 | 45.13 | **1.04** | **44.09** |
| o1tp | 27.39 | 8.85 | 18.54 |
| tp | 29.32 | 11.98 | 17.34 |

With the overlap and the default thread placement, the reads are almost fully hidden (1.04 s wait) but the compute
share rises from 18 s to 44 s: the reader threads and the compute threads contend for the same cores while both run,
and a callback boundary per layer adds a synchronization point. With the compute threads pinned to the big cores the
compute share stays at 18.5 s and the wait falls from 12.05 s to 8.85 s: the overlap hides about a quarter of the read
wait, because only the work between the router and the expert operators, and the other two roles' reads, can overlap
(next-layer prefetch, which would hide more, was out of scope). The prefetch counters in `ab5/*/health-after.json` show
1,840 reservations started by the callback in one trial (46 graphs x 40 layers), every routed operator finding its
layer already started (5,520 = 3 x 1,840), and no fallbacks.

## Decision

`--overlap topk --threadpool big --poll 0` is the launcher default (`AME_OVERLAP`, `AME_THREADPOOL`, `AME_POLL`); the server
binary's defaults stay off. The overlap must not be used without the pinned threadpool on this phone. A diagnostic follow-up is
recorded at the end of this file.

## Limits

One prompt pair, five trials per arm, charger connected, two instants of frequency per phase. The 7.8% gain is small
next to the Unit 4 and Unit 5 gains and rests on a pinned threadpool that other devices may not need or tolerate; on a
homogeneous-core device `--threadpool big` is ignored. `--poll 0` was chosen a priori and not swept.
