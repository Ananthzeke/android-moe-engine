# Unit 5: reader threads, phone A/B

Same device, artifact, method and trial structure as [Unit 4](../unit4/README.md): fresh process per
trial, a short request then a 32-token request, alternating arms, one binary (the committed Unit 5 build,
`ame-server` SHA-256 `110817154b043f84abb29dbb5f2e998d5f696e574b1a937550f28126e63512e0`). Every arm uses the
promoted cache: `--cache layered --eviction lru --layer-slots 32 --arena-mib 2200 --fadvise-random
--fadvise-dontneed`. Arms differ only by reader flags:

- `r1`: `--readers 1` (reads on the generation thread, as before)
- `r2a`: `--readers 2 --reader-cpus auto`
- `r4a`: `--readers 4 --reader-cpus auto`
- `r4n`: `--readers 4 --reader-cpus none` (pinning test, series 2)

`--reader-cpus auto` pinned the readers to the cores whose `cpuinfo_max_freq` is below the maximum: cpu0-3 here
(1,804,800 kHz against 2,016,000 kHz for cpu4-7), found at startup from the sysfs frequency files, not hard-coded.
Raw records: `ab3/` (series 1) and `ab4/` (series 2).

## Series 1: r1, r2a, r4a, five alternating trials each (15 trials, no failures)

Median [min, max].

| metric | r1 | r2a | r4a |
| --- | --- | --- | --- |
| short request, total s | 42.74 [40.33, 45.14] | 29.89 [28.38, 32.12] | **25.94 [23.46, 26.84]** |
| short prefill s | 39.45 [37.23, 41.89] | 27.22 [25.49, 29.42] | 23.66 [21.07, 24.44] |
| 32-token request, total s | 74.76 [72.67, 78.39] | 56.94 [56.10, 61.35] | **51.23 [50.83, 52.89]** |
| 32-token decode s | 39.52 [38.74, 40.70] | 32.19 [31.91, 35.30] | 29.95 [29.67, 30.73] |
| decode median step s | 1.173 [1.155, 1.211] | 0.9925 [0.9727, 1.050] | **0.916 [0.9131, 0.9539]** |
| decode tokens/s | 0.785 [0.762, 0.800] | 0.963 [0.878, 0.972] | **1.035 [1.009, 1.045]** |
| application bytes read, prefill / 32-token decode | 5.87 / 3.90 GB | 5.83 / 3.90 GB | 5.87 / 3.89 GB |
| process peak RSS (`VmHWM`) | 4.37 GB | 4.37 GB | 4.37 GB |
| `MemAvailable` after the trial | 3.48 GB | 3.37 GB | 3.40 GB |

Emitted token IDs are **identical in all 15 trials** and equal to the Unit 2 and Unit 4 values. The bytes read do not
change with the reader count, as they must not; only the time does.

- **r4a vs r1:** short request -39.3%, 32-token request -31.5%, decode step -21.9%, decode throughput +31.9%,
  prefill -40.0%. **r2a vs r1:** -30.1%, -23.8%, -15.4%, +22.8%. The ranges of the three arms do not overlap on the
  request totals (32-token: 72.7-78.4, 56.1-61.4, 50.8-52.9 s).
- Queue depth above one matters on this storage; two readers get most of the gain, four a further 10%. The reader
  count is capped at four by design, so no larger count was tried.

## Series 2: reader pinning, r4a vs r4n, five alternating trials each (10 trials, no failures)

| metric | r4a (readers pinned to cpu0-3) | r4n (unpinned) |
| --- | --- | --- |
| short request, total s | 23.82 [23.47, 25.30] | 23.95 [23.68, 24.26] |
| 32-token request, total s | 51.12 [50.53, 51.50] | 50.47 [50.27, 51.64] |
| decode median step s | 0.9216 [0.9095, 0.9285] | 0.915 [0.9104, 0.9420] |

No measurable effect: the ranges overlap and the medians differ by 1.3% (32-token request) and 0.5% (short). Pinning is
therefore not used by default. The pinning code stays available behind `--reader-cpus auto`.

## Decision

`--readers 4` is the launcher default (`AME_READERS`, `deploy/phone/start-35b.sh`); the server binary's own default stays
one reader. Memory is unchanged (four reader threads; stacks are outside the arena accounting, as documented). The gain
exceeds the run-to-run spread in every metric and the token IDs are identical.

## Host gates

`tools/verify.py` runs the reader pool, layered and legacy store tests under AddressSanitizer/UBSan and under
ThreadSanitizer (no reports; a deliberate race in a scratch program is reported in this environment). The tests cover
a read failure in one job while others are in flight, cancellation mid-batch, byte equality of every staged slice with
the file for 1, 2 and 4 readers, and a frozen destination after each call returns. See the commit message of the Unit 5
code commit for commands and results.

## Limits

One prompt pair, five trials per arm, charger connected. Storage queue-depth scaling is a property of this phone's UFS
or eMMC part and was not isolated with a standalone benchmark; the result here is the end-to-end effect. `O_DIRECT` was not
tried (it needs its own alignment self-test; see the plan).
