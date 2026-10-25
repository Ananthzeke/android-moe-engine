# Unit 4: layered expert cache, phone A/B

Device: vivo 1920 (Snapdragon 665, 8 GB RAM, Android 12), enrolled Qwen3.6-35B-A3B Q4_K_M, charger
connected, context 512, 4 threads, greedy. Server: the committed Unit 4 build
(`ame-server`, SHA-256 `9281f1a5b5fa812c78ccdce9bc8a77a44ea7c05a1274717ad25af2361326c298`). One
binary; the arms differ only by flags. **Each trial is a fresh process** (full 20 GB hash, then one 24-token-prompt
request with 8 new tokens and one 27-token-prompt request with 32 new tokens, in that order), so cache
state is cold at the start of every trial, and trials alternate between arms (`tools/ab_trials.py`).
Page-cache state is uncontrolled (an unprivileged shell cannot drop caches), which is why the
alternation matters. `lmkd`: no kill line mentions the server. zram: `/sys/block/zram0/mm_stat` is not
readable by the shell uid (permission denied), so zram growth is not measured; system `pswpin`/`SwapFree`
are recorded instead. Raw records: `ab1/` and `ab2/` (every trial's requests, health, memory state, the
server's own log), `smoke/` (one trial per arm, run before the main series; not in the tables).

Arms (server arguments after the common ones):

- `base`: `--cache legacy --slots 64 --arena-mib 256` (the shipped design)
- `l16`: `--cache layered --eviction lru --layer-slots 16 --arena-mib 1152 --fadvise-random --fadvise-dontneed` (1.13 GB)
- `l32`: the same with `--layer-slots 32 --arena-mib 2200` (2.26 GB)
- `l32k`: `l32` without `--fadvise-dontneed`, to test whether dropping the kernel's copy matters

## Series 1: base, l16, l32, five alternating trials each (15 trials, no failures)

Median [min, max] over five trials.

| metric | base | l16 | l32 |
| --- | --- | --- | --- |
| short request, total s | 55.36 [53.86, 57.21] | 47.22 [46.14, 49.58] | **40.03 [39.04, 42.35]** |
| short prefill s | 48.79 [47.37, 50.10] | 43.07 [42.21, 45.58] | 37.02 [36.05, 39.32] |
| short decode s (2 steps) | 6.34 [5.93, 6.96] | 3.86 [3.77, 4.02] | 2.86 [2.84, 2.91] |
| 32-token request, total s | 122.3 [117.4, 128.5] | 109.9 [107.3, 111.6] | **73.99 [71.70, 74.70]** |
| 32-token decode s | 60.85 [57.88, 63.81] | 64.46 [62.72, 64.79] | 39.31 [37.47, 40.40] |
| decode median step s | 1.740 [1.632, 1.947] | 2.107 [2.036, 2.131] | **1.186 [1.131, 1.213]** |
| decode tokens/s (steps / decode phase) | 0.510 [0.486, 0.536] | 0.481 [0.479, 0.494] | **0.789 [0.767, 0.827]** |
| prefill block-layer `read_bytes` | 15.8 GB | 6.65 GB | 5.04 GB |
| 32-token decode block-layer `read_bytes` | 9.82 GB | 8.91 GB | 3.87 GB |
| process peak RSS (`VmHWM`) | 2.14 GB | 3.27 GB | 4.37 GB |
| `MemAvailable` after the trial | 5.86 GB | 4.62 GB | 3.60 GB |
| prefill major faults | 474 | 611 | 255 |

Emitted token IDs: **identical in all 15 trials** for both requests (`[9419, 0, 248046]` and the 32-token
list recorded in `../unit2/README.md`).

- **l32 vs base:** short request -27.7%, 32-token request -39.5%, decode step -31.8%, decode throughput +54.8%.
  The ranges do not overlap on any of these metrics (base 32-token total 117.4-128.5 s, l32 71.7-74.7 s), and
  the run-to-run spread inside an arm is under 10%.
- **l16 is mixed:** prefill is 11.7% faster, but decode steps are 21% slower than base. With a 1.13 GB cache the
  kernel's copy of the working set (which `base` benefits from: block-layer reads are 45% below `rchar` in its
  decode) is dropped and the arena alone is too small to replace it. It is not promoted.
- **Prefill reads physical bytes once:** block-layer `read_bytes` in prefill falls from 15.8 GB (2.4x the 6.6 GB
  the engine asked for) to 6.65 GB and 5.04 GB, which matches the application bytes. The 2.4x excess in `base` is
  consistent with readahead around each 576 KiB read; `--fadvise-random` removes it. The cause is not isolated
  from the other flag, but together they remove it.

## Series 2: l32 vs l32k, five alternating trials each (10 trials, no failures)

| metric | l32 (`--fadvise-dontneed`) | l32k (kernel copy kept) |
| --- | --- | --- |
| short request, total s | 40.45 [40.18, 43.08] | 43.13 [40.99, 45.22] |
| 32-token request, total s | 73.88 [72.63, 74.61] | 80.11 [78.85, 81.23] |
| decode median step s | 1.171 [1.126, 1.197] | 1.213 [1.186, 1.214] |
| 32-token decode block-layer `read_bytes` | 3.63 GB | 3.90 GB |

Dropping the kernel's copy is 8.4% faster on the 32-token request and the ranges do not overlap, so the flag
stays on. Token IDs are identical across all 10 trials.

## Decision

`--cache layered --layer-slots 32 --arena-mib 2200 --fadvise-random --fadvise-dontneed` beats the shipped cache by more than
run-to-run variation and keeps the same token IDs, and the memory it adds is the arena the allowance covers: the
project-accounted cache is 2,264,924,160 bytes of arena plus 1,143,360 bytes of bookkeeping (resident-slot table, expert tables, maps, scratch), inside the 2,200
MiB (2,306,867,200 byte) allowance. The phone launcher (`deploy/phone/start-35b.sh`) now defaults to it.
Process peak RSS rises by 2.2 GB to 4.37 GB and `MemAvailable` after the run is 3.5-3.7 GB; no pressure
campaign was run, so behaviour with other large apps resident is not established. The server binary's own
defaults stay `legacy` with a 256 MiB arena, which is safe on any device; `AME_CACHE=legacy` restores the old
launcher behaviour.

## Limits

One prompt pair, 5 trials per arm, charger connected, battery temperature 30-35 C, no thermal trace between
the two instants per phase. The first request of each process is cold and the second is not; both
arms follow the same order. `read_bytes` and `rchar` are different quantities. Real-35B reference-logit
parity is still not established; token IDs equal the shipped cache's and the host's (see
`../host-parity-35b/`).
