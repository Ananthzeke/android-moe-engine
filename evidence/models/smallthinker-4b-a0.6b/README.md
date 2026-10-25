# SmallThinker-4B-A0.6B-Instruct (`smallthinker`), Q4_K

Wave 1, model 1. Status: **supported for CPU expert streaming at the tested shapes on the host and the vivo 1920, with a caveat for models that fit in RAM** (below).

## Artifact

| Field | Value |
| --- | --- |
| Repository | `Tiiny/SmallThinker-4BA0.6B-Instruct-GGUF`, revision `b6ca58deaddfaa18501040b24cc88294c26f84b8` |
| File | `SmallThinker-4B-A0.6B-Instruct.Q4_K.gguf` (llama.cpp's `Q4_K` is its `Q4_K_M` alias; the file mixes Q4_K and Q6_K tensors) |
| Bytes | 2,630,212,704 |
| SHA-256 | `f9e03828eaebdeef503b907f7bc0fabfd4392bdbd8edc9f5433c8530d7e5a6fb`, equal to the Hub LFS digest, computed again on the host after download and on the phone after transfer |
| License | Apache-2.0 (model card metadata) |
| Backend | llama.cpp `ff5a23376725e5a889fa0bfb987b7541dc304da0` with patches 0001 and 0002 |

## Layout (measured from the tensor table, `index.json`)

32 layers, all routed. 32 experts per layer, top-4. Three separate routed tensors per layer (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`), no shared expert, no routed companion tensors. Gate and up are Q4_K in every layer; down is Q4_K in 16 layers and Q6_K in the other 16, so expert sizes are 1,990,656 and 2,294,784 bytes per layer-expert (two size classes).

| Quantity | Value | Kind |
| --- | --- | --- |
| Routed bytes per decode token, no cache hits | 274,268,160 | derived from the tensor table (top-4 x sum of per-layer expert bytes) |
| Plan estimate | about 255 MB | estimated from `config.json`; the real table is 7.6% larger |
| Expert tensors | 2,194,145,280 bytes (83.6% of the file) | derived |
| Non-expert tensors | 430,115,328 bytes | derived |

## Fixture and host parity

- `fixture-parity.json`. Tiny upstream-generated fixture (`test-llama-archs -a smallthinker`), regenerated with 8 experts and top-2 so a 4-slot pool must evict (`AME_FIXTURE_EXPERTS=8`, `tools/generate_qwen35_fixture.sh`). Float, Q4_K_M and Q6_K each ran under the shipped cache, the layered cache, and the layered cache with 4 readers and `--overlap topk`, against a resident run at the same prefill unit: token IDs and logits identical in every case, with evictions in every streamed run. Capacity below top-k (1 slot) refused with `contract_prefill_refused` under both cache layouts. Command: `tests/arch_fixture_test.py`, wired into `tools/verify_inference.py`.
- `host-parity.json`. The real artifact on the host (`tools/model_parity.py`, prompt "Explain in one sentence why the sky appears blue.", 16 greedy tokens, context 512, 4 threads). Legacy cache with 8 slots, layered cache with 8 slots per layer, the same with 4 readers, and the same with 4 readers, `--overlap topk` and both fadvise flags, each against a resident run at prefill unit 2: **token IDs and logits are bit-identical for all four variants.**
- Matched schedule matters. The same artifact with the shipped 64-slot cache admits a prefill unit of 16 and produces different tokens (`17408, 700, ...`) from a unit of 2 (`3602, 279, ...`); that is the existing open changed-microbatch gate, not a streaming fault, and each streamed run matched the resident run at its own unit.
- The harness's default resident run keeps the CPU extra buffer types (repacked weights), which a streamed run cannot use. On this Q4_K artifact the repacked resident logits differ from the plain ones by up to 1.85 (unit 2) and 22.9 (unit 16), though the tokens agreed at unit 2. The parity reference therefore runs with `--no-extra-bufts`, added to the harness in this unit.

## Phone (vivo 1920, Snapdragon 665, 8 GB, charger connected)

Fresh process per run, one warmup round, then five alternating measured rounds, `tools/phone_harness_ab.py`. Chat template on, greedy, context 512, 4 threads, 16 new tokens, 19 prompt tokens, prefill unit 4 in every arm. Medians [min, max]; `phone-ab-summary.json` holds every run.

| Arm | Wall s | TTFT s | Decode tok/s | Peak RSS | Application read bytes | System `pgpgin` |
| --- | --- | --- | --- | --- | --- | --- |
| resident | 8.65 [8.59, 8.77] | 1.72 [1.68, 1.76] | **8.29** [8.02, 8.30] | 2.65 GB | 0 | 1.93 GB |
| streamed, shipped cache (64 slots, 256 MiB) | 23.4 [23.3, 23.5] | 3.00 [2.96, 3.01] | 4.44 [4.44, 4.48] | 0.49 GB | 7.76 GB | 0.04 MB |
| streamed, layered 16 slots, fadvise | 30.4 [30.2, 30.6] | 5.48 [5.40, 5.76] | 1.92 [1.89, 1.93] | 1.46 GB | 3.51 GB | 1.48 GB |
| streamed, layered 16 slots, 4 readers, overlap | 40.8 [33.2, 47.9] | 4.57 [4.56, 4.72] | 2.10 [2.04, 2.17] | 1.46 GB | 3.51 GB | 3.40 GB |

**Token IDs and the SHA-256 of the captured logits are identical in all 20 measured runs** (one logits hash across the four arms; warmup runs are not summarised). No run failed. Wall time includes the streamed arms' startup artifact hash (16 s, and 28 s in the overlap arm, which is also why its wall range is wide).

What this shows:

- Resident is the right mode for this model. It decodes 1.9x faster than the shipped streamed cache and 4.3x faster than the layered cache, in 2.65 GB of RSS with 5.7 GB still available.
- The layered cache's gain on the 35B comes from replacing flash reads with arena hits. Here the file fits in the page cache, so the shipped cache's 7.76 GB of application reads are mostly served from memory (`pgpgin` of 0.04 MB) while the layered preset, which drops the kernel's copy with `fadvise(DONTNEED)`, goes to flash for every miss (1.48 GB of block reads). The layered cache did halve application reads (3.51 vs 7.76 GB, 56% hits against 1.5%), but those reads are slower. This is evidence for the plan's recommendation to launch resident when the artifact fits, and against using the 35B launcher defaults for a model that fits in RAM.
- The four-reader overlap helped little here (decode 2.10 vs 1.92 tok/s, read-wait 3.5 vs 9.8 s) because decode compute, not the wait, dominates. No threadpool pinning was used in the harness.

## Limits

One prompt, five measured rounds per arm, charger connected, uncontrolled page cache (the file had just been written and hashed on the phone). The streamed arm sizes are the shipped 64 slots and a 16-slot layered pool (1.10 GB); other sizes were not measured. The resident reference is the same patched backend without the materializer and without the CPU extra buffer types, not a pristine upstream build. The 8-slot host runs and the 4-slot fixtures are the eviction-heavy cases; the phone runs use 16 slots. No quality evaluation, long context, or server (HTTP) measurement was run for this model. The artifact is an instruct model; its chat template was applied by the harness.
