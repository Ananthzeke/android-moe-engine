# Validation contract

This document defines future experiments. No inference measurements have been run for this repository.

## Define a comparison before running it

Compare the pinned unmodified backend with the patched engine using the same model artifact, prompt token IDs, chat template, quantization, context, thread count, output length, and sampler settings. A comparison between different conversions or backends is a separate characterization, not an isolated residency speedup.

Declare the primary question and metric first. For initial streaming work, the question is whether reference-correct inference completes inside the declared owned-memory budget. Report latency as a tradeoff. Do not invent a minimum tokens-per-second requirement for the phone.

Use one warm-up followed by three measured trials per variant and condition initially. Alternate order to reduce thermal and background drift. Save every trial and any exclusion with a reason. A larger campaign needs a concrete unresolved question.

## Prove correctness

For unchanged arithmetic and deterministic execution, require exact token IDs and exact router IDs under fixed greedy settings. Compare intermediate expert outputs and logits, not just rendered text. Reset KV, recurrent state, sampler, and RNG for independent requests.

If an established backend path changes floating-point order, declare absolute and relative logit tolerances before looking at candidate results. Set them from the reference's repeatability and arithmetic contract. Record top-1 margins and any token divergence. Do not widen tolerances after failure or claim exact parity from a seed alone.

A tiny fixture checks forced hits, misses, alternating experts, duplicate experts, multiple projections, shared experts, different layouts, and prefill batches. Test zero sizes and valid boundaries at N-1, N, and N+1 for capacity, chunks, quant blocks, and context. Test offsets beyond 4 GiB, multiplication overflow, truncated files, cancellation, and short-read cleanup.

For a real model too large for the phone's resident reference, generate reference logits or teacher-forced token results on an identified larger host with the same quantized artifact. Differences in ISA or arithmetic require explicit treatment. A forced token sequence compares execution costs but is not a free-generation quality test; keep both results distinct.

Validate prefill/decode phase boundaries against the backend's token-position and pending-input conventions. Test supported chunks or shape buckets at their boundaries, including prompt lengths that cross a boundary. Unsupported schedules must refuse before materialization. Token chunking does not prove a bounded MoE expert union; verify both.

Add lifetime fixtures for tied embeddings, shared experts, persistent children, aliased views, and packed copies. Inject cancellation or compute failure before and after a lease reaches the consumer. Prove that all tasks drain before release and that request-local KV/recurrent state is discarded. A next independent request must match a fresh-engine reference, or the API must require reopen. Do not infer cleanup from a successful post-hook alone.

## Record provenance

Each run records the engine SHA and dirty patch identity, backend SHA, converter version, NDK and compiler, flags, binary SHA-256 and byte size, every model shard hash, tokenizer and template hashes, prompt hashes or permitted text, seed, and all runtime options.

Tag execution as `physical_target`, `host`, or `emulator`. Host results never establish phone throughput. Synthetic memory limits or replay traces do not establish end-to-end phone feasibility.

Pin an exact Hugging Face artifact revision when selecting a model. Resolve instruction-tuned variants deliberately. Inspect the produced ELF, linkage, alignment, and instruction requirements before deployment. Hashing and model validation occur outside timed phases unless the metric explicitly includes them.

Also record load strategy, native artifact version, capability/phase-plan identity, stored and materialized representations, and state-allocation policy. A LiteRT-LM external characterization uses its own artifact/build identity; different containers or conversions are not the same-artifact reference/candidate experiment.

## Measure time with clear boundaries

Use a monotonic high-resolution clock. Define these fields in seconds:

| Metric | Boundary |
| --- | --- |
| `load_seconds` | Engine open begins to ready, including allocations and required weight loading. |
| `request_ttft_seconds` | Generate call begins to the first emitted token, including tokenization, prefill, required I/O, and sampling. |
| `startup_to_first_token_seconds` | Open begins to first emitted token. |
| `prefill_seconds` | Backend prompt evaluation, including its storage waits. Record tokenization separately. |
| `decode_seconds` | First emitted token to last emitted token. |
| `decode_tokens_per_second` | `(generated_tokens - 1) / decode_seconds` when at least two tokens are emitted and duration is positive; otherwise null with a reason. |
| `wall_seconds` | Whole run including setup scopes explicitly named in the record. |

Do not divide prompt tokens plus output tokens by decode time. Record EOS and actual output count. Future asynchronous kernels need synchronization at timing boundaries.

Use diagnostic events to distinguish storage read, materialization/decompression, backend packing, execution, and release wait. Do not sum overlapping event durations and label that total wall time. A next-unit prefetch must record which intervals overlap and keep its own byte reservations within coverage.

## Name memory and I/O scopes

Record a covered allocation high-water mark and its categories. Track expert cache payload, packed bytes, staging, index metadata, fixed backend state, KV/recurrent state, workspaces, stacks, alignment, and load-time peaks. List accounting exclusions.

Collect process RSS and PSS independently when permitted. State sampling interval and that sampled PSS is a lower bound on the true peak. Capture process high-water RSS with platform units where available. A new process per trial avoids mistaking a cumulative lifetime peak for one request.

Record per-phase high-water marks for load, prefill, and decode. Include the overlap of old/new state banks, compressed and expanded weights, packing copies, and read staging. Reused aliases share one backing charge; physically distinct copies do not. A small read chunk does not bound a whole-section allocation made before the read loop.

If an accelerator characterization is added later, label DMA-BUF or other partial monitors as subsets. Unsupported data is null with a reason. A sampled allocation subset is neither a total accelerator peak nor proof of an enforced cap.

Record these I/O quantities separately:

- Logical expert bytes requested by the model.
- Successful bytes returned by application read calls.
- Read calls, short reads, cache hits/misses, evictions, and wait time.
- Physical storage-read counters only if accessible, with collection scope and semantics.

Application `pread` bytes can come from the OS page cache and are not physical flash reads. Process I/O counters can include other work. Use null and a reason where a counter is inaccessible. A runtime-owned cache cap does not cap file-backed page-cache residency or process RSS/PSS.

## Control state honestly

A fresh process is `cold_process_os_cache_unknown` unless OS file-cache state is controlled. Rootless Android cannot reliably drop global caches. `warm_engine_cache` means declared warm-up state remains in the same process. Describe whether expert arena, backend tuning cache, model mappings, and OS cache persist.

Record battery state, charging state, temperatures or thermal status where available, elapsed sustained load, and relevant background pressure. Avoid benchmark comparisons that pair one cool run with one throttled run. Do not disable device protections. Distinguish an unavailable temperature reading from a zero reading.

## Persist complete outcomes

Each run directory starts with an atomic `started.json` before heavy allocation. Store raw stdout, stderr, token IDs, and permitted intermediate parity evidence with bounded buffering. Write a final `result.json` via temporary file, flush, and atomic rename. Flush the parent directory when supported.

The planned final record contains these objects:

```text
run: id, timestamp with timezone, variant, condition, repeat index,
     command, status, termination reason, exit code, seed
environment: execution tag, device snapshot, thermal observations
provenance: engine/backend/build/artifact identities and runtime settings
timing: clock, boundaries, phase durations, token counts
memory: owned coverage/categories, high-water RSS, sampled PSS, sampling scope
io: logical bytes, application-read bytes, physical counter or missing reason
outputs: token IDs, logits/reference paths, file hashes
validation: parity criterion/result, allocation checks, unavailable evidence
```

The provenance and memory objects also retain the phase plan, load strategy, accounting exclusions, and materialization identities described above. Artifact-preparation resume tests must reject a stale source/converter/options identity even if a completion marker exists.

A supervisor reconciles a start record without a final record as interrupted or unknown. Use `low_memory_kill` only when OS evidence establishes that cause. A killed process cannot reliably write its own final status. Save timeout and unsupported cases, not fabricated zero-throughput successes.

## Accept or reject a change

Require declared correctness and covered-memory ceilings first. Report median, minimum, maximum, trial count, and individual outcomes for the small campaign. Aggregate from machine-readable records with a script and verify every reported number against those records.

An intervention that saves memory but increases latency can be useful. Name that tradeoff instead of calling it a speedup. If a required counter is missing, narrow the conclusion or mark the experiment incomplete. Preserve negative results and the simpler working path.

## Coding foundation evidence (2026-10-19)

Host checks use GCC 11.4.0, Python 3.10.12, CMake 3.31.6, clang-format 18.1.8, and the `ninja==1.11.1.4` Python package (binary version `1.11.1.git.kitware.jobserver-1`). Commands from the repository root:

```sh
/tmp/ame-dev-tools/bin/python tools/verify.py --format
/tmp/ame-dev-tools/bin/python tools/verify.py
python3 -O tools/check_scaffold.py
python3 tests/check_scaffold_test.py
/tmp/ame-dev-tools/bin/cmake --list-presets=all
git diff --check
```

The full verifier passes formatting, metadata/link validation, and all three CTest cases in each of Debug, Release, and ASan/UBSan. Extent cases cover N-1/N/N+1, empty extents at EOF, offsets above 4 GiB, and UINT64_MAX without allocating large files. Compile-time checks exclude unchecked/default construction. CLI tests check help/version success and invalid arguments' nonzero exit/diagnostic behavior. Temporary configuration tests run the validator under Python `-O`, reject invalid support metadata and duplicate IDs, and allow extra configurations, anchors, and complete evidence references. These tests establish reference validation only, not the truth of a claimed external experiment.

The first sandbox sanitizer attempt failed because LeakSanitizer reported it cannot run under ptrace. The full verifier subsequently passed outside that sandbox with leak detection retained. No sanitizer checks were disabled. Environments using ptrace need an execution context that permits LeakSanitizer thread inspection.

Preset discovery confirms the three host configure presets; the Android configure preset is hidden without `ANDROID_NDK_HOME`. No NDK is installed in this environment, so cross-build, ELF architecture/linkage, physical-device probe, inference, model support, and streaming remain unverified. No weights or backends were downloaded. Phase 0 remains incomplete.
