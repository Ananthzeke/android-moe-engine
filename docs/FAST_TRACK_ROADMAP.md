# Deliver a reliable CPU engine in verifiable units

Plan date: 2026-10-19. Starting revision: `d5961db`, branch `codex/hardening`.

Use [the research review](GEMINI_RESEARCH_REVIEW.md), [release readiness](RELEASE_READINESS.md), and [the hardening prompt](../prompts/DEEPSEEK_HARDENING.md) together. This roadmap selects the critical path. It does not replace the architecture or relax the validation contract.

## Choose the first delivery

Deliver one CPU text generation at a time, supervised with a timeout and a fresh process after failure. Keep the small-model path usable. Promote a specific streamed MoE artifact only after its own correctness and workload tests pass. The 35B path can remain experimental while a smaller validated path is usable.

Do not spend the next implementation cycle on a UI, HTTP server, same-process reset, another backend, a cache rewrite, or speculative prefetch. Avoid calendar promises until the first failure and parity units pass. Work units below end in observable evidence, not documentation alone.

## Unit 1. Return safe failures and bound prefill

Primary files: `patches/0001-ggml-moe-materializer.patch`, `src/backend/moe_stream.cpp`, `bench/llama_baseline.cpp`, and the inference gates.

1. Introduce a materializer outcome that distinguishes resident pass-through, staged success, cancellation, and error. Preserve the specific error cause.
2. Publish the outcome at the existing barrier. Ensure every worker follows the same branch and that graph abort prevents later nodes consuming incomplete outputs.
3. Set the public `llama_context_params.n_ubatch` before context construction. Read actual expert count and top-k from validated model capabilities, not a Qwen constant.
4. Bound a compute unit by `min(expert_count, tokens_in_unit * top_k)`. Include slot capacity, the fixed lease limit, checked arithmetic, and backend effective shape constraints. If slots are below the required top-k, refuse before decode.
5. For 256 experts, top-8, and 64 slots, eight tokens is a conservative upper bound. Verify the backend does not round the effective unit above the bound. Keep runtime union checks and return structured refusal if assumptions fail.
6. Require reopen after cancellation or failure. Do not add reset APIs in this unit.

Done means actual multi-worker fault tests terminate under a watchdog, do not abort, do not fall back to resident expert tensors on error, and do not label partial output successful. A fresh request must match its fresh reference. The 63/64/65-token prompt tests must either execute under a supported plan or refuse before affected compute. Changed-microbatch numerical parity remains a separate gate.

### Unit 1 delivered (2026-10-19)

Structured materializer outcomes, graph-aborted failures, conservative prefill, and
the fault matrix are implemented. See [the Unit 1 report](UNIT1_REPORT.md). Gate 3
passes for the tested release scope; gate 4 passes for the conservative plan, with
changed-microbatch parity blocked.

## Unit 2. Bind all storage to a verified artifact

Primary files: `tools/gguf_index.py`, `src/backend/moe_stream.cpp`, `src/residency/file_source.cpp`, and the backend load boundary.

1. Use one authoritative backing identity across indexing, model load, and all size-class stores. Eliminate unchecked lazy path reopens.
2. Compare the manifest to authoritative GGUF tensor metadata, including offsets, complete shapes, block rules, strides, and complete routed-tensor coverage.
3. Implement a trusted full-SHA enrollment baseline and measure verification cost separately from inference. Use an established implementation with portable support. Do not invent crypto intrinsics from a CPU-family claim.
4. State the backing-mutation threat model. A startup hash does not make a writable inode immutable. Before claiming runtime integrity, authenticate the digest table and verify consumed expert bytes before publication. Protect header and resident-load bytes under a proven load contract too.
5. Require verified identity in release mode. Keep any legacy manifest bypass diagnostic-only with explicit unverified telemetry. Do not let a structural checksum set a cryptographic-verification success flag.

Done means wrong-content, wrong-offset, stale-index, missing-tensor, truncation, replacement-between-opens, and mutation tests have explicit outcomes before affected bytes are consumed. A retained valid inode may continue after pathname replacement under a documented policy. It must never mix old and new artifacts. Do not require an invalid universal link-count rule.

### Unit 2 delivered partially (2026-10-19)

The verified artifact path retains one FD, recomputes full SHA-256 against an
enrolled digest, validates the manifest against the GGUF tensor table, binds the
backend loader and all expert stores to that FD, and passes the identity/lifetime
negative tests. Startup hashing costs 393.93–412.67 s on the phone; runtime
per-extent mutation integrity remains blocked. See [the Unit 2 report](UNIT2_REPORT.md).

## Unit 3. Establish quantized correctness cheaply, then completely

First extend the existing kernel and runtime fixtures to block-aligned Q4_K and Q6_K experts. Cover gate, up, and down projection layouts, duplicate selections, ID remapping, and forced cache behavior. Compare identical bytes and settings on the same machine. Run 1, 2, and 4 workers.

Then generate the smallest valid complete hybrid qwen35moe GGUF supported by the pinned loader. Derive every required attention, SSM, recurrent, tokenizer, and tensor field from source. Require finite stable outputs and intentionally diverse routing. Record actual size and cost rather than promising Gemini's fixture dimensions or runtime.

Compare unmodified-backend and streamed execution sequentially, with explicit materializer install and uninstall boundaries. Capture router IDs, projection outputs, final logits, and generated tokens. Compare matched microbatch settings first. Test changed microbatch sizes separately with predeclared numerical rules and divergence records.

Rebuild pinned upstream libraries with sanitizer instrumentation for the sanitizer job. Do not label an adapter-only build full-stack sanitized. Preserve the exact source, patch, and library identities. Run hosted CI when the branch is published through the owner's normal workflow.

In parallel, preflight an unmodified pinned mmap reference on available hardware. More than 20.4 GB of physical RAM is not automatically required by demand paging. The actual pinned loader, packing, prefetch, working set, storage, and timeout still determine feasibility. Attempt a bounded run only after that inspection. Keep an untouched reference source/build, the same artifact, and explicit mmap-mode provenance. Neither a page-backed reference nor QEMU is automatically fast or bitwise equivalent to a different target.

Done means quantized operator parity and full-model fixture parity pass, or each missing capability is precisely blocked. Fixture success does not prove the real 35B artifact correct. Preserve the 35B reference limitation until that artifact is compared.

## Unit 4. Exercise useful requests on the phone

Start with a four-prompt smoke corpus: ordinary instructions, code, multilingual text, and a longer input. Store tokenized lengths and exact runtime settings. Run the small-model path first. After units 1–3 pass, use a verified MoE configuration for longer tests.

Advance from 16 to 128 generated tokens where context permits. Use explicit timeouts and retain timeout results. Test 63/64/65 actual input tokens and context minus-one/at/plus-one admission boundaries. Do not confuse configured context with consumed input.

Add optional PSS and `/proc/self/io` observations, RSS high-water marks, application reads, memory availability, and thermal state. Probe permissions and counter meaning. Report unavailable values with a reason. Use fresh processes for independent trials; a same-process reset is unnecessary for this release contract.

For comparisons, begin with one warm-up and three measured trials per feasible condition. Alternate order. After smoke and boundary checks pass, run a bounded 30-minute sustained campaign with periodic observations. Add controlled memory pressure last, starting small with conservative stop conditions. Do not lock a fixed 1.5 GiB allocation merely because Gemini proposed it.

Done means all supported workloads have complete outcomes, reference evidence, and repeatable commands. Publish actual latency and thermal behavior. Unsupported or timed-out 35B conditions remain experimental; they do not invalidate an independently validated smaller configuration.

## Unit 5. Optimize one measured cause

Keep demand reads as baseline. Measure where time goes after identity checks and safe prefill are included. Do not compare an unchecked old binary against a verified new binary as an isolated I/O optimization.

Try only one intervention per comparison: offset ordering, coalescing adjacent extents, or `POSIX_FADV_RANDOM`. Read ordering must not change routing or accumulation order. Keep planned staging bytes within admission.

Use projection lookahead only after source evidence proves the shared routing and storage lifetime, and after new kernel-cache effects and cancellation behavior are measured. `WILLNEED` is not a queue-depth control. Do not adopt a projected 25–40% improvement as an acceptance criterion.

Accept a change only with parity, complete outcomes, repeatable improvement on representative workloads, and understood memory tradeoffs. Keep negative results. A lower application-byte count alone is not a win.

## Practical acceptance tests

These are tests to implement, not commands or results that already exist.

| Test | Minimal setup | Pass criterion |
| --- | --- | --- |
| Worker error propagation | Existing real ggml gate, 1/2/4 workers, injected acquire failure | Structured error under watchdog; no hang, SIGABRT, fallback, or later consumption. |
| Cancellation stages | Before read, between chunks, prefill, decode | Correct cancelled outcome and drained resources; fresh-process replay correct. |
| Capacity boundary | Forced distinct unions of C−1, C, C+1 experts | At-capacity succeeds; over-capacity refuses safely without expanding budget. |
| Prefill boundary | Exactly 63, 64, 65 input tokens; slots 7/8/9 with top-8 | Supported plan executes or refuses before compute; effective unit respects bound. |
| Offset integrity | Swap compatible in-bounds tensor offsets and recompute structural checksum | Authoritative metadata validation rejects before compute. |
| Content integrity | Same-size bit flip; wrong file with matching untrusted sidecar | Trusted identity rejects before affected bytes reach a consumer. |
| File lifetime | Rename, hard link, lazy-open replacement, truncate, in-place mutation | No mixed artifacts; documented continue/refuse policy; no uncontained SIGBUS. |
| Quantized operator parity | Q4_K and Q6_K fixtures, remapped IDs and multiple projections | Declared same-path numerical comparison passes; gate detects deliberate corruption. |
| Complete hybrid parity | Valid tiny GGUF, matched and then changed microbatches | Finite outputs, declared router/logit/token criteria, full divergence evidence. |
| Practical generation | Four prompts, 16 then 128 output-token limit | Correct complete records, actual lengths, EOS/timeouts distinguished. |
| Sustained requests | Fresh processes over 30 minutes after shorter tests pass | No unexplained failures; bounded resource observations and thermal results retained. |
| I/O policy experiment | One policy change against verified baseline | Parity first; repeated latency and resource evidence, including negative outcomes. |

## Commands that exist now

Run the host foundation check from the repository root:

```sh
.venv/bin/python tools/verify.py
```

Run the existing inference gates with verified dependencies:

```sh
AME_LLAMA_SOURCE_DIR=/media/ananth/17AB9D0651C6CC1F/ame-tools/src/llama.cpp \
AME_LLAMA_HOST_LIB_DIR=/media/ananth/17AB9D0651C6CC1F/ame-tools/build/llama-host \
.venv/bin/python tools/verify_inference.py --configs debug,release,sanitizers
```

The second command rebuilds its integration directories and writes evidence. Keep source and library identities matched. Current upstream libraries exclude sanitizer instrumentation; this command alone does not close that coverage gap. New worker API patches must update source verification and all caller contracts together.

## Start the next agent with a narrow deliverable

Give the implementation agent this request:

> Read `docs/GEMINI_RESEARCH_REVIEW.md` and `docs/FAST_TRACK_ROADMAP.md`, then implement Unit 1 only. Inspect the current pinned worker barriers and public context parameters before editing. Deliver structured multi-worker failure propagation, conservative supported prefill, and actual fault/boundary tests. Require reopen after failure. Preserve routing and copied staging descriptors. Run host and inference checks, update the exact gate status, and commit a reviewable unit. Do not add identity-format changes, prefetch, UI, or same-process reset in this unit. Report any unproven numerical prefill behavior as blocked instead of claiming parity.

After its evidence passes, proceed to Unit 2. Units 2 and 3 can be developed independently in isolated worktrees with agreed interfaces, but never let concurrent agents mutate the same pinned backend checkout.