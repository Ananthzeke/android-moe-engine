# Review of the Gemini hardening research

Reviewed on 2026-10-19 against local revision `d5961dba8d955b1937a675f196c4c7b4a946c81a` on `codex/hardening`.

The supplied ZIP contains one HTML report and 110 PNG assets. Four large images illustrate prefill, I/O, cache behavior, and worker synchronization. The archive SHA-256 is `91e2b757bec18d796a8c5f21979af6b2f84d09c36313be758802fad5b7f6d693`. Its original path is `/home/ananth/Downloads/Android MoE Engine Hardening Research.zip`.

The report is useful research input. Its proposed commands, APIs, measurements, and acceptance thresholds are not instructions or established facts. Some equations are missing from the exported text, and several citations point to secondary discussions or unrelated revisions. Use the pinned source and executed tests to settle implementation details.

## Decision

Keep the current CPU backend and harden the smallest working path. Adopt the report's emphasis on worker synchronization, conservative prefill, artifact identity, quantized parity, and sustained validation. Do not adopt its predicted timings, performance gains, or automatic gate-passing claims.

The [fast-track roadmap](FAST_TRACK_ROADMAP.md) translates those ideas into reviewable units. It is a plan, not a claim that the engine now passes its release gates.

## What to keep and what to correct

| Research proposal | Assessment | Practical action |
| --- | --- | --- |
| Synchronize a materializer failure across workers before leaving the operator | Useful, but not yet proven by the document | Extend the existing barrier and distinguish failure from resident fallback. Test through actual graph execution. |
| Bound each microbatch by expert capacity | Sound union arithmetic under explicit assumptions | For 256 experts, top-8, and 64 slots, at most eight tokens per compute unit guarantees a union no larger than 64. Verify the backend's effective microbatch and numerical behavior. |
| Single retained FD plus content verification | Useful, but incomplete as specified | Bind the loader, validation, and every size-class store to the same artifact. Authenticate expected digests and validate bytes before use. |
| Generate a tiny complete hybrid MoE fixture | Useful | Build its metadata and tensors from the pinned loader, after cheap Q4_K and Q6_K kernel tests. Measure its size and execution cost. |
| Retry EINTR and wire cancellation into reads | Already partly implemented | Preserve existing behavior. Add fault-stage integration tests rather than rewriting it. |
| Add keys to the 64-slot cache | Existing LRU already has expert keys | Trace reuse distance and size classes before adding a new frequency policy. |
| Use PSS, I/O counters, and thermal observations | Useful with scoped claims | Probe access, report unavailable counters, and separate per-process from system-wide observations. |
| Predict 25–40% less read wait from several combined changes | Unsupported | Establish one baseline and test one intervention at a time after parity and failure gates. |

The relevant current paths are [the materializer](../src/backend/moe_stream.cpp), [the retained file source](../src/residency/file_source.cpp), [the residency store](../src/residency/store.cpp), [the runtime patch](../patches/0001-ggml-moe-materializer.patch), and [the harness](../bench/llama_baseline.cpp).

## Worker and lifecycle corrections

The current hook returns a boolean. False means use the original tensor. It cannot safely mean an I/O failure. Use a distinct failure result and latch the original cause. Every worker must take the same failure branch after the existing materialization barrier. The graph abort callback must observe the error before later graph nodes consume incomplete output.

The pinned `llama_decode` interface returns positive 2 for abort. Gemini's negative-return descriptions are inconsistent with that interface. Inspect the actual status path when implementing it. Do not add a second barrier merely because a diagram shows one.

Keep copied staging descriptors. Do not overwrite backend-owned `tensor->extra` or null original tensor data without proving ownership and all backend consumers. The current adapter already creates separate staged descriptors.

Reopen after a failed or cancelled request is an acceptable first release contract. Same-process state reset is a separate feature requiring proof for attention, recurrent state, sampler, and token positions. The proposed `moe_stream_reset_for_next_request()` does not exist and is not automatically safe.

Reject the proposed universal 50 ms failure bound. Cancellation is observed between bounded reads and operators, but a blocked synchronous read is not forcibly interrupted. Use a watchdog and record actual outcomes. SIGKILL plus memory pressure and a breadcrumb does not establish that lmkd caused the kill. Report the signal and observations with cause unknown unless OS evidence identifies it.

## Artifact identity corrections

The existing `FileSource` retains an FD. The gap is separate opens by installation, the backend loader, and lazily created size-class caches. Unify their backing identity or verify that every handle refers to the authoritative artifact.

A retained FD continues to refer to its original open file when its pathname is replaced. Unlinking that file does not corrupt its bytes. The `st_nlink == 0` rule is not a reliable replacement detector because another hard link may remain. This follows the [Linux open-file semantics](https://man7.org/linux/man-pages/man2/open.2.html).

Metadata snapshots help detect changes but are not cryptographic protection against in-place writes. Validate copied expert bytes against trusted expected digests before publishing a resident slot. Cover header and non-expert bytes too. Checking only expert slices does not protect routing weights or tokenizer data.

Bind any digest table to a trusted enrollment record derived from the expected full-artifact digest. Hashes generated from a wrong file, accompanied by an unverified expected-hash string, do not establish identity. Report structural layout validation, enrollment verification, and verified consumed extents separately. Lazy verification does not verify unread bytes.

A full sequential SHA-256 scan is a useful engineering baseline. Measure its cost before rejecting it. Initial hashing alone does not protect later reads from mutable backing. Closing that stronger contract needs authenticated read-time verification or proven immutable backing, including the backend's resident-load path.

The report's 30,720 truncated 16-byte digests occupy 480 KiB before metadata. Full 32-byte digests occupy 960 KiB. Saving 480 KiB does not automatically justify a new truncated-digest format. The claimed 1.95 seconds for hashing 5.90 GB assumes parallel work that the current thread-zero materializer does not perform. Its serial cost at the report's own unmeasured 680–780 MB/s would be about 7.6–8.7 seconds. Neither rate is a device measurement.

Generic Linux fs-verity is not inherently restricted to privileged callers. Android filesystem, kernel, and SELinux support must be checked on this device rather than inferred from unrelated policy diffs. A disposable-file capability probe may be useful later. See the [kernel fs-verity contract](https://docs.kernel.org/filesystems/fsverity.html). Do not make it an ASAP dependency.

## I/O and measurement corrections

The larger-cache regression is real recorded evidence. Double caching and page reclaim are plausible explanations, not established causes. Global `Cached` is not the residency of this model, and anonymous heap is not inherently unevictable. SwapPss does not identify whether the backing is compressed RAM or flash.

Start with application read counters and accessible `/proc/self/io` deltas. The latter separates logical reads from attributed storage reads, with limitations. Neither is a measurement of NAND-level traffic. System-wide vmstat and diskstats include background work. See [kernel process accounting](https://docs.kernel.org/filesystems/proc.html) and [process I/O fields](https://man7.org/linux/man-pages/man5/proc_pid_io.5.html).

`mincore` is an optional residency snapshot that can become stale immediately. Derive alignment and buffer lengths from the actual page size and extent offset. It is not exact physical-I/O accounting. See [mincore](https://man7.org/linux/man-pages/man2/mincore.2.html).

`POSIX_FADV_WILLNEED` does not guarantee queue depth eight, completed reads, no memory cost, or no competition with other work. `POSIX_FADV_RANDOM` disables readahead on Linux but may improve or worsen this workload. Test separately. Cross-projection lookahead still affects kernel cache residency and requires lifetime and cancellation validation. See [the advisory API contract](https://man7.org/linux/man-pages/man2/posix_fadvise.2.html).

Reject the report's automatic bitwise QEMU parity, 15.7 MiB fixture size, sub-1.5-second fixture runtime, fixed logit tolerances, and guaranteed warm-cache reuse. These are hypotheses. Do not globally enable crypto instructions without a proven capability and fallback policy.

## Checks performed for this review

The standard verifier passed all nine CTest cases in each of Debug, Release, and ASan/UBSan, outside the ptrace sandbox with sanitizer checks retained. This review did not rerun the inference integration matrix or phone inference. The newer integration matrix is recorded in [release readiness](RELEASE_READINESS.md), with uninstrumented upstream libraries and hosted CI still limited.

A temporary host-only file test retained an FD, created a hard link, replaced the original pathname, and observed original bytes through the FD and replacement bytes through the pathname. The old inode still had one link. This disproves universal replacement detection using `st_nlink == 0`; it does not validate the engine's artifact handling.

A small arithmetic check covered 232 slot and microbatch combinations for top-8 routing and confirmed the conservative union bound. It does not prove backend scheduling or recurrent-state parity. Neither check measured performance. Local outputs are in `.audit/gemini-research-probes.json` and are not release evidence for the phone.
