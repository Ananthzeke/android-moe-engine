# Patches

[`0001-ggml-moe-materializer.patch`](0001-ggml-moe-materializer.patch) is a small
ordered patch against llama.cpp `ff5a23376725e5a889fa0bfb987b7541dc304da0`.

It adds `ggml/include/ggml-moe.h`, an optional materializer call in
`ggml_compute_forward_mul_mat_id`. Thread 0 calls the hook before row grouping; after
a barrier all workers use the returned compact expert tensor and remapped ID array.
The tiled fast path is skipped when a materializer staged the tensor because that path
reads `dst->src[0]` directly. The `MUL_MAT_ID` graph workspace is sized to the
materializer's declared slot count so a cache may exceed a tensor's expert count. Weak default getters keep an unpatched build linkable;
the project supplies strong definitions.

Apply it to the pinned checkout:

```sh
cd <llama.cpp>
git apply /path/to/android-moe-engine/patches/0001-ggml-moe-materializer.patch
```

Limitations of this prototype:

- One active generation per process; the staged descriptors and materializer are
  process-global.
- The hook aborts on an I/O or capacity failure rather than silently falling back to
  the full expert table.
- The materializer stages one tensor's selected union at a time. It does not yet
  prove a bounded prefill microbatch, allocation coverage, or a safe post-cancel
  reset.
- It changes no routing, arithmetic, accumulation, or expert selection.
[`0002-llama-no-eager-mmap-prefetch.patch`](0002-llama-no-eager-mmap-prefetch.patch)
changes the model loader's `init_mappings` call to disable prefetch/`MAP_POPULATE`
**when the materializer is active**. Dense runs keep the prefetch behavior. On a
9.48 GB MoE this cut peak process RSS from 5.2-6.1 GiB to about 1.2 GiB and reduced
TTFT from 4.9-8.4 s to about 2.1 s. Apply it after 0001.