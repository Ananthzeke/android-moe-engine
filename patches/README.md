# Patches

[`0001-ggml-moe-materializer.patch`](0001-ggml-moe-materializer.patch) is a small
ordered patch against llama.cpp `ff5a23376725e5a889fa0bfb987b7541dc304da0`.

It adds `ggml/include/ggml-moe.h` and an optional materializer call in
`ggml_compute_forward_mul_mat_id`. Thread 0 calls the hook before row grouping; after
a barrier all workers use the returned compact expert tensor and remapped ID array.
The tiled fast path is skipped when a materializer staged the tensor because that path
reads `dst->src[0]` directly. Weak default getters keep an unpatched build linkable;
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
