# Unit 2 report: artifact-bound streamed weights

Date: 2026-10-19  
Branch: `codex/hardening`  
Base: `49e69a5`; backend `ff5a23376725e5a889fa0bfb987b7541dc304da0`

## Result

**Unit 2 partially passed.** Startup identity and authoritative layout validation are
implemented and exercised on the real 35B artifact. Runtime mutation/per-extent
verification and the full-hash startup cost remain blockers.

## The identity guarantee

For verified streaming, all of the following must hold:

1. The operator supplies an independently enrolled SHA-256 and byte count. Release
   commands take these from the pinned catalog/artifact record, not from the
   manifest.
2. `Artifact::open` retains one `O_RDONLY|O_CLOEXEC` FD, checks size, computes a
   full SHA-256 over the exact FD, and compares it to the enrollment.
3. The same FD's `/proc/self/fd/<fd>` path is used by the llama.cpp model loader.
   Every expert size-class source is `FileSource::open_fd` from that FD. No verified
   path reopens the artifact by pathname.
4. The parsed GGUF tensor table is authoritative. Every manifest row must match the
   tensor's name, type, complete dimensions, absolute offset, byte extent,
   quantization row arithmetic, and contiguous stride.
5. Every `_exps` tensor in the GGUF table must be present exactly once. Missing,
   duplicate, non-routed, or swapped rows fail before materialization.
6. The manifest's declared SHA must agree with the enrolled/computed SHA.
7. fstat metadata is rechecked before installation and before every materialization.
   This catches truncation and ordinary in-place writes that change metadata.

This is **not** a runtime content guarantee. A write between a recheck and a bounded
read, or one that preserves size/mtime/inode metadata, can still be consumed. There
is no authenticated per-expert digest table and no per-slot byte verification.

## Implemented

- Portable SHA-256 in `src/common/sha256` with standard test vectors.
- `src/backend/artifact`: retained FD, full digest, GGUF table parse, expected
  identity comparison, and fstat recheck.
- `FileSource::open_fd` and verified install overload.
- Harness requires `--artifact-sha` for streaming; missing/invalid identity refuses
  before model load. It loads the backend through the retained FD path.
- Real GGUF validation replaced trusting manifest offsets alone.
- Negative tests for same-size wrong content, matching untrusted sidecar, swapped
  in-bounds offsets, missing/duplicate rows, pathname replacement, rename, hard link,
  truncation, and in-place mutation before install.
- A replacement test rewrote the model pathname with a 17-byte file after identity
  verification; the 14.3B run still produced token `32671` from the retained FD.
- Regenerated a v2 manifest for the 35B artifact and ran it on the phone with the
  enrolled SHA.

## Tests executed

```sh
.venv/bin/python tools/verify.py
# 11/11 tests passed in Debug, Release, and ASan+UBSan

AME_LLAMA_SOURCE_DIR=/media/ananth/17AB9D0651C6CC1F/ame-tools/src/llama.cpp \
AME_LLAMA_HOST_LIB_DIR=/media/ananth/17AB9D0651C6CC1F/ame-tools/build/llama-host \
.venv/bin/python tools/verify_inference.py --configs debug,release,sanitizers
# passed, including the artifact identity tests and the Unit 1 fault matrix
```

The artifact gate reports 11/11 negative/lifetime checks passing in each
configuration. The Unit 1 1/2/4-worker fault matrix remains green.

## Verification overhead

Full SHA-256 measured separately from inference timing:

| System | Artifact | Build | Time |
| --- | --- | --- | ---: |
| Host | 14.3B, 9.49 GB | Release | 71.27 s |
| Host | 35B, 20.42 GB | Release | 120.55 s |
| Phone | 35B, 20.42 GB | Release | **393.93 s** (412.67 s on the prior run) |
| Host | 14.3B, 9.49 GB | Debug | 454.24 s |

The phone overhead is about 6.6–6.9 minutes before the model can load. This is a
practical release blocker; the full hash is correct but not presently usable as a
routine warm-start path. The hash is outside `load_seconds` and is reported as
`artifact_hash_seconds`.

## Evidence

- [35B phone verified run](../evidence/unit2/qwen36-unit2-verified.json): tokens
  `[11751, 11]`, artifact verified, v2 manifest, hash 393.93 s.
- [35B host verified run](../evidence/unit2/host-qwen36-v2.json): tokens
  `[11751, 11]`, hash 120.55 s.
- [Replacement-after-open run](../evidence/unit2/host-replacement-after-artifact.json):
  model path replaced with 17 bytes; token `32671` from the retained FD.
- [35B v2 manifest](../evidence/phase4/qwen36-35b-a3b-q4km-v2.manifest).
- [Artifact gate tests](../tests/artifact_gate_test.py) and
  [fixture generator](../tests/artifact_fixture.py).
- [Integration matrix](../evidence/hardening/gate1-integration.json).

## Remaining blockers

1. **Runtime mutation/per-extent integrity.** Startup hash plus fstat is not
   per-read content authentication. A post-recheck write can be consumed.
2. **Startup cost.** 393.93–412.67 s on the phone for the 35B artifact.
3. **Unit 1 remains partial.** Missing-model-metadata refusal and cancellation
   exactly between read chunks are not closed by this unit.
4. **Compute-failure coverage** remains separate.
5. **35B numerical correctness is not promoted** by identity validation. The
   reference parity gate remains blocked.
6. **Phone campaign** remains failed.

No identity pass makes the 35B output reference-correct, and no hash makes the
writable backing immutable.