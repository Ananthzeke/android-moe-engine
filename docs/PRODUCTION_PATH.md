# Production execution path

Date: 2026-10-19. Scope: contract-admitted harness, tiny sparse fixtures.

## One host command

```sh
build/host-release/ame run \
  --runner build/inference-release/llama_baseline_stream \
  --run-dir runs/qwen35-sparse \
  --timeout 300 \
  -- \
  --model /tmp/qwen35-sparse/qwen35moe-moe.gguf \
  --manifest evidence/unit3/qwen35-sparse.manifest \
  --artifact-sha "$(sha256sum /tmp/qwen35-sparse/qwen35moe-moe.gguf | cut -d' ' -f1)" \
  --stream --prompt a --max-new-tokens 3 \
  --context 128 --threads 4 --batch-size 64 \
  --moe-slots 4 --moe-arena-mib 64 \
  --output runs/qwen35-sparse/generation.json
```

Expected success: supervisor result exit 0; harness JSON `status: ok`;
`prefill_unit_tokens: 2`; `effective_ubatch_tokens: 2`;
tokens `[106, 57, 108]`.

The supervisor does not parse model options; every option after `--` is passed to
`llama_baseline_stream`.

## One phone command

```sh
adb shell '/data/local/tmp/ame-engine/llama_baseline \
  --model /data/local/tmp/ame-engine/qwen35-sparse.gguf \
  --manifest /data/local/tmp/ame-engine/qwen35-sparse.manifest \
  --artifact-sha 29527436f00bb4b3d4990427d26e4b9151c99d5a1768a5ebf7c22da37fa1b0a8 \
  --stream --prompt a --max-new-tokens 3 \
  --context 128 --threads 4 --batch-size 64 \
  --moe-slots 4 --moe-arena-mib 64 \
  --output /data/local/tmp/ame-engine/production.json'
```

Expected: `status ok`, admitted/effective 2, tokens `[106, 57, 108]`.
Failure records are structured harness JSON; supervisor timeout is a supervisor
`result.json` with `status: timeout`.

## Evidence

- Host production verification: `evidence/unit3/production-verification.json`
- Phone qwen35moe sparse: `evidence/unit3/qwen35-sparse-phone-production.json`
- Phone qwen2moe sparse: `evidence/unit3/qwen2-sparse-phone-production.json`
- Phone refusal/cancel/fresh: `evidence/unit3/phone-{refusal,cancel,fresh}.json`
- Identities: `evidence/unit3/phone-production-identities.json`
