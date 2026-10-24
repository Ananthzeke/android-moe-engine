"""Exercise the complete runner's batch boundary and the harness's result-write failure."""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from logit_io import read_logits, read_tokens  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--complete-runner", type=Path, required=True)
parser.add_argument("--harness", type=Path, required=True)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--layered-model", type=Path,
                    help="tiny MoE with separate gate, up and down expert tensors")
arguments = parser.parse_args()
model = arguments.model
digest = hashlib.sha256(model.read_bytes()).hexdigest()


def run(command, **kwargs):
    return subprocess.run([str(part) for part in command], capture_output=True, text=True,
                          **kwargs)


with tempfile.TemporaryDirectory() as scratch:
    scratch = Path(scratch)
    manifest = scratch / "model.manifest"
    index = run([sys.executable, ROOT / "tools" / "gguf_index.py", "--input", model,
                 "--file-size", model.stat().st_size, "--manifest-out", manifest,
                 "--artifact-sha256", digest, "--output", scratch / "index.json"])
    if index.returncode != 0:
        raise SystemExit(f"manifest generation failed: {index.stderr[-400:]}")

    def complete(batch, tag, prompt="1,2,3"):
        logits, tokens = scratch / f"{tag}.logits", scratch / f"{tag}.tokens"
        result = run([arguments.complete_runner, "--model", model, "--manifest", manifest,
                      "--sha", digest, "--prompt-ids", prompt, "--context", 64, "--threads", 1,
                      "--batch", batch, "--ubatch", batch, "--predict", 2, "--slots", 8,
                      "--output", logits, "--tokens-output", tokens])
        return result, logits, tokens

    # A prompt longer than the batch must be chunked, not written past the batch arrays.
    whole, whole_logits, whole_tokens = complete(8, "whole")
    chunked, chunked_logits, chunked_tokens = complete(1, "chunked")
    for label, result in (("whole", whole), ("chunked", chunked)):
        if result.returncode != 0 or "AddressSanitizer" in result.stderr:
            raise SystemExit(f"{label} prompt batch failed: {result.returncode} {result.stderr[-800:]}")
    if len(read_logits(chunked_logits)) != len(read_logits(whole_logits)) != 4:
        raise SystemExit("chunked prompt lost logit rows")
    if read_tokens(chunked_tokens) != read_tokens(whole_tokens):
        raise SystemExit("chunked prompt changed generated tokens")

    for flag in ("--batch", "--ubatch", "--context"):
        invalid = run([arguments.complete_runner, "--model", model, "--prompt-ids", "1",
                       "--output", scratch / "invalid.logits", flag, 0])
        if invalid.returncode != 2:
            raise SystemExit(f"{flag} 0 must be rejected with status 2: {invalid.returncode}")

    # The inference result is evidence only if the result file reaches disk.
    missing_output = scratch / "absent-directory" / "result.json"
    written = run([arguments.harness, "--model", model, "--manifest", manifest,
                   "--artifact-sha", digest, "--stream", "--prompt", "a", "--max-new-tokens", 2,
                   "--context", 64, "--threads", 1, "--batch-size", 8, "--moe-slots", 8,
                   "--moe-arena-mib", 64, "--output", missing_output])
    if written.returncode == 0 or missing_output.exists():
        raise SystemExit(f"harness reported success without a result file: {written}")
    control = run([arguments.harness, "--model", model, "--manifest", manifest,
                   "--artifact-sha", digest, "--stream", "--prompt", "a", "--max-new-tokens", 2,
                   "--context", 64, "--threads", 1, "--batch-size", 8, "--moe-slots", 8,
                   "--moe-arena-mib", 64, "--output", scratch / "result.json"])
    if control.returncode != 0 or not (scratch / "result.json").is_file():
        raise SystemExit(f"harness control run failed: {control.returncode} {control.stderr[-800:]}")


if arguments.layered_model is not None:
    # The layered cache must give the same tokens and logits as the shipped cache, and
    # refuse a model whose experts are fused.
    layered_model = arguments.layered_model
    layered_digest = hashlib.sha256(layered_model.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        layered_manifest = scratch / "layered.manifest"
        index = run([sys.executable, ROOT / "tools" / "gguf_index.py", "--input", layered_model,
                     "--file-size", layered_model.stat().st_size, "--manifest-out",
                     layered_manifest, "--artifact-sha256", layered_digest, "--output",
                     scratch / "index.json"])
        if index.returncode != 0:
            raise SystemExit(f"layered manifest failed: {index.stderr[-400:]}")

        def harness(name, *extra):
            output, logits = scratch / f"{name}.json", scratch / f"{name}.logits"
            result = run([arguments.harness, "--model", layered_model, "--manifest",
                          layered_manifest, "--artifact-sha", layered_digest, "--stream",
                          "--prompt", "a", "--max-new-tokens", 6, "--context", 64, "--threads", 2,
                          "--batch-size", 64, "--temperature", 0, "--seed", 1, "--moe-arena-mib",
                          64, "--logits-output", logits, "--output", output, *extra])
            if result.returncode != 0 or not output.is_file():
                raise SystemExit(f"{name} harness failed: {result.returncode} {result.stderr[-600:]}")
            return json.loads(output.read_text()), logits.read_bytes()

        legacy, legacy_logits = harness("legacy", "--moe-slots", 8)
        for name, extra in (("legacy-readers", ["--moe-slots", 8, "--readers", 4]),
                            ("layered", ["--cache", "layered"]),
                            ("layered-readers", ["--cache", "layered", "--readers", 4]),
                            ("layered-overlap", ["--cache", "layered", "--overlap", "topk",
                                                 "--readers", 2]),
                            ("layered-overlap-1", ["--cache", "layered", "--overlap", "topk"]),
                            ("layered-lfu", ["--cache", "layered", "--eviction", "lfu",
                                             "--fadvise-random", "--fadvise-dontneed"])):
            record, logits = harness(name, *extra)
            expected_layout = "layered" if "--cache" in extra else "legacy"
            if record["streaming"]["cache_layout"] != expected_layout:
                raise SystemExit(f"{name}: expected the {expected_layout} cache")
            if record["output"]["token_ids"] != legacy["output"]["token_ids"]:
                raise SystemExit(f"{name}: token IDs differ from the shipped cache")
            if logits != legacy_logits:
                raise SystemExit(f"{name}: logits differ from the shipped cache")
            if "--overlap" in extra:
                # The real scheduler delivered the router's top-k tensors and every routed
                # operator found its layer already started.
                stream = record["streaming"]
                if stream["overlap"] != "topk" or stream["prefetches_started"] == 0 \
                        or stream["prefetches_consumed"] == 0 or stream["prefetch_fallbacks"] != 0:
                    raise SystemExit(f"{name}: the top-k overlap did not run: {stream}")
        # The fused-expert fixture cannot be expressed as three role views: refused at install.
        refused = run([arguments.harness, "--model", model, "--manifest", manifest,
                       "--artifact-sha", digest, "--stream", "--prompt", "a", "--cache", "layered",
                       "--max-new-tokens", 2, "--context", 64, "--moe-arena-mib", 64,
                       "--output", scratch / "fused.json"])
        record = json.loads((scratch / "fused.json").read_text())
        if refused.returncode == 0 or record.get("status") == "ok":
            raise SystemExit("a fused gate_up model was accepted by the layered cache")
