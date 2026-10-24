"""Exercise the complete runner's batch boundary and the harness's result-write failure."""
import argparse
import hashlib
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
