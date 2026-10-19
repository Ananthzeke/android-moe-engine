#!/usr/bin/env python3
"""Negative integration test: missing expert-routing metadata must refuse."""
import argparse
import json
import pathlib
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=pathlib.Path, required=True)
    parser.add_argument("--runner", type=pathlib.Path, required=True)
    parser.add_argument("--remove-metadata", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    malformed = args.workdir / "missing-metadata.gguf"
    logits = args.workdir / "must-not-exist.bin"
    refusal = args.workdir / "refusal.json"
    removed = subprocess.run(
        [str(args.remove_metadata), str(args.fixture), str(malformed),
         "qwen35moe.expert_count", "qwen35moe.expert_used_count"],
        text=True, capture_output=True)
    if removed.returncode != 0:
        print(removed.stderr, file=sys.stderr)
        return 1
    run = subprocess.run(
        [str(args.runner), "--model", str(malformed), "--prompt-ids", "1,2",
         "--context", "64", "--threads", "1", "--batch", "8", "--ubatch", "8",
         "--predict", "1", "--output", str(logits), "--refusal-output", str(refusal)],
        text=True, capture_output=True)
    try:
        status = json.loads(refusal.read_text())
    except (OSError, json.JSONDecodeError):
        print("refusal record missing or malformed", file=sys.stderr)
        return 1
    failed = (run.returncode != 0 and status.get("status") == "refused"
              and status.get("stage") == "model_load" and not logits.exists())
    result = {"returncode": run.returncode, "status": status,
              "logits_absent": not logits.exists(), "passed": failed}
    print(json.dumps(result, indent=2))
    return 0 if failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
