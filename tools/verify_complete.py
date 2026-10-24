#!/usr/bin/env python3
"""Execute pristine-vs-streamed complete-model comparisons from an explicit matrix."""
import argparse, hashlib, json, pathlib, subprocess, sys

from logit_io import FormatError, count_mismatched_floats, read_logits, read_tokens

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--pristine-runner", type=pathlib.Path, required=True)
    parser.add_argument("--streamed-runner", type=pathlib.Path, required=True)
    parser.add_argument("--broken-runner", type=pathlib.Path, default=None)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, default=pathlib.Path("/tmp/ame-complete-verify"))
    args = parser.parse_args()
    matrix = json.loads(args.config.read_text())
    args.workdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in matrix["cases"]:
        model = pathlib.Path(case["model"]); manifest = pathlib.Path(case["manifest"])
        if not model.is_file() or not manifest.is_file():
            raise SystemExit(f"missing prerequisite for {case['name']}")
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        if digest != case["sha256"]:
            raise SystemExit(f"digest mismatch for {case['name']}")
        common = ["--model", str(model), "--prompt-ids", case["prompt_ids"],
                  "--context", str(case["context"]), "--threads", str(case["threads"]),
                  "--batch", str(case["batch"]), "--ubatch", str(case["ubatch"]),
                  "--predict", str(case["predict"])]
        pristine = args.workdir / f"{case['name']}-pristine"
        streamed = args.workdir / f"{case['name']}-streamed"
        p = subprocess.run([str(args.pristine_runner), *common, "--output", str(pristine) + ".logits",
                            "--tokens-output", str(pristine) + ".tokens"], text=True, capture_output=True)
        s = subprocess.run([str(args.streamed_runner), *common, "--manifest", str(manifest),
                            "--sha", digest, "--slots", str(case["slots"]),
                            "--output", str(streamed) + ".logits",
                            "--tokens-output", str(streamed) + ".tokens"], text=True, capture_output=True)
        if p.returncode != 0 or s.returncode != 0:
            raise SystemExit(f"run failed for {case['name']}: {p.stderr[-300:]} {s.stderr[-300:]}")
        try:
            left, right = read_logits(str(pristine) + ".logits"), read_logits(str(streamed) + ".logits")
            mismatched = count_mismatched_floats(left, right)
            pristine_tokens = read_tokens(str(pristine) + ".tokens")
            streamed_tokens = read_tokens(str(streamed) + ".tokens")
        except FormatError as error:
            raise SystemExit(f"invalid output for {case['name']}: {error}")
        if len(streamed_tokens) != case["predict"]:
            raise SystemExit(f"{case['name']}: expected {case['predict']} tokens, "
                             f"got {len(streamed_tokens)}")
        capacity_ok = None
        if case.get("capacity_slots"):
            cap = subprocess.run([str(args.streamed_runner), *common, "--manifest", str(manifest),
                                  "--sha", digest, "--slots", str(case["capacity_slots"]),
                                  "--output", str(args.workdir / f"{case['name']}-capacity.logits")],
                                 text=True, capture_output=True)
            capacity_ok = cap.returncode != 0 and not (args.workdir / f"{case['name']}-capacity.logits").exists()
        cancelled_ok = None
        fresh_ok = None
        if case.get("lifecycle"):
            cancel_out = args.workdir / f"{case['name']}-cancel.logits"
            cancel = subprocess.run([str(args.streamed_runner), *common, "--manifest", str(manifest),
                                     "--sha", digest, "--slots", str(case["slots"]),
                                     "--cancel-before-decode",
                                     "--output", str(cancel_out)], text=True, capture_output=True)
            cancelled_ok = cancel.returncode != 0 and not cancel_out.exists() and "cancelled" in cancel.stdout
            fresh = args.workdir / f"{case['name']}-fresh.logits"
            fresh_tokens = args.workdir / f"{case['name']}-fresh.tokens"
            fresh_run = subprocess.run([str(args.streamed_runner), *common, "--manifest", str(manifest),
                                        "--sha", digest, "--slots", str(case["slots"]),
                                        "--output", str(fresh), "--tokens-output", str(fresh_tokens)],
                                       text=True, capture_output=True)
            if fresh_run.returncode != 0:
                fresh_ok = False
            else:
                try:
                    fresh_ok = (right == read_logits(str(fresh))
                                and read_tokens(str(fresh_tokens)) == streamed_tokens)
                except (FormatError, OSError):
                    fresh_ok = False
        corruption_ok = None
        if case.get("corruption") and args.broken_runner is None:
            raise SystemExit(f"{case['name']} requires --broken-runner for its corruption check")
        if case.get("corruption"):
            broken = args.workdir / f"{case['name']}-broken.logits"
            b = subprocess.run([str(args.broken_runner), *common, "--manifest", str(manifest),
                                "--sha", digest, "--slots", str(case["slots"]),
                                "--output", str(broken)], text=True, capture_output=True)
            if b.returncode != 0:
                raise SystemExit(f"broken run failed for {case['name']}")
            try:
                corruption_ok = count_mismatched_floats(left, read_logits(str(broken))) > 0
            except FormatError as error:
                raise SystemExit(f"invalid broken output for {case['name']}: {error}")
        rows.append({"name": case["name"], "sha256": digest, "mismatched_floats": mismatched,
                     "capacity_refused": capacity_ok, "corruption_detected": corruption_ok,
                     "cancelled": cancelled_ok, "fresh_replay": fresh_ok,
                     "tokens_equal": pristine_tokens == streamed_tokens,
                     "tokens": streamed_tokens,
                     "streamed": json.loads(s.stdout.splitlines()[-1]) if s.stdout.strip() else {}})
    args.output.write_text(json.dumps({"schema_version": 1, "rows": rows}, indent=2) + "\n")
    failed = [r for r in rows if r["mismatched_floats"] != 0 or not r["tokens_equal"]
              or r["capacity_refused"] is False or r["corruption_detected"] is False
              or r["cancelled"] is False or r["fresh_replay"] is False]
    for row in rows:
        print(row["name"], "mismatched", row["mismatched_floats"], "tokens", row["tokens"])
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())