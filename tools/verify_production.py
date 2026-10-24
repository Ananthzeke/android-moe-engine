#!/usr/bin/env python3
"""Execute the production harness, not complete_runner, and compare its result."""
import argparse, hashlib, json, os, pathlib, subprocess, sys

from logit_io import FormatError, count_mismatched_floats, read_logits, read_tokens

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--harness", type=pathlib.Path, required=True)
    p.add_argument("--reference", type=pathlib.Path, required=True)
    p.add_argument("--supervisor", type=pathlib.Path, required=True)
    p.add_argument("--matrix", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--workdir", type=pathlib.Path, default=pathlib.Path("/tmp/ame-production"))
    a = p.parse_args(); a.workdir.mkdir(parents=True, exist_ok=True)
    matrix = json.loads(a.matrix.read_text()); rows = []
    for case in matrix["cases"]:
        model = pathlib.Path(case["model"]); manifest = pathlib.Path(case["manifest"])
        if not model.is_file() or not manifest.is_file():
            raise SystemExit("missing prerequisite: " + case["name"])
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        if digest != case["sha256"]:
            raise SystemExit("fixture digest mismatch: " + case["name"])
        work = a.workdir / case["name"]; work.mkdir(exist_ok=True)
        harness = work / "harness.json"; logits = work / "harness.logits"
        common = ["--model", str(model), "--manifest", str(manifest), "--artifact-sha", digest,
                  "--stream", "--prompt", "a", "--max-new-tokens", "3",
                  "--context", "128", "--threads", "1", "--batch-size", "64",
                  "--moe-slots", str(case["slots"]), "--moe-arena-mib", "64"]
        r = subprocess.run([str(a.harness), *common, "--output", str(harness),
                            "--logits-output", str(logits)], text=True, capture_output=True)
        if r.returncode != 0:
            raise SystemExit(f"harness failed {case['name']}: {r.stderr[-400:]}")
        h = json.loads(harness.read_text())
        if h.get("status") != "ok":
            raise SystemExit(f"harness status {h.get('status')} for {case['name']}")
        admitted = h["options"]["prefill_unit_tokens"]
        effective = h["options"]["effective_ubatch_tokens"]
        if admissible := case.get("admitted"):
            if admitted != admissible or effective != admissible:
                raise SystemExit(f"admission mismatch {case['name']}: {admitted}/{effective}")
        prompt_ids = h["prompt_token_ids"]
        ref_logits = work / "reference.logits"; ref_tokens = work / "reference.tokens"
        ref = subprocess.run([str(a.reference), "--model", str(model),
                              "--prompt-ids", ",".join(str(x) for x in prompt_ids),
                              "--context", "128", "--threads", "1", "--batch", "64",
                              "--ubatch", str(effective), "--predict", "3",
                              "--output", str(ref_logits), "--tokens-output", str(ref_tokens)],
                             text=True, capture_output=True)
        if ref.returncode != 0:
            raise SystemExit(f"reference failed {case['name']}: {ref.stderr[-400:]}")
        try:
            ref_ids = read_tokens(ref_tokens)
            lhs, rhs = read_logits(logits), read_logits(ref_logits)
            # harness rows are prefill-chunk last rows + generated rows.
            ref_rows = rhs[len(prompt_ids)-1:]
            count_mismatched_floats(lhs, ref_rows)
        except FormatError as error:
            raise SystemExit(f"invalid output for {case['name']}: {error}")
        harness_ids = h["output"]["token_ids"]
        if ref_ids != harness_ids:
            raise SystemExit(f"token mismatch {case['name']}: {harness_ids} != {ref_ids}")
        logits_equal = lhs == ref_rows
        if not logits_equal:
            raise SystemExit(f"logit mismatch {case['name']}")
        row = {"name": case["name"], "requested_batch": 64, "admitted": admitted,
               "effective": effective, "tokens": harness_ids,
               "logits_equal": logits_equal, "status": h["status"]}
        # Refusal below top-k.
        refusal = subprocess.run([str(a.harness), *common,
                                  "--moe-slots", "1", "--output", str(work / "refusal.json")],
                                 text=True, capture_output=True)
        refusal_json = json.loads((work / "refusal.json").read_text())
        if refusal_json.get("status") == "ok":
            raise SystemExit(f"capacity refusal did not fail {case['name']}")
        # Invalid artifact identity.
        bad = subprocess.run([str(a.harness), *common,
                              "--artifact-sha", "0" * 64, "--output", str(work / "bad.json")],
                             text=True, capture_output=True)
        bad_json = json.loads((work / "bad.json").read_text())
        if bad_json.get("status") == "ok":
            raise SystemExit(f"bad identity accepted {case['name']}")
        # Deterministic cancellation.
        cancel = subprocess.run([str(a.harness), *common, "--cancel-before-decode",
                                 "--output", str(work / "cancel.json")],
                                text=True, capture_output=True)
        cancel_json = json.loads((work / "cancel.json").read_text())
        if cancel_json.get("status") == "ok":
            raise SystemExit(f"cancellation reported success {case['name']}")
        # Missing routed consumer in the manifest proposal.
        lines = manifest.read_text().splitlines()
        tensor_index = next(i for i, line in enumerate(lines) if line.startswith("tensor "))
        bad_manifest = work / "missing-consumer.manifest"
        bad_manifest.write_text("\n".join(lines[:tensor_index] + lines[tensor_index + 1:]) + "\n")
        missing = subprocess.run([str(a.harness), *common, "--manifest", str(bad_manifest),
                                  "--output", str(work / "missing.json")],
                                 text=True, capture_output=True)
        missing_json = json.loads((work / "missing.json").read_text())
        if missing_json.get("status") == "ok":
            raise SystemExit(f"missing consumer accepted {case['name']}")
        # Fresh replay after cancellation.
        fresh = subprocess.run([str(a.harness), *common, "--output", str(work / "fresh.json")],
                               text=True, capture_output=True)
        fresh_json = json.loads((work / "fresh.json").read_text())
        if fresh_json.get("status") != "ok" or fresh_json["output"]["token_ids"] != harness_ids:
            raise SystemExit(f"fresh replay failed {case['name']}")
        # Injected read failure.
        env = dict(os.environ); env["AME_IO_FAIL_AFTER_BYTES"] = "1"
        failed = subprocess.run([str(a.harness), *common, "--output", str(work / "iofail.json")],
                                text=True, capture_output=True, env=env)
        failed_json = json.loads((work / "iofail.json").read_text())
        if failed_json.get("status") == "ok":
            raise SystemExit(f"read failure reported success {case['name']}")
        row.update({"capacity_refused": True, "bad_identity_refused": True,
                    "missing_consumer_refused": True, "cancelled": True,
                    "read_failure": True, "fresh_replay": True})
        rows.append(row)
    # Supervisor timeout, explicit non-success.
    run_dir = a.workdir / "supervisor"
    supervisor = subprocess.run([str(a.supervisor), "run", "--runner", "/bin/sleep",
                                 "--run-dir", str(run_dir), "--timeout", "1", "--", "3"],
                                text=True, capture_output=True)
    result = json.loads((run_dir / "result.json").read_text())
    if result.get("status") != "timeout" or supervisor.returncode == 0:
        raise SystemExit("supervisor timeout did not produce timeout")
    a.output.write_text(json.dumps({"schema_version": 1, "rows": rows,
                                    "supervisor_timeout": result.get("status")}, indent=2) + "\n")
    print(json.dumps({"rows": rows, "supervisor_timeout": result.get("status")}, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())