#!/usr/bin/env python3
"""Host parity of one real artifact: resident run versus every streamed cache variant.

Runs the harness (`llama_baseline_stream`) greedily on one prompt. Each streamed variant is compared
with a resident run at the same prefill unit: token IDs must be identical, and so must the captured
logits. Resident references run with the CPU extra buffer types off (`--no-extra-bufts`), because
a streamed run cannot use repacked weights; a resident run with them on is also recorded, with its
tokens compared and its logits difference reported, as information about the reference only.

A variant that fails, or that differs, makes the exit status nonzero. When the host cannot hold the
resident run, pass --no-resident: the streamed variants are then compared with each other only, and
the record says so.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_PROMPT = "Explain in one sentence why the sky appears blue."


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def float_difference(left, right):
    import array
    a, b = array.array("f"), array.array("f")
    a.frombytes(left)
    b.frombytes(right)
    if len(a) != len(b):
        return None
    count, largest = 0, 0.0
    for x, y in zip(a, b):
        if x != y:
            count += 1
            largest = max(largest, abs(x - y))
    return {"floats": len(a), "differing": count, "max_abs": largest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-sha", help="skip hashing when already known")
    parser.add_argument("--name", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--legacy-slots", type=int, default=64)
    parser.add_argument("--layer-slots", type=int, default=8)
    parser.add_argument("--arena-mib", type=int, default=2200)
    parser.add_argument("--no-resident", action="store_true",
                        help="the host cannot hold a resident run; compare streamed variants only")
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    sha = arguments.artifact_sha or sha256_file(arguments.model)

    def run(name, extra):
        output = arguments.output_dir / f"{name}.json"
        logits = arguments.output_dir / f"{name}.logits"
        command = [str(arguments.harness), "--model", str(arguments.model), "--artifact-sha", sha,
                   "--prompt", arguments.prompt, "--max-new-tokens", str(arguments.max_new_tokens),
                   "--context", str(arguments.context), "--threads", str(arguments.threads),
                   "--temperature", "0", "--seed", "1", "--logits-output", str(logits),
                   "--output", str(output), *extra]
        begin = time.monotonic()
        result = subprocess.run(command, capture_output=True, text=True)
        elapsed = time.monotonic() - begin
        (arguments.output_dir / f"{name}.stderr").write_text(result.stderr[-4000:])
        if result.returncode != 0 or not output.is_file():
            return {"name": name, "ok": False, "returncode": result.returncode,
                    "stderr_tail": result.stderr[-400:]}, None
        record = json.loads(output.read_text())
        if record.get("status") not in (None, "ok"):
            return {"name": name, "ok": False, "status": record.get("status"),
                    "error": record.get("error")}, None
        return {"name": name, "ok": True, "wall_seconds": round(elapsed, 1),
                "token_ids": record["output"]["token_ids"],
                "prompt_tokens": len(record["prompt_token_ids"]),
                "prefill_unit": record["options"]["prefill_unit_tokens"],
                "streaming": {key: value for key, value in (record.get("streaming") or {}).items()
                              if key != "manifest"}}, logits.read_bytes()

    stream_common = ["--stream", "--manifest", str(arguments.manifest),
                     "--moe-arena-mib", str(arguments.arena_mib)]
    variants = [
        ("legacy", ["--moe-slots", str(arguments.legacy_slots)]),
        ("layered", ["--cache", "layered", "--layer-slots", str(arguments.layer_slots)]),
        ("layered-readers4", ["--cache", "layered", "--layer-slots", str(arguments.layer_slots),
                              "--readers", "4"]),
        ("layered-readers4-overlap",
         ["--cache", "layered", "--layer-slots", str(arguments.layer_slots), "--readers", "4",
          "--overlap", "topk", "--fadvise-random", "--fadvise-dontneed"]),
    ]
    rows, failures = [], 0
    streamed = {}
    for name, extra in variants:
        row, logits = run(name, [*stream_common, *extra])
        print(json.dumps({k: row.get(k) for k in ("name", "ok", "token_ids", "prefill_unit")}),
              flush=True)
        streamed[name] = (row, logits)
        rows.append(row)
        failures += 0 if row["ok"] else 1

    resident = {}
    if not arguments.no_resident:
        for unit in sorted({r["prefill_unit"] for r, _ in streamed.values() if r["ok"]}):
            for label, extra in ((f"resident-u{unit}", ["--batch-size", str(unit), "--no-extra-bufts"]),
                                 (f"resident-extra-u{unit}", ["--batch-size", str(unit)])):
                row, logits = run(label, extra)
                print(json.dumps({k: row.get(k) for k in ("name", "ok", "token_ids")}), flush=True)
                rows.append(row)
                resident[label] = (row, logits)
                failures += 0 if row["ok"] else 1
    comparisons = []
    reference_tokens = None
    for name, (row, logits) in streamed.items():
        if not row["ok"]:
            continue
        unit = row["prefill_unit"]
        if arguments.no_resident:
            if reference_tokens is None:
                reference_tokens, reference_logits, reference_name = row["token_ids"], logits, name
                continue
            comparison = {"variant": name, "against": reference_name, "kind": "streamed_vs_streamed",
                          "tokens_equal": row["token_ids"] == reference_tokens,
                          "logits_equal": logits == reference_logits}
        else:
            ref_row, ref_logits = resident[f"resident-u{unit}"]
            extra_row, extra_logits = resident[f"resident-extra-u{unit}"]
            comparison = {"variant": name, "against": ref_row["name"], "kind": "resident_vs_streamed",
                          "tokens_equal": row["token_ids"] == ref_row["token_ids"],
                          "logits_equal": logits == ref_logits,
                          "resident_with_extra_buffers_tokens_equal":
                              extra_row["token_ids"] == row["token_ids"],
                          "resident_with_extra_buffers_logits_diff":
                              float_difference(extra_logits, logits)}
        comparisons.append(comparison)
        if not (comparison["tokens_equal"] and comparison["logits_equal"]):
            failures += 1
    summary = {"schema_version": 1, "model": arguments.name, "artifact_sha256": sha,
               "prompt": arguments.prompt, "no_resident": arguments.no_resident,
               "cache": {"legacy_slots": arguments.legacy_slots, "layer_slots": arguments.layer_slots,
                         "arena_mib": arguments.arena_mib},
               "rows": rows, "comparisons": comparisons, "failures": failures}
    (arguments.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for comparison in comparisons:
        print(json.dumps(comparison), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
