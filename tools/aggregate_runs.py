#!/usr/bin/env python3
"""Aggregate completed run records without inventing measurements."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def median(values):
    return statistics.median(values) if values else None


def summarize(values):
    if not values:
        return {"count": 0, "median": None, "minimum": None, "maximum": None}
    return {
        "count": len(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", action="append", required=True,
                        help="directory containing run subdirectories; repeatable")
    parser.add_argument("--output", help="write aggregate JSON here")
    parser.add_argument("--markdown", help="write a Markdown table here")
    parser.add_argument("--tag", help="only include this execution tag")
    arguments = parser.parse_args()

    runs = []
    problems = []
    for root in arguments.records:
        for result_path in sorted(Path(root).rglob("result.json")):
            result = json.loads(result_path.read_text())
            if arguments.tag and result.get("tag") != arguments.tag:
                continue
            measurement = result.get("measurement")
            entry = {
                "record": str(result_path),
                "run_id": result.get("run_id"),
                "tag": result.get("tag"),
                "status": result.get("status"),
                "termination_reason": result.get("termination_reason"),
                "exit_code": result.get("exit_code"),
            }
            if result.get("status") != "completed":
                problems.append({"run": entry["run_id"], "reason": "run_not_completed"})
                runs.append(entry)
                continue
            if not measurement:
                problems.append({"run": entry["run_id"], "reason": "measurement_missing"})
                runs.append(entry)
                continue
            measurement_path = Path(measurement.get("path", ""))
            if not measurement_path.is_file():
                problems.append({"run": entry["run_id"], "reason": "measurement_file_missing"})
                runs.append(entry)
                continue
            if sha256_file(measurement_path) != measurement.get("sha256"):
                problems.append({"run": entry["run_id"], "reason": "measurement_hash_mismatch"})
                runs.append(entry)
                continue
            document = json.loads(measurement_path.read_text())
            if document.get("status") not in ("ok", "pass"):
                problems.append({"run": entry["run_id"], "reason": "measurement_status_not_success"})
                runs.append(entry)
                continue
            timing = document.get("timing") or {}
            memory = document.get("memory") or {}
            entry["measurement_status"] = document.get("status")
            entry["request_ttft_seconds"] = timing.get("request_ttft_seconds")
            entry["prefill_seconds"] = timing.get("prefill_seconds")
            entry["decode_tokens_per_second"] = timing.get("decode_tokens_per_second")
            entry["load_seconds"] = timing.get("load_seconds")
            entry["vm_hwm_kib_end"] = memory.get("vm_hwm_kib_end")
            entry["output_token_ids"] = (document.get("output") or {}).get("token_ids")
            runs.append(entry)

    completed = [run for run in runs if run.get("measurement_status") in ("ok", "pass")]
    groups = {}
    for run in completed:
        groups.setdefault(run["tag"], []).append(run)
    summary = {}
    for tag, entries in groups.items():
        summary[tag] = {
            "runs": len(entries),
            "request_ttft_seconds": summarize([run["request_ttft_seconds"] for run in entries
                                               if run.get("request_ttft_seconds") is not None]),
            "decode_tokens_per_second": summarize([run["decode_tokens_per_second"] for run in entries
                                                   if run.get("decode_tokens_per_second") is not None]),
            "load_seconds": summarize([run["load_seconds"] for run in entries
                                       if run.get("load_seconds") is not None]),
            "vm_hwm_kib_end": summarize([run["vm_hwm_kib_end"] for run in entries
                                         if run.get("vm_hwm_kib_end") is not None]),
        }
        identical = True
        reference = None
        for run in entries:
            tokens = run.get("output_token_ids")
            if tokens is None:
                identical = False
                break
            if reference is None:
                reference = tokens
            elif tokens != reference:
                identical = False
        summary[tag]["token_ids_identical_across_runs"] = identical
    report = {
        "schema_version": 1,
        "runs": runs,
        "groups": summary,
        "problems": problems,
        "notes": [
            "Host results do not establish phone throughput.",
            "A physical_target tag is supplied by the run command, not inferred.",
            "Missing counters remain null; no zero is substituted.",
        ],
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        Path(arguments.output).write_text(text)
    else:
        sys.stdout.write(text)
    if arguments.markdown:
        lines = ["| tag | runs | TTFT median s | decode tok/s median | peak VmHWM KiB median | token IDs stable |",
                 "| --- | ---: | ---: | ---: | ---: | --- |"]
        for tag, group in summary.items():
            def value(field):
                item = group[field]["median"]
                return "null" if item is None else f"{item:.4f}"
            lines.append(f"| {tag} | {group['runs']} | {value('request_ttft_seconds')} | "
                         f"{value('decode_tokens_per_second')} | {value('vm_hwm_kib_end')} | "
                         f"{group['token_ids_identical_across_runs']} |")
        Path(arguments.markdown).write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
