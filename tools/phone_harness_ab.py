#!/usr/bin/env python3
"""Alternating fresh-process A/B of the on-device harness (resident versus streamed variants).

Each arm is `NAME=extra harness arguments`. One warmup round runs every arm once (not
summarised), then --trials measured rounds run every arm in order, so cache state alternates
between arms. Every run is a fresh process. Records are pulled from the device. The logits of
each run are hashed on the device and the hash is kept, so a hash mismatch between arms is
visible without pulling the large logits files. Memory is the harness's own process samples
(VmHWM and RSS) and system MemAvailable; `pgpgin` is system-wide block-layer reads from
/proc/vmstat around the run, not process I/O.
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


def adb(*command, timeout=3600):
    return subprocess.run(["adb", "shell", *command], capture_output=True, text=True, timeout=timeout)


def meminfo():
    values = {}
    for line in adb("cat", "/proc/meminfo").stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].rstrip(":") in ("MemAvailable", "SwapFree"):
            values[parts[0].rstrip(":")] = int(parts[1])
    return values


def vmstat():
    values = {}
    for line in adb("cat", "/proc/vmstat").stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in ("pgpgin", "pswpin", "pgmajfault"):
            values[parts[0]] = int(parts[1])
    return values


def summarise(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return {"median": statistics.median(values), "min": min(values), "max": max(values),
            "n": len(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-dir", required=True)
    parser.add_argument("--binary", default="llama_baseline")
    parser.add_argument("--model", required=True, help="file name inside the device directory")
    parser.add_argument("--manifest", help="manifest file name inside the device directory")
    parser.add_argument("--sha", required=True)
    parser.add_argument("--prompt-file", required=True, help="file name inside the device directory")
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    directory = arguments.device_dir
    arms = []
    for item in arguments.arm:
        name, _, rest = item.partition("=")
        arms.append((name, rest))

    results = []
    log = arguments.output_dir / "progress.log"

    def one(round_index, name, extra, measured):
        run_id = f"{'t' if measured else 'w'}{round_index:02d}-{name}"
        record_path, logits_path = f"{directory}/{run_id}.json", f"{directory}/{run_id}.logits"
        command = [f"{directory}/{arguments.binary}", "--model", f"{directory}/{arguments.model}",
                   "--artifact-sha", arguments.sha, "--prompt-file", f"{directory}/{arguments.prompt_file}",
                   "--max-new-tokens", str(arguments.max_new_tokens), "--context", str(arguments.context),
                   "--threads", str(arguments.threads), "--temperature", "0", "--seed", "1234",
                   "--logits-output", logits_path, "--output", record_path, "--run-id", run_id,
                   "--tag", "physical_target"]
        if arguments.chat:
            command.append("--chat")
        if "--stream" in extra.split():
            command += ["--manifest", f"{directory}/{arguments.manifest}"]
        command += extra.split()
        before_mem, before_vm = meminfo(), vmstat()
        begin = time.monotonic()
        done = adb(" ".join(command) + f" 2> {directory}/{run_id}.err; echo rc=$?")
        wall = time.monotonic() - begin
        after_vm = vmstat()
        entry = {"round": round_index, "arm": name, "measured": measured, "run_id": run_id,
                 "controller_wall_seconds": round(wall, 2), "failed": None,
                 "mem_available_before_kib": before_mem.get("MemAvailable"),
                 "pgpgin_kib": after_vm.get("pgpgin", 0) - before_vm.get("pgpgin", 0),
                 "pswpin_pages": after_vm.get("pswpin", 0) - before_vm.get("pswpin", 0),
                 "majflt": after_vm.get("pgmajfault", 0) - before_vm.get("pgmajfault", 0)}
        pulled = arguments.output_dir / f"{run_id}.json"
        subprocess.run(["adb", "pull", record_path, str(pulled)], capture_output=True)
        if "rc=0" not in done.stdout or not pulled.is_file():
            entry["failed"] = f"harness did not finish: {done.stdout[-200:]} {done.stderr[-200:]}"
        else:
            record = json.loads(pulled.read_text())
            if record.get("status") != "ok":
                entry["failed"] = f"status {record.get('status')}: {record.get('error')}"
            else:
                logits_hash = adb(f"sha256sum {logits_path}").stdout.split()
                entry.update({
                    "token_ids": record["output"]["token_ids"],
                    "prompt_tokens": record["prompt_token_count"],
                    "prefill_unit": record["options"]["prefill_unit_tokens"],
                    "logits_sha256": logits_hash[0] if logits_hash else None,
                    "load_seconds": record["timing"]["load_seconds"],
                    "prefill_seconds": record["timing"]["prefill_seconds"],
                    "ttft_seconds": record["timing"]["request_ttft_seconds"],
                    "decode_seconds": record["timing"]["decode_seconds"],
                    "decode_tokens_per_second": record["timing"]["decode_tokens_per_second"],
                    "generate_seconds": record["timing"]["generate_seconds"],
                    "wall_seconds": record["timing"]["wall_seconds"],
                    "vm_hwm_kib": record["memory"]["vm_hwm_kib_end"],
                    "mem_available_end_kib": record["memory"]["mem_available_kib_end"],
                    "app_read_bytes": (record.get("streaming") or {}).get("read_bytes"),
                    "hits": (record.get("streaming") or {}).get("hits"),
                    "misses": (record.get("streaming") or {}).get("misses"),
                    "evictions": (record.get("streaming") or {}).get("evictions"),
                    "arena_bytes": (record.get("streaming") or {}).get("arena_bytes"),
                    "read_wait_seconds": (record.get("streaming") or {}).get("read_wait_seconds"),
                    "artifact_hash_seconds": (record.get("streaming") or {}).get("artifact_hash_seconds")})
        adb(f"rm -f {logits_path} {record_path} {directory}/{run_id}.err")
        with open(log, "a") as stream:
            stream.write(json.dumps({k: entry.get(k) for k in
                                     ("round", "arm", "measured", "failed", "wall_seconds",
                                      "decode_tokens_per_second")}) + "\n")
        print(json.dumps({k: entry.get(k) for k in ("round", "arm", "measured", "failed")}), flush=True)
        results.append(entry)

    for index in range(arguments.warmup):
        for name, extra in arms:
            one(index, name, extra, False)
    for index in range(arguments.trials):
        for name, extra in arms:
            one(index, name, extra, True)

    summary = {"schema_version": 1, "arms": {name: extra for name, extra in arms},
               "trials": arguments.trials, "warmup": arguments.warmup,
               "device_dir": "<device-dir>", "model": arguments.model, "sha256": arguments.sha,
               "max_new_tokens": arguments.max_new_tokens, "context": arguments.context,
               "threads": arguments.threads, "chat": arguments.chat, "runs": results, "by_arm": {}}
    token_groups, logits_groups = {}, {}
    for name, _ in arms:
        rows = [r for r in results if r["arm"] == name and r["measured"] and not r["failed"]]
        summary["by_arm"][name] = {
            "ok_runs": len(rows),
            "failed_runs": len([r for r in results if r["arm"] == name and r["measured"] and r["failed"]]),
            **{key: summarise([r.get(key) for r in rows]) for key in (
                "wall_seconds", "ttft_seconds", "prefill_seconds", "decode_seconds",
                "decode_tokens_per_second", "generate_seconds", "vm_hwm_kib", "mem_available_end_kib",
                "app_read_bytes", "pgpgin_kib", "majflt", "read_wait_seconds")}}
        for row in rows:
            token_groups.setdefault(json.dumps(row["token_ids"]), []).append(name)
            logits_groups.setdefault(row["logits_sha256"], []).append(name)
    summary["token_groups"] = {key: sorted(set(value)) for key, value in token_groups.items()}
    summary["logits_groups"] = {str(key): sorted(set(value)) for key, value in logits_groups.items()}
    summary["token_ids_identical_across_all_runs"] = len(token_groups) == 1
    summary["failed"] = [r["run_id"] for r in results if r["failed"]]
    (arguments.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"tokens_identical": summary["token_ids_identical_across_all_runs"],
                      "failed": summary["failed"]}))
    return 1 if summary["failed"] or not summary["token_ids_identical_across_all_runs"] else 0


if __name__ == "__main__":
    sys.exit(main())
