#!/usr/bin/env python3
"""Alternating A/B trials of server variants on a phone, one fresh process per trial.

Each trial stops any server, records system state, starts the server with one variant's
arguments, waits for readiness, sends the same short and 32-token requests, records the
server's per-request records, `VmHWM`, memory, swap and zram state, and kills the process.
Variants alternate (A, B, A, B, ...), so slow drift hits both arms. A failed trial is
recorded and the run continues. `summarize` builds the table from the saved records.

Everything is read from files the run saved; nothing is computed from memory.
"""
import argparse
import json
import shlex
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PACKAGE = "/data/local/tmp/ame-api"
MODEL = "/data/local/tmp/ame-engine/models/Qwen3.6-35B-A3B-Q4_K_M.gguf"
MANIFEST = f"{PACKAGE}/qwen36-35b-v2.manifest"
SHA = "671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7"
SHORT = {"messages": [{"role": "user",
                       "content": "Say hello in one short sentence. Answer directly without reasoning."}],
         "max_tokens": 8}
LONG = {"messages": [{"role": "user",
                      "content": "Count from one to thirty, separated by commas. Answer directly without reasoning."}],
        "max_tokens": 32}


def adb(serial, *arguments, timeout=120, check=False):
    command = ["adb"] + (["-s", serial] if serial else []) + list(arguments)
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        raise RuntimeError(f"{' '.join(command)}: {result.stderr.strip()}")
    return result


def shell(serial, script, timeout=120):
    return adb(serial, "shell", script, timeout=timeout)


STATE_SCRIPT = (
    "grep -E 'MemTotal|MemAvailable|^Cached|SwapTotal|SwapFree' /proc/meminfo; "
    "grep -E '^(pswpin|pswpout|pgmajfault|workingset_refault)' /proc/vmstat; "
    "echo zram_mm_stat $(cat /sys/block/zram0/mm_stat 2>&1); "
    "echo battery $(dumpsys battery | grep -E 'level|temperature' | tr -s ' ' | tr '\\n' ' '); "
    "for c in 0 1 2 3 4 5 6 7; do echo cpu$c $(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq 2>&1); done"
)


def http(url, body=None, timeout=900):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data,
                                     headers={"content-type": "application/json"} if data else {})
    begin = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode(), time.monotonic() - begin
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(), time.monotonic() - begin


def wait_ready(base, limit):
    begin = time.monotonic()
    while time.monotonic() - begin < limit:
        try:
            status, text, _ = http(base + "/health", timeout=5)
            if status == 200 and json.loads(text).get("status") == "ready":
                return time.monotonic() - begin
        except (OSError, ValueError):
            pass
        time.sleep(5)
    return None


def run_trial(arguments, number, name, variant_args, directory):
    serial, base = arguments.serial, f"http://127.0.0.1:{arguments.port}"
    directory.mkdir(parents=True, exist_ok=True)
    record = {"trial": number, "variant": name, "args": variant_args, "failed": None}
    shell(serial, "pkill -x ame-server; sleep 3; ps -A | grep -c ame-server")
    (directory / "state-before.txt").write_text(shell(serial, STATE_SCRIPT).stdout)
    log = f"{PACKAGE}/ab-{number:02d}-{name}.jsonl"
    command = (f"cd {PACKAGE}; rm -f {log}; nohup ./ame-server --model {MODEL} "
               f"--model-id Qwen3.6-35B-A3B --stream --manifest {MANIFEST} --artifact-sha {SHA} "
               f"--context 512 --threads 4 --port {arguments.port} --request-log {log} "
               f"{variant_args} > ab-start.log 2>&1 < /dev/null & echo $!")
    pid = shell(serial, command).stdout.strip().splitlines()[-1]
    record["pid"] = pid
    adb(serial, "forward", f"tcp:{arguments.port}", f"tcp:{arguments.port}")
    ready = wait_ready(base, arguments.ready_timeout)
    record["ready_wait_seconds"] = ready
    if ready is None:
        record["failed"] = "server not ready"
        (directory / "server-start.log").write_text(
            shell(serial, f"tail -c 2000 {PACKAGE}/ab-start.log").stdout)
    else:
        (directory / "health-ready.json").write_text(http(base + "/health")[1])
        rows = []
        for kind, template in (("short", SHORT), ("long", LONG)):
            body = {"model": "Qwen3.6-35B-A3B", "temperature": 0, "ame_debug": True, **template}
            status, text, seconds = http(base + "/v1/chat/completions", body, arguments.timeout)
            try:
                parsed = json.loads(text)
            except ValueError:
                parsed = None
            ok = status == 200 and isinstance(parsed, dict) and "choices" in parsed
            (directory / f"{kind}.json").write_text(json.dumps(
                {"request": body, "http_status": status, "client_seconds": seconds,
                 "response": parsed if parsed is not None else text}, indent=2) + "\n")
            rows.append({"kind": kind, "http_status": status, "client_seconds": seconds, "ok": ok})
            if not ok:
                record["failed"] = f"{kind} request failed with HTTP {status}"
                break
        record["requests"] = rows
        (directory / "health-after.json").write_text(http(base + "/health")[1])
    status_text = shell(serial, f"grep -E 'VmHWM|VmRSS' /proc/{pid}/status 2>&1").stdout
    (directory / "process-status.txt").write_text(status_text)
    (directory / "state-after.txt").write_text(shell(serial, STATE_SCRIPT).stdout)
    lmkd = shell(serial, "logcat -d -b all 2>&1 | grep -i -E 'lmkd|lowmemorykiller|killing' | tail -n 20")
    (directory / "lmkd.txt").write_text(lmkd.stdout or "(no matching lines or logcat unavailable)\n")
    adb(serial, "pull", log, str(directory / "requests.jsonl"))
    shell(serial, "pkill -x ame-server")
    (directory / "trial.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def parse_state(path):
    values = {}
    if not path.is_file():
        return values
    for line in path.read_text().splitlines():
        parts = line.replace(":", " ").split()
        if len(parts) >= 2 and parts[1].isdigit():
            values[parts[0]] = int(parts[1])
        if parts and parts[0] == "zram_mm_stat" and len(parts) >= 3 and parts[1].isdigit():
            values["zram_orig_bytes"], values["zram_compr_bytes"] = int(parts[1]), int(parts[2])
    return values


def summarize(directory):
    rows = []
    for trial in sorted(directory.glob("trial-*")):
        if not (trial / "trial.json").is_file():
            continue  # still running, or the harness stopped before writing it
        info = json.loads((trial / "trial.json").read_text())
        row = {"trial": info["trial"], "variant": info["variant"], "failed": info["failed"]}
        for kind in ("short", "long"):
            path = trial / f"{kind}.json"
            if not path.is_file():
                continue
            document = json.loads(path.read_text())
            record = (document["response"] or {}).get("ame") if isinstance(document["response"], dict) else None
            if not record:
                continue
            phases = {p["name"]: p for p in record["phases"]}
            row[f"{kind}_total"] = record["total_seconds"]
            row[f"{kind}_tokens"] = record["token_ids"]
            for phase in ("prefill", "decode"):
                p = phases.get(phase)
                if p:
                    row[f"{kind}_{phase}_s"] = p["seconds"]
                    row[f"{kind}_{phase}_rchar"] = p["proc_delta"]["rchar"]
                    row[f"{kind}_{phase}_read_bytes"] = p["proc_delta"]["read_bytes"]
                    row[f"{kind}_{phase}_majflt"] = p["proc_delta"]["majflt"]
                    row[f"{kind}_{phase}_hits"] = (p["expert_delta"] or {}).get("hits")
                    row[f"{kind}_{phase}_misses"] = (p["expert_delta"] or {}).get("misses")
                    row[f"{kind}_{phase}_expert_bytes"] = (p["expert_delta"] or {}).get("expert_bytes")
                    row[f"{kind}_{phase}_read_s"] = (p["expert_delta"] or {}).get("read_seconds")
            if kind == "long" and record["phases"]:
                steps = record["decode_step_seconds"]
                row["long_decode_median_step_s"] = statistics.median(steps) if steps else None
                row["long_tokens_per_s"] = (len(steps) / row.get("long_decode_s") if steps and row.get("long_decode_s") else None)
        after = parse_state(trial / "state-after.txt")
        before = parse_state(trial / "state-before.txt")
        row["mem_available_kib_after"] = after.get("MemAvailable")
        row["mem_available_kib_before"] = before.get("MemAvailable")
        row["swap_in_pages"] = (after.get("pswpin", 0) - before.get("pswpin", 0)) if after and before else None
        row["zram_compr_bytes_after"] = after.get("zram_compr_bytes")
        status = (trial / "process-status.txt")
        hwm = [int(l.split()[1]) for l in status.read_text().splitlines() if l.startswith("VmHWM")] if status.is_file() else []
        row["vm_hwm_kib"] = hwm[0] if hwm else None
        lmkd = (trial / "lmkd.txt").read_text() if (trial / "lmkd.txt").is_file() else ""
        row["lmkd_lines"] = len([l for l in lmkd.splitlines() if "ame-server" in l])
        rows.append(row)
    return rows


def table(rows):
    variants = sorted({r["variant"] for r in rows})
    metrics = ["short_total", "short_prefill_s", "short_decode_s", "long_total", "long_decode_s",
               "long_decode_median_step_s", "long_tokens_per_s", "short_prefill_read_bytes",
               "long_decode_read_bytes", "short_prefill_majflt", "long_decode_majflt", "vm_hwm_kib",
               "mem_available_kib_after", "swap_in_pages"]
    lines = ["| metric | " + " | ".join(f"{v} median [min, max] (n)" for v in variants) + " |",
             "| --- | " + " | ".join("---" for _ in variants) + " |"]
    for metric in metrics:
        cells = []
        for variant in variants:
            values = [r[metric] for r in rows if r["variant"] == variant and not r["failed"]
                      and r.get(metric) is not None]
            cells.append("n/a" if not values else
                         f"{statistics.median(values):.4g} [{min(values):.4g}, {max(values):.4g}] ({len(values)})")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    failed = [f"trial {r['trial']} {r['variant']}: {r['failed']}" for r in rows if r["failed"]]
    tokens = {}
    for r in rows:
        if r["failed"]:
            continue
        for kind in ("short", "long"):
            tokens.setdefault((kind, json.dumps(r.get(f"{kind}_tokens"))), []).append(
                f"{r['variant']}#{r['trial']}")
    lines.append("")
    lines.append(f"Failed trials: {failed if failed else 'none'}")
    lines.append("Token-ID groups (identical lists share a group):")
    for (kind, ids), members in sorted(tokens.items()):
        lines.append(f"- {kind}: {len(members)} trials, ids {ids[:60]}{'...' if len(ids) > 60 else ''}: {', '.join(members)}")
    lmkd = [f"trial {r['trial']} {r['variant']}: {r['lmkd_lines']}" for r in rows if r.get("lmkd_lines")]
    lines.append(f"lmkd lines mentioning ame-server: {lmkd if lmkd else 'none'}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--variant", action="append", required=True,
                     help="NAME=server arguments, e.g. base='--slots 64 --arena-mib 256'")
    run.add_argument("--trials", type=int, default=5, help="trials per variant")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--serial")
    run.add_argument("--port", type=int, default=8081)
    run.add_argument("--ready-timeout", type=float, default=1800)
    run.add_argument("--timeout", type=float, default=900)
    summary = sub.add_parser("summarize")
    summary.add_argument("directory", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "summarize":
        rows = summarize(arguments.directory)
        (arguments.directory / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
        text = table(rows)
        (arguments.directory / "summary.md").write_text(text + "\n")
        print(text)
        return 0
    variants = []
    for item in arguments.variant:
        name, _, rest = item.partition("=")
        variants.append((name, rest))
    failures = 0
    number = 0
    for _ in range(arguments.trials):
        for name, variant_args in variants:
            number += 1
            directory = arguments.output_dir / f"trial-{number:02d}-{name}"
            try:
                record = run_trial(arguments, number, name, variant_args, directory)
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
                directory.mkdir(parents=True, exist_ok=True)
                record = {"trial": number, "variant": name, "args": variant_args,
                          "failed": f"harness error: {error}"}
                (directory / "trial.json").write_text(json.dumps(record, indent=2) + "\n")
            failures += 1 if record["failed"] else 0
            print(json.dumps({k: record.get(k) for k in ("trial", "variant", "failed")}), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
