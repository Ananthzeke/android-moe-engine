"""Exercise the run recorder's success, failure, timeout, and atomic records."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

root = Path(__file__).resolve().parents[1]
recorder = root / "tools/run_record.py"


def invoke(record_dir, run_id, command, timeout=None, measurement=None, overwrite=False,
           request_log=None):
    arguments = [sys.executable, str(recorder), "--record-dir", str(record_dir),
                 "--run-id", run_id, "--tag", "host"]
    if timeout is not None:
        arguments += ["--timeout", str(timeout)]
    if measurement is not None:
        arguments += ["--measurement", str(measurement)]
    if request_log is not None:
        arguments += ["--request-log", str(request_log)]
    if overwrite:
        arguments.append("--overwrite")
    arguments += ["--", *command]
    return subprocess.run(arguments, capture_output=True, text=True)


with tempfile.TemporaryDirectory() as directory:
    base = Path(directory)

    successful = base / "success"
    measurement = successful / "measurement.json"
    command = [sys.executable, "-c",
               f"import json; json.dump({{'schema_version': 1, 'status': 'ok', "
               f"'run': {{'id': 'fixture', 'tag': 'host'}}}}, open({str(measurement)!r}, 'w'))"]
    result = invoke(successful, "fixture-success", command, measurement=measurement)
    if result.returncode != 0:
        raise SystemExit(f"Success recorder failed: {result.stdout}{result.stderr}")
    started = json.loads((successful / "started.json").read_text())
    finished = json.loads((successful / "result.json").read_text())
    if started["schema_version"] != 1 or finished["schema_version"] != 1:
        raise SystemExit("Unexpected record schema")
    if finished["status"] != "completed" or finished["termination_reason"] != "exit_zero":
        raise SystemExit(f"Unexpected success status: {finished}")
    if finished["measurement"]["status"] != "ok" or finished["measurement"]["schema_version"] != 1:
        raise SystemExit("Measurement was not parsed")
    if not (successful / "stdout.log").is_file() or not (successful / "stderr.log").is_file():
        raise SystemExit("Logs were not created")
    for stream in ("stdout", "stderr"):
        record = finished[stream]
        if record is None or len(record["sha256"]) != 64:
            raise SystemExit(f"Missing {stream} hash")

    failed = base / "failure"
    result = invoke(failed, "fixture-failure", [sys.executable, "-c", "raise SystemExit(3)"])
    if result.returncode != 1:
        raise SystemExit("Failure recorder should return nonzero")
    finished = json.loads((failed / "result.json").read_text())
    if finished["status"] != "failed" or finished["exit_code"] != 3:
        raise SystemExit(f"Unexpected failure status: {finished}")

    started_at = time.monotonic()
    timed_out = base / "timeout"
    result = invoke(timed_out, "fixture-timeout",
                    [sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.4)
    elapsed = time.monotonic() - started_at
    if result.returncode != 1 or elapsed > 8:
        raise SystemExit("Timeout was not enforced")
    finished = json.loads((timed_out / "result.json").read_text())
    if finished["status"] != "timeout" or finished["termination_reason"] != "timeout":
        raise SystemExit(f"Unexpected timeout status: {finished}")

    missing = base / "missing"
    result = invoke(missing, "fixture-missing", [sys.executable, "-c", "pass"],
                    measurement=missing / "absent.json")
    if result.returncode != 0:
        raise SystemExit("Missing measurement should not fail the run")
    finished = json.loads((missing / "result.json").read_text())
    if finished["measurement"] is not None or not finished.get("measurement_missing"):
        raise SystemExit("Missing measurement was not recorded")

    second = invoke(successful, "fixture-success", [sys.executable, "-c", "pass"])
    if second.returncode != 2:
        raise SystemExit("Recorder overwrote an existing record")

    # A server request log: startup record, requests, and one corrupt line.
    log = base / "requests.jsonl"
    lines = [
        {"type": "startup", "first_ready_seconds": 12.5, "artifact_hash_seconds": 3.0,
         "manifest_validation_seconds": 0.5,
         "stages": [{"name": "artifact_open", "seconds": 3.1}, {"name": "model_load", "seconds": 8.0}]},
        {"type": "request", "request_id": "r1", "status": "completed", "prompt_tokens": 4,
         "completion_tokens": 3, "token_ids": [5, 6, 7], "total_seconds": 2.0,
         "phases": [{"name": "prefill", "seconds": 0.5}, {"name": "decode", "seconds": 1.4}]},
        {"type": "request", "request_id": "r2", "status": "failed", "error": "decode failed",
         "prompt_tokens": 4, "completion_tokens": 0, "token_ids": [], "total_seconds": 0.7,
         "phases": [{"name": "prefill", "seconds": 0.6}]},
    ]
    log.write_text("\n".join(json.dumps(line) for line in lines) + "\n{not json\n")
    logged = base / "logged"
    result = invoke(logged, "fixture-log", [sys.executable, "-c", "pass"], request_log=log)
    if result.returncode != 0:
        raise SystemExit(f"Request log recorder failed: {result.stdout}{result.stderr}")
    finished = json.loads((logged / "result.json").read_text())
    summary = finished["request_log"]
    if len(summary["sha256"]) != 64 or summary["startup"]["first_ready_seconds"] != 12.5:
        raise SystemExit(f"Startup record not summarized: {summary}")
    if summary["startup"]["stages"] != {"artifact_open": 3.1, "model_load": 8.0}:
        raise SystemExit(f"Startup stages lost: {summary['startup']}")
    first, second = summary["requests"]
    if first["token_ids_count"] != 3 or len(first["token_ids_sha256"]) != 64:
        raise SystemExit(f"Token IDs not summarized: {first}")
    if second["status"] != "failed" or second["error"] != "decode failed":
        raise SystemExit(f"Failed request not retained: {second}")
    if first["token_ids_sha256"] == second["token_ids_sha256"]:
        raise SystemExit("Different token lists produced the same hash")
    if len(summary["parse_errors"]) != 1 or summary["parse_errors"][0]["line"] != 4:
        raise SystemExit(f"Malformed line not reported: {summary['parse_errors']}")

    absent = base / "no-log"
    result = invoke(absent, "fixture-no-log", [sys.executable, "-c", "pass"],
                    request_log=base / "absent.jsonl")
    finished = json.loads((absent / "result.json").read_text())
    if finished["request_log"] is not None or not finished.get("request_log_missing"):
        raise SystemExit("Missing request log was not recorded")
