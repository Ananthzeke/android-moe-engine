"""Exercise the run recorder's success, failure, timeout, and atomic records."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

root = Path(__file__).resolve().parents[1]
recorder = root / "tools/run_record.py"


def invoke(record_dir, run_id, command, timeout=None, measurement=None, overwrite=False):
    arguments = [sys.executable, str(recorder), "--record-dir", str(record_dir),
                 "--run-id", run_id, "--tag", "host"]
    if timeout is not None:
        arguments += ["--timeout", str(timeout)]
    if measurement is not None:
        arguments += ["--measurement", str(measurement)]
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
