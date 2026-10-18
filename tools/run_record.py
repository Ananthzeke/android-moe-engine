#!/usr/bin/env python3
"""Record one command execution with atomic start/final records.

The command is opaque. A measurement JSON written by the command can be validated
and hashed. This tool does not interpret model results or claim device support.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

SCHEMA_VERSION = 1


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path, document):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    data = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    with open(temporary, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def file_record(path):
    path = Path(path)
    if not path.is_file():
        return None
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def measurement_record(path):
    record = file_record(path)
    if record is None:
        return None
    try:
        document = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        record["parse_error"] = str(error)
        return record
    record["schema_version"] = document.get("schema_version")
    record["status"] = document.get("status")
    record["run_id"] = (document.get("run") or {}).get("id")
    record["tag"] = (document.get("run") or {}).get("tag")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", required=True, help="directory for start, logs, and result")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tag", default="host", help="physical_target, host, or emulator")
    parser.add_argument("--timeout", type=float, default=None, help="seconds before termination")
    parser.add_argument("--measurement", help="JSON path the command writes")
    parser.add_argument("--cwd", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="command after --; it is executed without a shell")
    arguments = parser.parse_args()
    command = arguments.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")

    record_dir = Path(arguments.record_dir)
    record_dir.mkdir(parents=True, exist_ok=True)
    started_path = record_dir / "started.json"
    result_path = record_dir / "result.json"
    if not arguments.overwrite and (started_path.exists() or result_path.exists()):
        print(f"Refusing to overwrite existing record in {record_dir}", file=sys.stderr)
        return 2

    started_at = utc_now()
    atomic_write(started_path, {
        "schema_version": SCHEMA_VERSION,
        "run_id": arguments.run_id,
        "tag": arguments.tag,
        "started_at": started_at,
        "command": command,
        "cwd": str(Path(arguments.cwd or os.getcwd()).resolve()),
        "timeout_seconds": arguments.timeout,
        "measurement_path": arguments.measurement,
        "host_pid": os.getpid(),
    })

    stdout_path = record_dir / "stdout.log"
    stderr_path = record_dir / "stderr.log"
    begin = time.monotonic()
    status = "failed"
    exit_code = None
    termination = "not_started"
    error_text = None
    with open(stdout_path, "wb") as stdout_stream, open(stderr_path, "wb") as stderr_stream:
        try:
            process = subprocess.Popen(
                command,
                cwd=arguments.cwd,
                stdout=stdout_stream,
                stderr=stderr_stream,
                start_new_session=True,
            )
        except OSError as error:
            error_text = str(error)
            termination = "spawn_failed"
            process = None
        if process is not None:
            try:
                exit_code = process.wait(timeout=arguments.timeout)
                if exit_code == 0:
                    status = "completed"
                    termination = "exit_zero"
                else:
                    status = "failed"
                    termination = "exit_nonzero"
            except subprocess.TimeoutExpired:
                status = "timeout"
                termination = "timeout"
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                exit_code = process.returncode
            except KeyboardInterrupt:
                status = "interrupted"
                termination = "interrupt"
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                process.wait()
                exit_code = process.returncode

    finished_at = utc_now()
    measurement = measurement_record(arguments.measurement) if arguments.measurement else None
    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": arguments.run_id,
        "tag": arguments.tag,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": status,
        "termination_reason": termination,
        "exit_code": exit_code,
        "wall_seconds": time.monotonic() - begin,
        "command": command,
        "stdout": file_record(stdout_path),
        "stderr": file_record(stderr_path),
        "measurement": measurement,
    }
    if error_text is not None:
        result["error"] = error_text
    if arguments.measurement and measurement is None:
        result["measurement_missing"] = True
    atomic_write(result_path, result)
    print(json.dumps({"status": status, "exit_code": exit_code, "record": str(result_path)}))
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
