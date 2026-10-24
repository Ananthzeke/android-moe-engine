#!/usr/bin/env python3
"""Start a server once per variant and require identical emitted token IDs.

Each variant is `NAME=extra server arguments`. The first variant is the reference. Every variant
answers the same two greedy requests (a short one and a 32-token one); their sampled token IDs
must equal the reference's. Records are kept per variant. A variant that fails to start or to
answer is a failure. Used on a host with the pinned model to check that cache layout, readers,
overlap and threadpool options never change what is computed.
"""
import argparse
import json
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SHORT = {"messages": [{"role": "user",
                       "content": "Say hello in one short sentence. Answer directly without reasoning."}],
         "max_tokens": 8}
LONG = {"messages": [{"role": "user",
                      "content": "Count from one to thirty, separated by commas. Answer directly without reasoning."}],
        "max_tokens": 32}


def post(url, body, timeout):
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode())


def wait_ready(base, process, limit):
    begin = time.monotonic()
    while time.monotonic() - begin < limit:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as response:
                if json.loads(response.read().decode()).get("status") == "ready":
                    return True
        except (OSError, ValueError):
            pass
        time.sleep(3)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-sha", required=True)
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ready-timeout", type=float, default=1800)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{arguments.port}"
    reference = None
    failures = 0
    summary = []
    for item in arguments.variant:
        name, _, rest = item.partition("=")
        log = arguments.output_dir / f"{name}.server.log"
        record = {"variant": name, "args": rest, "ok": False, "tokens": {}}
        with open(log, "wb") as stream:
            process = subprocess.Popen(
                [str(arguments.server), "--model", str(arguments.model), "--model-id", "Qwen3.6-35B-A3B",
                 "--stream", "--manifest", str(arguments.manifest), "--artifact-sha",
                 arguments.artifact_sha, "--context", str(arguments.context), "--threads",
                 str(arguments.threads), "--port", str(arguments.port), *shlex.split(rest)],
                stdout=stream, stderr=stream)
            try:
                if not wait_ready(base, process, arguments.ready_timeout):
                    record["error"] = "server did not become ready"
                else:
                    record["ok"] = True
                    for kind, template in (("short", SHORT), ("long", LONG)):
                        body = {"model": "Qwen3.6-35B-A3B", "temperature": 0, "ame_debug": True,
                                **template}
                        status, payload = post(base + "/v1/chat/completions", body, 900)
                        if status != 200 or "ame" not in payload:
                            record["ok"] = False
                            record["error"] = f"{kind} request failed: HTTP {status}"
                            break
                        record["tokens"][kind] = payload["ame"]["token_ids"]
                        (arguments.output_dir / f"{name}.{kind}.json").write_text(
                            json.dumps(payload, indent=2) + "\n")
                    health = json.loads(urllib.request.urlopen(base + "/health", timeout=10).read().decode())
                    record["streaming"] = health.get("streaming")
            finally:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
        if record["ok"]:
            if reference is None:
                reference = record["tokens"]
            record["matches_reference"] = record["tokens"] == reference
            if not record["matches_reference"]:
                failures += 1
        else:
            failures += 1
        summary.append(record)
        print(json.dumps({k: record.get(k) for k in ("variant", "ok", "matches_reference", "error")}),
              flush=True)
    (arguments.output_dir / "summary.json").write_text(
        json.dumps({"schema_version": 1, "reference": reference, "variants": summary,
                    "failures": failures}, indent=2) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
