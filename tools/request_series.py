#!/usr/bin/env python3
"""Send a fixed series of chat requests to a running server and retain every record.

Each request asks for the server's per-request record (`ame_debug`). The script saves the
request body, the full response, the wall time seen by the client, and the /health
document before and after, so a series can be rebuilt without the server's log. A failed
request is saved and the series continues; the exit status is nonzero if any failed.
"""
import argparse
import json
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


def get(base, path, timeout=10):
    with urllib.request.urlopen(base + path, timeout=timeout) as response:
        return json.loads(response.read().decode())


def post(base, body, timeout):
    request = urllib.request.Request(base + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
    begin = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode(), time.monotonic() - begin
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(), time.monotonic() - begin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--model", default="Qwen3.6-35B-A3B")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--short", type=int, default=3, help="identical short requests")
    parser.add_argument("--long", type=int, default=1, help="32-token requests")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--label", default="series")
    arguments = parser.parse_args()
    base = arguments.base_url.rstrip("/")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    plan = [("short", SHORT)] * arguments.short + [("long", LONG)] * arguments.long
    failures = 0
    index_rows = []
    for number, (kind, template) in enumerate(plan, start=1):
        name = f"{arguments.label}-{number:02d}-{kind}"
        body = {"model": arguments.model, "temperature": 0, "ame_debug": True, **template}
        health_before = get(base, "/health")
        status, text, client_seconds = post(base, body, arguments.timeout)
        health_after = get(base, "/health")
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        ok = status == 200 and isinstance(parsed, dict) and "choices" in parsed
        failures += 0 if ok else 1
        document = {"schema_version": 1, "name": name, "kind": kind, "request": body,
                    "http_status": status, "client_seconds": client_seconds,
                    "response": parsed if parsed is not None else text,
                    "health_before": health_before, "health_after": health_after}
        (arguments.output_dir / f"{name}.json").write_text(json.dumps(document, indent=2) + "\n")
        record = (parsed or {}).get("ame") if isinstance(parsed, dict) else None
        index_rows.append({"name": name, "kind": kind, "http_status": status,
                           "client_seconds": client_seconds,
                           "status": (record or {}).get("status"),
                           "token_ids": (record or {}).get("token_ids")})
        print(json.dumps(index_rows[-1]), flush=True)
    (arguments.output_dir / f"{arguments.label}-index.json").write_text(
        json.dumps({"schema_version": 1, "rows": index_rows, "failures": failures}, indent=2) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
