#!/usr/bin/env python3
"""API smoke against a running local OpenAI-compatible server."""
import argparse, json, pathlib, socket, sys, urllib.error, urllib.request

def post(base, payload, timeout=120):
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()

def get(base, path):
    with urllib.request.urlopen(base + path, timeout=10) as response:
        return response.status, response.read().decode()

def parse_sse(text):
    events = [line[6:] for line in text.splitlines() if line.startswith("data: ")]
    done = events and events[-1].strip() == "[DONE]"
    chunks = [json.loads(e) for e in events if e.strip() != "[DONE]"]
    return done, chunks

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument("--model", default="Qwen3.5-2B")
    p.add_argument("--output", type=pathlib.Path, required=True)
    a = p.parse_args()
    base = a.base_url.rstrip("/")
    results = {}
    status, health = get(base, "/health")
    results["health"] = {"http": status, "body": json.loads(health)}
    status, models = get(base, "/v1/models")
    results["models"] = {"http": status, "body": json.loads(models)}
    prompt = {"model": a.model, "messages": [{"role": "system", "content": "Be brief."},
                                              {"role": "user", "content": "Name one color."}],
              "max_tokens": 8, "temperature": 0}
    status, body = post(base, prompt)
    plain = json.loads(body)
    results["chat"] = {"http": status, "status": plain.get("choices", [{}])[0].get("finish_reason"),
                       "content": plain.get("choices", [{}])[0].get("message", {}).get("content"),
                       "usage": plain.get("usage")}
    status, sse = post(base, {**prompt, "stream": True,
                              "stream_options": {"include_usage": True}})
    done, chunks = parse_sse(sse)
    results["stream"] = {"http": status, "done": done, "chunks": len(chunks),
                         "finish": chunks[-2]["choices"][0]["finish_reason"] if len(chunks) > 1 else None,
                         "usage": chunks[-1].get("usage") if chunks else None}
    status, body = post(base, {**prompt, "stop": ["<think>"]})
    results["stop"] = {"http": status, "finish": json.loads(body)["choices"][0]["finish_reason"]}
    status, body = post(base, {**prompt, "tools": []})
    results["unsupported_tools"] = {"http": status, "error": json.loads(body).get("error", {}).get("code")}
    status, body = post(base, {**prompt, "model": "missing"})
    results["unknown_model"] = {"http": status, "error": json.loads(body).get("error", {}).get("code")}
    status, body = post(base, {**prompt, "messages": []})
    results["malformed_messages"] = {"http": status}
    status, body = post(base, {**prompt, "max_tokens": 100000})
    results["context_overflow"] = {"http": status, "error": json.loads(body).get("error", {}).get("code")}
    # Independent fresh request after all failures.
    status, body = post(base, prompt)
    results["fresh_after_failures"] = {"http": status, "finish": json.loads(body)["choices"][0]["finish_reason"]}
    passed = (results["health"]["body"].get("status") == "ready"
              and results["chat"]["http"] == 200 and results["stream"]["done"]
              and results["unsupported_tools"]["http"] == 400
              and results["unknown_model"]["http"] == 404
              and results["context_overflow"]["http"] == 400
              and results["fresh_after_failures"]["http"] == 200)
    results["passed"] = passed
    a.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0 if passed else 1

if __name__ == "__main__":
    sys.exit(main())
