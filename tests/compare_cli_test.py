import json
import subprocess
import sys
import tempfile
from pathlib import Path

binary = sys.argv[1]
base = {
    "schema_version": 1,
    "status": "ok",
    "model": {"artifact_sha256": "aa", "file_bytes": 123},
    "options": {"context_tokens": 512, "max_new_tokens": 8, "threads": 4,
                "batch_size": 64, "seed": 1234, "temperature": 0,
                "top_k": 40, "top_p": 0.95, "mmap": True,
                "chat_template": False, "gpu_layers": 0},
    "prompt_token_ids": [1, 2],
    "output": {"token_ids": [3, 4], "finish_reason": "limit"},
    "timing": {"request_ttft_seconds": 1.0, "decode_tokens_per_second": 2.0},
    "memory": {"vm_hwm_kib_end": 100},
}


def compare(directory, left, right):
    paths = []
    for name, record in (("left.json", left), ("right.json", right)):
        path = Path(directory) / name
        path.write_text(json.dumps(record))
        paths.append(str(path))
    output = Path(directory) / "comparison.json"
    result = subprocess.run([binary, "compare", "--left", paths[0], "--right", paths[1],
                             "--output", str(output)], capture_output=True, text=True)
    return result, output


with tempfile.TemporaryDirectory() as directory:
    result, output = compare(directory, base, base)
    if result.returncode != 0 or not output.is_file():
        raise SystemExit(f"Identical comparison failed: {result}")
    document = json.loads(output.read_text())
    if document["compatible"] is not True or document["differences"] != []:
        raise SystemExit(f"Identical comparison reported differences: {document}")
    for side, expected in (("left", 100), ("right", 100)):
        if document["metrics"][side]["peak_rss_kib"] != expected:
            raise SystemExit(f"Metrics lost for {side}: {document['metrics']}")

    different = json.loads(json.dumps(base))
    different["output"]["token_ids"] = [3, 5]
    result, output = compare(directory, base, different)
    if result.returncode != 1:
        raise SystemExit(f"Different tokens must exit 1: {result}")
    document = json.loads(output.read_text())
    if document["compatible"] is not False or "output.token_ids" not in document["differences"]:
        raise SystemExit(f"Token difference not reported: {document}")
