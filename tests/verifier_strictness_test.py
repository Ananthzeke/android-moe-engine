"""Challenge the complete-model and production verifiers with bad runner output."""
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from logit_io import FormatError, count_mismatched_floats, read_logits, read_tokens  # noqa: E402

ame = sys.argv[1]


def logits_file(path, rows, trailing=b"", declared_rows=None, declared_widths=None):
    data = struct.pack("<Q", len(rows) if declared_rows is None else declared_rows)
    for index, row in enumerate(rows):
        width = len(row) if declared_widths is None else declared_widths[index]
        data += struct.pack("<Q", width) + struct.pack(f"<{len(row)}f", *row)
    path.write_bytes(data + trailing)
    return path


def expect_format_error(label, action):
    try:
        action()
    except FormatError:
        return
    raise SystemExit(f"{label}: accepted malformed data")


def check_parsers(directory):
    good = logits_file(directory / "good.logits", [[1.0, 2.0], [3.0, 4.0]])
    if len(read_logits(good)) != 2:
        raise SystemExit("valid logits were rejected")
    expect_format_error("truncated payload", lambda: read_logits(
        logits_file(directory / "a", [[1.0, 2.0]], declared_widths=[999])))
    expect_format_error("missing rows", lambda: read_logits(
        logits_file(directory / "b", [[1.0]], declared_rows=3)))
    expect_format_error("trailing bytes", lambda: read_logits(
        logits_file(directory / "c", [[1.0]], trailing=b"\0")))
    expect_format_error("empty row", lambda: read_logits(
        logits_file(directory / "d", [[]])))
    expect_format_error("non-finite", lambda: read_logits(
        logits_file(directory / "e", [[float("inf")]])))
    (directory / "short").write_bytes(b"\0\0")
    expect_format_error("short header", lambda: read_logits(directory / "short"))
    (directory / "tokens").write_bytes(struct.pack("<Q", 3) + struct.pack("<2i", 1, 2))
    expect_format_error("short tokens", lambda: read_tokens(directory / "tokens"))
    row = struct.pack("<f", 1.0)
    expect_format_error("row count", lambda: count_mismatched_floats([row, row], [row]))
    expect_format_error("row width", lambda: count_mismatched_floats([row], [row + row]))
    expect_format_error("empty", lambda: count_mismatched_floats([], []))
    if count_mismatched_floats([row], [struct.pack("<f", 2.0)]) != 1:
        raise SystemExit("different floats were not counted")


RUNNER = """#!/usr/bin/env python3
import struct, sys
args = sys.argv[1:]
def value(key):
    return args[len(args) - 1 - args[::-1].index(key) + 1]
rows = {rows!r}
def write_logits(path):
    data = struct.pack("<Q", {declared})
    for row in rows:
        data += struct.pack("<Q", len(row)) + struct.pack("<%df" % len(row), *row)
    open(path, "wb").write(data + {trailing!r})
tokens = {tokens!r}
if "--tokens-output" in args:
    open(value("--tokens-output"), "wb").write(
        struct.pack("<Q", len(tokens)) + struct.pack("<%di" % len(tokens), *tokens))
write_logits(value("--output"))
"""


def runner(path, rows, tokens=(5, 6), trailing=b"", declared=None):
    path.write_text(RUNNER.format(python=sys.executable, rows=[list(row) for row in rows],
                                  tokens=list(tokens), trailing=trailing,
                                  declared=len(rows) if declared is None else declared))
    path.chmod(0o755)
    return path


def run_complete(directory, name, pristine, streamed, corruption=False, broken=None, predict=2):
    model = directory / "model.gguf"
    model.write_bytes(b"fixture")
    manifest = directory / "model.manifest"
    manifest.write_text("fixture\n")
    case = {"name": name, "model": str(model), "manifest": str(manifest),
            "sha256": hashlib.sha256(b"fixture").hexdigest(), "prompt_ids": "1", "context": 8,
            "threads": 1, "batch": 8, "ubatch": 8, "predict": predict, "slots": 2}
    if corruption:
        case["corruption"] = True
    matrix = directory / f"{name}.json"
    matrix.write_text(json.dumps({"cases": [case]}))
    command = [sys.executable, str(ROOT / "tools" / "verify_complete.py"), "--config", str(matrix),
               "--pristine-runner", str(pristine), "--streamed-runner", str(streamed),
               "--output", str(directory / f"{name}.out"), "--workdir", str(directory / name)]
    if broken is not None:
        command += ["--broken-runner", str(broken)]
    return subprocess.run(command, capture_output=True, text=True)


def check_complete(directory):
    rows = [[1.0, 2.0], [3.0, 4.0]]
    good = runner(directory / "good", rows)
    result = run_complete(directory, "identical", good, good)
    if result.returncode != 0:
        raise SystemExit(f"identical runners must pass: {result}")
    cases = {
        "short_rows": runner(directory / "short", rows[:1]),
        "truncated_payload": runner(directory / "trunc", rows, declared=3),
        "trailing": runner(directory / "trail", rows, trailing=b"\1"),
        "token_count": runner(directory / "toks", rows, tokens=(5,)),
        "different_logits": runner(directory / "diff", [[1.0, 2.0], [3.0, 5.0]]),
    }
    for name, streamed in cases.items():
        if run_complete(directory, name, good, streamed).returncode == 0:
            raise SystemExit(f"verify_complete accepted {name}")
    if run_complete(directory, "no_broken", good, good, corruption=True).returncode == 0:
        raise SystemExit("verify_complete skipped a required corruption check")
    broken = runner(directory / "broken", [[1.0, 2.0], [3.0, 9.0]])
    if run_complete(directory, "with_broken", good, good, corruption=True,
                    broken=broken).returncode != 0:
        raise SystemExit("detectable corruption must pass")
    if run_complete(directory, "silent_broken", good, good, corruption=True,
                    broken=good).returncode == 0:
        raise SystemExit("undetected corruption must fail")


HARNESS = """#!/usr/bin/env python3
import json, os, struct, sys
args = sys.argv[1:]
def last(key):
    return args[len(args) - 1 - args[::-1].index(key) + 1]
failing = ("--cancel-before-decode" in args or last("--moe-slots") == "1"
           or last("--artifact-sha") == "0" * 64 or last("--manifest").endswith("missing-consumer.manifest")
           or "AME_IO_FAIL_AFTER_BYTES" in os.environ)
if failing:
    open(last("--output"), "w").write(json.dumps({{"status": "error"}}))
    sys.exit(1)
rows = {rows!r}
data = struct.pack("<Q", len(rows))
for row in rows:
    data += struct.pack("<Q", len(row)) + struct.pack("<%df" % len(row), *row)
if "--logits-output" in args:
    open(last("--logits-output"), "wb").write(data)
open(last("--output"), "w").write(json.dumps({{
    "status": "ok", "prompt_token_ids": [1],
    "options": {{"prefill_unit_tokens": 2, "effective_ubatch_tokens": 2}},
    "output": {{"token_ids": [5, 6]}}}}))
"""


def run_production(directory, name, harness_rows, reference_rows):
    harness = directory / f"{name}-harness"
    harness.write_text(HARNESS.format(python=sys.executable,
                                      rows=[list(row) for row in harness_rows]))
    harness.chmod(0o755)
    reference = runner(directory / f"{name}-reference", reference_rows)
    model = directory / "model.gguf"
    model.write_bytes(b"fixture")
    manifest = directory / "model.manifest"
    manifest.write_text("header\ntensor blk.0.ffn_up_exps.weight\n")
    matrix = directory / f"{name}.json"
    matrix.write_text(json.dumps({"cases": [{
        "name": "case", "model": str(model), "manifest": str(manifest),
        "sha256": hashlib.sha256(b"fixture").hexdigest(), "slots": 4, "admitted": 2}]}))
    return subprocess.run([sys.executable, str(ROOT / "tools" / "verify_production.py"),
                           "--harness", str(harness), "--reference", str(reference),
                           "--supervisor", ame, "--matrix", str(matrix),
                           "--output", str(directory / f"{name}.out"),
                           "--workdir", str(directory / name)], capture_output=True, text=True)


def check_production(directory):
    # The reference carries one row per prompt token before the harness rows start.
    rows = [[1.0, 2.0], [3.0, 4.0]]
    result = run_production(directory, "equal", rows, rows)
    if result.returncode != 0:
        raise SystemExit(f"equal logits must pass: {result}")
    if json.loads((directory / "equal.out").read_text())["rows"][0]["logits_equal"] is not True:
        raise SystemExit("equal logits were not recorded as equal")
    unequal = run_production(directory, "unequal", [[1.0, 2.0], [3.0, 5.0]], rows)
    if unequal.returncode == 0:
        raise SystemExit("verify_production accepted unequal logits")
    if run_production(directory, "short", rows[:1], rows).returncode == 0:
        raise SystemExit("verify_production accepted a missing logit row")


with tempfile.TemporaryDirectory() as scratch:
    scratch = Path(scratch)
    check_parsers(scratch)
    check_complete(scratch)
    check_production(scratch)
