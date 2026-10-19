#!/usr/bin/env python3
"""Exercise the v2 manifest validation before any tensor reaches a kernel."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def fnv1a64(text: str) -> str:
    value = 0xCBF29CE484222325
    for byte in text.encode():
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def row(name="fake.exps", layer=0, role="role", type_id=0, type_name="f32",
        experts=4, expert_bytes=1024, stride=1024, offset=0, nbytes=4096):
    return (name, layer, role, type_id, type_name, experts, expert_bytes, stride,
            offset, nbytes)


def canonical(rows):
    return "\n".join("|".join(str(value) for value in item) for item in rows)


def manifest(rows, artifact_bytes, version=2, artifact_sha="-", tensor_count=None,
             layout=None):
    text = "\n".join([
        f"# ame-expert-manifest {version}",
        f"# artifact_bytes {artifact_bytes}",
        f"# artifact_sha256 {artifact_sha}",
        f"# tensor_count {len(rows) if tensor_count is None else tensor_count}",
        f"# layout_fnv1a64 {fnv1a64(canonical(rows)) if layout is None else layout}",
    ]) + "\n"
    text += "".join("tensor " + " ".join(str(value) for value in item) + "\n" for item in rows)
    return text


def invoke(binary: Path, model: Path, manifest_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(binary), "--model", str(model), "--manifest", str(manifest_path)],
                          text=True, capture_output=True, check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    failures = 0

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        model = root / "model.bin"
        model.write_bytes(b"\x11" * 8192)
        valid_rows = [row()]
        valid = root / "valid.manifest"
        valid.write_text(manifest(valid_rows, model.stat().st_size))
        result = invoke(args.binary, model, valid)
        ok = result.returncode == 0 and json.loads(result.stdout).get("identity_checked") is True
        print(("PASS" if ok else "FAIL") + " valid v2 manifest")
        failures += 0 if ok else 1

        cases = [
            ("wrong tensor_count", manifest(valid_rows, 8192, tensor_count=2)),
            ("duplicate row", manifest([row(), row()], 8192)),
            ("nbytes mismatch", manifest([row(nbytes=4095)], 8192)),
            ("overflow", manifest([row(experts=0xFFFFFFFFFFFFFFFF)], 8192)),
            ("truncated extent", manifest([row(offset=4096, nbytes=4096)], 4096)),
            ("unsupported version", manifest(valid_rows, 8192, version=3)),
            ("missing layout", "# ame-expert-manifest 2\n# artifact_bytes 8192\n"
                              "# artifact_sha256 -\n# tensor_count 1\n"
                              "tensor fake.exps 0 role 0 f32 4 1024 1024 0 4096\n"),
            ("wrong layout", manifest(valid_rows, 8192, layout="0" * 16)),
        ]
        for label, text in cases:
            path = root / (label.replace(" ", "_") + ".manifest")
            path.write_text(text)
            expected_size = 4096 if label == "truncated extent" else 8192
            case_model = model
            if expected_size != 8192:
                case_model = root / "short.bin"
                case_model.write_bytes(b"\x22" * expected_size)
            result = invoke(args.binary, case_model, path)
            ok = result.returncode != 0
            print(("PASS" if ok else "FAIL") + f" {label}")
            failures += 0 if ok else 1

        sparse = root / "sparse.bin"
        with sparse.open("wb") as stream:
            stream.truncate(4 * 1024 * 1024 * 1024 + 8192)
        rows = [row(offset=4 * 1024 * 1024 * 1024)]
        sparse_manifest = root / "sparse.manifest"
        sparse_manifest.write_text(manifest(rows, sparse.stat().st_size))
        result = invoke(args.binary, sparse, sparse_manifest)
        ok = result.returncode == 0
        print(("PASS" if ok else "FAIL") + " offset above 4 GiB")
        failures += 0 if ok else 1

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
