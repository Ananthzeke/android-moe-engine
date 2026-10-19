#!/usr/bin/env python3
"""Negative and lifetime tests for the verified artifact path."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import artifact_fixture  # noqa: E402


def invoke(binary: Path, model: Path, manifest: Path, sha: str, extra=None):
    command = [str(binary), "--model", str(model), "--manifest", str(manifest),
               "--sha", sha]
    if extra:
        command.extend(extra)
    return subprocess.run(command, text=True, capture_output=True, check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    failures = 0

    def record(label, condition, payload):
        nonlocal failures
        print(("PASS" if condition else "FAIL") + f" {label}: {payload}")
        failures += 0 if condition else 1

    def fixture(name):
        directory = Path(tempfile.mkdtemp())
        return directory, *artifact_fixture.write_fixture(directory / name)

    with tempfile.TemporaryDirectory() as root_text:
        root = Path(root_text)
        model, manifest, sha = artifact_fixture.write_fixture(root / "valid")
        result = invoke(args.binary, model, manifest, sha)
        record("valid artifact", result.returncode == 0, result.stdout.strip())

        wrong_dir, wrong, wrong_manifest, wrong_sha = fixture("wrong")
        data = bytearray(wrong.read_bytes())
        data[0] ^= 0x01
        wrong.write_bytes(data)
        result = invoke(args.binary, wrong, wrong_manifest, sha)
        record("same-size bit flip", result.returncode != 0 and '"open"' in result.stdout,
               result.stdout.strip())

        # A matching sidecar for the wrong file must not override the expected identity.
        matching_dir, matching, matching_manifest, _ = fixture("matching")
        matching_data = bytearray(matching.read_bytes())
        matching_data[0] ^= 0x01
        matching.write_bytes(matching_data)
        matching_sha = hashlib.sha256(matching.read_bytes()).hexdigest()
        matching_manifest.write_text(artifact_fixture.manifest_text(
            artifact_fixture.manifest_rows(), matching.stat().st_size, matching_sha))
        result = invoke(args.binary, matching, matching_manifest, sha)
        record("wrong artifact with matching sidecar", result.returncode != 0,
               result.stdout.strip())

        swap_dir, swap, swap_manifest, _ = fixture("swap")
        rows = artifact_fixture.manifest_rows()
        first = list(rows[0])
        second = list(rows[1])
        first[8], second[8] = second[8], first[8]
        rows[0], rows[1] = tuple(first), tuple(second)
        swap_manifest.write_text(artifact_fixture.manifest_text(rows, swap.stat().st_size, sha))
        result = invoke(args.binary, swap, swap_manifest, sha)
        record("swapped in-bounds offsets", result.returncode != 0, result.stdout.strip())

        missing_dir, missing, missing_manifest, _ = fixture("missing")
        missing_manifest.write_text(artifact_fixture.manifest_text(
            artifact_fixture.manifest_rows()[:-1], missing.stat().st_size, sha))
        result = invoke(args.binary, missing, missing_manifest, sha)
        record("missing routed tensor", result.returncode != 0, result.stdout.strip())

        duplicate_dir, duplicate, duplicate_manifest, _ = fixture("duplicate")
        duplicate_manifest.write_text(artifact_fixture.manifest_text(
            artifact_fixture.manifest_rows() + [artifact_fixture.manifest_rows()[0]],
            duplicate.stat().st_size, sha))
        result = invoke(args.binary, duplicate, duplicate_manifest, sha)
        record("duplicate tensor", result.returncode != 0, result.stdout.strip())

        replace_dir, replace_model, replace_manifest, replace_sha = fixture("replace")
        replacement = replace_dir / "replacement.gguf"
        replacement.write_bytes(bytearray(replace_model.read_bytes()))
        replacement_data = bytearray(replacement.read_bytes())
        replacement_data[0] ^= 0x01
        replacement.write_bytes(replacement_data)
        result = invoke(args.binary, replace_model, replace_manifest, replace_sha,
                        ["--replace-after-open", str(replacement)])
        record("replacement after open uses retained FD", result.returncode == 0,
               result.stdout.strip())

        rename_dir, rename_model, rename_manifest, rename_sha = fixture("rename")
        result = invoke(args.binary, rename_model, rename_manifest, rename_sha,
                        ["--rename-after-open", str(rename_dir / "renamed.gguf")])
        record("rename after open uses retained FD", result.returncode == 0,
               result.stdout.strip())

        link_dir, link_model, link_manifest, link_sha = fixture("link")
        result = invoke(args.binary, link_model, link_manifest, link_sha,
                        ["--link-after-open", str(link_dir / "hardlink.gguf")])
        record("hard link does not replace retained FD", result.returncode == 0,
               result.stdout.strip())

        truncate_dir, truncate_model, truncate_manifest, truncate_sha = fixture("truncate")
        result = invoke(args.binary, truncate_model, truncate_manifest, truncate_sha,
                        ["--truncate-after-open"])
        record("truncation before install refused", result.returncode != 0,
               result.stdout.strip())

        mutate_dir, mutate_model, mutate_manifest, mutate_sha = fixture("mutate")
        result = invoke(args.binary, mutate_model, mutate_manifest, mutate_sha,
                        ["--mutate-after-open", "4", "0x42"])
        record("in-place mutation before install refused", result.returncode != 0,
               result.stdout.strip())

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())