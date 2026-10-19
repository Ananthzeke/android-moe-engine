#!/usr/bin/env python3
"""Inspect an Android ARM64 artifact for baseline ISA and linkage requirements."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ALLOWED_NEEDED = {"liblog.so", "libm.so", "libdl.so", "libc.so", "libandroid.so"}
FORBIDDEN_INSTRUCTIONS = re.compile(
    r"\b(sdot|udot|smmla|ummla|bfdot|bfmmla|sve|ptrue|whilelt)\b")


def run(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(command)}\n{result.stderr}")
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--readelf", required=True)
    parser.add_argument("--objdump", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evidence: dict[str, object] = {
        "schema_version": 1,
        "gate": "gate1_android_elf",
        "binary": str(args.binary),
        "sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "size_bytes": args.binary.stat().st_size,
    }
    try:
        header = run([args.readelf, "-h", str(args.binary)])
        evidence["elf64"] = "ELF64" in header
        evidence["machine_aarch64"] = "AArch64" in header
        evidence["type"] = next((line.split(":", 1)[1].strip() for line in header.splitlines()
                                 if "Type:" in line), None)
        program = run([args.readelf, "-l", str(args.binary)])
        interpreter = next((line.strip() for line in program.splitlines() if "interpreter" in line), "")
        evidence["interpreter"] = interpreter
        dynamic = run([args.readelf, "-d", str(args.binary)])
        needed = re.findall(r"Shared library: \[(.*?)\]", dynamic)
        evidence["needed"] = needed
        attributes = run([args.readelf, "-A", str(args.binary)])
        evidence["attributes_present"] = bool(attributes.strip())
        disassembly = run([args.objdump, "-d", str(args.binary)])
        matches = sorted(set(FORBIDDEN_INSTRUCTIONS.findall(disassembly.lower())))
        evidence["forbidden_instructions"] = matches
        evidence["passed"] = (
            evidence["elf64"]
            and evidence["machine_aarch64"]
            and evidence["type"] in {"DYN (Shared object file)", "EXEC (Executable file)"}
            and "/system/bin/linker64" in interpreter
            and set(needed).issubset(ALLOWED_NEEDED)
            and not matches
        )
    except Exception as error:  # noqa: BLE001 - preserve the inspection failure
        evidence["error"] = str(error)
        evidence["passed"] = False

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
    return 0 if evidence.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
