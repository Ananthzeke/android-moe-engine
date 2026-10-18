#!/usr/bin/env python3
"""Offline host acceptance checks; bootstrap tools separately."""

import argparse
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def tool(name):
    sibling = Path(sys.executable).parent / name
    found = str(sibling) if sibling.is_file() else shutil.which(name)
    if not found:
        raise RuntimeError(f"Missing {name}; install requirements-dev.txt as documented in CONTRIBUTING.md")
    return found


def run(command, env=None):
    print("+ " + " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", action="store_true", help="Fix C++ formatting only")
    options = parser.parse_args()
    formatter = tool("clang-format")
    version = subprocess.check_output([formatter, "--version"], text=True)
    if not re.search(r"version 18\.1\.8\b", version):
        raise RuntimeError("clang-format 18.1.8 required; install requirements-dev.txt")
    sources = sorted(str(p.relative_to(ROOT)) for base in ("src", "apps", "tests")
                     for p in (ROOT / base).rglob("*") if p.suffix in (".cpp", ".hpp"))
    run([formatter, *(["-i"] if options.format else ["--dry-run", "--Werror"]), *sources])
    if options.format:
        return
    cmake, ninja = tool("cmake"), tool("ninja")
    run([sys.executable, "tools/check_scaffold.py"])
    environment = os.environ.copy()
    environment["PATH"] = str(Path(ninja).parent) + os.pathsep + environment.get("PATH", "")
    for preset in ("host-debug", "host-release", "host-sanitizers"):
        run([cmake, "--preset", preset, f"-DCMAKE_MAKE_PROGRAM={ninja}",
             f"-DPython3_EXECUTABLE={sys.executable}"], environment)
        run([cmake, "--build", "--preset", preset], environment)
        run([str(Path(cmake).with_name("ctest")), "--preset", preset], environment)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        sys.exit(1)
