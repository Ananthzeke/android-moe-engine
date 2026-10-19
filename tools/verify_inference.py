#!/usr/bin/env python3
"""Build and run the pinned-llama.cpp inference integration gates.

This is separate from the offline verifier: it requires an existing, verified
llama.cpp source tree and host static libraries. It never downloads anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs" / "dependencies.lock.json"
PATCHES = [ROOT / "patches" / "0001-ggml-moe-materializer.patch",
           ROOT / "patches" / "0002-llama-no-eager-mmap-prefetch.patch"]


def run(command: list[str], *, cwd: Path = ROOT, timeout: int = 900,
        check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True,
                            timeout=timeout, env=environment())
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n"
                           f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result


def environment() -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = str(ROOT / ".venv" / "bin") + os.pathsep + env.get("PATH", "")
    return env


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_patch_set() -> str:
    digest = hashlib.sha256()
    for path in PATCHES:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def pinned_commit() -> str:
    lock = json.loads(LOCK.read_text())
    for backend in lock.get("backends", []):
        if backend.get("id") == "llama.cpp":
            return backend["commit"]
    raise RuntimeError("llama.cpp is not pinned in configs/dependencies.lock.json")


def verify_source(source: Path) -> dict[str, object]:
    head = run(["git", "rev-parse", "HEAD"], cwd=source).stdout.strip()
    expected = pinned_commit()
    if head != expected:
        raise RuntimeError(f"llama.cpp HEAD {head} != pinned {expected}")
    diff = run(["git", "diff", "--binary"], cwd=source).stdout.encode()
    diff_hash = hashlib.sha256(diff).hexdigest()
    patch_hash = sha256_patch_set()
    if diff_hash != patch_hash:
        raise RuntimeError(f"applied llama.cpp diff {diff_hash} != patches {patch_hash}")
    return {"head": head, "diff_sha256": diff_hash, "patch_sha256": patch_hash,
            "source": str(source)}


def gate_stream(binary: Path, workdir: Path) -> dict[str, object]:
    workdir.mkdir(parents=True, exist_ok=True)
    result = run([str(binary), "--workdir", str(workdir)], timeout=300)
    payload = json.loads(result.stdout)
    if payload.get("identical") is not True:
        raise RuntimeError(f"stream gate not identical: {payload}")
    if payload.get("materialize_calls", 0) < 1 or payload.get("reads", 0) < 1:
        raise RuntimeError(f"stream gate did not exercise materialization: {payload}")
    return payload


def gate_faults(binary: Path, workdir: Path) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    cases = []
    for threads in (1, 2, 4):
        for fault in ("none", "read", "cancel", "capacity", "eof"):
            cases.append(("fault", fault, threads, 4, 4))
    cases.extend([
        ("capacity-minus-one", "none", 4, 4, 3),
        ("capacity-at", "none", 4, 4, 4),
        ("slots-below-top-k", "capacity", 4, 7, 8),
        ("slots-at-top-k", "none", 4, 8, 8),
        ("slots-above-top-k", "none", 4, 9, 8),
    ])
    for name, fault, threads, slots, selected in cases:
        case_dir = workdir / f"{name}_{fault}_{threads}_{slots}_{selected}"
        case_dir.mkdir(parents=True, exist_ok=True)
        command = [str(binary), "--workdir", str(case_dir), "--fault", fault,
                   "--threads", str(threads), "--slots", str(slots),
                   "--selected", str(selected)]
        result = run(command, timeout=300, check=False)
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"fault case {name} returned non-JSON: {result.stdout}\n"
                               f"stderr: {result.stderr}") from error
        if result.returncode != 0 or payload.get("passed") is not True:
            raise RuntimeError(f"fault case {name} threads={threads} slots={slots} "
                               f"selected={selected} failed: {payload} stderr={result.stderr}")
        checks.append({"name": name, "fault": fault, "threads": threads,
                       "slots": payload.get("slots"), "selected": selected,
                       "candidate_status": payload.get("candidate_status"),
                       "failed": payload.get("failed"), "cancelled": payload.get("cancelled"),
                       "identical": payload.get("identical")})
        if fault != "none":
            fresh_selected = min(selected, slots)
            fresh_dir = workdir / f"fresh_after_{name}_{threads}_{slots}"
            fresh_dir.mkdir(parents=True, exist_ok=True)
            fresh = run([str(binary), "--workdir", str(fresh_dir), "--fault", "none",
                         "--threads", str(threads), "--slots", str(slots),
                         "--selected", str(fresh_selected)], timeout=300, check=False)
            fresh_payload = json.loads(fresh.stdout)
            if fresh.returncode != 0 or fresh_payload.get("passed") is not True:
                raise RuntimeError(f"fresh request after {name} failed: {fresh_payload}")
            checks.append({"name": f"fresh_after_{name}", "fault": "none",
                           "threads": threads, "slots": fresh_payload.get("slots"),
                           "selected": fresh_selected,
                           "candidate_status": fresh_payload.get("candidate_status"),
                           "failed": fresh_payload.get("failed"),
                           "cancelled": fresh_payload.get("cancelled"),
                           "identical": fresh_payload.get("identical")})
    return checks


def gate_broken(binary: Path, workdir: Path) -> dict[str, object]:
    workdir.mkdir(parents=True, exist_ok=True)
    result = run([str(binary), "--workdir", str(workdir)], timeout=300, check=False)
    if result.returncode == 0:
        raise RuntimeError("broken materializer gate unexpectedly passed")
    payload = json.loads(result.stdout)
    if payload.get("identical") is not False:
        raise RuntimeError(f"broken gate did not report mismatch: {payload}")
    return {"exit_code": result.returncode, "identical": payload.get("identical"),
            "mismatched": payload.get("mismatched")}


def gate_kernel(binary: Path, workdir: Path, output: Path) -> dict[str, object]:
    workdir.mkdir(parents=True, exist_ok=True)
    run([str(binary), "--workdir", str(workdir), "--output", str(output)], timeout=300)
    payload = json.loads(output.read_text())
    rounds = payload.get("rounds", [])
    if not rounds or any(item.get("identical") is not True for item in rounds):
        raise RuntimeError(f"kernel gate not identical: {payload}")
    return {"dimensions": payload.get("dimensions"),
            "memory": payload.get("memory"), "rounds": len(rounds),
            "all_identical": True}


def build(config: str, source: Path, libs: Path, build_dir: Path) -> dict[str, object]:
    shutil.rmtree(build_dir, ignore_errors=True)
    configure = [
        "cmake", "-S", str(ROOT), "-B", str(build_dir), "-GNinja",
        "-DAME_INFERENCE_INTEGRATION=ON",
        f"-DAME_LLAMA_SOURCE_DIR={source}",
        f"-DAME_LLAMA_HOST_LIB_DIR={libs}",
        f"-DCMAKE_BUILD_TYPE={'Debug' if config != 'release' else 'Release'}",
        "-DBUILD_TESTING=OFF",
    ]
    if config == "sanitizers":
        configure.append("-DAME_SANITIZERS=ON")
    configured = run(configure, timeout=900)
    built = run(["cmake", "--build", str(build_dir), "--target",
                 "moe_stream_gate", "moe_stream_gate_broken", "moe_kernel_gate",
                 "manifest_gate", "llama_baseline_stream"], timeout=1800)
    return {
        "configure_returncode": configured.returncode,
        "build_returncode": built.returncode,
        "configure_tail": configured.stdout[-2000:] + configured.stderr[-2000:],
        "build_tail": built.stdout[-4000:] + built.stderr[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-source", type=Path,
                        default=Path(os.environ.get("AME_LLAMA_SOURCE_DIR", "")))
    parser.add_argument("--llama-libs", type=Path,
                        default=Path(os.environ.get("AME_LLAMA_HOST_LIB_DIR", "")))
    parser.add_argument("--configs", default="debug,release,sanitizers")
    parser.add_argument("--evidence", type=Path,
                        default=ROOT / "evidence" / "hardening" / "gate1-integration.json")
    parser.add_argument("--skip-sanitizers", action="store_true")
    args = parser.parse_args()

    if not str(args.llama_source) or not args.llama_source.is_dir():
        print("error: --llama-source or AME_LLAMA_SOURCE_DIR is required", file=sys.stderr)
        return 2
    if not str(args.llama_libs) or not args.llama_libs.is_dir():
        print("error: --llama-libs or AME_LLAMA_HOST_LIB_DIR is required", file=sys.stderr)
        return 2

    configs = [item.strip() for item in args.configs.split(",") if item.strip()]
    if args.skip_sanitizers:
        configs = [item for item in configs if item != "sanitizers"]
    evidence: dict[str, object] = {
        "schema_version": 1,
        "gate": "gate1_inference_integration",
        "llama": verify_source(args.llama_source),
        "libs": str(args.llama_libs),
        "configs": {},
    }
    start = time.monotonic()
    try:
        for config in configs:
            build_dir = ROOT / "build" / f"inference-{config}"
            record: dict[str, object] = {"build": build(config, args.llama_source,
                                                        args.llama_libs, build_dir)}
            record["stream"] = gate_stream(build_dir / "moe_stream_gate",
                                           build_dir / "gate-stream")
            record["faults"] = gate_faults(build_dir / "moe_stream_gate",
                                           build_dir / "gate-faults")
            record["broken"] = gate_broken(build_dir / "moe_stream_gate_broken",
                                           build_dir / "gate-broken")
            record["kernel"] = gate_kernel(build_dir / "moe_kernel_gate",
                                            build_dir / "gate-kernel",
                                            build_dir / "gate-kernel" / "result.json")
            manifest = run([sys.executable, str(ROOT / "tests" / "manifest_gate_test.py"),
                            "--binary", str(build_dir / "manifest_gate")], timeout=300)
            record["manifest"] = {"returncode": manifest.returncode,
                                   "stdout": manifest.stdout[-4000:],
                                   "stderr": manifest.stderr[-2000:]}
            evidence["configs"][config] = record
            print(f"{config}: stream/kernel/broken gates passed")
    except Exception as error:  # noqa: BLE001 - report the failing gate
        evidence["error"] = str(error)
        evidence["elapsed_seconds"] = time.monotonic() - start
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(json.dumps(evidence, indent=2) + "\n")
        print(f"inference verification failed: {error}", file=sys.stderr)
        return 1
    evidence["elapsed_seconds"] = time.monotonic() - start
    evidence["passed"] = True
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"wrote {args.evidence}")
    return 0


if __name__ == "__main__":
    sys.exit(main())