#!/usr/bin/env python3
"""Validate repository links and configuration consistency, not inference claims."""

import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit


def require(condition, message):
    if not condition:
        raise ValueError(message)


def evidence(root, references):
    require(isinstance(references, list) and bool(references), "Evidence references required")
    for reference in references:
        require(isinstance(reference, str) and bool(reference.strip()), "Empty evidence reference")
        parsed = urlsplit(reference)
        require(not parsed.scheme and bool(parsed.path), "Evidence must name a committed local record")
        path = (root / unquote(parsed.path)).resolve()
        require(path.is_relative_to(root) and path.is_file() and path.stat().st_size > 0,
                f"Missing or empty evidence: {reference}")


def validate(root):
    configs = {path.stem: json.loads(path.read_text())
               for path in sorted((root / "configs").glob("*.json"))}
    required = {"model-candidates", "mvp.example", "device-vivo-s1-pro"}
    require(required <= configs.keys(), "Missing required configuration")
    for name in required:
        require(configs[name]["schema_version"] == 1, f"Unsupported schema: {name}")
    catalog = configs["model-candidates"]["models"]
    identifiers = [model["id"] for model in catalog]
    require(all(isinstance(value, str) and value for value in identifiers), "Invalid model ID")
    require(len(identifiers) == len(set(identifiers)), "Duplicate model IDs")
    for model in catalog:
        require(type(model["verified_on_target"]) is bool, "Support status must be boolean")
        require(urlsplit(model["source"]).scheme == "https", "Model source must be HTTPS")
        digest = model["artifact_sha256"]
        require(digest is None or (isinstance(digest, str) and re.fullmatch(r"[0-9a-fA-F]{64}", digest)),
                "Artifact SHA must be null or 64 hexadecimal characters")
        if model["verified_on_target"]:
            require(digest is not None, "Verified model requires artifact SHA")
            evidence(root, model.get("evidence"))
    options = configs["mvp.example"]
    require(options["model_candidate"] in identifiers, "Unknown model candidate")
    require(isinstance(options["status"], str) and bool(options["status"]), "Missing config status")
    require(options["execution"] == "cpu" and options["active_sessions"] == 1, "MVP is CPU/single session")
    require(options["prefetch"] is False, "MVP requires demand reads")
    require(0 < options["expert_cache_limit_mib"] < options["owned_memory_limit_mib"], "Invalid memory limits")
    require(0 < options["max_new_tokens"] < options["context_tokens"] <= 2048, "Invalid token limits")
    require(options["read_chunk_limit_kib"] > 0 and options["io_queue_depth"] == 1, "Invalid I/O limits")
    device = configs["device-vivo-s1-pro"]
    require(type(device["performance_measured"]) is bool, "Performance status must be boolean")
    if device["performance_measured"]:
        evidence(root, device.get("evidence"))
    markdown = sorted(path for path in root.rglob("*.md")
                      if not any(part.startswith(".") or part in {"build", "third_party"}
                                 for part in path.relative_to(root).parts))
    for path in markdown:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            parsed = urlsplit(target)
            if parsed.scheme or not parsed.path:
                continue
            destination = (path.parent / unquote(parsed.path)).resolve()
            require(destination.is_relative_to(root), f"Link escapes repository: {path}: {target}")
            require(destination.exists(), f"Missing link: {path}: {target}")
    print(f"Repository valid: {len(configs)} configurations, {len(catalog)} candidates, {len(markdown)} Markdown files")
    print("This check validates references and consistency; it does not establish inference or target support.")


if __name__ == "__main__":
    try:
        validate(Path(__file__).resolve().parents[1])
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"Repository validation failed: {error}", file=sys.stderr)
        sys.exit(1)
