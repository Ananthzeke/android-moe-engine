#!/usr/bin/env python3
"""Validate the design scaffold. This does not validate an inference engine."""

import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    configs = {}
    for path in sorted((root / "configs").glob("*.json")):
        configs[path.stem] = json.loads(path.read_text())
    assert len(configs) == 3, "Expected the three reviewed configuration files"
    assert all(item["schema_version"] == 1 for item in configs.values())

    catalog = configs["model-candidates"]["models"]
    identifiers = [model["id"] for model in catalog]
    assert len(identifiers) == len(set(identifiers)), "Duplicate model IDs"
    for model in catalog:
        assert model["verified_on_target"] is False, "Attach evidence before changing scaffold support status"
        assert model["artifact_sha256"] is None, "The scaffold contains no verified artifact"
        assert urlsplit(model["source"]).scheme == "https"

    options = configs["mvp.example"]
    assert options["model_candidate"] in identifiers
    assert options["status"] == "proposed_not_implemented"
    assert options["execution"] == "cpu" and options["active_sessions"] == 1
    assert options["prefetch"] is False
    assert 0 < options["expert_cache_limit_mib"] < options["owned_memory_limit_mib"]
    assert 0 < options["max_new_tokens"] < options["context_tokens"] <= 2048
    assert options["read_chunk_limit_kib"] > 0 and options["io_queue_depth"] == 1
    assert configs["device-vivo-s1-pro"]["performance_measured"] is False

    markdown = [root / "README.md", root / "AGENTS.md"]
    markdown += sorted((root / "docs").glob("*.md"))
    markdown += sorted((root / "prompts").glob("*.md"))
    for path in markdown:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            parsed = urlsplit(target)
            if parsed.scheme or not parsed.path:
                continue
            destination = (path.parent / unquote(parsed.path)).resolve()
            assert destination.is_relative_to(root), f"Link escapes repository: {path.name}: {target}"
            assert destination.exists(), f"Missing link: {path.name}: {target}"
    print(f"Scaffold valid: {len(configs)} configurations, {len(catalog)} model candidates, {len(markdown)} Markdown files")
    print("Inference, model support, and phone performance remain untested.")


if __name__ == "__main__":
    main()
