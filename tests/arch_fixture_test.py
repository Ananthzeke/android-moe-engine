"""Streamed-versus-resident parity for one tiny MoE architecture fixture.

The fixture comes from the pinned `test-llama-archs` with more experts than cache slots, so the
router has to choose and the pools have to evict. The float model and its Q4_K_M and Q6_K
quantizations each run through the shipped cache, the layered cache, the layered cache with
reader threads and the top-k overlap, and are compared with a resident run at the same prefill
unit (tokens and logits must be bit-identical). Capacity below the router's top-k must refuse.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--harness", type=Path, required=True)
parser.add_argument("--fixture", type=Path, required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--quantize", type=Path,
                    help="llama-quantize; without it only the float model is checked")
parser.add_argument("--slots", type=int, default=4)
parser.add_argument("--evidence", type=Path, help="write a JSON record of the comparisons")
arguments = parser.parse_args()

PROMPT = "The quick brown fox jumps over the lazy dog 0123456789 again and again"


def run(command, **kwargs):
    return subprocess.run([str(part) for part in command], capture_output=True, text=True, **kwargs)


def fail(message):
    raise SystemExit(f"{arguments.name}: {message}")


def check_variant(label, model, scratch):
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    manifest = scratch / f"{label}.manifest"
    index = run([sys.executable, ROOT / "tools" / "gguf_index.py", "--input", model,
                 "--file-size", model.stat().st_size, "--manifest-out", manifest,
                 "--artifact-sha256", digest, "--output", scratch / f"{label}.index.json"])
    if index.returncode != 0:
        fail(f"{label}: manifest failed: {index.stderr[-400:]}")
    layout = json.loads((scratch / f"{label}.index.json").read_text())["experts"]
    top_k, n_expert = layout["top_k"], layout["n_expert"]
    if n_expert <= arguments.slots:
        fail(f"{label}: {n_expert} experts do not exceed {arguments.slots} slots (not sparse)")

    def harness(name, streamed, *extra, unit=None):
        output, logits = scratch / f"{label}-{name}.json", scratch / f"{label}-{name}.logits"
        command = [arguments.harness, "--model", model, "--artifact-sha", digest, "--prompt", PROMPT,
                   "--max-new-tokens", 6, "--context", 128, "--threads", 2, "--temperature", 0,
                   "--seed", 1, "--logits-output", logits, "--output", output]
        if streamed:
            command += ["--stream", "--manifest", manifest, "--moe-arena-mib", 64, *extra]
        else:
            # No CPU extra buffer types: a streamed run cannot use repacked weights, so the
            # resident reference must run the same kernels.
            command += ["--batch-size", unit, "--no-extra-bufts"]
        result = run(command)
        if result.returncode != 0 or not output.is_file():
            fail(f"{label}/{name}: failed with {result.returncode}: {result.stderr[-600:]}")
        return json.loads(output.read_text()), logits.read_bytes()

    resident_cache = {}

    def resident(unit):
        if unit not in resident_cache:
            resident_cache[unit] = harness(f"resident-{unit}", False, unit=unit)
        return resident_cache[unit]

    rows = []
    variants = (
        ("legacy", ["--moe-slots", arguments.slots]),
        ("layered", ["--cache", "layered", "--layer-slots", arguments.slots]),
        ("layered-readers-overlap", ["--cache", "layered", "--layer-slots", arguments.slots,
                                     "--readers", 4, "--overlap", "topk"]),
    )
    for name, extra in variants:
        record, logits = harness(name, True, *extra)
        stream = record["streaming"]
        unit = record["options"]["prefill_unit_tokens"]
        reference, reference_logits = resident(unit)
        if len(reference["prompt_token_ids"]) < 8:
            fail(f"{label}: the prompt is too short to exercise multi-token prefill")
        if record["output"]["token_ids"] != reference["output"]["token_ids"]:
            fail(f"{label}/{name}: token IDs differ from the resident run at unit {unit}")
        if logits != reference_logits:
            fail(f"{label}/{name}: logits differ from the resident run at unit {unit}")
        if stream["evictions"] == 0:
            fail(f"{label}/{name}: the run did not evict, so it does not test sparse selection: {stream}")
        if "overlap" in name and (stream["prefetches_started"] == 0
                                  or stream["prefetch_fallbacks"] != 0):
            fail(f"{label}/{name}: the top-k overlap did not run: {stream}")
        rows.append({"variant": label, "cache": name, "prefill_unit": unit,
                     "token_ids": record["output"]["token_ids"], "tokens_equal": True,
                     "logits_equal": True, "misses": stream["misses"],
                     "evictions": stream["evictions"], "experts": n_expert, "top_k": top_k})

    if top_k > 1:
        for name, extra in (("legacy", ["--moe-slots", top_k - 1]),
                            ("layered", ["--cache", "layered", "--layer-slots", top_k - 1])):
            output = scratch / f"{label}-refuse-{name}.json"
            refused = run([arguments.harness, "--model", model, "--artifact-sha", digest, "--stream",
                           "--manifest", manifest, "--prompt", PROMPT, "--max-new-tokens", 2,
                           "--context", 128, "--moe-arena-mib", 64, "--batch-size", 64, *extra,
                           "--output", output])
            status = json.loads(output.read_text()).get("status") if output.is_file() else None
            if refused.returncode == 0 and status != "refused":
                fail(f"{label}: capacity below top-k was accepted by the {name} cache ({status})")
            rows.append({"variant": label, "cache": name, "refused_below_top_k": True})
    return rows


with tempfile.TemporaryDirectory() as scratch:
    scratch = Path(scratch)
    models = {"float": arguments.fixture}
    if arguments.quantize is not None:
        for label, kind in (("q4_k_m", "Q4_K_M"), ("q6_k", "Q6_K")):
            target = scratch / f"{label}.gguf"
            done = run([arguments.quantize, arguments.fixture, target, kind])
            if done.returncode != 0 or not target.is_file():
                fail(f"quantizing to {kind} failed: {done.stderr[-400:]}")
            models[label] = target
    records = []
    for label, model in models.items():
        records.extend(check_variant(label, model, scratch))
    summary = {"schema_version": 1, "architecture": arguments.name, "slots": arguments.slots,
               "fixture_sha256": hashlib.sha256(arguments.fixture.read_bytes()).hexdigest(),
               "variants": sorted(models), "rows": records}
    if arguments.evidence is not None:
        arguments.evidence.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"architecture": arguments.name, "variants": sorted(models),
                      "rows": len(records)}))
