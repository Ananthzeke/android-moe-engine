"""Check tools/replay_routing.py against the real Store and its own accounting."""
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from replay_routing import Refused, StoreSim, layer_units, parse_trace, run  # noqa: E402

driver = sys.argv[1]


def real_store(directory, capacity, steps):
    scenario = Path(directory) / "scenario.txt"
    scenario.write_text(f"capacity {capacity}\n" + "\n".join(" ".join(map(str, s)) for s in steps)
                        + "\n")
    result = subprocess.run([driver, str(scenario)], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"store driver failed: {result}")
    rows = []
    for line in result.stdout.splitlines():
        rows.append(None if line == "refused" else [int(x) for x in line.split()])
    return rows


def simulated(capacity, steps):
    sim = StoreSim(capacity)
    rows = []
    for step in steps:
        before = (sim.hits, sim.misses, sim.evictions)
        try:
            _, _, slots = sim.acquire(step)
        except Refused:
            rows.append(None)
            continue
        rows.append([sim.hits - before[0], sim.misses - before[1], sim.evictions - before[2],
                     *slots.values()])
    return rows


with tempfile.TemporaryDirectory() as directory:
    # The case from the review: capacity two, resident oldest to newest [A, B], then [C, A].
    # The real Store pins A as a hit, evicts B, and records one hit.
    a, b, c = 0, 1, 2
    steps = [[a], [b], [c, a]]
    real = real_store(directory, 2, steps)
    sim = simulated(2, steps)
    if real[2][:3] != [1, 1, 1]:
        raise SystemExit(f"real Store did not keep A and evict B: {real}")
    if real != sim:
        raise SystemExit(f"simulator differs from the real Store: {real} != {sim}")
    # A simulator that evicts before pinning hits would report zero hits here.

    # Random scenarios: duplicates, refusals (more distinct keys than slots), and reuse.
    generator = random.Random(1234)
    for case in range(60):
        capacity = generator.randint(1, 7)
        universe = generator.randint(capacity, capacity + 6)
        steps = [[generator.randrange(universe) for _ in range(generator.randint(1, capacity + 2))]
                 for _ in range(generator.randint(5, 40))]
        real = real_store(directory, capacity, steps)
        sim = simulated(capacity, steps)
        if real != sim:
            raise SystemExit(f"case {case} (capacity {capacity}): real {real} != simulated {sim}")

    # Trace parsing: a format-2 trace, replayed through every policy.
    trace = Path(directory) / "trace.txt"
    lines = ["# ame-moe-trace 2", "# columns: call request graph phase ubatch tensor n_experts experts..."]
    call = 0
    for graph, phase in enumerate(["prefill", "decode", "decode", "decode"], start=1):
        ubatch = 4 if phase == "prefill" else 1
        for layer in range(3):
            for role in ("gate", "up", "down"):
                experts = [(layer + graph + k) % 6 for k in range(2)]
                lines.append(f"{call} 1 {graph} {phase} {ubatch} blk.{layer}.ffn_{role}_exps.weight "
                             f"{len(experts)} {' '.join(map(str, experts))}")
                call += 1
    trace.write_text("\n".join(lines) + "\n")
    rows = parse_trace(trace)
    if len(rows) != 36 or rows[0]["phase"] != "prefill" or rows[-1]["phase"] != "decode":
        raise SystemExit("trace labels were not parsed")
    units = layer_units(rows)
    if len(units) != 12 or units[0]["roles"] != ["gate", "up", "down"]:
        raise SystemExit("roles of one layer were not grouped")
    results = run(rows, ["store_lru", "shared_layer_lru", "per_layer_lru", "per_layer_lfu",
                         "per_layer_lru_protect", "per_layer_lfu_protect"], [12, 48], 100, 4)
    by_name = {(r["policy"], r["budget_slices"]): r for r in results}
    for key, item in by_name.items():
        for phase in ("prefill", "decode"):
            data = item["phases"][phase]
            if data["hits"] + data["misses"] != data["lookups"]:
                raise SystemExit(f"{key} {phase}: hits and misses do not add up")
            if data["bytes_read"] > data["bytes_needed"] * 3 + 1:
                raise SystemExit(f"{key} {phase}: read more than three times what was needed")
    # Gate, up and down share one slot: the same cache holds three times the experts per byte.
    big = by_name[("per_layer_lru", 48)]["phases"]["decode"]
    small = by_name[("per_layer_lru", 12)]["phases"]["decode"]
    if big["bytes_read"] > small["bytes_read"]:
        raise SystemExit("a larger budget read more bytes")
    for name in ("per_layer_lru", "per_layer_lfu"):
        if by_name[(name, 48)]["phases"] != by_name[(name + "_protect", 48)]["phases"]:
            raise SystemExit(f"{name}: the protect variant differs from the plain policy")
    json.dumps(results)
