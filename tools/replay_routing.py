#!/usr/bin/env python3
"""Replay an AME_MOE_TRACE routing trace against expert-cache policies.

The baseline policy `store_lru` reproduces `ame::residency::Store::acquire`: hits are
pinned before any victim is chosen, free slots are used first, the victim is the unpinned
resident slot with the oldest tick, and a lease is released (stamping new ticks) just before
the next acquire. Its keys are (tensor, expert) and every slot holds one slice, which is the
shipped design. `tests/replay_routing_test.py` checks it against the real Store.

The candidate policies key on (layer, expert); the gate, up and down slices of one expert
share one slot and are read together. A replay only reproduces routing decisions that were
already made; it never changes which experts are computed.

Protect-until-consumed: the roles of one layer form a single acquisition, so every slot
loaded or hit for the layer is pinned until the layer's lease is released after it finishes.
The `_protect` variants state that constraint explicitly. Because gate, up and down use the
same expert IDs they cannot differ from the plain variants in a replay; equal results are
expected, and a difference would indicate a bug.

Counts are per phase (`prefill`, `decode`, or `unknown` for a format-1 trace). `slice hit
rate` is the share of needed slice bytes served from memory, so policies with different slot
sizes stay comparable. `bytes_read` is application bytes, not physical flash reads.
"""
import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

TENSOR = re.compile(r"blk\.(\d+)\.ffn_(gate_up|gate|up|down)_exps")
SLICE_BYTES = 589_824
DEFAULT_SLICES = (960, 1920, 2880, 3840)


class Refused(Exception):
    """An acquisition that cannot be admitted: more misses than evictable slots."""


class StoreSim:
    """Mirror of Store::acquire with tick-based LRU eviction (or LFU for candidates).

    Slots are kept in recency order (oldest first); a lease's slots move to the newest end
    when it is released, in pin order, which is the order Store stamps their ticks.
    """

    def __init__(self, slots, policy="lru", halve_every=0):
        self.slots = slots
        self.entry = [None] * slots
        self.pins = [0] * slots
        self.freq = [0] * slots
        self.where = {}
        self.free = list(range(slots))  # ascending: Store fills the lowest free slot first
        self.order = collections.OrderedDict()  # resident slots, least recently used first
        self.policy = policy
        self.halve_every = halve_every
        self.acquires = 0
        self.lease = []
        self.hits = self.misses = self.evictions = 0

    def release(self):
        for slot in self.lease:
            if self.pins[slot] > 0:
                self.pins[slot] -= 1
            self.order.move_to_end(slot)
        self.lease = []

    def _victim(self, taken):
        best, best_frequency = None, None
        for slot in self.order:
            if slot in taken or self.pins[slot] != 0:
                continue
            if self.policy != "lfu":
                return slot
            if best is None or self.freq[slot] < best_frequency:
                best, best_frequency = slot, self.freq[slot]
        return best

    def acquire(self, keys):
        """Return (hit_keys, miss_keys, slot per unique key). Raises Refused."""
        self.release()
        unique = list(dict.fromkeys(keys))
        hit_keys = [k for k in unique if k in self.where]
        miss_keys = [k for k in unique if k not in self.where]
        if len(miss_keys) > self.slots:
            raise Refused
        taken = {self.where[k] for k in hit_keys}
        used_free = min(len(miss_keys), len(self.free))
        chosen = self.free[:used_free]
        victims = []
        while len(chosen) + len(victims) < len(miss_keys):
            victim = self._victim(taken | set(victims))
            if victim is None:
                raise Refused
            victims.append(victim)
        chosen = chosen + victims
        del self.free[:used_free]
        self.acquires += 1
        if self.halve_every and self.acquires % self.halve_every == 0:
            self.freq = [count // 2 for count in self.freq]
        pins = []
        for key in hit_keys:
            slot = self.where[key]
            self.pins[slot] += 1
            self.freq[slot] += 1
            pins.append(slot)
            self.hits += 1
        for key, slot in zip(miss_keys, chosen):
            old = self.entry[slot]
            if old is not None:
                del self.where[old]
                self.evictions += 1
            self.entry[slot] = key
            self.where[key] = slot
            self.pins[slot] = 1
            self.freq[slot] = 1
            self.order[slot] = None
            self.order.move_to_end(slot)
            pins.append(slot)
            self.misses += 1
        self.lease = pins
        return hit_keys, miss_keys, {k: self.where[k] for k in unique}


def parse_trace(path):
    """Return rows: dicts with request, graph, phase, ubatch, layer, role, tensor, experts."""
    rows = []
    version = 1
    last_layer = -1
    graph = 0
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            if line.startswith("# ame-moe-trace"):
                version = int(line.split()[2])
            continue
        parts = line.split()
        if not parts:
            continue
        if version >= 2:
            call, request, graph_id, phase, ubatch, tensor, count = parts[:7]
            experts = [int(x) for x in parts[7:]]
            request, graph, ubatch = int(request), int(graph_id), int(ubatch)
        else:
            call, tensor, count = parts[:3]
            experts = [int(x) for x in parts[3:]]
            phase, request, ubatch = "unknown", 0, 0
            match = TENSOR.search(tensor)
            layer_now = int(match.group(1)) if match else 0
            if layer_now < last_layer:
                graph += 1
            last_layer = layer_now
        if int(count) != len(experts):
            raise ValueError(f"{path}: call {call} declares {count} experts, has {len(experts)}")
        match = TENSOR.search(tensor)
        if match is None:
            raise ValueError(f"{path}: unrecognized tensor name {tensor}")
        rows.append({"request": request, "graph": graph, "phase": phase, "ubatch": ubatch,
                     "layer": int(match.group(1)), "role": match.group(2), "tensor": tensor,
                     "experts": experts})
    return rows


def layer_units(rows):
    """Group consecutive rows of one (request, graph, layer) into one acquisition."""
    units = []
    for row in rows:
        key = (row["request"], row["graph"], row["layer"])
        if units and units[-1]["key"] == key:
            units[-1]["roles"].append(row["role"])
            units[-1]["experts"].extend(row["experts"])
        else:
            units.append({"key": key, "phase": row["phase"], "ubatch": row["ubatch"],
                          "layer": row["layer"], "roles": [row["role"]],
                          "experts": list(row["experts"]), "slice_accesses": 0})
        units[-1]["slice_accesses"] += len(set(row["experts"]))
    for unit in units:
        unit["experts"] = list(dict.fromkeys(unit["experts"]))
    return units


class Tally:
    def __init__(self):
        self.lookups = self.hits = self.misses = 0
        self.bytes_read = self.bytes_needed = 0
        self.refused = 0
        self.graphs = set()
        self.tokens = 0

    def result(self):
        rate = self.hits / self.lookups if self.lookups else None
        slice_rate = 1 - self.bytes_read / self.bytes_needed if self.bytes_needed else None
        return {"lookups": self.lookups, "hits": self.hits, "misses": self.misses,
                "hit_rate": rate, "bytes_read": self.bytes_read,
                "bytes_needed": self.bytes_needed, "slice_hit_rate": slice_rate,
                "refused_calls": self.refused, "graphs": len(self.graphs),
                "bytes_read_per_graph": (self.bytes_read / len(self.graphs)
                                         if self.graphs else None)}


def replay_store(rows, slots, slice_bytes):
    """Baseline: one pool keyed by (tensor, expert), one slice per slot, one acquire per row."""
    sim = StoreSim(slots)
    tallies = collections.defaultdict(Tally)
    for row in rows:
        tally = tallies[row["phase"]]
        tally.graphs.add((row["request"], row["graph"]))
        keys = [(row["tensor"], e) for e in row["experts"]]
        unique = len(set(keys))
        tally.bytes_needed += unique * slice_bytes
        try:
            hits, misses, _ = sim.acquire(keys)
        except Refused:
            tally.refused += 1
            tally.lookups += unique
            tally.misses += unique
            tally.bytes_read += unique * slice_bytes
            continue
        tally.lookups += unique
        tally.hits += len(hits)
        tally.misses += len(misses)
        tally.bytes_read += len(misses) * slice_bytes
    return tallies


def replay_layer(rows, policy, budget_slices, slice_bytes, halve_every):
    """Candidates keyed by (layer, expert); one slot holds every role's slice."""
    units = layer_units(rows)
    layers = sorted({u["layer"] for u in units})
    roles = max((len(set(u["roles"])) for u in units), default=3)
    slot_bytes = roles * slice_bytes
    total_slots = budget_slices * slice_bytes // slot_bytes
    protect = policy.endswith("_protect")
    base = policy[: -len("_protect")] if protect else policy
    eviction = "lfu" if base.endswith("lfu") else "lru"
    shared = base.startswith("shared")
    per_layer = max(1, total_slots // len(layers)) if layers else 0
    pools = {}
    tallies = collections.defaultdict(Tally)
    for unit in units:
        tally = tallies[unit["phase"]]
        tally.graphs.add(unit["key"][:2])
        pool_key = "all" if shared else unit["layer"]
        if pool_key not in pools:
            pools[pool_key] = StoreSim(total_slots if shared else per_layer, eviction,
                                       halve_every)
        sim = pools[pool_key]
        keys = [(unit["layer"], e) for e in unit["experts"]]
        needed = unit["slice_accesses"]
        tally.bytes_needed += needed * slice_bytes
        tally.lookups += len(keys)
        try:
            hits, misses, _ = sim.acquire(keys)
        except Refused:
            tally.refused += 1
            tally.misses += len(keys)
            tally.bytes_read += len(keys) * slot_bytes
            continue
        tally.hits += len(hits)
        tally.misses += len(misses)
        tally.bytes_read += len(misses) * slot_bytes
    for sim in pools.values():
        sim.release()
    return tallies, {"total_slots": total_slots,
                     "slots_per_layer": None if shared else per_layer,
                     "slot_bytes": slot_bytes, "protect_until_consumed": protect}


POLICIES = ("store_lru", "shared_layer_lru", "per_layer_lru", "per_layer_lfu",
            "per_layer_lru_protect", "per_layer_lfu_protect")


def run(rows, policies, budgets, slice_bytes, halve_every):
    results = []
    for budget in budgets:
        for policy in policies:
            if policy == "store_lru":
                tallies = replay_store(rows, budget, slice_bytes)
                extra = {"total_slots": budget, "slots_per_layer": None,
                         "slot_bytes": slice_bytes, "protect_until_consumed": False}
            else:
                tallies, extra = replay_layer(rows, policy, budget, slice_bytes, halve_every)
            results.append({"policy": policy, "budget_slices": budget,
                            "budget_bytes": budget * slice_bytes, **extra,
                            "phases": {phase: tally.result()
                                       for phase, tally in sorted(tallies.items())}})
    return results


def markdown(results, phase):
    lines = [f"| policy | budget GB | slots | {phase} slice hit rate | {phase} bytes read per graph (MB) |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for item in results:
        data = item["phases"].get(phase)
        if data is None:
            continue
        rate = "n/a" if data["slice_hit_rate"] is None else f"{data['slice_hit_rate'] * 100:.1f}%"
        per = "n/a" if data["bytes_read_per_graph"] is None else f"{data['bytes_read_per_graph'] / 1e6:.1f}"
        lines.append(f"| {item['policy']} | {item['budget_bytes'] / 1e9:.2f} | "
                     f"{item['total_slots']} | {rate} | {per} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--slice-bytes", type=int, default=SLICE_BYTES)
    parser.add_argument("--budget-slices", default=",".join(str(x) for x in DEFAULT_SLICES),
                        help="budgets in slice-sized units (default 0.57/1.13/1.70/2.26 GB)")
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--halve-every", type=int, default=32,
                        help="LFU: halve a pool's counts every N acquisitions of that pool")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    arguments = parser.parse_args()
    rows = parse_trace(arguments.trace)
    budgets = [int(x) for x in arguments.budget_slices.split(",")]
    policies = arguments.policies.split(",")
    for policy in policies:
        if policy not in POLICIES:
            parser.error(f"unknown policy {policy}")
    results = run(rows, policies, budgets, arguments.slice_bytes, arguments.halve_every)
    phases = collections.Counter(row["phase"] for row in rows)
    document = {
        "schema_version": 1,
        "trace": {"path": str(arguments.trace),
                  "sha256": hashlib.sha256(arguments.trace.read_bytes()).hexdigest(),
                  "rows": len(rows), "rows_by_phase": dict(phases),
                  "graphs": len({(r["request"], r["graph"]) for r in rows}),
                  "layers": len({r["layer"] for r in rows})},
        "parameters": {"slice_bytes": arguments.slice_bytes, "halve_every": arguments.halve_every,
                       "budget_slices": budgets},
        "results": results,
    }
    if arguments.json:
        arguments.json.write_text(json.dumps(document, indent=2) + "\n")
    tables = []
    for phase in sorted(phases):
        tables.append(f"### {phase}\n\n{markdown(results, phase)}\n")
    text = "\n".join(tables)
    if arguments.markdown:
        arguments.markdown.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
