#!/usr/bin/env python3
"""Print the startup stages and per-request phases from a request-series directory."""
import argparse
import glob
import json
import sys
from pathlib import Path


def value(counter):
    return "null" if counter is None else str(counter)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series_dir", type=Path)
    parser.add_argument("--health", type=Path, help="/health document holding the startup record")
    arguments = parser.parse_args()
    if arguments.health:
        startup = json.loads(arguments.health.read_text())["startup"]
        print(f"startup: first_ready {startup['first_ready_seconds']:.1f} s, hash "
              f"{startup.get('artifact_hash_seconds', float('nan')):.1f} s, manifest "
              f"{startup.get('manifest_validation_seconds', float('nan')):.3f} s")
        print("| stage | seconds | rchar | read_bytes | majflt | pswpin |\n| --- | ---: | ---: | ---: | ---: | ---: |")
        for stage in startup["stages"]:
            p = stage["proc_delta"]
            print(f"| {stage['name']} | {stage['seconds']:.2f} | {value(p['rchar'])} | "
                  f"{value(p['read_bytes'])} | {value(p['majflt'])} | {value(p['pswpin'])} |")
        print()
    print("| request | phase | seconds | rchar | read_bytes | majflt | pswpin | hits | misses | "
          "expert_bytes | pread_calls | read_seconds | big-core kHz start/end |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for path in sorted(glob.glob(str(arguments.series_dir / "*-[0-9][0-9]-*.json"))):
        document = json.loads(Path(path).read_text())
        record = (document["response"] or {}).get("ame") if isinstance(document["response"], dict) else None
        if not record:
            print(f"| {document['name']} | no record | | | | | | | | | | | |")
            continue
        for phase in record["phases"]:
            p, e = phase["proc_delta"], phase["expert_delta"] or {}
            start = phase["big_core_khz_start"][0]["khz"] if phase["big_core_khz_start"] else None
            end = phase["big_core_khz_end"][0]["khz"] if phase["big_core_khz_end"] else None
            print(f"| {document['name']} | {phase['name']} | {phase['seconds']:.2f} | "
                  f"{value(p['rchar'])} | {value(p['read_bytes'])} | {value(p['majflt'])} | "
                  f"{value(p['pswpin'])} | {e.get('hits')} | {e.get('misses')} | "
                  f"{e.get('expert_bytes')} | {e.get('pread_calls')} | "
                  f"{e.get('read_seconds', 0):.2f} | {start}/{end} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
