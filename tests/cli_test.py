import json
import subprocess
import sys
import tempfile
from pathlib import Path

binary = sys.argv[1]
for arguments in ([], ["--help"], ["--version"]):
    result = subprocess.run([binary, *arguments], capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip() or result.stderr:
        raise SystemExit(f"CLI success contract failed: {arguments}: {result}")
for arguments in (["run"], ["--unknown"], ["--help", "extra"], ["--version", "--help"],
                  ["probe", "--bogus"], ["probe", "--output"], ["probe", "--root", ""]):
    result = subprocess.run([binary, *arguments], capture_output=True, text=True)
    if result.returncode == 0 or not result.stderr.strip() or result.stdout:
        raise SystemExit(f"CLI rejection contract failed: {arguments}: {result}")

probe = subprocess.run([binary, "probe"], capture_output=True, text=True)
if probe.returncode != 0 or probe.stderr:
    raise SystemExit(f"Host probe failed: {probe}")
snapshot = json.loads(probe.stdout)
if snapshot["schema_version"] != 1:
    raise SystemExit("Unexpected probe schema version")
if snapshot["execution_tag"] != "host":
    raise SystemExit(f"Host probe mislabelled execution: {snapshot['execution_tag']}")
if not snapshot["observed_at"].endswith("Z"):
    raise SystemExit("Probe timestamp is not UTC")
if not isinstance(snapshot["unavailable"], list):
    raise SystemExit("Probe unavailable list missing")
for invalid in ({"os.machine": None}, {"cpu.features": None}):
    for field in invalid:
        section, key = field.split(".")
        if snapshot[section][key] is not None and not snapshot[section][key]:
            raise SystemExit(f"Empty optional observation: {field}")
with tempfile.TemporaryDirectory() as directory:
    destination = Path(directory) / "device.json"
    written = subprocess.run([binary, "probe", "--output", str(destination)],
                             capture_output=True, text=True)
    if written.returncode != 0 or written.stderr or not destination.is_file():
        raise SystemExit(f"Probe output contract failed: {written}")
    recorded = json.loads(destination.read_text())
    if recorded["schema_version"] != 1:
        raise SystemExit("Written probe record is invalid")
