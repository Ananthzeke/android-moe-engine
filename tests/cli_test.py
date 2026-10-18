import subprocess
import sys

binary = sys.argv[1]
for arguments in ([], ["--help"], ["--version"]):
    result = subprocess.run([binary, *arguments], capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip() or result.stderr:
        raise SystemExit(f"CLI success contract failed: {arguments}: {result}")
for arguments in (["run"], ["--unknown"], ["--help", "extra"], ["--version", "--help"]):
    result = subprocess.run([binary, *arguments], capture_output=True, text=True)
    if result.returncode == 0 or not result.stderr.strip() or result.stdout:
        raise SystemExit(f"CLI rejection contract failed: {arguments}: {result}")
