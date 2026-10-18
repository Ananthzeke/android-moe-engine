"""Exercise validator behavior with temporary repository records."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as directory:
    copy = Path(directory)
    shutil.copytree(root / "configs", copy / "configs")
    (copy / "tools").mkdir()
    shutil.copy(root / "tools/check_scaffold.py", copy / "tools/check_scaffold.py")
    (copy / "README.md").write_text("[Anchor](#local) [Web](https://example.com)\n")
    (copy / "configs/extra.json").write_text('{"future_schema": 2}\n')

    def check(accepted):
        result = subprocess.run([sys.executable, "-O", str(copy / "tools/check_scaffold.py")],
                                capture_output=True, text=True)
        if (result.returncode == 0) != accepted:
            raise SystemExit(f"Unexpected validator outcome: {result.stdout}{result.stderr}")

    path = copy / "configs/model-candidates.json"
    catalog = json.loads(path.read_text())
    check(True)
    model = catalog["models"][0]
    model["verified_on_target"] = True
    path.write_text(json.dumps(catalog))
    check(False)
    model["artifact_sha256"] = "not-a-hash"
    model["evidence"] = ["evidence.md"]
    path.write_text(json.dumps(catalog))
    check(False)
    model["artifact_sha256"] = "a" * 64
    path.write_text(json.dumps(catalog))
    check(False)
    (copy / "evidence.md").write_text("Temporary fixture record, not actual model support.\n")
    check(True)
    catalog["models"].append(dict(model))
    path.write_text(json.dumps(catalog))
    check(False)
