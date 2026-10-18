"""Exercise the metadata-first GGUF index on a synthetic fixture."""

import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
tool = root / "tools/gguf_index.py"
TYPE_Q4_0 = 2
TYPE_F32 = 0


def u32(value):
    return struct.pack("<I", value)


def u64(value):
    return struct.pack("<Q", value)


def text(value):
    encoded = value.encode()
    return u64(len(encoded)) + encoded


def kv_string(key, value):
    return text(key) + u32(8) + text(value)


def kv_u32(key, value):
    return text(key) + u32(4) + u32(value)


def kv_string_array(key, values):
    payload = b"".join(text(value) for value in values)
    return text(key) + u32(9) + u32(8) + u64(len(values)) + payload


def tensor(name, dims, type_id, offset):
    return (text(name) + u32(len(dims)) + b"".join(u64(dim) for dim in dims) +
            u32(type_id) + u64(offset))


def nbytes(dims, block, element):
    assert dims[0] % block == 0
    total = (dims[0] // block) * element
    for dim in dims[1:]:
        total *= dim
    return total


with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "fixture.gguf"
    tensors = [
        ("blk.0.ffn_gate_exps.weight", [32, 16, 4], TYPE_Q4_0),
        ("blk.0.ffn_up_exps.weight", [32, 16, 4], TYPE_Q4_0),
        ("blk.0.ffn_down_exps.weight", [32, 16, 4], TYPE_Q4_0),
        ("blk.0.ffn_gate_shexp.weight", [32, 16], TYPE_Q4_0),
        ("blk.0.ffn_gate_inp_shexp.weight", [32], TYPE_F32),
        ("output.weight", [32, 16], TYPE_F32),
    ]
    offsets = []
    running = 0
    for _, dims, type_id in tensors:
        offsets.append(running)
        block, element = (32, 18) if type_id == TYPE_Q4_0 else (1, 4)
        running += nbytes(dims, block, element)
    metadata = (
        kv_string("general.architecture", "testmoe") +
        kv_u32("general.alignment", 32) +
        kv_u32("testmoe.expert_count", 4) +
        kv_u32("testmoe.expert_used_count", 2) +
        kv_string_array("general.tags", ["fixture"])
    )
    header = b"GGUF" + u32(3) + u64(len(tensors)) + u64(5) + metadata
    header += b"".join(tensor(name, dims, type_id, offset)
                       for (name, dims, type_id), offset in zip(tensors, offsets))
    data_start = (len(header) + 31) // 32 * 32
    file_size = data_start + running + 1024
    path.write_bytes(header + b"\0" * (data_start - len(header)) + b"\0" * (running + 1024))

    output = Path(directory) / "index.json"
    manifest = Path(directory) / "experts.manifest"
    result = subprocess.run([sys.executable, str(tool), "--input", str(path),
                             "--file-size", str(file_size),
                             "--mem-available-kib", "1",
                             "--storage-available-bytes", "1",
                             "--output", str(output),
                             "--manifest-out", str(manifest)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"GGUF index failed: {result.stdout}{result.stderr}")
    report = json.loads(output.read_text())
    if report["gguf"]["architecture"] != "testmoe":
        raise SystemExit("Architecture was not parsed")
    experts = report["experts"]
    if experts is None or experts["n_layer"] != 1 or experts["n_expert"] != 4:
        raise SystemExit(f"Unexpected expert index: {experts}")
    if experts["top_k"] != 2 or experts["expert_bytes_per_layer"] != 864:
        raise SystemExit(f"Unexpected routed bytes: {experts}")
    if experts["shared_expert_bytes_per_layer"] != 288:
        raise SystemExit(f"Unexpected shared bytes: {experts}")
    if experts["decode_live_bytes_per_layer"] != 2016:
        raise SystemExit(f"Unexpected live bytes: {experts}")
    if experts["shared_expert_tensor_count"] != 1:
        raise SystemExit("Shared router was counted as a matrix")
    if report["admission"]["decision"] != "refuse":
        raise SystemExit(f"Admission did not refuse: {report['admission']}")
    if report["admission"]["streaming_decode_live_bytes_estimate"] != 4192:
        raise SystemExit(f"Unexpected streaming estimate: {report['admission']}")
    manifest_lines = [line for line in manifest.read_text().splitlines() if line.startswith("tensor ")]
    if len(manifest_lines) != 3:
        raise SystemExit(f"Unexpected manifest entries: {manifest_lines}")
    fields = manifest_lines[0].split()
    if fields[1] != "blk.0.ffn_down_exps.weight" or int(fields[7]) != 288:
        raise SystemExit(f"Unexpected manifest row: {manifest_lines[0]}")

    truncated = Path(directory) / "truncated.gguf"
    truncated.write_bytes(header[:16])
    result = subprocess.run([sys.executable, str(tool), "--input", str(truncated),
                             "--file-size", str(file_size)],
                            capture_output=True, text=True)
    if result.returncode == 0 or "header_truncated" not in result.stderr:
        raise SystemExit("Truncated header was not refused")