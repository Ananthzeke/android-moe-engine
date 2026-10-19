"""Generate a small valid GGUF MoE fixture and its v2 expert manifest."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

TYPE_F32 = 0
NE0 = 32
NE1 = 16
N_EXPERT = 4
EXPERT_BYTES = NE0 * NE1 * 4
TENSOR_BYTES = EXPERT_BYTES * N_EXPERT
NAMES = [f"blk.0.ffn_{role}_exps.weight" for role in ("gate", "up", "down")]


def u32(value: int) -> bytes:
    return struct.pack("<I", value)


def u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def string(value: str) -> bytes:
    encoded = value.encode()
    return u64(len(encoded)) + encoded


def metadata_string(key: str, value: str) -> bytes:
    return string(key) + u32(8) + string(value)


def metadata_u32(key: str, value: int) -> bytes:
    return string(key) + u32(4) + u32(value)


def fnv1a64(text: str) -> str:
    value = 0xCBF29CE484222325
    for byte in text.encode():
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


DATA_OFFSET = 0


def manifest_rows(offsets=None):
    offsets = offsets or [DATA_OFFSET + index * TENSOR_BYTES for index in range(len(NAMES))]
    return [(name, 0, "role", TYPE_F32, "f32", N_EXPERT, EXPERT_BYTES,
             EXPERT_BYTES, offset, TENSOR_BYTES)
            for name, offset in zip(NAMES, offsets)]


def manifest_text(rows, artifact_bytes: int, artifact_sha: str) -> str:
    canonical = "\n".join("|".join(str(value) for value in row) for row in rows)
    lines = [
        "# ame-expert-manifest 2",
        f"# artifact_bytes {artifact_bytes}",
        f"# artifact_sha256 {artifact_sha}",
        f"# tensor_count {len(rows)}",
        f"# layout_fnv1a64 {fnv1a64(canonical)}",
    ]
    lines.extend("tensor " + " ".join(str(value) for value in row) for row in rows)
    return "\n".join(lines) + "\n"


def write_fixture(directory: Path) -> tuple[Path, Path, str]:
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / "fixture.gguf"
    metadata = (
        metadata_string("general.architecture", "testmoe") +
        metadata_u32("testmoe.expert_count", N_EXPERT) +
        metadata_u32("testmoe.expert_used_count", 2) +
        metadata_u32("general.alignment", 32)
    )
    header = b"GGUF" + u32(3) + u64(len(NAMES)) + u64(4) + metadata
    for index, name in enumerate(NAMES):
        header += string(name) + u32(3) + u64(NE0) + u64(NE1) + u64(N_EXPERT)
        header += u32(TYPE_F32) + u64(index * TENSOR_BYTES)
    while len(header) % 32 != 0:
        header += b"\x00"
    payload = bytearray()
    for tensor in range(len(NAMES)):
        for expert in range(N_EXPERT):
            for value in range(NE0 * NE1):
                payload += struct.pack("<f", float((tensor * 131 + expert * 17 + value) % 1009) / 1009.0)
    global DATA_OFFSET
    DATA_OFFSET = len(header)
    model.write_bytes(header + bytes(payload))
    sha = hashlib.sha256(model.read_bytes()).hexdigest()
    manifest = directory / "fixture.manifest"
    manifest.write_text(manifest_text(manifest_rows(), model.stat().st_size, sha))
    return model, manifest, sha