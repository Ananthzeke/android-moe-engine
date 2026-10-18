#!/usr/bin/env python3
"""Metadata-first GGUF index and admission preflight.

Reads a prefix of a GGUF file, parses metadata and tensor information, groups
expert tensors, and checks declared file bounds. It never loads tensor payloads.
"""

import argparse
import hashlib
import json
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1

# ggml type IDs used by the pinned llama.cpp. Unknown IDs fail the index instead of
# guessing a block size.
TYPE_TABLE = {
    0: ("f32", 1, 4), 1: ("f16", 1, 2), 2: ("q4_0", 32, 18), 3: ("q4_1", 32, 20),
    6: ("q5_0", 32, 22), 7: ("q5_1", 32, 24), 8: ("q8_0", 32, 34), 9: ("q8_1", 32, 36),
    10: ("q2_K", 256, 84), 11: ("q3_K", 256, 110), 12: ("q4_K", 256, 144),
    13: ("q5_K", 256, 176), 14: ("q6_K", 256, 210), 15: ("q8_K", 256, 292),
    16: ("iq2_xxs", 256, 66), 17: ("iq2_xs", 256, 74), 18: ("iq3_xxs", 256, 98),
    19: ("iq1_s", 256, 50), 20: ("iq4_nl", 32, 18), 21: ("iq3_s", 256, 110),
    22: ("iq2_s", 256, 82), 23: ("iq4_xs", 256, 136), 24: ("i8", 1, 1),
    25: ("i16", 1, 2), 26: ("i32", 1, 4), 27: ("i64", 1, 8), 28: ("f64", 1, 8),
    29: ("iq1_m", 256, 56), 30: ("bf16", 1, 2), 34: ("tq1_0", 256, 54),
    35: ("tq2_0", 256, 66), 39: ("mxfp4", 32, 17), 40: ("nvfp4", 64, 36),
}

VALUE_UINT8, VALUE_INT8, VALUE_UINT16, VALUE_INT16 = 0, 1, 2, 3
VALUE_UINT32, VALUE_INT32, VALUE_FLOAT32, VALUE_BOOL = 4, 5, 6, 7
VALUE_STRING, VALUE_ARRAY, VALUE_UINT64, VALUE_INT64, VALUE_FLOAT64 = 8, 9, 10, 11, 12
SCALAR_FORMATS = {
    VALUE_UINT8: "<B", VALUE_INT8: "<b", VALUE_UINT16: "<H", VALUE_INT16: "<h",
    VALUE_UINT32: "<I", VALUE_INT32: "<i", VALUE_FLOAT32: "<f", VALUE_BOOL: "<B",
    VALUE_UINT64: "<Q", VALUE_INT64: "<q", VALUE_FLOAT64: "<d",
}


class Reader:
    def __init__(self, data):
        self.data = data
        self.offset = 0

    def take(self, count):
        if count < 0 or self.offset + count > len(self.data):
            raise ValueError("header_truncated")
        value = self.data[self.offset:self.offset + count]
        self.offset += count
        return value

    def unpack(self, fmt):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.take(size))[0]

    def string(self):
        length = self.unpack("<Q")
        return self.take(length).decode("utf-8", "replace")

    def value(self, value_type, want_value=True):
        if value_type == VALUE_STRING:
            text = self.string()
            return text if want_value else None
        if value_type == VALUE_ARRAY:
            element_type = self.unpack("<I")
            count = self.unpack("<Q")
            for _ in range(count):
                self.value(element_type, want_value=False)
            return {"array_type": element_type, "count": count} if want_value else None
        fmt = SCALAR_FORMATS.get(value_type)
        if fmt is None:
            raise ValueError(f"unknown_value_type:{value_type}")
        if not want_value:
            self.take(struct.calcsize(fmt))
            return None
        return self.unpack(fmt)


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def tensor_nbytes(dims, type_id):
    if type_id not in TYPE_TABLE:
        raise ValueError(f"unknown_tensor_type:{type_id}")
    _, block, element = TYPE_TABLE[type_id]
    if not dims:
        raise ValueError("empty_tensor_dims")
    if dims[0] % block != 0:
        raise ValueError("tensor_row_not_block_aligned")
    row = (dims[0] // block) * element
    total = row
    for dim in dims[1:]:
        total *= dim
    return total


def parse_data(data):
    reader = Reader(data)
    magic = reader.take(4)
    if magic != b"GGUF":
        raise ValueError("not_gguf")
    version = reader.unpack("<I")
    if version != 3:
        raise ValueError(f"unsupported_gguf_version:{version}")
    tensor_count = reader.unpack("<Q")
    kv_count = reader.unpack("<Q")
    metadata = {}
    metadata_arrays = {}
    for _ in range(kv_count):
        key = reader.string()
        value_type = reader.unpack("<I")
        value = reader.value(value_type)
        if value_type == VALUE_ARRAY:
            metadata_arrays[key] = value
        elif len(metadata) < 256:
            metadata[key] = value
    tensors = []
    for _ in range(tensor_count):
        name = reader.string()
        rank = reader.unpack("<I")
        if rank == 0 or rank > 4:
            raise ValueError(f"unsupported_tensor_rank:{rank}")
        dims = [reader.unpack("<Q") for _ in range(rank)]
        type_id = reader.unpack("<I")
        offset = reader.unpack("<Q")
        tensors.append({"name": name, "dims": dims, "type_id": type_id, "offset": offset})
    header_end = reader.offset
    alignment = metadata.get("general.alignment", 32)
    if not isinstance(alignment, int) or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("invalid_alignment")
    data_start = align_up(header_end, alignment)
    return {
        "version": version,
        "tensor_count": tensor_count,
        "kv_count": kv_count,
        "header_end": header_end,
        "data_start": data_start,
        "alignment": alignment,
        "metadata": metadata,
        "metadata_arrays": metadata_arrays,
        "tensors": tensors,
        "prefix_bytes": len(data),
    }


def parse(path, max_bytes=128 * 1024 * 1024):
    with open(path, "rb") as stream:
        data = stream.read(max_bytes)
    return parse_data(data)


def expert_index(parsed, file_size, top_k, arch):
    alignment = parsed["alignment"]
    data_start = parsed["data_start"]
    total_tensor_bytes = 0
    expert_bytes_total = 0
    layers = {}
    shared_by_layer = {}
    shared_tensor_count = 0
    for tensor in parsed["tensors"]:
        dims = tensor["dims"]
        nbytes = tensor_nbytes(dims, tensor["type_id"])
        total_tensor_bytes += nbytes
        absolute = data_start + tensor["offset"]
        if absolute < data_start or absolute + nbytes > file_size:
            raise ValueError(f"tensor_outside_file:{tensor['name']}")
        name = tensor["name"]
        if not name.startswith("blk."):
            continue
        parts = name.split(".")
        if len(parts) < 4:
            continue
        try:
            layer = int(parts[1])
        except ValueError:
            continue
        if "shexp" in parts[2]:
            if "inp" in parts[2]:
                continue
            if len(dims) != 2:
                raise ValueError(f"shared_expert_not_2d:{name}")
            shared_by_layer[layer] = shared_by_layer.get(layer, 0) + nbytes
            shared_tensor_count += 1
            continue
        if "exps" not in parts[2]:
            continue
        if len(dims) != 3:
            raise ValueError(f"expert_tensor_not_3d:{name}")
        n_expert = dims[2]
        expert_bytes = tensor_nbytes([dims[0], dims[1]], tensor["type_id"])
        if expert_bytes * n_expert != nbytes:
            raise ValueError(f"expert_stride_mismatch:{name}")
        role = parts[2]
        entry = {
            "name": name,
            "role": role,
            "type": TYPE_TABLE[tensor["type_id"]][0],
            "type_id": tensor["type_id"],
            "dims": dims,
            "offset": tensor["offset"],
            "absolute_offset": absolute,
            "expert_bytes": expert_bytes,
            "nbytes": nbytes,
        }
        layer_entry = layers.setdefault(layer, {"layer": layer, "roles": {}, "n_expert": n_expert,
                                                 "expert_bytes": 0})
        if layer_entry["n_expert"] != n_expert:
            raise ValueError(f"inconsistent_expert_count:{name}")
        layer_entry["roles"][role] = entry
        layer_entry["expert_bytes"] += expert_bytes
        expert_bytes_total += nbytes
    if not layers:
        return None
    ordered = [layers[key] for key in sorted(layers)]
    per_layer_min = min(layer["expert_bytes"] for layer in ordered)
    per_layer = max(layer["expert_bytes"] for layer in ordered)
    shared_per_layer = [shared_by_layer.get(layer["layer"], 0) for layer in ordered]
    shared_per_layer_bytes = max(shared_per_layer) if shared_per_layer else 0
    n_expert = ordered[0]["n_expert"]
    if top_k <= 0 or top_k > n_expert:
        raise ValueError("invalid_top_k")
    decode_union = top_k * per_layer
    return {
        "n_layer": len(ordered),
        "n_expert": n_expert,
        "top_k": top_k,
        "expert_bytes_per_layer": per_layer,
        "expert_bytes_per_layer_min": per_layer_min,
        "shared_expert_bytes_per_layer": shared_per_layer_bytes,
        "decode_union_bytes_per_layer": decode_union,
        "decode_live_bytes_per_layer": decode_union + shared_per_layer_bytes,
        "prefill_union_min_bytes_per_layer": decode_union,
        "prefill_union_max_bytes_per_layer": n_expert * per_layer,
        "layers": ordered,
        "expert_tensor_count": sum(len(layer["roles"]) for layer in ordered),
        "shared_expert_tensor_count": shared_tensor_count,
        "expert_bytes_total": expert_bytes_total,
        "shared_expert_bytes_total": sum(shared_by_layer.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="GGUF file or header prefix")
    parser.add_argument("--file-size", type=int, required=True, help="declared full file size")
    parser.add_argument("--output")
    parser.add_argument("--url", default=None)
    parser.add_argument("--top-k", type=int, default=0, help="0 reads expert_used_count")
    parser.add_argument("--mem-available-kib", type=int, default=0)
    parser.add_argument("--storage-available-bytes", type=int, default=0)
    parser.add_argument("--max-prefix-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--manifest-out", help="write the project expert manifest")
    arguments = parser.parse_args()

    with open(arguments.input, "rb") as stream:
        data = stream.read(arguments.max_prefix_bytes)
    prefix_hash = hashlib.sha256(data).hexdigest()
    parsed = parse_data(data)
    arch = parsed["metadata"].get("general.architecture")
    if not isinstance(arch, str):
        raise ValueError("missing_architecture")
    if parsed["data_start"] > parsed["prefix_bytes"]:
        raise ValueError("header_truncated")
    if parsed["prefix_bytes"] > arguments.file_size:
        raise ValueError("prefix_larger_than_file")
    top_k = arguments.top_k
    metadata_top_k = parsed["metadata"].get(f"{arch}.expert_used_count")
    declared_experts = parsed["metadata"].get(f"{arch}.expert_count")
    if top_k == 0:
        if isinstance(metadata_top_k, int):
            top_k = metadata_top_k
        elif declared_experts is None:
            top_k = 1
        else:
            raise ValueError("top_k_not_declared")
    experts = expert_index(parsed, arguments.file_size, top_k, arch)
    total_tensor_bytes = 0
    for tensor in parsed["tensors"]:
        nbytes = tensor_nbytes(tensor["dims"], tensor["type_id"])
        total_tensor_bytes += nbytes
    nonexpert = total_tensor_bytes - (experts["expert_bytes_total"] if experts else 0)
    mem_available_bytes = arguments.mem_available_kib * 1024
    admission = {
        "mem_available_bytes": mem_available_bytes,
        "full_resident_tensor_bytes": total_tensor_bytes,
        "streaming_integrated": False,
        "decision": "unknown",
        "reason": "no_memory_observation" if mem_available_bytes == 0 else None,
    }
    if mem_available_bytes > 0:
        if total_tensor_bytes > mem_available_bytes:
            admission["decision"] = "refuse"
            admission["reason"] = "tensor_bytes_exceed_available_memory"
        else:
            admission["decision"] = "admit_weights_only"
            admission["reason"] = "backend_workspace_and_state_not_covered"
    if experts:
        admission["streaming_decode_live_bytes_estimate"] = (
            nonexpert + experts["decode_union_bytes_per_layer"])
        admission["streaming_decision"] = "requires_backend_workspace_proof"
        admission["streaming_reason"] = "workspace_state_and_page_cache_not_covered"
    storage = {
        "artifact_bytes": arguments.file_size,
        "available_bytes": arguments.storage_available_bytes,
        "fits_available_storage": (arguments.file_size <= arguments.storage_available_bytes
                                   if arguments.storage_available_bytes > 0 else None),
        "transfer_and_temporary_space_reserved": False,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "tool": "gguf_index",
        "observed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {
            "url": arguments.url,
            "file_size": arguments.file_size,
            "prefix_bytes": parsed["prefix_bytes"],
            "prefix_sha256": prefix_hash,
        },
        "gguf": {
            "version": parsed["version"],
            "tensor_count": parsed["tensor_count"],
            "kv_count": parsed["kv_count"],
            "header_end": parsed["header_end"],
            "data_start": parsed["data_start"],
            "alignment": parsed["alignment"],
            "architecture": arch,
            "metadata": parsed["metadata"],
            "array_keys": sorted(parsed["metadata_arrays"].keys()),
        },
        "tensor_bytes": {
            "total": total_tensor_bytes,
            "expert": experts["expert_bytes_total"] if experts else 0,
            "nonexpert": nonexpert,
        },
        "experts": experts,
        "dense_without_expert_tensors": experts is None,
        "admission": admission,
        "storage": storage,
        "limitations": [
            "Only metadata and tensor information were read; tensor payloads were not loaded.",
            "Tensor bounds use the declared file size, not downloaded data.",
            "Backend workspace, KV/recurrent state, packing, and page-cache behavior are not covered.",
            "No inference or expert streaming was executed.",
        ],
    }
    if arguments.manifest_out:
        lines = ["# ame-expert-manifest 1",
                 f"# source {arguments.url or 'local'} {arguments.file_size} {prefix_hash}"]
        if experts:
            for layer in experts["layers"]:
                for role in sorted(layer["roles"]):
                    entry = layer["roles"][role]
                    lines.append(
                        f"tensor {entry['name']} {layer['layer']} {role} {entry['type_id']} "
                        f"{entry['type']} {experts['n_expert']} {entry['expert_bytes']} "
                        f"{entry['expert_bytes']} "
                        f"{entry['absolute_offset']} {entry['nbytes']}")
        Path(arguments.manifest_out).write_text("\n".join(lines) + "\n")
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        Path(arguments.output).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        print(f"gguf_index failed: {error}", file=sys.stderr)
        sys.exit(1)