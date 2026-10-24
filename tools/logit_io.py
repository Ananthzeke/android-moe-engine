"""Strict readers and comparisons for runner logit and token files."""
import math
import pathlib
import struct


class FormatError(ValueError):
    pass


def read_logits(path):
    """Return rows of raw float32 bytes; reject truncation, trailing bytes, and non-finite values."""
    data = pathlib.Path(path).read_bytes()
    if len(data) < 8:
        raise FormatError(f"{path}: missing row count")
    count = struct.unpack_from("<Q", data, 0)[0]
    offset = 8
    rows = []
    for index in range(count):
        if len(data) - offset < 8:
            raise FormatError(f"{path}: row {index} header is truncated")
        width = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        if width == 0:
            raise FormatError(f"{path}: row {index} is empty")
        if width > (len(data) - offset) // 4:
            raise FormatError(f"{path}: row {index} declares {width} floats beyond end of file")
        payload = data[offset:offset + 4 * width]
        offset += 4 * width
        if not all(math.isfinite(value) for (value,) in struct.iter_unpack("<f", payload)):
            raise FormatError(f"{path}: row {index} contains a non-finite logit")
        rows.append(payload)
    if offset != len(data):
        raise FormatError(f"{path}: {len(data) - offset} trailing bytes after {count} rows")
    return rows


def read_tokens(path):
    data = pathlib.Path(path).read_bytes()
    if len(data) < 8:
        raise FormatError(f"{path}: missing token count")
    count = struct.unpack_from("<Q", data, 0)[0]
    if count != (len(data) - 8) // 4 or (len(data) - 8) % 4 != 0:
        raise FormatError(f"{path}: token payload does not match declared count {count}")
    return list(struct.unpack_from("<%di" % count, data, 8))


def count_mismatched_floats(left, right):
    """Compare equal-shaped row lists; a shape difference is an error, never zero mismatches."""
    if not left or not right:
        raise FormatError("a comparison needs at least one logit row on each side")
    if len(left) != len(right):
        raise FormatError(f"row counts differ: {len(left)} != {len(right)}")
    mismatched = 0
    for index, (lhs, rhs) in enumerate(zip(left, right)):
        if len(lhs) != len(rhs):
            raise FormatError(f"row {index} widths differ: {len(lhs) // 4} != {len(rhs) // 4}")
        mismatched += sum(1 for a, b in zip(struct.iter_unpack("<f", lhs),
                                            struct.iter_unpack("<f", rhs)) if a != b)
    return mismatched
