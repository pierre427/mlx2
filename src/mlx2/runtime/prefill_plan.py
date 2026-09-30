"""Import-free geometry for prefill kernels (rows are not request lanes)."""

import hashlib
import json


def execution_identity(projection=None, scan=None):
    """Numerical law identity, excluding counters that change during a run."""
    if not projection and not scan:
        return None
    result = {"version": 1}
    if projection:
        result["projection"] = {
            key: projection[key]
            for key in ("kernel", "installed", "names_sha256", "tile")
        }
    if scan:
        result["scan"] = {
            key: scan[key] for key in ("chunk_size", "segment_max_rows", "layers")
        }
    return result


def apc_prefill_fingerprint(base, identity):
    """Separate candidate prefix states on disk and in memory; off is identity."""
    if identity is None:
        return base
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return (base, "prefill-execution-v1", hashlib.sha256(payload).hexdigest())


def recurrence_segments(length: int, *, maximum: int = 256, alignment: int = 8):
    """Bound recurrence workspace without leaving a 1..16-row final tile.

    The core GDN prefill admission starts at 17 rows. Borrow one aligned
    block from the preceding segment when needed; every token occurs once.
    A short entire input remains short and belongs to the reference kernel.
    """
    if type(length) is not int or length < 0:
        raise ValueError("length must be a nonnegative integer")
    if type(alignment) is not int or alignment not in (8, 16):
        raise ValueError("alignment must be 8 or 16")
    if type(maximum) is not int or maximum < 48 or maximum % alignment:
        raise ValueError("maximum must be aligned and at least 48")
    start = 0
    while start < length:
        size = min(maximum, length - start)
        tail = length - start - size
        if 0 < tail < 17:
            size -= ((17 - tail + alignment - 1) // alignment) * alignment
        yield start, start + size
        start += size


def prefill_rows(shape, *, minimum_sequence: int = 18):
    """Admit [batch, sequence, features], never a wide one-token decode batch."""
    if len(shape) != 3 or shape[0] < 1 or shape[1] < minimum_sequence or shape[2] < 1:
        return 0
    return int(shape[0]) * int(shape[1])
