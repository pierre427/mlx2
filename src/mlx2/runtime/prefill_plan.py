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


def depth_bounded_prefill_rows(step: int, depth: int, budget, *, floor: int = 128):
    """Prefill chunk rows for a lane whose KV cache already holds ``depth``.

    One chunk's attention costs about ``rows * (depth + rows)``; at deep KV a
    fixed chunk can run one Metal command buffer past the GPU watchdog
    (jundot/omlx#4149).  ``budget`` bounds that product: the configured
    ``step`` is kept while ``step * (depth + step) <= budget`` -- so every
    chunk ending at or before ``budget // step`` tokens is unchanged, and so
    are its output bits -- and beyond that the largest power of two that fits
    is used, never below ``min(step, floor)``.  ``budget=None`` is off.
    """
    if budget is None:
        return step
    if type(step) is not int or step < 1 or type(depth) is not int or depth < 0:
        raise ValueError("step must be positive and depth nonnegative")
    if type(budget) is not int or budget < 1:
        raise ValueError("budget must be a positive integer or None")
    if step * (depth + step) <= budget:
        return step
    rows = 1 << (step.bit_length() - 1)
    if rows == step:
        rows >>= 1
    while rows > floor and rows * (depth + rows) > budget:
        rows >>= 1
    return max(rows, min(step, floor))
