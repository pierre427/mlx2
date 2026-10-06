"""Import-free geometry for prefill kernels (rows are not request lanes)."""

import hashlib
import json


EXTERNAL_VARLEN_PREFILL_IDENTITY = {
    "schema": "mlx2.external-varlen-prefill.v1",
    "law": "right-padded-target-prefill-merge-private-extract-live",
}

# Model-independent fallback used only when neither the operator nor the
# adapter declares a prefill chunk.  The top rung is selected as the fallback
# default but remains unqualified at 128K; receipts distinguish this policy
# from an adapter selection or qualification evidence.
PROMPT_LENGTH_PREFILL_SCHEDULE = (
    (32768, 512),
    (65536, 2048),
    (131072, 8192),
)


def prompt_length_prefill_step(prompt_tokens: int, *, maximum: int = 8192) -> int:
    """Choose a bounded chunk from total prompt fill, independent of model."""
    if type(prompt_tokens) is not int or prompt_tokens < 1:
        raise ValueError("prompt_tokens must be a positive integer")
    if type(maximum) is not int or maximum < 1:
        raise ValueError("maximum must be a positive integer")
    for context_limit, step in PROMPT_LENGTH_PREFILL_SCHEDULE:
        if prompt_tokens <= context_limit:
            return min(step, maximum)
    return min(PROMPT_LENGTH_PREFILL_SCHEDULE[-1][1], maximum)


def single_round_prefill_rows(segment_lengths) -> int | None:
    """Rows needed to reach the reserved final-token generation boundary.

    The engine processes only the first segment in a prefill round. Insertion
    reserves the prompt's final token as a singleton segment for promotion, so
    exactly one non-final segment can finish in one round. More segments need
    more scheduler rounds regardless of their combined length.
    """
    lengths = tuple(segment_lengths)
    if not lengths or any(type(length) is not int or length < 1 for length in lengths):
        raise ValueError("segment lengths must be positive integers")
    if lengths == (1,):
        return 0
    if len(lengths) == 2 and lengths[-1] == 1:
        return lengths[0]
    return None


def shared_prefill_budget_width(
    base: int,
    rows: int,
    *,
    enabled: bool,
    token_budget: int | None = None,
) -> int:
    """Per-row width after an optional shared padded-token budget."""
    base = max(1, int(base))
    if not enabled:
        return base
    budget = base if token_budget is None else min(base, int(token_budget))
    return max(1, budget // max(1, int(rows)))


def next_prefill_checkpoint(positions, covered: int) -> int | None:
    """First planned checkpoint beyond ``covered``, without consuming it."""
    covered = int(covered)
    for position in positions:
        position = int(position)
        if position > covered:
            return position
    return None


def checkpoint_bounded_prefill_rows(
    rows: int, depth: int, next_checkpoint: int | None
) -> int:
    """Clamp a pure row budget to the next exact checkpoint boundary."""
    rows, depth = int(rows), int(depth)
    if rows < 0 or depth < 0:
        raise ValueError("rows and depth must be nonnegative")
    if rows == 0:
        return 0
    if next_checkpoint is None:
        return rows
    distance = int(next_checkpoint) - depth
    if distance < 1:
        raise ValueError("next checkpoint must follow the covered depth")
    return min(rows, distance)


def prefill_fits_one_chunk(
    remaining_tokens: int,
    total_tokens: int,
    scheduled_step: int,
    *,
    autoscale: bool,
    maximum: int,
    depth: int = 0,
    depth_budget=None,
    depth_floor: int = 128,
    next_checkpoint: int | None = None,
) -> bool:
    """Whether all actual prefill rows fit the next scheduled slice.

    Overflow admission must use the same per-prompt autoscale cap as the
    execution loop.  Comparing only with the configured maximum can admit a
    short-context request as a one-chunk overflow lane even though execution
    will deliberately slice it at 512 rows. ``remaining_tokens`` excludes the
    final singleton token reserved for generation; ``total_tokens`` retains it
    because the autoscale schedule is based on the complete prompt.
    """
    if type(remaining_tokens) is not int or remaining_tokens < 0:
        raise ValueError("remaining_tokens must be a nonnegative integer")
    if type(total_tokens) is not int or total_tokens < 1:
        raise ValueError("total_tokens must be a positive integer")
    if type(scheduled_step) is not int or scheduled_step < 1:
        raise ValueError("scheduled_step must be a positive integer")
    if type(autoscale) is not bool:
        raise ValueError("autoscale must be boolean")
    if type(maximum) is not int or maximum < 1:
        raise ValueError("maximum must be a positive integer")
    actual_step = min(scheduled_step, maximum)
    if autoscale:
        actual_step = min(
            actual_step,
            prompt_length_prefill_step(total_tokens, maximum=maximum),
        )
    actual_step = depth_bounded_prefill_rows(
        actual_step, depth, depth_budget, floor=depth_floor
    )
    actual_step = checkpoint_bounded_prefill_rows(
        actual_step, depth, next_checkpoint
    )
    return remaining_tokens <= actual_step


def execution_identity(
    projection=None,
    scan=None,
    invariant=None,
    varlen=None,
    external_varlen_prefill=None,
):
    """Numerical law identity, excluding counters that change during a run."""
    if (
        not projection
        and not scan
        and not invariant
        and not varlen
        and not external_varlen_prefill
    ):
        return None
    result = {"version": 1}
    if invariant:
        result["invariant"] = {key: invariant[key] for key in ("schema", "law")}
    if projection:
        result["projection"] = {
            key: projection[key]
            for key in ("kernel", "installed", "names_sha256", "tile")
        }
    if scan:
        result["scan"] = {
            key: scan[key] for key in ("chunk_size", "segment_max_rows", "layers")
        }
    if varlen:
        result["varlen"] = {
            key: varlen[key] for key in ("schema", "law", "policy")
        }
    if external_varlen_prefill:
        result["external_varlen_prefill"] = {
            key: external_varlen_prefill[key] for key in ("schema", "law")
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
