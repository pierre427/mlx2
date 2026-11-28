"""Fused device work of a lazy tree round (``device_round.tree_fused``).

Original mlx2 implementation (no mined code).  Default off; only
``speculative_device_tree`` calls into this module, and only when the
``tree_fused`` switch is on (it needs ``tree_decide``).

A ``tree_decide`` round (speculative_device_tree.py) adds about 30 small
device operations to a round: the verify input ``[anchor] + clip(tokens)``
(drafter concatenate and cast, clip, concatenate), and the decision packet
(float32 cast, argmax, finiteness, a parent gather, sibling and ancestor
masks and their reductions, a depth select, three host-built constant masks
uploaded every round).  Two Metal kernels replace them:

``verify_row``
    One launch: ``row = [anchor, clip(source[index], 0, bound)]`` (int32, the
    verify input the round already builds) and ``raw = source[index]`` (the
    unclipped drafts the host reads, so a draft outside the vocabulary still
    declines the round after the read).  ``source`` is the drafter's own
    branch grid (native MTP: ``[branches, depth]`` uint32), so its per-branch
    slices, concatenate and cast are never evaluated; ``index`` is a static
    per-topology table uploaded once.
``decide_rows``
    One launch over the ``[rows, vocab]`` verify law (bf16 logits, or the
    float32 processed rows under ``processors``): per row, ``mx.argmax``'s
    index (the first NaN if any, else the first index of the maximum) and the
    host verifier's finiteness verdict (no NaN, no ``+inf``, not all
    ``-inf``: exactly when ``logsumexp`` is finite, and exactly
    ``isfinite(peak) & ~any(isnan)``).  One threadgroup per row.

The accepted path is then walked on the host from ``selected`` and the host
tokens (first matching child per node, the host walk's tie-break), which is
what the ``tree_decide`` packet's device path selection computed; the host
reads ``selected`` in the same single read either way.  Response log
probabilities (``emit_logprobs``) stay the lazy ``values - logsumexp`` arrays
of the unfused round, so they are bitwise the same and are only evaluated
when a response needs them.

Off the GPU (CPU tests, ``MLX2_PREFLIGHT_CPU_ONLY``) both functions run the
same semantics as plain MLX operations.
"""

from __future__ import annotations

import functools

VERSION = "speculative-device-tree-fused-v1"
COUNTER = "external_device_tree_fused_lanes"
# Threads per row of ``decide_rows`` (a multiple of the 32-wide simdgroup).
_THREADS = 1024

_VERIFY_ROW_SOURCE = """
    uint i = thread_position_in_grid.x;
    if (i == 0) {
        row[0] = anchor[0];
        return;
    }
    if (i >= threads_per_grid.x) {
        return;
    }
    int token = static_cast<int>(source[index[i - 1]]);
    raw[i - 1] = token;
    row[i] = metal::clamp(token, 0, int(BOUND));
"""

_DECIDE_ROWS_SOURCE = """
    const uint row = threadgroup_position_in_grid.x;
    const uint lane = thread_position_in_threadgroup.x;
    const uint simd = lane / 32;
    const uint slot = lane % 32;
    const device T* x = values + size_t(row) * size_t(VOCAB);
    // Sentinel index VOCAB: nothing taken yet (an all -inf row ends here).
    float best = -INFINITY;
    uint index = uint(VOCAB);
    bool nan_seen = false;
    bool pinf_seen = false;
    for (uint v = lane; v < uint(VOCAB); v += uint(THREADS)) {
        float f = static_cast<float>(x[v]);
        bool f_nan = metal::isnan(f);
        nan_seen = nan_seen || f_nan;
        pinf_seen = pinf_seen || (f == INFINITY);
        // mx.argmax: the first NaN wins; else strictly greater (first max).
        bool best_nan = (index != uint(VOCAB)) && metal::isnan(best);
        if ((f_nan && !best_nan) || (!best_nan && !f_nan && f > best)) {
            best = f;
            index = v;
        }
    }
    // Reduce (best, index) and the flags across the simdgroup, then across
    // simdgroups: NaN first (smallest index), else larger value, ties to the
    // smaller index; the sentinel loses every comparison it can lose.
    for (ushort delta = 16; delta > 0; delta >>= 1) {
        float other = simd_shuffle_down(best, delta);
        uint other_index = simd_shuffle_down(index, delta);
        bool other_valid = other_index != uint(VOCAB);
        bool mine_valid = index != uint(VOCAB);
        bool other_nan = other_valid && metal::isnan(other);
        bool mine_nan = mine_valid && metal::isnan(best);
        bool take = false;
        if (other_valid) {
            if (!mine_valid) {
                take = true;
            } else if (other_nan) {
                take = !mine_nan || other_index < index;
            } else if (!mine_nan) {
                take = other > best || (other == best && other_index < index);
            }
        }
        if (take) {
            best = other;
            index = other_index;
        }
    }
    nan_seen = simd_any(nan_seen);
    pinf_seen = simd_any(pinf_seen);
    threadgroup float shared_best[THREADS / 32];
    threadgroup uint shared_index[THREADS / 32];
    threadgroup uint shared_flags[THREADS / 32];
    if (slot == 0) {
        shared_best[simd] = best;
        shared_index[simd] = index;
        shared_flags[simd] = (nan_seen ? 1u : 0u) | (pinf_seen ? 2u : 0u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        const uint groups = uint(THREADS) / 32;
        best = slot < groups ? shared_best[slot] : -INFINITY;
        index = slot < groups ? shared_index[slot] : uint(VOCAB);
        uint flags = slot < groups ? shared_flags[slot] : 0u;
        for (ushort delta = 16; delta > 0; delta >>= 1) {
            float other = simd_shuffle_down(best, delta);
            uint other_index = simd_shuffle_down(index, delta);
            bool other_valid = other_index != uint(VOCAB);
            bool mine_valid = index != uint(VOCAB);
            bool other_nan = other_valid && metal::isnan(other);
            bool mine_nan = mine_valid && metal::isnan(best);
            bool take = false;
            if (other_valid) {
                if (!mine_valid) {
                    take = true;
                } else if (other_nan) {
                    take = !mine_nan || other_index < index;
                } else if (!mine_nan) {
                    take = other > best || (other == best && other_index < index);
                }
            }
            if (take) {
                best = other;
                index = other_index;
            }
        }
        flags = simd_or(flags);
        if (slot == 0) {
            bool finite = (flags == 0u) && (index != uint(VOCAB));
            packet[row] = index == uint(VOCAB) ? 0 : int(index);
            packet[threadgroups_per_grid.x + row] = finite ? 1 : 0;
        }
    }
"""


def _on_metal(mx):
    try:
        return bool(mx.metal.is_available()) and mx.default_device() == mx.gpu
    except Exception:  # noqa: BLE001 - no Metal backend: the reference path
        return False


@functools.lru_cache(maxsize=1)
def _verify_row_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="mlx2_device_tree_verify_row",
        input_names=["source", "index", "anchor"],
        output_names=["row", "raw"],
        source=_VERIFY_ROW_SOURCE,
    )


@functools.lru_cache(maxsize=1)
def _decide_rows_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="mlx2_device_tree_decide_rows",
        input_names=["values"],
        output_names=["packet"],
        source=_DECIDE_ROWS_SOURCE,
    )


_INDEX_TABLES = {}


def index_table(mx, index):
    """The device ``int32`` table of one static index tuple (uploaded once)."""
    key = tuple(int(value) for value in index)
    table = _INDEX_TABLES.get(key)
    if table is None:
        if len(_INDEX_TABLES) >= 256:
            _INDEX_TABLES.clear()
        table = mx.array(list(key), dtype=mx.int32)
        _INDEX_TABLES[key] = table
    return table


def verify_row(mx, source, index, anchor, bound):
    """``([anchor] + clip(source[index], 0, bound), source[index])`` as int32.

    ``source`` is any integer device array (read flattened); ``index`` the
    static flat positions of the tree's nodes in node order."""
    count = len(index)
    if count < 1:
        raise ValueError("a fused verify row needs at least one node")
    flat = source.reshape(-1)
    table = index_table(mx, index)
    anchor_array = mx.array([int(anchor)], dtype=mx.int32)
    if _on_metal(mx):
        row, raw = _verify_row_kernel()(
            inputs=[flat, table, anchor_array],
            template=[("BOUND", int(bound))],
            grid=(count + 1, 1, 1),
            threadgroup=(count + 1, 1, 1),
            output_shapes=[(count + 1,), (count,)],
            output_dtypes=[mx.int32, mx.int32],
        )
        return row, raw
    raw = mx.take(flat, table).astype(mx.int32)
    row = mx.concatenate([anchor_array, mx.clip(raw, 0, int(bound))])
    return row, raw


def decide_rows(mx, values):
    """``[selected (rows), finite (rows)]`` int32 of a ``[rows, vocab]`` law.

    ``selected`` is ``mx.argmax(values.astype(float32), axis=-1)``; ``finite``
    is the host verifier's verdict that the row is a distribution."""
    if values.ndim != 2:
        raise ValueError("fused tree decisions take [rows, vocab] values")
    rows, vocab = int(values.shape[0]), int(values.shape[1])
    if _on_metal(mx) and values.dtype in (mx.bfloat16, mx.float16, mx.float32):
        (packet,) = _decide_rows_kernel()(
            inputs=[values],
            template=[("T", values.dtype), ("VOCAB", vocab), ("THREADS", _THREADS)],
            grid=(_THREADS * rows, 1, 1),
            threadgroup=(_THREADS, 1, 1),
            output_shapes=[(2 * rows,)],
            output_dtypes=[mx.int32],
        )
        return packet
    return reference_decide_rows(mx, values)


def reference_decide_rows(mx, values):
    """``decide_rows`` as plain MLX operations (the unfused arithmetic)."""
    values = values.astype(mx.float32)
    selected = mx.argmax(values, axis=-1)
    peak = mx.take_along_axis(values, selected[:, None], axis=-1)[:, 0]
    finite = mx.isfinite(peak) & ~mx.any(mx.isnan(values), axis=-1)
    return mx.concatenate([selected.astype(mx.int32), finite.astype(mx.int32)])


def walk(target_parents, tokens, selected):
    """The greedy accepted path's rows: from the anchor row, the first child
    (node order) whose token is its parent row's ``selected``.  The unvisited
    rows' validity is irrelevant (the host walk stops at the first invalid
    visited row before any child of it is taken)."""
    children = [[] for _ in target_parents]
    for row in range(1, len(target_parents)):
        children[target_parents[row]].append(row)
    path, row = [0], 0
    while True:
        want = selected[row]
        nxt = next((child for child in children[row] if tokens[child - 1] == want), None)
        if nxt is None:
            return path
        path.append(nxt)
        row = nxt


__all__ = ["COUNTER", "VERSION", "decide_rows", "index_table", "reference_decide_rows",
           "verify_row", "walk"]
