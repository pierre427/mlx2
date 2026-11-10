"""Synthetic, storage-only DS4.1 KV microbenchmarks; caller owns the GPU lease.

No MLX import, model, checkpoint, weights, attention, or APC publication. Context
labels are the POST-update frontier. Growth deliberately uses an exact-fill
restored pool, whereas the ordinary full-update cell uses rounded capacity.
"""
from __future__ import annotations

import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

from statistics import median
from time import perf_counter

from mlx2.runtime.models.deepseek_v4_cache import _copy
from mlx2.runtime.models.deepseek_v41_cache import DeepSeekV41LayerCache

CONTEXTS = (100, 128, 1024, 4096, 16384)
RATIOS = (1, 2)
REPETITIONS = 7
ARRAY_CAP_BYTES = 2 * 1024**3
MODES = (
    "concat", "local_owned_copy", "retention_view", "update_local",
    "append_no_grow", "append_growth", "full_kv_storage_update",
)


def _physical(state):
    return tuple(v for v in (
        state.keys, state.buf_kv, state.buf_gate,
        state._pooled, state._index_keys, state.hash_tail,
    ) if v is not None)


def _eval(mx, *arrays):
    values = tuple(v for v in arrays if v is not None)
    if values and hasattr(mx, "eval"):
        mx.eval(*values)


def _shape_bytes(frontier, ratio, mode):
    if type(frontier) is not int or frontier not in CONTEXTS:
        raise ValueError("unsupported synthetic context")
    if type(ratio) is not int or ratio not in RATIOS or mode not in MODES:
        raise ValueError("unsupported synthetic geometry/mode")
    old_count = (frontier - 1) // ratio
    capacity = old_count if mode == "append_growth" else ((old_count + 1 + 255) // 256) * 256
    # Physical cache + inputs + prebuilt local band; excludes allocator overhead.
    total = 4 * (capacity * (512 + 128) + min(frontier - 1, 128) * 512
                 + 2 * ((frontier - 1) % ratio) * 512 + 512 + 512 + 128 + 128 * 512)
    if total >= ARRAY_CAP_BYTES:
        raise ValueError("synthetic array budget exceeded")
    return old_count, capacity, total


def _make_case(mx, frontier, ratio, mode):
    """Allocate and complete the independent reset BEFORE starting a clock."""
    old_count, capacity, byte_count = _shape_bytes(frontier, ratio, mode)
    state = DeepSeekV41LayerCache(ratio, xp=mx, source=True,
                                 identity="synthetic-kv-diagnostic-only")
    state.offset = frontier - 1
    state.keys = mx.full((1, 1, min(frontier - 1, 128), 512), 0.125, dtype=mx.float32)
    rem = state.offset % ratio
    state.buf_kv = mx.full((1, rem, 512), 0.25, dtype=mx.float32)
    state.buf_gate = mx.full((1, rem, 512), 0.5, dtype=mx.float32)
    state._pool_count, state._capacity = old_count, capacity
    state._pooled = mx.full((1, capacity, 512), 0.125, dtype=mx.float32)
    state._index_keys = mx.full((1, capacity, 128), 0.25, dtype=mx.float32)
    local = mx.full((1, 1, 1, 512), 0.75, dtype=mx.float32)
    pooled = mx.full((1, 1, 512), 0.5, dtype=mx.float32)
    index = mx.full((1, 1, 128), 0.375, dtype=mx.float32)
    prior = state.keys[:, :, -127:, :]
    band = mx.concatenate((prior, local), axis=2)
    if mode.startswith("append_"):
        state.offset = frontier  # publish receives the advanced local frontier.
    _eval(mx, *_physical(state), local, pooled, index, band)

    def operation():
        if mode == "concat":
            return mx.concatenate((prior, local), axis=2)
        if mode == "local_owned_copy":
            return _copy(mx, band[:, :, -128:, :])
        if mode == "retention_view":
            return band[:, :, -128:, :]
        if mode == "update_local":
            return state.update_local(local)
        if mode in ("append_no_grow", "append_growth"):
            state.append_pools(pooled, index)
            return state.pooled
        result = state.update_local(local)
        state.append_pools(pooled, index)
        # Inputs are already compressed source rows; compressor math is excluded.
        # All chosen post-frontiers end a group for both source ratios.
        state.buf_kv = state.buf_gate = None
        return result

    return state, operation, {
        "input_array_bytes": byte_count, "prior_offset": frontier - 1,
        "post_frontier": frontier, "ratio": ratio,
        "prior_pool_count": old_count, "prior_pool_capacity": capacity,
        "pool_capacity_policy": "exact_fill_restored" if mode == "append_growth" else "rounded_256",
        "local_band_rows": min(frontier, 128),
    }


def _measure(mx, operation, state):
    start = perf_counter()
    result = operation()
    submitted = perf_counter()
    _eval(mx, result, *_physical(state))
    completed = perf_counter()
    # One finite scalar after completion is a safety check, not a parity oracle.
    sample = result.reshape(-1)[0].item()
    if not (-float("inf") < float(sample) < float("inf")):
        raise ValueError("nonfinite synthetic result")
    return {"submit_ms": (submitted - start) * 1000,
            "completion_ms": (completed - submitted) * 1000,
            "wall_ms": (completed - start) * 1000}


def run(mx, check_live=None):
    """Run seven interleaved repetitions; root-owned worker injects mlx.core.

    check_live is invoked before each reset; all allocations and reset evals are
    outside the measured operation. No GPU capability is acquired here.
    """
    cells = [(n, r, mode) for n in CONTEXTS for r in RATIOS for mode in MODES]
    results = {cell: {"context": cell[0], "ratio": cell[1], "mode": cell[2],
                      "samples": []} for cell in cells}
    order = []
    for repetition in range(REPETITIONS):
        rotation = repetition % len(cells)
        for cell in cells[rotation:] + cells[:rotation]:
            if check_live is not None:
                check_live()
            state, operation, metadata = _make_case(mx, *cell)
            sample = _measure(mx, operation, state)
            sample["repetition"] = repetition
            results[cell]["samples"].append(sample)
            results[cell]["geometry"] = metadata
            results[cell]["post_pool_capacity"] = state._capacity
            order.append([repetition, *cell])
            del operation, state
    for result in results.values():
        result["summary_ms"] = {
            field: {"median": median(s[field] for s in result["samples"]),
                    "minimum": min(s[field] for s in result["samples"]),
                    "maximum": max(s[field] for s in result["samples"])}
            for field in ("submit_ms", "completion_ms", "wall_ms")
        }
    return {
        "schema": "ds41-synthetic-kv-components-v1", "diagnostic_only": True,
        "seed": 41, "data_generation": "fixed finite constants (no RNG needed)",
        "checkpoint": "none-synthetic-no-weights", "model_startup": False,
        "head_dim": 512, "index_dim": 128, "window": 128,
        "context_scope": "post-one-token-update frontier", "repetitions": REPETITIONS,
        "array_cap_bytes": ARRAY_CAP_BYTES, "cells": list(results.values()),
        "execution_order": order,
        "timing_scope": "submit then eval physical state/output; reset outside timers",
        "full_update_scope": "storage only: local update plus already-compressed pool/index append",
        "excluded": ["compressor projection", "attention", "weights", "model", "APCv2 publication"],
        "peak_memory_bytes": mx.get_peak_memory() if hasattr(mx, "get_peak_memory") else None,
    }
