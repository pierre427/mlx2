#!/usr/bin/env python3
"""Owned full-backbone parity/timing probe for layerwise short segments."""

from __future__ import annotations

import argparse
import copy
import gc
import time
from pathlib import Path

import nemotron35_probe_common as common
import numpy as np


def _bits(value, mx):
    dtype = mx.uint32 if value.dtype == mx.float32 else mx.uint16
    return np.asarray(value.view(dtype))


def _mismatches(actual, expected, mx):
    if actual is None or expected is None:
        return int(actual is not expected)
    a, e = _bits(actual, mx), _bits(expected, mx)
    return max(a.size, e.size) if a.shape != e.shape else int(np.count_nonzero(a != e))


def _cache_mismatches(actual, expected, mx):
    from mlx2.runtime.models.cache import ArraysCache

    rows = []
    for index, (a, e) in enumerate(zip(actual, expected)):
        if isinstance(a, ArraysCache):
            for slot, (av, ev) in enumerate(zip(a.cache, e.cache)):
                rows.append(
                    {
                        "cache": index,
                        "slot": slot,
                        "kind": "mamba",
                        "mismatches": _mismatches(av, ev, mx),
                    }
                )
            continue
        rows.append(
            {
                "cache": index,
                "kind": "attention_offset",
                "mismatches": int(a.offset != e.offset),
                "actual": int(a.offset),
                "expected": int(e.offset),
            }
        )
        for slot, (av, ev) in enumerate(zip(a.state, e.state)):
            av = av[..., : a.offset, :]
            ev = ev[..., : e.offset, :]
            rows.append(
                {
                    "cache": index,
                    "slot": slot,
                    "kind": "attention",
                    "mismatches": _mismatches(av, ev, mx),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    parser.add_argument("--settle-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Refusing to overwrite nonempty probe evidence")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    owner = common.require_owner(args.cpg_owner_lock)
    before = common.host_snapshot()
    errors = common.admission_errors(
        before, minimum_available_bytes=4 * common.GIB, performance=True
    )
    if errors:
        raise RuntimeError("; ".join(errors))

    import mlx.core as mx

    from mlx2.adapters.registry import resolve_adapter

    policy = {
        "mamba_conv": "fused_short",
        "mamba_ssm": "fused_short_gate",
        "mamba_projection": "short_row_exact",
        "mamba_segment": "layerwise_short_exact",
    }
    adapter = resolve_adapter(args.model)(str(args.model), execution_policy=policy)
    model = adapter.model
    time.sleep(args.settle_seconds)
    capture_layers = (0, len(model.layers) // 2, len(model.layers) - 1)
    base = model.make_cache()
    model(mx.array([[1, 2, 3, 4]]), base)
    mx.eval([entry.state for entry in base])
    token_bank = list(range(101, 117))

    def reference(tokens, keep=None):
        caches = copy.deepcopy(base)
        logits, taps = [], []
        count = len(tokens) if keep is None else keep
        for token in tokens[:count]:
            out, hidden = model.forward_with_taps(
                mx.array([[token]]), caches, capture_layers
            )
            logits.append(out)
            taps.append(hidden)
        if not logits:
            return None, None, caches
        out = mx.concatenate(logits, axis=1)
        hidden = mx.concatenate(taps, axis=1)
        mx.eval(out, hidden, [entry.state for entry in caches])
        return out, hidden, caches

    def candidate(tokens):
        caches = copy.deepcopy(base)
        for entry in caches:
            entry.start_speculation()
        out, hidden = model.forward_with_taps(
            mx.array([tokens]), caches, capture_layers
        )
        mx.eval(out, hidden, [entry.state for entry in caches])
        return out, hidden, caches

    cells = []
    for steps in (2, 4, 8, 16):
        common.require_owner(args.cpg_owner_lock)
        tokens = token_bank[:steps]
        expected, expected_taps, expected_cache = reference(tokens)
        actual, actual_taps, actual_cache = candidate(tokens)
        full_cache = _cache_mismatches(actual_cache, expected_cache, mx)
        rollback = []
        for keep in range(steps + 1):
            _, _, prefix_cache = reference(tokens, keep=keep)
            _, _, rolled = candidate(tokens)
            removed = []
            for entry in rolled:
                removed.append(int(entry.trim(steps - keep)))
                entry.stop_speculation()
            mx.eval([entry.state for entry in rolled])
            cache_rows = _cache_mismatches(rolled, prefix_cache, mx)
            rollback.append(
                {
                    "keep": keep,
                    "removed": removed,
                    "cache_mismatches": sum(row["mismatches"] for row in cache_rows),
                    "cache_rows": cache_rows,
                }
            )

        def serial_call(value=tokens):
            return reference(value)

        def candidate_call(value=tokens):
            return candidate(value)

        calls = {"serial_b1": serial_call, "layerwise_short": candidate_call}
        for _ in range(2):
            for call in calls.values():
                out, hidden, caches = call()
                mx.eval(out, hidden, [entry.state for entry in caches])
        timings = {name: [] for name in calls}
        for repetition in range(6):
            order = tuple(calls) if repetition % 2 == 0 else tuple(reversed(calls))
            for name in order:
                started = time.perf_counter_ns()
                out, hidden, caches = calls[name]()
                mx.eval(out, hidden, [entry.state for entry in caches])
                timings[name].append((time.perf_counter_ns() - started) * 1e-6)
        medians = {name: float(np.median(values)) for name, values in timings.items()}
        cell = {
            "steps": steps,
            "logit_mismatches": _mismatches(actual, expected, mx),
            "tap_mismatches": _mismatches(actual_taps, expected_taps, mx),
            "cache_mismatches": sum(row["mismatches"] for row in full_cache),
            "cache_rows": full_cache,
            "rollback": rollback,
            "median_ms": medians,
            "speedup_vs_serial_b1": medians["serial_b1"] / medians["layerwise_short"],
            "timings_ms": timings,
        }
        cells.append(cell)
        common.atomic_json(
            args.out_dir / "result.json",
            {
                "schema": "mlx2.nemotron35-layerwise-short-exact.v1",
                "state": "running",
                "cells": cells,
            },
        )
        del expected, expected_taps, expected_cache, actual, actual_taps, actual_cache
        mx.clear_cache()

    status = model.short_segment_status()
    adapter = None
    model = None
    base = None
    gc.collect()
    mx.clear_cache()
    after = common.host_snapshot()
    contamination = [
        reason
        for reason in common.contamination(before, after)
        if reason != "swap-outs increased"
    ]
    exact = all(
        cell["logit_mismatches"] == 0
        and cell["tap_mismatches"] == 0
        and cell["cache_mismatches"] == 0
        and all(
            all(removed == cell["steps"] - row["keep"] for removed in row["removed"])
            and row["cache_mismatches"] == 0
            for row in cell["rollback"]
        )
        for cell in cells
    )
    receipt = {
        "schema": "mlx2.nemotron35-layerwise-short-exact.v1",
        "state": "completed",
        "owner": owner,
        "source": common.source_identity(),
        "dependencies": common.runtime_dependencies(),
        "model": str(args.model.resolve()),
        "policy": policy,
        "settle_seconds": args.settle_seconds,
        "capture_layers": list(capture_layers),
        "before": before,
        "after": after,
        "swap_ignored_by_user": True,
        "cells": cells,
        "candidate_status": status,
        "all_exact": exact,
        "contamination": contamination,
        "passed": exact and not contamination,
        "qualified": False,
        "scope": "One B1 target session; exact layerwise Mamba short segments.",
    }
    common.atomic_json(args.out_dir / "result.json", receipt)
    if contamination:
        raise RuntimeError("layerwise short probe contaminated")
    if not exact:
        raise RuntimeError("layerwise short target differed from serial B1")


if __name__ == "__main__":
    main()
