#!/usr/bin/env python3
"""Owned every-prefix oracle for Nemotron-H full-backbone block execution."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import mlx.core as mx
import nemotron35_probe_common as common
import numpy as np
from mlx.utils import tree_flatten

sys.path.insert(0, str(common.ROOT / "tests"))
from test_nemotron_external_taps_cpu import tiny_model


def array_result(actual, expected):
    a, e = np.asarray(actual), np.asarray(expected)
    if a.shape != e.shape:
        return {
            "exact": False,
            "mismatch_count": max(int(a.size), int(e.size)),
            "max_abs": None,
            "actual_shape": list(a.shape),
            "expected_shape": list(e.shape),
        }
    return {
        "exact": bool(np.array_equal(a, e)),
        "mismatch_count": int(np.count_nonzero(a != e)),
        "max_abs": float(np.max(np.abs(a.astype(np.float32) - e.astype(np.float32))))
        if a.size else 0.0,
    }


def cache_results(actual, expected):
    rows = []
    for index, (a, e) in enumerate(zip(actual, expected)):
        for kind, actual_tree, expected_tree in (
            ("state", a.state, e.state),
            ("meta_state", a.meta_state, e.meta_state),
        ):
            actual_leaves = dict(tree_flatten(actual_tree))
            expected_leaves = dict(tree_flatten(expected_tree))
            for path in sorted(actual_leaves.keys() | expected_leaves.keys()):
                av = actual_leaves.get(path)
                ev = expected_leaves.get(path)
                if av is None or ev is None:
                    result = {
                        "exact": av is None and ev is None,
                        "mismatch_count": int(not (av is None and ev is None)),
                        "max_abs": None,
                    }
                elif isinstance(av, (str, bytes)) or isinstance(ev, (str, bytes)):
                    result = {
                        "exact": av == ev,
                        "mismatch_count": int(av != ev),
                        "max_abs": None,
                    }
                else:
                    result = array_result(av, ev)
                rows.append(
                    {"cache": index, "kind": kind, "path": path, **result}
                )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Refusing to overwrite nonempty probe evidence")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    owner = common.require_owner(args.cpg_owner_lock)
    before = common.host_snapshot()
    errors = common.admission_errors(before, minimum_available_bytes=4 * common.GIB)
    if errors:
        raise RuntimeError("; ".join(errors))

    model = tiny_model()
    base = model.make_cache()
    model(mx.array([[1, 2, 3]]), base)
    mx.eval([c.state for c in base])
    cases = []
    for width in range(2, 9):
        tokens = mx.array([list(range(4, 4 + width))])
        block_cache = copy.deepcopy(base)
        serial_cache = copy.deepcopy(base)
        block_logits, block_taps = model.forward_with_taps(
            tokens, block_cache, [0, 1, 4]
        )
        serial_logits, serial_taps = [], []
        for step in range(width):
            logits, taps = model.forward_with_taps(
                tokens[:, step : step + 1], serial_cache, [0, 1, 4]
            )
            serial_logits.append(logits)
            serial_taps.append(taps)
        serial_logits = mx.concatenate(serial_logits, axis=1)
        serial_taps = [
            mx.concatenate([row[layer] for row in serial_taps], axis=-2)
            for layer in range(3)
        ]
        mx.eval(block_logits, block_taps, serial_logits, serial_taps,
                [c.state for c in block_cache], [c.state for c in serial_cache])
        logits_result = array_result(block_logits, serial_logits)
        taps_result = [array_result(a, e) for a, e in zip(block_taps, serial_taps)]
        caches_result = cache_results(block_cache, serial_cache)
        cases.append({
            "width": width,
            "logits": logits_result,
            "taps": taps_result,
            "caches": caches_result,
            "exact": logits_result["exact"]
            and all(row["exact"] for row in taps_result)
            and all(row["exact"] for row in caches_result),
        })

    after = common.host_snapshot()
    contamination = [
        reason for reason in common.contamination(before, after)
        if reason != "swap-outs increased"
    ]
    receipt = {
        "schema": "mlx2.nemotron35-super-full-backbone-block-parity.v1",
        "owner": owner,
        "source": common.source_identity(),
        "dependencies": common.runtime_dependencies(),
        "before": before,
        "after": after,
        "swap_ignored_by_user": True,
        "cases": cases,
        "all_exact": all(case["exact"] for case in cases),
        "contamination": contamination,
        "passed": not contamination,
        "note": "Diagnostic prerequisite: non-exact block arithmetic is retained as evidence and does not make this diagnostic run fail.",
    }
    common.atomic_json(args.out_dir / "result.json", receipt)
    if contamination:
        raise RuntimeError("Full-backbone block diagnostic was contaminated")


if __name__ == "__main__":
    main()
