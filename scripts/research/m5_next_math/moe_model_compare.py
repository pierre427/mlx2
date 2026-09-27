#!/usr/bin/env python3
"""Compare separate-process M=3 full-model stock/tile4/tile8 receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def _load(path: Path) -> dict:
    data = json.loads(path.read_text())
    if data.get("schema") != "mlx2.m5-next-math.qwen4-m3-model-gate.v1":
        raise RuntimeError(f"unexpected receipt schema: {path}")
    if data.get("completed") is not True:
        raise RuntimeError(f"incomplete model gate: {path}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--tile4", type=Path, required=True)
    parser.add_argument("--tile8", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np

    paths = {"stock": args.stock, "tile4": args.tile4, "tile8": args.tile8}
    receipts = {name: _load(path) for name, path in paths.items()}
    identity_fields = (
        "model", "model_identity", "model_config_sha256", "repo_head", "production_source_sha256",
        "path_source_sha256", "execution_profile", "tile8_source_sha256",
        "prompt_token_sha256", "block_token_ids",
        "moe_module_count",
    )
    reference = receipts["stock"]
    for name, receipt in receipts.items():
        if receipt["arm"] != name:
            raise RuntimeError(f"arm identity mismatch: {name}")
        if any(receipt[key] != reference[key] for key in identity_fields):
            raise RuntimeError(f"artifact/source/input mismatch: {name}")
        if not receipt["rounds"]:
            raise RuntimeError(f"no timing rounds: {name}")
        if any(
            not row.get("cache_positions_restored") or row.get("trimmed_entry_count") != row.get("cache_entry_count")
            for row in receipt["rounds"]
        ):
            raise RuntimeError(f"speculative cache restore incomplete: {name}")
        if name != "stock" and any(
            row["hook_calls"] < 1 or row["tile4_dispatches"] < 1
            for row in receipt["rounds"]
        ):
            raise RuntimeError(f"candidate disengaged: {name}")
        if name == "stock" and any(
            row["hook_calls"] or row["tile4_dispatches"]
            for row in receipt["rounds"]
        ):
            raise RuntimeError("stock arm engaged fused candidate")
    if receipts["tile4"]["sort_threshold_used"] != receipts["tile8"]["sort_threshold_used"]:
        raise RuntimeError("tile4/tile8 sorted-gather policy differs")
    if receipts["tile4"]["sort_threshold_used"] <= 30:
        raise RuntimeError("M=3 candidate sorted-gather threshold did not clear 30")
    if receipts["stock"]["sort_threshold_used"] != receipts["stock"]["sort_threshold_original"]:
        raise RuntimeError("stock arm did not retain production sort threshold")

    counts = {len(receipt["rounds"]) for receipt in receipts.values()}
    if len(counts) != 1:
        raise RuntimeError("model arms have different measured round counts")
    count = counts.pop()
    comparisons = []
    for index in range(count):
        arrays = {}
        for name, receipt in receipts.items():
            row = receipt["rounds"][index]
            output_path = Path(row["logits_path"])
            if hashlib.sha256(output_path.read_bytes()).hexdigest() != row["logits_sha256"]:
                raise RuntimeError(f"full-logit receipt hash mismatch: {name} round {index}")
            arrays[name] = np.load(output_path, allow_pickle=False)
        if len({array.shape for array in arrays.values()}) != 1:
            raise RuntimeError(f"logit shape mismatch at round {index}")
        if not all(np.isfinite(array).all() for array in arrays.values()):
            raise RuntimeError(f"nonfinite logits at round {index}")
        tile_delta = arrays["tile8"].astype(np.float64) - arrays["tile4"]
        stock_delta = arrays["tile8"].astype(np.float64) - arrays["stock"]
        comparisons.append({
            "round": index,
            "full_logit_count": int(arrays["tile8"].size),
            "tile4_vs_tile8_exact": bool(np.array_equal(arrays["tile4"], arrays["tile8"])),
            "tile4_vs_tile8_max_abs": float(np.max(np.abs(tile_delta))),
            "stock_vs_tile8_max_abs": float(np.max(np.abs(stock_delta))),
            "stock_vs_tile8_rel_l2": float(
                np.linalg.norm(stock_delta.ravel())
                / max(np.linalg.norm(arrays["stock"].astype(np.float64).ravel()), 1e-30)
            ),
        })

    exact = all(row["tile4_vs_tile8_exact"] for row in comparisons)
    medians = {
        name: statistics.median(
            row["elapsed_ns"] for row in receipt["rounds"]
        ) / 1e6
        for name, receipt in receipts.items()
    }
    report = {
        "schema": "mlx2.m5-next-math.qwen4-m3-model-compare.v1",
        "status": (
            "tile4_tile8_parity_passed_stock_unqualified"
            if exact else "tile4_tile8_parity_failed"
        ),
        "identity": {key: reference[key] for key in identity_fields},
        "sort_thresholds": {
            name: receipt["sort_threshold_used"]
            for name, receipt in receipts.items()
        },
        "receipts": {name: str(path) for name, path in paths.items()},
        "comparisons": comparisons,
        "tile4_vs_tile8_full_logits_exact": exact,
        "median_verify_ms": medians,
        "tile8_speedup_vs_tile4_exploratory": medians["tile4"] / medians["tile8"],
        "timing_caveat": (
            "The candidate arms raise the sorted-gather threshold to admit M=3 "
            "while stock retains production policy. Separate model loads and short "
            "runs; useful as a model-path screen, "
            "not sufficient for serving selection or throughput claims."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "exact": exact,
        "out": str(args.out),
    }))
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
