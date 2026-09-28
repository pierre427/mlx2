#!/usr/bin/env python3
"""Qualify the guarded QSA stage-one PR-91 candidates on Apple GPU.

This producer requires a parent process to hold both host GPU lock files.  It
checks exact scorer and selector parity, proves candidate dispatch, and records
interleaved component and full-route timings.  It never changes runtime
defaults or selects a production route.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlx2.runtime.models import qwen4_qsa_stage1 as stage1

LOCKS = (Path("/Users/Shared/mlxuag/gpu.lock"), Path("/tmp/gpu.lock"))
SCHEMA = "mlx2.qwen4-qsa-stage1-pr91-qualification.v1"
BLOCK_TOPK = 512
COMPRESS_RATIO = 4
MIN_GAIN = 0.03
MAX_ROUTE_REGRESSION = 0.02


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command(*args: str) -> str:
    return subprocess.run(
        args,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _thermal() -> dict:
    result = subprocess.run(
        ("pmset", "-g", "therm"), check=False, capture_output=True, text=True
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _require_parent_locks() -> dict:
    if os.environ.get("MLX2_GPU_DUAL_FLOCK_HELD") != "1":
        raise SystemExit("qualification requires MLX2_GPU_DUAL_FLOCK_HELD=1")
    evidence = {}
    for path in LOCKS:
        if not path.is_file():
            raise SystemExit(f"GPU lock is not a regular file: {path}")
        with path.open("a+") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                evidence[str(path)] = "contended_by_parent"
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                raise SystemExit(f"GPU lock is not held by parent: {path}")
    return evidence


def _evaluate(value):
    values = value if isinstance(value, tuple) else (value,)
    mx.eval(*values)
    return value


def _array_equal(left: mx.array, right: mx.array) -> tuple[bool, float]:
    _evaluate((left, right))
    left_np = np.asarray(left)
    right_np = np.asarray(right)
    equal = bool(np.array_equal(left_np, right_np))
    if left_np.dtype.kind == "f":
        maximum = float(np.max(np.abs(left_np.astype(np.float64) - right_np)))
    else:
        maximum = 0.0 if equal else float("inf")
    return equal, maximum


def _set_arm(name: str) -> None:
    stage1._KEYS_STATIONARY = name in {"keys_stationary", "combined"}
    stage1._ONEPASS_TOPK = name in {"onepass_topk", "combined"}


def _score_case(rows: int, blocks: int, dtype, seed: int) -> dict:
    mx.random.seed(seed)
    q = mx.random.normal((1, rows, 4, 128), dtype=dtype)
    pooled = mx.random.normal((1, blocks, 128), dtype=dtype)
    reference = stage1._mpp_scores(q, pooled)
    candidate = stage1._mpp_keys_stationary_scores(q, pooled)
    equal, maximum = _array_equal(reference, candidate)
    return {
        "rows": rows,
        "blocks": blocks,
        "dtype": str(dtype),
        "exact": equal,
        "max_abs": maximum,
    }


def _selector_case(
    name: str,
    scores_np: np.ndarray,
    *,
    valid_count: int,
    topk: int,
) -> dict:
    scores = mx.array(scores_np.astype(np.float32))
    rows, blocks = scores_np.shape
    positions = mx.full((rows,), valid_count * COMPRESS_RATIO - 1, dtype=mx.int32)
    stage1._ONEPASS_TOPK = False
    reference = stage1._select_scores(
        scores, positions, topk=topk, compress_ratio=COMPRESS_RATIO
    )
    stage1._ONEPASS_TOPK = True
    candidate = stage1._select_scores(
        scores, positions, topk=topk, compress_ratio=COMPRESS_RATIO
    )
    equal, _ = _array_equal(reference, candidate)
    return {
        "name": name,
        "rows": rows,
        "blocks": blocks,
        "valid_count": valid_count,
        "topk": topk,
        "exact": equal,
        "reference_sha256": hashlib.sha256(np.asarray(reference).tobytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(np.asarray(candidate).tobytes()).hexdigest(),
    }


def _full_inputs(rows: int, blocks: int, dtype, seed: int):
    mx.random.seed(seed)
    q = mx.random.normal((1, rows, 4, 128), dtype=dtype)
    pooled = mx.random.normal((1, blocks, 128), dtype=dtype)
    positions = mx.full(
        (1, rows), blocks * COMPRESS_RATIO - 1, dtype=mx.int32
    )
    _evaluate((q, pooled, positions))
    return q, pooled, positions


def _full_call(name: str, values):
    _set_arm(name)
    q, pooled, positions = values
    return stage1.qsa_stage1_select(
        q,
        pooled,
        positions,
        block_topk=BLOCK_TOPK,
        compress_ratio=COMPRESS_RATIO,
    )


def _timed(function) -> float:
    started = time.perf_counter_ns()
    _evaluate(function())
    return (time.perf_counter_ns() - started) / 1.0e6


def _interleaved(arms: dict[str, object], *, warmups: int, repeats: int) -> dict:
    names = list(arms)
    for _ in range(warmups):
        for name in names:
            _evaluate(arms[name]())
    samples = {name: [] for name in names}
    orders = []
    for repeat in range(repeats):
        offset = repeat % len(names)
        order = names[offset:] + names[:offset]
        if (repeat // len(names)) % 2:
            order.reverse()
        orders.append(order)
        for name in order:
            samples[name].append(_timed(arms[name]))
    report = {"orders": orders, "arms": {}}
    baseline = statistics.median(samples[names[0]])
    for name in names:
        ordered = sorted(samples[name])
        median = statistics.median(ordered)
        report["arms"][name] = {
            "samples_ms": samples[name],
            "median_ms": median,
            "minimum_ms": min(ordered),
            "maximum_ms": max(ordered),
            "gain_vs_baseline": 1.0 - median / baseline,
        }
    return report


def _benchmark_case(rows: int, blocks: int, warmups: int, repeats: int) -> dict:
    values = _full_inputs(rows, blocks, mx.bfloat16, 7001 + rows + blocks)
    q, pooled, positions = values
    scores = stage1._mpp_scores(q, pooled)
    _evaluate(scores)

    exact = {}
    _set_arm("baseline")
    baseline = _evaluate(_full_call("baseline", values))
    for name in ("keys_stationary", "onepass_topk", "combined"):
        candidate = _evaluate(_full_call(name, values))
        agrees, _ = _array_equal(baseline, candidate)
        exact[name] = agrees

    component = {
        "scorer": _interleaved(
            {
                "baseline": lambda: stage1._mpp_scores(q, pooled),
                "keys_stationary": lambda: stage1._mpp_keys_stationary_scores(q, pooled),
            },
            warmups=warmups,
            repeats=repeats,
        ),
        "selector": _interleaved(
            {
                "baseline": lambda: _selector_dispatch(False, scores, positions),
                "onepass_topk": lambda: _selector_dispatch(True, scores, positions),
            },
            warmups=warmups,
            repeats=repeats,
        ),
    }
    route = _interleaved(
        {
            name: (lambda name=name: _full_call(name, values))
            for name in ("baseline", "keys_stationary", "onepass_topk", "combined")
        },
        warmups=warmups,
        repeats=repeats,
    )
    return {
        "rows": rows,
        "blocks": blocks,
        "tokens": blocks * COMPRESS_RATIO,
        "dtype": "bfloat16",
        "route_exact": exact,
        "thermal_before": _thermal(),
        "component": component,
        "route": route,
        "thermal_after": _thermal(),
    }


def _selector_dispatch(candidate: bool, scores: mx.array, positions: mx.array):
    stage1._ONEPASS_TOPK = candidate
    return stage1._select_scores(
        scores,
        positions.reshape(-1),
        topk=BLOCK_TOPK + stage1._EXACT_BAND_EXTRA,
        compress_ratio=COMPRESS_RATIO,
    )


def _decision(
    *, exact: bool, observed: bool, component_gains: list[float], route_gains: list[float]
) -> dict:
    component_gain = statistics.median(component_gains)
    route_gain = statistics.median(route_gains)
    worst_route_gain = min(route_gains)
    component_gate_passed = (
        exact
        and observed
        and component_gain >= MIN_GAIN
        and worst_route_gain >= -MAX_ROUTE_REGRESSION
    )
    qualified = component_gate_passed and route_gain >= MIN_GAIN
    return {
        "implemented": True,
        "exact": exact,
        "observed_in_qualification": observed,
        "component_median_gain": component_gain,
        "component_gate_passed": component_gate_passed,
        "route_median_gain": route_gain,
        "worst_route_gain": worst_route_gain,
        "qualification_gate": {
            "minimum_component_gain": MIN_GAIN,
            "minimum_route_gain": MIN_GAIN,
            "maximum_route_regression": MAX_ROUTE_REGRESSION,
        },
        "qualified": qualified,
        "selected": False,
        "observed_used_in_production": False,
    }


def run(args: argparse.Namespace) -> dict:
    lock_evidence = _require_parent_locks()
    if _command("git", "status", "--porcelain"):
        raise SystemExit("qualification requires a clean Git worktree")
    mx.set_default_device(mx.gpu)
    if not stage1.qsa_stage1_kernel_available():
        raise SystemExit("Apple GPU Metal kernels are unavailable")

    stage1.qsa_stage1_candidate_status(reset=True)
    if args.mode == "quick":
        geometries = ((64, 4096),)
    else:
        geometries = ((64, 16256), (512, 16256), (64, 24576))

    score_parity = [
        _score_case(rows, blocks, dtype, 101 + rows + blocks)
        for rows, blocks in ((64, 2049), (65, 4096), (128, 16256))
        for dtype in (mx.float16, mx.bfloat16)
    ]
    rng = np.random.default_rng(991)
    selector_parity = [
        _selector_case(
            "distinct",
            rng.normal(size=(3, 4096)).astype(np.float32),
            valid_count=4096,
            topk=544,
        ),
        _selector_case(
            "all_ties_overflow",
            np.zeros((2, 4096), dtype=np.float32),
            valid_count=4096,
            topk=544,
        ),
        _selector_case(
            "patterned_ties_overflow",
            np.tile(np.arange(7, dtype=np.float32), (2, 4096 // 7 + 1))[:, :4096],
            valid_count=4096,
            topk=512,
        ),
        _selector_case(
            "partial_validity_padding",
            rng.normal(size=(2, 4096)).astype(np.float32),
            valid_count=1500,
            topk=544,
        ),
    ]
    benchmarks = [
        _benchmark_case(rows, blocks, args.warmups, args.repeats)
        for rows, blocks in geometries
    ]
    counts = stage1.qsa_stage1_candidate_status()["runtime_counts"]

    score_exact = all(row["exact"] for row in score_parity)
    selector_exact = all(row["exact"] for row in selector_parity)
    route_exact = all(
        value
        for row in benchmarks
        for value in row["route_exact"].values()
    )
    keys = _decision(
        exact=score_exact and route_exact,
        observed=counts.get("keys_stationary_dispatches", 0) > 0,
        component_gains=[
            row["component"]["scorer"]["arms"]["keys_stationary"][
                "gain_vs_baseline"
            ]
            for row in benchmarks
        ],
        route_gains=[
            row["route"]["arms"]["keys_stationary"]["gain_vs_baseline"]
            for row in benchmarks
        ],
    )
    onepass = _decision(
        exact=selector_exact and route_exact,
        observed=counts.get("onepass_topk_dispatches", 0) > 0,
        component_gains=[
            row["component"]["selector"]["arms"]["onepass_topk"][
                "gain_vs_baseline"
            ]
            for row in benchmarks
        ],
        route_gains=[
            row["route"]["arms"]["onepass_topk"]["gain_vs_baseline"]
            for row in benchmarks
        ],
    )
    combined_gains = [
        row["route"]["arms"]["combined"]["gain_vs_baseline"]
        for row in benchmarks
    ]
    combined = {
        "exact": route_exact,
        "median_route_gain": statistics.median(combined_gains),
        "worst_route_gain": min(combined_gains),
        "qualifies_for_selection": (
            keys["qualified"]
            and onepass["qualified"]
            and min(combined_gains) >= MIN_GAIN
        ),
        "selected": False,
        "observed_used_in_production": False,
    }
    report = {
        "schema": SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "commit": _command("git", "rev-parse", "HEAD"),
            "origin_main": _command("git", "rev-parse", "origin/main"),
            "merge_base": _command("git", "merge-base", "HEAD", "origin/main"),
            "branch": _command("git", "branch", "--show-current"),
            "module_sha256": _sha256(
                ROOT / "src/mlx2/runtime/models/qwen4_qsa_stage1.py"
            ),
            "producer_sha256": _sha256(Path(__file__).resolve()),
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mlx_version": version("mlx"),
            "device": mx.device_info(),
            "default_device": str(mx.default_device()),
            "lock_evidence": lock_evidence,
        },
        "controls": {
            "mode": args.mode,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "timing_order": "rotating interleaved arms with alternating direction",
            "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32", "unset_default_tf32"),
            "runtime_defaults_changed": False,
        },
        "parity": {"scorer": score_parity, "selector": selector_parity},
        "benchmarks": benchmarks,
        "candidate_runtime_counts": counts,
        "qualification": {
            "keys_stationary": keys,
            "onepass_topk": onepass,
            "combined": combined,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("quick", "full"), default="full")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["qualification"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
