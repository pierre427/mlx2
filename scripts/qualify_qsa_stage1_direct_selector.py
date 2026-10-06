#!/usr/bin/env python3
"""Qualify the default-off Qwen4 QSA direct selector candidates on Apple GPU.

Run this only through ``/Users/Shared/mlxuag/gpuq-reference.sh``.  The script
fails closed unless the process owns both queue locks, exercises the production
dispatch rather than probe-local kernels, and leaves the runtime default off.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlx2.runtime.models import qwen4_qsa_stage1 as stage1

SCHEMA = "mlx2.qwen4-qsa-stage1-direct-selector-qualification.v1"
COMPRESS_RATIO = 4
BLOCK_TOPK = 512
CANDIDATE_TOPK = BLOCK_TOPK + stage1._EXACT_BAND_EXTRA
MIN_COMPONENT_GAIN = {"direct8": 0.0, "direct4": 0.03}
MIN_ROUTE_GAIN = {"direct8": 0.0, "direct4": 0.03}
MAX_CELL_REGRESSION = 0.02


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command(*args: str) -> str:
    result = subprocess.run(
        args, cwd=ROOT, check=False, capture_output=True, text=True, timeout=30
    )
    return (result.stdout + result.stderr).strip()


def _lock_owner(path: str) -> dict:
    owner_path = Path(path) / "owner.json"
    if not owner_path.is_file():
        raise SystemExit(f"missing GPU lock owner: {owner_path}")
    return json.loads(owner_path.read_text())


def _prove_gpu_ownership() -> dict:
    expected_lease = os.environ.get("GPUQ_LEASE")
    expected_session = os.environ.get("GPUQ_SESSION")
    if not expected_lease:
        raise SystemExit("GPUQ_LEASE does not own both locks")
    if not expected_session:
        raise SystemExit("GPUQ_SESSION does not own both locks")
    shared = _lock_owner("/Users/Shared/mlxuag/gpu.lock")
    temporary = _lock_owner("/tmp/gpu.lock")
    if any(
        owner.get("lease_id") != expected_lease for owner in (shared, temporary)
    ):
        raise SystemExit("GPUQ_LEASE does not own both locks")
    if any(
        owner.get("session") != expected_session for owner in (shared, temporary)
    ):
        raise SystemExit("GPUQ_SESSION does not own both locks")
    return {"shared": shared, "temporary": temporary}


def _host_snapshot() -> dict:
    return {
        "swapusage": _command("sysctl", "-n", "vm.swapusage"),
        "vm_stat": _command("vm_stat"),
        "thermal": _command("pmset", "-g", "therm"),
        "active_memory_bytes": int(mx.get_active_memory()),
        "cache_memory_bytes": int(mx.get_cache_memory()),
    }


def _eval(value):
    values = value if isinstance(value, tuple) else (value,)
    mx.eval(*values)
    return value


def _equal(left: mx.array, right: mx.array) -> bool:
    _eval((left, right))
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def _set_selector(name: str) -> None:
    stage1._DIRECT_SELECTOR = name
    stage1._ONEPASS_TOPK = False


def _select(name: str, scores: mx.array, positions: mx.array, topk: int) -> mx.array:
    _set_selector(name)
    return stage1._select_scores(
        scores,
        positions,
        topk=topk,
        compress_ratio=COMPRESS_RATIO,
    )


def _timed(function: Callable[[], mx.array]) -> float:
    started = time.perf_counter_ns()
    _eval(function())
    return (time.perf_counter_ns() - started) / 1.0e6


def _interleaved(
    arms: dict[str, Callable[[], mx.array]], *, warmups: int, repeats: int
) -> dict:
    names = list(arms)
    for repeat in range(warmups):
        order = names[repeat % len(names) :] + names[: repeat % len(names)]
        for name in order:
            _eval(arms[name]())
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
    baseline = statistics.median(samples["off"])
    return {
        "orders": orders,
        "arms": {
            name: {
                "samples_ms": values,
                "median_ms": statistics.median(values),
                "minimum_ms": min(values),
                "maximum_ms": max(values),
                "gain_vs_off": 1.0 - statistics.median(values) / baseline,
            }
            for name, values in samples.items()
        },
    }


def _score_case(kind: str, rows: int, blocks: int) -> mx.array:
    if kind == "zeros":
        return mx.zeros((rows, blocks), dtype=mx.float32)
    mx.random.seed(1729 + rows + blocks)
    scores = mx.random.uniform(shape=(rows, blocks), dtype=mx.float32)
    if kind == "ties":
        return mx.floor(scores * 16.0) / 16.0
    return scores


def _correctness() -> list[dict]:
    cases = (
        ("distinct", 3, 4097, 512, [4097, 4097, 4097]),
        ("ties", 5, 4097, 512, [4097] * 5),
        ("zeros", 2, 4097, 512, [4097, 4097]),
        ("ties", 4, 2049, 544, [2049, 1900, 900, 0]),
        ("distinct", 3, 2049, 512, [0, 128, 511]),
    )
    results = []
    for kind, rows, blocks, topk, valid_counts in cases:
        scores = _score_case(kind, rows, blocks)
        positions = mx.array(
            [max(0, count * COMPRESS_RATIO - 1) for count in valid_counts],
            dtype=mx.int32,
        )
        _eval((scores, positions))
        reference = _eval(_select("off", scores, positions, topk))
        matches = {
            name: _equal(reference, _select(name, scores, positions, topk))
            for name in ("direct8", "direct4")
        }
        results.append(
            {
                "kind": kind,
                "rows": rows,
                "blocks": blocks,
                "topk": topk,
                "valid_counts": valid_counts,
                "matches": matches,
                "reference_sha256": hashlib.sha256(
                    np.asarray(reference).tobytes()
                ).hexdigest(),
            }
        )
        del scores, positions, reference
        gc.collect()
        mx.clear_cache()
    return results


def _full_inputs(rows: int, blocks: int):
    mx.random.seed(90210 + rows + blocks)
    query = mx.random.uniform(
        low=-1.0, high=1.0, shape=(1, rows, 4, 128), dtype=mx.bfloat16
    )
    pooled = mx.random.uniform(
        low=-1.0, high=1.0, shape=(1, blocks, 128), dtype=mx.bfloat16
    )
    positions = mx.full((1, rows), blocks * COMPRESS_RATIO - 1, dtype=mx.int32)
    return _eval((query, pooled, positions))


def _route(name: str, values) -> mx.array:
    _set_selector(name)
    query, pooled, positions = values
    return stage1.qsa_stage1_select(
        query,
        pooled,
        positions,
        block_topk=BLOCK_TOPK,
        compress_ratio=COMPRESS_RATIO,
    )


def _benchmark(rows: int, blocks: int, warmups: int, repeats: int) -> dict:
    values = _full_inputs(rows, blocks)
    query, pooled, positions = values
    _set_selector("off")
    scores = _eval(stage1._mpp_scores(query, pooled))
    reference = _eval(_route("off", values))
    exact = {
        name: _equal(reference, _route(name, values)) for name in ("direct8", "direct4")
    }
    selector_arms = {
        name: (
            lambda name=name, scores=scores, positions=positions: _select(
                name, scores, positions.reshape(-1), CANDIDATE_TOPK
            )
        )
        for name in ("off", "direct8", "direct4")
    }
    route_arms = {
        name: (lambda name=name, values=values: _route(name, values))
        for name in ("off", "direct8", "direct4")
    }
    result = {
        "rows": rows,
        "blocks": blocks,
        "tokens_at_ratio_4": blocks * COMPRESS_RATIO,
        "route_exact": exact,
        "selector": _interleaved(selector_arms, warmups=warmups, repeats=repeats),
        "route": _interleaved(route_arms, warmups=warmups, repeats=repeats),
    }
    del query, pooled, positions, scores, reference, values
    gc.collect()
    mx.clear_cache()
    return result


def _memory(rows: int, blocks: int) -> dict:
    scores = _eval(_score_case("distinct", rows, blocks))
    positions = _eval(mx.full((rows,), blocks * COMPRESS_RATIO - 1, dtype=mx.int32))
    arms = {}
    for name in ("off", "direct8", "direct4"):
        gc.collect()
        mx.clear_cache()
        active = int(mx.get_active_memory())
        mx.reset_peak_memory()
        elapsed = _timed(
            lambda name=name, scores=scores, positions=positions: _select(
                name, scores, positions, BLOCK_TOPK
            )
        )
        peak = int(mx.get_peak_memory())
        arms[name] = {
            "elapsed_ms": elapsed,
            "active_before_bytes": active,
            "peak_bytes": peak,
            "incremental_peak_bytes": max(0, peak - active),
        }
    del scores, positions
    gc.collect()
    mx.clear_cache()
    return {
        "rows": rows,
        "blocks": blocks,
        "input_score_bytes": rows * blocks * 4,
        "arms": arms,
    }


def _decision(name: str, benchmarks: list[dict], exact: bool, counts: dict) -> dict:
    selector_gains = [
        row["selector"]["arms"][name]["gain_vs_off"] for row in benchmarks
    ]
    route_gains = [row["route"]["arms"][name]["gain_vs_off"] for row in benchmarks]
    observed = counts.get(f"{name}_topk_dispatches", 0) > 0
    qualified = (
        exact
        and observed
        and statistics.median(selector_gains) >= MIN_COMPONENT_GAIN[name]
        and statistics.median(route_gains) >= MIN_ROUTE_GAIN[name]
        and min(route_gains) >= -MAX_CELL_REGRESSION
    )
    return {
        "implemented": True,
        "exact": exact,
        "observed_in_qualification": observed,
        "dispatch_count": counts.get(f"{name}_topk_dispatches", 0),
        "selector_gains": selector_gains,
        "selector_median_gain": statistics.median(selector_gains),
        "route_gains": route_gains,
        "route_median_gain": statistics.median(route_gains),
        "worst_route_gain": min(route_gains),
        "gates": {
            "minimum_component_gain": MIN_COMPONENT_GAIN[name],
            "minimum_route_gain": MIN_ROUTE_GAIN[name],
            "maximum_cell_regression": MAX_CELL_REGRESSION,
        },
        "qualified_synthetic_stage1": qualified,
        "selected": False,
        "observed_used_in_production": False,
    }


def run(args: argparse.Namespace) -> dict:
    lock_evidence = _prove_gpu_ownership()
    mx.set_default_device(mx.gpu)
    if not stage1.qsa_stage1_kernel_available():
        raise SystemExit("Apple GPU Metal kernels are unavailable")
    if os.environ.get("MLX_QWEN4_QSA_STAGE1_DIRECT_SELECTOR", "off") != "off":
        raise SystemExit(
            "qualification requires the imported runtime default to be off"
        )

    report = {
        "schema": SCHEMA,
        "status": "running",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "commit": _command("git", "rev-parse", "HEAD"),
            "branch": _command("git", "branch", "--show-current"),
            "origin_main": _command("git", "rev-parse", "origin/main"),
            "worktree_dirty": bool(_command("git", "status", "--porcelain")),
            "stage1_sha256": _sha256(
                ROOT / "src/mlx2/runtime/models/qwen4_qsa_stage1.py"
            ),
            "selector_sha256": _sha256(
                ROOT / "src/mlx2/runtime/models/qwen4_qsa_selector.py"
            ),
            "qualifier_sha256": _sha256(Path(__file__).resolve()),
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mlx_version": version("mlx"),
            "device": mx.device_info(),
            "lock_evidence": lock_evidence,
        },
        "controls": {
            "mode": args.mode,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "compress_ratio": COMPRESS_RATIO,
            "block_topk": BLOCK_TOPK,
            "candidate_topk": CANDIDATE_TOPK,
            "runtime_default": "off",
            "runtime_default_changed": False,
            "direct8_route_gate_authority": (
                "user accepted the previously observed 2.87 percent complete-stage "
                "gain on 2026-10-04; performance is recorded separately from "
                "qualification, and both component and complete stage are required "
                "not to regress materially"
            ),
            "timing_order": "rotating interleaved arms with alternating direction",
        },
        "host_before": _host_snapshot(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    try:
        stage1.qsa_stage1_candidate_status(reset=True)
        report["correctness"] = _correctness()
        geometries = (
            ((64, 4096),)
            if args.mode == "quick"
            else ((64, 16256), (512, 16256), (512, 24576))
        )
        report["benchmarks"] = [
            _benchmark(rows, blocks, args.warmups, args.repeats)
            for rows, blocks in geometries
        ]
        if args.mode == "full":
            report["memory"] = [
                _memory(4096, 32768),
                _memory(8192, 16256),
                _memory(8192, 32768),
            ]
        counts = stage1.qsa_stage1_candidate_status()["runtime_counts"]
        correctness_exact = all(
            all(row["matches"].values()) for row in report["correctness"]
        )
        decisions = {}
        for name in ("direct8", "direct4"):
            route_exact = all(row["route_exact"][name] for row in report["benchmarks"])
            decisions[name] = _decision(
                name,
                report["benchmarks"],
                correctness_exact and route_exact,
                counts,
            )
        report["candidate_runtime_counts"] = counts
        report["qualification"] = decisions
        report["status"] = (
            "passed"
            if all(
                decision["qualified_synthetic_stage1"]
                for decision in decisions.values()
            )
            else "failed"
        )
    except BaseException as exc:
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _set_selector("off")
        report["host_after"] = _host_snapshot()
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
