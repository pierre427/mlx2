#!/usr/bin/env python3
"""Synthetic GPU benchmark of the GVR-style QSA selector against direct8/direct4.

Run only through ``scripts/run_with_gpu_locks.py``; the script refuses to run
unless its parent process owns both GPU locks.  No model weights are loaded:
every input is a synthetic tensor.  Every timed cell is first checked for
exact equality across all arms (and against the NumPy selector law on sampled
rows), and GVR's per-row path diagnostics are reported as fallback rates.

Sections (``--sections``):

* ``selector``: selector-only timing over rows x blocks at k=512, plus the
  544-to-512 exact-band refine shape;
* ``stage1``: the complete production ``qsa_stage1_select`` route with the
  selector switched per arm (MPP scorer, 544 candidates, exact repair);
* ``adversarial``: selector-only timing and fallback rates on adversarial
  score distributions at one shape.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlx2.runtime.models import qwen4_qsa_gvr_reference as gvr_ref
from mlx2.runtime.models import qwen4_qsa_selector as selector
from mlx2.runtime.models import qwen4_qsa_stage1 as stage1

SCHEMA = "mlx2.qwen4-qsa-stage1-gvr-selector-bench.v1"
RATIO = 4
TOPK = 512
CANDIDATE_TOPK = TOPK + stage1._EXACT_BAND_EXTRA
LOCKS = ("/Users/Shared/mlxuag/gpu.lock", "/tmp/gpu.lock")


def _command(*args: str) -> str:
    result = subprocess.run(
        args, cwd=ROOT, check=False, capture_output=True, text=True, timeout=30
    )
    return (result.stdout + result.stderr).strip()


def _prove_lease() -> list[dict]:
    owners = []
    for lock in LOCKS:
        path = Path(lock) / "owner.json"
        if not path.is_file():
            raise SystemExit(f"GPU lock not held: {path}")
        owner = json.loads(path.read_text())
        if int(owner.get("pid", -1)) != os.getppid():
            raise SystemExit(f"GPU lock {lock} is owned by {owner}")
        owners.append(owner)
    return owners


def _device_info() -> dict | None:
    for owner in (mx, getattr(mx, "metal", None)):
        function = getattr(owner, "device_info", None)
        if function is not None:
            try:
                return {key: str(value) for key, value in function().items()}
            except Exception:  # noqa: BLE001 - informational only
                return None
    return None


def _swapouts() -> int:
    for line in _command("vm_stat").splitlines():
        if line.startswith("Swapouts"):
            return int(line.split(":")[1].strip().rstrip("."))
    return -1


def _host() -> dict:
    return {
        "swapusage": _command("sysctl", "-n", "vm.swapusage"),
        "swapouts": _swapouts(),
        "thermal": _command("pmset", "-g", "therm"),
        "active_memory_bytes": int(mx.get_active_memory()),
    }


def _gvr(width: int, capacity: int):
    def run(scores, positions, *, topk, compress_ratio):
        return selector.select_scores_gvr(
            scores,
            positions,
            topk=topk,
            compress_ratio=compress_ratio,
            width=width,
            capacity=capacity,
        )

    return run


SELECTOR_ARMS: dict[str, Callable] = {
    "direct8": selector.select_scores_direct8,
    "direct4": selector.select_scores_direct4,
    "gvr": _gvr(1024, 2048),
    "gvr_c1024": _gvr(1024, 1024),
    "gvr_w512": _gvr(512, 2048),
}
GVR_VARIANTS = {"gvr": (1024, 2048), "gvr_c1024": (1024, 1024), "gvr_w512": (512, 2048)}


def _summary(values: list[float]) -> dict:
    array = np.asarray(values)
    return {
        "samples_ms": values,
        "p50_ms": float(np.percentile(array, 50)),
        "p99_ms": float(np.percentile(array, 99)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def _interleaved(arms: dict[str, Callable[[], mx.array]], warmups: int, repeats: int):
    names = list(arms)
    for repeat in range(warmups):
        for name in names[repeat % len(names) :] + names[: repeat % len(names)]:
            mx.eval(arms[name]())
    samples = {name: [] for name in names}
    for repeat in range(repeats):
        offset = repeat % len(names)
        order = names[offset:] + names[:offset]
        if (repeat // len(names)) % 2:
            order.reverse()
        for name in order:
            started = time.perf_counter_ns()
            mx.eval(arms[name]())
            samples[name].append((time.perf_counter_ns() - started) / 1.0e6)
    result = {name: _summary(values) for name, values in samples.items()}
    base = result["direct8"]["p50_ms"]
    for arm in result.values():
        arm["speedup_vs_direct8"] = base / arm["p50_ms"]
    if "direct4" in result:
        for arm in result.values():
            arm["speedup_vs_direct4"] = result["direct4"]["p50_ms"] / arm["p50_ms"]
    return result


def _path_rates(diag: np.ndarray) -> dict:
    paths = diag[:, 0]
    rows = max(1, paths.size)
    sampled = diag[paths == gvr_ref.PATH_SAMPLED]
    return {
        "rows": int(paths.size),
        **{
            f"{gvr_ref.PATH_NAMES[code]}_rows": int(np.count_nonzero(paths == code))
            for code in gvr_ref.PATH_NAMES
        },
        "fallback_rate": float(np.count_nonzero(paths == gvr_ref.PATH_FALLBACK) / rows),
        "sampled_candidates_median": (
            float(np.median(sampled[:, 1])) if sampled.size else None
        ),
        "sampled_candidates_max": int(sampled[:, 1].max()) if sampled.size else None,
        "threshold_index_histogram": {
            str(index): int(np.count_nonzero(sampled[:, 2] == index))
            for index in range(gvr_ref.NUM_THRESHOLDS)
        },
    }


def _check_cell(
    scores: mx.array, positions: mx.array, topk: int, arms: list[str], law_rows: int
) -> dict:
    outputs = {
        name: SELECTOR_ARMS[name](scores, positions, topk=topk, compress_ratio=RATIO)
        for name in arms
    }
    mx.eval(*outputs.values())
    reference = np.asarray(outputs["direct8"])
    matches = {
        name: bool(np.array_equal(reference, np.asarray(value)))
        for name, value in outputs.items()
    }
    rows = int(scores.shape[0])
    pick = np.unique(np.linspace(0, rows - 1, num=min(rows, law_rows)).astype(int))
    host_scores = np.asarray(scores[mx.array(pick)])
    host_positions = np.asarray(positions)[pick]
    law = gvr_ref.selector_law(
        host_scores, host_positions, topk=topk, compress_ratio=RATIO
    )
    matches["selector_law_sampled_rows"] = bool(np.array_equal(law, reference[pick]))
    paths = {}
    for name in arms:
        if name not in GVR_VARIANTS:
            continue
        width, capacity = GVR_VARIANTS[name]
        _, diag = selector.select_scores_gvr(
            scores,
            positions,
            topk=topk,
            compress_ratio=RATIO,
            width=width,
            capacity=capacity,
            return_diagnostics=True,
        )
        diag = np.asarray(diag)
        paths[name] = _path_rates(diag)
        _, expected = gvr_ref.gvr_reference_select(
            host_scores,
            host_positions,
            topk=topk,
            compress_ratio=RATIO,
            width=width,
            capacity=capacity,
        )
        paths[name]["diagnostics_match_reference_sampled_rows"] = bool(
            np.array_equal(expected, diag[pick])
        )
    return {"matches": matches, "paths": paths}


def _selector_calls(arms, scores, positions) -> dict[str, Callable[[], mx.array]]:
    def bind(name):
        return lambda: SELECTOR_ARMS[name](
            scores, positions, topk=TOPK, compress_ratio=RATIO
        )

    return {name: bind(name) for name in arms}


def _route_calls(arms, query, pooled, positions) -> dict[str, Callable[[], mx.array]]:
    def bind(arm):
        return lambda: _route(arm, query, pooled, positions)

    return {arm: bind(arm) for arm in arms}


def _uniform(rows: int, blocks: int, seed: int) -> mx.array:
    mx.random.seed(seed)
    return mx.random.uniform(shape=(rows, blocks), dtype=mx.float32)


def _release(*values) -> None:
    del values
    gc.collect()
    mx.clear_cache()


def selector_section(args) -> list[dict]:
    cells = []
    arms = list(SELECTOR_ARMS)
    for rows in args.rows:
        for blocks in args.blocks:
            scores = _uniform(rows, blocks, 1729 + rows + blocks)
            positions = mx.full((rows,), blocks * RATIO - 1, dtype=mx.int32)
            mx.eval(scores, positions)
            check = _check_cell(scores, positions, TOPK, arms, args.law_rows)
            timing = _interleaved(
                _selector_calls(arms, scores, positions),
                args.warmups,
                args.repeats,
            )
            cell = {"rows": rows, "blocks": blocks, "topk": TOPK, **check}
            cell["arms"] = timing
            cells.append(cell)
            print(_brief("selector", cell), flush=True)
            _release(scores, positions)
    # The exact-band refine call: 544 exact candidates down to 512.
    for rows in args.rows:
        mx.random.seed(77 + rows)
        scores = mx.random.uniform(shape=(rows, CANDIDATE_TOPK), dtype=mx.float32)
        positions = mx.full((rows,), CANDIDATE_TOPK * RATIO - 1, dtype=mx.int32)
        mx.eval(scores, positions)
        check = _check_cell(scores, positions, TOPK, arms, args.law_rows)
        timing = _interleaved(
            _selector_calls(arms, scores, positions),
            args.warmups,
            args.repeats,
        )
        cell = {"rows": rows, "blocks": CANDIDATE_TOPK, "topk": TOPK, "refine": True}
        cell.update(check)
        cell["arms"] = timing
        cells.append(cell)
        print(_brief("refine", cell), flush=True)
        _release(scores, positions)
    return cells


def _adversarial_scores(kind: str, rows: int, blocks: int) -> tuple[mx.array, mx.array]:
    full = mx.full((rows,), blocks * RATIO - 1, dtype=mx.int32)
    mx.random.seed(4242 + len(kind))
    base = mx.random.uniform(shape=(rows, blocks), dtype=mx.float32)
    if kind == "uniform":
        return base, full
    if kind == "relu_normal":
        normal = mx.random.normal(shape=(rows, blocks, 4), dtype=mx.float32)
        return mx.sum(mx.maximum(normal, 0), axis=-1) / 11.3137, full
    if kind == "ties16":
        return mx.floor(base * 16.0) / 16.0, full
    if kind == "ties4096":
        return mx.floor(base * 4096.0) / 4096.0, full
    if kind == "all_equal":
        return mx.full((rows, blocks), 0.5, dtype=mx.float32), full
    if kind == "ascending":
        return mx.broadcast_to(
            mx.arange(blocks, dtype=mx.float32)[None, :], (rows, blocks)
        ) + 0.0, full
    if kind == "outlier_spikes":
        spikes = mx.random.uniform(shape=(rows, blocks)) < (64.0 / blocks)
        return mx.where(spikes, base * 1.0e30, base * 1.0e-3), full
    if kind == "sentinel_tail":
        column = mx.arange(blocks)[None, :]
        limit = mx.random.randint(600, blocks, shape=(rows, 1))
        return mx.where(column < limit, base, -mx.inf), full
    if kind == "causal_positions":
        positions = (
            mx.arange(rows, dtype=mx.int32) * ((blocks * RATIO) // rows)
            + (blocks * RATIO) // rows
            - 1
        )
        return base, positions
    raise ValueError(kind)


ADVERSARIAL = (
    "uniform",
    "relu_normal",
    "ties16",
    "ties4096",
    "all_equal",
    "ascending",
    "outlier_spikes",
    "sentinel_tail",
    "causal_positions",
)


def adversarial_section(args) -> list[dict]:
    cells = []
    arms = ["direct8", "direct4", "gvr"]
    rows, blocks = args.adversarial_shape
    for kind in ADVERSARIAL:
        scores, positions = _adversarial_scores(kind, rows, blocks)
        mx.eval(scores, positions)
        check = _check_cell(scores, positions, TOPK, arms, args.law_rows)
        timing = _interleaved(
            _selector_calls(arms, scores, positions),
            args.warmups,
            args.repeats,
        )
        cell = {"kind": kind, "rows": rows, "blocks": blocks, "topk": TOPK, **check}
        cell["arms"] = timing
        cells.append(cell)
        print(_brief(f"adv:{kind}", cell), flush=True)
        _release(scores, positions)
    return cells


def _route(arm: str, query, pooled, positions) -> mx.array:
    stage1._ONEPASS_TOPK = False
    stage1._DIRECT_SELECTOR = arm
    return stage1.qsa_stage1_select(
        query, pooled, positions, block_topk=TOPK, compress_ratio=RATIO
    )


def stage1_section(args) -> list[dict]:
    cells = []
    arms = ["off", "direct8", "direct4", "gvr"]
    for rows in args.rows:
        for blocks in args.blocks:
            mx.random.seed(90210 + rows + blocks)
            query = mx.random.uniform(
                low=-1.0, high=1.0, shape=(1, rows, 4, 128), dtype=mx.bfloat16
            )
            pooled = mx.random.uniform(
                low=-1.0, high=1.0, shape=(1, blocks, 128), dtype=mx.bfloat16
            )
            positions = mx.full((1, rows), blocks * RATIO - 1, dtype=mx.int32)
            mx.eval(query, pooled, positions)
            route = stage1.qsa_stage1_route(query, pooled, block_topk=TOPK)
            outputs = {arm: _route(arm, query, pooled, positions) for arm in arms}
            mx.eval(*outputs.values())
            reference = np.asarray(outputs["off"])
            matches = {
                arm: bool(np.array_equal(reference, np.asarray(value)))
                for arm, value in outputs.items()
            }
            approximate = stage1._mpp_scores(query, pooled)
            _, diag = selector.select_scores_gvr(
                approximate,
                positions.reshape(-1),
                topk=CANDIDATE_TOPK,
                compress_ratio=RATIO,
                return_diagnostics=True,
            )
            primary_paths = _path_rates(np.asarray(diag))
            del approximate, diag, outputs
            stage1.qsa_stage1_candidate_status(reset=True)
            timing = _interleaved(
                _route_calls(arms, query, pooled, positions),
                args.warmups,
                args.repeats,
            )
            counts = stage1.qsa_stage1_candidate_status(reset=True)["runtime_counts"]
            stage1._DIRECT_SELECTOR = "off"
            cell = {
                "rows": rows,
                "blocks": blocks,
                "topk": TOPK,
                "candidate_topk": CANDIDATE_TOPK,
                "route": route,
                "matches": matches,
                "gvr_primary_paths": primary_paths,
                "dispatch_counts": counts,
                "arms": timing,
            }
            cells.append(cell)
            print(_brief("stage1", cell), flush=True)
            _release(query, pooled, positions)
    return cells


def _brief(label: str, cell: dict) -> str:
    timing = " ".join(
        f"{name}={arm['p50_ms']:.3f}" for name, arm in cell["arms"].items()
    )
    paths = cell.get("paths", {}).get("gvr") or cell.get("gvr_primary_paths") or {}
    return (
        f"[{label}] rows={cell['rows']} blocks={cell['blocks']} "
        f"exact={all(cell['matches'].values())} "
        f"fallback={paths.get('fallback_rate')} p50ms {timing}"
    )


def _all_exact(cells: list[dict]) -> bool:
    for cell in cells:
        if not all(cell["matches"].values()):
            return False
        for paths in cell.get("paths", {}).values():
            if not paths["diagnostics_match_reference_sampled_rows"]:
                return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sections", default="selector,adversarial,stage1")
    parser.add_argument("--rows", type=int, nargs="+", default=[1024, 4096, 8192])
    parser.add_argument("--blocks", type=int, nargs="+", default=[4096, 16384, 32768])
    parser.add_argument("--adversarial-shape", type=int, nargs=2, default=[4096, 32768])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--law-rows", type=int, default=24)
    args = parser.parse_args()

    owners = _prove_lease()
    mx.set_cache_limit(4 << 30)
    mx.set_default_device(mx.gpu)
    if not mx.metal.is_available():
        raise SystemExit("Metal is unavailable")
    report = {
        "schema": SCHEMA,
        "status": "running",
        "revision": _command("git", "rev-parse", "HEAD"),
        "dirty": bool(_command("git", "status", "--porcelain", "--untracked-files=no")),
        "mlx_version": mx.__version__,
        "device": _device_info(),
        "gpu_lock_owners": owners,
        "synthetic_inputs_only": True,
        "model_weights_loaded": False,
        "ratio": RATIO,
        "topk": TOPK,
        "candidate_topk": CANDIDATE_TOPK,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "timing_order": "rotating interleaved arms with alternating direction",
        "gvr_variants": {
            name: {
                "width": w,
                "capacity": c,
                "fractions256": list(gvr_ref.threshold_fractions256(TOPK, c)),
            }
            for name, (w, c) in GVR_VARIANTS.items()
        },
        "host_before": _host(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        for section in args.sections.split(","):
            function = {
                "selector": selector_section,
                "adversarial": adversarial_section,
                "stage1": stage1_section,
            }[section]
            report[section] = function(args)
            args.output.write_text(json.dumps(report, indent=1, default=str) + "\n")
        report["all_exact"] = all(
            _all_exact(report[name])
            for name in ("selector", "adversarial", "stage1")
            if name in report
        )
        report["status"] = "completed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_s"] = time.time() - started
        report["host_after"] = _host()
        report["swapouts_delta"] = (
            report["host_after"]["swapouts"] - report["host_before"]["swapouts"]
        )
        args.output.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(
        json.dumps({"status": report["status"], "all_exact": report.get("all_exact")})
    )
    return 0 if report.get("all_exact") else 1


if __name__ == "__main__":
    raise SystemExit(main())
