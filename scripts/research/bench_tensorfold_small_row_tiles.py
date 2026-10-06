#!/usr/bin/env python3
"""M5 tile-width crossover bench for TensorFold #196 and mlx2's lane kernel.

The production mlx2 lane law is already 32 output columns wide.  This isolated
bench asks the remaining question: would TensorFold's 64-wide cooperative tile
be better for any row widths mlx2 admits?  It compares mlx2 NT=32, TensorFold
NT=32, TensorFold NT=64, and stock MLX without changing installation or policy.

Default mode is metadata-only.  The GPU arm requires the repository's external
GPU ownership wrapper and a local TensorFold checkout containing the pinned MIT
kernel revision.  A component result is not a serving selection or model
qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tensorfold_viability_common import (
    environment_snapshot,
    git_revision,
    pinned_checkout_snapshot,
    require_distribution,
    require_matching_gpu_receipts,
    timed_arms,
)

SOURCE_PR = "https://github.com/ashhart/TensorFold/pull/196"
SOURCE_HEAD = "deb1057662d69b76d27bf479997744b855868e54"
KERNEL_REVISION = "bb4b4a35863af562fc4ccb2586300d8f94b5d6de"
DEFAULT_TENSORFOLD_ROOT = Path.home() / "Desktop" / "TensorFold"
DEFAULT_ROWS = (1, 9, 16, 17, 32, 64, 128)
EXPECTED_MLX_VERSION = "0.32.2.dev20260919+39400a0d4"
KERNEL_RELATIVE_PATH = "src/tensorfold/kernels/qwen/dense/v1/lane_qmm.py"
KERNEL_SHA256 = "39107d55af048de9f37d3a48bdfe0dfad720d66eeac1b08da5f170198d591b0e"
SHAPES = {
    "down_proj": (17408, 5120),
    "out_proj": (6144, 5120),
}


def narrow_strategy(*, streams: int, widest: int, split_k: int) -> bool:
    """TensorFold #196 selection law for weights with more than four slices."""

    if streams < 1 or widest < 1 or split_k < 1:
        raise ValueError("streams, widest and split_k must be positive")
    return streams * widest <= 32 and split_k > 4


def _parse_rows(value: str) -> tuple[int, ...]:
    rows = tuple(int(item) for item in value.split(",") if item)
    if not rows or min(rows) < 1 or max(rows) > 128:
        raise ValueError("--rows must stay within 1..128")
    return rows


def _parse_shapes(value: str) -> tuple[str, ...]:
    names = tuple(item for item in value.split(",") if item)
    if not names:
        raise ValueError("--shapes must not be empty")
    unknown = sorted(set(names) - set(SHAPES))
    if unknown:
        raise ValueError(f"unknown shapes: {unknown}")
    return names


def execution_plan(
    *,
    rows: tuple[int, ...] = DEFAULT_ROWS,
    shapes: tuple[str, ...] = tuple(SHAPES),
    warmups: int = 2,
    rounds: int = 7,
    seed: int = 196,
) -> dict:
    """Describe exact allocations and candidate calls without importing MLX."""

    if warmups < 0 or rounds < 1:
        raise ValueError("warmups must be nonnegative and rounds positive")
    cases = []
    for name in shapes:
        k, n = SHAPES[name]
        packed_weight = n * k // 2
        scale_or_bias = n * (k // 64) * 2
        cases.append(
            {
                "name": name,
                "k": k,
                "n": n,
                "rows": list(rows),
                "input_allocation": [max(rows), k],
                "input_dtype": "bfloat16",
                "quantization": {"bits": 4, "group_size": 64},
                "estimated_bytes": {
                    "plain_weight": packed_weight,
                    "tensorfold_nt32_weight": packed_weight,
                    "tensorfold_nt64_weight": packed_weight,
                    "one_scale_array": scale_or_bias,
                    "one_bias_array": scale_or_bias,
                    "tensorfold_packed_scale_bias": scale_or_bias * 2,
                    "three_weight_layouts": packed_weight * 3,
                },
                "arm_calls": {
                    "mlx2_nt32": "mlx2 lane_matmul(inputs, prepared_mpp_weight)",
                    "tensorfold_nt32": (
                        "TensorFold lane_matmul(inputs, tiled_nt32, packed_scale_bias, "
                        "tiled=True, sk=shape_split_k, nt=32)"
                    ),
                    "tensorfold_nt64": (
                        "TensorFold lane_matmul(inputs, tiled_nt64, packed_scale_bias, "
                        "tiled=True, sk=shape_split_k, nt=64)"
                    ),
                    "stock": (
                        "mx.quantized_matmul(inputs, plain_weight, scales, biases, "
                        "transpose=True, group_size=64, bits=4)"
                    ),
                },
            }
        )
    return {
        "seed": seed,
        "warmups_per_arm": warmups,
        "measured_rounds_per_arm": rounds,
        "counterbalance": "forward arm order on even rounds, reverse on odd rounds",
        "synchronization": "mx.eval each output",
        "case_order": "shape order then row order from command line",
        "cases": cases,
    }


def deferred_gpu_command(args=None) -> list[str]:
    tensorfold_root = DEFAULT_TENSORFOLD_ROOT if args is None else args.tensorfold_root
    rows = ",".join(map(str, DEFAULT_ROWS)) if args is None else args.rows
    shapes = ",".join(SHAPES) if args is None else args.shapes
    warmups = 2 if args is None else args.warmups
    rounds = 7 if args is None else args.rounds
    seed = 196 if args is None else args.seed
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_with_gpu_locks.py"),
        "--session",
        "upstream-viability-benches-20261004",
        "--label",
        "tensorfold-small-row-196",
        "--receipt",
        str(ROOT / "artifacts" / "research" / "tensorfold-small-row-196-lock.json"),
        "--",
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-gpu",
        "--i-hold-gpu-lease",
        "--tensorfold-root",
        str(tensorfold_root),
        "--rows",
        rows,
        "--shapes",
        shapes,
        "--warmups",
        str(warmups),
        "--rounds",
        str(rounds),
        "--seed",
        str(seed),
        "--output",
        str(ROOT / "artifacts" / "research" / "tensorfold-small-row-196.json"),
    ]


def metadata(args=None) -> dict:
    rows = DEFAULT_ROWS if args is None else _parse_rows(args.rows)
    shapes = tuple(SHAPES) if args is None else _parse_shapes(args.shapes)
    tensorfold_root = DEFAULT_TENSORFOLD_ROOT if args is None else args.tensorfold_root
    return {
        "schema": "mlx2.tensorfold-small-row-tiles-bench.v1",
        "status": "research_harness_unexecuted",
        "source": {
            "pr": SOURCE_PR,
            "pr_head": SOURCE_HEAD,
            "kernel_revision": KERNEL_REVISION,
            "kernel_path": KERNEL_RELATIVE_PATH,
            "kernel_sha256": KERNEL_SHA256,
        },
        "environment": environment_snapshot(mlx_version=EXPECTED_MLX_VERSION),
        "deferred_gpu_command": deferred_gpu_command(args),
        "tensorfold_checkout": pinned_checkout_snapshot(
            tensorfold_root,
            expected_revision=KERNEL_REVISION,
            files={KERNEL_RELATIVE_PATH: KERNEL_SHA256},
        ),
        "current_mlx2": {
            "already_implemented": True,
            "detail": (
                "mlx2 runtime/lane/matmul.py fixes NT=32, ROW_BLOCK=16 "
                "for every admitted M5 lane call; #196's small-row tile is not missing"
            ),
            "remaining_question": (
                "whether an isolated 64-wide arm wins at 64..128 rows enough to "
                "justify a future crossover without changing mlx2's numerical law"
            ),
            "production_change": False,
        },
        "arms": {
            "mlx2_nt32": "current untiled mlx2 lane kernel",
            "tensorfold_nt32": "pinned TensorFold tiled 32-column kernel",
            "tensorfold_nt64": "pinned TensorFold cooperative 64-column kernel",
            "stock": "MLX affine q4/group-64 quantized matmul",
        },
        "execution_plan": execution_plan(
            rows=rows,
            shapes=shapes,
            warmups=2 if args is None else args.warmups,
            rounds=7 if args is None else args.rounds,
            seed=196 if args is None else args.seed,
        ),
        "gates": [
            "TensorFold NT32 and NT64 must be array_equal at every row width",
            "every output must be finite",
            "record mlx2-vs-TensorFold numerics instead of assuming a shared law",
            "no production default or prepared weight is changed",
        ],
        "known_caveats": [
            "#196 measured only M5 Max and an independent comment measured M5 Ultra but not its crossover",
            "the independent M5 Ultra DFlash2 end-to-end result was within noise despite a large no-draft gain",
            "a per-call crossover can add a second tiled weight layout; residency cost must be charged before adoption",
        ],
    }


def _tensorfold_module(root: Path, checkout=None):
    if checkout is None:
        checkout = pinned_checkout_snapshot(
            root,
            expected_revision=KERNEL_REVISION,
            files={KERNEL_RELATIVE_PATH: KERNEL_SHA256},
            require_match=True,
        )
    source = root / "src"
    sys.path.insert(0, str(source))
    module = importlib.import_module("tensorfold.kernels.qwen.dense.v1.lane_qmm")
    loaded = Path(module.__file__).resolve()
    if source.resolve() not in loaded.parents:
        raise RuntimeError(
            f"imported TensorFold from {loaded}, expected under {source}"
        )
    escaped = []
    for name, loaded_module in sys.modules.items():
        if name != "tensorfold" and not name.startswith("tensorfold."):
            continue
        loaded_file = getattr(loaded_module, "__file__", None)
        if loaded_file and source.resolve() not in Path(loaded_file).resolve().parents:
            escaped.append({"module": name, "path": str(Path(loaded_file).resolve())})
    if escaped:
        raise RuntimeError(f"TensorFold dependency escaped pinned checkout: {escaped}")
    for attr in ("lane_matmul", "pack_scales", "tile_weight", "split_k"):
        if not hasattr(module, attr):
            raise RuntimeError(f"pinned TensorFold kernel lacks {attr}")
    return module, checkout


def _evaluate(mx, value) -> None:
    mx.eval(value)


def _module(mx, nn, k: int, n: int):
    module = nn.QuantizedLinear(k, n, bias=False, group_size=64, bits=4)
    module.weight = mx.random.randint(0, 2**31 - 1, shape=(n, k // 8)).astype(mx.uint32)
    module.scales = (
        mx.random.uniform(low=0.001, high=0.05, shape=(n, k // 64))
    ).astype(mx.bfloat16)
    module.biases = (mx.random.uniform(low=-0.1, high=0.1, shape=(n, k // 64))).astype(
        mx.bfloat16
    )
    mx.eval(module.parameters())
    return module


def _run_shape(mx, nn, mlx2_lane, upstream, name, k, n, rows_to_run, args):
    module = _module(mx, nn, k, n)
    lw = mlx2_lane.prepare(module, "mpp")
    sk = upstream.split_k(n, k)
    plain_weight = module.weight
    sbt = upstream.pack_scales(module.scales, module.biases)
    # The pinned kernel infers its fixed q4/group-64 layout; only the output
    # tile width is configurable at this revision.
    weight32 = upstream.tile_weight(plain_weight, 32)
    weight64 = upstream.tile_weight(plain_weight, 64)
    x = mx.random.normal((max(rows_to_run), k)).astype(mx.bfloat16)
    mx.eval(lw.scale_bias, sbt, weight32, weight64, x)
    records = []
    for rows in rows_to_run:
        inputs = x[:rows]
        arms = {
            "mlx2_nt32": lambda inputs=inputs: mlx2_lane.lane_matmul(inputs, lw),
            "tensorfold_nt32": lambda inputs=inputs: upstream.lane_matmul(
                inputs, weight32, sbt, tiled=True, sk=sk, nt=32
            ),
            "tensorfold_nt64": lambda inputs=inputs: upstream.lane_matmul(
                inputs, weight64, sbt, tiled=True, sk=sk, nt=64
            ),
            "stock": lambda inputs=inputs: mx.quantized_matmul(
                inputs,
                plain_weight,
                module.scales,
                module.biases,
                transpose=True,
                group_size=64,
                bits=4,
            ),
        }
        outputs = {arm: call() for arm, call in arms.items()}
        mx.eval(*outputs.values())
        checks = {}
        for arm, value in outputs.items():
            delta = mx.abs(
                value.astype(mx.float32) - outputs["tensorfold_nt32"].astype(mx.float32)
            )
            checks[arm] = {
                "finite": bool(mx.all(mx.isfinite(value)).item()),
                "bit_equal_to_tensorfold_nt32": bool(
                    mx.array_equal(value, outputs["tensorfold_nt32"]).item()
                ),
                "max_abs_to_tensorfold_nt32": float(mx.max(delta).item()),
            }
        timings = timed_arms(
            lambda value: _evaluate(mx, value),
            arms,
            warmups=args.warmups,
            rounds=args.rounds,
        )
        base = timings["tensorfold_nt64"]["median_ms"]
        timings["tensorfold_nt32"]["speedup_vs_tensorfold_nt64"] = (
            base / timings["tensorfold_nt32"]["median_ms"]
        )
        records.append(
            {
                "rows": rows,
                "checks": checks,
                "timing": timings,
                "upstream_exact_gate": checks["tensorfold_nt64"][
                    "bit_equal_to_tensorfold_nt32"
                ],
            }
        )
    return {
        "name": name,
        "k": k,
        "n": n,
        "layout_bytes": {
            "plain_weight": int(plain_weight.nbytes),
            "tensorfold_nt32": int(weight32.nbytes),
            "tensorfold_nt64": int(weight64.nbytes),
            "benchmark_three_layout_total": int(
                plain_weight.nbytes + weight32.nbytes + weight64.nbytes
            ),
        },
        "mlx2_split_k": lw.split_k,
        "tensorfold_split_k": sk,
        "tensorfold_196_narrow_strategy_for_one_stream": narrow_strategy(
            streams=1, widest=32, split_k=sk
        ),
        "rows": records,
    }


def _run(args) -> dict:
    if not args.i_hold_gpu_lease:
        raise RuntimeError("--run-gpu requires --i-hold-gpu-lease")
    owner = require_matching_gpu_receipts()
    runtime = require_distribution("mlx", EXPECTED_MLX_VERSION)
    rows_to_run = _parse_rows(args.rows)
    names = _parse_shapes(args.shapes)
    execution_plan(
        rows=rows_to_run,
        shapes=names,
        warmups=args.warmups,
        rounds=args.rounds,
        seed=args.seed,
    )
    upstream_checkout = pinned_checkout_snapshot(
        args.tensorfold_root,
        expected_revision=KERNEL_REVISION,
        files={KERNEL_RELATIVE_PATH: KERNEL_SHA256},
        require_match=True,
    )

    import mlx.core as mx
    from mlx import nn

    from mlx2.runtime.lane import matmul as mlx2_lane

    mx.set_default_device(mx.gpu)
    info = mx.device_info()
    if not mx.metal.is_available() or "M5" not in str(info.get("device_name", "")):
        raise RuntimeError("TensorFold #196 bench requires an M5 Metal GPU")
    upstream, upstream_checkout = _tensorfold_module(
        args.tensorfold_root, checkout=upstream_checkout
    )
    mx.random.seed(args.seed)
    cases = [
        _run_shape(mx, nn, mlx2_lane, upstream, name, *SHAPES[name], rows_to_run, args)
        for name in names
    ]
    exact = all(row["upstream_exact_gate"] for case in cases for row in case["rows"])
    finite = all(
        check["finite"]
        for case in cases
        for row in case["rows"]
        for check in row["checks"].values()
    )
    return {
        **metadata(args),
        "status": "component_benchmark_complete_unqualified",
        "source_head": git_revision(ROOT),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tensorfold_checkout": upstream_checkout,
        "gpu_owner": owner,
        "mlx_version": mx.__version__,
        "runtime_pin": runtime,
        "device": info,
        "parameters": {"warmups": args.warmups, "rounds": args.rounds},
        "upstream_exact_gate": exact,
        "numerical_gate": exact and finite,
        "cases": cases,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--i-hold-gpu-lease", action="store_true")
    parser.add_argument("--tensorfold-root", type=Path, default=DEFAULT_TENSORFOLD_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rows", default=",".join(map(str, DEFAULT_ROWS)))
    parser.add_argument("--shapes", default=",".join(SHAPES))
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=196)
    args = parser.parse_args(argv)
    result = _run(args) if args.run_gpu else metadata(args)
    rendered = json.dumps(result, indent=2, default=str) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if not args.run_gpu or result["numerical_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
