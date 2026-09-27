#!/usr/bin/env python3
"""Research-only Qwen4 fused-MoE output-row tiling benchmark.

This harness loads the current production scalar and tile-4 Metal sources at
run time.  Tile 8 and tile 16 are strict, counted transformations of the
current tile-4 source.  It does not alter admission, routing, or production
code.  Widths M=2/4/8 are explicitly research-only and unadmitted.

The default action is metadata-only and does not import MLX.  GPU execution
must be wrapped by the repository's CPG lease plus owned_exec.py lock guard.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATH = ROOT / "src/mlx2/runtime/models/qwen4_fused_moe.py"
LOCK_RECEIPT = Path("/Users/Shared/mlxuag/gpu.lock/owner.json")
TILES = (1, 4, 8, 16)
TOKEN_WIDTHS = (1, 2, 3, 4, 8)
HIDDEN_SIZE = 2560
EXPERT_HIDDEN_SIZE = 640
TOP_K = 10
NUM_EXPERTS = 512


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _requires_exported_width1_check(token_widths: tuple[int, ...]) -> bool:
    return 1 in token_widths


def _source_literals() -> dict[str, str]:
    """Read kernel strings without importing MLX or the production module."""
    tree = ast.parse(SOURCE_PATH.read_text())
    wanted = {"_DOWN_REDUCE_SOURCE", "_DOWN_REDUCE_TILE4_SOURCE"}
    found: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        for name in names:
            if name in wanted:
                value = ast.literal_eval(node.value)
                if not isinstance(value, str):
                    raise TypeError(f"{name} is no longer a literal string")
                found[name] = value
    missing = wanted - found.keys()
    if missing:
        raise RuntimeError(f"missing production kernel literals: {sorted(missing)}")
    return found


def _replace_exact(source: str, old: str, new: str, expected: int) -> str:
    count = source.count(old)
    if count != expected:
        raise RuntimeError(
            f"source-generation guard failed for {old!r}: expected {expected}, got {count}"
        )
    return source.replace(old, new)


def _widen_tile4_source(source: str, tile: int) -> str:
    """Widen only output-row arrays, loops, and indices in current tile-4."""
    if tile not in (8, 16):
        raise ValueError(f"only tile 8/16 are generated, got {tile}")
    result = source
    replacements = (
        ("thread_position_in_grid.y * 4;", f"thread_position_in_grid.y * {tile};", 1),
        ("partials[TOPK * 4];", f"partials[TOPK * {tile}];", 1),
        ("float values[8];", f"float values[{tile * 2}];", 1),
        ("i < 8", f"i < {tile * 2}", 2),
        ("local_row < 4", f"local_row < {tile}", 2),
        ("local_slot * 4 + local_row", f"local_slot * {tile} + local_row", 2),
        ("slot * 4 + local_row", f"slot * {tile} + local_row", 1),
        ("lane < 4", f"lane < {tile}", 1),
        ("slot * 4 + lane", f"slot * {tile} + lane", 1),
    )
    for old, new, expected in replacements:
        result = _replace_exact(result, old, new, expected)
    if result == source:
        raise RuntimeError("widening produced unchanged source")
    return result


def _sources() -> dict[int, str]:
    literals = _source_literals()
    tile4 = literals["_DOWN_REDUCE_TILE4_SOURCE"]
    return {
        1: literals["_DOWN_REDUCE_SOURCE"],
        4: tile4,
        8: _widen_tile4_source(tile4, 8),
        16: _widen_tile4_source(tile4, 16),
    }


def _static_description() -> dict:
    sources = _sources()
    return {
        "schema": "mlx2.m5-next-math.moe-output-tile.v1",
        "status": "isolated_research_harness_unqualified",
        "production_source": str(SOURCE_PATH),
        "production_source_sha256": _sha256_path(SOURCE_PATH),
        "geometry": {
            "experts": NUM_EXPERTS,
            "hidden_size": HIDDEN_SIZE,
            "expert_hidden_size": EXPERT_HIDDEN_SIZE,
            "top_k": TOP_K,
            "quantization": "affine-q4-group64",
        },
        "tiles": {
            str(tile): {
                "source_sha256": _sha256_bytes(source.encode()),
                "source_origin": (
                    "exact current scalar literal"
                    if tile == 1
                    else "exact current tile4 literal"
                    if tile == 4
                    else "strict counted transformation of current tile4 literal"
                ),
            }
            for tile, source in sources.items()
        },
        "token_widths": {
            "M=1": "currently qualified and selected tile4 production width",
            "M=3": "current explicit candidate width; auto maps to tile4 when requested, not production-qualified",
            "M=2/4/8": "research-only private launches; unselected and unadmitted",
        },
        "arithmetic_invariants": [
            "per-word accum_x nibble addition order unchanged",
            "per-row accum_q nibble addition order unchanged",
            "simd_sum reduction unchanged",
            "BF16 expert and weighted boundaries unchanged",
            "slot reduction order 0..9 unchanged",
        ],
    }


def _require_ownership() -> dict:
    owner = json.loads(LOCK_RECEIPT.read_text())
    parent = os.getppid()
    grandparent = int(
        subprocess.check_output(["ps", "-o", "ppid=", "-p", str(parent)], text=True).strip()
    )
    parent_command = subprocess.check_output(
        ["ps", "-o", "command=", "-p", str(parent)], text=True
    )
    if owner.get("pid") != grandparent or not owner.get("cpg_generation"):
        raise RuntimeError("refusing GPU work without matching CPG lease ancestry")
    if "owned_exec.py" not in parent_command:
        raise RuntimeError("refusing GPU work without owned_exec.py advisory-lock guard")
    return owner


def _compare(mx, reference, candidate) -> dict:
    import numpy as np

    mx.eval(reference, candidate)
    left = np.asarray(reference.astype(mx.float32))
    right = np.asarray(candidate.astype(mx.float32))
    delta = right.astype(np.float64) - left.astype(np.float64)
    left_bits = left.view(np.uint32)
    right_bits = right.view(np.uint32)
    ul = (left_bits >> 16).astype(np.int64)
    ur = (right_bits >> 16).astype(np.int64)
    ol = np.where(ul & 0x8000, 0x8000 - (ul & 0x7FFF), 0x8000 + ul)
    or_ = np.where(ur & 0x8000, 0x8000 - (ur & 0x7FFF), 0x8000 + ur)
    norm = float(np.linalg.norm(left.astype(np.float64).reshape(-1)))
    error = float(np.linalg.norm(delta.reshape(-1)))
    return {
        "count": int(left.size),
        "bit_mismatches": int(np.count_nonzero(left_bits != right_bits)),
        "max_abs": float(np.max(np.abs(delta))),
        "rms_error": float(np.sqrt(np.mean(delta * delta))),
        "relative_l2": error / norm if norm else (0.0 if error == 0.0 else None),
        "max_bf16_ulps": int(np.max(np.abs(ol - or_))),
        "reference_nonfinite": int(np.count_nonzero(~np.isfinite(left))),
        "candidate_nonfinite": int(np.count_nonzero(~np.isfinite(right))),
    }


def _time_arms(mx, arms: dict[str, object], *, rounds: int, inner: int, seed: int) -> dict:
    import numpy as np

    for function in arms.values():
        for _ in range(3):
            mx.eval(function())
    mx.synchronize()
    samples = {name: [] for name in arms}
    orders: list[list[str]] = []
    rng = random.Random(seed)
    for _ in range(rounds):
        order = list(arms)
        rng.shuffle(order)
        orders.append(order)
        for name in order:
            started = time.perf_counter_ns()
            for _ in range(inner):
                mx.eval(arms[name]())
            samples[name].append((time.perf_counter_ns() - started) / inner / 1_000.0)
    bootstrap = np.random.default_rng(seed).integers(0, rounds, size=(4000, rounds))
    result = {
        "method": "interleaved synchronous wall time per fresh invocation; Python dispatch included",
        "rounds": rounds,
        "inner": inner,
        "orders": orders,
        "arms": {},
    }
    baseline = np.asarray(samples["tile4"])
    for name, values in samples.items():
        array = np.asarray(values)
        ratios = baseline / array
        result["arms"][name] = {
            "samples_us": values,
            "median_us": float(np.median(array)),
            "p10_us": float(np.percentile(array, 10)),
            "p90_us": float(np.percentile(array, 90)),
            "paired_median_speedup_vs_tile4": float(np.median(ratios)),
            "paired_bootstrap_95pct": np.percentile(
                np.median(ratios[bootstrap], axis=1), (2.5, 97.5)
            ).tolist(),
        }
    return result


def _routing_inputs(mx, tokens: int, *, stress: bool):
    indices = [
        [int((token * 67 + slot * 47 + 11) % NUM_EXPERTS) for slot in range(TOP_K)]
        for token in range(tokens)
    ]
    if stress:
        score_rows = [[0.1] * TOP_K for _ in range(tokens)]
        base = mx.where(
            mx.arange(EXPERT_HIDDEN_SIZE) % 2 == 0,
            mx.array(8.0),
            mx.array(-8.0),
        )
        hidden = mx.stack(
            [
                mx.stack(
                    [base * (-1.0 if (token + slot) % 2 else 1.0) for slot in range(TOP_K)]
                )
                for token in range(tokens)
            ]
        ).astype(mx.bfloat16)
        label = "alternating-sign magnitude-8 hidden, equal BF16 scores"
    else:
        raw_scores = [
            [float(1 + ((slot * 7 + token * 3) % 13)) for slot in range(TOP_K)]
            for token in range(tokens)
        ]
        score_rows = [[value / sum(row) for value in row] for row in raw_scores]
        hidden = (mx.random.normal((tokens, TOP_K, EXPERT_HIDDEN_SIZE)) * 0.75).astype(
            mx.bfloat16
        )
        label = "normal hidden sd=0.75, deterministic normalized positive BF16 scores"
    return (
        hidden,
        mx.array(indices, dtype=mx.int32),
        mx.array(score_rows, dtype=mx.bfloat16),
        label,
    )


def _run(args) -> dict:
    owner = _require_ownership()
    source_file_hash = _sha256_path(SOURCE_PATH)
    static_sources = _sources()

    import mlx.core as mx
    import numpy as np

    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.runtime.models import qwen4_fused_moe as production

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("Metal GPU unavailable; refusing to label this a GPU result")
    if production.NUM_EXPERTS != NUM_EXPERTS:
        raise RuntimeError(f"expert geometry drifted to {production.NUM_EXPERTS}")
    if production._DOWN_REDUCE_SOURCE != static_sources[1]:
        raise RuntimeError("runtime scalar source differs from AST-read source")
    if production._DOWN_REDUCE_TILE4_SOURCE != static_sources[4]:
        raise RuntimeError("runtime tile4 source differs from AST-read source")

    kernels = {
        tile: mx.fast.metal_kernel(
            name=f"research_qwen4_q4_down_tile{tile}_v1",
            input_names=[
                "hidden",
                "down_weight",
                "down_scales",
                "down_biases",
                "indices",
                "scores",
            ],
            output_names=["out"],
            source=static_sources[tile],
            ensure_row_contiguous=False,
        )
        for tile in TILES
    }
    counters = {f"tile{tile}": 0 for tile in TILES}
    counters.update({f"compiled_tile{tile}": 0 for tile in TILES})
    counters.update({"exported_scalar": 0, "exported_tile4": 0})

    def launch(tile, hidden, weight, scales, biases, indices, scores):
        counters[f"tile{tile}"] += 1
        tokens = hidden.shape[0]
        if tile == 1:
            grid = (32, HIDDEN_SIZE, tokens)
            threadgroup = (32, 1, 1)
        else:
            grid = (160, HIDDEN_SIZE // tile, tokens)
            threadgroup = (160, 1, 1)
        return kernels[tile](
            inputs=[hidden, weight, scales, biases, indices, scores],
            template=[("T", hidden.dtype), ("W", scales.dtype)],
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=[(tokens, HIDDEN_SIZE)],
            output_dtypes=[hidden.dtype],
        )[0]

    def compiled_for(tile):
        def body(hidden, weight, scales, biases, indices, scores):
            return launch(tile, hidden, weight, scales, biases, indices, scores)

        compiled_body = mx.compile(body)

        def called(*inputs):
            counters[f"compiled_tile{tile}"] += 1
            return compiled_body(*inputs)

        return called

    compiled = {tile: compiled_for(tile) for tile in TILES}

    mx.random.seed(args.seed)
    packed_shape = (NUM_EXPERTS, HIDDEN_SIZE, EXPERT_HIDDEN_SIZE // 8)
    table_shape = (NUM_EXPERTS, HIDDEN_SIZE, EXPERT_HIDDEN_SIZE // 64)
    weight = mx.random.randint(-(2**31), 2**31 - 1, shape=packed_shape).astype(mx.uint32)
    scales = mx.random.uniform(0.002, 0.02, shape=table_shape).astype(mx.bfloat16)
    biases = mx.random.uniform(-0.15, 0.0, shape=table_shape).astype(mx.bfloat16)
    mx.eval(weight, scales, biases)

    result = {
        **_static_description(),
        "device": mx.metal.device_info(),
        "mlx_version": importlib.metadata.version("mlx"),
        "repo_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "ownership": owner,
        "seed": args.seed,
        "parameters": {
            "rounds": args.rounds,
            "inner": args.inner,
            "packed_weight_bytes": int(np.prod(packed_shape) * 4),
            "scale_bytes": int(np.prod(table_shape) * 2),
            "bias_bytes": int(np.prod(table_shape) * 2),
            "packed_random_bits": "uniform signed-int32 bit patterns cast to uint32",
            "selected_experts": "token/slot deterministic spread over all 512 ids",
        },
        "cases": [],
        "mechanism_counters": counters,
        "completed": False,
    }

    def save():
        result["mechanism_counters"] = dict(counters)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    if args.profile_tile is not None:
        tokens = 3
        hidden, indices, scores, distribution = _routing_inputs(mx, tokens, stress=False)
        inputs = (hidden, weight, scales, biases, indices, scores)
        reference = launch(4, *inputs)
        candidate = launch(8, *inputs)
        mx.eval(reference, candidate)
        random_parity = _compare(mx, reference, candidate)

        stress_hidden, stress_indices, stress_scores, stress_distribution = (
            _routing_inputs(mx, tokens, stress=True)
        )
        stress_inputs = (
            stress_hidden,
            weight,
            scales,
            biases,
            stress_indices,
            stress_scores,
        )
        stress_reference = launch(4, *stress_inputs)
        stress_candidate = launch(8, *stress_inputs)
        mx.eval(stress_reference, stress_candidate)
        stress_parity = _compare(mx, stress_reference, stress_candidate)
        for label, parity in (("random", random_parity), ("stress", stress_parity)):
            if parity["bit_mismatches"] or parity["reference_nonfinite"] or parity["candidate_nonfinite"]:
                raise RuntimeError(f"{label} parity/finiteness gate failed: {parity}")

        # Compile and warm both arms equally, then exercise only the selected
        # uniquely named kernel during the counter-capture window.
        for tile in (4, 8):
            for _ in range(10):
                mx.eval(launch(tile, *inputs))
        mx.synchronize()
        started = time.perf_counter_ns()
        output = None
        for _ in range(args.profile_iterations):
            output = launch(args.profile_tile, *inputs)
            mx.eval(output)
        mx.synchronize()
        elapsed_ns = time.perf_counter_ns() - started
        output_bytes = np.asarray(output.astype(mx.float32)).tobytes()
        result["profile"] = {
            "status": "counter-capture workload, not primary timing evidence",
            "tokens": tokens,
            "selected_tile": args.profile_tile,
            "kernel_name": f"research_qwen4_q4_down_tile{args.profile_tile}_v1",
            "iterations": args.profile_iterations,
            "elapsed_ns": elapsed_ns,
            "distribution": distribution,
            "stress_distribution": stress_distribution,
            "random_tile4_vs_tile8": random_parity,
            "stress_tile4_vs_tile8": stress_parity,
            "output_float32_sha256": _sha256_bytes(output_bytes),
        }
        expected_counts = {
            "tile4": 12 + (args.profile_iterations if args.profile_tile == 4 else 0),
            "tile8": 12 + (args.profile_iterations if args.profile_tile == 8 else 0),
        }
        if any(counters[name] != count for name, count in expected_counts.items()):
            raise RuntimeError(f"profile mechanism engagement failure: {counters}")
        if _sha256_path(SOURCE_PATH) != source_file_hash:
            raise RuntimeError("production source changed during the profile workload")
        result["peak_memory_bytes"] = int(mx.get_peak_memory())
        result["completed"] = True
        save()
        return result

    for tokens in args.token_widths:
        if tokens not in TOKEN_WIDTHS:
            raise ValueError(f"unsupported research token width {tokens}")
        hidden, indices, scores, distribution = _routing_inputs(mx, tokens, stress=False)
        mx.eval(hidden, indices, scores)
        inputs = (hidden, weight, scales, biases, indices, scores)
        outputs = {tile: launch(tile, *inputs) for tile in TILES}
        compiled_outputs = {tile: compiled[tile](*inputs) for tile in TILES}
        mx.eval(*outputs.values(), *compiled_outputs.values())
        case = {
            "tokens": tokens,
            "route_status": (
                "M=1 production-qualified/selected tile4 geometry"
                if tokens == 1
                else "current explicit candidate width; auto maps to tile4 when requested, not production-qualified"
                if tokens == 3
                else "research-only private launch; unselected and unadmitted"
            ),
            "distribution": distribution,
            "selected_expert_ids": np.asarray(indices).tolist(),
            "parity_vs_scalar": {
                f"tile{tile}": _compare(mx, outputs[1], outputs[tile]) for tile in TILES
            },
            "compiled_vs_direct": {
                f"tile{tile}": _compare(mx, outputs[tile], compiled_outputs[tile])
                for tile in TILES
            },
        }
        if tokens == 1:
            counters["exported_scalar"] += 1
            exported_scalar = production.qwen4_fused_down(
                hidden, indices, scores, weight, scales, biases, variant="scalar"
            )
            counters["exported_tile4"] += 1
            exported_tile4 = production.qwen4_fused_down(
                hidden, indices, scores, weight, scales, biases, variant="tile4"
            )
            case["current_exported_parity"] = {
                "private_tile1_vs_exported_scalar": _compare(
                    mx, exported_scalar, outputs[1]
                ),
                "private_tile4_vs_exported_tile4": _compare(
                    mx, exported_tile4, outputs[4]
                ),
                "exported_scalar_vs_exported_tile4": _compare(
                    mx, exported_scalar, exported_tile4
                ),
            }
        direct_arms = {
            f"tile{tile}": (
                lambda tile=tile: launch(tile, hidden, weight, scales, biases, indices, scores)
            )
            for tile in TILES
        }
        case["direct_timing"] = _time_arms(
            mx, direct_arms, rounds=args.rounds, inner=args.inner, seed=args.seed + tokens
        )
        compiled_arms = {
            f"tile{tile}": (
                lambda tile=tile: compiled[tile](
                    hidden, weight, scales, biases, indices, scores
                )
            )
            for tile in TILES
        }
        try:
            case["compiled_timing"] = _time_arms(
                mx,
                compiled_arms,
                rounds=args.rounds,
                inner=args.inner,
                seed=args.seed + 100 + tokens,
            )
        except Exception as error:
            case["compiled_timing"] = {
                "available": False,
                "error": f"{type(error).__name__}: {error}",
            }

        stress_hidden, stress_indices, stress_scores, stress_label = _routing_inputs(
            mx, tokens, stress=True
        )
        stress_inputs = (
            stress_hidden,
            weight,
            scales,
            biases,
            stress_indices,
            stress_scores,
        )
        stress_outputs = {tile: launch(tile, *stress_inputs) for tile in TILES}
        mx.eval(*stress_outputs.values())
        case["stress"] = {
            "distribution": stress_label,
            "parity_vs_scalar": {
                f"tile{tile}": _compare(mx, stress_outputs[1], stress_outputs[tile])
                for tile in TILES
            },
        }
        result["cases"].append(case)
        save()
        medians = {
            name: arm["median_us"]
            for name, arm in case["direct_timing"]["arms"].items()
        }
        print(json.dumps({"tokens": tokens, "direct_medians_us": medians}), flush=True)
        del hidden, indices, scores, outputs, compiled_outputs
        del stress_hidden, stress_indices, stress_scores, stress_outputs
        mx.clear_cache()

    required = [f"tile{tile}" for tile in TILES]
    if any(counters[name] <= 0 for name in required):
        raise RuntimeError(f"mechanism engagement failure: {counters}")
    # The exported production kernels are a width-1 reference check.  Narrow
    # follow-up runs may intentionally select only wider token batches.
    if _requires_exported_width1_check(args.token_widths) and (
        counters["exported_scalar"] <= 0 or counters["exported_tile4"] <= 0
    ):
        raise RuntimeError(f"current exported baseline was not exercised: {counters}")
    if _sha256_path(SOURCE_PATH) != source_file_hash:
        raise RuntimeError("production source changed during the benchmark")
    result["peak_memory_bytes"] = int(mx.get_peak_memory())
    result["completed"] = True
    save()
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--inner", type=int, default=3)
    parser.add_argument("--profile-tile", type=int, choices=(4, 8))
    parser.add_argument("--profile-iterations", type=int, default=20_000)
    parser.add_argument("--token-widths", type=int, nargs="+", default=list(TOKEN_WIDTHS))
    return parser


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.run_gpu:
        print(json.dumps(_static_description(), indent=2, sort_keys=True))
        return 0
    if args.out is None:
        parser.error("--run-gpu requires --out")
    if args.rounds < 3 or args.inner < 1:
        parser.error("--rounds must be >=3 and --inner must be >=1")
    if args.profile_iterations < 1:
        parser.error("--profile-iterations must be >=1")
    report = _run(args)
    print(json.dumps({"output": str(args.out), "completed": report["completed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
