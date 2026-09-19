#!/usr/bin/env python3
"""GPU correctness and latency probe for Qwen4 compact GDN rollback.

This is an isolated kernel qualification. It requires the external GPU lease
receipts and never changes a serving route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlx2.runtime.models.qwen4_fused_gdn import (  # noqa: E402
    CONV_DIM,
    CONV_KERNEL,
    KEY_HEAD_DIM,
    NUM_VALUE_HEADS,
    VALUE_DIM,
    VALUE_HEAD_DIM,
)
from mlx2.runtime.models.qwen4_fused_gdn_verify import (  # noqa: E402
    probe_qwen4_fused_gdn_replay_verify,
    probe_qwen4_fused_gdn_verify,
    qwen4_fused_gdn_reconstruct,
    qwen4_fused_gdn_replay_verify,
    qwen4_fused_gdn_verify,
)


def _require_lock() -> list[dict]:
    paths = (
        Path("/tmp/gpu.lock/owner.json"),
        Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"GPU benchmark requires both lock receipts; missing {missing}")
    return [json.loads(path.read_text()) for path in paths]


def _source_hash() -> str:
    digest = hashlib.sha256()
    for relative in (
        "src/mlx2/runtime/models/qwen4_fused_gdn.py",
        "src/mlx2/runtime/models/qwen4_fused_gdn_verify.py",
    ):
        path = ROOT / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _random_inputs(width: int, seed: int) -> dict:
    mx.random.seed(seed)
    bf16 = mx.bfloat16
    values = {
        "qkv": mx.random.uniform(-1.5, 1.5, (1, width, CONV_DIM)).astype(bf16),
        "z": mx.random.uniform(-1.5, 1.5, (1, width, VALUE_DIM)).astype(bf16),
        "b": mx.random.uniform(-2.0, 2.0, (1, width, NUM_VALUE_HEADS)).astype(bf16),
        "a": mx.random.uniform(-2.0, 2.0, (1, width, NUM_VALUE_HEADS)).astype(bf16),
        "conv_state": mx.random.uniform(
            -1.5, 1.5, (1, CONV_KERNEL - 1, CONV_DIM)
        ).astype(bf16),
        "conv_weight": mx.random.uniform(
            -0.25, 0.25, (CONV_DIM, CONV_KERNEL, 1)
        ).astype(bf16),
        "A_log": mx.random.uniform(-2.0, 0.0, (NUM_VALUE_HEADS,)).astype(mx.float32),
        "dt_bias": mx.random.uniform(-2.0, 2.0, (NUM_VALUE_HEADS,)).astype(bf16),
        "recurrent_state": mx.random.uniform(
            -0.25,
            0.25,
            (1, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM),
        ).astype(mx.float32),
        "norm_weight": mx.random.uniform(0.5, 1.5, (VALUE_HEAD_DIM,)).astype(bf16),
    }
    mx.eval(*values.values())
    return values


def _dispatch(function, values: dict, threadgroup_y: int):
    return function(
        values["qkv"],
        values["z"],
        values["b"],
        values["a"],
        values["conv_state"],
        values["conv_weight"],
        values["A_log"],
        values["dt_bias"],
        values["recurrent_state"],
        values["norm_weight"],
        1e-6,
        threadgroup_y=threadgroup_y,
    )


def _comparison(left, right) -> dict:
    delta = mx.abs(left.astype(mx.float32) - right.astype(mx.float32))
    maximum = mx.max(delta)
    exact = mx.all(left == right)
    finite = mx.all(mx.isfinite(left)) & mx.all(mx.isfinite(right))
    mx.eval(maximum, exact, finite)
    return {
        "exact": bool(exact.item()),
        "finite": bool(finite.item()),
        "max_abs": float(maximum.item()),
    }


def _measure(function, warmups: int, rounds: int) -> dict:
    for _ in range(warmups):
        outputs = function()
        mx.eval(*outputs)
    samples = []
    for _ in range(rounds):
        started = time.perf_counter_ns()
        outputs = function()
        mx.eval(*outputs)
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return {
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "samples_ms": samples,
    }


def _peak_delta(function) -> int:
    mx.clear_cache()
    mx.reset_peak_memory()
    active = mx.get_active_memory()
    outputs = function()
    mx.eval(*outputs)
    return max(0, int(mx.get_peak_memory() - active))


def _case(width: int, seed: int, warmups: int, rounds: int) -> dict:
    values = _random_inputs(width, seed)
    snapshot_ty = probe_qwen4_fused_gdn_verify(mx.bfloat16, width)
    replay_ty = probe_qwen4_fused_gdn_replay_verify(mx.bfloat16, width)
    if snapshot_ty is None or replay_ty is None:
        raise RuntimeError(
            f"Metal probe declined width {width}: snapshot={snapshot_ty}, replay={replay_ty}"
        )

    snapshot = _dispatch(qwen4_fused_gdn_verify, values, snapshot_ty)
    replay = _dispatch(qwen4_fused_gdn_replay_verify, values, replay_ty)
    mx.eval(*snapshot, *replay)
    parity = {
        "output": _comparison(snapshot[0], replay[0]),
        "final_conv_state": _comparison(snapshot[1], replay[1]),
        "final_recurrent_state": _comparison(snapshot[2], replay[2]),
        "partial": [],
    }
    for accepted in range(1, width):
        restored_state = qwen4_fused_gdn_reconstruct(
            values["recurrent_state"],
            replay[3],
            replay[4],
            replay[5],
            accepted,
            threadgroup_y=replay_ty,
        )
        restored_conv = mx.contiguous(
            mx.concatenate([values["conv_state"], values["qkv"]], axis=1)[
                :, accepted : accepted + CONV_KERNEL - 1
            ]
        )
        mx.eval(restored_state, restored_conv)
        parity["partial"].append(
            {
                "accepted": accepted,
                "recurrent_state": _comparison(
                    snapshot[3][:, accepted - 1], restored_state
                ),
                "conv_state": _comparison(snapshot[4][:, accepted - 1], restored_conv),
            }
        )

    def snapshot_verify():
        return _dispatch(qwen4_fused_gdn_verify, values, snapshot_ty)

    def replay_verify():
        return _dispatch(qwen4_fused_gdn_replay_verify, values, replay_ty)

    timing = {
        "snapshot_verify": _measure(snapshot_verify, warmups, rounds),
        "replay_verify": _measure(replay_verify, warmups, rounds),
        "partial_rollback": {},
    }
    for accepted in sorted({1, max(1, (width - 1) // 2), width - 1}):
        def replay_with_rollback(accepted=accepted):
            outputs = replay_verify()
            restored = qwen4_fused_gdn_reconstruct(
                values["recurrent_state"],
                outputs[3],
                outputs[4],
                outputs[5],
                accepted,
                threadgroup_y=replay_ty,
            )
            return (*outputs, restored)

        timing["partial_rollback"][str(accepted)] = _measure(
            replay_with_rollback, warmups, rounds
        )

    snapshot_bytes = sum(int(array.nbytes) for array in snapshot[3:])
    replay_bytes = sum(int(array.nbytes) for array in replay[3:])
    timing["replay_verify"]["speedup_vs_snapshot"] = (
        timing["snapshot_verify"]["median_ms"]
        / timing["replay_verify"]["median_ms"]
    )
    return {
        "width": width,
        "threadgroup_y": {"snapshot": snapshot_ty, "replay": replay_ty},
        "parity": parity,
        "rollback_output_bytes": {
            "snapshot": snapshot_bytes,
            "replay": replay_bytes,
            "reduction_ratio": snapshot_bytes / replay_bytes,
        },
        "timing": timing,
        "peak_delta_bytes": {
            "snapshot_verify": _peak_delta(snapshot_verify),
            "replay_verify": _peak_delta(replay_verify),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", default="3,4,8")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1709)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    locks = _require_lock()
    mx.set_default_device(mx.gpu)
    results = [
        _case(width, args.seed + width, args.warmups, args.rounds)
        for width in map(int, args.widths.split(","))
    ]
    exact = all(
        case["parity"][name]["exact"]
        for case in results
        for name in ("output", "final_conv_state", "final_recurrent_state")
    ) and all(
        item[name]["exact"]
        for case in results
        for item in case["parity"]["partial"]
        for name in ("recurrent_state", "conv_state")
    )
    finite = all(
        case["parity"][name]["finite"]
        for case in results
        for name in ("output", "final_conv_state", "final_recurrent_state")
    ) and all(
        item[name]["finite"]
        for case in results
        for item in case["parity"]["partial"]
        for name in ("recurrent_state", "conv_state")
    )
    report = {
        "schema": "mlx2.qwen4-gdn-replay-gpu-microbench.v1",
        "status": "isolated_kernel_qualification",
        "route_selected": False,
        "source_hash": _source_hash(),
        "device": mx.device_info(),
        "locks": locks,
        "parameters": {
            "widths": list(map(int, args.widths.split(","))),
            "warmups": args.warmups,
            "rounds": args.rounds,
            "seed": args.seed,
        },
        "results": results,
        "correctness": {"exact": exact, "finite": finite},
        "passed": exact and finite,
        "limitations": [
            "Synthetic production-geometry kernel inputs, not model activations.",
            "Peak deltas are allocator observations for one layer, not process RSS.",
            "This probe does not select a serving route.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "passed": report["passed"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
