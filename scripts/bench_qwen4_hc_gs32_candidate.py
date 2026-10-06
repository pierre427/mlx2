#!/usr/bin/env python3
"""Deferred Metal probe for an exact group-size-32 mlx2 HC route.

The production route remains untouched.  In this process only, the existing
mlx2 two-launch, composed-bit-exact HC kernel source is parameterized by one
homogeneous affine group size.  This combines the useful pointer arithmetic
from oMLX #4245/#4248 with mlx2's stricter composed-MLX operation order.

Run the artifact census first.  A synthetic pass can establish arithmetic and
micro-latency, but it cannot establish value for an artifact we do not serve.

  PYTHONPATH=src .venv/bin/python scripts/bench_qwen4_hc_gs32_candidate.py \
      --i-own-the-gpu --out /tmp/qwen4-hc-gs32.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

LOCK_RECEIPTS = (
    Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
    Path("/tmp/gpu.lock/owner.json"),
)


def require_matching_gpu_receipts(paths=LOCK_RECEIPTS) -> dict:
    """Refuse Metal unless both repository lock receipts name one lease."""

    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise RuntimeError(
            "GPU benchmark requires matching ownership receipts from "
            f"scripts/run_with_gpu_locks.py; missing {missing}"
        )
    receipts = [json.loads(Path(path).read_text()) for path in paths]
    if receipts[0] != receipts[1] or not receipts[0].get("lease_id"):
        raise RuntimeError("GPU ownership receipts do not name one matching lease")
    return receipts[0]


def candidate_sources(header: str, norm_down: str, up_mix: str) -> tuple[str, str, str]:
    """Parameterize the group pointers; exported for CPU/static regression tests."""
    header_old = (
        "template <typename T, int BITS, typename P>\ninline float hcd_wide_row("
    )
    header_new = "template <typename T, int BITS, int GS, typename P>\ninline float hcd_wide_row("
    if header.count(header_old) != 1:
        raise ValueError("HC wide-row template source changed")
    header = header.replace(header_old, header_new)
    header = header.replace("g * 64 + sc * 8", "g * GS + sc * 8")
    header = header.replace("sc < 8", "sc < GS / 8")

    def projection_source(source: str) -> str:
        source = source.replace("hcd_wide_row<T, DB>(", "hcd_wide_row<T, DB, GS>(")
        source = source.replace("hcd_wide_row<T, UB>(", "hcd_wide_row<T, UB, GS>(")
        source = source.replace(" / 64", " / GS")
        source = source.replace("64 / ", "GS / ")
        if " / 64" in source or "64 / " in source:
            raise ValueError("unparameterized HC group pointer remains")
        return source

    return header, projection_source(norm_down), projection_source(up_mix)


def install_candidate(hcd, group_size: int = 32) -> None:
    """Install a process-local candidate; never called by production code."""
    if group_size != 32:
        raise ValueError("this intake probe is intentionally scoped to group size 32")
    hcd.HEADER, hcd.NORM_DOWN_SOURCE, hcd.UP_MIX_SOURCE = candidate_sources(
        hcd.HEADER, hcd.NORM_DOWN_SOURCE, hcd.UP_MIX_SOURCE
    )
    hcd.GROUP_SIZE = group_size
    original = hcd._build_plan

    def build_plan(module, law=0):
        plan = original(module, law)
        plan["a_template"] = [*plan["a_template"], ("GS", group_size)]
        plan["b_template"] = [*plan["b_template"], ("GS", group_size)]
        return plan

    hcd._build_plan = build_plan
    hcd._KERNELS.clear()
    hcd._COMPILED.clear()
    hcd.reset_for_tests()


def _bits_equal(mx, left, right) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and bool(mx.array_equal(left.view(mx.uint16), right.view(mx.uint16)).item())
    )


def run(*, cases: int, reps: int, iterations: int, gpu_owner: dict) -> dict:
    import mlx.core as mx
    from mlx import nn

    from mlx2.runtime.models import qwen4_exp as qwen4
    from mlx2.runtime.models import qwen4_hc_decode as hcd

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("Metal GPU is unavailable")
    install_candidate(hcd)
    args = SimpleNamespace(
        hc_count=4, hidden_size=2560, hc_lowrank=320, rms_norm_eps=1e-6
    )
    mx.random.seed(4245)
    module = qwen4.GatedResidual(args, use_combine=True)
    module.set_dtype(mx.bfloat16)
    nn.quantize(
        module,
        group_size=32,
        bits=4,
        class_predicate=lambda _path, child: isinstance(child, nn.Linear),
    )
    module.eval()
    mx.eval(module.parameters())
    reason = hcd.static_admission(module)
    if reason is not None:
        raise AssertionError(f"synthetic gs32 module was not admitted: {reason}")

    exact = []
    sample = None
    for case in range(cases):
        mx.random.seed(4300 + case)
        x = mx.random.normal((1, 1, 10240)).astype(mx.bfloat16)
        expected = module._composed(x, False, False)
        mixed, inject = hcd.hc_decode_launch(module, x.reshape(1, -1))
        got = (mixed.reshape(1, 1, 2560), x, inject.reshape(1, 1, 4))
        mx.eval(*expected, *got)
        equal = [_bits_equal(mx, a, b) for a, b in zip(expected, got)]
        exact.append(equal)
        sample = x
    if not all(all(row) for row in exact):
        raise AssertionError("gs32 candidate differs from composed MLX bits")

    def composed():
        return module._composed(sample, False, False)

    def candidate():
        mixed, inject = hcd.hc_decode_launch(module, sample.reshape(1, -1))
        return mixed, inject

    timings = {"composed": [], "candidate": []}
    # Alternating order avoids giving one arm every early warm sample.
    for rep in range(reps + 1):
        order = ("composed", "candidate") if rep % 2 == 0 else ("candidate", "composed")
        for arm in order:
            fn = composed if arm == "composed" else candidate
            started = time.perf_counter_ns()
            for _ in range(iterations):
                out = fn()
                mx.eval(*out)
            elapsed_us = (time.perf_counter_ns() - started) / iterations / 1e3
            if rep:
                timings[arm].append(elapsed_us)
    medians = {name: statistics.median(values) for name, values in timings.items()}
    return {
        "schema": "mlx2.qwen4-hc-gs32-candidate.v1",
        "candidate": "process-local exact mlx2 HC kernel with GS=32",
        "sources": ["jundot/omlx#4245", "jundot/omlx#4248"],
        "geometry": {
            "hidden": 2560,
            "streams": 4,
            "lowrank": 320,
            "bits": 4,
            "group_size": 32,
        },
        "exact_cases": cases,
        "all_bit_identical": True,
        "timings_us": timings,
        "median_us": medians,
        "gpu_owner": gpu_owner,
        "candidate_delta_pct": 100 * (medians["candidate"] / medians["composed"] - 1),
        "qualification": False,
        "selected": False,
        "artifact_condition": "run qwen4_upstream_artifact_probe.py; no local artifact benefit is inferred",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=16)
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=48)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()
    if not args.i_own_the_gpu:
        parser.error("refusing Metal execution without --i-own-the-gpu")
    owner = require_matching_gpu_receipts()
    report = run(
        cases=args.cases,
        reps=args.reps,
        iterations=args.iterations,
        gpu_owner=owner,
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
