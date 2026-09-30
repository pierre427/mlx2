#!/usr/bin/env python3
"""Offline Metal compilation, or explicit locked GPU numerical/timing probes.

Default mode never dispatches GPU work. --gpu acquires both host locks and
refuses a running inference/qualification process. These synthetic gates do
not qualify a production model route or claim an HTTP throughput improvement.
"""

import argparse
import fcntl
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["MLX_ENABLE_TF32"] = "0"


def offline_compile(prefill):
    cases = [((7,), (False,)), ((33, 48, 64, 7), (True, False, True, False))]
    sources = [("project", *prefill.source_for(*case), len(case[0])) for case in cases]
    sources += [
        ("swiglu", *prefill.swiglu_source(33, biases), 1)
        for biases in ((False, False), (True, True), (True, False), (False, True))
    ]
    receipts = []
    with tempfile.TemporaryDirectory(prefix="mlx2-prefill-metal-") as folder:
        for i, (kind, source, names, outputs) in enumerate(sources):
            arguments = []
            for name in names:
                if name == "M":
                    arguments.append("constant int& M")
                else:
                    dtype = "uint" if name.startswith("W") else "bfloat"
                    arguments.append(f"const device {dtype}* {name}")
            arguments += [
                f"device bfloat* {'Y' + str(j) if kind == 'project' else 'OUT'}"
                for j in range(outputs)
            ]
            arguments += [
                "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]",
                "uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]",
                "uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]",
            ]
            text = (
                "#include <metal_stdlib>\n"
                + prefill.HEADER
                + "\nusing namespace metal;\n"
                + "kernel void check("
                + ",\n".join(arguments)
                + ") {\nconstexpr int K=512;\n"
                + source
                + "\n}\n"
            )
            path = Path(folder) / f"{i}.metal"
            path.write_text(text)
            result = subprocess.run(
                [
                    "xcrun",
                    "-sdk",
                    "macosx",
                    "metal",
                    "-std=metal4.0",
                    "-c",
                    str(path),
                    "-o",
                    str(path.with_suffix(".air")),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            receipts.append(
                {
                    "kind": kind,
                    "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "exit_code": result.returncode,
                    "diagnostics": result.stderr,
                }
            )
    return receipts


def comparison(mx, a, b):
    af, bf = a.astype(mx.float32), b.astype(mx.float32)
    delta = af - bf
    return {
        "finite": bool(mx.all(mx.isfinite(af)).item()),
        "max_abs": float(mx.max(mx.abs(delta)).item()),
        "relative_l2": float(
            (
                mx.sqrt(mx.sum(delta * delta))
                / mx.maximum(mx.sqrt(mx.sum(bf * bf)), 1e-12)
            ).item()
        ),
        "bit_equal": bool(mx.array_equal(a, b).item()),
    }


def gpu_probes(mx, nn, prefill):
    from collections import Counter

    from mlx2.runtime.models import gated_delta as gdn
    from mlx2.runtime.models.activations import swiglu

    def quantized(k, n, bias):
        linear = nn.Linear(k, n, bias=bias)
        linear.weight = linear.weight.astype(mx.bfloat16)
        if bias:
            linear.bias = linear.bias.astype(mx.bfloat16)
        return nn.QuantizedLinear.from_linear(linear, group_size=64, bits=4)

    mx.random.seed(2910)
    rows = []
    for batch, length, k, n in (
        (1, 19, 64, 7),
        (4, 65, 512, 33),
        (1, 257, 512, 48),
        (1, 512, 5120, 6144),
    ):
        x = mx.random.normal((batch, length, k)).astype(mx.bfloat16)
        gate, up = quantized(k, n, True), quantized(k, n, False)
        mx.eval(x, gate.parameters(), up.parameters())
        fns = {
            "stock": lambda x=x, gate=gate, up=up: (gate(x), up(x)),
            "grouped": lambda x=x, gate=gate, up=up: prefill.project(x, (gate, up)),
            "stock_swiglu": lambda x=x, gate=gate, up=up: swiglu(gate(x), up(x)),
            "fused_swiglu": lambda x=x, gate=gate, up=up: prefill.project_swiglu(
                x, gate, up
            ),
        }
        outputs = {key: fn() for key, fn in fns.items()}
        for y in outputs.values():
            mx.eval(y)
        checks = [
            comparison(mx, a, b) for a, b in zip(outputs["grouped"], outputs["stock"])
        ]
        checks.append(comparison(mx, outputs["fused_swiglu"], outputs["stock_swiglu"]))
        samples = {key: [] for key in fns}
        for repeat in range(7):
            order = list(fns) if repeat % 2 == 0 else list(reversed(fns))
            for key in order:
                start = time.perf_counter()
                mx.eval(fns[key]())
                samples[key].append((time.perf_counter() - start) * 1000)
        rows.append(
            {
                "shape": [batch, length, k, n],
                "checks": checks,
                "milliseconds": {
                    key: statistics.median(v) for key, v in samples.items()
                },
            }
        )
    scans = []
    for batch, length in ((1, 257), (2, 513)):
        q, k = gdn.normalize_gdn_qk(
            mx.random.normal((batch, length, 16, 128)).astype(mx.bfloat16),
            mx.random.normal((batch, length, 16, 128)).astype(mx.bfloat16),
        )
        v = mx.random.normal((batch, length, 16, 128)).astype(mx.bfloat16)
        gate = mx.full((batch, length, 16), 0.98, dtype=mx.float32)
        beta = mx.full((batch, length, 16), 0.5, dtype=mx.float32)
        state = mx.random.normal((batch, 16, 128, 128)) * 0.01
        reference = gdn.gated_delta_kernel(q, k, v, gate, beta, state)
        for tile in (8, 16):
            stats = Counter()
            candidate = gdn._chunked_prefill(
                q, k, v, gate, beta, state, None, tile, stats
            )
            if candidate is None:
                scans.append(
                    {"shape": [batch, length], "tile": tile, "skipped": dict(stats)}
                )
                continue
            checks = [comparison(mx, a, b) for a, b in zip(candidate, reference)]
            # Follow both states with identical ordinary decode operations.
            for _ in range(4):
                candidate = gdn.gated_delta_kernel(
                    q[:, :1], k[:, :1], v[:, :1], gate[:, :1], beta[:, :1], candidate[1]
                )
                reference = gdn.gated_delta_kernel(
                    q[:, :1], k[:, :1], v[:, :1], gate[:, :1], beta[:, :1], reference[1]
                )
                checks.extend(
                    comparison(mx, a, b) for a, b in zip(candidate, reference)
                )
            scans.append(
                {
                    "shape": [batch, length],
                    "tile": tile,
                    "checks": checks,
                    "counters": dict(stats),
                }
            )
            reference = gdn.gated_delta_kernel(q, k, v, gate, beta, state)
    checks = [check for row in rows + scans for check in row.get("checks", [])]
    return {
        "projections": rows,
        "scans": scans,
        "peak_memory_bytes": mx.get_peak_memory(),
        "numerical_gate": all(c["finite"] and c["relative_l2"] <= 0.01 for c in checks),
        "numerical_gate_limit_relative_l2": 0.01,
        "bit_exact_gate": all(c["bit_equal"] for c in checks),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    from mlx import nn

    from mlx2.runtime.models import tensorfold_prefill as prefill

    receipt = {
        "schema": "mlx2.tensorfold-prefill.v1",
        "qualification": "candidate",
        "offline_compile": offline_compile(prefill),
        "gpu_executed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.gpu:
            if any(row["exit_code"] for row in receipt["offline_compile"]):
                raise RuntimeError("offline Metal compilation failed")
            with ExitStack() as stack:
                for path in ("/Users/Shared/mlxuag/gpu.lock", "/tmp/gpu.lock"):
                    handle = stack.enter_context(open(path, "a"))
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                processes = subprocess.check_output(
                    ["ps", "-axo", "command"], text=True
                )
                if any(
                    "-m mlx2.server" in line
                    or "tensorfold serve" in line
                    or "run_perf.py" in line
                    for line in processes.splitlines()
                ):
                    raise RuntimeError("an inference/qualification process is running")
                receipt["initial_thermal"] = json.loads(
                    subprocess.check_output(
                        ["swift", str(ROOT / "scripts/thermal_probe.swift")], text=True
                    )
                )
                if receipt["initial_thermal"]["thermal_state"] != 0:
                    raise RuntimeError("GPU must cool to nominal thermal state")
                mx.set_default_device(mx.gpu)
                receipt["gpu"] = gpu_probes(mx, nn, prefill)
                receipt["gpu_executed"] = True
                receipt["final_thermal"] = json.loads(
                    subprocess.check_output(
                        ["swift", str(ROOT / "scripts/thermal_probe.swift")], text=True
                    )
                )
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(args.output)
    return int(
        any(row["exit_code"] for row in receipt["offline_compile"])
        or (args.gpu and not receipt["gpu"]["numerical_gate"])
    )


if __name__ == "__main__":
    raise SystemExit(main())
