#!/usr/bin/env python3
"""Bounded synthetic Metal functional probe; no model qualification or defaults.

Run under cpg_job and run_with_gpu_locks. Refuses nonempty output directories.
Records the exact source and synthetic seed; these are not checkpoint results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path


def atomic(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def swapouts():
    text = subprocess.check_output(["vm_stat"], text=True)
    return int(next(line for line in text.splitlines() if line.startswith("Swapouts:")).split()[-1].rstrip("."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    parser.add_argument("--timing", action="store_true", help="also explore resident/rotating synthetic projection costs")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError("refusing to overwrite nonempty evidence directory")
    args.out.mkdir(parents=True, exist_ok=True)
    from qualification_gpu_ownership import require_qualification_lease
    from run_single_model_smoke_ladder import thermal_sample
    from run_with_gpu_locks import foreign_model_processes

    owner = json.loads((args.cpg_owner_lock / "owner.json").read_text())
    ownership = require_qualification_lease(task_id=owner["cpg_task"],
                                          generation=owner["cpg_generation"],
                                          cpg_owner_lock=args.cpg_owner_lock)
    if foreign_model_processes():
        raise RuntimeError("foreign model process contamination")
    thermal = thermal_sample()
    if thermal["thermal_state"] not in (0, 1):
        raise RuntimeError("thermal admission refused")
    swap0 = swapouts()
    root = Path(__file__).resolve().parents[1]
    files = ["src/mlx2/runtime/fused_mlp_stage_a.py", "src/mlx2/runtime/int8_prefill.py",
             "src/mlx2/runtime/models/row_exact_qmv.py", "src/mlx2/runtime/models/qwen3_next.py",
             "scripts/probe_uzu_kernels.py"]
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}
    result = {"schema": "mlx2.uzu-synthetic-probe.v1", "checkpoint": "not-applicable:synthetic-fixtures",
              "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "source_files": hashes, "seed": 1027, "qualification": False,
              "ownership": ownership, "thermal_start": thermal, "cells": [], "status": "running"}
    atomic(args.out / "result.json", result)
    try:
        import mlx.core as mx
        import numpy as np
        from mlx import nn

        from mlx2.runtime import fused_mlp_stage_a as fused
        from mlx2.runtime import int8_prefill as ip
        from mlx2.runtime.models import row_exact_qmv as qmv
        from mlx2.runtime.models.activations import swiglu
        from mlx2.runtime.models.qwen3_next import Qwen3NextMLP

        mx.set_default_device(mx.gpu)
        mx.random.seed(1027)
        result["device"] = mx.device_info().get("device_name")
        result["mlx_version"] = mx.__version__

        def equal(a, b):
            mx.eval(a, b)
            if a.dtype in (mx.bfloat16, mx.float16):
                a, b = a.view(mx.uint16), b.view(mx.uint16)
            return a.shape == b.shape and bool(mx.array_equal(a, b).item())

        def record(cell):
            result["cells"].append(cell)
            atomic(args.out / "result.json", result)

        def timed(fn):
            mx.eval(fn())
            samples = []
            for _ in range(5):
                start = time.perf_counter()
                mx.eval(fn())
                mx.synchronize()
                samples.append((time.perf_counter() - start) * 1e6)
            return statistics.median(samples)

        # Every finite BF16 gate encoding in [-32,32], paired with varying up
        # values; catches the dtype/rounding boundaries that random-only tests miss.
        bits = np.arange(65536, dtype=np.uint16)
        vals = (bits.astype(np.uint32) << 16).view(np.float32)
        bits = bits[np.isfinite(vals) & (np.abs(vals) <= 32)]
        bits = np.resize(bits, ((len(bits) + 511) // 512) * 512)
        gate = mx.array(bits).view(mx.bfloat16).reshape(-1, 512)
        up = (mx.random.normal(gate.shape) * 1.7).astype(mx.bfloat16)
        for layout, stage in (("q8", ip.q8_stage_a), ("q45", ip.q45_stage_a)):
            for scale in ip.Q8_ACT_SCALES:
                ordinary = stage(swiglu(gate, up), scale)
                candidate = fused.prepare(gate, up, layout=layout, act_scale=scale)
                matches = [equal(a, b) for a, b in zip(ordinary, candidate)]
                record({"kind": "producer", "layout": layout, "act_scale": scale,
                        "shape": list(gate.shape), "exact_qa_sa_ra": matches})
                if not all(matches):
                    raise AssertionError(f"fused producer parity: {layout}/{scale}: {matches}")

        for k in (64, 192, 576):
            gate = mx.random.normal((3, k)).astype(mx.bfloat16)
            up = mx.random.normal(gate.shape).astype(mx.bfloat16)
            for layout, stage in (("q8", ip.q8_stage_a), ("q45", ip.q45_stage_a)):
                for scale in ip.Q8_ACT_SCALES:
                    ordinary = stage(swiglu(gate, up), scale)
                    candidate = fused.prepare(gate, up, layout=layout, act_scale=scale)
                    matches = [equal(a, b) for a, b in zip(ordinary, candidate)]
                    record({"kind": "producer_tail", "layout": layout, "act_scale": scale,
                            "shape": list(gate.shape), "exact_qa_sa_ra": matches})
                    if not all(matches):
                        raise AssertionError("fused producer partial-tile parity")

        # QMV explicit tilings must keep each row's ordinary one-token bits.
        for bits in (4, 5, 6, 8):
            fixtures = [(65, 512, 64, mx.bfloat16), (256, 1024, 64, mx.bfloat16),
                        (512, 576, 64, mx.bfloat16), (65, 1024, 32, mx.float16),
                        (65, 1024, 128, mx.float16)]
            for n, k, group_size, dtype in fixtures:
                base = nn.Linear(k, n, bias=False)
                base.set_dtype(dtype)
                linear = nn.QuantizedLinear.from_linear(base, bits=bits, group_size=group_size)
                x = mx.random.normal((3, k)).astype(dtype)
                if qmv.decline_reason(linear, x) is not None:
                    continue
                reference = qmv.per_row(linear, x)
                fast = qmv.qmv_fast_layout(k, n, bits)
                for rows_per_group in range(1, min(3, qmv.row_group_limit(bits, fast)) + 1):
                    for rps in (1, 2, 4, 8):
                        geometry = qmv.TileGeometry(rows_per_group, rps)
                        got, route = qmv.quantized_linear(linear, x, geometry=geometry)
                        match = equal(got, reference)
                        record({"kind": "qmv", "bits": bits, "m": 3, "n": n, "k": k,
                                "group_size": group_size, "dtype": str(dtype),
                                "rows_per_group": rows_per_group, "results_per_simdgroup": rps,
                                "route": route, "exact": match})
                        if not match:
                            raise AssertionError("explicit QMV tile changed serial arithmetic")

        # Actual adapter-owned dense MLP hook, with real packed consumers.
        class Model(nn.Module):
            def __init__(self, bits):
                super().__init__()
                self.mlp = Qwen3NextMLP(256, 512)
                self.set_dtype(mx.bfloat16)
                self.mlp.down_proj = nn.QuantizedLinear.from_linear(self.mlp.down_proj, bits=bits, group_size=64)
                self.eval()

            def __call__(self, x):
                return self.mlp(x)

        for bits in (4, 5, 8):
            model = Model(bits)
            x = mx.random.normal((1, 64, 256)).astype(mx.bfloat16)
            options = {"enabled": True, "row_threshold": 32,
                       "q8_inplace" if bits == 8 else "q45_inplace": True,
                       "q8_only" if bits == 8 else "inplace_only": True}
            handle = ip.apply(model, options)
            try:
                ordinary = model(x)
                mx.eval(ordinary)
                baseline_us = timed(lambda model=model, x=x: model(x))
                ordinary_one = model(x[:, :1])
                mx.eval(ordinary_one)
            finally:
                ip.remove(handle)
            handle = ip.apply(model, {**options, "fused_mlp_stage_a": True})
            try:
                got = model(x)
                match = equal(got, ordinary)
                one_match = equal(model(x[:, :1]), ordinary_one)
                candidate_us = timed(lambda model=model, x=x: model(x))
                receipt = handle.receipt()
                record({"kind": "mlp", "bits": bits, "shape": list(x.shape),
                        "exact_vs_unfused_int8": match, "ordinary_decode_exact": one_match,
                        "receipt": receipt, "exploratory_resident_us": {
                            "unfused": baseline_us, "fused": candidate_us}})
                if not match or not one_match or not receipt["fused_mlp_stage_a"]["calls"]:
                    raise AssertionError("fused MLP integration parity/engagement failed")
            finally:
                ip.remove(handle)
        if args.timing:
            # Both arms use the same consumer arithmetic. Alternate timing
            # order; never install a winning tile into the serving policy.
            n, k, rows = 5120, 17408, 3
            copies = (256 * 1024**2 + n * k // 2 - 1) // (n * k // 2)
            pool = []
            for _ in range(copies):
                linear = nn.QuantizedLinear(k, n, bits=4, group_size=64, bias=False)
                linear.scales = linear.scales.astype(mx.bfloat16)
                linear.biases = linear.biases.astype(mx.bfloat16)
                mx.eval(linear.parameters())
                pool.append(linear)
            x = mx.random.normal((rows, k)).astype(mx.bfloat16)
            functions = {"default": lambda: [qmv.quantized_linear(lin, x)[0] for lin in pool]}
            for rps in (1, 2, 4, 8):
                geometry = qmv.TileGeometry(2, rps)
                functions[f"rows2_rps{rps}"] = lambda geometry=geometry: [
                    qmv.quantized_linear(lin, x, geometry=geometry)[0] for lin in pool]
            samples = {name: [] for name in functions}
            for iteration in range(5):
                names = list(functions) if iteration % 2 == 0 else list(functions)[::-1]
                for name in names:
                    samples[name].append(timed(functions[name]) / copies)
            record({"kind": "qmv_timing", "shape": [rows, n, k], "bits": 4,
                    "working_set": "rotating distinct allocations; not proof of cold DRAM",
                    "copies": copies, "packed_weight_bytes": copies * n * k // 2,
                    "exploratory_us_per_projection": samples})
            del pool, functions
            gate = mx.random.normal((512, 17408)).astype(mx.bfloat16)
            up = mx.random.normal(gate.shape).astype(mx.bfloat16)
            mx.eval(gate, up)
            functions = {
                "unfused": lambda: ip.q45_stage_a(swiglu(gate, up), "group64"),
                "fused": lambda: fused.prepare(gate, up, layout="q45", act_scale="group64"),
            }
            samples = {name: [] for name in functions}
            for iteration in range(5):
                for name in (list(functions) if iteration % 2 == 0 else list(functions)[::-1]):
                    samples[name].append(timed(functions[name]))
            record({"kind": "producer_timing", "shape": list(gate.shape),
                    "working_set": "resident", "exploratory_us": samples})
        result["thermal_end"] = thermal_sample()
        result["swapouts_delta"] = swapouts() - swap0
        result["status"] = "passed"
        result["performance_scope"] = "exploratory resident synthetic timings; no model or throughput claim"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        atomic(args.out / "result.json", result)


if __name__ == "__main__":
    main()
