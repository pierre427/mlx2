#!/usr/bin/env python3
"""Offline compile, or a locked GPU check, of the metal_q4native prefill tile.

Default mode never dispatches GPU work: it compiles every generated
``metal_q4native`` source (and, for reference, the ``metal`` sources) with
``xcrun -sdk macosx metal`` and writes a receipt.

``--gpu`` (run only by the GPU owner) takes both host GPU locks, refuses a
running inference/qualification process and a non-nominal thermal state,
then for each 27B projection shape (K x N) 5120x6144, 17408x5120,
5120x17408 and synthetic affine q4/g64 weights:

* accuracy: max |candidate - fp32 reference| <= 2 * max |MLX - fp32 reference|
  + one bf16 ulp of the output scale, where the reference dequantizes with
  MLX and multiplies in fp32 -- the rule of ``scripts/lane_matmul_gate.py``.
  Checked for ``metal_q4native`` and, for comparison, ``metal``;
* row behaviour: every row of an R-row call is bitwise equal to the same row
  computed in a 32-row call and in a call with a different row count
  (tiles are row-independent by construction: per-row products, per-row
  group sums, groups in fixed order); row counts include partial 32-row
  tiles;
* fused SwiGLU: ``metal_q4native`` fused gate/up/SwiGLU vs fp32 reference;
* timing: interleaved (ABAB / BABA) median ms for ``metal``,
  ``metal_q4native`` and MLX's ``quantized_matmul`` at 512, 2048 and 8192
  rows.

Synthetic gates only: this does not qualify a model route.
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

SHAPES = ((5120, 6144), (17408, 5120), (5120, 17408))   # (K, N)
TIMING_ROWS = (512, 2048, 8192)
ROW_CHECK = (1, 17, 31, 32, 33, 63, 64, 65, 100, 512)


def offline_compile():
    from mlx2.runtime.models import prefill_metal, prefill_metal_q4native

    cases = [((7,), (False,)), ((33, 48, 64, 7), (True, False, True, False)),
             ((6144,), (False,))]
    receipts = []
    for law, module in (("metal", prefill_metal), ("metal_q4native", prefill_metal_q4native)):
        sources = [("project", *module.source_for(*case), len(case[0])) for case in cases]
        sources += [("swiglu", *module.swiglu_source(33, biases), 1)
                    for biases in ((False, False), (True, True), (True, False), (False, True))]
        with tempfile.TemporaryDirectory(prefix="mlx2-prefill-q4native-") as folder:
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
                ]
                text = ("#include <metal_stdlib>\n" + module.HEADER
                        + "\nkernel void check(" + ",\n".join(arguments)
                        + ") {\nconstexpr int K=5120;\n" + source + "\n}\n")
                path = Path(folder) / f"{i}.metal"
                path.write_text(text)
                result = subprocess.run(
                    ["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-c",
                     str(path), "-o", str(path.with_suffix(".air"))],
                    capture_output=True, text=True, check=False)
                receipts.append({"law": law, "kind": kind,
                                 "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                 "exit_code": result.returncode,
                                 "diagnostics": result.stderr[-4000:]})
    return receipts


def _timed(mx, fn, reps=7):
    for _ in range(2):
        mx.eval(fn())
    out = []
    for _ in range(reps):
        start = time.perf_counter()
        mx.eval(fn())
        out.append((time.perf_counter() - start) * 1000)
    return out


def gpu_checks(mx, nn, prefill):
    import numpy as np

    def quantized(k, n, key):
        linear = nn.Linear(k, n, bias=False)
        linear.weight = (mx.random.normal((n, k), key=key) * 0.02).astype(mx.bfloat16)
        return nn.QuantizedLinear.from_linear(linear, group_size=64, bits=4)

    key = mx.random.key(1028)
    cases = []
    for k, n in SHAPES:
        key, kw, ku, kx = mx.random.split(key, 4)
        module, up = quantized(k, n, kw), quantized(k, n, ku)
        x = mx.random.normal((1, max(TIMING_ROWS), k), key=kx).astype(mx.bfloat16)
        mx.eval(x, module.parameters(), up.parameters())
        ref_w = mx.dequantize(module.weight, module.scales, module.biases,
                              group_size=64, bits=4).astype(mx.float32)
        xs = x[:, :64]
        ref = xs.astype(mx.float32) @ ref_w.T
        scale = float(mx.max(mx.abs(ref)).item())
        ulp = scale * 2.0 ** -7

        def err(y, ref=ref):
            return float(mx.max(mx.abs(y.astype(mx.float32) - ref)).item())

        stock_err = err(module(xs))
        record = {"k": k, "n": n, "stock_max_err": stock_err, "ref_max_abs": scale}
        for backend in ("metal", "metal_q4native"):
            got = err(prefill.project(xs, (module,), backend=backend)[0])
            record[f"{backend}_max_err"] = got
            record[f"{backend}_accurate"] = got <= 2 * stock_err + ulp
        # Row behaviour: rows are independent of how many share the call.
        full = prefill.project(x[:, :512], (module,), backend="metal_q4native")[0]
        mx.eval(full)
        bad = []
        for rows in ROW_CHECK:
            # Exactly ``rows`` rows in a fresh buffer: a read past the last
            # row cannot land in valid memory.
            fresh = mx.array(np.array(x[:, :rows].astype(mx.float32))).astype(mx.bfloat16)
            got = prefill.project(fresh, (module,), backend="metal_q4native")[0]
            if not bool(mx.array_equal(got, full[:, :rows]).item()):
                bad.append(rows)
        record["row_fail_counts"] = bad
        # Fused SwiGLU against the fp32 reference of the same law family.
        ref_u = mx.dequantize(up.weight, up.scales, up.biases,
                              group_size=64, bits=4).astype(mx.float32)
        g32, u32 = ref, xs.astype(mx.float32) @ ref_u.T
        ref_swiglu = g32 * mx.sigmoid(g32) * u32
        g_stock, u_stock = module(xs), up(xs)
        stock_swiglu = g_stock * mx.sigmoid(g_stock) * u_stock
        fused = prefill.project_swiglu(xs, module, up, backend="metal_q4native")
        s_err = float(mx.max(mx.abs(stock_swiglu.astype(mx.float32) - ref_swiglu)).item())
        f_err = float(mx.max(mx.abs(fused.astype(mx.float32) - ref_swiglu)).item())
        s_scale = float(mx.max(mx.abs(ref_swiglu)).item())
        record.update(swiglu_stock_max_err=s_err, swiglu_q4native_max_err=f_err,
                      swiglu_accurate=f_err <= 2 * s_err + s_scale * 2.0 ** -7)
        timing = {}
        for rows in TIMING_ROWS:
            xr = x[:, :rows]
            fns = {
                "metal": lambda xr=xr, m=module: prefill.project(xr, (m,), backend="metal")[0],
                "metal_q4native": lambda xr=xr, m=module: prefill.project(
                    xr, (m,), backend="metal_q4native")[0],
                "stock": lambda xr=xr, m=module: m(xr),
            }
            samples = {name: [] for name in fns}
            for repeat in range(6):
                order = list(fns) if repeat % 2 == 0 else list(reversed(fns))
                for name in order:
                    samples[name].extend(_timed(mx, fns[name], reps=3))
            timing[str(rows)] = {name: round(statistics.median(v), 4)
                                 for name, v in samples.items()}
        record["median_ms"] = timing
        record["passed"] = bool(record["metal_q4native_accurate"] and not bad
                                and record["swiglu_accurate"])
        print(json.dumps(record), flush=True)
        cases.append(record)
    return {"cases": cases, "passed": all(c["passed"] for c in cases),
            "peak_memory_bytes": mx.get_peak_memory()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    from mlx import nn

    from mlx2.runtime.models import prefill_metal_q4native
    from mlx2.runtime.models import tensorfold_prefill as prefill

    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=False).stdout.strip()
    receipt = {
        "schema": "mlx2.prefill-q4native-check.v1",
        "qualification": "candidate",
        "mlx2_commit": git,
        "checkpoint": "synthetic-affine-q4g64@seed1028",
        "kernel_identity": prefill_metal_q4native.KERNEL_IDENTITY,
        "offline_compile": offline_compile(),
        "gpu_executed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite nonempty receipt {args.output}")
    try:
        if args.gpu:
            if any(row["exit_code"] for row in receipt["offline_compile"]):
                raise RuntimeError("offline Metal compilation failed")
            with ExitStack() as stack:
                for path in ("/Users/Shared/mlxuag/gpu.lock", "/tmp/gpu.lock"):
                    if Path(path).is_dir():
                        # scripts/run_with_gpu_locks.py holds the window: the
                        # sentinels are owner directories, not flock files.
                        owner = Path(path) / "owner.json"
                        if not owner.is_file():
                            raise RuntimeError(f"GPU lock directory without owner receipt: {path}")
                        receipt.setdefault("gpu_lock_owner", json.loads(owner.read_text()))
                        continue
                    handle = stack.enter_context(open(path, "a"))
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                processes = subprocess.check_output(["ps", "-axo", "command"], text=True)
                if any("-m mlx2.server" in line or "tensorfold serve" in line
                       or "run_perf.py" in line for line in processes.splitlines()):
                    raise RuntimeError("an inference/qualification process is running")
                probe = ["swift", str(ROOT / "scripts/thermal_probe.swift")]
                receipt["initial_thermal"] = json.loads(subprocess.check_output(probe, text=True))
                if receipt["initial_thermal"]["thermal_state"] != 0:
                    raise RuntimeError("GPU must cool to nominal thermal state")
                mx.set_default_device(mx.gpu)
                info = mx.device_info()
                receipt["device"] = {k: str(info.get(k)) for k in ("device_name", "architecture")}
                receipt["mlx"] = mx.__version__
                receipt["gpu"] = gpu_checks(mx, nn, prefill)
                receipt["gpu_executed"] = True
                receipt["final_thermal"] = json.loads(subprocess.check_output(probe, text=True))
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(args.output)
    return int(any(row["exit_code"] for row in receipt["offline_compile"])
               or (args.gpu and not receipt["gpu"]["passed"]))


if __name__ == "__main__":
    raise SystemExit(main())
