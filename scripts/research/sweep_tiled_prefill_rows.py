#!/usr/bin/env python3
"""Row-count crossover of lane ``tiled_prefill`` vs stock quantized_matmul.

The F4 bench (qualification/runs/f4-fused-prefill-20261105) found lane
``tiled_prefill`` bit-exact with stock and +20% at 512 rows but -17% at 8192
on a 27B gate/up. This sweep times every 4-bit projection shape of
Qwen3.8-27B (layers 0-3 incl. MTP) at prefill row counts above the lane
kernel's window, in the ordinary layout the lane installer uses (nt=0) and
the TensorFold tiled layout (nt=32), and records bit-exactness against stock,
to choose a row ceiling for a short-prefill-only admission.

Weights are seeded random BF16 quantized with mx.quantize (affine q4/g64);
activations unit-normal BF16. Component timings, median of 9, no thermal hold.

Usage: ``sweep_tiled_prefill_rows.py owned OUT``. Original mlx2 code.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# (name, N, K): Qwen3.8-27B 4-bit projections.
SHAPES = (("k_or_v_proj", 1024, 5120), ("o_proj", 5120, 6144), ("in_proj_qkv", 10240, 5120),
          ("q_proj", 12288, 5120), ("gate_or_up", 17408, 5120), ("mtp_down", 5120, 17408))
ROWS = (160, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192)


def checkout_env() -> dict:
    environment = dict(os.environ, MLX2_TILED_PREFILL="1")
    environment["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(ROOT / "src"), environment.get("PYTHONPATH", "")) if p)
    return environment


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def source() -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    return {"commit": git("rev-parse", "HEAD"),
            "dirty_tracked": bool(git("status", "--porcelain", "--untracked-files=no"))}


def host_sample() -> dict:
    return {"thermal": subprocess.check_output(["pmset", "-g", "therm"], text=True),
            "swap": subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()}


def unit(out: Path) -> int:
    import mlx.core as mx

    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.runtime.lane import tiled_prefill
    from mlx2.runtime.tensorfold_qwen38.lane_qmm import tile_weight

    receipt = {"schema": "mlx2.tiled-prefill-row-sweep.v1", "source": source(),
               "checkpoint": "synthetic-qwen38-27b-q4g64-shapes", "device": mx.device_info(),
               "mlx": mx.__version__, "host_before": host_sample(), "cells": []}

    def timed(fn, x):
        mx.eval(fn(x))
        for _ in range(2):
            mx.eval(fn(x))
        times = []
        for _ in range(9):
            mx.synchronize()
            start = time.perf_counter()
            mx.eval(fn(x))
            mx.synchronize()
            times.append(time.perf_counter() - start)
        return sorted(times)[4]

    try:
        key = mx.random.key(20261105)
        for name, n, k in SHAPES:
            dense = (mx.random.normal((n, k), key=key) * 0.02).astype(mx.bfloat16)
            weight, scales, biases = mx.quantize(dense, group_size=64, bits=4)
            tiled = tile_weight(weight, 32, 64, bits=4)
            mx.eval(weight, scales, biases, tiled)
            del dense

            class Ordinary:
                bits, group_size = 4, 64

            Ordinary.weight, Ordinary.scales, Ordinary.biases = weight, scales, biases

            class Tiled(Ordinary):
                pass

            Tiled.weight = tiled

            def stock(x, weight=weight, scales=scales, biases=biases):
                return mx.quantized_matmul(x, weight, scales=scales, biases=biases, transpose=True,
                                           group_size=64, bits=4)

            def nt0(x):
                y = tiled_prefill.project(x, Ordinary, 0)
                if y is None:
                    raise RuntimeError("declined")
                return y

            def nt32(x):
                y = tiled_prefill.project(x, Tiled, 32)
                if y is None:
                    raise RuntimeError("declined")
                return y

            for rows in ROWS:
                x = mx.random.normal((rows, k), key=key).astype(mx.bfloat16)
                reference = stock(x)
                mx.eval(x, reference)
                cell = {"shape": name, "N": n, "K": k, "rows": rows}
                for arm, fn in (("stock", stock), ("nt0", nt0), ("nt32", nt32)):
                    try:
                        seconds = timed(fn, x)
                        y = fn(x)
                        mx.eval(y)
                        cell[arm] = {"ms": seconds * 1e3, "tflops": 2 * rows * n * k / seconds / 1e12,
                                     "exact": bool(mx.array_equal(y, reference).item())}
                    except Exception as exc:  # noqa: BLE001 - a refusal is evidence
                        cell[arm] = {"error": repr(exc)}
                for arm in ("nt0", "nt32"):
                    if "ms" in cell.get(arm, {}):
                        cell[arm]["speedup_vs_stock"] = cell["stock"]["ms"] / cell[arm]["ms"]
                receipt["cells"].append(cell)
                print(json.dumps({"shape": name, "rows": rows,
                                  **{a: round(cell[a].get("speedup_vs_stock", 1.0), 3)
                                     for a in ("nt0", "nt32")},
                                  "exact": [cell[a].get("exact") for a in ("nt0", "nt32")]}), flush=True)
            atomic_json(out / "receipt.json", receipt)
        receipt["passed"] = True
    except BaseException as exc:
        receipt.update(passed=False, error=repr(exc))
        raise
    finally:
        receipt["host_after"] = host_sample()
        atomic_json(out / "receipt.json", receipt)
    return 0


def owned(out: Path, job_id: str) -> int:
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    with DesktopCPGLease(out, job_id, checkpoint="synthetic-qwen38-27b-q4g64-shapes",
                         source_revision=source()["commit"]) as lease:
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                   "--session", lease.owner["cpg_session"], "--label", job_id,
                   "--desktop-cpg-lease", str(lease.path),
                   "--receipt", str(out / "paired-locks.json"), "--",
                   sys.executable, "-u", str(Path(__file__)), "unit", str(out)]
        child = subprocess.Popen(command, cwd=ROOT, env=checkout_env(), start_new_session=True)
        try:
            while child.poll() is None:
                lease.check()
                time.sleep(1)
            lease.success = child.returncode == 0
            return child.returncode
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=120)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("owned", "unit"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--job-id", default="tiled-prefill-row-sweep-20261105")
    args = parser.parse_args()
    out = args.out.resolve()
    return unit(out) if args.mode == "unit" else owned(out, args.job_id)


if __name__ == "__main__":
    raise SystemExit(main())
