#!/usr/bin/env python3
"""F4 microbench: IronMLX fused affine4 prefill/SwiGLU vs existing mlx2 paths.

Feature 4 (d5593d878, ``experimental/ironmlx_tiled_prefill.py``) is a fused
gate/up GEMM + SwiGLU epilogue over the existing tiled q4 layout. Before
asking whether it helps, measure what the existing paths already achieve
(lesson 2026-11-05: check baseline TFLOP/s before expecting a port to win).

One MLP gate/up + activation of a real Qwen3.8-27B layer (K=5120,
N=2x17408, affine q4/g64 BF16), random unit-scale BF16 activations:

  stock        mx.quantized_matmul (fused gate/up) + compiled silu(g)*u
  lane_tiled   runtime/lane/tiled_prefill.project (NAX, tiled nt32) + silu*u
  tf_metal     runtime/models/tensorfold_prefill.project_swiglu (fused, M5)
  f4_fused     experimental/ironmlx_tiled_prefill.project (fused, tiled nt32)

Per arm and row count: median ms, effective TFLOP/s (2*M*K*N), and parity
against ``stock`` (max |diff| / max |ref|, mean |diff|, exact-equal share);
plus F4 against its CPU oracle at partial-tile shapes. Not a thermal-gated
measurement; component timings only.

Usage: ``bench_f4_fused_prefill.py owned OUT --model PATH [--layer 10]``.
Original mlx2 code.
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
ROWS = (512, 2048, 4096, 8192)
PARTIAL_ROWS = (129, 4100)


def checkout_env() -> dict:
    environment = dict(os.environ, MLX2_TILED_PREFILL="1", MLX2_IRONMLX_FUSED_PREFILL="1")
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


def load_layer(model: Path, layer: int):
    import mlx.core as mx

    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = None
    for key in index:
        if key.endswith(f"layers.{layer}.mlp.gate_proj.weight"):
            prefix = key[: -len("gate_proj.weight")]
    if prefix is None:
        raise ValueError(f"layer {layer} gate_proj not found")
    out, shards = {}, {}
    for proj in ("gate_proj", "up_proj"):
        for part in ("weight", "scales", "biases"):
            key = f"{prefix}{proj}.{part}"
            shard = index[key]
            if shard not in shards:
                shards[shard] = mx.load(str(model / shard))
            out[(proj, part)] = shards[shard][key]
    return prefix, out


def unit(out: Path, model: Path, layer: int) -> int:
    import mlx.core as mx
    from mlx import nn

    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.adapters.registry import inspect_model
    from mlx2.experimental import ironmlx_tiled_prefill as f4
    from mlx2.runtime.lane import tiled_prefill as lane_tiled
    from mlx2.runtime.models import tensorfold_prefill as tf
    from mlx2.runtime.tensorfold_qwen38.lane_qmm import tile_weight

    receipt = {"schema": "mlx2.f4-fused-prefill-bench.v1", "source": source(),
               "checkpoint": f"{model.name}@sha256:{inspect_model(model).artifact_fingerprint}",
               "device": mx.device_info(), "mlx": mx.__version__, "layer": layer,
               "host_before": host_sample(), "rows": {}, "partial": {}, "qualified": False}
    try:
        prefix, w = load_layer(model, layer)
        receipt["prefix"] = prefix
        gw, gs, gb = w[("gate_proj", "weight")], w[("gate_proj", "scales")], w[("gate_proj", "biases")]
        uw, us, ub = w[("up_proj", "weight")], w[("up_proj", "scales")], w[("up_proj", "biases")]
        n, kw = gw.shape
        k = kw * 8
        if gs.shape != (n, k // 64) or gw.dtype != mx.uint32 or gs.dtype != mx.bfloat16:
            raise ValueError(f"layer {layer} gate/up is not affine q4/g64 bf16: {gw.shape} {gs.shape}")
        fw, fs, fb = (mx.concatenate(pair, axis=0) for pair in ((gw, uw), (gs, us), (gb, ub)))
        tiled = tile_weight(fw, 32, 64, bits=4)
        mx.eval(fw, fs, fb, tiled)
        receipt["geometry"] = {"K": k, "N_each": n, "N_fused": 2 * n}

        def as_linear(weight, scales, biases):
            linear = nn.QuantizedLinear(k, int(weight.shape[0]), bias=False, group_size=64, bits=4)
            linear.weight, linear.scales, linear.biases = weight, scales, biases
            return linear

        gate_lin, up_lin = as_linear(gw, gs, gb), as_linear(uw, us, ub)

        class Tiled:
            weight, scales, biases, bits, group_size = tiled, fs, fb, 4, 64

        swiglu = mx.compile(lambda y: nn.silu(y[..., :n]) * y[..., n:])

        def stock(x):
            return swiglu(mx.quantized_matmul(x, fw, scales=fs, biases=fb, transpose=True,
                                              group_size=64, bits=4))

        def lane(x):
            y = lane_tiled.project(x, Tiled, 32)
            if y is None:
                raise RuntimeError("lane tiled_prefill declined")
            return swiglu(y)

        def tf_metal(x):
            return tf.project_swiglu(x, gate_lin, up_lin, backend="metal")

        def f4_fused(x):
            return f4.project(x, tiled, fs, fb, ntcol=32)

        arms = {"stock": stock, "lane_tiled": lane, "tf_metal": tf_metal, "f4_fused": f4_fused}

        def compare(ref, got):
            ref32, got32 = ref.astype(mx.float32), got.astype(mx.float32)
            diff = mx.abs(ref32 - got32)
            scale = mx.max(mx.abs(ref32)).item()
            return {"max_abs": mx.max(diff).item(), "max_rel_to_peak": mx.max(diff).item() / scale,
                    "mean_abs": mx.mean(diff).item(), "exact_share": mx.mean(ref32 == got32).item(),
                    "finite": bool(mx.all(mx.isfinite(got32)).item())}

        key = mx.random.key(20261105)
        for rows in ROWS:
            x = mx.random.normal((rows, k), key=key).astype(mx.bfloat16)
            mx.eval(x)
            cell = {}
            reference = stock(x)
            mx.eval(reference)
            for name, fn in arms.items():
                try:
                    y = fn(x)
                    mx.eval(y)
                    for _ in range(3):
                        mx.eval(fn(x))
                    times = []
                    for _ in range(10):
                        mx.synchronize()
                        start = time.perf_counter()
                        mx.eval(fn(x))
                        mx.synchronize()
                        times.append(time.perf_counter() - start)
                    times.sort()
                    median = times[len(times) // 2]
                    cell[name] = {"median_ms": median * 1e3, "min_ms": times[0] * 1e3,
                                  "tflops": 2 * rows * k * 2 * n / median / 1e12,
                                  "parity_vs_stock": compare(reference, y)}
                except Exception as exc:  # noqa: BLE001 - a refusal is evidence
                    cell[name] = {"error": repr(exc)}
            receipt["rows"][str(rows)] = cell
            atomic_json(out / "receipt.json", receipt)
            print(json.dumps({"rows": rows, **{a: round(c.get("tflops", 0), 2) for a, c in cell.items()}}),
                  flush=True)

        import numpy as np

        for rows in PARTIAL_ROWS:
            x = mx.random.normal((rows, k), key=key).astype(mx.bfloat16)
            got = f4_fused(x)
            reference = stock(x)
            mx.eval(got, reference)
            entry = {"parity_vs_stock": compare(reference, got)}
            # CPU oracle on a 256-column slice (gate cols 0..127 + matching up cols).
            cols = mx.concatenate([mx.arange(128), n + mx.arange(128)])
            sw, ss, sb = fw[cols], fs[cols], fb[cols]
            small = f4.project(x[:min(rows, 129)], tile_weight(sw, 32, 64, bits=4), ss, sb, ntcol=32)
            oracle = f4.reference(np.array(x[:min(rows, 129)].astype(mx.float32)),
                                  np.array(tile_weight(sw, 32, 64, bits=4)),
                                  np.array(ss.astype(mx.float32)), np.array(sb.astype(mx.float32)))
            mx.eval(small)
            got_small = np.array(small.astype(mx.float32))
            entry["cpu_oracle_slice"] = {
                "max_abs": float(np.max(np.abs(got_small - oracle))),
                "max_rel_to_peak": float(np.max(np.abs(got_small - oracle)) / np.max(np.abs(oracle))),
                "exact_share": float(np.mean(got_small == oracle))}
            receipt["partial"][str(rows)] = entry
        receipt["passed"] = True
    except BaseException as exc:
        receipt.update(passed=False, error=repr(exc))
        raise
    finally:
        receipt["host_after"] = host_sample()
        atomic_json(out / "receipt.json", receipt)
    return 0


def owned(out: Path, model: Path, layer: int, job_id: str) -> int:
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    with DesktopCPGLease(out, job_id, checkpoint=model.name, source_revision=source()["commit"]) as lease:
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                   "--session", lease.owner["cpg_session"], "--label", job_id,
                   "--desktop-cpg-lease", str(lease.path),
                   "--receipt", str(out / "paired-locks.json"), "--",
                   sys.executable, "-u", str(Path(__file__)), "unit", str(out), "--model", str(model),
                   "--layer", str(layer)]
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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--job-id", default="f4-fused-prefill-bench-20261105")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.mode == "unit":
        return unit(out, args.model, args.layer)
    return owned(out, args.model, args.layer, args.job_id)


if __name__ == "__main__":
    raise SystemExit(main())
