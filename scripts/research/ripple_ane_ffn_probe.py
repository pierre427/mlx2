#!/usr/bin/env python3
"""Ripple-65B FFN: GPU-alone versus GPU || ANE token split, one layer, synthetic weights.

Feasibility probe, not a serving route and not qualification. Three sub-commands:

``prepare`` (mlx2 venv, MLX on CPU)
    Draws seeded synthetic weights with Ripple's exact FFN geometry
    (hidden 5120, intermediate 40960), quantizes them with MLX affine
    quantization (oQ4e: 4-bit, group 64; down_proj optionally 5/6-bit as in the
    real checkpoint), and writes the packed MLX tensors, the dequantized FP16
    weights the Core ML build consumes, and a BF16 activation block.

``run`` (mlx2 venv)
    Times, for each prefill chunk of R rows and each ANE share s:
      (i)   MLX alone running the whole FFN on R rows (ricursive.MLP, quantized);
      (ii)  GPU || ANE: the first k ~= s*R rows go to a Core ML package in a
            separate worker process while MLX runs the remaining R-k rows.
            Every hand-off cost is inside the timed span: BF16->FP16 conversion,
            device->host copy into shared memory, Core ML input/output
            conversion in the worker, waiting, FP16->BF16 conversion and the
            row merge back into one MLX array;
      (iii) accuracy of the merged output against the MLX-alone output.
    It also times the ANE slice alone (with hand-off) and the MLX remainder
    alone, so the overlap can be judged against its ideal.
    Core ML packages are loaded one at a time; every load, compute-plan and
    prediction is bounded by a timeout, after which the worker process group
    is killed. One JSON receipt is written atomically; an existing receipt is
    never overwritten.

    Real runs need ``--owned-window`` (the caller asserts it holds the GPU
    ownership window and accepts the ANE hazard). ``--dry-run`` instead forces
    MLX onto the CPU and Core ML onto CPU_ONLY and exercises the same code.

``coreml-worker`` (mlx-uag venv; internal)
    Loads one package, reports its compute plan, answers predictions over a
    line protocol with tensors in POSIX shared memory. Never imports MLX.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import select
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCHEMA = "mlx2.ripple-ane-ffn-probe.v1"
HIDDEN = 5120
INTER = 40960
CLAMP = 20.0
GROUP = 64
REPO = Path(__file__).resolve().parents[2]
COREML_PY = "~/Desktop/mlx-uag/.venv/bin/python"
DEFAULT_SCRATCH = "/private/tmp/claude-501/ripple-ane"
TAG = "@@"  # worker protocol lines; anything else on stdout is library noise


# --------------------------------------------------------------------------- common


def _sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def _atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False, default=str)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.link(tmp, path)  # fails if the receipt appeared meanwhile: never overwrite
    tmp.unlink()


def _git(*args):
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              timeout=20, check=False).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc}"


def _host():
    def sysctl(k):
        try:
            return subprocess.run(["sysctl", "-n", k], capture_output=True, text=True, timeout=5, check=False).stdout.strip()
        except Exception:  # noqa: BLE001
            return None
    return {"chip": sysctl("machdep.cpu.brand_string"), "memsize": sysctl("hw.memsize"),
            "macos": platform.mac_ver()[0], "swapusage": sysctl("vm.swapusage")}


def _vm_free_bytes():
    """Free + inactive + speculative pages (system-wide; ANE buffers live outside our RSS)."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5, check=False).stdout
    except Exception:  # noqa: BLE001
        return None
    page = 16384
    vals = {}
    for line in out.splitlines():
        if "page size of" in line:
            page = int(line.split("page size of")[1].split()[0])
        elif ":" in line:
            k, v = line.split(":", 1)
            try:
                vals[k.strip()] = int(v.strip().rstrip("."))
            except ValueError:
                pass
    keys = ("Pages free", "Pages inactive", "Pages speculative")
    return sum(vals.get(k, 0) for k in keys) * page


def _round_rows(n, quantum=64):
    return max(quantum, round(n / quantum) * quantum)


def _stats(xs):
    xs = list(xs)
    if not xs:
        return None
    return {"n": len(xs), "median_ms": statistics.median(xs), "min_ms": min(xs), "max_ms": max(xs),
            "all_ms": xs}


def _compare(a, b):
    """a, b: float32 arrays of equal shape. Returns error and direction metrics."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    d = np.abs(a - b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return {"finite": finite, "max_abs": float(np.nanmax(d)), "mean_abs": float(np.nanmean(d)),
            "ref_max_abs": float(np.nanmax(np.abs(b))),
            "rel_l2": float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)),
            "cosine": float(a @ b / denom) if denom else None}


def _row_cosine_min(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return float(np.min(num / np.maximum(den, 1e-30)))


# --------------------------------------------------------------------------- prepare (MLX)


def _mlx(cpu):
    import mlx.core as mx
    if cpu:
        mx.set_default_device(mx.cpu)
    return mx


def cmd_prepare(args):
    cpu_only = os.environ.get("MLX2_PREFLIGHT_CPU_ONLY") == "1" or not args.allow_gpu
    mx = _mlx(cpu_only)
    out = Path(args.scratch) / "weights"
    if (out / "weights.json").exists():
        raise SystemExit(f"refusing to overwrite prepared weights in {out}")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    meta = {"schema": SCHEMA + ".weights", "seed": args.seed, "hidden": HIDDEN, "intermediate": INTER,
            "group_size": GROUP, "bits": {"gate_proj": 4, "up_proj": 4, "down_proj": args.down_bits},
            "mode": "affine", "device": "cpu" if cpu_only else "gpu",
            "synthetic": "normal(0, 0.02) * per-output-row lognormal(0, 0.25), cast BF16, then MLX affine quantize",
            "tensors": {}}
    tensors = {}
    for name, shape, bits in (("gate_proj", (INTER, HIDDEN), 4), ("up_proj", (INTER, HIDDEN), 4),
                              ("down_proj", (HIDDEN, INTER), args.down_bits)):
        w = rng.standard_normal(shape, dtype=np.float32) * 0.02
        w *= np.exp(rng.normal(0.0, 0.25, size=(shape[0], 1))).astype(np.float32)
        wb = mx.array(w).astype(mx.bfloat16)
        del w
        q, s, b = mx.quantize(wb, group_size=GROUP, bits=bits)
        deq = mx.dequantize(q, s, b, group_size=GROUP, bits=bits)  # BF16, what the GPU multiplies by
        mx.eval(q, s, b, deq)
        tensors[f"{name}.weight"] = q
        tensors[f"{name}.scales"] = s
        tensors[f"{name}.biases"] = b
        deq16 = np.array(deq.astype(mx.float32)).astype(np.float16)  # BF16 -> FP16 for Core ML
        np.save(out / f"{name}.deq_fp16.npy", deq16)
        meta["tensors"][name] = {"shape": list(shape), "bits": bits,
                                 "fp16_nonfinite": int((~np.isfinite(deq16)).sum()),
                                 "max_abs": float(np.abs(deq16).max())}
        del deq16, deq, wb
    mx.save_safetensors(str(out / "ffn_q.safetensors"), tensors)
    # Activations: post-norm-like BF16 rows; 2% of rows scaled x16 so the gate clamp
    # (max 20) and the up clip ([-20, 20]) are actually exercised.
    x = rng.standard_normal((args.max_rows, HIDDEN), dtype=np.float32)
    hot = rng.choice(args.max_rows, size=max(1, args.max_rows // 50), replace=False)
    x[hot] *= 16.0
    xb = mx.array(x).astype(mx.bfloat16)
    mx.save_safetensors(str(out / "x.safetensors"), {"x": xb})
    np.save(out / "x_bf16_as_fp32.npy", np.array(xb.astype(mx.float32)))
    meta["x"] = {"rows": args.max_rows, "hot_rows": sorted(int(i) for i in hot), "hot_scale": 16.0}
    meta["files"] = {p.name: _sha256_file(p) for p in sorted(out.iterdir()) if p.is_file()}
    meta["fingerprint"] = hashlib.sha256(json.dumps(meta["files"], sort_keys=True).encode()).hexdigest()
    meta["checkpoint"] = f"synthetic-ripple65b-ffn-layer@sha256:{meta['fingerprint']}"
    _atomic_json(out / "weights.json", meta)
    print(json.dumps({"checkpoint": meta["checkpoint"], "dir": str(out)}))


def _fp32_reference(wdir, xb, rows):
    """NumPy FP32 Ripple MLP on the GPU's dequantized weights (both arms are judged against it)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ripple_ane_ffn_build import (
        reference,  # numpy only; coremltools is imported lazily there
    )

    x = np.load(wdir / "x_bf16_as_fp32.npy")[:rows]
    w = {n: np.load(wdir / f"{n}.deq_fp16.npy").astype(np.float32) for n in ("gate_proj", "up_proj", "down_proj")}
    out = reference(x, w["gate_proj"], w["up_proj"], w["down_proj"])
    del w
    return out


def _build_mlp(mx, wdir, fused):
    from mlx import nn
    sys.path.insert(0, str(REPO / "src"))
    from mlx2.runtime.models import ricursive  # the production MLP / _swiglu

    meta = json.loads((wdir / "weights.json").read_text())
    args = ricursive.ModelArgs()
    assert args.hidden_size == HIDDEN and args.intermediate_size == INTER and args.ffn_clamp_threshold == CLAMP
    mlp = ricursive.MLP(args, nvfp4=False)
    for name in ("gate_proj", "up_proj", "down_proj"):
        lin = getattr(mlp, name)
        out_dim, in_dim = lin.weight.shape
        bits = meta["bits"][name]
        setattr(mlp, name, nn.QuantizedLinear(in_dim, out_dim, bias=False, group_size=GROUP, bits=bits))
    mlp.load_weights(str(wdir / "ffn_q.safetensors"), strict=True)
    mlp.fused = bool(fused)
    mx.eval(mlp.parameters())
    xb = mx.load(str(wdir / "x.safetensors"))["x"]
    return mlp, xb, meta


# --------------------------------------------------------------------------- Core ML worker client


class WorkerError(RuntimeError):
    pass


class CoreMLWorker:
    """One Core ML package in its own process group; every call bounded."""

    def __init__(self, package, compute_units, max_rows, python, scratch):
        from multiprocessing import shared_memory
        nbytes = max_rows * HIDDEN * 2
        self.shm_in = shared_memory.SharedMemory(create=True, size=nbytes)
        self.shm_out = shared_memory.SharedMemory(create=True, size=nbytes)
        self.buf_in = np.ndarray((max_rows, HIDDEN), np.float16, self.shm_in.buf)
        self.buf_out = np.ndarray((max_rows, HIDDEN), np.float16, self.shm_out.buf)
        self.log = open(Path(scratch) / f"worker-{os.getpid()}-{int(time.time())}.stderr", "w")
        self.proc = subprocess.Popen(
            [python, "-u", str(Path(__file__).resolve()), "coreml-worker", "--package", str(package),
             "--compute-units", compute_units, "--shm-in", self.shm_in.name, "--shm-out", self.shm_out.name,
             "--max-rows", str(max_rows)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, start_new_session=True,
            cwd=str(scratch))
        self._fd = self.proc.stdout.fileno()
        self._pending = b""
        self.predictions = 0
        self.log_path = self.log.name

    def _read(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            while b"\n" in self._pending:
                line, self._pending = self._pending.split(b"\n", 1)
                text = line.decode(errors="replace")
                if text.startswith(TAG):
                    msg = json.loads(text[len(TAG):])
                    if msg.get("error"):
                        raise WorkerError(msg["error"])
                    return msg
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"Core ML worker silent for {timeout:.0f}s")
            ready, _, _ = select.select([self._fd], [], [], left)
            if ready:
                chunk = os.read(self._fd, 65536)
                if not chunk:
                    raise WorkerError(f"Core ML worker exited (rc={self.proc.poll()}); see {self.log_path}")
                self._pending += chunk

    def send(self, msg):
        self.proc.stdin.write((json.dumps(msg) + "\n").encode())
        self.proc.stdin.flush()

    def wait_ready(self, timeout):
        return self._read(timeout)

    def request(self, msg, timeout):
        self.send(msg)
        return self._read(timeout)

    def start_predict(self, rows):
        self.send({"op": "predict", "rows": int(rows)})

    def finish_predict(self, timeout):
        reply = self._read(timeout)
        self.predictions += 1
        return reply

    def close(self):
        try:
            if self.proc.poll() is None:
                try:
                    self.send({"op": "exit"})
                    self.proc.wait(timeout=10)
                except Exception:  # noqa: BLE001
                    pass
            for sig, wait in ((signal.SIGTERM, 3), (signal.SIGKILL, 5)):
                if self.proc.poll() is not None:
                    break
                try:
                    os.killpg(self.proc.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    self.proc.wait(timeout=wait)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            self.log.close()
            for shm in (self.shm_in, self.shm_out):
                try:
                    shm.close()
                    shm.unlink()
                except Exception:  # noqa: BLE001
                    pass


# --------------------------------------------------------------------------- run


def _time_gpu(mx, mlp, x):
    t0 = time.perf_counter()
    y = mlp(x)
    mx.eval(y)
    return (time.perf_counter() - t0) * 1e3, y


def _time_split(mx, mlp, x, k, worker, timeout, *, ane_only=False):
    """One GPU || ANE pass over x with the first k rows on the ANE. All hand-off inside the span."""
    t0 = time.perf_counter()
    xa = x[:k].astype(mx.float16)                           # BF16 -> FP16 (on the MLX device)
    yg = None if ane_only else mlp(x[k:])
    mx.eval(xa)
    if yg is not None:
        mx.async_eval(yg)                                    # GPU remainder starts now
    t_conv = time.perf_counter()
    worker.buf_in[:k] = np.asarray(xa)             # device -> host into the Core ML input
    t_copy = time.perf_counter()
    worker.start_predict(k)
    if yg is not None:
        mx.eval(yg)
    t_gpu = time.perf_counter()
    reply = worker.finish_predict(timeout)                   # waiting
    t_wait = time.perf_counter()
    ya = mx.array(worker.buf_out[:k]).astype(mx.bfloat16)   # host -> MLX, FP16 -> BF16
    y = ya if yg is None else mx.concatenate([ya, yg], axis=0)
    mx.eval(y)
    t1 = time.perf_counter()
    parts = {"convert_ms": (t_conv - t0) * 1e3, "host_copy_ms": (t_copy - t_conv) * 1e3,
             "gpu_remainder_return_ms": (t_gpu - t_copy) * 1e3, "ane_wait_after_gpu_ms": (t_wait - t_gpu) * 1e3,
             "merge_ms": (t1 - t_wait) * 1e3, "worker": reply}
    return (t1 - t0) * 1e3, y, parts


def cmd_run(args):
    if not args.dry_run and not args.owned_window:
        raise SystemExit("real runs need --owned-window (GPU ownership + ANE hazard accepted); "
                         "use --dry-run for the CPU-only exercise")
    if not args.dry_run and os.environ.get("MLX2_PREFLIGHT_CPU_ONLY") == "1":
        raise SystemExit("MLX2_PREFLIGHT_CPU_ONLY=1 is set: refusing a real (GPU/ANE) run")
    out = Path(args.out)
    if out.exists() or out.with_name(out.name + ".tmp").exists():
        raise SystemExit(f"refusing to overwrite receipt {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(args.scratch)
    wdir = scratch / "weights"
    manifest = json.loads(Path(args.manifest).read_text())
    mx = _mlx(args.dry_run)
    compute_units = "CPU_ONLY" if args.dry_run else args.compute_units

    mlp, xb, wmeta = _build_mlp(mx, wdir, args.fused)
    ref32_all = _fp32_reference(wdir, xb, max(int(r) for r in args.rows.split(",")))
    if manifest.get("weights_checkpoint") != wmeta["checkpoint"]:
        raise SystemExit("package manifest was built from different weights than the prepared set")
    rows_list = [int(r) for r in args.rows.split(",")]
    shares = [float(s) for s in args.shares.split(",")]
    if max(rows_list) > xb.shape[0]:
        raise SystemExit("prepared activation block is shorter than the largest chunk")

    receipt = {
        "schema": SCHEMA, "kind": "dry_run_cpu_only" if args.dry_run else "owned_window_gpu_ane",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mlx2_commit": _git("rev-parse", "HEAD"), "mlx2_dirty": bool(_git("status", "--porcelain", "--", "scripts")),
        "checkpoint": wmeta["checkpoint"], "weights_bits": wmeta["bits"], "host": _host(),
        "mlx": {"version": mx.__version__, "device": str(mx.default_device()), "fused_swiglu": bool(args.fused),
                "route": "mlx2.runtime.models.ricursive.MLP with nn.QuantizedLinear (affine, group 64)"},
        "coreml_compute_units": compute_units, "rows": rows_list, "shares": shares,
        "repeats": args.repeats, "warmup": args.warmup,
        "dry_run_caveat": ("MLX's CPU bf16 quantized matmul accumulates in reduced precision (one projection is "
                           "~0.997 cosine to FP32), so in a dry run 'gpu_alone' is not a precise reference; "
                           "judge both arms against *_vs_fp32_ref") if args.dry_run else None,
        "claims": {"implemented": True, "qualified": False, "selected": False, "observed_used": False,
                   "note": "one synthetic layer; placement is MLComputePlan anticipation, not an execution trace"},
        "formats": {}, "errors": [],
    }
    receipt["vm_free_bytes_start"] = _vm_free_bytes()

    wanted = set(args.formats.split(",")) if args.formats else None
    packages = []
    for p in manifest["packages"]:
        if wanted is not None and p["format"] not in wanted and p["label"] not in wanted:
            continue
        if p.get("status") != "built" or (p.get("cpu_verify_passed") is False and not args.include_failed_verify):
            receipt["errors"].append(f"skipped {p['label']}: status={p.get('status')} "
                                     f"cpu_verify_passed={p.get('cpu_verify_passed')}")
            continue
        packages.append(p)
    try:
        for pkg in packages:  # one Core ML package resident at a time
            receipt["formats"][pkg["label"]] = _run_format(mx, mlp, xb, ref32_all, pkg, rows_list, shares,
                                                           args, compute_units, scratch)
    except KeyboardInterrupt:
        receipt["errors"].append("interrupted")
    finally:
        receipt["vm_free_bytes_end"] = _vm_free_bytes()
        receipt["host_end"] = _host()
        receipt["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _atomic_json(out, receipt)
        print(json.dumps({"receipt": str(out)}))


def _run_format(mx, mlp, xb, ref32_all, pkg, rows_list, shares, args, compute_units, scratch):
    res = {"package": pkg, "cells": {}, "ane_alone": {}, "errors": []}
    shapes = sorted(pkg["row_shapes"])
    max_rows = max(shapes)
    vm0 = _vm_free_bytes()
    t0 = time.perf_counter()
    worker = CoreMLWorker(pkg["path"], compute_units, max_rows, args.coreml_python, scratch)
    try:
        ready = worker.wait_ready(args.load_timeout)
        res["load"] = {"wall_s": time.perf_counter() - t0, "worker": ready,
                       "vm_free_delta_bytes": (None if vm0 is None else vm0 - (_vm_free_bytes() or 0))}
        plan = worker.request({"op": "plan"}, args.load_timeout)
        res["compute_plan"] = plan
        if not args.dry_run and not args.allow_off_ane:
            heavy = plan.get("heavy_ops_by_device", {})
            off = {d: n for d, n in heavy.items() if d != "MLNeuralEngineComputeDevice"}
            if off or not heavy:
                res["errors"].append(f"refused: matmul ops not all ANE-preferred {heavy}; "
                                     "timing would measure a CPU fallback, not the ANE")
                return res
        for rows in rows_list:
            x = xb[:rows]
            mx.eval(x)
            ks = {}
            for s in shares:
                want = _round_rows(rows * s)
                k = min(shapes, key=lambda v: (abs(v - want), v))
                if 0 < k < rows:
                    ks[s] = k
            cell = {"rows": rows, "gpu_alone": [], "splits": {}}
            # warmup (also the first ANE call per shape, which may compile/specialize)
            for _ in range(args.warmup):
                _time_gpu(mx, mlp, x)
                for k in sorted(set(ks.values())):
                    _time_split(mx, mlp, x, k, worker, args.predict_timeout)
            ref_y = None
            split_runs = {s: [] for s in ks}
            parts_last = {}
            ys = {}
            for rep in range(args.repeats):
                order = list(ks.items())
                if rep % 2:
                    order.reverse()
                gpu_first = rep % 2 == 0
                if gpu_first:
                    ms, ref_y = _time_gpu(mx, mlp, x)
                    cell["gpu_alone"].append(ms)
                for s, k in order:
                    ms, y, parts = _time_split(mx, mlp, x, k, worker, args.predict_timeout)
                    split_runs[s].append(ms)
                    parts_last[s] = parts
                    ys[s] = y
                if not gpu_first:
                    ms, ref_y = _time_gpu(mx, mlp, x)
                    cell["gpu_alone"].append(ms)
            ref = np.array(ref_y.astype(mx.float32))
            ref32 = ref32_all[:rows]
            cell["gpu_alone_vs_fp32_ref"] = _compare(ref, ref32)
            gpu_med = statistics.median(cell["gpu_alone"])
            cell["gpu_alone"] = _stats(cell["gpu_alone"])
            for s, k in ks.items():
                merged = np.array(ys[s].astype(mx.float32))
                # MLX on the GPU remainder alone (same rows), for the ideal-overlap bound
                rem = [_time_gpu(mx, mlp, x[k:])[0] for _ in range(max(1, args.repeats // 2))]
                med = statistics.median(split_runs[s])
                cell["splits"][str(s)] = {
                    "share_requested": s, "ane_rows": k, "share_actual": k / rows,
                    "split": _stats(split_runs[s]), "gpu_remainder_alone": _stats(rem),
                    "split_over_gpu_alone": med / gpu_med,
                    "last_pass_breakdown": parts_last[s],
                    "accuracy_merged_vs_gpu_alone": _compare(merged, ref),
                    "accuracy_ane_rows_vs_gpu_alone": _compare(merged[:k], ref[:k]),
                    "accuracy_gpu_rows_identical": bool(np.array_equal(merged[k:], ref[k:])),
                    "ane_rows_min_row_cosine": _row_cosine_min(merged[:k], ref[:k]),
                    "ane_rows_vs_fp32_ref": _compare(merged[:k], ref32[:k]),
                    "gpu_alone_same_rows_vs_fp32_ref": _compare(ref[:k], ref32[:k]),
                }
            res["cells"][str(rows)] = cell
        # ANE slice alone, with the same hand-off, as the capacity curve
        for k in shapes:
            if k > max(rows_list):
                continue
            x = xb[:k]
            mx.eval(x)
            times = []
            for _ in range(max(1, args.repeats)):
                ms, _, parts = _time_split(mx, mlp, x, k, worker, args.predict_timeout, ane_only=True)
                times.append(ms)
            res["ane_alone"][str(k)] = {"with_handoff": _stats(times), "last_breakdown": parts}
    except (TimeoutError, WorkerError) as exc:
        res["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        res["ane_predictions_completed"] = worker.predictions
        worker.close()
        res["worker_stderr"] = worker.log_path
    if res["cells"] and worker.predictions == 0:
        res["errors"].append("gate: split cells recorded with zero completed ANE predictions")
    return res


# --------------------------------------------------------------------------- Core ML worker (mlx-uag venv)


def cmd_worker(args):
    from multiprocessing import resource_tracker, shared_memory

    def say(msg):
        sys.stdout.write(TAG + json.dumps(msg, default=str) + "\n")
        sys.stdout.flush()

    try:
        import coremltools as ct
        shm_in = shared_memory.SharedMemory(name=args.shm_in)
        shm_out = shared_memory.SharedMemory(name=args.shm_out)
        for shm in (shm_in, shm_out):  # the parent owns and unlinks them
            try:
                resource_tracker.unregister(shm._name, "shared_memory")
            except Exception:  # noqa: BLE001
                pass
        buf_in = np.ndarray((args.max_rows, HIDDEN), np.float16, shm_in.buf)
        buf_out = np.ndarray((args.max_rows, HIDDEN), np.float16, shm_out.buf)
        cu = getattr(ct.ComputeUnit, args.compute_units)
        t0 = time.perf_counter()
        model = ct.models.MLModel(args.package, compute_units=cu)
        load_s = time.perf_counter() - t0
        import resource
        say({"ready": True, "coremltools": ct.__version__, "load_s": load_s, "compute_units": args.compute_units,
             "maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    except Exception as exc:  # noqa: BLE001
        say({"error": f"load: {type(exc).__name__}: {exc}"})
        return 1
    for line in sys.stdin:
        msg = json.loads(line)
        op = msg.get("op")
        try:
            if op == "exit":
                break
            if op == "plan":
                say(_compute_plan(ct, model, cu))
            elif op == "predict":
                k = msg["rows"]
                t0 = time.perf_counter()
                outd = model.predict({"x": buf_in[:k]})
                t1 = time.perf_counter()
                y = outd["y"]
                buf_out[:k] = y if y.dtype == np.float16 else y.astype(np.float16)
                t2 = time.perf_counter()
                say({"rows": k, "predict_ms": (t1 - t0) * 1e3, "out_copy_ms": (t2 - t1) * 1e3,
                     "out_dtype": str(y.dtype), "out_finite": bool(np.isfinite(y).all())})
            else:
                say({"error": f"unknown op {op}"})
        except Exception as exc:  # noqa: BLE001
            say({"error": f"{op}: {type(exc).__name__}: {exc}"})
    for shm in (shm_in, shm_out):
        shm.close()
    return 0


def _compute_plan(ct, model, cu):
    """Anticipated per-op placement (MLComputePlan). Not an execution trace."""
    try:
        from coremltools.models.compute_plan import MLComputePlan
        compiled = model.get_compiled_model_path()
        plan = MLComputePlan.load_from_path(path=compiled, compute_units=cu)
        program = plan.model_structure.program
        by_dev, heavy = {}, {}
        ops = []
        for fname, fn in program.functions.items():
            for op in fn.block.operations:
                base = op.operator_name.split(".")[-1]  # e.g. "ios18.linear" -> "linear"
                if base.startswith("const"):
                    continue
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                dev = type(usage.preferred_compute_device).__name__ if usage else "none"
                by_dev[dev] = by_dev.get(dev, 0) + 1
                if base in ("linear", "matmul", "conv"):
                    heavy[dev] = heavy.get(dev, 0) + 1
                ops.append([fname, op.operator_name, dev])
        return {"ops_by_device": by_dev, "heavy_ops_by_device": heavy, "ops": ops[:200]}
    except Exception as exc:  # noqa: BLE001
        return {"plan_error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- CLI


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="synthetic Ripple FFN weights + activations (mlx2 venv)")
    p.add_argument("--scratch", default=DEFAULT_SCRATCH)
    p.add_argument("--seed", type=int, default=20261105)
    p.add_argument("--max-rows", type=int, default=2048)
    p.add_argument("--down-bits", type=int, default=4, choices=(4, 5, 6),
                   help="oQ4e uses 4-bit down_proj on 44 layers, 5-bit on 16, 6-bit on 20")
    p.add_argument("--allow-gpu", action="store_true", help="quantize on the default device (default: CPU)")
    r = sub.add_parser("run", help="time GPU-alone vs GPU||ANE and write one receipt (mlx2 venv)")
    r.add_argument("--scratch", default=DEFAULT_SCRATCH)
    r.add_argument("--manifest", default=f"{DEFAULT_SCRATCH}/packages/manifest.json")
    r.add_argument("--out", required=True)
    r.add_argument("--formats", default="", help="comma list of formats or package labels (default: every verified package)")
    r.add_argument("--include-failed-verify", action="store_true",
                   help="also time packages whose CPU_ONLY verification failed (nondeterministic or bad rows)")
    r.add_argument("--rows", default="512,1024,2048")
    r.add_argument("--shares", default="0.10,0.17,0.25,0.35,0.47")
    r.add_argument("--repeats", type=int, default=5)
    r.add_argument("--warmup", type=int, default=2)
    r.add_argument("--fused", action="store_true", help="ricursive fused SwiGLU region (precise sigmoid)")
    r.add_argument("--compute-units", default="CPU_AND_NE", choices=("CPU_AND_NE", "ALL", "CPU_ONLY"))
    r.add_argument("--coreml-python", default=COREML_PY)
    r.add_argument("--load-timeout", type=float, default=900.0)
    r.add_argument("--predict-timeout", type=float, default=60.0)
    r.add_argument("--allow-off-ane", action="store_true", help="time even if matmuls are not ANE-preferred")
    r.add_argument("--dry-run", action="store_true", help="MLX on CPU, Core ML CPU_ONLY")
    r.add_argument("--owned-window", action="store_true", help="caller holds the GPU window; ANE hazard accepted")
    w = sub.add_parser("coreml-worker")
    w.add_argument("--package", required=True)
    w.add_argument("--compute-units", required=True)
    w.add_argument("--shm-in", required=True)
    w.add_argument("--shm-out", required=True)
    w.add_argument("--max-rows", type=int, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "prepare":
        return cmd_prepare(args)
    if args.cmd == "run":
        if args.dry_run:
            if args.repeats == 5:
                args.repeats = 1
            if args.warmup == 2:
                args.warmup = 0
            # MLX's CPU quantized matmul is single-threaded scalar code (~6+ min for one
            # 512-row Ripple FFN), so the CPU exercise uses one small chunk by default.
            if args.rows == "512,1024,2048":
                args.rows = "128"
            if args.shares == "0.10,0.17,0.25,0.35,0.47":
                args.shares = "0.5"
        return cmd_run(args)
    return cmd_worker(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
