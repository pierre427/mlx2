#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Kernel-only A/B of Kimi MLA flash192 (``kimi_linear_mla_flash.expanded``)
laws v1 and v2 on synthetic BF16 tensors in the Kimi-Linear MLA geometry
(32 heads, qk 192 = 128 NoPE + 64 shared, v 128; head-major kv as served).

Per case (a bottom-right causal prompt chunk of ``rows`` after a prefix of
``context - rows`` keys): median ms of interleaved reps, TFLOP/s over the
visible keys, error against the FP32 ops reference on sampled rows, v2 vs v1
relative difference on the full output, and bitwise chunk invariance (one
launch vs four row slices at their offsets).

Usage: ``bench_kimi_flash192_law_ab.py owned OUT_DIR`` (CPG lease + paired
GPU locks) or ``... unit OUT_DIR``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GEOMETRY_OF = "Kimi-Linear-48B-A3B-Instruct-8bit@553a7a5673491df2b2d38f7380188a9d8e9f9a66"
CASES = [(8192, 2048), (8192, 8192), (32768, 2048), (32768, 8192), (131072, 2048), (131072, 8192)]
REPS = 5
REF_ROWS = 12
LAWS = ("v1", "v2")


def host_sample():
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    m = re.search(r"Swapouts:\s+(\d+)", vm)
    return {"time": time.time(), "swapouts": int(m.group(1)) if m else None}


def source():
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True,  # noqa: E731
                                    text=True).stdout
    diff = git("diff", "HEAD", "--", "src")
    files = ("scripts/bench_kimi_flash192_law_ab.py",
             "src/mlx2/runtime/models/kimi_linear_mla_flash.py")
    return {"commit": git("rev-parse", "HEAD").strip(), "src_dirty": bool(diff),
            "src_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
            "files": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}}


def atomic(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))
    os.replace(tmp, path)


def make_case(context, rows, seed=20261105):
    import mlx.core as mx
    from mlx2.runtime.models import kimi_linear_mla_flash as fl

    k1, k2, k3 = mx.random.split(mx.random.key(seed), 3)
    q = (mx.random.normal((1, rows, fl.HEADS * (fl.NOPE + fl.SHARED)), key=k1) * 0.5).astype(mx.bfloat16)
    kv = mx.random.normal((1, fl.HEADS, context, fl.NOPE + fl.VDIM), key=k2).astype(mx.bfloat16)
    sh = mx.random.normal((1, 1, context, fl.SHARED), key=k3).astype(mx.bfloat16)
    mx.eval(q, kv, sh)
    return q, kv, sh


def run(law, q, kv, sh, *, offset=None):
    from mlx2.runtime.models import kimi_linear_mla_flash as fl

    return fl.expanded(q, kv, sh, "causal", scale=(fl.NOPE + fl.SHARED) ** -0.5, offset=offset,
                       law=law)


def time_fn(fn):
    import mlx.core as mx

    mx.synchronize()
    t0 = time.perf_counter()
    mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) * 1e3


def rel(a, b):
    import mlx.core as mx

    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return float(mx.sqrt(mx.sum((a - b) ** 2)) / mx.maximum(mx.sqrt(mx.sum(b ** 2)), 1e-20))


def ref_rows(q, kv, sh, rows_idx, context, rows):
    """FP32 reference for single rows at their own causal offsets."""
    import mlx.core as mx
    from mlx2.runtime.models import kimi_linear_mla_flash as fl

    outs = []
    natural = context - rows
    for r in rows_idx:
        outs.append(fl.expanded_reference(q[:, r:r + 1], kv, sh, "causal",
                                          scale=(fl.NOPE + fl.SHARED) ** -0.5, offset=natural + r))
    out = mx.concatenate(outs, axis=1)
    mx.eval(out)
    return out


def one_case(context, rows, reps, ref_n, laws, log):
    import mlx.core as mx
    from mlx2.runtime.models import kimi_linear_mla_flash as fl

    q, kv, sh = make_case(context, rows)
    fns = {law: (lambda law=law: run(law, q, kv, sh)) for law in laws}
    for law in laws:
        time_fn(fns[law]); time_fn(fns[law])
    times = {law: [] for law in laws}
    for r in range(reps):
        for law in (laws if r % 2 == 0 else laws[::-1]):
            times[law].append(time_fn(fns[law]))
    prefix = context - rows
    visible = rows * prefix + rows * (rows + 1) // 2
    flops = 2.0 * fl.HEADS * visible * ((fl.NOPE + fl.SHARED) + fl.VDIM)
    step = max(1, rows // ref_n)
    idx = list(range(0, rows, step))[:ref_n - 1] + [rows - 1]
    ref = ref_rows(q, kv, sh, idx, context, rows)
    row = {"context": context, "rows": rows, "visible_scores": visible, "gflop": flops / 1e9,
           "arms": {}}
    full = {}
    for law in laws:
        med = statistics.median(times[law])
        out = fns[law]()
        mx.eval(out)
        full[law] = out
        got = mx.concatenate([out[:, i:i + 1] for i in idx], axis=1)
        # Chunk invariance: four row slices at their own offsets.
        part = rows // 4
        pieces = [run(law, q[:, s:s + part], kv, sh, offset=prefix + s) for s in range(0, rows, part)]
        whole = mx.concatenate(pieces, axis=1)
        row["arms"][law] = {
            "ms": med, "ms_all": times[law], "tflops": flops / (med * 1e-3) / 1e12,
            "rel_l2_vs_fp32_ref": rel(got, ref),
            "finite": bool(mx.all(mx.isfinite(out.astype(mx.float32))).item()),
            "chunk_invariant_bitwise": bool(mx.array_equal(whole, out).item()),
        }
    if "v1" in full and "v2" in full:
        row["v2_vs_v1"] = {"rel_l2": rel(full["v2"], full["v1"]),
                           "bitwise_equal": bool(mx.array_equal(full["v2"], full["v1"]).item())}
        row["speedup_v2"] = row["arms"]["v1"]["ms"] / row["arms"]["v2"]["ms"]
    log(json.dumps({"context": context, "rows": rows,
                    **{law: {"ms": round(e["ms"], 3), "tflops": round(e["tflops"], 2),
                             "err": f"{e['rel_l2_vs_fp32_ref']:.2e}",
                             "chunk_inv": e["chunk_invariant_bitwise"]}
                       for law, e in row["arms"].items()},
                    "speedup_v2": round(row.get("speedup_v2", 0), 3)}))
    # Rebind instead of ``del``: the timing lambdas in ``fns`` close over these.
    fns = q = kv = sh = full = None
    mx.clear_cache()
    return row


def smoke(log):
    """The module's own probe shape for each law (and a token-major kv)."""
    import mlx.core as mx
    from mlx2.runtime.models import kimi_linear_mla_flash as fl

    ok, detail = {}, {}
    k1, k2, k3 = mx.random.split(mx.random.key(20261102), 3)
    rows, width = 40, 100
    q = (mx.random.normal((1, rows, fl.HEADS * (fl.NOPE + fl.SHARED)), key=k1) * 0.5).astype(mx.bfloat16)
    kv = mx.random.normal((1, width, fl.HEADS * (fl.NOPE + fl.VDIM)), key=k2).astype(mx.bfloat16)
    sh = mx.random.normal((1, 1, width, fl.SHARED), key=k3).astype(mx.bfloat16)
    kvh = mx.contiguous(kv.reshape(1, width, fl.HEADS, fl.NOPE + fl.VDIM).transpose(0, 2, 1, 3))
    scale = (fl.NOPE + fl.SHARED) ** -0.5
    ref = fl.expanded_reference(q, kv, sh, "causal", scale=scale)
    for law in LAWS:
        try:
            a = fl.expanded(q, kv, sh, "causal", scale=scale, law=law)
            b = fl.expanded(q, kvh, sh, "causal", scale=scale, law=law)
            mx.eval(a, b)
            e = {"token_major": rel(a, ref), "head_major": rel(b, ref)}
            ok[law] = max(e.values()) < 2e-2
        except Exception as exc:  # noqa: BLE001 -- recorded; the law is skipped
            e, ok[law] = {"error": repr(exc)[:500]}, False
        detail[law] = e
    log("smoke " + json.dumps(detail))
    return ok, detail


def unit(out_dir: Path, quick: bool):
    import mlx.core as mx

    out_dir.mkdir(parents=True, exist_ok=True)
    mx.set_cache_limit(4 << 30)
    log_path = out_dir / "bench.log"

    def log(msg):
        with log_path.open("a") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        print(msg, flush=True)

    report = {"source": source(), "device": mx.device_info().get("device_name"),
              "mlx": mx.__version__, "geometry_of": GEOMETRY_OF,
              "scope": "kernel-only research microbench on synthetic BF16 tensors; not "
              "qualification, no model checkpoint loaded", "before": host_sample(), "rows": []}
    ok, detail = smoke(log)
    report["smoke"] = {"ok": ok, "detail": detail}
    laws = tuple(law for law in LAWS if ok.get(law))
    if "v1" not in laws:
        report["aborted"] = "v1 failed the smoke gate"
        atomic(out_dir / "report.json", report)
        return 1
    for context, rows in (CASES[2:3] if quick else CASES):
        before = host_sample()["swapouts"]
        report["rows"].append(one_case(context, rows, 2 if quick else REPS,
                                       4 if quick else REF_ROWS, laws, log))
        after = host_sample()["swapouts"]
        atomic(out_dir / "report.partial.json", report)
        if before is not None and after is not None and after > before:
            report["aborted"] = f"swapouts rose {before} -> {after}"
            break
    report["after"] = host_sample()
    atomic(out_dir / "report.json", report)
    return 0


def owned(out_dir: Path, job_id: str, quick: bool):
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    with DesktopCPGLease(out_dir, job_id, checkpoint="synthetic-kimi-flash192",
                         source_revision=source()["commit"]) as lease:
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                   "--session", lease.owner["cpg_session"], "--label", job_id,
                   "--desktop-cpg-lease", str(lease.path),
                   "--receipt", str(out_dir / "paired-locks.json"), "--",
                   sys.executable, "-u", str(Path(__file__)), "unit", str(out_dir)]
        if quick:
            command.append("--quick")
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        child = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
        try:
            while child.poll() is None:
                lease.check()
                time.sleep(1)
            lease.success = child.returncode == 0
            return child.returncode
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("owned", "unit"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--job-id", default="kimi-flash192-law-ab-20261105")
    args = parser.parse_args()
    if args.mode == "unit":
        return unit(args.out, args.quick)
    return owned(args.out, args.job_id, args.quick)


if __name__ == "__main__":
    raise SystemExit(main())
