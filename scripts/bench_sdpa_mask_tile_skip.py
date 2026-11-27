#!/usr/bin/env python3
"""Microbenchmark of stock MLX SDPA prefill on mlx2's real mask shapes.

Written for the ml-explore/mlx#4983 intake (skip fully masked KV tiles in the
fused prefill kernels).  Run it once per MLX build (one venv each) into its
own empty output directory, then compare the two ``results.json`` files with
``--compare``.  Every case builds its mask with mlx2's own mask functions
(``technologies.window_attention`` and the cache ``make_mask`` shapes), so
the shapes match what the routes pass to ``mx.fast.scaled_dot_product_attention``.

Cases (bfloat16):

* ``swa_ring``    sliding window over a RotatingKVCache prefill chunk:
                  qL = chunk, kL = window - 1 + chunk (after the first chunk).
* ``swa_full``    sliding window over a full-length key view (kL = context).
* ``pad_b4``      B=4 left-padded ragged batch, causal + left padding mask
                  ``[4, 1, qL, kL]`` (BatchKVCache with nonzero left padding).
* ``dense_true``  all-true array mask (no empty tile): the scan-cost control.
* ``native_win``  ``mask="causal", window_size=W`` (MLX 0.32.5 native window,
                  no array) for the swa cases, as the no-PR alternative.

Outputs of every case are hashed (and saved, ``--save``) so the two builds
can be checked for bit-exactness.  GPU use: run inside an owned window
(scripts/amoeba_owned.py); no thermal hold, this is not a publishable
performance run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _cases(quick: bool):
    """(name, B, Hq, Hkv, qL, kL, D, kind, extra)."""
    out = []
    contexts = (4096, 8192) if quick else (4096, 8192, 16384, 32768)
    chunk = 2048
    # Gemma 4 26B-A4B sliding layers: Hq 16, Hkv 8, D 256, W 1024.
    for w in (1024,):
        out.append(("swa_ring_g4", 1, 16, 8, chunk, w - 1 + chunk, 256, "swa", {"window": w}))
        out.append(("swa_first_g4", 1, 16, 8, chunk, chunk, 256, "swa", {"window": w}))
        for c in contexts:
            out.append((f"swa_full_g4_{c}", 1, 16, 8, chunk, c, 256, "swa", {"window": w}))
    # gpt-oss-20b sliding layers: Hq 64, Hkv 8, D 64, W 128, sink logits.
    out.append(("swa_ring_gptoss", 1, 64, 8, chunk, 127 + chunk, 64, "swa", {"window": 128, "sinks": True}))
    for c in contexts:
        out.append((f"swa_full_gptoss_{c}", 1, 64, 8, chunk, c, 64, "swa", {"window": 128, "sinks": True}))
    # Padded B4, ragged lengths (1, .75, .5, .25 of kL), last chunk of prefill.
    for (tag, hq, hkv, d) in (("d128", 32, 8, 128), ("g4full_d512", 16, 2, 512)):
        for c in contexts:
            out.append((f"pad_b4_{tag}_{c}", 4, hq, hkv, chunk, c, d, "pad", {}))
    # Built-in causal (no array): the PR says these kernels are untouched.
    for (tag, hq, hkv, d) in (("g4full_d512", 16, 2, 512), ("d256", 16, 8, 256), ("d128", 32, 8, 128)):
        for c in contexts:
            out.append((f"causal_{tag}_{c}", 1, hq, hkv, chunk, c, d, "causal", {}))
    # Control: an array mask with no empty tile.
    for (tag, hq, hkv, d) in (("d128", 32, 8, 128), ("d256", 16, 8, 256)):
        out.append((f"dense_true_{tag}", 1, hq, hkv, chunk, 8192, d, "dense", {}))
    return out


def _mask(mx, wa, kind, B, qL, kL, extra):
    offset = kL - qL
    if kind == "swa":
        return wa.causal_window_mask(qL, offset, extra["window"])
    if kind == "pad":
        # Rows end together; row b has left padding kL * b / 4.
        left = mx.array([kL * b // 4 for b in range(B)])
        return wa.causal_window_mask(qL, offset, None, left_padding=left)
    if kind == "dense":
        return mx.ones((qL, kL), dtype=mx.bool_)
    if kind == "causal":
        return wa.causal_window_mask(qL, offset, None)  # for keep stats only
    raise ValueError(kind)


def _valid_rows(mx, kind, B, qL, kL):
    """Query rows whose outputs a route consumes (pad query rows are dropped)."""
    if kind != "pad":
        return None
    offset = kL - qL
    pos = mx.arange(offset, offset + qL)[None, :]
    left = mx.array([kL * b // 4 for b in range(B)])[:, None]
    return pos >= left  # [B, qL]


def run(args) -> int:
    import mlx.core as mx
    import numpy as np

    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.runtime.technologies import window_attention as wa

    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit("refusing nonempty output directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    info = mx.device_info()
    report = {
        "mlx_version": mx.__version__,
        "mlx_file": mx.__file__,
        "device": info.get("device_name"),
        "nax": bool(mx.metal.is_nax_available()),
        "mlx2_commit": os.environ.get("MLX2_COMMIT", "unknown"),
        "label": args.label,
        "reps": args.reps,
        "rounds": args.rounds,
        "dtype": "bfloat16",
        "cases": [],
    }
    # Warm the GPU clocks a little; this is a sanity A/B, not a thermal run.
    w = mx.random.normal((2048, 2048), dtype=mx.bfloat16)
    for _ in range(10):
        w = w @ w * 1e-4
    mx.eval(w)

    for (name, B, Hq, Hkv, qL, kL, D, kind, extra) in _cases(args.quick):
        if args.only and not any(s in name for s in args.only.split(",")):
            continue
        mx.random.seed(0)
        q = (0.5 * mx.random.normal((B, Hq, qL, D))).astype(mx.bfloat16)
        k = (0.5 * mx.random.normal((B, Hkv, kL, D))).astype(mx.bfloat16)
        v = (0.5 * mx.random.normal((B, Hkv, kL, D))).astype(mx.bfloat16)
        sinks = (mx.random.normal((Hq,)) * 0.5).astype(mx.bfloat16) if extra.get("sinks") else None
        mask = _mask(mx, wa, kind, B, qL, kL, extra)
        mx.eval(q, k, v, mask)
        scale = D ** -0.5
        arms = {"array": dict(mask=mask)} if kind != "causal" else {"causal": dict(mask="causal")}
        if kind == "swa" and args.native:
            arms["native_window"] = dict(mask="causal", window_size=int(extra["window"]))
        rows = _valid_rows(mx, kind, B, qL, kL)
        for arm, kw in arms.items():
            def call(q=q, k=k, v=v):
                return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, sinks=sinks, **kw)
            try:
                o = call()
                mx.eval(o)
            except Exception as exc:  # noqa: BLE001 -- record and keep going
                report["cases"].append({"case": name, "arm": arm, "error": str(exc)})
                continue
            for _ in range(2):
                mx.eval(call())
            mx.synchronize()
            samples = []
            for _ in range(args.rounds):
                t0 = time.perf_counter()
                outs = [call() for _ in range(args.reps)]
                mx.eval(*outs)
                mx.synchronize()
                samples.append((time.perf_counter() - t0) * 1e3 / args.reps)
                del outs
            o32 = o.astype(mx.float32)
            finite = bool(mx.all(mx.isfinite(o32)).item())
            if rows is not None:
                keep = rows[:, None, :, None]
                o_valid = mx.where(keep, o32, mx.zeros_like(o32))
            else:
                o_valid = o32
            mx.eval(o_valid)
            # SDPA may return a transposed layout: hash a C-contiguous copy.
            digest = hashlib.sha256(np.ascontiguousarray(np.array(o.view(mx.uint16))).tobytes()).hexdigest()
            vdigest = hashlib.sha256(np.ascontiguousarray(np.array(o_valid)).tobytes()).hexdigest()
            if args.save:
                mx.save(str(out_dir / f"{name}__{arm}.npy"), o_valid)
            rec = {
                "case": name, "arm": arm, "B": B, "Hq": Hq, "Hkv": Hkv, "qL": qL, "kL": kL,
                "D": D, "kind": kind, **extra,
                "mask_shape": list(mask.shape), "mask_dtype": str(mask.dtype),
                "mask_true_fraction": float(mx.mean(mask.astype(mx.float32)).item()),
                "ms_median": statistics.median(samples), "ms_min": min(samples),
                "ms_samples": samples, "out_sha256": digest, "valid_rows_sha256": vdigest,
                "finite": finite,
            }
            report["cases"].append(rec)
            print(f"{name:28s} {arm:14s} {rec['ms_median']:9.3f} ms  keep={rec['mask_true_fraction']:.3f}",
                  flush=True)
        del q, k, v, mask, call
        mx.clear_cache()
    (out_dir / "results.json").write_text(json.dumps(report, indent=1) + "\n")
    return 0


def compare(paths) -> int:
    """Print base vs candidate per case/arm: speedup and output identity."""
    base, cand = (json.loads(Path(p, "results.json").read_text()) for p in paths)
    index = {(c["case"], c["arm"]): c for c in base["cases"] if "error" not in c}
    print(f"base {base['mlx_version']}  cand {cand['mlx_version']}")
    print(f"{'case':28s} {'arm':14s} {'keep':>6s} {'base ms':>9s} {'cand ms':>9s} {'x':>6s} exact valid_exact")
    for c in cand["cases"]:
        b = index.get((c["case"], c["arm"]))
        if b is None or "error" in c:
            print(f"{c['case']:28s} {c['arm']:14s} {'n/a':>6s} error={c.get('error')}")
            continue
        print(f"{c['case']:28s} {c['arm']:14s} {c['mask_true_fraction']:6.3f} {b['ms_median']:9.3f} "
              f"{c['ms_median']:9.3f} {b['ms_median'] / c['ms_median']:6.2f} "
              f"{b['out_sha256'] == c['out_sha256']!s:5s} {b['valid_rows_sha256'] == c['valid_rows_sha256']}")
    # Within each build: the native window against the dense array.
    for tag, rep in (("base", base), ("cand", cand)):
        arms = {(c["case"], c["arm"]): c for c in rep["cases"] if "error" not in c}
        for (case, arm), c in arms.items():
            if arm != "native_window" or (case, "array") not in arms:
                continue
            a = arms[(case, "array")]
            print(f"{tag} {case:28s} native vs array x{a['ms_median'] / c['ms_median']:6.2f} "
                  f"exact {a['out_sha256'] == c['out_sha256']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--label", default="")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--native", action="store_true", help="also time mask='causal'+window_size")
    ap.add_argument("--save", action="store_true", help="save valid-row outputs as .npy")
    ap.add_argument("--compare", nargs=2, metavar=("BASE_DIR", "CAND_DIR"))
    a = ap.parse_args()
    if a.compare:
        return compare(a.compare)
    if not a.out:
        ap.error("--out is required")
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
