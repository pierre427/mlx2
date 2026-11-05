#!/usr/bin/env python3
"""Build Core ML ML Program packages for one Ripple-65B FFN layer (ANE token split).

Run with the mlx-uag venv (coremltools 9.0); it never imports MLX. Inputs are the
dequantized FP16 weights written by ``ripple_ane_ffn_probe.py prepare`` (the
values the GPU's oQ4e affine weights decode to, cast BF16 -> FP16).

Each package is the whole SwiGLU MLP for a slice of rows, as Ripple computes it
(src/mlx2/runtime/models/ricursive.py, MLP / _swiglu):

    gate = min(x @ Wg^T, 20); up = clip(x @ Wu^T, -20, 20)
    y    = (silu(gate) * up) @ Wd^T

Differences from the GPU route, stated up front:
  * activations and intermediates are FP16, not BF16 (more mantissa, far less
    range: a pre-clamp gate or a down-projection output above 65504 overflows);
  * SiLU runs in FP16 by default (Ripple runs it in FP32 and casts back);
    ``--silu-fp32`` reproduces that but an FP32 cast is not ANE-resident;
  * weights are the GPU's dequantized values in FP16, or a compressed
    re-encoding of them (int8 per-channel, or grouped-channel LUT8/LUT4), which
    cannot reproduce group-64 affine Q4 exactly. Affine uint4 blockwise
    (constexpr_blockwise_shift_scale) is deliberately not offered: the lab found
    it compiles but makes every matmul CPU-preferred.

Shapes: one package per format with EnumeratedShapes over the row counts the
probe needs (ANE shares of 512/1024/2048-row chunks, rounded to 64), or fixed
single-shape packages with ``--shape-mode fixed``.

Models are converted with ``skip_model_load=True`` and verified only with
``ComputeUnit.CPU_ONLY``: nothing here compiles for or runs on the ANE or GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HIDDEN = 5120
INTER = 40960
CLAMP = 20.0
DEFAULT_SCRATCH = "/private/tmp/claude-501/ripple-ane"
# k = round64(R * s) for R in (512, 1024, 2048), s in (0.10, 0.17, 0.25, 0.35, 0.47),
# plus the whole-chunk sizes for the ANE capacity curve.
DEFAULT_ROWS = "64,128,192,256,320,384,512,704,960,1024,2048"


# --------------------------------------------------------------------------- numerics


def bf16_round(a):
    """Round float32 to the nearest BF16 (ties to even), returned as float32."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    u = a.view(np.uint32)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    out = r.view(np.float32).copy()
    nan = np.isnan(a)
    out[nan] = np.nan
    return out


def fp16_round(a):
    return np.asarray(a, dtype=np.float32).astype(np.float16).astype(np.float32)


def reference(x, wg, wu, wd, rnd=None, silu_fp32=True):
    """Ripple MLP in NumPy. rnd: None (FP32 throughout), bf16_round or fp16_round
    applied where the device rounds (projection outputs, SiLU result, product, output)."""
    r = (lambda a: a) if rnd is None else rnd
    with np.errstate(over="ignore"):
        return _reference(x, wg, wu, wd, r, rnd, silu_fp32)


def _reference(x, wg, wu, wd, r, rnd, silu_fp32):
    x = x.astype(np.float32)
    g = r(x @ wg.T)
    u = r(x @ wu.T)
    g = np.minimum(g, CLAMP)
    u = np.clip(u, -CLAMP, CLAMP)
    if silu_fp32 or rnd is None:
        s = r(g / (1.0 + np.exp(-g)))
    else:  # FP16 SiLU: sigmoid rounded, then product rounded
        s = r(g * r(1.0 / (1.0 + np.exp(-g))))
    p = r(s * u)
    return r(p @ wd.T)


def compare(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    d = np.abs(a - b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return {"finite": bool(np.isfinite(a).all() and np.isfinite(b).all()), "max_abs": float(np.nanmax(d)),
            "mean_abs": float(np.nanmean(d)), "ref_max_abs": float(np.nanmax(np.abs(b))),
            "rel_l2": float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)),
            "cosine": float(a @ b / den) if den else None}


# --------------------------------------------------------------------------- weight encodings


def kmeans_lut_grouped(w, nbits, group, iters=40):
    """Grouped-channel scalar LUT along the output axis (axis 0).

    For each block of ``group`` output rows: quantile-initialised 1-D Lloyd
    k-means (sorted-segment means, exact assignment by midpoints). Returns
    indices (uint8, w.shape) and lut (float16, [out/group, 1, 2**nbits, 1]).
    """
    out_dim = w.shape[0]
    assert out_dim % group == 0
    k = 1 << nbits
    lut = np.empty((out_dim // group, 1, k, 1), np.float16)
    idx = np.empty(w.shape, np.uint8)
    for gi in range(out_dim // group):
        blk = w[gi * group:(gi + 1) * group].astype(np.float32)
        v = np.sort(blk.ravel())
        n = v.size
        uniq = np.unique(v)
        if uniq.size <= k:  # exact: every distinct value gets its own code
            c = np.concatenate([uniq, np.repeat(uniq[-1], k - uniq.size)]).astype(np.float64)
        else:
            # init on distinct values (reaches the tails), then 1-D Lloyd on all values
            c = uniq[np.linspace(0, uniq.size - 1, k).round().astype(np.int64)].astype(np.float64)
            csum = np.concatenate([[0.0], np.cumsum(v, dtype=np.float64)])
            for _ in range(iters):
                mids = (c[1:] + c[:-1]) / 2
                edges = np.concatenate([[0], np.searchsorted(v, mids), [n]])
                cnt = np.diff(edges)
                sums = csum[edges[1:]] - csum[edges[:-1]]
                newc = np.where(cnt > 0, sums / np.maximum(cnt, 1), c)
                done = np.max(np.abs(newc - c)) < 1e-7
                c = newc
                if done:
                    break
        c16 = np.sort(c.astype(np.float16))
        cf = c16.astype(np.float32)
        mids = (cf[1:] + cf[:-1]) / 2
        idx[gi * group:(gi + 1) * group] = np.searchsorted(mids, blk).astype(np.uint8)
        lut[gi, 0, :, 0] = c16
    return idx, lut


def lut_dense(idx, lut, group):
    out_dim = idx.shape[0]
    tables = lut[:, 0, :, 0].astype(np.float32)  # [G, K]
    rows = np.repeat(np.arange(out_dim // group), group)
    return np.take_along_axis(tables[rows], idx.astype(np.int64), axis=1)


def int8_per_channel(w):
    w = w.astype(np.float32)
    scale = np.abs(w).max(axis=1, keepdims=True) / 127.0
    scale = np.where(scale == 0, 1.0, scale).astype(np.float16)
    q = np.clip(np.round(w / scale.astype(np.float32)), -127, 127).astype(np.int8)
    return q, scale


# --------------------------------------------------------------------------- program


def encode(fmt, w, lut_group):
    """Returns (encoding dict for the MIL builder, dense float32 weight it decodes to)."""
    if fmt == "fp16":
        w16 = w.astype(np.float16)
        return {"kind": "fp16", "w": w16}, w16.astype(np.float32)
    if fmt == "int8":
        q, s = int8_per_channel(w)
        return {"kind": "int8", "q": q, "scale": s}, q.astype(np.float32) * s.astype(np.float32)
    if fmt in ("lut8", "lut6", "lut4"):
        nbits = int(fmt[3:])
        idx, lut = kmeans_lut_grouped(w, nbits, lut_group)
        return {"kind": "lut", "idx": idx, "lut": lut}, lut_dense(idx, lut, lut_group)
    raise ValueError(fmt)


def slice_enc(enc, rows=None, cols=None):
    """Slice an encoded weight along output rows or input cols (for inner tiles)."""
    def sl(a):
        if rows is not None:
            a = a[rows]
        if cols is not None:
            a = a[:, cols]
        return np.ascontiguousarray(a)
    k = enc["kind"]
    if k == "fp16":
        return {"kind": k, "w": sl(enc["w"])}
    if k == "int8":
        return {"kind": k, "q": sl(enc["q"]), "scale": enc["scale"][rows] if rows is not None else enc["scale"]}
    if k == "lut":
        assert cols is None or rows is None
        lut = enc["lut"]
        if rows is not None:
            g = enc["idx"].shape[0] // lut.shape[0]
            lut = lut[rows.start // g:rows.stop // g]
        return {"kind": k, "idx": sl(enc["idx"]), "lut": np.ascontiguousarray(lut)}
    raise ValueError(k)


def build_program(encs, tiles, silu_fp32, row_dim):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend._utils import _construct_constexpr_lut_op
    from coremltools.converters.mil.mil import types

    def weight(e, name):
        if e["kind"] == "fp16":
            return mb.const(val=e["w"], name=name)
        if e["kind"] == "int8":
            return mb.constexpr_blockwise_shift_scale(data=e["q"], scale=e["scale"], name=name)
        return _construct_constexpr_lut_op(e["idx"], e["lut"], name=name)

    @mb.program(input_specs=[mb.TensorSpec(shape=(row_dim, HIDDEN), dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def prog(x):
        width = INTER // tiles
        acc = None
        for t in range(tiles):
            rs = slice(t * width, (t + 1) * width)
            eg = encs["gate_proj"] if tiles == 1 else slice_enc(encs["gate_proj"], rows=rs)
            eu = encs["up_proj"] if tiles == 1 else slice_enc(encs["up_proj"], rows=rs)
            if tiles == 1:
                ed = encs["down_proj"]
            elif encs["down_proj"]["kind"] == "lut":
                ed = encs["down_tiles"][t]
            else:
                ed = slice_enc(encs["down_proj"], cols=rs)
            g = mb.linear(x=x, weight=weight(eg, f"wg{t}"), name=f"gate{t}")
            u = mb.linear(x=x, weight=weight(eu, f"wu{t}"), name=f"up{t}")
            g = mb.minimum(x=g, y=np.float16(CLAMP), name=f"gate_clamp{t}")
            u = mb.clip(x=u, alpha=np.float16(-CLAMP), beta=np.float16(CLAMP), name=f"up_clip{t}")
            if silu_fp32:
                g32 = mb.cast(x=g, dtype="fp32")
                h = mb.cast(x=mb.mul(x=g32, y=mb.sigmoid(x=g32)), dtype="fp16", name=f"silu{t}")
            else:
                h = mb.mul(x=g, y=mb.sigmoid(x=g), name=f"silu{t}")
            p = mb.mul(x=h, y=u, name=f"prod{t}")
            d = mb.linear(x=p, weight=weight(ed, f"wd{t}"), name=f"down{t}")
            acc = d if acc is None else mb.add(x=acc, y=d, name=f"acc{t}")
        return mb.identity(x=acc, name="y")

    return prog


def convert(prog, shapes, shape_mode):
    import coremltools as ct
    if shape_mode == "enumerated" and len(shapes) > 1:
        shape = ct.EnumeratedShapes(shapes=[(r, HIDDEN) for r in shapes], default=(max(shapes), HIDDEN))
    else:
        shape = (shapes[0], HIDDEN)
    return ct.convert(prog, inputs=[ct.TensorType(name="x", shape=shape, dtype=np.float16)],
                      outputs=[ct.TensorType(name="y", dtype=np.float16)],
                      minimum_deployment_target=ct.target.macOS15,
                      compute_precision=ct.precision.FLOAT16,
                      compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)


def tree_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())


def tree_sha256(root):
    root = Path(root)
    h = hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        h.update(str(p.relative_to(root)).encode())
        fh = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                fh.update(b)
        h.update(fh.hexdigest().encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch", default=DEFAULT_SCRATCH)
    ap.add_argument("--out-dir", default=None, help="default: <scratch>/packages")
    ap.add_argument("--formats", default="fp16,int8,lut8,lut4")
    ap.add_argument("--rows", default=DEFAULT_ROWS)
    ap.add_argument("--shape-mode", default="enumerated", choices=("enumerated", "fixed"))
    ap.add_argument("--inner-tiles", type=int, default=1, help="split the 40960 intermediate into N matmul tiles")
    ap.add_argument("--lut-group", type=int, default=16, help="output channels per LUT")
    ap.add_argument("--silu-fp32", action="store_true")
    ap.add_argument("--verify-rows", type=int, default=256, help="rows checked on CPU_ONLY (0 = skip)")
    ap.add_argument("--keep-going", action="store_true")
    args = ap.parse_args(argv)

    import coremltools as ct

    scratch = Path(args.scratch)
    wdir = scratch / "weights"
    wmeta = json.loads((wdir / "weights.json").read_text())
    out_dir = Path(args.out_dir or scratch / "packages")
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {
        "schema": "mlx2.ripple-ane-ffn-build.v1", "weights_checkpoint": wmeta["checkpoint"], "packages": []}
    if manifest["weights_checkpoint"] != wmeta["checkpoint"]:
        raise SystemExit("existing manifest was built from different weights")
    rows = sorted({int(r) for r in args.rows.split(",")})
    if INTER % args.inner_tiles or (INTER // args.inner_tiles) % args.lut_group:
        raise SystemExit("inner tiles must divide 40960 into LUT-group multiples")

    W = {n: np.load(wdir / f"{n}.deq_fp16.npy").astype(np.float32) for n in ("gate_proj", "up_proj", "down_proj")}
    xv = np.load(wdir / "x_bf16_as_fp32.npy")
    hot = set(wmeta["x"]["hot_rows"])
    vr = args.verify_rows
    if vr:
        # include some clamp-exercising rows in the verified slice
        sel = sorted(hot)[: max(1, vr // 50)]
        cold = [i for i in range(xv.shape[0]) if i not in hot][: vr - len(sel)]
        order = sorted(sel + cold)
        xs = xv[order]
        hot_mask = np.array([i in hot for i in order])
        ref32 = reference(xs, W["gate_proj"], W["up_proj"], W["down_proj"])
        ref_bf16 = reference(xs, W["gate_proj"], W["up_proj"], W["down_proj"], rnd=bf16_round)
        g = xs @ W["gate_proj"].T
        u = xs @ W["up_proj"].T
        clamp_stats = {"gate_clamped_frac": float((g > CLAMP).mean()), "up_clipped_frac": float((np.abs(u) > CLAMP).mean()),
                       "gate_preclamp_max": float(g.max()), "verified_rows": int(xs.shape[0]),
                       "hot_rows_in_slice": len(sel)}
        del g, u
        manifest["gpu_bf16_emulation_vs_fp32"] = compare(ref_bf16, ref32)
        manifest["clamp_stats"] = clamp_stats

    for fmt in args.formats.split(","):
        label = f"{fmt}-t{args.inner_tiles}" + ("-silu32" if args.silu_fp32 else "") + f"-{args.shape_mode}"
        shape_sets = [rows] if args.shape_mode == "enumerated" else [[r] for r in rows]
        t_enc = time.perf_counter()
        encs, dense = {}, {}
        for n, w in W.items():
            if fmt.startswith("lut") and n == "down_proj" and args.inner_tiles > 1:
                # LUT groups run along output rows; tile down along its input columns per tile
                width = INTER // args.inner_tiles
                tiles = []
                dd = []
                for t in range(args.inner_tiles):
                    e, d = encode(fmt, w[:, t * width:(t + 1) * width], args.lut_group)
                    tiles.append(e)
                    dd.append(d)
                encs["down_tiles"] = tiles
                encs[n] = {"kind": "lut"}
                dense[n] = np.concatenate(dd, axis=1)
            else:
                encs[n], dense[n] = encode(fmt, w, args.lut_group)
        enc_s = time.perf_counter() - t_enc
        werr = {n: compare(dense[n], W[n]) for n in W}
        for shapes in shape_sets:
            name = label if len(shape_sets) == 1 else f"{label}-r{shapes[0]}"
            path = out_dir / f"ripple_ffn_{name}.mlpackage"
            entry = {"label": name, "format": fmt, "inner_tiles": args.inner_tiles, "silu_fp32": args.silu_fp32,
                     "shape_mode": args.shape_mode, "row_shapes": shapes, "path": str(path),
                     "lut_group": args.lut_group if fmt.startswith("lut") else None,
                     "weight_encode_s": enc_s, "weight_error_vs_q4_dequant": werr}
            try:
                if path.exists():
                    raise RuntimeError(f"refusing to overwrite {path}")
                from coremltools.converters.mil.mil import get_new_symbol
                row_dim = get_new_symbol() if len(shapes) > 1 else shapes[0]
                t0 = time.perf_counter()
                prog = build_program(encs, args.inner_tiles, args.silu_fp32, row_dim)
                model = convert(prog, shapes, args.shape_mode)
                model.save(str(path))
                entry["build_s"] = time.perf_counter() - t0
                entry["package_bytes"] = tree_bytes(path)
                entry["package_sha256"] = tree_sha256(path)
                del model, prog
                if vr:
                    entry["verify"] = verify(ct, path, xs, shapes, dense, ref32, ref_bf16, args.silu_fp32, hot_mask)
                    ok = entry["verify"].get("repeat_bitwise_identical", True) and \
                        entry["verify"].get("min_row_cosine_vs_fp32", 1.0) > 0.9
                    entry["cpu_verify_passed"] = bool(ok)
                entry["status"] = "built"
            except Exception as exc:
                entry["status"] = f"failed: {type(exc).__name__}: {exc}"
                if path.exists() and entry.get("package_bytes") is None:
                    shutil.rmtree(path, ignore_errors=True)
                if not args.keep_going:
                    manifest["packages"].append(entry)
                    _write_manifest(man_path, manifest)
                    raise
            manifest["packages"] = [p for p in manifest["packages"] if p["label"] != name] + [entry]
            _write_manifest(man_path, manifest)
            print(json.dumps({k: entry.get(k) for k in ("label", "status", "package_bytes", "build_s")}), flush=True)
            if "verify" in entry:
                print(json.dumps(entry["verify"]["summary"]), flush=True)
        del encs, dense
    print(json.dumps({"manifest": str(man_path)}))


def verify(ct, path, xs, shapes, dense, ref32, ref_bf16, silu_fp32, hot_mask):
    """CPU_ONLY prediction at one supported row count; never the ANE or GPU."""
    rows = max(r for r in shapes if r <= xs.shape[0]) if any(r <= xs.shape[0] for r in shapes) else None
    if rows is None:
        return {"skipped": "no package shape fits the verify slice", "summary": {}}
    x = xs[:rows]
    t0 = time.perf_counter()
    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    y = model.predict({"x": x.astype(np.float16)})["y"].astype(np.float32)
    pred_s = time.perf_counter() - t0
    y2 = model.predict({"x": x.astype(np.float16)})["y"].astype(np.float32)
    deterministic = bool(np.array_equal(y, y2))
    row_cos = (y * ref32[:rows]).sum(1) / np.maximum(
        np.linalg.norm(y, axis=1) * np.linalg.norm(ref32[:rows], axis=1), 1e-30)
    # same decoded weights, FP16 rounding at the device's rounding points
    own16 = reference(x, dense["gate_proj"], dense["up_proj"], dense["down_proj"], rnd=fp16_round,
                      silu_fp32=silu_fp32)
    res = {
        "compute_units": "CPU_ONLY", "rows": rows, "load_s": load_s, "predict_s": pred_s,
        "repeat_bitwise_identical": deterministic, "min_row_cosine_vs_fp32": float(row_cos.min()),
        "second_predict_vs_first": compare(y2, y),
        "coreml_cpu_vs_fp32_ref_q4_weights": compare(y, ref32[:rows]),
        "coreml_cpu_vs_gpu_bf16_emulation": compare(y, ref_bf16[:rows]),
        "coreml_cpu_vs_fp16_emulation_own_weights": compare(y, own16),
        "gpu_bf16_emulation_vs_fp32_ref": compare(ref_bf16[:rows], ref32[:rows]),
        "fp16_emulation_own_weights_vs_fp32_ref": compare(own16, ref32[:rows]),
    }
    cold = ~hot_mask[:rows]
    res["cold_rows_only"] = {"rows": int(cold.sum()),
                             "coreml_cpu_vs_fp32_ref": compare(y[cold], ref32[:rows][cold]),
                             "gpu_bf16_emulation_vs_fp32_ref": compare(ref_bf16[:rows][cold], ref32[:rows][cold])}
    res["summary"] = {
        "label_rows": rows, "deterministic": deterministic, "min_row_cos": float(row_cos.min()),
        "cos_vs_fp32": res["coreml_cpu_vs_fp32_ref_q4_weights"]["cosine"],
        "maxabs_vs_fp32": res["coreml_cpu_vs_fp32_ref_q4_weights"]["max_abs"],
        "cos_vs_bf16gpu": res["coreml_cpu_vs_gpu_bf16_emulation"]["cosine"],
        "bf16gpu_cos_vs_fp32": res["gpu_bf16_emulation_vs_fp32_ref"]["cosine"],
        "bf16gpu_maxabs_vs_fp32": res["gpu_bf16_emulation_vs_fp32_ref"]["max_abs"],
        "cold_cos_vs_fp32": res["cold_rows_only"]["coreml_cpu_vs_fp32_ref"]["cosine"],
        "cold_bf16gpu_cos_vs_fp32": res["cold_rows_only"]["gpu_bf16_emulation_vs_fp32_ref"]["cosine"],
        "fp16emu_cos_vs_fp32": res["fp16_emulation_own_weights_vs_fp32_ref"]["cosine"],
    }
    del model
    return res


def _write_manifest(path, manifest):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    os.replace(tmp, path)


if __name__ == "__main__":
    sys.exit(main() or 0)
