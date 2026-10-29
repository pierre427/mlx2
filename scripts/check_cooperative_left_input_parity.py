#!/usr/bin/env python3
"""GPU parity for the cooperative-left-input guard (c5308a2a5).

Compares, bitwise, the guarded TensorFold attention kernels against the
previous (unguarded, register-P) kernels of a reference revision, and the
register-P path against the threadgroup-P path, at representative shapes:

* lane_attention ``partial_direct`` (D=256) and ``partial_direct_128`` (D=128)
* stream_attention ``partial`` (via ``tree_sdpa``)

It also records which arm ``is_compatible_as_left_input`` selects on this GPU
with a probe kernel (the guarded source plus a flag output).

Usage: check_cooperative_left_input_parity.py --i-own-the-gpu --ref 026a67023 --out OUT.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
PKG = "src/mlx2/runtime/tensorfold_qwen38"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _load_ref_package(rev: str, tmp: Path):
    dest = tmp / "tf_ref"
    dest.mkdir()
    for name in _git("ls-tree", "--name-only", f"{rev}:{PKG}").splitlines():
        if name.endswith(".py"):
            (dest / name).write_text(_git("show", f"{rev}:{PKG}/{name}") + "\n")
    sys.path.insert(0, str(tmp))
    return importlib.import_module("tf_ref.lane_attention"), importlib.import_module("tf_ref.stream_attention")


def _bits(a: mx.array) -> bytes:
    a = mx.contiguous(a)
    mx.eval(a)
    if a.dtype == mx.bfloat16:
        a = a.view(mx.uint16)
    elif a.dtype == mx.float32:
        a = a.view(mx.uint32)
    import numpy as np
    return np.array(a).tobytes()


def _cmp(a: mx.array, b: mx.array) -> dict:
    ba, bb = _bits(a), _bits(b)
    out = {"bitwise_equal": ba == bb}
    if ba != bb:
        fa, fb = a.astype(mx.float32), b.astype(mx.float32)
        finite = mx.isfinite(fa) & mx.isfinite(fb)
        diff = mx.where(finite, mx.abs(fa - fb), 0.0)
        out["max_abs_diff"] = float(mx.max(diff).item())
        out["n_diff"] = int(mx.sum(fa != fb).item())
        out["nonfinite_mismatch"] = int(mx.sum(mx.isfinite(fa) != mx.isfinite(fb)).item())
    return out


def _lane_variant(la, attr: str, source: str):
    """Swap one lane_attention source global and drop compiled kernels (names carry the source hash)."""
    setattr(la, attr, source)
    la._kernels.clear()


def lane_case(la_new, la_ref, *, H, HKV, T, L, D, seed):
    key = mx.random.key(seed)
    k1, k2, k3 = mx.random.split(key, 3)
    q = (mx.random.normal((1, H, T, D), key=k1) * 1.0).astype(mx.bfloat16)
    k = (mx.random.normal((1, HKV, L, D), key=k2) * 1.0).astype(mx.bfloat16)
    v = mx.random.normal((1, HKV, L, D), key=k3).astype(mx.bfloat16)
    scale = 1.0 / (D ** 0.5)
    G, R = H // HKV, (H // HKV) * T
    RP = 16 * ((R + 15) // 16)
    SGA = RP // 16
    SG = min(SGA, la_new.TILES_PER_GROUP)
    nch = -(-L // la_new.CHUNK)
    qp = q.reshape(HKV, G, T, D).transpose(0, 2, 1, 3).reshape(HKV, R, D)
    if RP != R:
        qp = mx.concatenate([qp, mx.zeros((HKV, RP - R, D), dtype=q.dtype)], axis=1)
    qp = mx.contiguous(qp)
    dims = mx.array([L, nch, T, 1, SGA], dtype=mx.int32)
    sc = mx.array([scale], dtype=mx.float32)
    results = {}
    for tag, mod, direct in (("guarded", la_new, True), ("ref_registers", la_ref, True),
                             ("threadgroup", la_new, False)):
        mod.DIRECT_P = direct
        parts = mod._partial(qp, k, v, sc, dims, G=G, D=D, SG=SG, SGA=SGA, HKV=HKV, nch=nch, RP=RP)
        final = mod.lane_sdpa(q, k, v, scale)
        mx.eval(parts, final)
        results[tag] = (parts, final)
    la_new.DIRECT_P = la_ref.DIRECT_P = True
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal" if T > 1 else None)
    out = {"shape": {"H": H, "HKV": HKV, "T": T, "L": L, "D": D, "seed": seed}}
    for a, b in (("guarded", "ref_registers"), ("guarded", "threadgroup"), ("ref_registers", "threadgroup")):
        pa, fa = results[a]
        pb, fb = results[b]
        out[f"{a}_vs_{b}"] = {
            "PO": _cmp(pa[0], pb[0]), "PM": _cmp(pa[1], pb[1]), "PL": _cmp(pa[2], pb[2]), "out": _cmp(fa, fb),
        }
    g = results["guarded"][1].astype(mx.float32)
    out["guarded_vs_mx_sdpa_max_abs"] = float(mx.max(mx.abs(g - ref.astype(mx.float32))).item())
    return out


def stream_case(sa_new, sa_ref, la_new, *, H, HKV, parents_list, starts, D, seed):
    key = mx.random.key(seed)
    keys_ = mx.random.split(key, 1 + 2 * len(starts))
    rows = sum(len(p) for p in parents_list)
    q = mx.random.normal((1, H, rows, D), key=keys_[0]).astype(mx.bfloat16)
    kv = []
    for s, (parents, start) in enumerate(zip(parents_list, starts)):
        cap = start + len(parents) + 37
        kv.append((mx.random.normal((1, HKV, cap, D), key=keys_[1 + 2 * s]).astype(mx.bfloat16),
                   mx.random.normal((1, HKV, cap, D), key=keys_[2 + 2 * s]).astype(mx.bfloat16)))
    scale = 1.0 / (D ** 0.5)
    tg_source = sa_new._PARTIAL_REGISTERS.replace(la_new._P_REGISTERS, la_new._P_STORE).replace(
        la_new._PV_REGISTERS, la_new._PV_THREADGROUP)
    guarded_spec = sa_new._SPECS["partial"]
    variants = {}
    for tag, mod, spec in (("guarded", sa_new, guarded_spec), ("ref_registers", sa_ref, sa_ref._SPECS["partial"]),
                           ("threadgroup", sa_new, (tg_source, *guarded_spec[1:]))):
        saved = mod._SPECS["partial"]
        mod._SPECS["partial"] = spec
        mod._kernels.clear()
        captured = {}
        orig_kernel = mod._kernel

        def wrapped(name, _orig=orig_kernel, _cap=captured):
            k = _orig(name)
            if name != "partial":
                return k

            def call(**kw):
                # zero-init: in a multi-stream plan, chunk slots past a stream's own count are never written
                res = k(**kw, init_value=0)
                _cap["partial"] = res
                return res
            return call
        mod._kernel = wrapped
        try:
            plan = mod.Plan(parents_list, starts, H, HKV)
            out = mod.tree_sdpa(q, kv, scale, plan)
            mx.eval(out)
            variants[tag] = (captured.get("partial"), out)
        finally:
            mod._kernel = orig_kernel
            mod._SPECS["partial"] = saved
            mod._kernels.clear()
    res = {"shape": {"H": H, "HKV": HKV, "parents": parents_list, "starts": starts, "D": D, "seed": seed}}
    for a, b in (("guarded", "ref_registers"), ("guarded", "threadgroup"), ("ref_registers", "threadgroup")):
        pa, fa = variants[a]
        pb, fb = variants[b]
        entry = {"out": _cmp(fa, fb)}
        if pa is not None and pb is not None:
            entry.update({"PO": _cmp(pa[0], pb[0]), "PM": _cmp(pa[1], pb[1]), "PL": _cmp(pa[2], pb[2])})
        else:
            entry["partial_ran"] = False
        res[f"{a}_vs_{b}"] = entry
    plan = sa_new.Plan(parents_list, starts, H, HKV)
    ordinary = sa_new.ordinary_tree_sdpa(q, kv, scale, plan)
    res["guarded_vs_ordinary_max_abs"] = float(
        mx.max(mx.abs(variants["guarded"][1].astype(mx.float32) - ordinary.astype(mx.float32))).item())
    return res


def flag_probe(la_new, *, D):
    """Which arm the guarded kernel takes: FLAG[0] = 1 if any simdgroup chose registers, FLAG[1] if threadgroup."""
    src = la_new._PARTIAL_DIRECT if D == 256 else la_new._PARTIAL_DIRECT_128
    anchor = "    const bool p_in_registers = opO.template is_compatible_as_left_input<half, bfloat, float>(Pc);\n"
    assert anchor in src
    src = src.replace(anchor, anchor + "    FLAG[p_in_registers ? 0 : 1] = 1;\n")
    name = "lane_attention_flagprobe_" + hashlib.sha256((la_new._HEADER + src).encode()).hexdigest()[:16]
    kern = mx.fast.metal_kernel(name=name, input_names=["Qp", "K", "V", "scale", "dims"],
                                output_names=["PO", "PM", "PL", "FLAG"], source=src, header=la_new._HEADER,
                                ensure_row_contiguous=False)
    H, HKV, T, L = (24, 4, 8, 777) if D == 256 else (32, 2, 4, 777)
    G, R = H // HKV, (H // HKV) * T
    RP = 16 * ((R + 15) // 16)
    SGA = RP // 16
    SG = min(SGA, la_new.TILES_PER_GROUP)
    nch = -(-L // la_new.CHUNK)
    qp = mx.random.normal((HKV, RP, D)).astype(mx.bfloat16)
    k = mx.random.normal((1, HKV, L, D)).astype(mx.bfloat16)
    dims = mx.array([L, nch, T, 1, SGA], dtype=mx.int32)
    outs = kern(inputs=[qp, k, k, mx.array([0.0625], dtype=mx.float32), dims],
                template=[("G", G), ("D", D), ("SG", SG), ("CK", la_new.CHUNK), ("TK", la_new.TILE)],
                grid=(HKV * 32 * SG, nch, -(-SGA // SG)), threadgroup=(32 * SG, 1, 1),
                output_shapes=[(HKV * nch * RP * D,), (HKV * nch * RP,), (HKV * nch * RP,), (2,)],
                output_dtypes=[mx.float32, mx.float32, mx.float32, mx.int32], init_value=0)
    mx.eval(outs)
    flag = [int(x) for x in outs[3].tolist()]
    return {"D": D, "flag": flag, "registers_arm_taken": bool(flag[0]), "threadgroup_arm_taken": bool(flag[1])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-own-the-gpu", action="store_true", required=True)
    ap.add_argument("--ref", default="026a67023")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_path = Path(args.out)
    if out_path.exists() and out_path.stat().st_size:
        raise SystemExit(f"refusing to overwrite {out_path}")
    mx.set_cache_limit(4 << 30)
    sys.path.insert(0, str(ROOT / "src"))
    la_new = importlib.import_module("mlx2.runtime.tensorfold_qwen38.lane_attention")
    sa_new = importlib.import_module("mlx2.runtime.tensorfold_qwen38.stream_attention")
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        la_ref, sa_ref = _load_ref_package(args.ref, Path(tmp))
        report = {
            "head": _git("rev-parse", "HEAD"),
            "ref": _git("rev-parse", args.ref),
            "src_dirty": bool(_git("status", "--porcelain", "--", "src")),
            "mlx": mx.__version__,
            "device": mx.device_info() if hasattr(mx, "device_info") else str(mx.default_device()),
            "macos": platform.mac_ver()[0],
            "sources_sha256": {
                "guarded_partial_direct": hashlib.sha256(la_new._PARTIAL_DIRECT.encode()).hexdigest(),
                "ref_partial_direct": hashlib.sha256(la_ref._PARTIAL_DIRECT.encode()).hexdigest(),
                "guarded_stream_partial": hashlib.sha256(sa_new._PARTIAL.encode()).hexdigest(),
                "ref_stream_partial": hashlib.sha256(sa_ref._PARTIAL.encode()).hexdigest(),
            },
            "flag_probe": [flag_probe(la_new, D=256), flag_probe(la_new, D=128)],
            "lane": [],
            "stream": [],
        }
        seed = 0
        for D, H, HKV in ((256, 24, 4), (128, 32, 2)):
            for T in (1, 3, 8, 16):
                for L in (T + 5, 300, 1031, 4100):
                    if L < T:
                        continue
                    seed += 1
                    report["lane"].append(lane_case(la_new, la_ref, H=H, HKV=HKV, T=T, L=L, D=D, seed=seed))
        trees = [
            ([[-1, 0, 1, 2]], [700]),
            ([[-1, 0, 0, 1, 1, 2, 3, 4]], [1500]),
            ([[-1, 0, 1], [-1, 0, 0, 1, 2]], [300, 2049]),
            ([[-1, 0, 1, 2, 3, 4, 5, 6], [-1], [-1, 0, 0, 0]], [513, 65, 4000]),
        ]
        for parents_list, starts in trees:
            seed += 1
            report["stream"].append(stream_case(sa_new, sa_ref, la_new, H=24, HKV=4, parents_list=parents_list,
                                                starts=starts, D=256, seed=seed))

    def all_eq(rows, pair):
        return all(all(v.get("bitwise_equal", True) for v in r[pair].values() if isinstance(v, dict)) for r in rows)

    rows = report["lane"] + report["stream"]
    report["summary"] = {
        "cases_lane": len(report["lane"]), "cases_stream": len(report["stream"]),
        "guarded_vs_ref_registers_bitwise_all": all_eq(rows, "guarded_vs_ref_registers"),
        "guarded_vs_threadgroup_bitwise_all": all_eq(rows, "guarded_vs_threadgroup"),
        "ref_registers_vs_threadgroup_bitwise_all": all_eq(rows, "ref_registers_vs_threadgroup"),
        "max_abs_vs_mx_sdpa": max(r["guarded_vs_mx_sdpa_max_abs"] for r in report["lane"]),
        "max_abs_vs_ordinary_tree": max(r["guarded_vs_ordinary_max_abs"] for r in report["stream"]),
        "elapsed_s_exploratory": round(time.time() - t0, 1),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmpf = out_path.with_suffix(".tmp")
    tmpf.write_text(json.dumps(report, indent=1, default=str))
    os.replace(tmpf, out_path)
    print(json.dumps({"flag_probe": report["flag_probe"], "summary": report["summary"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
