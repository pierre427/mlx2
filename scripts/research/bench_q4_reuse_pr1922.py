#!/usr/bin/env python3
"""Isolated M5 comparison of 4-bit affine weight reuse with stock and sp_qmm.

The Metal calculation is adapted from ml-explore/mlx-lm PR #1922 at
187370a64fa11264880024b1dc227a7a767b5ba3 (MIT); see
provenance/q4-reuse-pr1922.json for license notice and modifications.
No model class or production dispatch is changed.

Run --dry-run without importing MLX. GPU execution requires --i-own-the-gpu
and a unique --out path; the owner must separately hold the CPG GPU claim and
/Users/Shared/mlxuag/gpu.lock. A successful microbench is not a model gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE_REVISION = "187370a64fa11264880024b1dc227a7a767b5ba3"
GS = 64
SHAPES = {
    "attn_kv_b4": (1024, 5120),
    "gate_up_b4": (17408, 5120),
    "down_b4": (5120, 17408),
}
ARTIFACT_NAME = "Qwen3.8-27B-oQ4e-mtp"
MS = (4, 6, 8)
ARMS = ("stock", "sp_qmm", "reuse")

# One 32-lane SIMD group owns four output rows. Each lane traverses different
# packed K words; every decoded 4-bit weight is reused across all M inputs.
_SOURCE = r"""
    uint lane = thread_position_in_grid.x;
    uint block = thread_position_in_grid.y;
    const uint row0 = block * NB;

    const uint k_pack = K / 8;
    const uint groups = K / GROUP;
    const uint u_per_group = GROUP / 8;

    float acc[NB][MAX_M];
    for (uint r = 0; r < NB; r++) {
        for (uint m = 0; m < M; m++) {
            acc[r][m] = 0.0f;
        }
    }

    // Two packed words per iteration: more independent loads in flight.
    auto accumulate = [&](uint i) {
        uint g = i / u_per_group;
        uint k_base = i * 8;
        uint u[NB];
        float sc[NB];
        float bi[NB];
        for (uint r = 0; r < NB; r++) {
            u[r] = w[(row0 + r) * k_pack + i];
            sc[r] = static_cast<float>(scales[(row0 + r) * groups + g]);
            bi[r] = static_cast<float>(biases[(row0 + r) * groups + g]);
        }
        for (uint j = 0; j < 8; j++) {
            float v[NB];
            for (uint r = 0; r < NB; r++) {
                v[r] = static_cast<float>((u[r] >> (4 * j)) & 0xF) * sc[r] + bi[r];
            }
            for (uint m = 0; m < M; m++) {
                float xv = static_cast<float>(x[m * K + k_base + j]);
                for (uint r = 0; r < NB; r++) {
                    acc[r][m] += v[r] * xv;
                }
            }
        }
    };

    for (uint i = lane; i < k_pack; i += 64) {
        accumulate(i);
        if (i + 32 < k_pack) {
            accumulate(i + 32);
        }
    }

    for (uint r = 0; r < NB; r++) {
        for (uint m = 0; m < M; m++) {
            float total = simd_sum(acc[r][m]);
            if (lane == 0) {
                out[m * N + row0 + r] = static_cast<T>(total);
            }
        }
    }
"""


def eligible(m: int, n: int, k: int, bits: int, group: int, dtype: str) -> bool:
    """Admission for this isolated kernel, before any device allocation."""
    return (4 <= m <= 8 and n > 0 and n % 4 == 0 and k > 0 and k % 8 == 0
            and group > 0 and group % 8 == 0 and k % group == 0
            and bits == 4 and dtype in ("bfloat16", "float16"))


def packed_reference(x, packed, scales, biases, group: int):
    """Small NumPy reference, used only by CPU tests; no MLX import."""
    import numpy as np

    m, k = x.shape
    n, packs = packed.shape
    if packs * 8 != k:
        raise ValueError("packed K mismatch")
    out = np.zeros((m, n), np.float64)
    for row in range(n):
        for p in range(packs):
            g = (p * 8) // group
            word = int(packed[row, p])
            for j in range(8):
                q = (word >> (4 * j)) & 15
                out[:, row] += (q * scales[row, g] + biases[row, g]) * x[:, p * 8 + j]
    return out


def stock_path(m: int, n: int, k: int) -> str:
    """Expected dispatch on current M5 MLX 39400a0d4; receipt labels it inferred."""
    if m <= 16 and k % 128 == 0 and (m >= 12 or (m >= 8 and n * k >= 1 << 24)):
        return "qmv_nax"
    return "qmv_wide"


def comparison_scope(m: int, n: int, k: int) -> str:
    """Keep inferred NAX stock comparisons outside the non-NAX decision set."""
    return ("non_nax_primary" if stock_path(m, n, k) == "qmv_wide"
            else "nax_contaminated_exploratory")


def plan(shapes: list[str], ms: list[int]) -> list[dict]:
    rows = []
    for shape in shapes:
        n, k = SHAPES[shape]
        for m in ms:
            rows.append({"shape": shape, "M": m, "N": n, "K": k,
                         "stock_path_inferred": stock_path(m, n, k),
                         "comparison_scope": comparison_scope(m, n, k),
                         "sp_qmm_policy_expected": m >= 8 and n >= 256,
                         "omlx_ksplit_eligible": m in (4, 6) and n >= 16384,
                         "reuse_eligible": eligible(m, n, k, 4, GS, "bfloat16")})
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def artifact_catalog(path: Path) -> dict:
    """Inspect safetensors headers and quantization metadata without loading weights."""
    from safetensors import safe_open

    config_path = path / "config.json"
    index_path = path / "model.safetensors.index.json"
    config = json.loads(config_path.read_text())
    qcfg = config["quantization"]
    found = {name: [] for name in SHAPES}
    suffix = {"attn_kv_b4": ("self_attn.k_proj", "self_attn.v_proj"),
              "gate_up_b4": ("mlp.gate_proj", "mlp.up_proj"),
              "down_b4": ("mlp.down_proj",)}
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(shard, framework="np", device="cpu") as f:
            keys = set(f.keys())
            for key in keys:
                if not key.endswith(".weight"):
                    continue
                base = key[:-7]
                quant = qcfg.get(base, qcfg)
                if (quant.get("bits") != 4 or quant.get("group_size") != GS
                        or quant.get("mode") != "affine"):
                    continue
                if not (base + ".scales" in keys and base + ".biases" in keys):
                    continue
                for name, (n, k) in SHAPES.items():
                    if not base.endswith(suffix[name]) or f.get_slice(key).get_shape() != [n, k // 8]:
                        continue
                    if (f.get_slice(key).get_dtype() != "U32"
                            or f.get_slice(base + ".scales").get_dtype() != "BF16"
                            or f.get_slice(base + ".biases").get_dtype() != "BF16"):
                        continue
                    found[name].append(key)
    return {"artifact": str(path.resolve()), "config_sha256": _sha256(config_path),
            "index_sha256": _sha256(index_path),
            "counts": {name: len(keys) for name, keys in found.items()},
            "examples": {name: keys[0] if keys else None for name, keys in found.items()},
            "header_only": True}


def _kernel(mx):
    return mx.fast.metal_kernel(name="mlx2_research_q4_reuse_1922",
                                input_names=["x", "w", "scales", "biases"],
                                output_names=["out"], source=_SOURCE)


def reuse(mx, kernel, x, q):
    w, scales, biases = q
    m, k = x.shape
    n = w.shape[0]
    if not eligible(m, n, k, 4, GS, str(x.dtype).removeprefix("mlx.core.")):
        raise ValueError("unsupported shape or dtype for q4 reuse")
    if w.shape != (n, k // 8) or scales.shape != (n, k // GS) or biases.shape != scales.shape:
        raise ValueError("packed q4 affine layout mismatch")
    if scales.dtype != x.dtype or biases.dtype != x.dtype:
        raise ValueError("scales/biases must match activation dtype")
    return kernel(inputs=[x, w, scales, biases],
                  template=[("T", x.dtype), ("M", m), ("MAX_M", 8), ("NB", 4), ("K", k), ("N", n), ("GROUP", GS)],
                  grid=(32, n // 4, 1), threadgroup=(32, 1, 1),
                  output_shapes=[(m, n)], output_dtypes=[x.dtype])[0]


def metrics(mx, y, stock):
    delta = y.astype(mx.float32) - stock.astype(mx.float32)
    ref = stock.astype(mx.float32)
    peak = max(float(mx.max(mx.abs(ref)).item()), 1e-9)
    l2 = max(float(mx.sqrt(mx.sum(ref * ref)).item()), 1e-9)
    max_abs = float(mx.max(mx.abs(delta)).item())
    rel_l2 = float((mx.sqrt(mx.sum(delta * delta))).item()) / l2
    top1 = float(mx.mean((mx.argmax(y, axis=-1) == mx.argmax(stock, axis=-1)).astype(mx.float32)).item())
    return {"max_abs": max_abs, "max_abs_over_stock_peak": max_abs / peak,
            "relative_l2": rel_l2, "top1_agreement": top1,
            "parity_pass": rel_l2 <= 0.005 and max_abs / peak <= 0.01}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--i-own-the-gpu", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--shapes", nargs="+", choices=tuple(SHAPES), default=list(SHAPES))
    ap.add_argument("--ms", type=int, nargs="+", default=list(MS))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--target-mib", type=int, default=256)
    ap.add_argument("--artifact", type=Path, help="header-only eligibility check for a pinned model")
    ap.add_argument("--omlx-ksplit", action="store_true",
                    help="also benchmark the pinned #3958 split-K geometry where eligible")
    args = ap.parse_args(argv)
    if not args.ms or not all(4 <= m <= 8 for m in args.ms):
        ap.error("--ms must contain only 4..8")
    if min(args.reps, args.iters, args.target_mib) < 1:
        ap.error("reps, iters, and target-mib must be positive")
    rows = plan(args.shapes, args.ms)
    catalog = artifact_catalog(args.artifact) if args.artifact else None
    if catalog and any(catalog["counts"][name] == 0 for name in args.shapes):
        ap.error("one or more requested shapes lack an eligible q4/BF16 artifact projection")
    if args.dry_run:
        if catalog:
            print(json.dumps({"kind": "artifact", **catalog}, sort_keys=True))
        for row in rows:
            print(json.dumps(row, sort_keys=True))
        print(json.dumps({"cases": len(rows), "arms": ARMS,
                          "optional_arm": "omlx_ksplit" if args.omlx_ksplit else None,
                          "note": "M8 stock may dispatch qmv_nax; actual path requires GPU profiler"}))
        return 0
    if not args.i_own_the_gpu or not args.out:
        ap.error("GPU run requires --i-own-the-gpu and --out")
    if args.out.exists():
        ap.error(f"refusing to overwrite existing receipt: {args.out}")

    import mlx.core as mx
    from mlx2.runtime.models import sp_qmm
    if args.omlx_ksplit:
        import omlx3958_ksplit

    mx.set_default_device(mx.gpu)
    kernel = _kernel(mx)
    script = Path(__file__).resolve()
    provenance = ROOT / "provenance/q4-reuse-pr1922.json"
    receipt = {"kind": "source", "git_head": _git("rev-parse", "HEAD"),
               "git_status": _git("status", "--short"), "script_sha256": _sha256(script),
               "provenance_sha256": _sha256(provenance),
               "upstream_metal_sha256": hashlib.sha256(_SOURCE.encode()).hexdigest(),
               "upstream_revision": SOURCE_REVISION,
               "upstream_path": "mlx_lm/models/qgemm.py", "mlx": mx.__version__,
               "device": mx.device_info(), "group_size": GS, "bits": 4,
               "target_mib": args.target_mib, "reps": args.reps, "iters": args.iters,
               "qualification": "microbench_only",
               "decision_scope": "non_nax_primary cases only; inferred NAX stock cases are exploratory"}
    if catalog:
        receipt["artifact"] = catalog
    if args.omlx_ksplit:
        receipt["omlx_ksplit_sha256"] = _sha256(script.with_name("omlx3958_ksplit.py"))
        receipt["omlx_ksplit_provenance_sha256"] = _sha256(ROOT / "provenance/q4-omlx3958-ksplit.json")
        receipt["omlx_ksplit_source_revision"] = "3e9703518984c294c2ed989cf968727d0e0ffa56"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    with args.out.open("x") as sink:
        def emit(item):
            line = json.dumps(item, sort_keys=True, default=str)
            print(line, flush=True)
            sink.write(line + "\n")
            sink.flush()

        emit(receipt)
        mx.random.seed(1922)
        for shape in args.shapes:
            n, k = SHAPES[shape]
            # Packed synthetic weights avoid a full N x K temporary per copy.
            weight_bytes = n * (k // 8) * 4 + 2 * n * (k // GS) * 2
            copies = max(2, min(128, (args.target_mib << 20) // weight_bytes))
            qlist = []
            for _ in range(copies):
                lo = mx.random.randint(0, 1 << 30, shape=(n, k // 8), dtype=mx.int32).astype(mx.uint32)
                hi = mx.random.randint(0, 4, shape=(n, k // 8), dtype=mx.int32).astype(mx.uint32)
                w = lo | (hi << 30)
                sc = (mx.random.uniform(shape=(n, k // GS)) * 0.02).astype(mx.bfloat16)
                bi = (mx.random.uniform(shape=(n, k // GS)) * 0.02 - 0.01).astype(mx.bfloat16)
                mx.eval(w, sc, bi)
                qlist.append((w, sc, bi))
            for m in args.ms:
                x = mx.random.normal((m, k)).astype(mx.bfloat16)
                mx.eval(x)
                arms = ARMS + (("omlx_ksplit",) if args.omlx_ksplit
                               and omlx3958_ksplit.eligible(m, n, k) else ())
                def fn(arm, q):
                    if arm == "stock":
                        return mx.quantized_matmul(x, *q, transpose=True, group_size=GS, bits=4)
                    if arm == "sp_qmm":
                        return sp_qmm.qmm(x, *q, group_size=GS, bits=4)
                    if arm == "omlx_ksplit":
                        return omlx3958_ksplit.qmm(mx, x, q)
                    return reuse(mx, kernel, x, q)

                stock = fn("stock", qlist[0]); mx.eval(stock)
                numeric = {"stock": {"parity_pass": True}}
                for arm in arms[1:]:
                    y = fn(arm, qlist[0]); mx.eval(y)
                    numeric[arm] = metrics(mx, y, stock)
                    if not numeric[arm]["parity_pass"]:
                        failures.append((shape, m, arm))
                timings = {arm: [] for arm in arms}
                for rep in range(args.reps):
                    order = arms if rep % 2 == 0 else arms[::-1]
                    for arm in order:
                        mx.eval(*[fn(arm, q) for q in qlist])
                        for _ in range(args.iters):
                            start = time.perf_counter()
                            mx.eval(*[fn(arm, q) for q in qlist])
                            timings[arm].append((time.perf_counter() - start) / copies)
                base = statistics.median(timings["stock"])
                for arm in arms:
                    median = statistics.median(timings[arm])
                    emit({"kind": "case", "shape": shape, "M": m, "N": n, "K": k,
                          "arm": arm, "path": stock_path(m, n, k) if arm == "stock" else arm,
                          "comparison_scope": comparison_scope(m, n, k),
                          "path_is_inferred": arm == "stock",
                          "sp_qmm_policy_prefer": sp_qmm.prefer(m, n, k, 4),
                          "us_per_op": round(median * 1e6, 2),
                          "speedup_vs_stock": round(base / median, 4),
                          "copies": copies, "weight_bytes_each": weight_bytes,
                          "numerical_vs_stock": numeric[arm]})
            qlist.clear(); mx.clear_cache()
        emit({"kind": "end", "parity_failures": failures,
              "result": "pass" if not failures else "fail"})
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
