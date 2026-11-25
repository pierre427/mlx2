#!/usr/bin/env python3
"""GPU repro: 32-bit KV offset overflow in the M5 FP verify kernel (not run yet).

Same shape as Neywa-Labs/mlx-vlm PR #2831.  ``mlx2_fp_verify_tensor`` read
K/V at ``((b*HK + hk)*N + t)*128`` in 32-bit ``int``.  Once
``(b*HK + hk)*N*128`` passes 2^31 - 1, a lane reads the wrong rows: NaN or
garbage output with no error.  The fix (2026-11-25) promotes ``b`` to
``ulong`` at the first multiply.

The check compares the kernel with itself.  The last lane of a large batch
must be bitwise equal to the same lane run alone as a batch of one.  The
single-lane run starts at offset 0, so it never overflows.  The ``fixed`` arm
is the shipped ``fp_verify_tensor.SOURCE``.  The ``unfixed`` arm rebuilds the
pre-fix source by undoing the promotion; it should show the failure, which
proves the comparator can fail.  A slice-level fp32 SDPA reference is also
recorded as a sanity bound (it is not bitwise).

Default geometry: B=5, HQ=HK=8, L=2, N=524288, D=128, fp16.  The largest
(b*HK + hk) is 39, so 39 * 2^19 * 2^7 is about 2.6e9, above 2^31.  Lane 4 is
fully past the limit.  K and V are 2.7e9 elements each, about 5.4 GB each in
fp16.  Peak is roughly 12 GB plus partials.

Requirements:
- M5-class GPU (``_device_supported``) with no TF32 override.
- An owned GPU window: pass ``--i-own-the-gpu`` and run under the GPU lease,
  e.g. ``scripts/run_with_gpu_locks.py``.
- Writes one ``receipt.json`` into a new or empty ``--out`` directory and
  refuses a nonempty one.

This is a synthetic kernel-arithmetic probe, not model qualification.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROMOTED = ("ulong(b)*HK+hk", "ulong(b)*HQ+h")
UNPROMOTED = ("b*HK+hk", "b*HQ+h")


def _unfixed_source(source: str) -> str:
    for promoted, plain in zip(PROMOTED, UNPROMOTED):
        if promoted not in source:
            raise SystemExit(f"fixed source lacks {promoted!r}; repro is stale")
        source = source.replace(promoted, plain)
    return source


def _launch(mx, fp, kernel, q, k, v, scale):
    b, h, l, d = q.shape
    hk, n = k.shape[1], k.shape[2]
    g = h // hk
    splits = min(32, max(1, (n + 2047) // 2048))
    args = [("T", q.dtype), ("B", b), ("HQ", h), ("HK", hk), ("L", l), ("N", n),
            ("G", g), ("SPLITS", splits), ("MASKED", False), ("CAUSAL", False)]
    partial, m, s = kernel(
        inputs=[q, k, v, mx.zeros((1,), mx.bool_), mx.array([scale], mx.float32)],
        template=args, grid=(((l * g + 15) // 16) * 32, b * hk, splits),
        threadgroup=(32, 1, 1),
        output_shapes=[(splits, b, h, l, d), (splits, b, h, l), (splits, b, h, l)],
        output_dtypes=[mx.float32] * 3,
    )
    return fp.merge(partial, m, s).astype(q.dtype)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--i-own-the-gpu", action="store_true", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--heads", type=int, default=8, help="HQ = HK (G = 1)")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=524288)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing nonempty evidence directory {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx
    from mlx2.runtime.models import fp_verify_tensor as fp
    from mlx2.runtime.gguf.source_dense_legacy import _device_supported

    mx.set_default_device(mx.gpu)
    if not _device_supported():
        raise SystemExit("fp_verify_tensor needs an M5-class GPU without TF32 override")
    b, h, l, n, d = args.batch, args.heads, args.rows, args.tokens, 128
    peak = (b * h - 1) * n * d + (n - 1) * d
    shape = {"B": b, "HQ": h, "HK": h, "L": l, "N": n, "D": d,
             "max_kv_offset": peak, "exceeds_int32": peak > 2**31 - 1}
    if not shape["exceeds_int32"]:
        raise SystemExit(f"geometry does not reach 2^31: {shape}")

    mx.random.seed(20261125)
    # Generated in fp16 directly: an fp32 intermediate would double the peak.
    # Lane-varying random K/V so a wrong-lane read cannot coincidentally match.
    q = mx.random.normal((b, h, l, d), dtype=mx.float16)
    k = mx.random.normal((b, h, n, d), dtype=mx.float16, scale=0.5)
    v = mx.random.normal((b, h, n, d), dtype=mx.float16)
    mx.eval(q, k, v)
    scale = d ** -0.5
    last = slice(b - 1, b)
    q1, k1, v1 = (mx.contiguous(x[last]) for x in (q, k, v))
    mx.eval(q1, k1, v1)

    fixed = mx.fast.metal_kernel(
        name="mlx2_fp_verify_tensor_repro_fixed", input_names=["q", "k", "v", "mask", "scale"],
        output_names=["partial", "maxima", "sums"], header=fp.HEADER, source=fp.SOURCE,
        ensure_row_contiguous=True)
    unfixed = mx.fast.metal_kernel(
        name="mlx2_fp_verify_tensor_repro_unfixed", input_names=["q", "k", "v", "mask", "scale"],
        output_names=["partial", "maxima", "sums"], header=fp.HEADER,
        source=_unfixed_source(fp.SOURCE), ensure_row_contiguous=True)

    reference = mx.fast.scaled_dot_product_attention(
        q1.astype(mx.float32), k1.astype(mx.float32), v1.astype(mx.float32), scale=scale)
    alone = _launch(mx, fp, fixed, q1, k1, v1, scale)
    mx.eval(reference, alone)
    checks = {}
    for name, kernel in (("fixed", fixed), ("unfixed", unfixed)):
        started = time.perf_counter()
        full = _launch(mx, fp, kernel, q, k, v, scale)[last]
        mx.eval(full)
        checks[name] = {
            "last_lane_bitwise_equal_to_alone": bool(mx.array_equal(full, alone).item()),
            "nonfinite": int(mx.sum(~mx.isfinite(full)).item()),
            "max_abs_vs_alone": float(mx.max(mx.abs(full.astype(mx.float32) - alone.astype(mx.float32))).item()),
            "max_abs_vs_fp32_sdpa": float(mx.max(mx.abs(full.astype(mx.float32) - reference)).item()),
            "seconds": time.perf_counter() - started,
        }
    checks["alone_vs_fp32_sdpa_max_abs"] = float(
        mx.max(mx.abs(alone.astype(mx.float32) - reference)).item())
    verdict = (
        "pass" if checks["fixed"]["last_lane_bitwise_equal_to_alone"]
        and checks["fixed"]["nonfinite"] == 0
        and not checks["unfixed"]["last_lane_bitwise_equal_to_alone"]
        else "fail"
    )
    revision = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    receipt = {
        "probe": "fp_verify_tensor 32-bit KV offset overflow",
        "checkpoint": "synthetic:none",
        "source_revision": revision,
        "host": platform.node(), "mlx": getattr(mx, "__version__", "unknown"),
        "geometry": shape, "checks": checks, "verdict": verdict,
        "comparator_can_fail": not checks["unfixed"]["last_lane_bitwise_equal_to_alone"],
    }
    temporary = args.out / ".receipt.json.tmp"
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    os.replace(temporary, args.out / "receipt.json")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
