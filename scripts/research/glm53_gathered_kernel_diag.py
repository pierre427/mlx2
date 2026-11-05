# SPDX-License-Identifier: MIT
"""Where the GLM-5.3 gathered attention kernels spend their time (diagnostic).

Builds timing-only variants of the fp16-operand flash partial kernel by
source substitution (never the production kernel; outputs of the variants
are meaningless) and times each on one ``--rows``-row block at ``--total``
latent rows, ``--layers`` launches per sample, median of ``--reps``:

``full``          the production flash f16 partial (FLASH_OUTER slots) + combine;
``fp32_full``     the production fp32 per-split fused_v2 partial + combine;
``no_combine``    flash partial only (the combine's cost is the difference);
``no_gather``     the tile fills read no device rows (constant values);
``no_pass2``      the value pass never runs (scores + statistics only);
``no_mma``        both MMA loops removed (fills, barriers, statistics only);
``combine_only``  the combine kernel on fixed partials.

The receipts kernel-diag-16k and kernel-diag-16k-split32 (2026-11-02) were
taken with this script's first version on the per-split fp16 twin of
fused_v2 (the kernel 127cd63a6 replaced): there, the partials write and the
combine read were ~40% of the time, the gather fills ~25%, the MMAs ~7%.

A/B signal, not publishable statistics; the GPU must be free.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def variant_sources(src):
    gather_fill = "v = src[c * (DW1 / 4) + c4];"
    gather_fill2 = "v = src[c4];"
    assert src.count(gather_fill) == 1 and src.count(gather_fill2) == 1
    no_gather = src.replace(gather_fill, "v = half4(1.0h);").replace(gather_fill2, "v = half4(1.0h);")
    pass2_loop = "for (uint t = 0; t < SPLIT / 8; t++) {"
    assert src.count(pass2_loop) == 1
    no_pass2 = src.replace(pass2_loop, "for (uint t = 0; t < 0; t++) {")
    mma1 = "simdgroup_multiply_accumulate(s, qf, kf, s);"
    mma2 = "simdgroup_multiply_accumulate(o[j], pf, vf, o[j]);"
    assert src.count(mma1) == 1 and src.count(mma2) == 1
    no_mma = src.replace(mma1, "").replace(mma2, "")
    return {"no_gather": no_gather, "no_pass2": no_pass2, "no_mma": no_mma}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--total", type=int, default=16384)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--layers", type=int, default=11)
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--split", type=int, default=64)
    parser.add_argument("--dim-split", type=int, default=4)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to reuse a nonempty evidence directory")
    args.out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx
    import numpy as np

    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from glm53_gathered_f16_bench import selection
    from mlx2.runtime.models import glm53_sparse_attention as sa
    from mlx2.runtime.models import glm53_sparse_attention_v2 as v2

    mx.set_default_device(mx.gpu)
    rng = np.random.default_rng(5306)
    width = sa.WIDTH
    spec = v2.kernel_spec()
    kernels = {
        "f16": mx.fast.metal_kernel(**spec["partial_flash"], ensure_row_contiguous=True),
        "fp32": mx.fast.metal_kernel(**spec["partial"], ensure_row_contiguous=True),
        "combine": mx.fast.metal_kernel(**spec["combine"], ensure_row_contiguous=True),
    }
    for name, source in variant_sources(spec["partial_flash"]["source"]).items():
        kernels[name] = mx.fast.metal_kernel(**{**spec["partial_flash"], "name": f"diag_{name}", "source": source},
                                             ensure_row_contiguous=True)
    plan = v2.launch_plan(args.rows, width, split=args.split, dim_split=args.dim_split, kind="flash_f16")
    plan32 = v2.launch_plan(args.rows, width, split=args.split, dim_split=args.dim_split)
    ids = mx.array(selection(rng, args.total, args.rows, width))
    meta = mx.array([[args.total, 0, 0, args.total - args.rows + r] for r in range(args.rows)], dtype=mx.int32)
    tails32 = mx.zeros((1, 512), mx.float32)
    layers = []
    for _ in range(args.layers):
        base = mx.array(rng.normal(size=(args.total, 512)).astype(np.float32))
        q = mx.array(rng.normal(size=(args.rows, 64, 512)).astype(np.float32))
        layers.append((base, base.astype(mx.float16), q, q.astype(mx.float16)))
    mx.eval(ids, meta, tails32, [a for layer in layers for a in layer])
    tails16 = tails32.astype(mx.float16)
    mx.eval(tails16)

    def partial(kernel, base, tails, q):
        p = plan32 if kernel == "fp32" else plan
        return kernels[kernel](inputs=[q, base, tails, ids, meta], template=p["template"],
                               grid=p["partial"]["grid"], threadgroup=p["partial"]["threadgroup"],
                               output_shapes=p["partial"]["output_shapes"], output_dtypes=[mx.float32] * 3)

    def combine(m, l, o):
        p = plan32 if m.shape[2] == plan32["splits"] else plan
        return kernels["combine"](inputs=[m, l, o], template=p["template"], grid=p["combine"]["grid"],
                                  threadgroup=p["combine"]["threadgroup"],
                                  output_shapes=p["combine"]["output_shapes"], output_dtypes=[mx.float32])[0]

    fixed = [partial("f16", b16, tails16, q16) for _, b16, _, q16 in layers]
    mx.eval(fixed)

    def run(variant):
        outs = []
        for b32, b16, q32, q16 in layers:
            if variant == "fp32_full":
                outs.append(combine(*partial("fp32", b32, tails32, q32)))
            elif variant == "full":
                outs.append(combine(*partial("f16", b16, tails16, q16)))
            elif variant == "no_combine":
                outs.append(partial("f16", b16, tails16, q16)[2])
            elif variant == "combine_only":
                outs.append(combine(*fixed[len(outs)]))
            else:
                outs.append(partial(variant, b16, tails16, q16)[2])
        mx.eval(outs)

    variants = ["full", "fp32_full", "no_combine", "no_gather", "no_pass2", "no_mma", "combine_only"]
    for variant in variants:
        run(variant)
    record = {"schema": "mlx2.glm53-gathered-kernel-diag.v1", "total": args.total, "rows": args.rows,
              "layers": args.layers, "split": args.split, "dim_split": args.dim_split, "reps": args.reps,
              "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
              "device": mx.device_info().get("device_name"), "mlx": mx.__version__,
              "partials_bytes_per_row": {"flash_f16": plan["splits"] * 64 * 514 * 4,
                                         "fused_v2": v2.workspace_bytes(1, split=args.split)}, "variants": {}}
    for rep in range(args.reps):
        for variant in (variants if rep % 2 == 0 else variants[::-1]):
            mx.clear_cache()
            t0 = time.perf_counter()
            run(variant)
            record["variants"].setdefault(variant, []).append(time.perf_counter() - t0)
    summary = {}
    for variant, seconds in record["variants"].items():
        summary[variant] = statistics.median(seconds) / (args.rows * args.layers) * 1e6
    record["us_per_row_layer"] = summary
    (args.out / "diag.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"us_per_row_layer": {k: round(v, 1) for k, v in summary.items()},
                      "partials_MB_per_row": {k: v / 1e6 for k, v in record["partials_bytes_per_row"].items()}},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
