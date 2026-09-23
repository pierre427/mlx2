"""GPU exactness check for the HySparse2 prefill paths at paper scale.

Random weights, like bench_hysparse2_prefill.py. Reports, per dtype and length,
the max |logit| difference of paper_exit / suffix_bound against the serving
loop, and the serving loop's top-1/top-2 margin (a flipped argmax with a margin
below the difference is a near-tie, not a divergence).
"""

import argparse
import json

import mlx.core as mx

from bench_hysparse2_prefill import build, prefill, serving_loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["float32:32768", "bfloat16:131072"])
    ap.add_argument("--output")
    args = ap.parse_args()
    rows = []
    for spec in args.runs:
        dtype_name, T = spec.split(":")
        T = int(T)
        model = build(4096, getattr(mx, dtype_name))
        mx.random.seed(1)
        toks = mx.random.randint(0, 32000, (1, T))
        ref = serving_loop(model, toks, 2048).astype(mx.float32)
        top2 = mx.sort(ref, axis=-1)[..., -2:]
        row = {
            "dtype": dtype_name,
            "tokens": T,
            "serving_loop_top2_margin": float(top2[..., 1] - top2[..., 0]),
        }
        for name, bound in (("paper_exit", False), ("suffix_bound", True)):
            out = prefill(model, toks, 2048, bound).astype(mx.float32)
            row[name] = {
                "max_abs_diff": float(mx.abs(out - ref).max()),
                "argmax_agrees": bool(mx.argmax(out, -1).item() == mx.argmax(ref, -1).item()),
            }
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model
        mx.clear_cache()
    if args.output:
        with open(args.output, "w") as fh:
            json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
