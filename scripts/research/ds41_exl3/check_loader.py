#!/usr/bin/env python3
"""Admit encoder output through mlx2's upstream EXL3 loader and (optionally) run it on Metal.

Builds a header-only checkpoint view (config.json + index) beside one encoded
expert file, runs runtime/trellis/upstream.plan_checkpoint and load_linear,
and compares the loaded tensors' reference decode with the file. With --metal
it also runs each projection through Exl3DenseLinear (the shipped trellis
kernels, reference law) on a few rows: a short GPU burst.

usage: check_loader.py <expert.safetensors> [--metal]
"""
import argparse, json, sys, tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from mlx2.runtime.trellis.upstream import ShardReader, load_linear, plan_checkpoint  # noqa: E402
from mlx2.runtime.models.exl3_reference import effective_weight  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("file"); ap.add_argument("--metal", action="store_true")
    a = ap.parse_args()
    src = Path(a.file).resolve()
    from safetensors import safe_open
    with safe_open(str(src), "np") as f:
        names = list(f.keys())
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / src.name).symlink_to(src)
        trellis = [n for n in names if n.endswith(".trellis")]
        with safe_open(str(src), "np") as f:
            words = sorted({f.get_slice(n).get_shape()[-1] for n in trellis})
        (tmp / "config.json").write_text(json.dumps({"model_type": "deepseek_v4", "quantization_config": {
            "quant_method": "exl3", "version": "0.0.x", "bits": words[0] / 16, "head_bits": 6, "codebook": "mcg"}}))
        weight_map = {n: src.name for n in names}
        plan = plan_checkpoint(tmp, weight_map)
        read = ShardReader(tmp, weight_map)
        report = {"receipt": plan.receipt(), "linears": {}}
        import mlx.core as mx
        for key, spec in sorted(plan.linears.items()):
            t = load_linear(read, spec)
            tr = np.array(t["trellis"]); suh = np.array(t["suh"]); svh = np.array(t["svh"])
            with safe_open(str(src), "np") as f:
                raw = f.get_tensor(f"{key}.trellis")
            row = {"in": spec.input_dims, "out": spec.output_dims, "words": spec.words,
                   "trellis_bits_equal_file": bool(np.array_equal(tr, raw.view(np.uint16)))}
            if a.metal:
                from mlx2.runtime.models.exl3 import Exl3DenseLinear
                w = effective_weight(tr, suh, svh, "mcg")
                x = np.random.default_rng(0).standard_normal((4, spec.input_dims)).astype(np.float16)
                with mx.stream(mx.gpu):
                    lin = Exl3DenseLinear(t["trellis"], t["suh"], t["svh"], law="reference")
                    lin.eval()
                    y = np.array(lin(mx.array(x)).astype(mx.float32))
                ref = x.astype(np.float64) @ w
                row["metal_vs_reference_rel"] = float(np.linalg.norm(y - ref) / np.linalg.norm(ref))
            report["linears"][key] = row
        print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
