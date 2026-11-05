#!/usr/bin/env python3
"""CPU per-module repro: does a TensorFold lane install change what Kimi reads?

Real 48B MLX q8 tensors of one layer (MLA ``kv_b_proj`` and the MoE router
``mlp.gate``, both read through ``kimi_linear.dense_weight`` rather than a
module call). Builds each QuantizedLinear from the checkpoint, takes the
dense matrix, installs raw TensorFold's ``lane_qmm`` (which re-tiles the
packed weights in place, as ``--lane-matmul tensorfold`` does), and takes the
dense matrix again. CPU only: no lane kernel runs, only the weight layout.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path.home() / "Models/Kimi-Linear-48B-A3B-Instruct-8bit")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import mlx.core as mx
    from mlx import nn

    mx.set_default_device(mx.cpu)
    from mlx2.runtime.models.kimi_linear import dense_weight
    from mlx2.runtime.tensorfold_qwen38 import lane_qmm

    index = json.loads((args.model/"model.safetensors.index.json").read_text())["weight_map"]
    names = {"kv_b_proj": f"model.layers.{args.layer}.self_attn.kv_b_proj",
             "router": f"model.layers.{args.layer}.mlp.gate"}
    holder = nn.Module()
    modules = {}
    for label, stem in names.items():
        shard = mx.load(str(args.model/index[stem+".weight"]))
        weight, scales, biases = (shard[stem+"."+part] for part in ("weight", "scales", "biases"))
        out_dim, in_dim = int(weight.shape[0]), int(weight.shape[1])*4
        module = nn.QuantizedLinear(in_dim, out_dim, bias=False, group_size=64, bits=8)
        module.weight, module.scales, module.biases = weight, scales, biases
        modules[label] = module
        setattr(holder, label, module)
    def naive(m):  # the pre-fix dense_weight law: dequantize whatever .weight holds
        return mx.dequantize(m.weight, m.scales, m.biases, group_size=m.group_size, bits=m.bits,
                             mode=m.mode)

    before = {label: naive(m) for label, m in modules.items()}
    for module in modules.values():
        module.__dict__.pop("_kimi_dense", None)
    try:
        lane_qmm.install(holder, tile=True)
        stale = {label: naive(m) for label, m in modules.items()}
        fixed = {label: dense_weight(m) for label, m in modules.items()}
        tiled = {label: bool(getattr(m, "_lane_tiled", False)) for label, m in modules.items()}
    finally:
        lane_qmm.uninstall()
    report = {"model": str(args.model), "layer": args.layer, "modules": {}}
    for label, module in modules.items():
        report["modules"][label] = {
            "shape": list(before[label].shape), "tiled_by_install": tiled[label],
            "prefix_dequantize_after_tile_max_abs_diff": float(mx.max(mx.abs(before[label]-stale[label])).item()),
            "dense_weight_after_tile_identical": bool(mx.array_equal(before[label], fixed[label]).item())}
    report["identical"] = all(m["dense_weight_after_tile_identical"] for m in report["modules"].values())
    text = json.dumps(report, indent=1)
    print(text)
    if args.output:
        args.output.write_text(text+"\n")
    return 0 if report["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
