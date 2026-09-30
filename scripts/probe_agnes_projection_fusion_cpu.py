"""Bounded real-artifact Agnes projection probe; CPU default, no route selection."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
from mlx import nn
from safetensors import safe_open

from mlx2.adapters.agnes_3_flash import inspect_artifact
from mlx2.runtime.agnes_projection_fusion import (
    install_agnes_projection_fusion,
    probe_agnes_projection_layer,
)
from mlx2.runtime.models.qwen3_5 import GatedDeltaNet


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _layer(path: Path, mapping: dict, index: int, quant: dict, hidden: int, device):
    prefix = f"language_model.model.layers.{index}.delta_attn."
    names = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
    keys = tuple(prefix + name + "." + field for name in names
                 for field in ("weight", "scales", "biases"))
    shards = {mapping[key] for key in keys}
    if len(shards) != 1:
        raise ValueError(f"layer {index} projection weights span shards: {sorted(shards)}")
    loaded = mx.load(path / shards.pop(), stream=mx.cpu)
    mx.eval(*(loaded[key] for key in keys))
    layer = nn.Module()
    for name in names:
        stem = prefix + name + "."
        weight = mx.array(loaded[stem + "weight"])
        module = nn.QuantizedLinear(
            hidden, weight.shape[0], bias=False, group_size=quant["group_size"],
            bits=quant["bits"], mode=quant["mode"],
        )
        module.weight = weight
        module.scales = mx.array(loaded[stem + "scales"])
        module.biases = mx.array(loaded[stem + "biases"])
        setattr(layer, name, module)
    layer.sharding_group = None
    del loaded
    return layer


def _audit_geometry(path: Path, mapping: dict, text: dict, quant: dict) -> int:
    """Check every GDN projection quartet from shard headers, without loading it."""
    hidden = text["hidden_size"]
    key_dim = text["linear_num_key_heads"] * text["linear_key_head_dim"]
    value_dim = text["linear_num_value_heads"] * text["linear_value_head_dim"]
    outputs = {"in_proj_qkv": key_dim * 2 + value_dim,
               "in_proj_z": value_dim,
               "in_proj_b": text["linear_num_value_heads"],
               "in_proj_a": text["linear_num_value_heads"]}
    indices = [i for i, kind in enumerate(text["layer_types"])
               if kind == "agnes_delta_attention"]
    expected = {}
    for index in indices:
        for name, rows in outputs.items():
            prefix = f"language_model.model.layers.{index}.delta_attn.{name}."
            for field in ("weight", "scales", "biases"):
                key = prefix + field
                expected[key] = (index, name, field, rows)
                if key not in mapping:
                    raise ValueError(f"missing Agnes projection tensor {key}")
    inspected = {}
    for shard in sorted({mapping[key] for key in expected}):
        with safe_open(path / shard, framework="numpy") as opened:
            for key in expected:
                if mapping[key] == shard:
                    tensor = opened.get_slice(key)
                    inspected[key] = (tuple(tensor.get_shape()), tensor.get_dtype())
    for key, (_index, _name, field, rows) in expected.items():
        shape, dtype = inspected[key]
        if shape[0] != rows:
            raise ValueError(f"Agnes projection row mismatch: {key}: {shape[0]} != {rows}")
        if field == "weight" and (shape[1] != hidden * quant["bits"] // 32
                                   or dtype != "U32"):
            raise ValueError(f"Agnes packed projection geometry mismatch: {key}")
        if field in ("scales", "biases") and shape[1] != hidden // quant["group_size"]:
            raise ValueError(f"Agnes projection group geometry mismatch: {key}")
        if field in ("scales", "biases") and dtype not in ("BF16", "F16"):
            raise ValueError(f"Agnes projection scale dtype mismatch: {key}")
    return len(indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--layers", type=int, nargs="+", default=(0, 70))
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--bench-reps", type=int, default=0)
    args = parser.parse_args()
    if args.bench_reps < 0 or args.bench_reps > 30:
        raise ValueError("bench reps must be between 0 and 30")
    device = mx.cpu if args.device == "cpu" else mx.gpu
    mx.set_default_device(device)
    path = args.artifact.expanduser().resolve()
    artifact = inspect_artifact(path)
    config = artifact["config"]
    plan = config["text_config"]["layer_types"]
    audited = _audit_geometry(path, artifact["weight_map"], config["text_config"],
                              config["quantization"])
    rows = []
    for index in (() if args.metadata_only else args.layers):
        if index < 0 or index >= len(plan) or plan[index] != "agnes_delta_attention":
            raise ValueError(f"layer {index} is not an Agnes GDN layer")
        layer = _layer(path, artifact["weight_map"], index, config["quantization"],
                       config["text_config"]["hidden_size"], device)
        supported = probe_agnes_projection_layer(layer)
        names = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
        stock_modules = tuple(getattr(layer, name) for name in names)
        cases = {}
        for dtype_name, dtype in (("float16", mx.float16), ("bfloat16", mx.bfloat16)):
            if dtype_name not in supported:
                continue
            for width in (9, 16):
                inputs = mx.random.normal(
                    (1, width, config["text_config"]["hidden_size"]),
                    key=mx.random.key(1000 + index + width),
                ).astype(dtype)
                stock = tuple(module(inputs) for module in stock_modules)
                mx.eval(stock)
                cases[(dtype_name, width)] = (inputs, stock)
        receipt = install_agnes_projection_fusion(
            SimpleNamespace(model_type="agnes", layers=[SimpleNamespace(
                is_linear=True, delta_attn=layer)]), enabled=True,
        )
        parity = None
        parity_cases = {}
        if receipt.fused_layers:
            for (dtype_name, width), (inputs, stock) in cases.items():
                fused = GatedDeltaNet._input_projections(layer, inputs)
                mx.eval(fused)
                parity_cases[f"{dtype_name}/{width}"] = all(
                    bool(mx.array_equal(candidate, reference).item())
                    for candidate, reference in zip(fused, stock)
                )
            parity = all(parity_cases.values()) if parity_cases else None
        microbench = None
        if receipt.fused_layers and args.bench_reps:
            one = cases[("float16", 9)][0][:, :1, :]

            def timed(call):
                start = time.perf_counter()
                mx.eval(call())
                return (time.perf_counter() - start) * 1000

            stock_call = lambda one=one, modules=stock_modules: tuple(module(one) for module in modules)
            fused_call = lambda layer=layer, one=one: GatedDeltaNet._input_projections(layer, one)
            for _ in range(3):
                timed(stock_call)
                timed(fused_call)
            stock_ms, fused_ms = [], []
            for repeat in range(args.bench_reps):
                if repeat % 2:
                    fused_ms.append(timed(fused_call))
                    stock_ms.append(timed(stock_call))
                else:
                    stock_ms.append(timed(stock_call))
                    fused_ms.append(timed(fused_call))
            microbench = {
                "scope": "one-row GDN input projections only, not serving throughput",
                "paired_repetitions": args.bench_reps,
                "stock_median_ms": statistics.median(stock_ms),
                "fused_median_ms": statistics.median(fused_ms),
            }
            microbench["stock_over_fused"] = (
                microbench["stock_median_ms"] / microbench["fused_median_ms"]
            )
        rows.append({"layer": index, "probe_dtypes": list(supported),
                     "fused": receipt.fused_layers == 1,
                     "wide_rows_exact": parity,
                     "wide_rows_exact_by_dtype": parity_cases,
                     "microbenchmark": microbench})
        del layer
        gc.collect()
        mx.clear_cache()
    print(json.dumps({
        "artifact": str(path),
        "config_sha256": _sha256(path / "config.json"),
        "index_sha256": _sha256(path / "model.safetensors.index.json"),
        "device": args.device.upper(), "effective_device": str(mx.default_device()),
        "selected_for_serving": False,
        "metadata_audited_gdn_layers": audited, "layers": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
