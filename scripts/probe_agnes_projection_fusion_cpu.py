"""Bounded real-artifact Agnes projection probe; CPU only, no route selection."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
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


def _layer(path: Path, mapping: dict, index: int, quant: dict, hidden: int):
    prefix = f"language_model.model.layers.{index}.delta_attn."
    names = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
    keys = tuple(prefix + name + "." + field for name in names
                 for field in ("weight", "scales", "biases"))
    shards = {mapping[key] for key in keys}
    if len(shards) != 1:
        raise ValueError(f"layer {index} projection weights span shards: {sorted(shards)}")
    loaded = mx.load(path / shards.pop(), stream=mx.cpu)
    layer = nn.Module()
    for name in names:
        stem = prefix + name + "."
        weight = loaded[stem + "weight"]
        module = nn.QuantizedLinear(
            hidden, weight.shape[0], bias=False, group_size=quant["group_size"],
            bits=quant["bits"], mode=quant["mode"],
        )
        module.weight = weight
        module.scales = loaded[stem + "scales"]
        module.biases = loaded[stem + "biases"]
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
    args = parser.parse_args()
    mx.set_default_device(mx.cpu)
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
                       config["text_config"]["hidden_size"])
        supported = probe_agnes_projection_layer(layer)
        receipt = install_agnes_projection_fusion(
            SimpleNamespace(model_type="agnes", layers=[SimpleNamespace(
                is_linear=True, delta_attn=layer)]), enabled=True,
        )
        parity = None
        if receipt.fused_layers and "float16" in supported:
            inputs = mx.random.normal((1, 9, config["text_config"]["hidden_size"]),
                                      key=mx.random.key(1000 + index)).astype(mx.float16)
            fused = GatedDeltaNet._input_projections(layer, inputs)
            parts = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
            bounds = layer._gdn_fused_bounds
            lower = 0
            exact = []
            for name, upper, candidate in zip(parts, bounds, fused):
                packed = layer.in_proj_fused
                reference = mx.quantized_matmul(
                    inputs, packed.weight[lower:upper], packed.scales[lower:upper],
                    packed.biases[lower:upper], transpose=True,
                    group_size=packed.group_size, bits=packed.bits,
                )
                mx.eval(candidate, reference)
                exact.append(bool(mx.array_equal(candidate, reference).item()))
                lower = upper
            parity = all(exact)
        rows.append({"layer": index, "probe_dtypes": list(supported),
                     "fused": receipt.fused_layers == 1,
                     "wide_rows_exact": parity})
        del layer
        gc.collect()
        mx.clear_cache()
    print(json.dumps({
        "artifact": str(path),
        "config_sha256": _sha256(path / "config.json"),
        "index_sha256": _sha256(path / "model.safetensors.index.json"),
        "device": "CPU", "selected_for_serving": False,
        "metadata_audited_gdn_layers": audited, "layers": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
