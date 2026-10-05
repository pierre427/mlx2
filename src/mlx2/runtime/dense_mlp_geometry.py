"""CPU-safe semantic declaration and loaded dense-GLU geometry inference.

Adapters declare names and tensor semantics. The runtime derives concrete
dimensions and quantization layout from loaded modules; it never selects an
implementation from a model name.
"""
from __future__ import annotations

import hashlib
import json


SCHEMA = "mlx2.dense-glu-semantics.v1"
GEOMETRY_SCHEMA = "mlx2.loaded-dense-glu-geometry.v1"
_FIELDS = {
    "schema", "gate_projection", "up_projection", "down_projection",
    "activation", "combination",
}


def validate_semantics(value):
    if (
        type(value) is not dict
        or set(value) != _FIELDS
        or value.get("schema") != SCHEMA
        or value.get("activation") != "silu"
        or value.get("combination") != "activated_gate_times_up"
        or any(
            type(value.get(name)) is not str or not value[name].isidentifier()
            for name in ("gate_projection", "up_projection", "down_projection")
        )
        or len({value[name] for name in (
            "gate_projection", "up_projection", "down_projection")}) != 3
    ):
        raise ValueError("exact dense SwiGLU semantic declaration required")
    return dict(value)


def _dtype(value):
    return str(getattr(value, "dtype", "")).split(".")[-1]


def _projection(module, name):
    weight = getattr(module, "weight", None)
    scales = getattr(module, "scales", None)
    biases = getattr(module, "biases", None)
    shape = getattr(weight, "shape", None)
    scale_shape = getattr(scales, "shape", None)
    bias_shape = getattr(biases, "shape", None)
    group_size = getattr(module, "group_size", None)
    bits = getattr(module, "bits", None)
    mode = getattr(module, "mode", None)
    if (
        type(shape) not in (tuple, list)
        or len(shape) != 2
        or any(type(n) is not int or n <= 0 for n in shape)
        or type(group_size) is not int
        or group_size <= 0
        or type(bits) is not int
        or bits <= 0
        or 32 % bits
        or mode != "affine"
        or _dtype(weight) != "uint32"
        or _dtype(scales) != "bfloat16"
        or _dtype(biases) != "bfloat16"
    ):
        raise ValueError(f"{name} is not loaded affine quantized geometry")
    output, packed_input = map(int, shape)
    input_width = packed_input * (32 // bits)
    groups = input_width // group_size
    if (
        input_width % group_size
        or tuple(scale_shape or ()) != (output, groups)
        or tuple(bias_shape or ()) != (output, groups)
    ):
        raise ValueError(f"{name} affine scale/bias geometry differs")
    linear_bias = getattr(module, "bias", None) if hasattr(module, "bias") else None
    if linear_bias is not None and (
        tuple(getattr(linear_bias, "shape", ())) != (output,)
        or _dtype(linear_bias) != "bfloat16"
    ):
        raise ValueError(f"{name} linear bias geometry differs")
    return {
        "input_width": input_width,
        "output_width": output,
        "packed_input_width": packed_input,
        "bits": bits,
        "group_size": group_size,
        "mode": mode,
        "weight_dtype": "uint32",
        "scale_dtype": "bfloat16",
        "bias_dtype": "bfloat16",
        "linear_bias": linear_bias is not None,
    }


def infer_dense_glu_geometry(layer, semantics):
    semantics = validate_semantics(semantics)
    names = tuple(semantics[name] for name in (
        "gate_projection", "up_projection", "down_projection"))
    modules = tuple(getattr(layer, name, None) for name in names)
    if any(module is None for module in modules):
        raise ValueError("loaded dense SwiGLU projections are absent")
    gate, up, down = (
        _projection(module, name) for module, name in zip(modules, names)
    )
    if gate != up:
        raise ValueError("gate and up loaded geometry differs")
    if (
        down["input_width"] != gate["output_width"]
        or down["output_width"] != gate["input_width"]
        or any(down[key] != gate[key] for key in (
            "bits", "group_size", "mode", "weight_dtype", "scale_dtype",
            "bias_dtype"))
    ):
        raise ValueError("down projection does not close loaded dense SwiGLU geometry")
    geometry = {
        "schema": GEOMETRY_SCHEMA,
        "semantics": semantics,
        "hidden_size": gate["input_width"],
        "intermediate_size": gate["output_width"],
        "quantization": {
            key: gate[key] for key in (
                "bits", "group_size", "mode", "weight_dtype", "scale_dtype",
                "bias_dtype")
        },
        "linear_biases": {
            name: projection["linear_bias"]
            for name, projection in zip(names, (gate, up, down))
        },
        "projection_shapes": {
            names[0]: [gate["output_width"], gate["input_width"]],
            names[1]: [up["output_width"], up["input_width"]],
            names[2]: [down["output_width"], down["input_width"]],
        },
    }
    geometry["sha256"] = hashlib.sha256(
        json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return geometry


def infer_uniform_dense_glu_geometry(layers, semantics):
    layers = tuple(layers)
    if not layers:
        raise ValueError("at least one loaded dense SwiGLU layer is required")
    values = tuple(infer_dense_glu_geometry(layer, semantics) for layer in layers)
    if any(value != values[0] for value in values[1:]):
        raise ValueError("loaded dense SwiGLU geometry differs across layers")
    return {
        "schema": "mlx2.loaded-dense-glu-cohort.v1",
        "layer_count": len(values),
        "geometry": values[0],
    }
