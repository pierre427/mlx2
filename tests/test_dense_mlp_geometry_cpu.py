"""Loaded dense-GLU geometry is inferred without importing MLX."""
from types import SimpleNamespace as NS

import pytest

from mlx2.runtime.dense_mlp_geometry import (
    infer_dense_glu_geometry,
    infer_uniform_dense_glu_geometry,
    validate_semantics,
)


SEMANTICS = {
    "schema": "mlx2.dense-glu-semantics.v1",
    "gate_projection": "gate_proj",
    "up_projection": "up_proj",
    "down_projection": "down_proj",
    "activation": "silu",
    "combination": "activated_gate_times_up",
}


def array(shape, dtype):
    return NS(shape=shape, dtype=dtype)


def projection(output, input_width, *, bits=4, group_size=64, bias=False):
    packed = input_width // (32 // bits)
    groups = input_width // group_size
    value = NS(
        weight=array((output, packed), "uint32"),
        scales=array((output, groups), "bfloat16"),
        biases=array((output, groups), "bfloat16"),
        bits=bits,
        group_size=group_size,
        mode="affine",
    )
    if bias:
        value.bias = array((output,), "bfloat16")
    return value


def layer(hidden=5120, intermediate=17408):
    return NS(
        gate_proj=projection(intermediate, hidden),
        up_proj=projection(intermediate, hidden),
        down_proj=projection(hidden, intermediate),
    )


def test_adapter_declares_semantics_and_loader_infers_real_geometry():
    value = infer_dense_glu_geometry(layer(), SEMANTICS)
    assert value["hidden_size"] == 5120
    assert value["intermediate_size"] == 17408
    assert value["quantization"] == {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
        "weight_dtype": "uint32",
        "scale_dtype": "bfloat16",
        "bias_dtype": "bfloat16",
    }
    assert value["projection_shapes"] == {
        "gate_proj": [17408, 5120],
        "up_proj": [17408, 5120],
        "down_proj": [5120, 17408],
    }
    assert len(value["sha256"]) == 64
    cohort = infer_uniform_dense_glu_geometry((layer(), layer()), SEMANTICS)
    assert cohort["layer_count"] == 2
    assert cohort["geometry"] == value


def test_semantic_and_loaded_geometry_mismatches_fail_closed():
    for bad in (
        None,
        {**SEMANTICS, "activation": "gelu"},
        {**SEMANTICS, "gate_projection": "up_proj"},
        {**SEMANTICS, "extra": True},
    ):
        with pytest.raises(ValueError):
            validate_semantics(bad)
    cases = []
    wrong_up = layer()
    wrong_up.up_proj = projection(8192, 5120)
    cases.append(wrong_up)
    wrong_down = layer()
    wrong_down.down_proj = projection(4096, 17408)
    cases.append(wrong_down)
    wrong_quant = layer()
    wrong_quant.down_proj = projection(5120, 17408, bits=8)
    cases.append(wrong_quant)
    wrong_scale = layer()
    wrong_scale.gate_proj.scales = array((17408, 79), "bfloat16")
    cases.append(wrong_scale)
    for value in cases:
        with pytest.raises(ValueError):
            infer_dense_glu_geometry(value, SEMANTICS)
    with pytest.raises(ValueError):
        infer_uniform_dense_glu_geometry((layer(), layer(4096, 11008)), SEMANTICS)
