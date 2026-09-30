"""CPU-only boundaries for the default-off Agnes projection candidate."""

from types import SimpleNamespace

import mlx.core as mx
from mlx import nn

from mlx2.runtime.agnes_projection_fusion import install_agnes_projection_fusion


class _GDN(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_qkv = nn.Linear(32, 64, bias=False)
        self.in_proj_z = nn.Linear(32, 32, bias=False)
        self.in_proj_b = nn.Linear(32, 8, bias=False)
        self.in_proj_a = nn.Linear(32, 8, bias=False)
        self.sharding_group = None


def _model():
    mx.set_default_device(mx.cpu)
    layer = _GDN()
    nn.quantize(layer, group_size=32, bits=4)
    return SimpleNamespace(
        model_type="agnes",
        layers=[SimpleNamespace(is_linear=True, delta_attn=layer)],
    )


def test_agnes_fusion_is_default_off_and_preserves_stock_weights():
    model = _model()
    receipt = install_agnes_projection_fusion(model)
    assert receipt.expected_layers == 1
    assert receipt.fused_layers == 0
    assert receipt.selected is False
    assert not hasattr(model.layers[0].delta_attn, "in_proj_fused")


def test_agnes_fusion_probes_and_matches_ordinary_projection_on_cpu():
    model = _model()
    layer = model.layers[0].delta_attn
    inputs = mx.random.normal((1, 9, 32)).astype(mx.float16)
    names = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
    stock = tuple(getattr(layer, name)(inputs) for name in names)
    receipt = install_agnes_projection_fusion(model, enabled=True)
    assert receipt.fused_layers == 1
    assert receipt.declined_layers == ()
    assert receipt.selected is False
    assert "float16" in receipt.activation_dtypes
    from mlx2.runtime.models.qwen3_5 import GatedDeltaNet

    fused = GatedDeltaNet._input_projections(layer, inputs)
    mx.eval(stock, fused)
    assert all(mx.array_equal(a, b).item() for a, b in zip(stock, fused))


def test_agnes_fusion_declines_sharded_layer():
    model = _model()
    layer = model.layers[0].delta_attn
    layer.sharding_group = object()
    receipt = install_agnes_projection_fusion(model, enabled=True)
    assert receipt.fused_layers == 0
    assert receipt.declined_layers == (0,)
    assert hasattr(layer, "in_proj_qkv")
