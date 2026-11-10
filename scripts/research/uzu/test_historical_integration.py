"""Focused CPU contracts for the Uzu-inspired producer and QMV geometry."""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[3] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.runtime import fused_mlp_stage_a as fused
from mlx2.runtime import int8_prefill as ip
from mlx2.runtime.models.activations import swiglu
from mlx2.runtime.models.qwen3_next import Qwen3NextMLP
from mlx2.runtime.models.row_exact_qmv import TileGeometry


@pytest.mark.parametrize("kwargs", [
    {"rows_per_group": True}, {"rows_per_group": 0}, {"rows_per_group": 9},
    {"rows_per_group": 2, "results_per_simdgroup": 3},
    {"rows_per_group": 2, "results_per_simdgroup": True},
])
def test_bad_tile_geometry(kwargs):
    with pytest.raises(ValueError):
        TileGeometry(**kwargs)


def test_tile_register_budget():
    TileGeometry(2, 8).validate(bits=4, fast=True, rows=3)
    with pytest.raises(ValueError):
        TileGeometry(3, 8).validate(bits=4, fast=True, rows=3)
    with pytest.raises(ValueError):
        TileGeometry(2).validate(bits=8, fast=True, rows=1)


BASE = {"enabled": True, "q45_inplace": True, "inplace_only": True}


def test_fusion_policy_is_explicit_and_revision_bound():
    base = ip.Int8PrefillPolicy(**BASE)
    selected = ip.Int8PrefillPolicy(**BASE, fused_mlp_stage_a=True)
    assert "fused_mlp_stage_a" not in base.as_dict()
    assert selected.as_dict()["fused_mlp_stage_a"] is True
    assert base.revision != selected.revision
    assert ip.apc_semantic_fingerprint("m", base) != ip.apc_semantic_fingerprint("m", selected)
    for kwargs in ({}, {"enabled": True}, {**BASE, "hadamard_rotation": True}):
        with pytest.raises(ValueError):
            ip.Int8PrefillPolicy(**kwargs, fused_mlp_stage_a=True)
    with pytest.raises(TypeError):
        ip.Int8PrefillPolicy(**BASE, fused_mlp_stage_a=1)


def test_fused_producer_rejects_unsupported_inputs_without_gpu_work():
    gate = mx.zeros((2, 64), dtype=mx.bfloat16)
    for up, layout in ((gate, "unknown"), (gate.astype(mx.float32), "q8"),
                       (mx.zeros((2, 128), dtype=mx.bfloat16), "q45")):
        with pytest.raises(ValueError):
            fused.prepare(gate, up, layout=layout)
    with pytest.raises(ValueError, match="requires Metal"):
        fused.prepare(gate, gate, layout="q8")


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = Qwen3NextMLP(256, 512)
        self.set_dtype(mx.bfloat16)
        self.mlp.down_proj = nn.QuantizedLinear.from_linear(self.mlp.down_proj, bits=4, group_size=64)
        self.eval()

    def __call__(self, x):
        return self.mlp(x)


def test_producer_install_is_scoped_and_reversible(monkeypatch):
    monkeypatch.setattr(ip, "require_supported_device", lambda: "fake-M5")
    model, unrelated = Model(), Model()
    x = mx.random.normal((1, 4, 256)).astype(mx.bfloat16)
    ordinary = model(x)
    calls = []

    def prepare(gate, up, **kwargs):
        calls.append(kwargs["layout"])
        return swiglu(gate, up), None, None

    def gemm(product, sa, ra, weight, st, bt, **kwargs):
        return mx.quantized_matmul(product, weight, st.T, bt.T,
                                   bits=kwargs["bits"], group_size=64, transpose=True)

    monkeypatch.setattr(fused, "prepare", prepare)
    monkeypatch.setattr(ip, "q45_gemm", gemm)
    handle = ip.apply(model, {**BASE, "row_threshold": 4, "fused_mlp_stage_a": True})
    try:
        result = model(x)
        assert mx.array_equal(result, ordinary).item()
        assert calls == ["q45"]
        assert handle.receipt()["fused_mlp_stage_a"]["calls"] == 1
        assert "_mlx2_int8_prefill" not in unrelated.mlp.down_proj.__dict__
        assert mx.array_equal(model(x[:, :1]), ordinary[:, :1]).item()
        assert calls == ["q45"]
    finally:
        ip.remove(handle)
    assert mx.array_equal(model(x), ordinary).item()
    assert "_mlx2_int8_prefill" not in model.mlp.down_proj.__dict__


def test_fusion_refuses_unreachable_producer_before_mutation(monkeypatch):
    monkeypatch.setattr(ip, "require_supported_device", lambda: "fake-M5")
    model = Model()
    model.mlp.__class__ = type("DifferentMLP", (Qwen3NextMLP,), {})
    before = type(model.mlp.down_proj)
    with pytest.raises(ip.Int8PrefillError, match="no supported"):
        ip.apply(model, {**BASE, "fused_mlp_stage_a": True})
    assert type(model.mlp.down_proj) is before


def test_fusion_refuses_fp16_only_producer_before_mutation(monkeypatch):
    monkeypatch.setattr(ip, "require_supported_device", lambda: "fake-M5")
    model = Model()
    model.set_dtype(mx.float16)
    before = type(model.mlp.down_proj)
    with pytest.raises(ip.Int8PrefillError, match="no supported"):
        ip.apply(model, {**BASE, "fused_mlp_stage_a": True})
    assert type(model.mlp.down_proj) is before
