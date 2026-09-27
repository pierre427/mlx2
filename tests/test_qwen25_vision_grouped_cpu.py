"""Qwen vision grouping geometry and execution with real MLX imports blocked."""

import importlib.abc
import sys
import types

import numpy as np
import pytest

from mlx2.adapters.qwen25_vision_grouped import (
    _grouped_sdpa,
    grouped_vision_counters,
    install_grouped_vision_attention,
    plan_equal_windows,
)


class _BlockMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise AssertionError("real MLX import in CPU-only Qwen vision test")
        return None


@pytest.fixture(autouse=True)
def block_real_mlx(monkeypatch):
    assert "mlx.core" not in sys.modules
    monkeypatch.setattr(sys, "meta_path", [_BlockMLX(), *sys.meta_path])


def _softmax_sdpa(q, k, v, *, scale, mask):
    assert mask is None
    if q.shape[-2] == 0:
        return q
    scores = q @ np.swapaxes(k, -1, -2) * scale
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ v


def test_planner_bounds_and_mixed_window_order():
    assert plan_equal_windows([0, 2, 4, 7, 10, 11], 11) == [
        (0, 4, 2, 2), (4, 10, 3, 2), (10, 11, 1, 1)
    ]
    assert plan_equal_windows([0, 2, 4], 5) is None
    assert plan_equal_windows([0, 2, 2, 4], 4) == [(0, 4, 2, 2)]
    assert plan_equal_windows([0, 0, 2, 2, 4, 4], 4) == [(0, 4, 2, 2)]
    assert plan_equal_windows([0, 2, 1, 4], 4) is None
    assert plan_equal_windows([0, 2, 3], 3) is None
    assert plan_equal_windows([0, True, 2], 2) is None
    assert plan_equal_windows(list(range(34)), 33) == [
        (0, 16, 1, 16), (16, 32, 1, 16), (32, 33, 1, 1)
    ]
    assert plan_equal_windows([0, 3000, 6000], 6000) is None


def test_grouped_sdpa_matches_independent_windows():
    rng = np.random.default_rng(147)
    q, k, v = [rng.normal(size=(1, 3, 11, 4)) for _ in range(3)]
    boundaries = [0, 2, 4, 7, 10, 11]
    mx = types.SimpleNamespace(
        fast=types.SimpleNamespace(scaled_dot_product_attention=_softmax_sdpa),
        concatenate=np.concatenate,
    )
    grouped = _grouped_sdpa(mx, q, k, v, plan_equal_windows(boundaries, 11), 0.5)
    reference = np.concatenate([
        _softmax_sdpa(q[:, :, start:end], k[:, :, start:end],
                      v[:, :, start:end], scale=0.5, mask=None)
        for start, end in zip(boundaries, boundaries[1:])
    ], axis=2)
    np.testing.assert_allclose(grouped, reference, rtol=1e-12, atol=1e-12)


def test_grouped_sdpa_skips_only_empty_source_windows():
    rng = np.random.default_rng(148)
    q, k, v = [rng.normal(size=(1, 3, 8, 4)) for _ in range(3)]
    boundaries = [0, 0, 2, 4, 4, 6, 8, 8]
    mx = types.SimpleNamespace(
        fast=types.SimpleNamespace(scaled_dot_product_attention=_softmax_sdpa),
        concatenate=np.concatenate,
    )
    groups = plan_equal_windows(boundaries, 8)
    assert groups == [(0, 8, 2, 4)]
    grouped = _grouped_sdpa(mx, q, k, v, groups, 0.5)
    reference = np.concatenate([
        _softmax_sdpa(q[:, :, start:end], k[:, :, start:end],
                      v[:, :, start:end], scale=0.5, mask=None)
        for start, end in zip(boundaries, boundaries[1:])
    ], axis=2)
    np.testing.assert_allclose(grouped, reference, rtol=1e-12, atol=1e-12)


def test_install_is_opt_in_and_reuses_boundaries_per_forward(monkeypatch):
    core = types.ModuleType("mlx.core")
    core.split = np.split
    core.expand_dims = np.expand_dims
    core.concatenate = np.concatenate
    fast = types.SimpleNamespace(calls=0)

    def sdpa(*args, **kwargs):
        fast.calls += 1
        return _softmax_sdpa(*args, **kwargs)

    fast.scaled_dot_product_attention = sdpa
    core.fast = fast
    mlx = types.ModuleType("mlx")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)

    class Boundaries:
        def __init__(self, values):
            self.values = values
            self.reads = 0

        def tolist(self):
            self.reads += 1
            return list(self.values)

    class Attention:
        def __init__(self):
            self.num_heads = 2
            self.scale = 0.5
            self.qkv = lambda x: np.concatenate((x, x, x), axis=-1)
            self.proj = lambda x: x

        def __call__(self, x, cu_seqlens, rotary_pos_emb=None):
            seq_length = x.shape[0]
            qkv = self.qkv(x).reshape(seq_length, 3, self.num_heads, -1).transpose(1, 0, 2, 3)
            q, k, v = np.split(qkv, 3)
            q = q.transpose(0, 2, 1, 3)
            k = k.transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            edges = cu_seqlens.tolist()
            outputs = [sdpa(q[:, :, start:end], k[:, :, start:end],
                             v[:, :, start:end], scale=self.scale, mask=None)
                       for start, end in zip(edges, edges[1:])]
            return self.proj(np.concatenate(outputs, axis=2).transpose(0, 2, 1, 3).reshape(seq_length, -1))

    class VisionModel:
        def __init__(self):
            self.blocks = [types.SimpleNamespace(attn=Attention()) for _ in range(4)]

        def __call__(self, x, window, full):
            for index, block in enumerate(self.blocks):
                x = block.attn(x, full if index == 1 else window, None)
            return x

    source = types.ModuleType("mlx_vlm.models.qwen2_5_vl.vision")
    source.Attention = Attention
    source.VisionModel = VisionModel
    source.apply_rotary_pos_emb_vision = lambda tensor, rotary: tensor
    monkeypatch.setitem(sys.modules, source.__name__, source)

    x = np.random.default_rng(100).normal(size=(10, 4))
    window = Boundaries([0, 2, 2, 4, 6, 8, 10])
    full = Boundaries([0, 10])
    reference = VisionModel()
    expected = reference(x, window, full)
    assert install_grouped_vision_attention(reference) == 0
    candidate = types.SimpleNamespace(vision_tower=VisionModel())
    before = grouped_vision_counters()
    assert install_grouped_vision_attention(candidate, enable=True) == 4
    window.reads = full.reads = 0
    fast.calls = 0
    actual = candidate.vision_tower(x, window, full)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert window.reads == 1
    assert full.reads == 2  # one candidate lookup plus source fallback
    assert fast.calls == 4  # three grouped windows calls and one full call
    after = grouped_vision_counters()
    assert after["grouped_calls"] - before["grouped_calls"] == 3
    assert after["grouped_windows"] - before["grouped_windows"] == 15
    assert after["reference_fallbacks"] - before["reference_fallbacks"] == 1
    assert after["boundary_materializations"] - before["boundary_materializations"] == 2
    assert grouped_vision_counters(candidate)["grouped_calls"] == 3
    with pytest.raises(TypeError, match="pinned Qwen2.5-VL VisionModel"):
        install_grouped_vision_attention(candidate, enable=True)


def test_adapter_candidate_flag_defaults_off_and_reports_engagement(monkeypatch):
    from mlx2.adapters import qwen25_vl
    from mlx2.adapters.pinned_vlm_candidate import PinnedVisionCandidateAdapter

    pinned = types.SimpleNamespace(vision_tower=object(), language_model=object())

    def fake_base_init(self, model_path, *, execution_policy=None):
        self.model = types.SimpleNamespace(_model=pinned)
        self.mlx_vlm_runtime = {"revision": "fake"}

    calls = []
    monkeypatch.setattr(PinnedVisionCandidateAdapter, "__init__", fake_base_init)
    monkeypatch.setattr(qwen25_vl, "install_grouped_vision_attention",
                        lambda model, *, enable: calls.append((model, enable)))
    monkeypatch.setattr(qwen25_vl, "grouped_vision_counters",
                        lambda model: {"grouped_calls": 2})
    monkeypatch.delenv("MLX2_QWEN25_GROUPED_VISION_CANDIDATE", raising=False)
    ordinary = qwen25_vl.Qwen25VLCandidateAdapter("ignored")
    assert calls == []
    assert ordinary.diagnostics()["qwen25_grouped_vision"] == "source"
    assert ordinary.diagnostics()["qwen25_grouped_vision_counts"] is None
    assert ordinary.execution_config(max_lanes=1, prefill_step=32)[
        "grouped_vision_attention"
    ] == "source"

    monkeypatch.setenv("MLX2_QWEN25_GROUPED_VISION_CANDIDATE", "1")
    candidate = qwen25_vl.Qwen25VLCandidateAdapter("ignored")
    assert calls == [(pinned, True)]
    assert candidate.diagnostics()["qwen25_grouped_vision"] == "candidate"
    assert candidate.diagnostics()["qwen25_grouped_vision_counts"] == {"grouped_calls": 2}
    assert candidate.execution_config(max_lanes=1, prefill_step=32)[
        "grouped_vision_attention"
    ] == "candidate_v1"
    monkeypatch.setenv("MLX2_QWEN25_GROUPED_VISION_CANDIDATE", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        qwen25_vl.Qwen25VLCandidateAdapter("ignored")
