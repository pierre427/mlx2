"""CPU-only candidate checks; fake MLX arrays prevent device initialization."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _budget_class():
    path = Path(__file__).resolve().parents[1] / "src/mlx2/adapters/xing_memory.py"
    spec = importlib.util.spec_from_file_location("xing_memory_cpu", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.XingCacheBudget


def _candidate(monkeypatch):
    core = types.ModuleType("mlx.core")
    core.uint32 = np.uint32
    core.bfloat16 = np.float16  # Two-byte CPU stand-in for BF16 storage.
    core.array = lambda x: np.array(x, copy=True)
    core.zeros = lambda shape, dtype: np.zeros(shape, dtype=dtype)
    core.contiguous = np.ascontiguousarray
    core.concatenate = np.concatenate

    def quantize(x, *, group_size, bits):
        assert (group_size, bits) == (64, 8)
        groups = x.reshape((*x.shape[:-1], 8, 64)).astype(np.float32)
        bias = groups.min(axis=-1)
        scale = (groups.max(axis=-1) - bias) / 255
        scale = np.where(scale == 0, 1, scale)
        q = np.clip(np.rint((groups - bias[..., None]) / scale[..., None]), 0, 255)
        q = q.astype(np.uint32).reshape((*x.shape[:-1], 128, 4))
        packed = q[..., 0] | (q[..., 1] << 8) | (q[..., 2] << 16) | (q[..., 3] << 24)
        return packed, scale.astype(x.dtype), bias.astype(x.dtype)

    def dequantize(packed, scale, bias, *, group_size, bits):
        assert (group_size, bits) == (64, 8)
        q = np.stack(tuple((packed >> shift) & 255 for shift in (0, 8, 16, 24)), axis=-1)
        q = q.reshape((*packed.shape[:-1], 512)).astype(np.float32)
        return (q * np.repeat(scale.astype(np.float32), 64, axis=-1)
                + np.repeat(bias.astype(np.float32), 64, axis=-1)).astype(scale.dtype)

    core.quantize = quantize
    core.dequantize = dequantize
    mlx = types.ModuleType("mlx")
    mlx.core = core

    class Base:
        @classmethod
        def from_state(cls, state, meta_state):
            obj = cls.__new__(cls)
            obj.state = state
            obj.meta_state = meta_state
            return obj

    class Exact(Base):
        def __init__(self, keys, values):
            self.keys, self.values = keys, values
            self.offset = keys.shape[2]

        def keys_and_values(self):
            return self.keys, self.values

    class Segmented:
        pass

    modules = {
        "mlx": mlx,
        "mlx.core": core,
        "mlx2.runtime.models.base": types.SimpleNamespace(create_attention_mask=lambda *a, **k: None),
        "mlx2.runtime.models.cache": types.SimpleNamespace(
            KVCache=Exact, _BaseCache=Base,
            create_attention_mask=lambda *a, **k: None,
            _note_recovery_rewind=lambda cache: None,
        ),
        "mlx2.runtime.segmented_plain_kv": types.SimpleNamespace(SegmentedBatchKVCache=Segmented),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[1] / "src/mlx2/runtime/models/xing_latent_kv8.py"
    spec = importlib.util.spec_from_file_location("mlx2.runtime.models.xing_latent_kv8", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, Exact


def test_kv8_cache_private_conversion_append_rollback_and_restore(monkeypatch):
    module, Exact = _candidate(monkeypatch)
    rng = np.random.default_rng(417)
    latent = rng.normal(size=(1, 1, 5, 512)).astype(np.float16)
    rope = rng.normal(size=(1, 1, 5, 64)).astype(np.float16)
    source = Exact(latent.copy(), rope.copy())
    cache = module.XingLatentKV8Cache.from_exact(source)
    assert cache.offset == 5
    assert cache.values.dtype == np.float16
    assert cache.keys[0].shape[-1] == 128
    assert cache.nbytes > 0
    assert np.array_equal(source.keys, latent)
    assert np.array_equal(source.values, rope)
    got, positional = cache.keys_and_values()
    np.testing.assert_allclose(got, latent, atol=0.025)
    np.testing.assert_allclose(positional, rope, atol=0.002)

    assert cache.trim(2) == 2
    replacement = np.full((1, 1, 2, 512), 0.25, np.float16)
    replacement_rope = np.full((1, 1, 2, 64), 0.5, np.float16)
    cache.update_and_fetch(replacement, replacement_rope)
    got, positional = cache.keys_and_values()
    np.testing.assert_allclose(got[..., :3, :], latent[..., :3, :], atol=0.025)
    np.testing.assert_allclose(got[..., 3:, :], replacement, atol=0.025)
    np.testing.assert_allclose(positional[..., 3:, :], replacement_rope)

    clone = module.XingLatentKV8Cache.from_state(cache.state, cache.meta_state)
    assert clone.offset == cache.offset
    np.testing.assert_allclose(clone.keys_and_values()[0], got)
    assert cache.trim(5) == 5
    assert clone.offset == 5
    with pytest.raises(ValueError, match="metadata version"):
        module.XingLatentKV8Cache.from_state(clone.state, ("other", "5"))
    assert module.latent_kv8_stats()["pack_calls"] >= 2
    assert module.latent_kv8_stats()["dequant_calls"] >= 2
    assert module.latent_kv8_stats()["max_cache_bytes"] > 0


def test_kv8_storage_projection_is_not_admission_budget():
    XingCacheBudget = _budget_class()
    config = {
        "num_hidden_layers": 40,
        "num_nextn_predict_layers": 1,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
    }
    budget = XingCacheBudget.from_config(config, mtp=False)
    ordinary = budget.project(8192)
    candidate = budget.candidate_latent_kv8_storage_bytes(8192)
    assert candidate < ordinary
    assert budget.project(8192) == ordinary
    assert budget.candidate_latent_kv8_storage_bytes(16384) > candidate
    with pytest.raises(ValueError):
        budget.candidate_latent_kv8_storage_bytes(-1)


def test_candidate_operation_is_revision_bound_and_private(monkeypatch):
    cache_module, Exact = _candidate(monkeypatch)
    monkeypatch.setitem(
        sys.modules, "mlx2.runtime.models.xing_latent_kv8", cache_module
    )
    state_errors = types.SimpleNamespace(ApproximateStateError=ValueError)
    monkeypatch.setitem(sys.modules, "mlx2.runtime.approximate_state", state_errors)
    path = Path(__file__).resolve().parents[1] / "src/mlx2/runtime/approximate_kv.py"
    spec = importlib.util.spec_from_file_location("mlx2.runtime.approximate_kv", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    latent = np.ones((1, 1, 3, 512), dtype=np.float16)
    rope = np.ones((1, 1, 3, 64), dtype=np.float16)
    source = Exact(latent, rope)
    first = module.XingLatentKV8Operation(adapter_fingerprint="artifact-a")
    second = module.XingLatentKV8Operation(adapter_fingerprint="artifact-b")
    assert first.revision != second.revision
    assert first.descriptor.cache_layout != "xing4-0-mla-latent-layer-segments-v1"
    state = module.LaneKVState("exact-source", (source,))
    converted = first.apply(state)
    assert converted.revision != state.revision
    assert converted.quantized_planes == 1
    assert type(converted.planes[0]) is cache_module.XingLatentKV8Cache
    assert type(state.planes[0]) is Exact
    with pytest.raises(ValueError, match="homogeneous exact"):
        first.apply(converted)
