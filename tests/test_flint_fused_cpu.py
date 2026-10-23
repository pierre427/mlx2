"""Host-only admission, state geometry and numerical namespace contracts."""

import copy
from pathlib import Path

import mlx.core as mx
import pytest

from mlx2.adapters import flint
from mlx2.runtime.models.cache import (
    BatchKVCache,
    BatchRotatingKVCache,
    KVCache,
    RotatingKVCache,
)
from mlx2.runtime.models.flint_fused import (
    MODE,
    cache_geometry,
    shader_identity,
    validate_config,
)


def config():
    return {
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_attention_heads": 32,
        "num_hidden_layers": 4,
        "layer_types": ["full_attention"] * 2 + ["sliding_attention"] * 2,
        "no_rope_layer_indices": [0, 1],
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_experts": 256,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 1024,
        "sliding_window": 513,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "quantization": {"bits": 4, "group_size": 64},
    }


def test_policy_is_explicit_and_cannot_mix_routes():
    assert flint._validate_execution_policy({"fused_decode": MODE}) == {}
    for policy in (
        {"fused_decode": "unknown"},
        {"fused_decode": MODE, "draft_model": "prompt_lookup"},
    ):
        with pytest.raises(ValueError):
            flint._validate_execution_policy(policy)
    adapter = flint.FlintAdapter.__new__(flint.FlintAdapter)
    adapter.fused_decode = MODE
    with pytest.raises(ValueError, match="max_lanes=1"):
        adapter.execution_config(max_lanes=2, prefill_step=512)
    assert adapter.execution_config(max_lanes=1, prefill_step=512)["num_draft"] == 0


def test_topology_and_mixed_bit_layer_refusal():
    validate_config(config())
    mixed = config()
    mixed["quantization"]["model.layers.0.self_attn.q_proj"] = {
        "bits": 6,
        "group_size": 64,
    }
    with pytest.raises(ValueError, match="mixed-bit"):
        validate_config(mixed)
    wrong = config()
    wrong["rope_theta"] = 100000
    with pytest.raises(ValueError, match="topology"):
        validate_config(wrong)


def test_adapter_numeric_contract_changes_cache_namespace():
    adapter = flint.FlintAdapter.__new__(flint.FlintAdapter)
    adapter.fused_decode = "off"
    assert adapter.execution_numerics_contract() is None
    adapter.fused_decode = MODE
    contract = adapter.execution_numerics_contract()
    assert contract["shader_sha256"] == shader_identity()
    assert contract["scope"] == "B1 one-token decode" and contract["qualified"] is False
    from mlx2.serving import adapter_numerical_contract

    assert adapter_numerical_contract(adapter) == contract


@pytest.mark.parametrize(
    "kind", [KVCache, BatchKVCache, RotatingKVCache, BatchRotatingKVCache]
)
def test_cache_read_map_excludes_only_replaced_sliding_column(kind):
    if kind is KVCache:
        cache = kind()
    elif kind is BatchKVCache:
        cache = kind([0])
    elif kind is RotatingKVCache:
        cache = kind(max_size=513)
    else:
        cache = kind(max_size=513, left_padding=[0])
    assert cache_geometry(cache) == (0, 0, 0, 0)
    # Unique position labels let us compare the fused read map to the actual
    # canonical update. Cover post-prefill overhang and ring wraps.
    for length in [400, 200, 1, 1, 1, 1]:
        copy.deepcopy(cache)
        prior = 0 if cache.keys is None else cache.keys.size
        vals = mx.arange(length, dtype=mx.float32) + 10000 + prior
        kv = mx.broadcast_to(vals[None, None, :, None], (1, 4, length, 128)).astype(
            mx.bfloat16
        )
        if length == 1 and cache.keys is not None:
            _cap, count, start, wrap = cache_geometry(cache)
            logical = [(start + j) % wrap if wrap else start + j for j in range(count)]
            before = cache.keys[0, 0, logical, 0].tolist()
            result, _ = cache.update_and_fetch(kv, kv)
            actual = result[0, 0, :, 0].tolist()
            assert sorted(before + [float(kv[0, 0, 0, 0].item())]) == sorted(actual)
        else:
            cache.update_and_fetch(kv, kv)


def test_no_mutable_input_kv_banks_or_model_wide_embedding():
    root = Path(__file__).parents[1] / "src/mlx2/runtime/models/flint_fused"
    source = (
        (root / "qkv.metal").read_text()
        + (root / "gate_up.metal").read_text()
        + (root / "down_combine.metal").read_text()
    )
    assert (
        "bank0" not in source and "embedding" not in source and "logits" not in source
    )
    assert "new_keys:new_values" in source and "attention_partial" not in source
    assert (root / "LICENSE").is_file()


def test_worker_geometry_preserves_qkv_heads():
    root = Path(__file__).parents[1] / "src/mlx2/runtime/models/flint_fused"
    body = (root / "qkv.metal").read_text()
    header = (root / "layer_header.metal").read_text()
    assert "head<40;head+=36" in body
    assert "head<36?head-32:head-36" in body
    assert "sync_all" not in header and "atomic_" not in header
    assert "threadgroup_barrier" in header


def test_incompatible_cache_refusal():
    cache = BatchKVCache([1])
    with pytest.raises(ValueError, match="unpadded"):
        cache_geometry(cache)
    cache = RotatingKVCache(max_size=513, keep=1)
    cache.keys = mx.zeros((1, 4, 1, 128))
    cache.values = cache.keys
    cache.offset = 1
    with pytest.raises(ValueError, match="sink"):
        cache_geometry(cache)
    wrong = config()
    wrong["no_rope_layer_indices"] = []
    with pytest.raises(ValueError, match="no-RoPE"):
        validate_config(wrong)


def test_device_envelope_is_explicit(monkeypatch):
    from mlx2.runtime.models.flint_fused import validate_device

    monkeypatch.setattr(
        mx,
        "device_info",
        lambda: {"architecture": "applegpu_g15s", "device_name": "Apple M3 Pro"},
    )
    validate_device()
    monkeypatch.setattr(
        mx,
        "device_info",
        lambda: {"architecture": "applegpu_g16s", "device_name": "Apple M5 Max"},
    )
    with pytest.raises(ValueError, match="M3 Pro"):
        validate_device()
