"""Qwen3 MoE admission and sampling stay bound to the inspected artifact."""

import json

import pytest

from mlx2.adapters import process_globals
from mlx2.adapters.process_globals import ProcessGlobalConflict
from mlx2.adapters.standard_decoder import StandardDecoderAdapter, inspect_artifact
from mlx2.sampling_defaults import SamplingDefaults, VendorSampling, resolve_sampling


def _artifact(tmp_path, **overrides):
    config = {
        "model_type": "qwen3_moe",
        "hidden_size": 64,
        "num_hidden_layers": 3,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 128,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "decoder_sparse_step": 2,
        "mlp_only_layers": [],
        "moe_intermediate_size": 32,
        "norm_topk_prob": True,
    }
    config.update(overrides)
    (tmp_path / "config.json").write_text(json.dumps(config))
    weights = {
        "model.embed_tokens.weight": "model.safetensors",
        "model.norm.weight": "model.safetensors",
        "model.layers.0.self_attn.q_proj.weight": "model.safetensors",
        "model.layers.1.mlp.gate.weight": "model.safetensors",
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weights})
    )
    (tmp_path / "model.safetensors").write_bytes(b"test-shard")
    return tmp_path


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("attention_bias", True, "biased projections"),
        ("mlp_bias", True, "biased projections"),
        ("rope_traditional", True, "nontraditional RoPE"),
        ("use_sliding_window", True, "full attention"),
        ("sliding_window", 128, "full attention"),
        ("layer_types", ["full_attention", "sliding_attention"], "full attention"),
        ("hidden_act", "gelu", "SwiGLU experts"),
        ("decoder_sparse_step", 4, "no sparse expert layer"),
    ],
)
def test_refuses_topology_the_pinned_moe_math_ignores(tmp_path, flag, value, message):
    with pytest.raises(ValueError, match=message):
        inspect_artifact(_artifact(tmp_path, **{flag: value}), expected="qwen3_moe")


def test_checks_router_at_first_actual_sparse_layer(tmp_path):
    path = _artifact(tmp_path)
    assert inspect_artifact(path, expected="qwen3_moe")["config"]["decoder_sparse_step"] == 2
    index_path = path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"].pop("model.layers.1.mlp.gate.weight")
    index["weight_map"]["model.layers.0.mlp.gate.weight"] = "model.safetensors"
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="first sparse-layer router"):
        inspect_artifact(path, expected="qwen3_moe")


def test_uses_artifact_generation_defaults_with_explicit_override(tmp_path):
    path = _artifact(tmp_path)
    (path / "generation_config.json").write_text(
        json.dumps({"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20})
    )
    artifact = inspect_artifact(path, expected="qwen3_moe")
    profile = VendorSampling.single(
        SamplingDefaults(
            **artifact["sampling_defaults"], source="generation_config.json"
        ),
        model="artifact-bound Qwen3 MoE",
    )
    effective, receipt = resolve_sampling({}, profile, thinking=False)
    assert (effective["temperature"], effective["top_p"], effective["top_k"]) == (
        0.6, 0.95, 20
    )
    assert receipt["sources"]["top_p"] == "generation_config.json"
    explicit, _ = resolve_sampling({"temperature": 0}, profile, thinking=False)
    assert explicit["temperature"] == 0


def test_invalid_generation_defaults_fail_before_model_load(tmp_path):
    path = _artifact(tmp_path)
    (path / "generation_config.json").write_text(
        json.dumps({"do_sample": False, "temperature": 0.6})
    )
    with pytest.raises(ValueError, match="conflicts"):
        inspect_artifact(path, expected="qwen3_moe")


def test_stock_moe_selectors_are_claimed_rolled_back_and_released(monkeypatch):
    from mlx2.runtime.models import moe_nax_gather, switch_layers

    monkeypatch.setattr(process_globals, "_HOLDERS", {})
    monkeypatch.setattr(process_globals, "_PENDING", {})
    monkeypatch.setattr(moe_nax_gather, "MODE", "fused")
    monkeypatch.setattr(switch_layers, "_RHS_PAD_POLICY", "adaptive")

    failed = StandardDecoderAdapter.__new__(StandardDecoderAdapter)

    def fail_after_claim():
        failed._claim_moe_globals()
        raise RuntimeError("model load failed")

    with pytest.raises(RuntimeError, match="model load failed"):
        process_globals.guarded_construction(failed, fail_after_claim)
    assert (moe_nax_gather.MODE, switch_layers._RHS_PAD_POLICY) == (
        "fused", "adaptive"
    )
    assert process_globals.live_selections() == []

    live = StandardDecoderAdapter.__new__(StandardDecoderAdapter)
    process_globals.guarded_construction(live, live._claim_moe_globals)
    assert (moe_nax_gather.MODE, switch_layers._RHS_PAD_POLICY) == ("off", "floor")
    assert process_globals.live_selections()[0][0] == "the Qwen3 MoE ordinary adapter"
    conflicting = StandardDecoderAdapter.__new__(StandardDecoderAdapter)
    with pytest.raises(ProcessGlobalConflict):
        process_globals.claim(
            conflicting,
            "another route",
            {process_globals.MOE_NAX_GATHER: ("fused", moe_nax_gather.set_mode)},
        )
    live.close()
    assert process_globals.live_selections() == []


def test_moe_constructor_guards_claim_lifecycle(tmp_path, monkeypatch):
    from mlx2.runtime.models import moe_nax_gather, switch_layers

    path = _artifact(tmp_path)
    monkeypatch.setattr(process_globals, "_HOLDERS", {})
    monkeypatch.setattr(process_globals, "_PENDING", {})
    monkeypatch.setattr(moe_nax_gather, "MODE", "fused")
    monkeypatch.setattr(switch_layers, "_RHS_PAD_POLICY", "adaptive")

    def fail(self, *_args, **_kwargs):
        self._claim_moe_globals()
        raise RuntimeError("stub load failed")

    monkeypatch.setattr(StandardDecoderAdapter, "_initialize", fail)
    with pytest.raises(RuntimeError, match="stub load failed"):
        StandardDecoderAdapter(str(path))
    assert (moe_nax_gather.MODE, switch_layers._RHS_PAD_POLICY) == (
        "fused", "adaptive"
    )
    assert process_globals.live_selections() == []

    monkeypatch.setattr(
        StandardDecoderAdapter,
        "_initialize",
        lambda self, *_args, **_kwargs: self._claim_moe_globals(),
    )
    live = StandardDecoderAdapter(str(path))
    assert (moe_nax_gather.MODE, switch_layers._RHS_PAD_POLICY) == ("off", "floor")
    live.close()
    assert process_globals.live_selections() == []
