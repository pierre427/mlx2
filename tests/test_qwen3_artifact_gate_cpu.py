"""Dense Qwen3 admission follows the pinned ordinary tensor implementation."""

import json

import pytest

from mlx2.adapters.standard_decoder import inspect_artifact
from mlx2.sampling_defaults import SamplingDefaults, VendorSampling, resolve_sampling


def _artifact(tmp_path, **overrides):
    config = {
        "model_type": "qwen3",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 128,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
    }
    config.update(overrides)
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors").write_bytes(b"test-shard")
    return tmp_path


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("num_experts", 4, "expert layers"),
        ("attention_bias", True, "biased projections"),
        ("mlp_bias", True, "biased projections"),
        ("rope_traditional", True, "nontraditional RoPE"),
        ("use_sliding_window", True, "full attention"),
        ("sliding_window", 128, "full attention"),
        ("layer_types", ["full_attention", "sliding_attention"], "full attention"),
    ],
)
def test_qwen3_refuses_topology_not_implemented(tmp_path, flag, value, message):
    with pytest.raises(ValueError, match=message):
        inspect_artifact(_artifact(tmp_path, **{flag: value}), expected="qwen3")


def test_qwen3_accepts_declared_stock_full_attention(tmp_path):
    artifact = inspect_artifact(
        _artifact(
            tmp_path,
            attention_bias=False,
            use_sliding_window=False,
            layer_types=["full_attention", "full_attention"],
        ),
        expected="qwen3",
    )
    assert artifact["config"]["model_type"] == "qwen3"
    assert artifact["qualification"] == "pending"


def test_qwen3_uses_artifact_generation_defaults_with_receipt_source(tmp_path):
    path = _artifact(tmp_path)
    (path / "generation_config.json").write_text(
        json.dumps({"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20})
    )
    artifact = inspect_artifact(path, expected="qwen3")
    profile = VendorSampling.single(
        SamplingDefaults(
            **artifact["sampling_defaults"], source="generation_config.json"
        ),
        model="artifact-bound Qwen3",
    )
    effective, receipt = resolve_sampling({}, profile, thinking=False)
    assert (effective["temperature"], effective["top_p"], effective["top_k"]) == (
        0.6,
        0.95,
        20,
    )
    assert receipt["sources"]["temperature"] == "generation_config.json"
    explicit, _ = resolve_sampling({"temperature": 0}, profile, thinking=False)
    assert explicit["temperature"] == 0


@pytest.mark.parametrize(
    "generation",
    [
        {"do_sample": "false"},
        {"top_p": 1.2},
        {"temperature": float("nan")},
        {"top_k": True},
        {"do_sample": False, "temperature": 0.6},
    ],
)
def test_qwen3_rejects_invalid_generation_defaults(tmp_path, generation):
    path = _artifact(tmp_path)
    (path / "generation_config.json").write_text(json.dumps(generation))
    with pytest.raises(ValueError):
        inspect_artifact(path, expected="qwen3")


def test_qwen3_without_generation_config_keeps_legacy_fallback(tmp_path):
    assert (
        inspect_artifact(_artifact(tmp_path), expected="qwen3")["sampling_defaults"]
        is None
    )
