"""Llama preflight guards without constructing MLX models."""

import json
from pathlib import Path

import pytest

from mlx2.adapters.standard_decoder import descriptor_for, inspect_artifact
from mlx2.contracts import Capability


def _artifact(tmp_path: Path, **changes: object) -> Path:
    config = {
        "model_type": "llama",
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "intermediate_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 256,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
    }
    config.update(changes)
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors").write_bytes(b"test-shard")
    return tmp_path


def test_full_attention_llama_stays_ordinary_and_unqualified(tmp_path: Path) -> None:
    path = _artifact(
        tmp_path,
        attention_bias=True,
        mlp_bias=True,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
        },
    )
    artifact = inspect_artifact(path, expected="llama")
    descriptor = descriptor_for("llama")
    assert artifact["qualification"] == "pending"
    assert artifact["has_mtp"] is False
    assert descriptor.variant == "ordinary"
    assert Capability.APC_V2 in descriptor.capabilities
    assert Capability.MTP not in descriptor.capabilities


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"num_experts": 8}, "dense SwiGLU"),
        ({"hidden_act": "gelu"}, "dense SwiGLU"),
        ({"attention_bias": "true"}, "attention_bias must be a boolean"),
        ({"mlp_bias": 1}, "mlp_bias must be a boolean"),
        ({"rope_traditional": "false"}, "rope_traditional must be a boolean"),
        ({"use_sliding_window": True}, "sliding Llama"),
        ({"layer_types": ["full_attention", "sliding_attention"]}, "sliding Llama"),
        ({"rope_scaling": []}, "rope_scaling must be an object"),
        ({"rope_scaling": {"rope_type": "unknown"}}, "unsupported Llama RoPE"),
        ({"rope_scaling": {"rope_type": "llama3", "factor": 0}}, "RoPE factor"),
        (
            {
                "rope_scaling": {
                    "rope_type": "llama3",
                    "factor": 32,
                    "high_freq_factor": 1,
                }
            },
            "frequency geometry",
        ),
    ],
)
def test_unsupported_llama_config_fails_before_load(
    tmp_path: Path, changes: dict, message: str
) -> None:
    path = _artifact(tmp_path, **changes)
    with pytest.raises(ValueError, match=message):
        inspect_artifact(path, expected="llama")
