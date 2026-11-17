"""Granite (GraniteForCausalLM) on the standard decoder: gate and CPU parity.

The parity test builds a tiny random float32 Granite with non-1.0 multipliers
and compares mlx2 logits with transformers' GraniteForCausalLM (the reference
named in provenance/standard-decoder-granite.json).
"""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.registry import inspect_model
from mlx2.adapters.standard_decoder import (
    StandardDecoderAdapter,
    descriptor_for,
    inspect_artifact,
)
from mlx2.contracts import Capability
from mlx2.runtime.models.standard_decoder import Model, ModelArgs

TINY = {
    "model_type": "granite",
    "architectures": ["GraniteForCausalLM"],
    "hidden_size": 64,
    "num_hidden_layers": 3,
    "intermediate_size": 160,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 97,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-5,
    "rope_theta": 50000.0,
    "hidden_act": "silu",
    "tie_word_embeddings": False,
    "attention_bias": False,
    "mlp_bias": False,
    # Deliberately far from the defaults so each scalar is observable.
    "attention_multiplier": 0.03125,
    "embedding_multiplier": 12.0,
    "residual_multiplier": 0.22,
    "logits_scaling": 8.0,
}


def _artifact(tmp_path: Path, **changes: object) -> Path:
    config = {**TINY, **changes}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors").write_bytes(b"test-shard")
    return tmp_path


def test_granite_resolves_to_standard_decoder_unqualified(tmp_path):
    path = _artifact(tmp_path)
    resolution = inspect_model(path)
    assert resolution.adapter_type is StandardDecoderAdapter
    assert resolution.descriptor.model_type == "granite"
    assert resolution.descriptor.metadata["qualification"] == "unqualified"
    assert Capability.MTP not in resolution.descriptor.capabilities
    config = resolution.artifact["config"]
    assert config["head_dim"] == 16
    assert config["attention_multiplier"] == 0.03125
    assert descriptor_for("granite").variant == "ordinary"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"attention_multiplier": None}, "attention_multiplier"),
        ({"logits_scaling": 0}, "logits_scaling"),
        ({"residual_multiplier": float("inf")}, "residual_multiplier"),
        ({"embedding_multiplier": "1.0"}, "embedding_multiplier"),
        ({"num_local_experts": 4}, "expert layers"),
        ({"head_dim": 32}, "head_dim"),
        ({"hidden_act": "gelu"}, "dense SwiGLU"),
        ({"use_sliding_window": True}, "sliding"),
    ],
)
def test_granite_gate_fails_closed(tmp_path, changes, message):
    config = {**TINY, **changes}
    config = {k: v for k, v in config.items() if v is not None}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors").write_bytes(b"test-shard")
    with pytest.raises(ValueError, match=message):
        inspect_artifact(tmp_path)


def test_other_families_refuse_granite_multipliers(tmp_path):
    path = _artifact(tmp_path, model_type="llama")
    with pytest.raises(ValueError, match="does not implement Granite multipliers"):
        inspect_artifact(path)
    with pytest.raises(ValueError, match="only Granite"):
        ModelArgs.from_dict({**TINY, "model_type": "llama"})


def test_granite_exposes_structural_mlp_surface():
    model = Model(ModelArgs.from_dict(TINY))
    inner = model.model
    assert len(inner.layers) == TINY["num_hidden_layers"]
    assert inner.embed_tokens.weight.shape == (97, 64)
    assert hasattr(inner, "norm") and hasattr(model, "lm_head")
    mlp = inner.layers[0].mlp
    assert mlp.gate_proj.weight.shape == (160, 64)
    assert mlp.up_proj.weight.shape == (160, 64)
    assert mlp.down_proj.weight.shape == (64, 160)
    assert inner.layers[0].self_attn.scale == 0.03125


def _reference(config):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    hf_config = transformers.GraniteConfig(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        hidden_act="silu",
        max_position_embeddings=config["max_position_embeddings"],
        rms_norm_eps=config["rms_norm_eps"],
        tie_word_embeddings=config["tie_word_embeddings"],
        rope_parameters={"rope_type": "default", "rope_theta": config["rope_theta"]},
        attention_bias=config["attention_bias"],
        mlp_bias=config["mlp_bias"],
        embedding_multiplier=config["embedding_multiplier"],
        logits_scaling=config["logits_scaling"],
        residual_multiplier=config["residual_multiplier"],
        attention_multiplier=config["attention_multiplier"],
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    reference = transformers.GraniteForCausalLM(hf_config).float().eval()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(0.0, 0.25)
    return torch, reference


@pytest.mark.parametrize("bias", [False, True])
def test_granite_forward_matches_transformers_float32(bias):
    config = {**TINY, "attention_bias": bias, "mlp_bias": bias}
    torch, reference = _reference(config)
    model = Model(ModelArgs.from_dict(config))
    weights = {
        name: mx.array(tensor.detach().numpy())
        for name, tensor in reference.state_dict().items()
    }
    model.load_weights(list(model.sanitize(weights).items()), strict=True)
    tokens = np.random.default_rng(1).integers(0, config["vocab_size"], (2, 11))
    with torch.no_grad():
        expected = reference(torch.tensor(tokens)).logits.numpy()
    actual = np.array(model(mx.array(tokens)).astype(mx.float32))
    assert actual.shape == expected.shape
    assert np.max(np.abs(actual - expected)) < 1e-5
    assert (actual.argmax(-1) == expected.argmax(-1)).all()

    # Incremental decode through the KV cache reproduces the full forward.
    cache = model.make_cache()
    rows = [model(mx.array(tokens[:, :6]), cache=cache)[:, -1]]
    for index in range(6, tokens.shape[1]):
        rows.append(model(mx.array(tokens[:, index : index + 1]), cache=cache)[:, -1])
    stepped = np.stack([np.array(row) for row in rows], axis=1)
    assert np.max(np.abs(stepped - expected[:, 5:])) < 1e-5

    # The multipliers are observable: Llama-equivalent scalars on the same
    # weights miss the reference by orders of magnitude more.
    plain = Model(ModelArgs.from_dict({
        **config, "attention_multiplier": (64 // 4) ** -0.5,
        "embedding_multiplier": 1.0, "residual_multiplier": 1.0,
        "logits_scaling": 1.0,
    }))
    plain.load_weights(list(model.sanitize(weights).items()), strict=True)
    assert np.max(np.abs(np.array(plain(mx.array(tokens))) - expected)) > 0.1
