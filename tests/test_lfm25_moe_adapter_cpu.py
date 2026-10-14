"""CPU-only LFM2.5 24B-A2B artifact and adapter contracts."""

import json
from pathlib import Path

import pytest

from mlx2.adapters.lfm25_moe import (
    DESCRIPTOR,
    LAYER_TYPES,
    MODEL_REVISION,
    LFM25MoEAdapter,
    LFM25MoECacheBudget,
    inspect_artifact,
)
from mlx2.adapters.registry import inspect_model
from mlx2.contracts import Capability, StatePlane


def _config():
    return {
        "architectures": ["Lfm2MoeForCausalLM"],
        "model_type": "lfm2_moe",
        "vocab_size": 128000,
        "hidden_size": 2048,
        "intermediate_size": 11776,
        "moe_intermediate_size": 1536,
        "num_hidden_layers": 40,
        "num_experts": 64,
        "num_experts_per_tok": 4,
        "norm_topk_prob": True,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 131072,
        "use_expert_bias": True,
        "num_dense_layers": 2,
        "norm_eps": 1e-5,
        "conv_bias": False,
        "conv_L_cache": 3,
        "routed_scaling_factor": 1.0,
        "tie_word_embeddings": True,
        "layer_types": list(LAYER_TYPES),
        "rope_parameters": {
            "rope_theta": 10_000_000.0,
            "rope_type": "default",
        },
        "quantization": {"group_size": 64, "bits": 8, "mode": "affine"},
        "quantization_config": {
            "group_size": 64,
            "bits": 8,
            "mode": "affine",
        },
    }


def _artifact(tmp_path):
    config = _config()
    weights = {
        "model.embed_tokens.weight": "model.safetensors",
        "model.embedding_norm.weight": "model.safetensors",
    }
    for index, layer_type in enumerate(LAYER_TYPES):
        prefix = f"model.layers.{index}."
        weights[prefix + "operator_norm.weight"] = "model.safetensors"
        weights[prefix + "ffn_norm.weight"] = "model.safetensors"
        weights[
            prefix
            + (
                "self_attn.q_proj.weight"
                if layer_type == "full_attention"
                else "conv.conv.weight"
            )
        ] = "model.safetensors"
        if index < 2:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                weights[prefix + f"feed_forward.{projection}.weight"] = (
                    "model.safetensors"
                )
        else:
            weights[prefix + "feed_forward.gate.weight"] = "model.safetensors"
            weights[prefix + "feed_forward.expert_bias"] = "model.safetensors"
            for projection in ("gate_proj", "up_proj", "down_proj"):
                weights[prefix + f"feed_forward.switch_mlp.{projection}.weight"] = (
                    "model.safetensors"
                )
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weights})
    )
    (tmp_path / "model.safetensors").write_bytes(b"metadata-only")
    return config, weights


def test_exact_official_8bit_artifact_and_registry(tmp_path):
    _artifact(tmp_path)
    artifact = inspect_artifact(tmp_path)
    resolved = inspect_model(tmp_path)
    assert artifact["quantization"]["bits"] == 8
    assert artifact["model_revision"] == MODEL_REVISION
    assert artifact["qualification"] == "pending"
    assert resolved.adapter_type is LFM25MoEAdapter
    assert resolved.default_route == "ordinary"
    assert resolved.descriptor is DESCRIPTOR


def test_capability_and_evidence_boundaries():
    assert StatePlane.RECURRENT in DESCRIPTOR.state_planes
    assert StatePlane.ATTENTION_KV in DESCRIPTOR.state_planes
    assert Capability.APC_V2 in DESCRIPTOR.capabilities
    assert Capability.REASONING in DESCRIPTOR.capabilities
    assert Capability.MTP not in DESCRIPTOR.capabilities
    assert Capability.CONTINUOUS_BATCH in DESCRIPTOR.capabilities
    assert Capability.PROMPT_LOOKUP in DESCRIPTOR.capabilities
    assert Capability.GRAMMAR in DESCRIPTOR.capabilities
    assert Capability.TOOLS in DESCRIPTOR.capabilities
    assert LFM25MoEAdapter.tool_call_open_marker == "<|tool_call_start|>"
    assert DESCRIPTOR.metadata["implemented"] is True
    assert DESCRIPTOR.metadata["qualification"] == "pending"
    assert DESCRIPTOR.metadata["selected"] is False
    assert DESCRIPTOR.metadata["observed_used"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda config, weights: config.update(num_experts_per_tok=2), "topology"),
        (
            lambda config, weights: config["quantization"].update(bits=4),
            "8-bit",
        ),
        (
            lambda config, weights: config["rope_parameters"].update(
                rope_type="linear", factor=2.0
            ),
            "rotary",
        ),
        (
            lambda config, weights: weights.__setitem__(
                "model.layers.2.feed_forward.gate.scales", "model.safetensors"
            ),
            "routing gates",
        ),
        (
            lambda config, weights: weights.pop(
                "model.layers.39.feed_forward.expert_bias"
            ),
            "incomplete",
        ),
    ],
)
def test_artifact_mismatch_fails_closed(tmp_path, mutation, message):
    config, weights = _artifact(tmp_path)
    mutation(config, weights)
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weights})
    )
    with pytest.raises(ValueError, match=message):
        inspect_artifact(tmp_path)


def test_hybrid_cache_budget_and_route_config():
    budget = LFM25MoECacheBudget.from_config(_config(), mtp=False)
    assert (budget.attention_layers, budget.conv_layers) == (10, 30)
    assert budget.project(4096) > budget.project(0)
    assert budget.recurrent_live_bytes == 30 * 2 * 2048 * 4
    with pytest.raises(ValueError, match="MTP"):
        LFM25MoECacheBudget.from_config(_config(), mtp=True)
    config = LFM25MoEAdapter.execution_config(object(), max_lanes=4, prefill_step=512)
    assert config["segment_aware_cohort_size"] == 4
    assert config["backend"] == "ordinary"
    assert config["lfm_shortconv"] == "source"
    assert config["moe_nax_gather"] == "off"


def test_attention_topology_and_kv_quantization_are_declared():
    from mlx2.runtime.attention_topology import FULL_KV, RECURRENT

    adapter = object.__new__(LFM25MoEAdapter)
    adapter.config = _config()
    adapter.identity = {"fingerprint": "f" * 64}
    adapter.layout = DESCRIPTOR.cache_layout
    topology = adapter.attention_topology()
    assert topology.layers.count(FULL_KV) == 10
    assert topology.layers.count(RECURRENT) == 30
    assert topology.state_revision == "f" * 64
    assert set(adapter.approximate_kv_operations()) == {"kv_q8", "kv_k8v4"}


def test_present_official_artifact_when_download_complete():
    path = Path("~/mlx-models/LFM2.5-24B-A2B-MLX-8bit")
    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        pytest.skip("official local artifact is not present")
    index = json.loads(index_path.read_text())
    shards = {path / name for name in index["weight_map"].values()}
    if not all(shard.is_file() for shard in shards):
        pytest.skip("official local artifact download is still in progress")
    record = inspect_artifact(path)
    assert len(record["identity"]["files"]) == 5
    assert record["identity"]["fingerprint"]


@pytest.mark.parametrize(
    ("env", "expected"),
    [(None, "gather"), ("off", "off"), ("gather", "gather"), ("fused", ValueError)],
)
def test_nax_gather_defaults_on_and_operator_value_wins(monkeypatch, env, expected):
    from mlx2.adapters import process_globals

    claims = []
    monkeypatch.setattr(
        process_globals, "claim",
        lambda holder, owner, selections: claims.append(selections) or object(),
    )
    if env is None:
        monkeypatch.delenv("MLX2_MOE_NAX_GATHER", raising=False)
    else:
        monkeypatch.setenv("MLX2_MOE_NAX_GATHER", env)
    adapter = object.__new__(LFM25MoEAdapter)
    if expected is ValueError:
        with pytest.raises(ValueError, match="fused"):
            adapter._claim_moe_globals()
        assert not claims
        return
    adapter._claim_moe_globals()
    assert adapter.moe_nax_gather == expected
    assert claims[0][process_globals.MOE_NAX_GATHER][0] == expected
