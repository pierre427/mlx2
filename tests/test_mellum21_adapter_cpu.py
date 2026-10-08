"""Mellum 2.1 metadata, dispatch, prompt and output contracts."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from mlx2.adapters.mellum21 import (
    CACHE_LAYOUT,
    MELLUM21_SAMPLING,
    MELLUM21_THINKING,
    SOURCE_REVISION,
    Mellum21ThinkingAdapter,
    inspect_artifact,
)
from mlx2.adapters.registry import inspect_model, resolve_adapter
from mlx2.contracts import Capability


def mellum_config():
    return {
        "architectures": ["MellumForCausalLM"],
        "attention_bias": False,
        "bos_token_id": 0,
        "dtype": "bfloat16",
        "eos_token_id": 28,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2304,
        "intermediate_size": 7168,
        "layer_types": [
            "full_attention" if index % 4 == 3 else "sliding_attention"
            for index in range(28)
        ],
        "mlp_layer_types": ["sparse"] * 28,
        "max_position_embeddings": 131072,
        "model_type": "mellum",
        "moe_intermediate_size": 896,
        "norm_topk_prob": True,
        "num_attention_heads": 32,
        "num_experts": 64,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 28,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "full_attention": {
                "rope_type": "yarn",
                "rope_theta": 500000.0,
                "factor": 16.0,
                "original_max_position_embeddings": 8192,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "attention_factor": 1.2772588722239782,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 500000.0,
            },
        },
        "sliding_window": 1024,
        "tie_word_embeddings": False,
        "use_sliding_window": True,
        "vocab_size": 98304,
    }


def artifact(path, *, packed=False):
    (path / "config.json").write_text(json.dumps(mellum_config()))
    (path / "chat_template.jinja").write_text(
        "{% if enable_thinking %}<think>{% endif %}<tool_call>"
    )
    (path / "tokenizer.json").write_text("{}")
    shard = path / "model.safetensors"
    shard.write_bytes(b"metadata-only fixture")
    required = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.mlp.gate.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
    required += (
        [
            "model.layers.0.mlp.switch_mlp.gate_proj.weight",
            "model.layers.27.mlp.switch_mlp.down_proj.weight",
        ]
        if packed
        else [
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.27.mlp.experts.63.down_proj.weight",
        ]
    )
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard.name for name in required}})
    )
    return path


def test_registry_dispatch_is_metadata_only(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import mlx2.adapters.registry; "
                "assert 'mlx.core' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    resolved = inspect_model(artifact(tmp_path))
    assert resolved.adapter_type is Mellum21ThinkingAdapter
    assert resolved.default_route == "ordinary"
    assert resolved.artifact["qualification"] == "pending"
    with pytest.raises(ValueError, match="native MTP"):
        resolve_adapter(tmp_path, mtp=True)


def test_topology_and_hybrid_cache_contract(tmp_path):
    record = inspect_artifact(artifact(tmp_path))
    assert (record["sliding_layers"], record["global_layers"]) == (21, 7)
    assert record["sliding_window"] == 1024
    assert record["supports_native_mtp"] is False
    config = mellum_config()
    config["num_experts_per_tok"] = 4
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="topology"):
        inspect_artifact(tmp_path)


def test_packed_quantized_expert_layout_is_admitted(tmp_path):
    config = mellum_config()
    config["quantization"] = {"group_size": 64, "bits": 4, "mode": "affine"}
    root = artifact(tmp_path, packed=True)
    (root / "config.json").write_text(json.dumps(config))
    assert inspect_artifact(root)["config"]["quantization"]["bits"] == 4


def test_layer_rope_template_and_shards_fail_closed(tmp_path):
    root = artifact(tmp_path)
    config = mellum_config()
    config["layer_types"][0] = "full_attention"
    (root / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="layer order"):
        inspect_artifact(root)

    artifact(root)
    (root / "chat_template.jinja").write_text("no reasoning contract")
    with pytest.raises(ValueError, match="chat template"):
        inspect_artifact(root)

    artifact(root)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    index["weight_map"]["lm_head.weight"] = "../outside.safetensors"
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="unsafe shard"):
        inspect_artifact(root)


def test_artifact_identity_covers_tokenizer_and_source_pin(tmp_path):
    root = artifact(tmp_path)
    before = inspect_artifact(root)["identity"]["fingerprint"]
    (root / "tokenizer.json").write_text('{"changed":true}')
    after = inspect_artifact(root)["identity"]["fingerprint"]
    assert after != before
    assert SOURCE_REVISION == "92ddae9fc7665e9f801d141d2e5a6b2caf2460c4"


def test_descriptor_reports_implemented_unqualified_ordinary_slice():
    assert MELLUM21_THINKING.cache_layout == CACHE_LAYOUT
    assert {
        Capability.TEXT,
        Capability.STREAMING,
        Capability.TOOLS,
        Capability.REASONING,
        Capability.CONTINUOUS_BATCH,
        Capability.APC_V2,
        Capability.LAYERED_CACHE,
        Capability.PROMPT_LOOKUP,
        Capability.GRAMMAR,
    } <= MELLUM21_THINKING.capabilities
    assert Capability.MTP not in MELLUM21_THINKING.capabilities
    assert MELLUM21_THINKING.metadata["qualification"] == "pending"
    defaults = MELLUM21_SAMPLING.profiles[MELLUM21_SAMPLING.general]
    assert (defaults.temperature, defaults.top_p, defaults.top_k) == (0.6, 0.95, 20)


def test_json_tool_output_and_thinking_channel():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "sum",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    adapter = object.__new__(Mellum21ThinkingAdapter)
    parser = adapter.output_parser(
        {"messages": [{"role": "user", "content": "x"}], "tools": tools}
    )
    events = parser.push(
        '<think>calculate</think><tool_call>{"name":"sum","arguments":{"x":3}}</tool_call>',
        final=True,
    )
    assert "".join(event.get("reasoning_content", "") for event in events) == "calculate"
    calls = [call for event in events for call in event.get("tool_calls", ())]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "sum"
    assert json.loads(calls[0]["function"]["arguments"]) == {"x": 3}


def test_thinking_defaults_on_for_chat_and_can_be_disabled():
    assert Mellum21ThinkingAdapter.thinking_enabled(
        {"messages": [{"role": "user", "content": "x"}]}
    )
    assert not Mellum21ThinkingAdapter.thinking_enabled(
        {
            "messages": [{"role": "user", "content": "x"}],
            "enable_thinking": False,
        }
    )
    assert not Mellum21ThinkingAdapter.thinking_enabled({"prompt": "x"})
