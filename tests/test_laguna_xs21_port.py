"""Laguna XS 2.1 static/CPU adapter contracts."""

import json
import subprocess
import sys

import pytest

from mlx2.adapters.laguna_xs21 import (
    LAGUNA_XS21,
    LagunaXS21Adapter,
    inspect_artifact,
)
from mlx2.adapters.registry import inspect_model, resolve_adapter
from mlx2.contracts import Capability
from mlx2.runtime.tool_parsers.laguna import parse_tool_call


def laguna_config():
    return {
        "architectures": ["LagunaForCausalLM"],
        "model_type": "laguna",
        "vocab_size": 100352,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": 40,
        "num_attention_heads": 48,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "sliding_window": 512,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "mlp_only_layers": [0],
        "moe_routed_scaling_factor": 2.5,
        "norm_topk_prob": True,
        "tie_word_embeddings": False,
        "gating": "per-head",
        "max_position_embeddings": 262144,
        "layer_types": [
            "full_attention" if index % 4 == 0 else "sliding_attention"
            for index in range(40)
        ],
        "mlp_layer_types": ["dense", *("sparse" for _ in range(39))],
        "num_attention_heads_per_layer": [
            48 if index % 4 == 0 else 64 for index in range(40)
        ],
    }


def artifact(path):
    (path / "config.json").write_text(json.dumps(laguna_config()))
    shard = path / "model.safetensors"
    shard.write_bytes(b"static inspection does not load payload")
    keys = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.mlp.gate.proj.weight",
        "model.layers.1.mlp.gate.e_score_correction_bias",
        "model.layers.1.mlp.switch_mlp.gate_proj.weight",
        "model.layers.1.mlp.shared_expert.gate_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard.name for key in keys}})
    )
    return path


def test_registry_dispatch_is_metadata_only(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import mlx2.adapters.registry; assert 'mlx.core' not in sys.modules"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    resolved = inspect_model(artifact(tmp_path))
    assert resolved.adapter_type is LagunaXS21Adapter
    assert resolved.artifact["qualification"] == "pending"
    with pytest.raises(ValueError, match="native MTP"):
        resolve_adapter(tmp_path, mtp=True)


def test_exact_topology_and_shard_closure(tmp_path):
    record = inspect_artifact(artifact(tmp_path))
    assert (record["global_layers"], record["sliding_layers"]) == (10, 30)
    config = laguna_config()
    config["num_experts"] = 128
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="topology"):
        inspect_artifact(tmp_path)


def test_missing_and_traversing_shards_fail_closed(tmp_path):
    artifact(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="local safetensors"):
        inspect_artifact(tmp_path)
    outside = tmp_path.parent / "outside.safetensors"
    outside.write_bytes(b"x")
    artifact(tmp_path)
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    index["weight_map"]["lm_head.weight"] = "../outside.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="local safetensors"):
        inspect_artifact(tmp_path)


def test_applicable_capabilities_and_pending_boundary():
    assert {Capability.TEXT, Capability.TOOLS, Capability.REASONING,
            Capability.APC_V2, Capability.PROMPT_LOOKUP, Capability.GRAMMAR} <= LAGUNA_XS21.capabilities
    assert Capability.MTP not in LAGUNA_XS21.capabilities
    assert LAGUNA_XS21.metadata["qualification"] == "pending"


def test_poolside_tool_parser_preserves_declared_strings():
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {
        "type": "object", "properties": {"code": {"type": "string"}, "limit": {"type": "integer"}}
    }}}]
    parsed = parse_tool_call(
        "<tool_call>lookup\n<arg_key>code</arg_key><arg_value>001</arg_value>"
        "<arg_key>limit</arg_key><arg_value>2</arg_value></tool_call>", tools
    )
    assert parsed == {"name": "lookup", "arguments": {"code": "001", "limit": 2}}
