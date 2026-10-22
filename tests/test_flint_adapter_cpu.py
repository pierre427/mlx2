import importlib.util
import json
from pathlib import Path

import mlx.core as mx
import pytest

from mlx2.adapters import flint
from mlx2.adapters.registry import inspect_model
from mlx2.output import OutputParser
from mlx2.runtime.models.cache import KVCache, RotatingKVCache
from mlx2.runtime.models.flint import Model, ModelArgs
from mlx2.runtime.tool_parsers.laguna import parse_tool_call


def tiny_args():
    return ModelArgs.from_dict(
        {
            "model_type": "flint",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0,
            "layer_types": ["full_attention", "sliding_attention", "sliding_attention"],
            "no_rope_layer_indices": [0],
            "mlp_only_layers": [0],
            "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 12,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "sliding_window": 5,
            "tie_word_embeddings": False,
        }
    )


def artifact(tmp_path: Path, *, missing=None, quantized=False):
    config = dict(flint._EXPECTED)
    config.update(
        {
            "layer_types": list(flint._LAYER_TYPES),
            "no_rope_layer_indices": list(flint._FULL_LAYERS),
            "use_sliding_window": True,
            "attention_dropout": 0.0,
            "rms_norm_eps": 1e-6,
        }
    )
    if quantized:
        config["quantization"] = {"group_size": 64, "bits": 4, "mode": "affine"}
    weights = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for layer in range(49):
        prefix = f"model.layers.{layer}."
        weights.update(
            prefix + suffix
            for suffix in (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "self_attn.softmax_offset",
            )
        )
        if layer < 2:
            weights.update(
                prefix + f"mlp.{projection}_proj.weight"
                for projection in ("gate", "up", "down")
            )
        else:
            weights.update(
                {
                    prefix + "mlp.gate.weight",
                    prefix + "mlp.shared_expert.gate_proj.weight",
                    prefix + "mlp.shared_expert.up_proj.weight",
                    prefix + "mlp.shared_expert.down_proj.weight",
                }
            )
            for expert in range(256):
                weights.update(
                    prefix + f"mlp.experts.{expert}.{projection}_proj.weight"
                    for projection in ("gate", "up", "down")
                )
    if missing:
        weights.remove(missing)
    if quantized:
        weights.update({"model.embed_tokens.scales", "model.embed_tokens.biases"})
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {key: "model.safetensors" for key in weights},
            }
        )
    )
    (tmp_path / "model.safetensors").write_bytes(b"fixture")
    return tmp_path


def test_inspector_and_registry_admit_exact_pinned_topology(tmp_path):
    result = inspect_model(artifact(tmp_path))
    assert result.adapter_type is flint.FlintAdapter
    assert result.descriptor.family == "flint"
    assert result.artifact["source_revision"] == flint.SOURCE_REVISION
    assert result.artifact["source_layout"] == "hf-per-expert"
    assert result.artifact["qualification"] == "pending"


def test_inspector_fails_closed_on_missing_numerical_law_tensor(tmp_path):
    with pytest.raises(ValueError, match="incomplete"):
        flint.inspect_artifact(
            artifact(tmp_path, missing="model.layers.17.self_attn.softmax_offset")
        )


def test_tiny_forward_and_hybrid_cache_contract():
    model = Model(tiny_args())
    cache = model.make_cache()
    assert [type(item) for item in cache] == [KVCache, RotatingKVCache, RotatingKVCache]
    assert [getattr(item, "max_size", None) for item in cache] == [None, 5, 5]
    logits = model(mx.array([[1, 2, 3]]), cache)
    mx.eval(logits)
    assert logits.shape == (1, 3, 32)
    contract = model.exact_prefix_reuse_contract(cache)
    assert contract == {
        "algorithm": "flint-full-swa-segmented-kv-v1",
        "full_layers": 1,
        "full_cache": {"kind": "unbounded", "max_size": None, "keep": 0},
        "sliding_layers": 2,
        "sliding_window": 5,
        "committed_offset": 3,
        "transaction": "SegmentedKVRows exact rotating-window snapshot/replay",
    }
    bounded = model.exact_prefix_reuse_contract(model.make_cache(max_kv_size=17))
    assert bounded["full_cache"] == {
        "kind": "rotating",
        "max_size": 17,
        "keep": 0,
    }


def test_tiny_incremental_decode_matches_full_hybrid_attention_after_rollover():
    mx.random.seed(0)
    model = Model(tiny_args())
    mx.eval(model.parameters())
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    full = model(tokens)
    cache = model.make_cache()
    incremental = mx.concatenate(
        [model(tokens[:, position : position + 1], cache) for position in range(8)],
        axis=1,
    )
    mx.eval(full, incremental)
    assert mx.allclose(full, incremental, atol=1e-5, rtol=1e-5).item()
    assert [entry.offset for entry in cache] == [8, 8, 8]


def test_quantized_expert_metadata_packs_in_expert_order():
    args = tiny_args()
    model = Model(args)
    weights = {}
    for layer in (1, 2):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            out = 8 if projection != "down_proj" else 16
            width = 16 if projection != "down_proj" else 8
            for expert in range(args.num_experts):
                base = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
                weights[base + ".weight"] = mx.full(
                    (out, width // 8), expert, mx.uint32
                )
                weights[base + ".scales"] = mx.full(
                    (out, width // 8), expert + 1, mx.bfloat16
                )
                weights[base + ".biases"] = mx.full(
                    (out, width // 8), expert + 2, mx.bfloat16
                )
    packed = model.sanitize(weights)
    key = "model.layers.1.mlp.switch_mlp.gate_proj.scales"
    assert packed[key].shape[0] == args.num_experts
    assert packed[key][3, 0, 0].item() == 4
    assert not any(".experts." in key for key in packed)


def test_optional_reasoning_and_matching_tool_wire_format():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            },
        }
    ]
    parser = OutputParser(
        chat=True, thinking=False, tools=tools, parse_tool=parse_tool_call
    )
    events = parser.push(
        "<think>check first</think>answer<tool_call>lookup"
        "<arg_key>q</arg_key><arg_value>flint</arg_value></tool_call>",
        final=True,
    )
    assert events[0] == {"reasoning_content": "check first"}
    assert events[1] == {"content": "answer"}
    call = events[2]["tool_calls"][0]
    assert call["function"]["name"] == "lookup"
    assert json.loads(call["function"]["arguments"]) == {"q": "flint"}


def test_provenance_pins_released_source():
    root = Path(__file__).resolve().parents[1]
    path = root / "provenance" / "flint.json"
    if not path.is_file():
        pytest.skip("private material absent: Flint provenance (not exported to the public mirror)")
    record = json.loads(path.read_text())
    assert record["source_revision"] == flint.SOURCE_REVISION
    assert record["source_index_sha256"] == flint.SOURCE_INDEX_SHA256
    assert record["source_wheel_sha256"] == flint.SGLANG_WHEEL_SHA256
    assert record["status"] == {
        "implemented": True,
        "qualified": False,
        "selected": False,
        "observed_used": True,
        "performance_claim": False,
    }


def test_oq4e_profile_protects_routing_and_uses_fidelity_tiers():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "convert_flint", root / "scripts" / "convert_flint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    matrix = mx.zeros((64, 64), dtype=mx.bfloat16)
    vector = mx.zeros((64,), dtype=mx.bfloat16)
    assert module._policy("model.layers.7.mlp.experts.3.gate_proj.weight", matrix) == (
        4,
        "routed_experts_q4",
    )
    assert module._policy(
        "model.layers.7.mlp.shared_expert.up_proj.weight", matrix
    ) == (
        5,
        "shared_expert_q5",
    )
    assert module._policy("model.layers.7.self_attn.q_proj.weight", matrix) == (
        6,
        "attention_q6",
    )
    assert module._policy("model.layers.7.mlp.gate.weight", matrix) == (
        None,
        "protected_bf16",
    )
    assert module._policy("model.layers.7.self_attn.softmax_offset", vector) == (
        None,
        "protected_bf16",
    )


@pytest.mark.parametrize("chunk", [3, 7, 16, 33])
def test_chunked_prefill_then_decode_matches_full_forward_past_the_window(chunk):
    # Chunks shorter and longer than the sliding window, then one-token
    # decode, against one cache-free forward with the window in the mask.
    mx.random.seed(0)
    model = Model(tiny_args())
    mx.eval(model.parameters())
    tokens = [(index * 7 + 3) % 31 + 1 for index in range(40)]
    full = model(mx.array([tokens]))
    cache = model.make_cache()
    prompt = 30
    rows = [
        model(mx.array([tokens[start : min(start + chunk, prompt)]]), cache=cache)
        for start in range(0, prompt, chunk)
    ]
    rows += [model(mx.array([[token]]), cache=cache) for token in tokens[prompt:]]
    incremental = mx.concatenate(rows, axis=1)
    mx.eval(full, incremental)
    assert mx.allclose(full, incremental, atol=1e-5, rtol=1e-5).item()
