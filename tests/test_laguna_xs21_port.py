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
    assert resolved.default_route == "ordinary"
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


def _poolside_tools(schema):
    return [{"type": "function", "function": {"name": "f", "parameters": {
        "type": "object", "properties": {"x": schema},
    }}}]


def _poolside_call(value):
    # The template's layout for one argument.
    return f"<tool_call>f\n<arg_key>x</arg_key>\n<arg_value>{value}</arg_value>\n</tool_call>"


def _poolside_events(tools, text, split, *, tolerant=False):
    """Events the Laguna output parser emits for ``text``, pushed ``split``
    characters at a time."""
    from mlx2.output import OutputParser

    parser = OutputParser(
        chat=True, tools=tools, parse_tool=parse_tool_call,
        tolerant_tool_markers=tolerant,
    )
    events = []
    for start in range(0, len(text), split):
        events += parser.push(text[start : start + split])
    return parser, events + parser.push("", final=True)


@pytest.mark.parametrize(
    ("schema", "value"),
    [
        ({"type": "object"}, '{"s": "a</arg_value>b"}'),
        ({"type": "object"}, '{"s": "a</tool_call>b", "t": "</arg_value></tool_call>"}'),
        ({"type": "array"}, '["say \\"</tool_call>\\"", "</arg_value>\\\\"]'),
        ({"type": "array"}, '["<tool_call>g</tool_call>"]'),
    ],
)
def test_poolside_json_values_keep_quoted_closing_tags(schema, value):
    """The template writes non-string arguments with ``tojson``, which leaves
    ``<`` raw, so a JSON string may quote ``</arg_value>`` or
    ``</tool_call>``.  The parser cut the value at the first
    ``</arg_value>`` and served ``{"s": "a`` as a string, and a quoted
    ``</tool_call>`` cut the call, which was served without the argument.  A
    JSON value now ends at the first closer outside its strings, and a call
    whose text ends inside one runs on to the next ``</tool_call>``, however
    it is chunked."""
    tools = _poolside_tools(schema)
    text = _poolside_call(value)
    both = text + "\n" + text
    for split in (len(both), 1, 7):
        for sample, count in ((text, 1), (both, 2)):
            _, events = _poolside_events(tools, sample, split)
            calls = [c for e in events for c in e.get("tool_calls", ())]
            assert [json.loads(c["function"]["arguments"]) for c in calls] == [
                {"x": json.loads(value)}
            ] * count
            # The newline between two calls is template structure, not content.
            assert "".join(e.get("content", "") for e in events) == ""


@pytest.mark.parametrize("value", ["a</arg_value>b", "a</tool_call>b"])
def test_poolside_raw_values_that_spell_a_closer_fail_closed(value):
    """The template writes a string argument raw, so a closer inside it cannot
    be told from markup and the value still ends at the first one.  What
    followed was dropped: ``a</arg_value>b`` was served as ``a``, and
    ``a</tool_call>b`` as a call without arguments with the rest leaking into
    content.  Such a call is now malformed: an error, or all content under
    the tolerant fallback."""
    tools = _poolside_tools({"type": "string"})
    text = _poolside_call(value)
    for split in (len(text), 1, 7):
        with pytest.raises(ValueError):
            _poolside_events(tools, text, split)
        parser, events = _poolside_events(tools, text, split, tolerant=True)
        assert not any("tool_calls" in e for e in events)
        assert "".join(e.get("content", "") for e in events) == text
        assert parser.tool_call_parse_fallbacks == 1


@pytest.mark.parametrize("schema", [{"type": "object"}, {"description": "any value"}])
@pytest.mark.parametrize(
    ("value", "served"),
    [
        ("(1, 2)", "(1, 2)"),
        ("{1: 2}", "{1: 2}"),
        ("{'a': (1,)}", "{'a': (1,)}"),
        ("NaN", "NaN"),
        ("[Infinity]", "[Infinity]"),
        ("{1, 2}", "{1, 2}"),
        ("b'x'", "b'x'"),
        ("{'a': [1, None, True, 2.5]}", {"a": [1, None, True, 2.5]}),
    ],
)
def test_poolside_values_the_arguments_cannot_carry_stay_text(schema, value, served):
    """A non-string argument is decoded best-effort, as JSON and then as a
    Python literal.  A tuple or a key such as ``1`` decoded that way was
    served as what JSON makes of it (``[1, 2]``, ``{"1": 2}``), a value the
    model did not write, and a non-finite float, a set or bytes made the
    arguments unserializable, so the call failed.  Such text now stays the
    string the model wrote, however it is chunked; a literal made only of
    JSON types still decodes."""
    tools = _poolside_tools(schema)
    text = _poolside_call(value)
    for split in (len(text), 1, 7):
        _, events = _poolside_events(tools, text, split)
        calls = [c for e in events for c in e.get("tool_calls", ())]
        assert [json.loads(c["function"]["arguments"]) for c in calls] == [{"x": served}]


@pytest.mark.parametrize("schema", [{"type": "object"}, {"description": "any value"}])
@pytest.mark.parametrize("value", ["{[1]: 2}", "{1, [2]}"])
def test_poolside_literals_that_fail_to_build_stay_text(schema, value):
    """A Python literal with an unhashable key or set member raises
    ``TypeError`` while it is built, which the best-effort decode did not
    catch, so the call failed instead of serving the text.  It now stays the
    string the model wrote, however it is chunked."""
    tools = _poolside_tools(schema)
    text = _poolside_call(value)
    for split in (len(text), 1, 7):
        _, events = _poolside_events(tools, text, split)
        calls = [c for e in events for c in e.get("tool_calls", ())]
        assert [json.loads(c["function"]["arguments"]) for c in calls] == [{"x": value}]
