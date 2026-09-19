"""CPU-only port contracts. Never import the tensor model or MLX."""

import ast
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from mlx2.adapters.muse_glimmer import (
    MUSE_GLIMMER,
    MuseGlimmerAdapter,
    inspect_artifact,
    normalize_messages,
)
from mlx2.adapters.muse_glimmer_config import ModelArgs
from mlx2.adapters.muse_glimmer_output import MuseOutputParser, parse_atem
from mlx2.contracts import Capability


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "functions.echo",
            "parameters": {
                "type": "object",
                "required": ["text"],
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string"},
                    "count": {"type": "integer"},
                    "flag": {"type": "boolean"},
                    "data": {"type": "object"},
                },
            },
        },
    }
]
ATEM = '<atem:invoke name="functions.echo"><atem:parameter name="text">  a<b  </atem:parameter><atem:parameter name="count">2</atem:parameter></atem:invoke>'


def collect(text, split=1, **kwargs):
    parser = MuseOutputParser(chat=True, **kwargs)
    events = []
    for start in range(0, len(text), split):
        events.extend(parser.push(text[start : start + split]))
    events.extend(parser.push("", final=True))
    return parser, events


def test_import_has_no_tensor_side_effects():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import mlx2.adapters.muse_glimmer; import mlx2.adapters.muse_glimmer_output; assert 'mlx.core' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_default_topology_and_layout():
    args = ModelArgs()
    assert args.layer_types.count("sliding_attention") == 39
    assert args.layer_types.count("full_attention") == 13
    assert args.sliding_window == 2048
    assert ModelArgs(sliding_window=1024).cache_layout != args.cache_layout


@pytest.mark.parametrize(
    "kwargs",
    [
        {"layer_types": ["full_attention"]},
        {"hidden_activation": "gelu"},
        {"sliding_window": 0},
        {"num_hidden_layers": 0},
        {"layer_rope_theta": [1.0] * 52},
        {"layer_types": ["unknown"] * 52},
        {"final_logit_softcapping": 0},
        {"num_attention_heads": 31},
    ],
)
def test_invalid_topology_fails_closed(kwargs):
    with pytest.raises(ValueError):
        ModelArgs(**kwargs)


def test_nested_configuration_is_target_identity():
    args = ModelArgs.from_dict(
        {
            "model_type": "muse_glimmer",
            "text_config": {
                "model_type": "muse_glimmer_text",
                "num_hidden_layers": 4,
            },
        }
    )
    assert args.model_type == "muse_glimmer"
    assert args.layer_types == ["sliding_attention"] * 3 + ["full_attention"]


def test_inspector_rejects_drafter_and_missing_weights(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "dflash_config": {}})
    )
    with pytest.raises(ValueError, match="target artifact"):
        inspect_artifact(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "muse_glimmer"}))
    with pytest.raises(FileNotFoundError):
        inspect_artifact(tmp_path)


def test_identity_bound_to_topology_and_tokenizer(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "muse_glimmer"}))
    (tmp_path / "model.safetensors").write_bytes(b"metadata-only fixture")
    (tmp_path / "tokenizer.json").write_text("{}")
    first = inspect_artifact(tmp_path)
    (tmp_path / "tokenizer.json").write_text('{"changed":true}')
    assert inspect_artifact(tmp_path)["fingerprint"] != first["fingerprint"]
    assert first["qualification"] == "pending"


@pytest.mark.parametrize("weights", [{}, [], None])
def test_empty_or_invalid_index_rejected_before_tensor_load(tmp_path, weights):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "muse_glimmer"}))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weights})
    )
    with pytest.raises(ValueError, match="nonempty weight index"):
        inspect_artifact(tmp_path)


def test_direct_inspection_rejects_drafter_metadata(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "muse_glimmer", "dflash_config": {}})
    )
    with pytest.raises(ValueError, match="drafter"):
        inspect_artifact(tmp_path)


def test_qualification_not_inferred_and_self_mtp_refused():
    assert MUSE_GLIMMER.metadata["qualification"] == "pending"
    assert Capability.MTP not in MUSE_GLIMMER.capabilities
    assert Capability.SEGMENTED_MTP not in MUSE_GLIMMER.capabilities
    assert Capability.APC_V2 in MUSE_GLIMMER.capabilities
    assert MuseGlimmerAdapter.profile_name(False) == "muse-glimmer-apcv2-ordinary"
    with pytest.raises(ValueError, match="DFlash2"):
        MuseGlimmerAdapter.profile_name(True)


def test_execution_policy_is_ordinary_and_no_speculation():
    adapter = MuseGlimmerAdapter.__new__(MuseGlimmerAdapter)
    config = adapter.execution_config(max_lanes=4, prefill_step=2048)
    assert config["num_draft"] == 0
    assert config["segment_aware_live_tip"] is False


def test_execution_policy_rejected_before_artifact_or_tensor_loading():
    with pytest.raises(ValueError, match="policy overrides"):
        MuseGlimmerAdapter("/not/a/model", execution_policy={"num_draft": 2})


def test_tool_history_is_normalized_without_mutation():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "functions.echo",
                        "arguments": '{"text":"hello"}',
                    }
                }
            ],
        }
    ]
    before = copy.deepcopy(messages)
    normalized = normalize_messages(messages)
    assert normalized[0]["tool_calls"][0]["function"]["arguments"] == {"text": "hello"}
    assert messages == before


def test_direct_answer_and_effort_mapping_are_native_prompt_policy():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            self.strength = kwargs["reasoning_strength"]
            assert kwargs["tokenize"] is False
            return "<|start|>assistant"

        def encode(self, prompt, **kwargs):
            self.prompt = prompt
            return [1, 2]

    adapter = MuseGlimmerAdapter.__new__(MuseGlimmerAdapter)
    adapter.tokenizer = Tokenizer()
    adapter.prompt_tokens({"messages": []})
    assert adapter.tokenizer.strength == "low"
    assert adapter.tokenizer.prompt.endswith(" to=user<|message|>")
    adapter.prompt_tokens({"messages": [], "reasoning_effort": "none"})
    assert adapter.tokenizer.strength == "low"
    assert adapter.tokenizer.prompt.endswith(" to=user<|message|>")
    adapter.prompt_tokens({"messages": [], "enable_thinking": True})
    assert adapter.tokenizer.strength == "high"
    assert adapter.tokenizer.prompt == "<|start|>assistant"
    adapter.prompt_tokens({"messages": [], "reasoning_effort": "ultra"})
    assert adapter.tokenizer.strength == "high"
    assert adapter.tokenizer.prompt == "<|start|>assistant"


def test_multimodal_content_refused():
    with pytest.raises(ValueError, match="text only"):
        normalize_messages([{"content": [{"type": "image_url", "image_url": "x"}]}])


@pytest.mark.parametrize("split", [1, 2, 3, 11, 4096])
def test_recipient_channels_are_chunk_boundary_safe(split):
    _, events = collect(
        "to=self<|message|>reason<|eom|><|start|>assistant to=user<|message|>answer<|eot|>",
        split=split,
    )
    assert "".join(e.get("reasoning_content", "") for e in events) == "reason"
    assert "".join(e.get("content", "") for e in events) == "answer"


def test_plain_content_and_bare_user_header():
    for raw, expected in [
        ("Hi.", "Hi."),
        ("to=user<|message|>Hi.", "Hi."),
        ("to=do list: Hi.", "to=do list: Hi."),
    ]:
        _, events = collect(raw)
        assert "".join(e.get("content", "") for e in events) == expected


def test_atem_tools_chunked_and_string_whitespace_preserved():
    parser, events = collect(
        "to=functions.echo<|message|><atem:function_calls>"
        + ATEM
        + "</atem:function_calls><|eot|>",
        tools=TOOLS,
    )
    assert parser.tool_count == 1
    tool = next(e["tool_calls"][0] for e in events if "tool_calls" in e)
    assert tool["function"]["name"] == "functions.echo"
    assert json.loads(tool["function"]["arguments"]) == {"text": "  a<b  ", "count": 2}


def test_muse_auto_parallel_false_rejects_a_second_call():
    body = (
        "<atem:function_calls>"
        + ATEM
        + ATEM
        + "</atem:function_calls>"
    )
    with pytest.raises(ValueError, match="at most one"):
        collect(body, tools=TOOLS, parallel_tool_calls=False)


@pytest.mark.parametrize(
    "body",
    [
        ATEM.replace("functions.echo", "functions.other"),
        ATEM.replace(">2</atem:parameter>", ">true</atem:parameter>"),
        ATEM.replace('name="count"', 'name="text"'),
        "bad" + ATEM,
        ATEM + "bad",
        "",
    ],
)
def test_malformed_or_undeclared_tool_calls_fail_closed(body):
    with pytest.raises(ValueError):
        parse_atem(body, TOOLS)


def test_stop_spans_chunk_boundary():
    parser, events = collect("to=user<|message|>helloSTOPignored", stops=["STOP"])
    assert parser.stopped
    assert "".join(e.get("content", "") for e in events) == "hello"


def test_incomplete_tool_and_header_fail_closed():
    for raw in [
        "to=user<|mess",
        "to=functions.echo<|message|><atem:function_calls>" + ATEM,
    ]:
        with pytest.raises(ValueError):
            collect(raw, tools=TOOLS)


def test_tensor_source_is_standalone_and_ordinary_cache_topology():
    path = Path(__file__).parents[1] / "src/mlx2/runtime/models/muse_glimmer.py"
    tree = ast.parse(path.read_text())
    imports = [
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    assert not any(
        module and module.startswith(("mlx_lm", "mlx_vlm")) for module in imports
    )
    assert {"RotatingKVCache", "KVCache"} <= {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
