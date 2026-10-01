"""Equivalent tool schemas must render the same prompt (omlx #4138).

Agent clients (Trae in the report) resend the same tool definitions each turn
with JSON object members in a different order: ``required`` before
``properties``, properties listed in another order, ``parameters`` before
``name``.  Chat templates serialize tools with ``tojson``, which keeps member
order, so the prompt diverged inside the tool block and APCv2 re-prefilled the
whole conversation.  ``validate_request`` now canonicalizes each tool's member
order once, so the rendered prompt and the strict-tool grammar see one object.
"""

import copy
import json
from pathlib import Path

import pytest

from mlx2.server import validate_request

MODELS = Path.home() / "mlx-models"


def _schema_a():
    return {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search text"},
            "filters": {
                "type": "array",
                "description": "Optional filters",
                "items": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer", "description": "Days ahead"},
                    },
                    "required": ["city", "days"],
                    "additionalProperties": False,
                },
            },
            "mode": {"type": "string", "enum": ["fast", "exact", "auto"]},
        },
        "required": ["query"],
        "additionalProperties": False,
    }


def _schema_b():
    # The same schema with every object's members in another order.
    return {
        "additionalProperties": False,
        "required": ["query"],
        "properties": {
            "mode": {"enum": ["fast", "exact", "auto"], "type": "string"},
            "filters": {
                "items": {
                    "additionalProperties": False,
                    "required": ["city", "days"],
                    "properties": {
                        "days": {"description": "Days ahead", "type": "integer"},
                        "city": {"type": "string"},
                    },
                    "type": "object",
                },
                "description": "Optional filters",
                "type": "array",
            },
            "query": {"description": "Search text", "type": "string"},
        },
        "type": "object",
    }


def _tools(reordered, strict=False):
    if not reordered:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the index",
                    "parameters": _schema_a(),
                    **({"strict": True} if strict else {}),
                },
            },
            {
                "type": "function",
                "function": {"name": "ping", "parameters": {"type": "object"}},
            },
        ]
    return [
        {
            "function": {
                **({"strict": True} if strict else {}),
                "parameters": _schema_b(),
                "description": "Search the index",
                "name": "search",
            },
            "type": "function",
        },
        {
            "function": {"parameters": {"type": "object"}, "name": "ping"},
            "type": "function",
        },
    ]


def _body(tools):
    return {
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "find the weather in Paris"},
        ],
        "tools": tools,
        "enable_thinking": False,
    }


def test_reordered_tools_validate_to_identical_serialization():
    first = validate_request(_body(_tools(False)))["tools"]
    second = validate_request(_body(_tools(True)))["tools"]
    assert json.dumps(first) == json.dumps(second)


@pytest.mark.parametrize("reordered", [False, True])
def test_canonical_order_preserves_values_lists_and_defaults(reordered):
    original = _tools(reordered)
    snapshot = json.dumps(original)
    tools = validate_request(_body(original))["tools"]
    # The caller's objects are not mutated, member order included.
    assert json.dumps(original) == snapshot
    # Same values (dict equality ignores member order).
    expected = _tools(False)
    expected[1]["function"]["description"] = ""
    assert tools == expected
    # Tool-list order and array order are kept.
    assert [tool["function"]["name"] for tool in tools] == ["search", "ping"]
    parameters = tools[0]["function"]["parameters"]
    assert parameters["properties"]["mode"]["enum"] == ["fast", "exact", "auto"]
    assert parameters["properties"]["filters"]["items"]["required"] == ["city", "days"]
    # The wrapper keeps the conventional member order templates were trained on.
    assert list(tools[0]) == ["type", "function"]
    assert list(tools[0]["function"]) == ["name", "description", "parameters"]
    assert list(tools[1]["function"]) == ["name", "description", "parameters"]
    # Schema keywords are sorted; required properties precede optional ones.
    assert list(parameters) == ["additionalProperties", "properties", "required", "type"]
    assert list(parameters["properties"]) == ["query", "filters", "mode"]
    assert list(parameters["properties"]["filters"]["items"]["properties"]) == [
        "city",
        "days",
    ]


def test_canonical_tool_definition_unit():
    from mlx2.server import canonical_tool_definition

    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "$defs": {
                    "Zone": {
                        "type": "object",
                        "properties": {"tz": {"type": "string"}, "name": {"type": "string"}},
                        "required": ["tz"],
                    },
                    "Alpha": {"type": "string"},
                },
                "properties": {
                    "zone": {"$ref": "#/$defs/Zone"},
                    "alpha": {"$ref": "#/$defs/Alpha", "default": {"b": 1, "a": [3, 1, 2]}},
                    # A property literally named like a schema keyword.
                    "properties": {"type": "string"},
                },
                "required": ["zone", "properties"],
                "anyOf": [{"required": ["zone"]}, {"required": ["alpha"]}],
            },
            "x-vendor": {"z": 1, "a": 2},
        },
    }
    snapshot = copy.deepcopy(tool)
    canonical = canonical_tool_definition(tool)
    assert tool == snapshot and json.dumps(tool) == json.dumps(snapshot)
    assert canonical == tool
    function = canonical["function"]
    assert list(function) == ["name", "parameters", "x-vendor"]
    assert list(function["x-vendor"]) == ["a", "z"]
    parameters = function["parameters"]
    assert list(parameters) == ["$defs", "anyOf", "properties", "required", "type"]
    assert list(parameters["$defs"]) == ["Alpha", "Zone"]
    assert list(parameters["$defs"]["Zone"]["properties"]) == ["tz", "name"]
    # Required names first (sorted), then optional ones (sorted).
    assert list(parameters["properties"]) == ["properties", "zone", "alpha"]
    assert list(parameters["properties"]["alpha"]) == ["$ref", "default"]
    assert list(parameters["properties"]["alpha"]["default"]) == ["a", "b"]
    assert parameters["properties"]["alpha"]["default"]["a"] == [3, 1, 2]
    assert parameters["required"] == ["zone", "properties"]
    assert parameters["anyOf"] == [{"required": ["zone"]}, {"required": ["alpha"]}]
    # Idempotent.
    assert json.dumps(canonical_tool_definition(canonical)) == json.dumps(canonical)


def test_strict_tool_with_optional_property_sorting_before_required_stays_valid():
    # Plain alphabetical order would put optional ``limit`` before required
    # ``query``, which the constrained-output compiler rejects.
    def tools(reordered):
        properties = {"query": {"type": "string"}, "limit": {"type": "integer"}}
        if reordered:
            properties = dict(reversed(list(properties.items())))
        return [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    first = validate_request(_body(tools(False)))["tools"]
    second = validate_request(_body(tools(True)))["tools"]
    assert json.dumps(first) == json.dumps(second)
    assert list(first[0]["function"]["parameters"]["properties"]) == ["query", "limit"]


def test_strict_tool_grammar_compiles_from_canonical_schema():
    from mlx2.structured_output import compile_constraint

    tools = validate_request(_body(_tools(True, strict=True)))["tools"]
    parameters = tools[0]["function"]["parameters"]
    compile_constraint(
        {
            "type": "json_schema",
            "json_schema": {"name": "search", "strict": True, "schema": parameters},
        }
    )


def _anthropic_tools(reordered):
    schema = _schema_b() if reordered else _schema_a()
    tool = {"name": "search", "description": "Search the index", "input_schema": schema}
    if reordered:
        tool = dict(reversed(list(tool.items())))
    return [tool]


def test_anthropic_and_responses_tools_reach_canonical_order():
    from mlx2.anthropic_compat import anthropic_request_to_chat
    from mlx2.openai_compat import responses_to_chat_request

    def anthropic(reordered):
        chat = anthropic_request_to_chat(
            {
                "model": "fixture",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": _anthropic_tools(reordered),
            }
        )
        return validate_request(chat)["tools"]

    assert json.dumps(anthropic(False)) == json.dumps(anthropic(True))

    def responses(reordered):
        schema = _schema_b() if reordered else _schema_a()
        tool = {
            "type": "function",
            "name": "search",
            "description": "Search the index",
            "parameters": schema,
        }
        if reordered:
            tool = dict(reversed(list(tool.items())))
        body, _ = responses_to_chat_request(
            {"model": "fixture", "input": "hi", "tools": [tool]}
        )
        return validate_request(body)["tools"]

    assert json.dumps(responses(False)) == json.dumps(responses(True))


def _template_adapter(name):
    root = MODELS / name
    if not (root / "tokenizer_config.json").is_file():
        pytest.skip(f"{name} is not available locally")
    from transformers import AutoTokenizer

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.tokenizer_integrity import repair_loaded_tokenizer
    from mlx2.runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper

    adapter_class = resolve_adapter(root)
    adapter = object.__new__(adapter_class)
    tokenizer = AutoTokenizer.from_pretrained(
        root, local_files_only=True, trust_remote_code=False
    )
    repair_loaded_tokenizer(tokenizer, root)
    # The adapters' own load wraps the tokenizer the same way.
    adapter.tokenizer = TokenizerWrapper(
        tokenizer, detokenizer_class=BPEStreamingDetokenizer, eos_token_ids=[]
    )
    return adapter


@pytest.mark.parametrize(
    "name",
    [
        "Qwen3.8-Flash-Next-MLX-4bit-MTP",
        "Qwen3.6-35B-A3B-MLX-4bit-uniform",
        "North-Mini-Code-1.0-mlx-4bit",
    ],
)
@pytest.mark.parametrize("strict", [False, True])
def test_reordered_tools_render_identical_prompt_tokens(name, strict):
    adapter = _template_adapter(name)
    raw = [_body(_tools(reordered, strict)) for reordered in (False, True)]
    # Falsifier: without canonicalization the template renders differ.
    raw_tokens = [list(adapter.prompt_tokens(body)) for body in raw]
    assert raw_tokens[0] != raw_tokens[1]
    tokens = [list(adapter.prompt_tokens(validate_request(body))) for body in raw]
    assert tokens[0] == tokens[1]
    # The canonical tool block is still rendered in full.
    text = adapter.render_prompt(validate_request(raw[1]))
    assert "Search the index" in text and '"days"' in text
