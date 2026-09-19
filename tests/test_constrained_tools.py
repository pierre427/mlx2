import pytest

from mlx2.adapters.muse_glimmer_output import (
    constrained_tool_grammar as muse_grammar,
    parse_atem,
)
from mlx2.adapters.north_output import constrained_tool_grammar as north_grammar
from mlx2.runtime.tool_parsers._schema import schema_value_matches
from mlx2.runtime.tool_parsers.qwen3_coder import (
    constrained_tool_grammar as qwen_grammar,
    parse_tool_call,
)
from mlx2.server import validate_request
from mlx2.structured_output import compile_constraint


def _tools(*, strict=True):
    return [{
        "type": "function",
        "function": {
            "name": "sum",
            "strict": strict,
            "parameters": {
                "$defs": {"count": {"type": "integer"}},
                "type": "object",
                "properties": {
                    "x": {"$ref": "#/$defs/count"},
                    "label": {"type": "string"},
                },
                "required": ["x"],
                "additionalProperties": False,
            },
        },
    }]


def _matches(grammar, text):
    return compile_constraint(grammar=grammar).fullmatch(text) is not None


def test_validation_accepts_required_and_named_tools_and_resolves_refs():
    base = {"messages": [{"role": "user", "content": "add"}], "tools": _tools()}
    required = validate_request(
        {**base, "tool_choice": "required", "parallel_tool_calls": False}
    )
    assert required["tool_choice"] == "required"
    assert required["parallel_tool_calls"] is False
    assert required["tools"][0]["function"]["parameters"]["properties"]["x"] == {
        "type": "integer"
    }
    named = validate_request({
        **base,
        "tool_choice": {"type": "function", "function": {"name": "sum"}},
    })
    assert named["tool_choice"]["function"]["name"] == "sum"


def test_non_strict_tool_reference_resolution_is_best_effort():
    messages = [{"role": "user", "content": "inspect"}]
    sibling = {
        "type": "function",
        "function": {
            "name": "sibling",
            "parameters": {
                "$defs": {"word": {"type": "string"}},
                "type": "object",
                "properties": {
                    "value": {
                        "$ref": "#/$defs/word",
                        "description": "annotation",
                    }
                },
            },
        },
    }
    recursive = {
        "type": "function",
        "function": {
            "name": "tree",
            "parameters": {
                "$defs": {
                    "node": {
                        "type": "object",
                        "properties": {
                            "children": {
                                "type": "array",
                                "items": {"$ref": "#/$defs/node"},
                            }
                        },
                    }
                },
                "type": "object",
                "properties": {"root": {"$ref": "#/$defs/node"}},
            },
        },
    }
    accepted = validate_request({"messages": messages, "tools": [sibling]})
    assert accepted["tools"][0]["function"]["parameters"] == sibling[
        "function"
    ]["parameters"]
    strict_sibling = {
        **sibling,
        "function": {**sibling["function"], "strict": True},
    }
    assert validate_request(
        {
            "messages": messages,
            "tools": [strict_sibling],
            "tool_choice": "required",
        }
    )
    accepted = validate_request({"messages": messages, "tools": [recursive]})
    assert "$defs" in accepted["tools"][0]["function"]["parameters"]

    recursive["function"]["strict"] = True
    with pytest.raises(ValueError, match="recursive JSON schema"):
        validate_request(
            {"messages": messages, "tools": [recursive], "tool_choice": "required"}
        )


@pytest.mark.parametrize(
    "schema,candidates",
    [
        ({"type": "string"}, ["", "x", '"quoted"']),
        ({"type": "string", "enum": ["red", "blue"]}, ["red", "blue", "green"]),
        ({"type": "string", "const": "fixed"}, ["fixed", "other"]),
        ({"type": "string", "minLength": 1, "maxLength": 3}, ["", "a", "abc", "abcd"]),
        ({"anyOf": [{"type": "string"}, {"type": "integer"}]}, ['"x"', "12", "true"]),
        ({"type": ["string", "null"]}, ['"hello"', "null", "12"]),
    ],
)
def test_adapter_strict_string_grammars_only_parse_schema_valid_values(
    schema, candidates
):
    import regex

    tools = [{
        "type": "function",
        "function": {
            "name": "f",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"value": schema},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }]
    # The HTTP boundary accepts every schema exercised by the adapter property
    # check, so direct grammar coverage cannot bypass stricter request rules.
    validate_request({
        "messages": [{"role": "user", "content": "call"}],
        "tools": tools,
        "tool_choice": "required",
        "parallel_tool_calls": False,
    })
    if "anyOf" in schema:
        with pytest.raises(ValueError, match="post-generation validator"):
            validate_request(
                {
                    "messages": [{"role": "user", "content": "call"}],
                    "tools": tools,
                    "tool_choice": "required",
                },
                constrained_tool_grammar=True,
            )
        with pytest.raises(ValueError, match="post-generation validator"):
            qwen_grammar(tools, "required", parallel_tool_calls=False)
        return
    qwen = qwen_grammar(tools, "required", parallel_tool_calls=False)
    muse = muse_grammar(tools, "required", parallel_tool_calls=False)
    for candidate in candidates:
        qwen_text = (
            "<tool_call>\n<function=f>\n<parameter=value>\n"
            + candidate
            + "\n</parameter>\n</function>\n</tool_call>"
        )
        if regex.fullmatch(qwen, qwen_text):
            parsed = parse_tool_call(
                qwen_text.split("<tool_call>\n", 1)[1].rsplit("\n</tool_call>", 1)[0],
                tools,
            )
            assert schema_value_matches(parsed["arguments"]["value"], schema)

        muse_text = (
            '<atem:function_calls><atem:invoke name="f">'
            '<atem:parameter name="value">'
            + candidate
            + "</atem:parameter></atem:invoke></atem:function_calls>"
        )
        if regex.fullmatch(muse, muse_text):
            parsed = parse_atem(
                muse_text[len("<atem:function_calls>") : -len("</atem:function_calls>")],
                tools,
            )[0]
            assert schema_value_matches(parsed["arguments"]["value"], schema)


def test_raw_string_patterns_fail_closed_for_strict_tool_adapters():
    tools = [{
        "type": "function",
        "function": {
            "name": "f",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string", "pattern": "^[a-z]+$"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }]
    request = {
        "messages": [{"role": "user", "content": "call"}],
        "tools": tools,
        "tool_choice": "required",
    }
    with pytest.raises(ValueError, match="unsupported JSON schema keywords"):
        validate_request(request)
    with pytest.raises(ValueError, match="pattern is unsupported"):
        qwen_grammar(tools, "required")
    with pytest.raises(ValueError, match="pattern is unsupported"):
        muse_grammar(tools, "required")


def test_qwen_strict_number_grammar_excludes_nonfinite_float_literals():
    tools = [{
        "type": "function",
        "function": {
            "name": "f",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "number"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }]
    grammar = qwen_grammar(tools, "required", parallel_tool_calls=False)

    def call(value):
        return (
            "<tool_call>\n<function=f>\n<parameter=value>\n"
            f"{value}\n</parameter>\n</function>\n</tool_call>"
        )

    assert _matches(grammar, call("1e99"))
    assert not _matches(grammar, call("1e400"))


def test_validation_rejects_unsafe_or_conflicting_tool_constraint_modes():
    base = {"messages": [{"role": "user", "content": "add"}], "tools": _tools()}
    assert validate_request(base)
    with pytest.raises(ValueError, match="declared function"):
        validate_request({
            **base,
            "tool_choice": {"type": "function", "function": {"name": "missing"}},
        })
    assert validate_request({
        **base,
        "tool_choice": "required",
        "response_format": {"type": "json_object"},
    })
    with pytest.raises(ValueError, match="parallel_tool_calls"):
        validate_request({**base, "tool_choice": "required", "parallel_tool_calls": 1})


def test_qwen_required_strict_grammar_uses_xml_wire_schema_and_parallel_bound():
    one = (
        "<tool_call>\n<function=sum>\n<parameter=x>\n3\n</parameter>"
        "\n<parameter=label>\nok\n</parameter>\n</function>\n</tool_call>"
    )
    grammar = qwen_grammar(_tools(), "required", parallel_tool_calls=False)
    assert _matches(grammar, one)
    assert not _matches(grammar, one.replace("\n3\n", "\nthree\n"))
    assert not _matches(grammar, one + "\n" + one)
    parallel = qwen_grammar(_tools(), "required", parallel_tool_calls=True)
    assert _matches(parallel, one + "\n" + one)


def test_named_grammars_cover_muse_atem_and_north_action_json():
    choice = {"type": "function", "function": {"name": "sum"}}
    muse = (
        '<atem:function_calls><atem:invoke name="sum">'
        '<atem:parameter name="x">4</atem:parameter>'
        '</atem:invoke></atem:function_calls>'
    )
    assert _matches(muse_grammar(_tools(), choice, parallel_tool_calls=False), muse)
    north = (
        '<|START_ACTION|>[{"tool_name":"sum","parameters":{"x":4}}]'
        '<|END_ACTION|>'
    )
    assert _matches(north_grammar(_tools(), choice, parallel_tool_calls=False), north)


def test_non_strict_north_required_grammar_keeps_recursive_json_arguments():
    grammar = north_grammar(_tools(strict=False), "required", parallel_tool_calls=False)
    text = (
        '<|START_ACTION|>[{"tool_name":"sum","parameters":'
        '{"nested":{"items":[1,true,null]}}}]<|END_ACTION|>'
    )
    assert _matches(grammar, text)
