"""Token-exhausted tool identities obey the same contracts across grammars."""

import json

import pytest

from mlx2.adapters.muse_glimmer_output import MuseOutputParser
from mlx2.adapters.north_output import NorthOutputParser
from mlx2.adapters.xing_output import XingOutputParser, parse_tool_block
from mlx2.openai_compat import ToolContractError, enforce_tool_contract
from mlx2.output import OutputParser
from mlx2.runtime.tool_parsers.laguna import parse_tool_call as laguna_parse
from mlx2.runtime.tool_parsers.qwen3_coder import parse_tool_call as qwen_parse


def tools(*names):
    return [
        {"type": "function", "function": {"name": name, "parameters": {}}}
        for name in names
    ]


def named(name):
    return {"type": "function", "function": {"name": name}}


def parser_and_partial(family, name, definitions):
    if family == "north":
        parser = NorthOutputParser(chat=True, thinking=False, tools=definitions)
        text = (
            '<|START_ACTION|>[{"tool_name":' + json.dumps(name) + ',"parameters":{"q":"'
        )
    elif family == "muse":
        parser = MuseOutputParser(chat=True, tools=definitions)
        # ATEM names use markup attributes rather than JSON escaping.
        text = (
            'to=user<|message|><atem:function_calls><atem:invoke name="'
            + name
            + '"><atem:parameter name="q">'
        )
    else:
        parse = (
            qwen_parse
            if family == "qwen"
            else laguna_parse
            if family == "laguna"
            else parse_tool_block
        )
        if family == "xing":
            parser = XingOutputParser(chat=True, thinking=False, tools=definitions)
        else:
            parser = OutputParser(
                chat=True, tools=definitions, parse_tool=parse, constrained_tools=True
            )
        if family == "qwen":
            text = "<tool_call><function=" + name + "><parameter=q>"
        elif family == "laguna":
            text = "<tool_call>" + name + "<arg_key>q</arg_key><arg_value>"
        elif family in ("xing-json", "xing"):
            text = '<tool_call>{"name":' + json.dumps(name) + ',"arguments":{"q":"'
        else:
            text = "<tool_call>" + name + "<param_key>q</param_key><param_value>"
    return parser, text


FAMILIES = ["qwen", "xing-plain", "xing-json", "xing", "laguna", "muse", "north"]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "name", ["weather", "news", "undeclared", "db:lookup", "a/b", "my tool", "café"]
)
@pytest.mark.parametrize("split", [1, 7, 1000])
def test_length_cut_records_grammar_owned_name_and_judges_contract(family, name, split):
    definitions = tools("weather", "news", "db:lookup", "a/b", "my tool", "café")
    parser, text = parser_and_partial(family, name, definitions)
    events = []
    for offset in range(0, len(text), split):
        events.extend(parser.push(text[offset : offset + split]))
    events.extend(parser.finish("", "length"))
    assert not any(event.get("tool_calls") for event in events)
    record = parser.truncated_tool_call
    unreadable = family == "muse" and name in ("db:lookup", "a/b", "my tool")
    committed = "my" if family == "qwen" and name == "my tool" else name
    expected = (
        {"function": None, "name_unreadable": True}
        if unreadable
        else {"function": committed}
    )
    assert record == {**expected, "cause": "max_tokens"}
    assert getattr(parser, "stop_truncated_tool_call", None) is None
    body = {"tools": definitions, "tool_choice": named("weather")}
    if name == "weather":
        enforce_tool_contract(
            body, [], finish_reason="length", truncated_tool_call=record
        )
    else:
        with pytest.raises(ToolContractError):
            enforce_tool_contract(
                body, [], finish_reason="length", truncated_tool_call=record
            )
    if name == "undeclared" or committed == "my" or unreadable:
        with pytest.raises(ToolContractError):
            enforce_tool_contract(
                {"tools": definitions},
                [],
                finish_reason="length",
                truncated_tool_call=record,
            )
    else:
        enforce_tool_contract(
            {"tools": definitions},
            [],
            finish_reason="length",
            truncated_tool_call=record,
        )
    with pytest.raises(ToolContractError):
        enforce_tool_contract(
            {"tools": definitions, "tool_choice": "none"},
            [],
            finish_reason="length",
            truncated_tool_call=record,
        )


@pytest.mark.parametrize("family", FAMILIES)
def test_cut_after_opener_has_incomplete_evidence_without_fulfilling_a_call(family):
    parser, text = parser_and_partial(family, "weather", tools("weather"))
    opener = (
        "<|START_ACTION|>"
        if family == "north"
        else "<atem:function_calls>"
        if family == "muse"
        else "<tool_call>"
    )
    text = text[: text.index(opener) + len(opener)]
    events = parser.finish(text, "length")
    assert not any(event.get("tool_calls") for event in events)
    assert parser.tool_count == 0
    assert parser.truncated_tool_call == {"function": None, "cause": "max_tokens"}
    enforce_tool_contract(
        {"tools": tools("weather"), "tool_choice": "required"},
        [],
        finish_reason="length",
        truncated_tool_call=parser.truncated_tool_call,
    )


@pytest.mark.parametrize("family", ["xing-json", "xing", "north"])
@pytest.mark.parametrize("name", ['say"hi"', "a\\\\b", "雪"])
def test_json_name_escaping_uses_decoded_identity(family, name):
    parser, text = parser_and_partial(family, name, tools(name))
    parser.finish(text, "length")
    assert parser.truncated_tool_call["function"] == name


@pytest.mark.parametrize(
    "payload, record",
    [
        (
            '[{"tool_name":"weather","parameters":{"tool_name":"news","x":"',
            {"function": "weather"},
        ),
        ('[{"parameters":{"tool_name":"news","x":"', {"function": None}),
        (
            '[{"tool_name":"weather","parameters":{}},{"tool_name":"news","parameters":',
            {"function": "weather", "functions": ["weather", "news"]},
        ),
        (
            '[{"tool_name":"weather","parameters":{}},{"tool_name":"ne',
            {"function": "weather"},
        ),
        (
            '[{"tool_name":"weather","tool_name":"ne',
            {"function": None, "name_unreadable": True},
        ),
        ('[{"tool_name":7,', {"function": None, "name_unreadable": True}),
    ],
)
def test_north_nested_members_array_siblings_and_unreadable_names(payload, record):
    parser = NorthOutputParser(
        chat=True, thinking=False, tools=tools("weather", "news")
    )
    assert parser.finish("<|START_ACTION|>" + payload, "length") == []
    assert parser.truncated_tool_call == {**record, "cause": "max_tokens"}
    if record.get("name_unreadable"):
        with pytest.raises(ToolContractError, match="could not be read"):
            enforce_tool_contract(
                {"tools": tools("weather", "news")},
                [],
                finish_reason="length",
                truncated_tool_call=parser.truncated_tool_call,
            )


@pytest.mark.parametrize(
    "finish, record",
    [
        ("length", {"function": "weather", "cause": "stop_sequence"}),
        ("stop", {"function": "weather", "cause": "max_tokens"}),
        ("tool_calls", {"function": "weather", "cause": "max_tokens"}),
        ("length", {"function": "weather"}),
        ("length", []),
    ],
)
def test_receipt_cause_must_match_terminal_finish(finish, record):
    with pytest.raises(ToolContractError, match="contradicts"):
        enforce_tool_contract(
            {"tools": tools("weather")},
            [],
            finish_reason=finish,
            truncated_tool_call=record,
        )


def test_legacy_and_generalized_stop_evidence_must_agree():
    with pytest.raises(ToolContractError, match="disagree"):
        enforce_tool_contract(
            {"tools": tools("weather", "news")},
            [],
            finish_reason="stop",
            stop_truncated_tool_call={"function": "news"},
            truncated_tool_call={"function": "weather", "cause": "stop_sequence"},
        )


@pytest.mark.parametrize("family", FAMILIES)
def test_plain_prose_length_finish_has_no_tool_evidence(family):
    parser, _ = parser_and_partial(family, "weather", tools("weather"))
    # Muse begins in its channel header; its ordinary answer uses that header.
    text = "hello" if family != "muse" else "to=user<|message|>hello"
    events = parser.finish(text, "length")
    assert "".join(event.get("content", "") for event in events) == "hello"
    assert parser.truncated_tool_call is None
