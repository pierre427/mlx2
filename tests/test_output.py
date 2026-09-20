import json

import pytest

from mlx2.adapters.flash_next import FlashNextAdapter
from mlx2.adapters.laguna_xs21 import LagunaXS21Adapter
from mlx2.adapters.mlx_vlm import Gemma3nAdapter, MiniCPMOAdapter
from mlx2.adapters.muse_glimmer import MuseGlimmerAdapter
from mlx2.adapters.north_mini_code import NorthMiniCodeAdapter
from mlx2.adapters.xing import XingAdapter
from mlx2.output import OutputParser
from mlx2.runtime.tool_parsers.qwen3_coder import parse_tool_call
from mlx2.memory import available_execution_bytes


def _adapter_parser(adapter_type):
    adapter = adapter_type.__new__(adapter_type)
    return adapter.output_parser(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "enable_thinking": False,
            "stop": ["STOP"],
        }
    )


@pytest.mark.parametrize(
    "adapter_type",
    [
        pytest.param(FlashNextAdapter, id="flash-next-generic"),
        pytest.param(LagunaXS21Adapter, id="laguna-generic"),
        pytest.param(Gemma3nAdapter, id="gemma3n-generic"),
        pytest.param(MiniCPMOAdapter, id="minicpmo-generic"),
        pytest.param(NorthMiniCodeAdapter, id="north"),
        pytest.param(MuseGlimmerAdapter, id="muse"),
        pytest.param(XingAdapter, id="xing"),
    ],
)
def test_every_adapter_parser_records_the_matched_stop_sequence(adapter_type):
    parser = _adapter_parser(adapter_type)
    events = parser.push("hello ST") + parser.push("OP hidden", final=True)
    assert "".join(event.get("content", "") for event in events) == "hello "
    assert parser.stopped is True
    assert parser.stop_sequence == "STOP"


def test_reasoning_markers_at_every_chunk_boundary():
    text = "<think>reason</think>answer"
    for split in range(len(text) + 1):
        parser = OutputParser(chat=True)
        events = parser.push(text[:split]) + parser.push(text[split:], final=True)
        assert "".join(e.get("reasoning_content", "") for e in events) == "reason"
        assert "".join(e.get("content", "") for e in events) == "answer"


def test_stop_at_every_chunk_boundary():
    text = "hello STOP hidden"
    for split in range(len(text) + 1):
        parser = OutputParser(stops=["STOP"])
        events = parser.push(text[:split]) + parser.push(text[split:], final=True)
        assert "".join(e.get("content", "") for e in events) == "hello "
        assert parser.stopped
        assert parser.stop_sequence == "STOP"


def test_tool_call_at_every_boundary():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "sum",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                },
            },
        }
    ]
    text = (
        "<tool_call><function=sum><parameter=x>123</parameter></function></tool_call>"
    )
    for split in range(len(text) + 1):
        parser = OutputParser(chat=True, tools=tools, parse_tool=parse_tool_call)
        events = parser.push(text[:split]) + parser.push(text[split:], final=True)
        calls = [c for e in events for c in e.get("tool_calls", [])]
        assert len(calls) == 1
        assert json.loads(calls[0]["function"]["arguments"]) == {"x": 123}


def test_unfinished_tool_falls_back_to_content_when_unconstrained():
    parser = OutputParser(
        chat=True, tools=[{}], parse_tool=parse_tool_call,
        tolerant_tool_markers=True,
    )
    assert parser.push("<tool_call><function=sum>", final=True) == [
        {"content": "<tool_call><function=sum>"}
    ]
    assert parser.tool_call_parse_fallbacks == 1


def test_footprint_is_admission_floor():
    assert (
        available_execution_bytes(
            available=80, recommended=100, active=20, cached=10, footprint=90
        )
        == 10
    )
    assert (
        available_execution_bytes(
            available=80, recommended=100, active=20, cached=10, footprint=None
        )
        == 0
    )


def test_client_stop_inside_tool_call_is_a_stop_not_a_server_error():
    tools = [{"type": "function", "function": {"name": "sum", "parameters": {}}}]
    parser = OutputParser(
        chat=True, tools=tools, parse_tool=parse_tool_call, stops=["\n\n"]
    )
    events = parser.push("<tool_call><function=sum>\n\n<parameter=x>1")
    assert parser.stopped
    assert events == []
    assert parser.tool_count == 0
    # Further pushes are inert once stopped.
    assert parser.push("</parameter></function></tool_call>", final=True) == []


def test_model_truncated_tool_call_still_fails_closed_without_a_stop():
    parser = OutputParser(
        chat=True, tools=[{}], parse_tool=parse_tool_call, stops=["\n\n"],
        constrained_tools=True,
    )
    with pytest.raises(ValueError, match="incomplete"):
        parser.push("<tool_call><function=sum>", final=True)


def test_tool_markers_inside_reasoning_are_never_executable():
    tools = [{"type": "function", "function": {"name": "sum", "parameters": {}}}]
    text = (
        "<think>consider <tool_call><function=sum></function></tool_call> carefully"
        "</think>answer"
    )
    for split in range(len(text) + 1):
        parser = OutputParser(
            chat=True, thinking=True, tools=tools, parse_tool=parse_tool_call,
            tolerant_tool_markers=True,
        )
        events = parser.push(text[:split]) + parser.push(text[split:], final=True)
        assert not any(event.get("tool_calls") for event in events)
        reasoning = "".join(event.get("reasoning_content", "") for event in events)
        assert "<tool_call>" in reasoning and "<function=sum>" in reasoning
        assert "".join(event.get("content", "") for event in events) == "answer"


@pytest.mark.parametrize(
    "body",
    [
        "<function=missing></function>",
        "not a function",
        "<function=sum><parameter=x>bad",
    ],
)
def test_malformed_or_undeclared_tool_blocks_fall_back_as_raw_content(body):
    tools = [{"type": "function", "function": {"name": "sum", "parameters": {}}}]
    parser = OutputParser(
        chat=True, tools=tools, parse_tool=parse_tool_call,
        tolerant_tool_markers=True,
    )
    raw = f"<tool_call>{body}</tool_call>"
    assert parser.push(raw, final=True) == [{"content": raw}]
    assert parser.tool_call_parse_fallbacks == 1


def test_constrained_tool_parse_failure_stays_fail_closed():
    tools = [{"type": "function", "function": {"name": "sum", "parameters": {}}}]
    parser = OutputParser(
        chat=True,
        tools=tools,
        parse_tool=parse_tool_call,
        constrained_tools=True,
    )
    with pytest.raises(ValueError, match="undeclared"):
        parser.push(
            "<tool_call><function=missing></function></tool_call>", final=True
        )


def test_parallel_tool_calls_false_drops_a_second_auto_call():
    """Over-calling is a model behaviour the bound answers, not a failure.

    Raising here reached the client as HTTP 502 ``server_error`` (and, on a
    stream, as an error event after the first call had already been written).
    The bound is honoured instead: at most one call leaves the parser, so the
    terminal ``enforce_tool_contract`` check can never disagree with it.
    """
    tools = [
        {"type": "function", "function": {"name": name, "parameters": {}}}
        for name in ("one", "two")
    ]
    parser = OutputParser(
        chat=True,
        tools=tools,
        parse_tool=parse_tool_call,
        parallel_tool_calls=False,
    )
    first = "<tool_call><function=one></function></tool_call>"
    second = "<tool_call><function=two></function></tool_call>"
    assert parser.push(first)[0]["tool_calls"][0]["function"]["name"] == "one"
    events = parser.push(second, final=True)
    assert [call for event in events for call in event.get("tool_calls", ())] == []
    assert parser.tool_count == 1
    assert parser.tool_call_constraint_truncations == 1


def test_parallel_tool_calls_false_keeps_the_first_of_a_batched_pair():
    """Two calls inside one flush: the first is kept, the surplus counted."""
    tools = [
        {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
    ]
    parser = OutputParser(
        chat=True,
        tools=tools,
        parse_tool=parse_tool_call,
        parallel_tool_calls=False,
    )
    two = (
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nToronto\n"
        "</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    events = parser.push(two, final=True)
    calls = [call for event in events for call in event.get("tool_calls", ())]
    assert len(calls) == 1
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Toronto"}
    assert parser.tool_call_constraint_truncations == 1


def test_qwen_tool_argument_types_resolve_local_schema_refs():
    tools = [{
        "type": "function",
        "function": {
            "name": "sum",
            "parameters": {
                "$defs": {"count": {"type": "integer"}},
                "type": "object",
                "properties": {"x": {"$ref": "#/$defs/count"}},
            },
        },
    }]
    call = parse_tool_call(
        "<function=sum><parameter=x>123</parameter></function>", tools
    )
    assert call == {"name": "sum", "arguments": {"x": 123}}


def test_qwen_non_strict_recursive_ref_keeps_legacy_raw_argument_fallback():
    tools = [{
        "type": "function",
        "function": {
            "name": "sum",
            "parameters": {
                "$defs": {"loop": {"$ref": "#/$defs/loop"}},
                "type": "object",
                "properties": {"x": {"$ref": "#/$defs/loop"}},
            },
        },
    }]
    assert parse_tool_call(
        "<function=sum><parameter=x>123</parameter></function>", tools
    ) == {"name": "sum", "arguments": {"x": "123"}}


_SUM_TOOLS = [{
    "type": "function",
    "function": {
        "name": "sum",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "integer"}, "note": {"type": "string"}},
        },
    },
}]
_SUM_CALL = (
    "<tool_call>\n<function=sum>\n<parameter=x>\n3\n</parameter>\n"
    "</function>\n</tool_call>"
)


def _split_everywhere(text, **kwargs):
    for split in range(len(text) + 1):
        parser = OutputParser(
            chat=True, tools=_SUM_TOOLS, parse_tool=parse_tool_call, **kwargs
        )
        events = parser.push(text[:split]) + parser.push(text[split:], final=True)
        content = "".join(e.get("content", "") for e in events)
        calls = [e["tool_calls"][0] for e in events if "tool_calls" in e]
        yield split, content, calls


@pytest.mark.parametrize("example", [
    "Call it like this:\n```xml\n" + _SUM_CALL + "\n```\nThat is all.",
    "Call it like this:\n  ~~~~\n" + _SUM_CALL + "\n  ~~~~\nThat is all.",
    "Inline: `" + _SUM_CALL.replace("\n", " ") + "` is the wire format.",
    "Inline: ``a ` " + _SUM_CALL.replace("\n", " ") + "`` done.",
])
def test_tool_markup_inside_code_is_content_not_a_phantom_call(example):
    # vllm #57553 / sglang #38624: documented or quoted examples of the tool
    # wire format must not be harvested as calls when no grammar is active.
    for split, content, calls in _split_everywhere(example):
        assert calls == [], split
        assert content == example, split


def test_real_tool_call_after_a_closed_fence_or_code_span_still_parses():
    text = (
        "Example:\n```\n" + _SUM_CALL + "\n```\nand `inline`\n\n" + _SUM_CALL
    )
    for split, content, calls in _split_everywhere(text):
        assert [call["function"]["name"] for call in calls] == ["sum"], split
        assert content == "Example:\n```\n" + _SUM_CALL + "\n```\nand `inline`\n\n"
    # An inline span never runs past its line, so a stray backtick in prose
    # does not swallow a later call.
    stray = "costs 5` per call\n" + _SUM_CALL
    for split, _content, calls in _split_everywhere(stray):
        assert len(calls) == 1, split


def test_qwen_unclosed_trailing_parameter_is_closed_by_the_function_end():
    # vllm #57707: the model closes </function> without </parameter>.
    tools = _SUM_TOOLS
    body = (
        "\n<function=sum>\n<parameter=x>\n3\n</parameter>\n"
        "<parameter=note>\nline one\nline two\n</function>\n"
    )
    assert parse_tool_call(body, tools) == {
        "name": "sum", "arguments": {"x": 3, "note": "line one\nline two"},
    }
    only = "\n<function=sum>\n<parameter=x>\n4\n</function>\n"
    assert parse_tool_call(only, tools) == {"name": "sum", "arguments": {"x": 4}}
    # An unclosed parameter in the middle is still malformed.
    middle = (
        "\n<function=sum>\n<parameter=x>\n3\n<parameter=note>\nok\n"
        "</parameter>\n</function>\n"
    )
    with pytest.raises(ValueError):
        parse_tool_call(middle, tools)
