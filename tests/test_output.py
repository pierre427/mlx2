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


def _channels_at_every_split(text, **kwargs):
    """Parse ``text`` whole, at every two-chunk split and char by char; all
    ways must agree.  Returns the (reasoning, content) pair."""

    def run(chunks):
        parser = OutputParser(chat=True, **kwargs)
        events = []
        for chunk in chunks:
            events += parser.push(chunk)
        events += parser.push("", final=True)
        return (
            "".join(e.get("reasoning_content", "") for e in events),
            "".join(e.get("content", "") for e in events),
        )

    whole = run([text])
    for split in range(1, len(text)):
        assert run([text[:split], text[split:]]) == whole, split
    assert run(list(text)) == whole
    return whole


@pytest.mark.parametrize("thinking", [False, True])
def test_reasoning_markers_inside_a_json_answer_are_literal(thinking):
    """Qwen tokenizers encode ``<think>``/``</think>`` as single ordinary
    added tokens, so a JSON grammar admits them inside a string.  They used to
    switch channels: the rest of the answer went to reasoning, or the closing
    tag vanished, and the delivered JSON no longer parsed."""
    for answer in ('{"tag":"<think>"}', '{"tag":"</think>"}', '{"a":"<think>x</think>y"}'):
        prefix = "plan</think>" if thinking else ""
        reasoning, content = _channels_at_every_split(prefix + answer, thinking=thinking)
        assert content == answer
        assert json.loads(content)
        assert reasoning == ("plan" if thinking else "")


def test_only_the_first_close_ends_reasoning_and_later_markers_are_content():
    text = "<think>weigh <think> it</think>answer <think>x</think> `</think>` y"
    assert _channels_at_every_split(text, thinking=True) == (
        "weigh <think> it",
        "answer <think>x</think> `</think>` y",
    )
    # Implied open (the template opened the block): the same without the
    # leading marker.
    assert _channels_at_every_split(text[len("<think>"):], thinking=True) == (
        "weigh <think> it",
        "answer <think>x</think> `</think>` y",
    )


def test_with_thinking_off_only_a_leading_open_marker_is_reasoning():
    assert _channels_at_every_split("<think>r</think>a</think>") == ("r", "a</think>")
    assert _channels_at_every_split("a <think>r</think>b") == ("", "a <think>r</think>b")
    assert _channels_at_every_split("r</think>a") == ("", "r</think>a")
    assert _channels_at_every_split("<thin") == ("", "<thin")


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


@pytest.mark.parametrize(
    "adapter_type, call",
    [
        pytest.param(
            FlashNextAdapter, "<tool_call><function={}></function></tool_call>",
            id="flash-next",
        ),
        pytest.param(LagunaXS21Adapter, "<tool_call>{}</tool_call>", id="laguna"),
    ],
)
def test_generic_adapter_parsers_carry_the_request_tool_bounds(adapter_type, call):
    """Laguna did not hand ``parallel_tool_calls`` or ``constrained_tools``
    to its parser, so a second call under ``parallel_tool_calls:false`` left
    the parser and the terminal contract check turned the request into a 502,
    and a malformed call under ``tool_choice:required`` became content."""
    from mlx2.openai_compat import enforce_tool_contract

    tools = [
        {"type": "function", "function": {"name": name, "parameters": {}}}
        for name in ("a", "b")
    ]
    request = {
        "messages": [{"role": "user", "content": "x"}],
        "tools": tools,
        "parallel_tool_calls": False,
        "enable_thinking": False,
    }
    adapter = adapter_type.__new__(adapter_type)
    parser = adapter.output_parser(request)
    events = parser.push(call.format("a") + call.format("b"), final=True)
    calls = [call for event in events for call in event.get("tool_calls", ())]
    assert [c["function"]["name"] for c in calls] == ["a"]
    assert parser.tool_call_constraint_truncations == 1
    enforce_tool_contract(request, calls, finish_reason="tool_calls")

    required = adapter.output_parser({
        **request, "tool_choice": "required", "parallel_tool_calls": True,
        "_tolerant_tool_markers": True,
    })
    with pytest.raises(ValueError, match="undeclared"):
        required.push(call.format("zzz"), final=True)


def test_boolean_property_subschemas_parse_instead_of_crashing():
    """``{"x": true}`` is valid JSON Schema and passes admission; Laguna and
    Muse called ``.get`` on it, and the AttributeError escaped every
    tolerant fallback as a 502."""
    from mlx2.adapters.muse_glimmer_output import MuseOutputParser
    from mlx2.runtime.tool_parsers.laguna import parse_tool_call as laguna_parse
    from mlx2.server import validate_request

    tools = validate_request({
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {
            "type": "object", "properties": {"x": True, "y": False},
        }}}],
    })["tools"]

    def arguments(events):
        (call,) = [c for e in events for c in e.get("tool_calls", ())]
        return json.loads(call["function"]["arguments"])

    laguna = OutputParser(chat=True, tools=tools, parse_tool=laguna_parse)
    assert arguments(laguna.push(
        "<tool_call>f<arg_key>x</arg_key><arg_value>[1]</arg_value></tool_call>",
        final=True,
    )) == {"x": [1]}
    muse_call = (
        '<|message|><atem:function_calls><atem:invoke name="f">'
        '<atem:parameter name="{}">[1]</atem:parameter>'
        "</atem:invoke></atem:function_calls>"
    )
    muse = MuseOutputParser(chat=True, tools=tools)
    assert arguments(muse.push(muse_call.format("x"), final=True)) == {"x": [1]}
    # ``false`` admits no value, so a call that supplies one is malformed.
    with pytest.raises(ValueError, match="forbidden"):
        MuseOutputParser(chat=True, tools=tools).push(muse_call.format("y"), final=True)


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


def test_qwen_untyped_parameter_decodes_objects_and_arrays_only():
    # mlx-lm#1910: a parameter schema without ``type`` admits any JSON value;
    # returning a structured argument as its source text handed clients a
    # string where the model wrote an object.
    tools = [{
        "type": "function",
        "function": {
            "name": "configure",
            "parameters": {
                "type": "object",
                "properties": {
                    "config": {"description": "settings"},
                    "ids": {"description": "targets"},
                    "label": {"description": "free text"},
                    "count": {},
                },
            },
        },
    }]
    call = parse_tool_call(
        "<function=configure>"
        '<parameter=config>{"depth": 2, "tags": ["a"]}</parameter>'
        "<parameter=ids>[1, 2]</parameter>"
        "<parameter=label>not {json</parameter>"
        "<parameter=count>123</parameter>"
        "</function>",
        tools,
    )
    assert call == {
        "name": "configure",
        "arguments": {
            "config": {"depth": 2, "tags": ["a"]},
            "ids": [1, 2],
            "label": "not {json",
            "count": "123",
        },
    }


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
