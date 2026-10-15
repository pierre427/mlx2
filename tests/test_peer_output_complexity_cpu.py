"""Chunk-boundary and bounded-work tests of the real generic tool parser."""

import codecs
import json

import pytest

from mlx2.output import OutputParser, ToolCallConstraintError
from mlx2.runtime.tool_parsers.qwen3_coder import parse_tool_call

TOOLS = [{"type": "function", "function": {
    "name": "db:lookup_é", "strict": True,
    "parameters": {
        "type": "object", "properties": {"query": {
            "type": "object", "properties": {"text": {"type": "string"}},
            "required": ["text"], "additionalProperties": False,
        }},
        "required": ["query"], "additionalProperties": False,
    },
}}]


def tool_text(query):
    return (
        '<tool_call><function=db:lookup_é><parameter=query>'
        + json.dumps({"text": query}, ensure_ascii=False)
        + '</parameter></function></tool_call>'
    )


def summarize(events):
    return {
        "content": "".join(e.get("content", "") for e in events),
        "reasoning_content": "".join(e.get("reasoning_content", "") for e in events),
        "tools": [call["function"] for e in events for call in e.get("tool_calls", [])],
    }


def parse(chunks, **kwargs):
    parser = OutputParser(chat=True, tools=TOOLS, parse_tool=parse_tool_call, **kwargs)
    events = []
    for chunk in chunks:
        events += parser.push(chunk)
    events += parser.finish("", "stop")
    return summarize(events)


@pytest.mark.parametrize("text", [
    tool_text('quoted </tool_call> and \\" escaped ☃'),
    '<think>reason about <tool_call></think>' + tool_text('é'),
    '```xml\n<tool_call>example</tool_call>\n```\n' + tool_text('valid'),
    tool_text('one </tool_call>') + '\n' + tool_text('two'),
])
def test_full_and_streamed_parsing_agree_at_every_utf8_byte_split(text):
    expected = parse([text])
    encoded = text.encode()
    for split in range(len(encoded) + 1):
        decoder = codecs.getincrementaldecoder("utf-8")()
        chunks = [decoder.decode(encoded[:split]), decoder.decode(encoded[split:], final=True)]
        assert parse(chunks) == expected, split
    assert parse(list(text)) == expected


def test_long_tool_body_retains_chunks_and_never_reparses_a_quoted_closer():
    calls = []

    def counted(text, tools):
        calls.append(len(text))
        return parse_tool_call(text, tools)

    parser = OutputParser(chat=True, tools=TOOLS, parse_tool=counted)
    text = tool_text('</tool_call>' + 'x' * 16384)
    events = []
    for char in text:
        events += parser.push(char)
        if parser.channel == "tool":
            assert len(parser.buffer) < len(parser.tool_close)
    events += parser.finish("", "stop")
    assert len(calls) == 2  # quoted closer once, then the actual closer
    assert parser.tool_parse_work_chars == sum(calls)
    assert parser.tool_marker_scan_chars <= len(text) * len(parser.tool_close)
    assert summarize(events)["tools"][0]["arguments"] == json.dumps({
        "query": {"text": '</tool_call>' + 'x' * 16384},
    })


@pytest.mark.parametrize("tolerant", [False, True])
def test_many_distinct_quoted_closers_fail_closed_at_grammar_work_budget(tolerant):
    parser = OutputParser(
        chat=True, tools=TOOLS, parse_tool=parse_tool_call,
        tolerant_tool_markers=tolerant, max_tool_parse_chars=1024,
    )
    with pytest.raises(ToolCallConstraintError, match="character-work budget"):
        for char in tool_text('</tool_call>' * 100):
            parser.push(char)
    assert parser.tool_count == 0
    assert parser.tool_call_parse_fallbacks == 0


def test_unfinished_chunked_call_keeps_exact_length_stop_record():
    parser = OutputParser(chat=True, tools=TOOLS, parse_tool=parse_tool_call)
    text = tool_text('long ' * 1024).removesuffix('</parameter></function></tool_call>')
    for char in text:
        assert not parser.push(char)
    assert parser.finish('', 'length') == []
    assert parser.truncated_tool_call == {"function": "db:lookup_é", "cause": "max_tokens"}
    assert parser.tool_count == 0


def test_quoted_closer_without_real_closer_preserves_terminal_error():
    text = tool_text('</tool_call> unfinished').removesuffix('</parameter></function></tool_call>')
    for chunks in ([text], list(text)):
        with pytest.raises(ValueError):
            parse(chunks)
