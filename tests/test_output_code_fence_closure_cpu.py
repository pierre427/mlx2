"""Quoted tool examples remain inert across incremental fence boundaries."""

import json

import pytest

from mlx2.output import OutputParser


TOOLS = [{"type": "function", "function": {"name": "example", "parameters": {}}}]
CALL = '<tool_call>{"name":"example","arguments":{}}</tool_call>'


def parse_chunks(chunks):
    parser = OutputParser(chat=True, tools=TOOLS, parse_tool=lambda value, _: json.loads(value))
    events = []
    for chunk in chunks:
        events.extend(parser.push(chunk))
    events.extend(parser.push("", final=True))
    return (
        "".join(event.get("content", "") for event in events),
        [call for event in events for call in event.get("tool_calls", ())],
    )


@pytest.mark.parametrize("fence", ["```", "~~~"])
@pytest.mark.parametrize("trailing", ["example ", " ", "\t"])
def test_non_whitespace_after_closing_fence_keeps_tool_example_in_code(fence, trailing):
    text = f"{fence}xml\n{fence}{trailing}{CALL}\n{fence}\n"
    for split in range(len(text) + 1):
        content, calls = parse_chunks([text[:split], text[split:]])
        assert calls == [], (split, content)
        assert content == text
    assert parse_chunks(list(text)) == (text, [])


@pytest.mark.parametrize("fence", ["```", "~~~"])
@pytest.mark.parametrize("trailing", ["", " \t", "\r"])
def test_real_closing_fence_releases_following_tool_call(fence, trailing):
    prefix = f"{fence}xml\nexample\n{fence}{trailing}\n"
    text = prefix + CALL
    for split in range(len(text) + 1):
        content, calls = parse_chunks([text[:split], text[split:]])
        assert content == prefix
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "example"
