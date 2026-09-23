"""Xing4.0 streaming reasoning / content / tool-call parser."""

from __future__ import annotations

import json
import random

import pytest

from mlx2.adapters.xing_output import XingOutputParser, parse_tool_block

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
                "scale": {"type": "number"},
                "live": {"type": "boolean"},
                "opts": {"type": "object"},
                "tags": {"type": "array"},
                "unit": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "note": {"type": ["string", "null"]},
            },
            "required": ["city"],
        },
    },
}
STRICT = {
    "type": "function",
    "function": {
        "name": "strict",
        "parameters": {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "additionalProperties": False,
        },
    },
}
TOOLS = [WEATHER, STRICT, {"type": "function", "function": {"name": "get", "parameters": {}}}]


def _call(name, **params):
    body = "".join(
        f"<param_key>{k}</param_key><param_value>{v}</param_value>" for k, v in params.items()
    )
    return f"<tool_call>{name}{body}</tool_call>"


def run(text, *, chunks=None, chat=True, thinking=True, tools=TOOLS, stops=()):
    parser = XingOutputParser(chat=chat, thinking=thinking, tools=tools, stops=stops)
    pieces = chunks if chunks is not None else [text]
    events = []
    for piece in pieces:
        events.extend(parser.push(piece))
    events.extend(parser.push("", final=True))
    out = {"reasoning_content": "", "content": "", "tool_calls": []}
    for event in events:
        assert len(event) == 1
        key, value = next(iter(event.items()))
        if key == "tool_calls":
            out["tool_calls"].extend(value)
        else:
            assert value, "empty deltas must not be emitted"
            out[key] += value
    out["stopped"] = parser.stopped
    return out


def split_every(text, size):
    return [text[i : i + size] for i in range(0, len(text), size)]


def calls(result):
    return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in result["tool_calls"]]


def test_thinking_on_starts_in_reasoning_and_drops_template_newline():
    out = run("First, 2+2.\nThen check.\n</think>The answer is 4.")
    assert out["reasoning_content"] == "First, 2+2.\nThen check."
    assert out["content"] == "The answer is 4."
    assert out["tool_calls"] == []


def test_thinking_off_is_all_content_and_stray_markers_do_not_leak():
    out = run("Hello</think> world", thinking=False)
    assert out == {"reasoning_content": "", "content": "Hello world", "tool_calls": [], "stopped": False}
    out = run("A<think>hidden\n</think>B", thinking=False)
    assert out["content"] == "AB" and out["reasoning_content"] == "hidden"


def test_unterminated_reasoning_stays_reasoning():
    # vLLM/HF behaviour; SGLang's force_nonempty_content is deliberately not copied.
    out = run("still thinking\n")
    assert out["reasoning_content"] == "still thinking\n"
    assert out["content"] == ""


def test_tool_calls_typed_by_schema_back_to_back():
    text = (
        "Checking.\n</think>Let me look."
        + _call("get_weather", city="  Paris ", days="3", scale="1.5", live="true",
                opts='{"a": [1, null]}', tags='["x", "y"]', unit="null", note="null")
        + "\n"
        + _call("get_weather", city="123", days="'7'", extra="{'k': (1, 2)}")
        + _call("get")
    )
    out = run(text)
    assert out["reasoning_content"] == "Checking."
    assert out["content"] == "Let me look."
    assert calls(out) == [
        ("get_weather", {"city": "Paris", "days": 3, "scale": 1.5, "live": True,
                          "opts": {"a": [1, None]}, "tags": ["x", "y"], "unit": None, "note": None}),
        ("get_weather", {"city": "123", "days": "7", "extra": {"k": [1, 2]}}),
        ("get", {}),
    ]
    assert [c["index"] for c in out["tool_calls"]] == [0, 1, 2]
    for call in out["tool_calls"]:
        assert call["type"] == "function" and call["id"].startswith("call_")
        assert isinstance(call["function"]["arguments"], str)


def test_parallel_false_keeps_first_of_two_xing_calls():
    from mlx2.openai_compat import enforce_tool_contract

    parser = XingOutputParser(
        chat=True, thinking=False, tools=TOOLS, parallel_tool_calls=False
    )
    events = parser.push(_call("get") + _call("get"), final=True)
    calls = [call for event in events for call in event.get("tool_calls", ())]

    assert len(calls) == 1
    assert parser.tool_call_constraint_truncations == 1
    enforce_tool_contract(
        {
            "tools": TOOLS,
            "tool_choice": "required",
            "parallel_tool_calls": False,
        },
        calls,
    )


def test_raw_string_values_keep_json_looking_text():
    out = run("</think>" + _call("get_weather", city='{"not": "decoded"}', note="[1, 2]"))
    assert calls(out) == [("get_weather", {"city": '{"not": "decoded"}', "note": "[1, 2]"})]


def test_json_body_forms_are_accepted():
    text = (
        '</think><tool_call>{"name": "get_weather", "arguments": {"city": "Rome", "days": 2}}</tool_call>'
        '<tool_call>get_weather {"city": "Oslo"}</tool_call>'
        '<tool_call>{"name": "get", "arguments": "{\\"a\\": 1}"}</tool_call>'
    )
    assert calls(run(text)) == [
        ("get_weather", {"city": "Rome", "days": 2}),
        ("get_weather", {"city": "Oslo"}),
        ("get", {"a": 1}),
    ]


def test_non_finite_and_unserializable_values_stay_text():
    out = run("</think>" + _call("get", a="NaN", b="{1, 2}", c="b'x'", d="-Infinity"))
    assert calls(out) == [("get", {"a": "NaN", "b": "{1, 2}", "c": "b'x'", "d": "-Infinity"})]


@pytest.mark.parametrize(
    "block, message",
    [
        (_call("nope", a="1"), "undeclared tool"),
        (_call("get_weather", days="1"), "missing a required"),
        (_call("strict", q="x", z="1"), "undeclared parameter"),
        ("<tool_call>get<param_key>a</param_key><param_value>1</param_value>junk</tool_call>", "Malformed"),
        ("<tool_call>get<param_key>a</param_key><param_value>1</param_value>"
         "<param_key>a</param_key><param_value>2</param_value></tool_call>", "Duplicate"),
        ("<tool_call>  </tool_call>", "Empty"),
        ("<tool_call>{bad json</tool_call>", "Malformed"),
        # A raw or untyped value ends at the first closer, which cannot be
        # told from one it quotes, and a JSON value quoting one must decode.
        (_call("get_weather", city="a</param_value>b"), "Malformed"),
        (_call("get_weather", city="a</tool_call>b"), "Malformed"),
        (_call("get", a='{"s": "</param_value>"}'), "Malformed"),
        (_call("get_weather", city="x", opts='{"s": "</param_value>" junk}'), "Malformed"),
    ],
)
def test_malformed_tool_calls_fail_closed(block, message):
    with pytest.raises(ValueError, match=message):
        run("</think>" + block)


@pytest.mark.parametrize(
    "opts, tags",
    [
        ('{"s": "a</param_value>b"}', '["a</tool_call>b"]'),
        ('{"s": "</param_value></tool_call>", "t": "<tool_call>get</tool_call>"}', "[]"),
        ("{}", '["say \\"</tool_call>\\"", "</param_value>\\\\"]'),
    ],
)
def test_json_values_keep_quoted_closing_tags(opts, tags):
    """The template writes non-string values with ``tojson``, which leaves
    ``<`` raw, so a JSON string may quote ``</param_value>`` or
    ``</tool_call>``.  The parser cut the value at the first
    ``</param_value>`` and the block at the first ``</tool_call>``, quoted or
    not, and the call failed with 502.  A JSON value now ends at the first
    closer outside its strings, however the text is chunked."""
    call = _call("get_weather", city="Oslo", opts=opts, tags=tags)
    text = "</think>" + call + call + "done"
    expected = {"city": "Oslo", "opts": json.loads(opts), "tags": json.loads(tags)}
    for size in (len(text), 1, 7):
        out = run(text, chunks=split_every(text, size))
        assert calls(out) == [("get_weather", expected)] * 2
        assert out["content"] == "done"


def test_incomplete_tool_call_fails_at_final_or_turn_end():
    with pytest.raises(ValueError, match="incomplete"):
        run("</think><tool_call>get<param_key>a")
    with pytest.raises(ValueError, match="incomplete"):
        run("</think><tool_call>get<_end>")


def test_tool_markup_without_declared_tools_is_content():
    text = "</think>" + _call("get", a="1")
    assert run(text, tools=None)["content"] == _call("get", a="1")


def test_tool_markup_inside_reasoning_stays_reasoning():
    out = run("maybe " + _call("get", a="1") + "\n</think>ok")
    assert out["reasoning_content"] == "maybe " + _call("get", a="1")
    assert out["tool_calls"] == [] and out["content"] == "ok"


def test_whitespace_around_tool_calls_is_not_content():
    out = run("</think>\n\n" + _call("get") + "\n" + _call("get") + "\n\n")
    assert out["content"] == "" and len(out["tool_calls"]) == 2
    out = run("</think>\n\nPlain answer.\n")
    assert out["content"] == "\n\nPlain answer.\n"


def test_turn_end_markers_stop_the_stream():
    out = run("r\n</think>done<_end>ignored", thinking=True)
    assert out["content"] == "done" and out["stopped"]
    out = run("hi<_user>fake turn", thinking=False)
    assert out["content"] == "hi" and out["stopped"]
    parser = XingOutputParser(chat=True, thinking=False, tools=TOOLS)
    assert parser.push("x<_end>") == [{"content": "x"}]
    assert parser.push("more") == []


def test_stop_strings_apply_to_content_only():
    out = run("STOP in reasoning\n</think>ab STOP cd", stops=["STOP"])
    assert out["reasoning_content"] == "STOP in reasoning"
    assert out["content"] == "ab " and out["stopped"]
    out = run("</think>abSTcd", chunks=["</think>ab", "S", "T", "cd"], stops=["STOP", "ST"])
    assert out["content"] == "ab" and out["stopped"]


def test_completion_mode_passes_text_through():
    out = run("a</think><tool_call>b", chat=False)
    assert out["content"] == "a</think><tool_call>b" and out["reasoning_content"] == ""


STREAM_CASES = [
    ("Plan:\n1. a\n2. b\n</think>Answer." + _call("get_weather", city="东京", days="2") + _call("get"), True),
    ("Hmm <think> nested?\n</think>\n\nfinal text 😀 with </thi-lookalike and <tool_", True),
    ("Direct answer <b>bold</b> < > <tool_call>" + '{"name": "get", "arguments": {}}' + "</tool_call>", False),
    ("x\n\n</think>" + _call("get_weather", city="a<b", opts='{"k": "</param"}'), True),
]


@pytest.mark.parametrize("text, thinking", STREAM_CASES)
def test_streaming_equals_one_shot_for_every_chunking(text, thinking):
    whole = run(text, thinking=thinking)
    strip_ids = lambda r: {**r, "tool_calls": calls(r)}  # noqa: E731
    for size in (1, 2, 3, 5, 7, 13):
        assert strip_ids(run(text, chunks=split_every(text, size), thinking=thinking)) == strip_ids(whole)
    rng = random.Random(len(text))
    for _ in range(50):
        cuts = sorted(rng.sample(range(1, len(text)), rng.randint(1, min(12, len(text) - 1))))
        chunks = [text[a:b] for a, b in zip([0] + cuts, cuts + [len(text)])]
        assert strip_ids(run(text, chunks=chunks, thinking=thinking)) == strip_ids(whole)


def test_partial_markers_never_leak_into_deltas():
    text = "r\n</think>A" + _call("get", a="1") + "B"
    parser = XingOutputParser(chat=True, thinking=True, tools=TOOLS)
    for char in text:
        for event in parser.push(char):
            for key, value in event.items():
                if key != "tool_calls":
                    assert "<" not in value, (key, value)
    parser.push("", final=True)


def test_parse_tool_block_prefers_longest_declared_name_for_json_suffix():
    tools = [{"type": "function", "function": {"name": "get"}},
             {"type": "function", "function": {"name": "get_all"}}]
    assert parse_tool_block('get_all{"x": 1}', tools) == {"name": "get_all", "arguments": {"x": 1}}
    assert parse_tool_block("get_all", tools) == {"name": "get_all", "arguments": {}}


def test_detokenized_model_stream_end_to_end():
    """Token-level: ids -> XingStreamingDetokenizer -> parser (needs tokenizer.model)."""
    from test_xing_tokenizer import MODEL_DIR

    if MODEL_DIR is None or not (MODEL_DIR / "tokenizer.json").exists():
        pytest.skip("converted Xing4.0 tokenizer not available")
    from mlx2.adapters import xing_tokenizer as xt

    tokenizer = xt.load_fast_unverified(MODEL_DIR)
    wrapper = xt.make_tokenizer_wrapper(tokenizer)
    output = ("The user wants weather.\n</think>I'll check."
              + _call("get_weather", city="Paris", days="3") + _call("get_weather", city="巴黎"))
    ids = tokenizer.encode(output, add_special_tokens=False)
    detok = wrapper.detokenizer
    parser = XingOutputParser(chat=True, thinking=True, tools=TOOLS)
    events = []
    for token in ids:
        detok.add_token(token)
        events.extend(parser.push(detok.last_segment))
    detok.finalize()
    events.extend(parser.push(detok.last_segment, final=True))
    reasoning = "".join(e.get("reasoning_content", "") for e in events)
    content = "".join(e.get("content", "") for e in events)
    got = [(c["function"]["name"], json.loads(c["function"]["arguments"]))
           for e in events for c in e.get("tool_calls", [])]
    assert reasoning == "The user wants weather."
    assert content == "I'll check."
    assert got == [("get_weather", {"city": "Paris", "days": 3}), ("get_weather", {"city": "巴黎"})]


WEATHER = [{"type": "function", "function": {"name": "weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def _run(chunks, final=True):
    from mlx2.adapters.xing_output import XingOutputParser

    parser = XingOutputParser(chat=True, thinking=False, tools=WEATHER)
    events = []
    for index, chunk in enumerate(chunks):
        events.extend(parser.push(chunk, final=final and index == len(chunks) - 1))
    return parser, events


@pytest.mark.parametrize("split", [1, 3, 7, 1000])
def test_turn_end_closes_a_structurally_complete_tool_call(split):
    # GPU 2026-09-18: warm-cache numerics made the model end the turn right
    # after the last </param_value>, skipping </tool_call> (was HTTP 502).
    text = "<tool_call>weather<param_key>city</param_key><param_value>Toronto</param_value><_end>"
    parser, events = _run([text[i : i + split] for i in range(0, len(text), split)])
    calls = [call for event in events for call in event.get("tool_calls", [])]
    assert len(calls) == 1 and parser.turn_closed_tool_calls == 1
    # EOS is not delivered as text: generation simply ends (final=True).
    text = text[: -len("<_end>")]
    parser, events = _run([text[i : i + split] for i in range(0, len(text), split)])
    calls = [call for event in events for call in event.get("tool_calls", [])]
    assert len(calls) == 1 and calls[0]["function"]["name"] == "weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Toronto"}
    assert parser.turn_closed_tool_calls == 1 and parser.stopped


@pytest.mark.parametrize("text", [
    "<tool_call>weather<param_key>city</param_key><param_value>Toro<_end>",
    "<tool_call>weather<param_key>city</param_key><_end>",
    "<tool_call>weather<_end>",
    '<tool_call>{"name": "weather", "arguments": {"city": "To<_end>',
])
def test_truncated_tool_call_at_turn_end_still_fails_closed(text):
    with pytest.raises(ValueError, match="incomplete"):
        _run([text])
    with pytest.raises(ValueError, match="incomplete"):
        _run([text[: -len("<_end>")]])


def test_json_tool_call_closed_by_turn_end():
    parser, events = _run(['<tool_call>{"name": "weather", "arguments": {"city": "Oslo"}}<_end>'])
    calls = [call for event in events for call in event.get("tool_calls", [])]
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Oslo"}
