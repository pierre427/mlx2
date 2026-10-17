"""LFM2.5 pythonic tool calling: parser, output channels, grammar, template."""

import json
from pathlib import Path

import pytest
import regex

from mlx2.adapters.lfm25_output import (
    TOOL_CALL_END,
    TOOL_CALL_START,
    LfmOutputParser,
    make_output_parser,
    make_tool_constraint,
    normalize_messages_for_template,
    render_chat,
    validate_tool_request,
)
from mlx2.output import THINKING_CLOSE_MARKER, OutputParser
from mlx2.runtime.tool_parsers.lfm_pythonic import (
    UnclosedPythonString,
    constrained_tool_grammar,
    parse_tool_call,
    partial_function_names,
)

MODEL = Path.home() / "mlx-models/LFM2.5-24B-A2B-MLX-8bit"

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "units": {"type": "string", "enum": ["c", "f"]},
                "days": {"type": "integer"},
            },
            "required": ["city"],
        },
    },
}
SEARCH = {
    "type": "function",
    "function": {
        "name": "search",
        "parameters": {
            "type": "object",
            "properties": {
                "q": {"type": "string"},
                "n": {"type": "integer"},
                "flags": {"type": "array", "items": {"type": "integer"}},
                "opts": {"type": "object"},
            },
            "required": ["q"],
        },
    },
}
TOOLS = [WEATHER, SEARCH]


def _calls(parsed):
    return parsed if isinstance(parsed, list) else [parsed]


# ---------------------------------------------------------------------------
# Parser


@pytest.mark.parametrize(
    "body, expected",
    [
        ("[get_weather(city='Paris')]", [("get_weather", {"city": "Paris"})]),
        (
            """[get_weather(city='Paris', units="c"), search(q='x', n=3, flags=[1,2], opts={"a": true})]""",
            [
                ("get_weather", {"city": "Paris", "units": "c"}),
                ("search", {"q": "x", "n": 3, "flags": [1, 2], "opts": {"a": True}}),
            ],
        ),
        ("[f()]", [("f", {})]),
        ("  [ f( ) ]  ", [("f", {})]),
        ("f(a=1)", [("f", {"a": 1})]),  # unbracketed, leniently
        (r"[f(s='it\'s a \"q\"\\n')]", [("f", {"s": "it's a \"q\"\\n"})]),
        (r"[f(s='line1\nline2\r')]", [("f", {"s": "line1\nline2\r"})]),
        ("[f(a=True, b=true, c=False, d=false)]",
         [("f", {"a": True, "b": True, "c": False, "d": False})]),
        ("[f(a=None, b=null)]", [("f", {"a": None, "b": None})]),
        ("[f(a=-3, b=-2.5, c=+1, d=1e-05, e=0.0)]",
         [("f", {"a": -3, "b": -2.5, "c": 1, "d": 1e-05, "e": 0.0})]),
        ('[f(o={"k": [1, {"n": null}], "s": "a\\/b"}, l=[true, "x"])]',
         [("f", {"o": {"k": [1, {"n": None}], "s": "a/b"}, "l": [True, "x"]})]),
        ("[f(o={'k': (1, 2), 'b': True})]", [("f", {"o": {"k": [1, 2], "b": True}})]),
        ("[f(s='a, b) ], c')]", [("f", {"s": "a, b) ], c"})]),
        ("[f(s='<|tool_call_end|>')]", [("f", {"s": "<|tool_call_end|>"})]),
        ("[db.lookup(from='x'), get-weather(city='y')]",
         [("db.lookup", {"from": "x"}), ("get-weather", {"city": "y"})]),
        ("[f(s='café ☃')]", [("f", {"s": "café ☃"})]),
        ("[f(a=1,), g(),]", [("f", {"a": 1}), ("g", {})]),
    ],
)
def test_parses_template_language(body, expected):
    parsed = _calls(parse_tool_call(body))
    assert [(call["name"], call["arguments"]) for call in parsed] == expected


@pytest.mark.parametrize(
    "body",
    [
        "[f(1)]",  # positional
        "[f('x')]",
        "[f(*a)]",
        "[f(**k)]",
        "[f(a=b)]",  # a variable
        "[f(a=g(1))]",  # a call
        "[f(a=1+2)]",
        "[f(a=f'x')]",
        "[f(a=b'x')]",
        "[f(a=1j)]",
        "[f(a={1, 2})]",
        "[f(a={1: 2})]",
        "[f(a=1e999)]",
        "[f(a=...)]",
        "[f(a=1, a=2)]",
        "[f(a=)]",
        "[f(a=1 b=2)]",
        "[f(a=[1)]",
        "[f(a=1)",
        "[f(a=1)] trailing",
        "[f(a=1) g()]",
        "[]",
        "",
        "[(a=1)]",
        "[get weather(a=1)]",
        "[f(s='raw\nnewline')]",
        "[f(a=__import__('os'))]",
    ],
)
def test_rejects_non_literal_or_malformed_calls(body):
    with pytest.raises(ValueError):
        parse_tool_call(body)


def test_unclosed_string_marks_a_quoted_closer():
    with pytest.raises(UnclosedPythonString) as caught:
        parse_tool_call("[f(s='half")
    assert caught.value.tool_call_close_quoted is True


@pytest.mark.parametrize(
    "partial, names",
    [
        ("", []),
        ("[", []),
        ("[get_wea", []),
        ("[get_weather", []),
        ("[get_weather(", ["get_weather"]),
        ("[get_weather(city='Par", ["get_weather"]),
        ("[get_weather(city='a, b(c'), search(", ["get_weather", "search"]),
        ("[a(x=[1, 2]), b(y={'k': 'v'}), c", ["a", "b"]),
        ("[a(x=1)]", ["a"]),
    ],
)
def test_partial_function_names(partial, names):
    assert partial_function_names(partial) == names


@pytest.mark.parametrize("partial", ["[(", "[get weather(", "['x'"])
def test_partial_function_names_unreadable(partial):
    with pytest.raises(ValueError):
        partial_function_names(partial)


# ---------------------------------------------------------------------------
# Output parser


def _request(**extra):
    return {"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS, **extra}


def _strip_ids(events):
    out = []
    for event in events:
        if "tool_calls" in event:
            call = dict(event["tool_calls"][0])
            assert call.pop("id").startswith("call_")
            out.append({"tool_calls": [call]})
        else:
            out.append(event)
    return out


def _merge(events):
    """Concatenate adjacent text deltas of one channel (chunking-invariant view)."""
    merged = []
    for event in _strip_ids(events):
        key = next(iter(event))
        if key != "tool_calls" and merged and key in merged[-1] and len(merged[-1]) == 1:
            merged[-1] = {key: merged[-1][key] + event[key]}
        else:
            merged.append(event)
    return [event for event in merged if "tool_calls" in event or next(iter(event.values()))]


def _run(parser, chunks, finish_reason="stop"):
    events = []
    for chunk in chunks[:-1]:
        events += parser.push(chunk)
    events += parser.finish(chunks[-1] if chunks else "", finish_reason)
    return events


def _call_event(index, name, arguments):
    return {"tool_calls": [{"index": index, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments)}}]}


TEXT = (
    "Plan: maybe <|tool_call_start|>[get_weather(city='x')]<|tool_call_end|> later."
    "</think>Checking.\n"
    "<|tool_call_start|>[get_weather(city='Paris', units='c'), "
    "search(q='a) <|tool_call_end|> b', n=-2, flags=[1, 2], opts={\"x\": null})]"
    "<|tool_call_end|>\n"
)


def test_streams_reasoning_content_then_calls():
    events = _merge(_run(make_output_parser(_request()), [TEXT]))
    assert events == [
        {"reasoning_content": "Plan: maybe <|tool_call_start|>[get_weather(city='x')]"
                              "<|tool_call_end|> later."},
        {"content": "Checking.\n"},
        _call_event(0, "get_weather", {"city": "Paris", "units": "c"}),
        _call_event(1, "search", {"q": "a) <|tool_call_end|> b", "n": -2,
                                  "flags": [1, 2], "opts": {"x": None}}),
    ]


@pytest.mark.parametrize("text", [TEXT, TEXT + "Done.", "r</think>" + "<|tool_call_start|>[f(1)]<|tool_call_end|>tail"])
def test_every_split_matches_one_shot(text):
    tolerant = _request(_tolerant_tool_markers=True)
    whole = _merge(_run(make_output_parser(tolerant), [text]))
    by_char = _merge(_run(make_output_parser(tolerant), list(text)))
    assert by_char == whole
    for split in range(len(text) + 1):
        parser = make_output_parser(tolerant)
        assert _merge(_run(parser, [text[:split], text[split:]])) == whole, split


def test_malformed_call_falls_back_to_content_when_tolerant():
    raw = "<|tool_call_start|>[get_weather('Paris')]<|tool_call_end|>"
    parser = make_output_parser(_request(_tolerant_tool_markers=True))
    events = _merge(_run(parser, ["r</think>" + raw + " ok"]))
    assert events == [{"reasoning_content": "r"}, {"content": raw + " ok"}]
    assert parser.tool_call_parse_fallbacks == 1
    assert parser.tool_count == 0


def test_undeclared_tool_falls_back_and_strict_parsers_raise():
    raw = "r</think><|tool_call_start|>[rm_rf(path='/')]<|tool_call_end|>"
    parser = make_output_parser(_request(_tolerant_tool_markers=True))
    _run(parser, [raw])
    assert parser.tool_call_parse_fallbacks == 1
    with pytest.raises(ValueError):
        _run(make_output_parser(_request()), [raw])
    with pytest.raises(ValueError):
        _run(make_output_parser(_request(tool_choice="required",
                                         _tolerant_tool_markers=True)), [raw])


def test_unterminated_call_at_eos_is_fallback_or_error():
    raw = "r</think><|tool_call_start|>[get_weather(city='P')"
    parser = make_output_parser(_request(_tolerant_tool_markers=True))
    events = _merge(_run(parser, [raw]))
    assert events[-1] == {"content": "<|tool_call_start|>[get_weather(city='P')"}
    assert parser.tool_call_parse_fallbacks == 1
    with pytest.raises(ValueError):
        _run(make_output_parser(_request()), [raw])


def test_max_tokens_cut_records_the_committed_call():
    parser = make_output_parser(_request())
    events = _run(parser, ["r</think>", "<|tool_call_start|>[search(q='x'), get_weather(ci"],
                  finish_reason="length")
    assert not any("tool_calls" in event for event in events)
    assert parser.truncated_tool_call == {
        "function": "search", "functions": ["search", "get_weather"], "cause": "max_tokens",
    }
    assert parser.stop_truncated_tool_call is None


def test_max_tokens_right_after_opener():
    parser = make_output_parser(_request())
    _run(parser, ["r</think><|tool_call_start|>"], finish_reason="length")
    assert parser.truncated_tool_call == {"function": None, "cause": "max_tokens"}


def test_stop_string_cut_records_the_call():
    parser = make_output_parser(_request(stop=["STOP"]))
    events = _run(parser, ["r</think>a <|tool_call_start|>[get_weather(city='STOP')]",
                           "<|tool_call_end|>"])
    assert parser.stopped and parser.stop_sequence == "STOP"
    assert _merge(events) == [{"reasoning_content": "r"}, {"content": "a "}]
    assert parser.stop_truncated_tool_call == {"function": "get_weather"}
    assert parser.truncated_tool_call["cause"] == "stop_sequence"


def test_stop_strings_do_not_apply_to_reasoning():
    parser = make_output_parser(_request(stop=["STOP"]))
    events = _merge(_run(parser, ["think STOP more</think>ans STOP x"]))
    assert events == [{"reasoning_content": "think STOP more"}, {"content": "ans "}]


def test_parallel_bound_keeps_first_call_and_counts_the_drop():
    parser = make_output_parser(_request(parallel_tool_calls=False))
    events = _merge(_run(parser, [(
        "r</think><|tool_call_start|>[get_weather(city='a'), get_weather(city='b')]"
        "<|tool_call_end|><|tool_call_start|>[search(q='c')]<|tool_call_end|>"
    )]))
    assert [e for e in events if "tool_calls" in e] == [
        _call_event(0, "get_weather", {"city": "a"})
    ]
    assert parser.tool_call_constraint_truncations == 2
    assert parser.tool_count == 1


def test_tool_choice_none_leaves_markers_as_text():
    parser = make_output_parser(_request(tool_choice="none"))
    text = "<|tool_call_start|>[get_weather(city='a')]<|tool_call_end|>"
    assert _merge(_run(parser, ["r</think>" + text])) == [
        {"reasoning_content": "r"}, {"content": text}]


def test_raw_completion_is_plain_content():
    parser = make_output_parser({"prompt": "x", "tools": TOOLS})
    text = "</think><|tool_call_start|>[f()]<|tool_call_end|>"
    assert _merge(_run(parser, [text])) == [{"content": text}]


def test_quoted_example_in_code_is_content():
    text = "r</think>Use:\n```\n<|tool_call_start|>[get_weather(city='a')]<|tool_call_end|>\n```\n"
    parser = make_output_parser(_request())
    assert _merge(_run(parser, [text])) == [
        {"reasoning_content": "r"}, {"content": text[len("r</think>"):]}]
    assert parser.tool_count == 0


# Differential check: the LFM transcription of ``OutputParser.push`` must
# behave exactly like the generic parser on the same stream with the generic
# markers, under every option and chunking.
DIFFERENTIAL = [
    ("<think>r</think>hi <|tool_call_start|>[get_weather(city='a')]<|tool_call_end|>", {}, "stop"),
    (TEXT + "after", {}, "stop"),
    (("r</think>\n<|tool_call_start|>[get_weather(city='a'), search(q='b')]<|tool_call_end|>\n"
      "<|tool_call_start|>[search(q='c')]<|tool_call_end|>\n"), {"parallel_tool_calls": False}, "stop"),
    ("r</think><|tool_call_start|>[get_weather(1)]<|tool_call_end|>tail",
     {"_tolerant_tool_markers": True}, "stop"),
    ("r</think><|tool_call_start|>[nope()]<|tool_call_end|>tail",
     {"_tolerant_tool_markers": True}, "stop"),
    ("r</think>x END y <|tool_call_start|>[search(q='a')]<|tool_call_end|>", {"stop": ["END"]}, "stop"),
    ("r</think>x <|tool_call_start|>[search(q='aEND')]<|tool_call_end|>", {"stop": ["END"]}, "stop"),
    ("r</think>x <|tool_call_start|>[search(q='a', n=", {}, "length"),
    ("r</think>x <|tool_call_start|>[search(q='a'", {"_tolerant_tool_markers": True}, "stop"),
    ("r <|tool_call_start|>[search(q='a')]<|tool_call_end|></think>`<|tool_call_start|>`", {}, "stop"),
    ("r</think>\n\n", {}, "stop"),
]


def _pieces(text):
    """Characters, with each marker kept whole (one token, as decoded)."""
    pattern = regex.compile(
        "|".join(regex.escape(m) for m in (TOOL_CALL_START, TOOL_CALL_END, THINKING_CLOSE_MARKER))
        + r"|[\s\S]"
    )
    return pattern.findall(text)


def _generic(text):
    return text.replace(TOOL_CALL_START, "<tool_call>").replace(TOOL_CALL_END, "</tool_call>")


def _events_or_error(parser, chunks, finish_reason):
    try:
        events = _strip_ids(_run(parser, chunks, finish_reason))
    except ValueError as exc:
        return ("error", str(exc))
    state = (
        parser.tool_count, parser.tool_call_parse_fallbacks,
        parser.tool_call_constraint_truncations, parser.truncated_tool_call,
        parser.stop_truncated_tool_call, parser.stopped, parser.stop_sequence,
    )
    return events, state


@pytest.mark.parametrize("text, extra, finish_reason", DIFFERENTIAL)
def test_matches_generic_output_parser(text, extra, finish_reason):
    request = _request(**extra)
    for chunks in ([text], _pieces(text)):
        lfm = _events_or_error(make_output_parser(request), chunks, finish_reason)
        generic_parser = OutputParser(
            chat=True, thinking=True, tools=TOOLS, parse_tool=parse_tool_call,
            stops=request.get("stop", ()),
            parallel_tool_calls=request.get("parallel_tool_calls", True),
            tolerant_tool_markers=request.get("_tolerant_tool_markers", False),
            think_close_separator="\r\n",
        )
        generic = _events_or_error(
            generic_parser, [_generic(chunk) for chunk in chunks], finish_reason
        )
        if lfm[0] == "error":
            assert generic[0] == "error"
            continue
        lfm_events = json.loads(_generic(json.dumps(lfm[0])))
        assert lfm_events == generic[0]
        assert lfm[1] == generic[1]


def test_parser_is_an_output_parser():
    parser = make_output_parser(_request())
    assert isinstance(parser, LfmOutputParser) and isinstance(parser, OutputParser)
    assert parser.channel == "reasoning_content"
    assert parser.partial_names is partial_function_names


# ---------------------------------------------------------------------------
# Template inputs


def test_normalizer_decodes_json_string_arguments_without_mutating():
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function",
             "function": {"name": "f", "arguments": '{"x": 1}'}},
            {"id": "b", "type": "function", "function": {"name": "g", "arguments": ""}},
            {"id": "c", "type": "function", "function": {"name": "h", "arguments": {"y": 2}}},
            {"id": "d", "type": "function", "function": {"name": "k"}},
        ]},
    ]
    original = json.dumps(messages)
    out = normalize_messages_for_template(messages)
    assert json.dumps(messages) == original
    assert [c["function"]["arguments"] for c in out[0]["tool_calls"]] == [
        {"x": 1}, {}, {"y": 2}, {}]


@pytest.mark.parametrize("arguments", ["{not json", "[1, 2]", "3", ["x"]])
def test_normalizer_rejects_non_object_arguments(arguments):
    messages = [{"role": "assistant", "tool_calls": [
        {"type": "function", "function": {"name": "f", "arguments": arguments}}]}]
    with pytest.raises(ValueError):
        normalize_messages_for_template(messages)


@pytest.mark.parametrize("name", ["get weather", "f(x)", "a,b", "", None])
def test_validate_tool_request_rejects_unrepresentable_names(name):
    tool = {"type": "function", "function": {"name": name, "parameters": {}}}
    with pytest.raises(ValueError):
        validate_tool_request({"tools": [tool]})


def test_validate_tool_request_rejects_unrepresentable_required_parameter():
    tool = {"type": "function", "function": {"name": "f", "parameters": {
        "type": "object", "properties": {"a b": {"type": "string"}}, "required": ["a b"]}}}
    with pytest.raises(ValueError):
        validate_tool_request({"tools": [tool]})
    validate_tool_request({"tools": TOOLS})


@pytest.fixture(scope="module")
def tokenizer():
    if not (MODEL / "tokenizer.json").exists():
        pytest.skip("LFM2.5 artifact not present")
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)


def test_render_chat_with_prior_call_and_tool_result(tokenizer):
    request = {
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "get_weather",
                             "arguments": '{"city": "Paris", "units": "c"}'},
            }]},
            {"role": "tool", "tool_call_id": "c1", "content": '{"t": 20}'},
        ],
        "tools": [WEATHER],
    }
    tools_json = json.dumps(WEATHER)
    assert render_chat(tokenizer, request, tokenize=False) == (
        "<|startoftext|><|im_start|>system\nBe brief.\nList of tools: ["
        + tools_json + "]<|im_end|>\n"
        "<|im_start|>user\nWeather?<|im_end|>\n"
        "<|im_start|>assistant\n<|tool_call_start|>[get_weather(city='Paris', units='c')]"
        "<|tool_call_end|><|im_end|>\n"
        "<|im_start|>tool\n{\"t\": 20}<|im_end|>\n"
        "<|im_start|>assistant\n<think>"
    )
    # tool_choice "none": no tool list, history still renders.
    rendered = render_chat(tokenizer, {**request, "tool_choice": "none"}, tokenize=False)
    assert "List of tools" not in rendered and "get_weather(city='Paris'" in rendered


def test_raw_template_refuses_json_string_arguments(tokenizer):
    messages = [{"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}}]}]
    with pytest.raises(Exception, match="must be a mapping"):
        tokenizer.apply_chat_template(messages, tokenize=False)


def test_markers_are_single_tokens(tokenizer):
    ids = tokenizer.encode(TOOL_CALL_START + TOOL_CALL_END + THINKING_CLOSE_MARKER,
                           add_special_tokens=False)
    assert ids == [124905, 124906, 124902]


ROUND_TRIP = [
    {"s": "it's \"quoted\" \\ back\\slash\nnew\rline\ttab", "e": ""},
    {"u": "café ☃ \U0001f600", "lt": "<|tool_call_end|> </think> ) ] ,"},
    {"i": -7, "f": 2.5, "big": 12345678901234, "tiny": 1e-07, "t": True, "n": None},
    {"l": [1, "a", None, True, {"k": [False]}], "d": {"x": {"y": "z'\""}, "e": {}}},
]


@pytest.mark.parametrize("arguments", ROUND_TRIP)
def test_template_rendering_round_trips_through_the_parser(tokenizer, arguments):
    request = {"messages": [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "f", "arguments": json.dumps(arguments)}},
            {"type": "function", "function": {"name": "g", "arguments": "{}"}},
        ]},
    ]}
    text = render_chat(tokenizer, request, tokenize=False)
    start = text.index(TOOL_CALL_START) + len(TOOL_CALL_START)
    end = text.rindex(TOOL_CALL_END)
    parsed = parse_tool_call(text[start:end])
    assert parsed == [{"name": "f", "arguments": arguments}, {"name": "g", "arguments": {}}]
    # Through the streaming parser too, char by char.
    parser = make_output_parser({"messages": [], "tools": [
        {"type": "function", "function": {"name": "f"}},
        {"type": "function", "function": {"name": "g"}}]})
    events = _run(parser, list("r</think>" + text[start - len(TOOL_CALL_START):end + len(TOOL_CALL_END)]))
    calls = [e["tool_calls"][0]["function"] for e in events if "tool_calls" in e]
    assert [json.loads(call["arguments"]) for call in calls] == [arguments, {}]


# ---------------------------------------------------------------------------
# Grammar


def _admits(pattern, text):
    """Serving's view: the scanner pattern, and the exact automaton agreeing."""
    from mlx2.structured_automaton import AutomatonUnsupported, automaton_for
    from mlx2.structured_output import _request_constraint

    compiled = _request_constraint(None, None, pattern, None).pattern
    admitted = compiled.fullmatch(text) is not None
    try:
        automaton = automaton_for(compiled)
    except AutomatonUnsupported:
        return admitted
    assert automaton.fullmatch(text) == admitted, text
    return admitted


def _block(body):
    return TOOL_CALL_START + body + TOOL_CALL_END


def test_required_grammar_accepts_template_calls_and_rejects_others():
    pattern = make_tool_constraint({"tools": TOOLS, "tool_choice": "required"})
    good = [
        "[get_weather(city='Paris')]",
        "[get_weather(city='it\\'s', units='c', days=3)]",
        "[get_weather(city='a', days=-1), search(q='x', flags=[1, 2], opts={\"a\": true})]",
        "[search(q='x', n=0)]",
    ]
    for body in good:
        assert _admits(pattern, _block(body)), body
        assert parse_tool_call(body)
    bad = [
        "[get_weather()]",  # missing required
        "[get_weather(units='c', city='a')]",  # out of declared order
        "[get_weather(city=\"a\")]",  # not the template's quoting
        "[get_weather(city='a', days='3')]",  # wrong type
        "[get_weather(city='a', extra=1)]",  # unknown argument
        "[delete(city='a')]",  # undeclared tool
        "[get_weather(city='a')] ",
        "[]",
        "[get_weather(city='a\nb')]",
        "[search(q='x', n=None)]",  # integer, not nullable
    ]
    for body in bad:
        assert not _admits(pattern, _block(body)), body
    assert not _admits(pattern, "[get_weather(city='a')]")  # markers required


def test_named_and_single_call_grammar():
    pattern = make_tool_constraint({
        "tools": TOOLS, "tool_choice": {"type": "function", "function": {"name": "search"}},
        "parallel_tool_calls": False,
    })
    assert _admits(pattern, _block("[search(q='x')]"))
    assert not _admits(pattern, _block("[get_weather(city='x')]"))
    assert not _admits(pattern, _block("[search(q='x'), search(q='y')]"))


def test_strict_grammar_constrains_values_by_schema():
    tool = {"type": "function", "function": {"name": "f", "strict": True, "parameters": {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["a", "it's"]},
            "count": {"type": ["integer", "null"]},
            "ok": {"type": "boolean"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "short": {"type": "string", "maxLength": 3},
        },
        "required": ["mode", "count", "ok", "tags", "short"],
        "additionalProperties": False,
    }}}
    pattern = constrained_tool_grammar([tool], "required")
    good = "[f(mode='it\\'s', count=None, ok=True, tags=[\"x\",\"y\"], short='abc')]"
    assert _admits(pattern, _block(good))
    parsed = parse_tool_call(good)
    assert parsed["arguments"] == {"mode": "it's", "count": None, "ok": True,
                                   "tags": ["x", "y"], "short": "abc"}
    for body in [
        "[f(mode='b', count=None, ok=True, tags=[], short='abc')]",
        "[f(mode='a', count=1.5, ok=True, tags=[], short='abc')]",
        "[f(mode='a', count=1, ok=true, tags=[], short='abc')]",
        "[f(mode='a', count=1, ok=True, tags=[1], short='abc')]",
        "[f(mode='a', count=1, ok=True, tags=[], short='abcd')]",
    ]:
        assert not _admits(pattern, _block(body)), body


@pytest.mark.parametrize("schema", [
    {"type": "object", "properties": {"a": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
     "required": ["a"], "additionalProperties": False},
    {"type": "object", "properties": {"a": {"type": "string", "pattern": "x"}},
     "required": ["a"], "additionalProperties": False},
    {"type": "object", "properties": {"a b": {"type": "string"}},
     "required": ["a b"], "additionalProperties": False},
])
def test_strict_grammar_fails_closed_on_unrepresentable_schemas(schema):
    tool = {"type": "function", "function": {"name": "f", "strict": True, "parameters": schema}}
    with pytest.raises(ValueError):
        constrained_tool_grammar([tool], "required")


def test_free_argument_grammar_for_schemaless_tools():
    tool = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
    pattern = constrained_tool_grammar([tool], "required")
    for body in ["[f()]", "[f(a=1, b='x', c=[1, {\"k\": null}], d=None)]"]:
        assert _admits(pattern, _block(body)), body
    assert not _admits(pattern, _block("[f(1)]"))


def test_auto_and_answer_shapes_compose():
    from mlx2.tool_grammar import plan_tool_grammar

    strict = {**WEATHER, "function": {**WEATHER["function"], "strict": True, "parameters": {
        **WEATHER["function"]["parameters"], "additionalProperties": False}}}
    for extra in ({}, {"response_format": {"type": "json_object"}}):
        pattern, status, _receipt = plan_tool_grammar(
            {"tools": [strict], **extra}, make_tool_constraint,
            open_marker=TOOL_CALL_START, leading_whitespace=True,
        )
        assert status == "engaged", (extra, status)
        call = _block("[get_weather(city='a')]")
        assert _admits(pattern, call if extra else "Sure. " + call + " done")
        if not extra:
            assert not _admits(pattern, "text <|tool_call_start|>[oops")


def test_constraint_is_none_unless_forced():
    assert make_tool_constraint({"tools": TOOLS}) is None
    assert make_tool_constraint({"tools": TOOLS, "tool_choice": "auto"}) is None
    assert make_tool_constraint({"tools": TOOLS, "tool_choice": "required"})


def test_every_grammar_compiles_to_the_exact_automaton():
    from mlx2.structured_automaton import automaton_for

    for parallel in (True, False):
        pattern = constrained_tool_grammar(TOOLS, "required", parallel_tool_calls=parallel)
        automaton_for(regex.compile(rf"(?:{pattern})"))


def test_answer_drops_the_newlines_the_model_writes_after_think_close():
    from mlx2.adapters.lfm25_output import make_output_parser

    request = {"messages": [{"role": "user", "content": "hi"}]}
    for chunks in (["reasoning</think>\n\n51"], list("reasoning</think>\n\n51")):
        parser = make_output_parser(request)
        events = []
        for chunk in chunks:
            events.extend(parser.push(chunk))
        events.extend(parser.finish("", "stop"))
        content = "".join(e.get("content", "") for e in events)
        reasoning = "".join(e.get("reasoning_content", "") for e in events)
        assert (reasoning, content) == ("reasoning", "51")
    # Newlines inside the answer are kept.
    parser = make_output_parser(request)
    events = parser.push("r</think>\n\na\n\nb") + parser.finish("", "stop")
    assert "".join(e.get("content", "") for e in events) == "a\n\nb"


@pytest.mark.parametrize("value", ["'a\\x00b'", "'\\u0000'", "'\\0'", '["a\\u0000"]', "{'k\\x00': 1}"])
def test_parser_refuses_nul_smuggled_through_escapes(value):
    with pytest.raises(ValueError, match="NUL"):
        parse_tool_call(f"[f(x={value})]")


def test_parser_keeps_other_escapes():
    assert parse_tool_call("[f(x='a\\tb\\'c')]")["arguments"] == {"x": "a\tb'c"}


def test_unrepresentable_optional_parameter_is_refused():
    from mlx2.adapters.lfm25_output import validate_tool_request

    tool = {"type": "function", "function": {"name": "f", "parameters": {
        "type": "object", "properties": {"q": {"type": "string"},
                                         "output format": {"type": "string"}},
        "required": ["q"]}}}
    with pytest.raises(ValueError, match="output format"):
        validate_tool_request({"tools": [tool]})


def _channels(events):
    out = {}
    for event in events:
        for key, value in event.items():
            if isinstance(value, str):
                out[key] = out.get(key, "") + value
    return out


CHAT = {"messages": [{"role": "user", "content": "x"}]}


def test_reasoning_off_holds_late_reasoning_until_the_model_closes_it():
    # Pierre, 2026-10-15: with reasoning off the always-reasoning model may
    # still reason and write its own </think>; hold the output back until the
    # marker (held text becomes reasoning) or the end (it is the answer).
    parser = make_output_parser(CHAT, thinking=False)
    assert parser.push("We need to read ") == []
    assert parser.push("the note.") == []
    events = parser.push("</think>\n\nNELL-1")
    assert events[0] == {"reasoning_content": "We need to read the note."}
    assert parser.push(" done") == [{"content": " done"}]
    out = _channels(events + parser.finish("", "stop"))
    assert out["content"] == "NELL-1"
    assert parser.late_reasoning_chars == len("We need to read the note.")


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_reasoning_off_releases_a_direct_answer_at_the_end(finish_reason):
    parser = make_output_parser(CHAT, thinking=False)
    events = parser.push("\n\nNEEDLE-1") + parser.push(" is the code.")
    assert events == []
    out = _channels(events + parser.finish("", finish_reason))
    assert out == {"content": "NEEDLE-1 is the code."}
    assert parser.late_reasoning_chars == 0


def test_reasoning_on_and_completions_do_not_hold_back():
    on = make_output_parser(CHAT, thinking=True)
    assert on.push("plan</think>\n\nanswer") != []
    completion = make_output_parser({"prompt": "x"}, thinking=False)
    assert completion.push("raw text") == [{"content": "raw text"}]


@pytest.mark.parametrize("split", [1, 2, 3, 5, 7, 11])
def test_late_reasoning_hold_finds_the_close_at_any_chunk_boundary(split):
    # Sweep 2026-10-16: the hold now scans only a carried tail plus the new
    # chunk; a marker straddling any boundary must still be found once.
    text = "We need to read it" + THINKING_CLOSE_MARKER + "\n\nNEEDLE-7D13C8-3</think> tail"
    chunks = [text[i:i + split] for i in range(0, len(text), split)]
    parser = make_output_parser({"messages": [{"role": "user", "content": "x"}]}, thinking=False)
    channels = _channels(_run(parser, chunks))
    assert channels == {
        "reasoning_content": "We need to read it",
        "content": "NEEDLE-7D13C8-3</think> tail",
    }
    assert parser.late_reasoning_chars == len("We need to read it")


def test_late_reasoning_hold_releases_unmarked_text_at_the_end():
    parser = make_output_parser({"messages": [{"role": "user", "content": "x"}]}, thinking=False)
    assert _channels(_run(parser, list("plain answer </thin"))) == {"content": "plain answer </thin"}


@pytest.mark.parametrize("tool", [
    {"type": "function", "function": {"name": "f"}},
    {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "required": ["a"]}}},
])
def test_forced_free_arguments_admitted_by_the_grammar_parse(tool):
    # Sweep 2026-10-16: the grammar's free-argument tail admits a repeated
    # name; the parser used to reject it, failing a grammar-valid forced call.
    request = {"messages": [{"role": "user", "content": "x"}], "tools": [tool], "tool_choice": "required"}
    text = f"{TOOL_CALL_START}[f(a=1, a=2)]{TOOL_CALL_END}"
    assert regex.fullmatch(make_tool_constraint(request), text)
    parser = make_output_parser(request, thinking=False)
    calls = [e for e in parser.push(text) + parser.finish("", "stop") if "tool_calls" in e]
    assert json.loads(calls[0]["tool_calls"][0]["function"]["arguments"]) == {"a": 2}


def test_late_reasoning_hold_scans_only_a_bounded_tail_per_push():
    # Codex review of ADAPTERS-2: the hold must not recopy or rescan the
    # whole held answer on every token; the marker search's examined
    # characters stay linear in the input.
    parser = make_output_parser({"messages": [{"role": "user", "content": "x"}]}, thinking=False)
    total = 0
    for index in range(2000):
        chunk = f"t{index} "
        total += len(chunk)
        assert parser.push(chunk) == []
    assert parser.held_scan_chars <= total + 2000 * (len(THINKING_CLOSE_MARKER) - 1)
    assert _channels(parser.finish("", "stop"))["content"].startswith("t0 t1 ")


def test_reasoning_off_hold_stops_at_a_client_stop_string():
    # Sweep 2026-10-16 ADAPTERS-1 (Pierre's decision): the hold kept decoding
    # past a client stop string until EOS/max_tokens; it now stops there.
    request = {"messages": [{"role": "user", "content": "x"}], "stop": ["\n"]}
    parser = make_output_parser(request, thinking=False)
    events = []
    for index, token in enumerate(["Answer", " one", "\n", "Answer two"]):
        events += parser.push(token)
        if parser.stopped:
            break
    assert index == 2 and parser.stopped and parser.stop_sequence == "\n"
    assert _channels(events) == {"content": "Answer one"}
    assert parser.late_reasoning_stop_releases == 1


def test_reasoning_off_hold_still_splits_reasoning_closed_before_a_stop():
    request = {"messages": [{"role": "user", "content": "x"}], "stop": ["\n"]}
    parser = make_output_parser(request, thinking=False)
    events = parser.push("read it</think>") + parser.push("NEEDLE-1\nmore")
    assert parser.stopped
    assert _channels(events) == {"reasoning_content": "read it", "content": "NEEDLE-1"}
    assert parser.late_reasoning_stop_releases == 0
