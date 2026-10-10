"""A stop-cut call's committed name is read by its grammar's own rules.

Codex integration review, item 2 (P1).  Tool admission accepts any nonempty
name up to 128 characters, but the name recovery behind the stop-cut receipt
(``stop_truncated_tool_call``) was one generic regex: ``[\\w.-]+`` for the
markup and plain-name forms, and the unescaped contents of a JSON string.  A
legal name such as ``db:lookup``, ``a/b``, ``my tool`` or an escaped JSON
name decoded to ``None`` after its delimiter had been seen, the receipt said
"cut before the name", and ``enforce_tool_contract`` accepted a required or
named choice without ever seeing the wrong function the model had committed
to.

Now each grammar publishes the committed name through its actual parser
rules (the JSON form decodes the name string incrementally, escapes
included), and a name delimiter followed by a name the grammar cannot read
fails the terminal contract (502, as for a missing required call) instead of
degrading to ``None``.  A cut before the delimiter stays exempt: the api
group's documented case, Qwen ``<tool_call>\\n`` cut by ``stop: ["\\n"]``.
"""

import json

import pytest
from test_serving_contract import TOOLS
from test_sweep_20261009_api_stop_cut_required_call import (  # noqa: F401
    CUT,
    StopCutEngine,
    _chunks,
    _drive,
    _post,
    endpoint,
)

from mlx2.adapters.muse_glimmer_output import MuseOutputParser
from mlx2.adapters.xing_output import parse_tool_block
from mlx2.openai_compat import ToolContractError, enforce_tool_contract
from mlx2.output import OutputParser
from mlx2.runtime.tool_parsers.laguna import parse_tool_call as laguna_parse
from mlx2.runtime.tool_parsers.qwen3_coder import parse_tool_call as qwen_parse
from mlx2.server import collect_nonstream_job
from mlx2.serving import Job

NAMED = {"type": "function", "function": {"name": "weather"}}
UNREADABLE = {"function": None, "name_unreadable": True}

# Names tool admission accepts that the generic ``[\w.-]+`` rule did not.
NAMES = ["db:lookup", "a/b", "my tool", 'say"hi"', "a\\b", "café"]


def _tools(*names):
    return TOOLS + [
        {"type": "function", "function": {"name": name, "parameters": {}}}
        for name in names
    ]


def _choice(name):
    return {"type": "function", "function": {"name": name}}


def _qwen(name):
    return (
        f"<tool_call>\n<function={name}>\n<parameter=q>\nParis\n</parameter>\n"
        "</function>\n</tool_call>"
    )


def _xing_plain(name):
    return f"<tool_call>{name}<param_key>q</param_key><param_value>Paris</param_value></tool_call>"


def _xing_bare(name):
    return f'<tool_call>{name}{{"q": "Paris"}}</tool_call>'


def _xing_json(name):
    return f'<tool_call>{{"name": {json.dumps(name)}, "arguments": {{"q": "Paris"}}}}</tool_call>'


def _laguna(name):
    return f"<tool_call>{name}<arg_key>q</arg_key><arg_value>Paris</arg_value></tool_call>"


GRAMMARS = {
    "qwen": (qwen_parse, _qwen),
    "xing-plain": (parse_tool_block, _xing_plain),
    "xing-bare": (parse_tool_block, _xing_bare),
    "xing-json": (parse_tool_block, _xing_json),
    "laguna": (laguna_parse, _laguna),
}


def _parser(parse_tool, tools, stops):
    return OutputParser(
        chat=True, tools=tools, parse_tool=parse_tool, stops=stops,
        constrained_tools=True,
    )


def _cut(parse_tool, call, tools, stop="Paris"):
    parser = _drive(_parser(parse_tool, tools, [stop]), call)
    assert parser.stopped and parser.tool_count == 0
    return parser.stop_truncated_tool_call


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("grammar", list(GRAMMARS))
def test_every_grammar_reads_the_committed_name_the_admission_accepts(grammar, name):
    parse_tool, call = GRAMMARS[grammar]
    if grammar == "qwen" and " " in name:
        pytest.skip("Qwen markup cannot carry a space inside <function=...>")
    tools = _tools(name)
    record = _cut(parse_tool, call(name), tools)
    assert record == {"function": name}
    # The model committed to a function other than the one selected: the
    # contract sees it (this was the fail-open: the name read as None).
    with pytest.raises(ToolContractError, match="exclusively"):
        enforce_tool_contract(
            {"tools": tools, "tool_choice": NAMED}, [], finish_reason="stop",
            stop_truncated_tool_call=record,
        )
    # The cut call was the one selected, or any declared call under required.
    enforce_tool_contract(
        {"tools": tools, "tool_choice": _choice(name)}, [], finish_reason="stop",
        stop_truncated_tool_call=record,
    )
    enforce_tool_contract(
        {"tools": tools, "tool_choice": "required"}, [], finish_reason="stop",
        stop_truncated_tool_call=record,
    )


@pytest.mark.parametrize("grammar", ["qwen", "xing-plain", "xing-json", "laguna"])
def test_an_undeclared_committed_name_fails_under_every_grammar(grammar):
    parse_tool, call = GRAMMARS[grammar]
    record = _cut(parse_tool, call("db:lookup"), TOOLS)
    assert record == {"function": "db:lookup"}
    for choice in ("required", NAMED, "auto"):
        with pytest.raises(ToolContractError, match="undeclared"):
            enforce_tool_contract(
                {"tools": TOOLS, "tool_choice": choice}, [], finish_reason="stop",
                stop_truncated_tool_call=record,
            )


def test_xing_bare_name_is_read_before_its_declaration_is_judged():
    # The grammar resolves ``name{`` against the declared names; an undeclared
    # prefix is still the name the model wrote, and the contract fails it.
    record = _cut(parse_tool_block, _xing_bare("db:lookup"), TOOLS)
    assert record == {"function": "db:lookup"}
    with pytest.raises(ToolContractError, match="undeclared"):
        enforce_tool_contract(
            {"tools": TOOLS, "tool_choice": "required"}, [], finish_reason="stop",
            stop_truncated_tool_call=record,
        )
    # A declared name that is a prefix of the written one is not the name.
    record = _cut(parse_tool_block, _xing_bare("weatherx"), TOOLS)
    assert record == {"function": "weatherx"}


@pytest.mark.parametrize(
    "partial, expected",
    [
        # Escapes decode as JSON does: the name is the decoded string.
        ('{"name": "say\\"hi\\"", "arguments": {"q": "', 'say"hi"'),
        ('{"name": "a\\\\b", "arguments": {"q": "', "a\\b"),
        ('{"name": "caf\\u00e9", "arguments": {"q": "', "café"),
        ('{"name": "a\\/b", "arguments": {"q": "', "a/b"),
        ('{"n\\u0061me": "db:lookup", "arguments": {"q": "', "db:lookup"),
        # Member order does not bind the name: a nested "name" is not it, the
        # top-level one is wherever it appears.
        ('{"arguments": {"name": "weather", "q": "', None),
        ('{"arguments": {"name": "weather"}, "name": "db:lookup", "q": "', "db:lookup"),
        # Cut inside the name string, or before the member: not committed.
        ('{"name": "db:lo', None),
        ('{"name": "db:lookup\\', None),
        ('{"name": "db:lookup\\u00', None),
        ('{"name"', None),
        ('{"name": ', None),
        ("{", None),
    ],
)
def test_xing_json_name_is_decoded_incrementally(partial, expected):
    parser = _parser(parse_tool_block, TOOLS, ["Paris"])
    _drive(parser, "<tool_call>" + partial + "Paris")
    assert parser.stopped
    assert parser.stop_truncated_tool_call == {"function": expected}


@pytest.mark.parametrize(
    "grammar, partial",
    [
        # Qwen: the name run ended with nothing in it.
        ("qwen", "<tool_call>\n<function=>\n<parameter=q>\n"),
        ("qwen", "<tool_call>\n<function=<parameter=q>\n"),
        # Xing plain / Laguna: the argument marker came before any name.
        ("xing-plain", "<tool_call><param_key>q</param_key><param_value>"),
        ("laguna", "<tool_call><arg_key>q</arg_key><arg_value>"),
        # Xing JSON: the name member holds something the grammar rejects.
        ("xing-json", '<tool_call>{"name": 5, "arguments": {"q": "'),
        ("xing-json", '<tool_call>{"name": null, "arguments": {"q": "'),
        ("xing-json", '<tool_call>{"name": "a\\x", "arguments": {"q": "'),
        ("xing-json", '<tool_call>{"name": ["a"], "arguments": {"q": "'),
        ("xing-json", '<tool_call>{"name": "a", "name": 5, "arguments": {"q": "'),
        ("xing-json", '<tool_call>{"name": "ok" "arguments": {"q": "'),
    ],
)
def test_a_delimiter_without_a_readable_name_fails_the_contract(grammar, partial):
    parse_tool, _call = GRAMMARS[grammar]
    parser = _parser(parse_tool, TOOLS, ["Paris"])
    _drive(parser, partial + "Paris")
    assert parser.stopped and parser.tool_count == 0
    assert parser.stop_truncated_tool_call == UNREADABLE
    for choice in ("required", NAMED, "auto"):
        with pytest.raises(ToolContractError, match="could not be read"):
            enforce_tool_contract(
                {"tools": TOOLS, "tool_choice": choice}, [], finish_reason="stop",
                stop_truncated_tool_call=parser.stop_truncated_tool_call,
            )


def test_qwen_name_ends_at_the_grammars_delimiters():
    # ``<function=my tool>``: the grammar reads ``my`` and the rest as markup
    # it rejects; the committed name is ``my``, undeclared.
    record = _cut(qwen_parse, _qwen("my tool"), _tools("my tool"))
    assert record == {"function": "my"}
    # The grammar also accepts whitespace or an opener as the name's end.
    tools = _tools("db:lookup")
    for text in (
        "<tool_call>\n<function=db:lookup\n<parameter=q>\nParis",
        "<tool_call>\n<function=db:lookup<parameter=q>\nParis",
        "<tool_call>\n<function= db:lookup>\n<parameter=q>\nParis",
    ):
        assert _cut(qwen_parse, text, tools) == {"function": "db:lookup"}
    # Cut inside the name, or right after the opener: not committed.
    assert _cut(qwen_parse, "<tool_call>\n<function=db:lookParis", tools) == CUT
    assert _cut(qwen_parse, "<tool_call>\n<function=Paris", tools) == CUT
    assert _cut(qwen_parse, "<tool_call>\nParis", tools) == CUT


def test_qwen_block_with_several_functions_judges_each_committed_name():
    tools = _tools("db:lookup")
    text = (
        "<tool_call>\n<function=weather>\n<parameter=city>\nRome\n</parameter>\n"
        "</function>\n<function=db:lookup>\n<parameter=q>\nParis"
    )
    record = _cut(qwen_parse, text, tools)
    assert record == {"function": "weather", "functions": ["weather", "db:lookup"]}
    with pytest.raises(ToolContractError, match="exclusively"):
        enforce_tool_contract(
            {"tools": tools, "tool_choice": NAMED}, [], finish_reason="stop",
            stop_truncated_tool_call=record,
        )
    enforce_tool_contract(
        {"tools": tools, "tool_choice": "required"}, [], finish_reason="stop",
        stop_truncated_tool_call=record,
    )
    with pytest.raises(ToolContractError, match="undeclared"):
        enforce_tool_contract(
            {"tools": TOOLS, "tool_choice": "required"}, [], finish_reason="stop",
            stop_truncated_tool_call=record,
        )
    # The second function's name not yet committed: the first is judged.
    text = (
        "<tool_call>\n<function=weather>\n<parameter=city>\nRome\n</parameter>\n"
        "</function>\n<function=db:lParis"
    )
    assert _cut(qwen_parse, text, tools) == {"function": "weather"}


@pytest.mark.parametrize("grammar", ["xing-plain", "laguna"])
def test_plain_name_grammars_bind_at_their_own_delimiters(grammar):
    parse_tool, _call = GRAMMARS[grammar]
    tools = _tools("db:lookup", "my tool")
    # Whitespace around the name is not part of it.
    marker = "<param_key>" if grammar == "xing-plain" else "<arg_key>"
    text = f"<tool_call>\n  my tool \n{marker}q</{marker[1:]}Paris"
    assert _cut(parse_tool, text, tools) == {"function": "my tool"}
    # Cut inside the name, or inside the marker: not committed.
    for text in ("<tool_call>db:looParis", f"<tool_call>db:lookup{marker[:4]}Paris"):
        assert _cut(parse_tool, text, tools) == CUT
    # After the name with no marker yet.  Laguna's parser reads a block with
    # no argument marker as a call on its first line, so the first newline
    # commits the name (codex round 1); Xing's binds a bare name only when
    # the block closes, so nothing is committed.
    record = _cut(parse_tool, "<tool_call>db:lookup\nParis", tools)
    if grammar == "laguna":
        assert record == {"function": "db:lookup"}
        with pytest.raises(ToolContractError, match="exclusively"):
            enforce_tool_contract(
                {"tools": tools, "tool_choice": NAMED}, [], finish_reason="stop",
                stop_truncated_tool_call=record,
            )
        assert _cut(parse_tool, "<tool_call>\n\n db:lookup \n\nParis", tools) == {
            "function": "db:lookup"
        }
    else:
        assert record == CUT


def test_qwen_walks_past_an_unclosed_function_to_later_committed_names():
    # Codex round 1: the reader stopped at the first unclosed function, so a
    # later delimited name in the same block was never judged.
    tools = _tools("db:lookup")
    text = (
        "<tool_call>\n<function=weather>\n<parameter=city>\nRome\n</parameter>\n"
        "<function=db:lookup>\n<parameter=q>\nParis"
    )
    record = _cut(qwen_parse, text, tools)
    assert record == {"function": "weather", "functions": ["weather", "db:lookup"]}
    with pytest.raises(ToolContractError, match="exclusively"):
        enforce_tool_contract(
            {"tools": tools, "tool_choice": NAMED}, [], finish_reason="stop",
            stop_truncated_tool_call=record,
        )
    # An opener quoted inside a closed value is value text, as the parser
    # reads it; one inside the value the cut landed in is too.
    quoted = (
        "<tool_call>\n<function=weather>\n<parameter=city>\nsee <function=db:lookup>"
        "\n</parameter>\n<parameter=q>\nParis"
    )
    assert _cut(qwen_parse, quoted, tools) == {"function": "weather"}
    open_value = "<tool_call>\n<function=weather>\n<parameter=city>\n<function=db:lookup>Paris"
    assert _cut(qwen_parse, open_value, tools) == {"function": "weather"}


def test_xing_json_repeated_name_cut_inside_its_string_is_unreadable():
    # Codex round 1: ``json.loads`` keeps the last ``name``; a repeat cut
    # inside its string leaves the name the call ends with unknown, and the
    # earlier one must not read as "no name yet".
    parser = _parser(parse_tool_block, TOOLS, ["Paris"])
    _drive(parser, '<tool_call>{"name": "news", "name": "weParis')
    assert parser.stopped and parser.stop_truncated_tool_call == UNREADABLE
    parser = _parser(parse_tool_block, TOOLS, ["Paris"])
    _drive(parser, '<tool_call>{"name": "news", "name": "weather", "arguments": {"q": "Paris')
    assert parser.stop_truncated_tool_call == {"function": "weather"}


MUSE_TOOLS = [
    {"type": "function", "function": {"name": "functions.weather", "parameters": {}}},
    {"type": "function", "function": {"name": "functions.db_lookup", "parameters": {}}},
]


def _atem(name):
    return (
        "to=functions.weather<|message|><atem:function_calls>"
        f'<atem:invoke name="{name}"><atem:parameter name="q">Paris'
        "</atem:parameter></atem:invoke></atem:function_calls><|eot|>"
    )


def test_muse_reads_the_name_by_its_invoke_rule():
    def cut(text, stop="Paris"):
        parser = _drive(MuseOutputParser(chat=True, tools=MUSE_TOOLS, stops=[stop]), text)
        assert parser.stopped and parser.tool_count == 0
        return parser.stop_truncated_tool_call

    record = cut(_atem("functions.db_lookup"))
    assert record == {"function": "functions.db_lookup"}
    with pytest.raises(ToolContractError, match="exclusively"):
        enforce_tool_contract(
            {"tools": MUSE_TOOLS, "tool_choice": _choice("functions.weather")}, [],
            finish_reason="stop", stop_truncated_tool_call=record,
        )
    # The ATEM invoke rule admits only ``[\w.-]+``: a name it cannot carry
    # is unreadable once its closing quote is seen, never "no name yet".
    for name in ("db:lookup", "a/b", "my tool", 'say"hi"', "a\\b", ""):
        assert cut(_atem(name)) == UNREADABLE, name
    with pytest.raises(ToolContractError, match="could not be read"):
        enforce_tool_contract(
            {"tools": MUSE_TOOLS, "tool_choice": "required"}, [],
            finish_reason="stop", stop_truncated_tool_call=UNREADABLE,
        )
    # Cut inside the name, or before the invoke: not committed.
    assert cut(_atem("functions.db_lookup"), stop="db_") == CUT
    assert cut(_atem("functions.db_lookup"), stop="<atem:invoke") == CUT
    # Several invokes in one block: every committed name is carried.
    text = (
        "to=functions.weather<|message|><atem:function_calls>"
        '<atem:invoke name="functions.weather"></atem:invoke>'
        '<atem:invoke name="functions.db_lookup"><atem:parameter name="q">Paris'
    )
    assert cut(text) == {
        "function": "functions.weather",
        "functions": ["functions.weather", "functions.db_lookup"],
    }


def test_tool_choice_none_rejects_a_stop_cut_call_too():
    """Integration review round 2: ``tool_choice: none`` rejected completed
    calls only; a call the model committed to before a client stop cut it
    was merely checked for membership in the declared tools."""
    body = {"tools": TOOLS, "tool_choice": "none"}
    for record in ({"function": "weather"}, {"function": None}, UNREADABLE):
        with pytest.raises(ToolContractError, match="tool_choice was none|could not be read"):
            enforce_tool_contract(
                body, [], finish_reason="stop", stop_truncated_tool_call=record
            )
    # No record, no calls: a plain answer under ``none`` is fine.
    enforce_tool_contract(body, [], finish_reason="stop", stop_truncated_tool_call=None)


def test_contract_fails_closed_on_a_record_it_cannot_read():
    body = {"tools": TOOLS, "tool_choice": "required"}
    for record in (
        UNREADABLE,
        {},  # no "function" member at all
        {"function": 5},
        {"function": ""},
        {"function": "weather", "functions": ["weather", 5]},
        {"function": "weather", "functions": "weather"},
        # Codex round 1: ``functions`` is read only in its canonical shape
        # (two or more names, the first being ``function``); a record whose
        # two members disagree cannot say what the model committed to.
        {"function": "news", "functions": []},
        {"function": "news", "functions": ["weather"]},
        {"function": "news", "functions": ["weather", "news"]},
        # Integration review round 2: a present ``functions`` that is not a
        # list (an explicit null) is not "absent"; the shape is unreadable.
        {"function": "weather", "functions": None},
        {"function": None, "functions": None},
    ):
        with pytest.raises(ToolContractError, match="could not be read"):
            enforce_tool_contract(
                body, [], finish_reason="stop", stop_truncated_tool_call=record
            )
    # The record is still evidence only under a stop finish.
    call = {"function": {"name": "weather", "arguments": json.dumps({"city": "Paris"})}}
    enforce_tool_contract(
        body, [call], finish_reason="tool_calls", stop_truncated_tool_call=UNREADABLE
    )


def test_http_terminal_fails_an_unreadable_or_wrong_committed_name(endpoint):  # noqa: F811
    engine, base = endpoint
    engine.stop_sequence = "\n"
    two = _tools("db:lookup")
    body = {
        "model": "fixture",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": two,
        "tool_choice": NAMED,
        "stop": ["\n"],
    }
    for record, expected in (
        (CUT, 200),
        ({"function": "weather"}, 200),
        ({"function": "db:lookup"}, 502),
        ({"function": "weather", "functions": ["weather", "db:lookup"]}, 502),
        (UNREADABLE, 502),
    ):
        engine.stop_truncated_tool_call = record
        for stream in (False, True):
            status, wire = _post(base, "/v1/chat/completions", {**body, "stream": stream})
            assert status == expected, (record, stream, wire)
        status, wire = _post(base, "/v1/messages", {
            "model": "fixture", "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
            "stop_sequences": ["\n"],
            "tools": [{"name": "weather", "input_schema": TOOLS[0]["function"]["parameters"]},
                      {"name": "db:lookup", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "tool", "name": "weather"},
        })
        assert status == expected, (record, "messages", wire)
    # Under required, the unreadable name still fails: a required call may
    # have been in progress, but the receipt cannot say to what.
    engine.stop_truncated_tool_call = UNREADABLE
    status, wire = _post(base, "/v1/chat/completions", {**body, "tool_choice": "required"})
    assert status == 502, wire


def test_collector_fails_an_unreadable_committed_name():
    body = {"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS,
            "tool_choice": "required"}
    job = Job(body)
    job.prompt_tokens, job.completion_tokens = 5, 2
    job.events.put({"text": ""})
    job.events.put({"finish_reason": "stop", "receipt": {
        "cache": "apcv2", "stop_sequence": "\n", "stop_truncated_tool_call": UNREADABLE,
    }})
    with pytest.raises(ToolContractError, match="could not be read"):
        collect_nonstream_job(job, body, chat=True)


def test_unbuffered_auto_chat_stream_judges_the_stop_cut_record(endpoint):  # noqa: F811
    # Codex round 1: the ordinary (auto, non-strict, unbuffered) Chat stream
    # ran the terminal contract only when the tool grammar had engaged, so a
    # stop-cut record naming an undeclared function, or none the grammar
    # could read, streamed a clean finish.  The contract now runs whenever
    # the terminal receipt carries the record.
    engine, base = endpoint
    engine.stop_sequence = "\n"
    body = {
        "model": "fixture",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "weather", "parameters": {}}}],
        "stop": ["\n"],
        "stream": True,
    }
    for record, failed in (
        (CUT, False),
        ({"function": "weather"}, False),
        ({"function": "db:lookup"}, True),
        (UNREADABLE, True),
    ):
        engine.stop_truncated_tool_call = record
        status, wire = _post(base, "/v1/chat/completions", body)
        assert status == 200, wire
        errors = [chunk["error"]["message"] for chunk in _chunks(wire) if "error" in chunk]
        finishes = [
            choice["finish_reason"]
            for chunk in _chunks(wire)
            for choice in chunk.get("choices", ())
            if choice.get("finish_reason")
        ]
        if failed:
            assert errors and finishes == [], (record, wire)
        else:
            assert not errors and finishes == ["stop"], (record, wire)


# Codex round 2 (integ-api): the readers over-reported names in shapes their
# parsers read as value text, so a valid named call cut there was a 502.
JSON_VALUE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "weather",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "where": {
                        "type": "object",
                        "properties": {"note": {"type": "string"}},
                        "required": ["note"],
                        "additionalProperties": False,
                    },
                },
                "required": ["city", "where"],
                "additionalProperties": False,
            },
        },
    },
    {"type": "function", "function": {"name": "news", "parameters": {}}},
]


def test_qwen_strict_json_value_quoting_a_closer_and_an_opener_is_value_text():
    # ``_walk_function`` ends a strict JSON value at the first closer outside
    # its strings; a reader ending it at the first closer read the quoted
    # ``<function=news>`` as a second committed name.
    quoted = (
        "<tool_call>\n<function=weather>\n<parameter=where>\n"
        '{"note": "</parameter>\\n<function=news>"}\n</parameter>\n'
        "<parameter=city>\nParis"
    )
    record = _cut(qwen_parse, quoted, JSON_VALUE_TOOLS)
    assert record == {"function": "weather"}
    enforce_tool_contract(
        {"tools": JSON_VALUE_TOOLS, "tool_choice": NAMED}, [], finish_reason="stop",
        stop_truncated_tool_call=record,
    )
    # The same text in the value the cut landed in, inside or outside its
    # string, is value text too.
    for cut_value in (
        '<tool_call>\n<function=weather>\n<parameter=where>\n{"note": "</parameter>\\n<function=news>Paris',
        '<tool_call>\n<function=weather>\n<parameter=where>\n{"note": "x"}\n<function=news>Paris',
    ):
        assert _cut(qwen_parse, cut_value, JSON_VALUE_TOOLS) == {"function": "weather"}
    # A real second function after the closed JSON value is still committed,
    # and a raw value still ends at its first closer.
    sibling = (
        "<tool_call>\n<function=weather>\n<parameter=where>\n"
        '{"note": "</parameter>"}\n</parameter>\n<function=news>\n<parameter=q>\nParis'
    )
    assert _cut(qwen_parse, sibling, JSON_VALUE_TOOLS) == {
        "function": "weather", "functions": ["weather", "news"],
    }
    raw = (
        "<tool_call>\n<function=weather>\n<parameter=city>\nRome\n</parameter>\n"
        "<function=news>\n<parameter=q>\nParis"
    )
    assert _cut(qwen_parse, raw, JSON_VALUE_TOOLS) == {
        "function": "weather", "functions": ["weather", "news"],
    }


MUSE_VALUE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "functions.weather",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}, "where": {"type": "object"}},
            },
        },
    },
    {"type": "function", "function": {"name": "functions.db_lookup", "parameters": {}}},
]


def test_muse_invoke_quoted_inside_a_parameter_value_is_value_text():
    # ``parse_atem`` reads an invoke tag inside a parameter value as the
    # value's own text; a reader counting every ``<atem:invoke name="`` read
    # it as a sibling invoke, and a named call cut there failed.
    def cut(text, stop="Paris"):
        parser = _drive(
            MuseOutputParser(chat=True, tools=MUSE_VALUE_TOOLS, stops=[stop]), text
        )
        assert parser.stopped and parser.tool_count == 0
        return parser.stop_truncated_tool_call

    head = "to=functions.weather<|message|><atem:function_calls>"
    quoted = '<atem:invoke name="functions.db_lookup">'
    for body in (
        # Raw value, closed; the cut lands in the next parameter.
        (
            f'<atem:invoke name="functions.weather"><atem:parameter name="q">see {quoted}'
            '</atem:parameter><atem:parameter name="where">{"c": "Paris'
        ),
        # Raw value the cut landed in.
        f'<atem:invoke name="functions.weather"><atem:parameter name="q">see {quoted} Paris',
        # JSON value whose string quotes a parameter closer and the invoke.
        (
            '<atem:invoke name="functions.weather"><atem:parameter name="where">'
            f'{{"c": "</atem:parameter>{quoted}"}}</atem:parameter>'
            '<atem:parameter name="q">Paris'
        ),
        # JSON value the cut landed in, inside its string.
        (
            '<atem:invoke name="functions.weather"><atem:parameter name="where">'
            f'{{"c": "{quoted} Paris'
        ),
    ):
        record = cut(head + body)
        assert record == {"function": "functions.weather"}, body
        enforce_tool_contract(
            {"tools": MUSE_VALUE_TOOLS, "tool_choice": _choice("functions.weather")}, [],
            finish_reason="stop", stop_truncated_tool_call=record,
        )
    # A sibling invoke after a closed invoke is still committed.
    sibling = (
        f'<atem:invoke name="functions.weather"><atem:parameter name="q">see {quoted}'
        f'</atem:parameter></atem:invoke>{quoted}<atem:parameter name="q">Paris'
    )
    assert cut(head + sibling) == {
        "function": "functions.weather",
        "functions": ["functions.weather", "functions.db_lookup"],
    }
    # A raw value cannot contain an invoke closer (``parse_atem`` rejects
    # the parameter as unclosed): the walk stops there, and an invoke after
    # it is still committed.
    unclosed = (
        f'<atem:invoke name="functions.weather"><atem:parameter name="q">see'
        f'</atem:invoke>{quoted}<atem:parameter name="q">Paris'
    )
    assert cut(head + unclosed) == {
        "function": "functions.weather",
        "functions": ["functions.weather", "functions.db_lookup"],
    }
