"""A client ``stop`` string that cuts a required tool call is a stop, not a 502.

``OutputParser`` drops a tool call that a client stop string lands inside as a
requested stop (its docstring and ``test_client_stop_inside_tool_call_is_a_stop
_not_a_server_error`` say so), the engine reports ``finish_reason: "stop"`` with
``receipt.stop_sequence``, but ``enforce_tool_contract`` exempted only a
``length`` finish from the ``required`` / named-function contract.  The same
call cut by ``max_tokens`` was a 200 ``finish_reason: "length"`` while the one
cut by the client's own ``stop`` was a 502 "model did not emit a required tool
call" on Chat (non-stream and buffered stream) and on ``/v1/messages``.  The
model did what the request asked; the client ended the turn.  The parser now
records the dropped call (``stop_truncated_tool_call: {"function": name}``,
the function the partial call had named or ``None``), the receipt carries it,
and the contract exempts exactly that under a ``stop`` finish.  A matched stop
on its own excuses nothing: a turn the model spent on prose, or ended on its
own, without the call it owed is still the model's violation and stays a
502; so is a cut call that had already named an undeclared function or one
other than the named choice (review round 3).
"""

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from test_serving_contract import TOOLS, FakeEngine

from mlx2.adapters.muse_glimmer_output import MuseOutputParser
from mlx2.openai_compat import ToolContractError, enforce_tool_contract
from mlx2.output import OutputParser, stop_cut_record
from mlx2.runtime.tool_parsers.qwen3_coder import parse_tool_call
from mlx2.server import collect_nonstream_job, handler_for
from mlx2.serving import Job

NAMED = {"type": "function", "function": {"name": "weather"}}
ANTHROPIC_TOOLS = [
    {"name": "weather", "input_schema": TOOLS[0]["function"]["parameters"]}
]
RESPONSES_TOOLS = [
    {"type": "function", "name": "weather", "parameters": TOOLS[0]["function"]["parameters"]}
]
CUT = {"function": None}  # the stop landed before the call named its function


class StopCutEngine(FakeEngine):
    """Emits what serving does after the parser dropped a stop-cut call."""

    stop_truncated_tool_call = None

    def submit(self, request, *, tenant_id="default"):
        job = super().submit(request, tenant_id=tenant_id)
        # Replace the terminal with one whose receipt carries the record.
        events = []
        while not job.events.empty():
            events.append(job.events.get_nowait())
        for event in events:
            if "finish_reason" in event and self.stop_truncated_tool_call:
                event["receipt"]["stop_truncated_tool_call"] = (
                    self.stop_truncated_tool_call
                )
            job.events.put(event)
        return job


@pytest.fixture
def endpoint():
    engine = StopCutEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield engine, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def _post(base, path, body):
    request = Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request) as response:
            return response.status, response.read().decode()
    except HTTPError as exc:
        return exc.code, exc.read().decode()


def _chunks(wire):
    return [
        json.loads(line[len("data: "):])
        for line in wire.splitlines()
        if line.startswith("data: {")
    ]


@pytest.mark.parametrize("choice", ["required", NAMED])
def test_contract_exempts_a_stop_cut_call_like_a_length_cut(choice):
    body = {"tools": TOOLS, "tool_choice": choice}
    # The control that already passed: max_tokens cut the call.
    enforce_tool_contract(body, [], finish_reason="length")
    # The client's stop string cut the call: the same requested truncation,
    # whether the cut came before the name or after the right one.
    enforce_tool_contract(body, [], finish_reason="stop", stop_truncated_tool_call=CUT)
    enforce_tool_contract(
        body, [], finish_reason="stop", stop_truncated_tool_call={"function": "weather"}
    )
    # No cut call: the model ended its turn without the call it owed, with or
    # without a stop string having matched in its prose.
    with pytest.raises(ToolContractError):
        enforce_tool_contract(body, [], finish_reason="stop")
    with pytest.raises(ToolContractError):
        enforce_tool_contract(
            body, [], finish_reason="stop", stop_truncated_tool_call=None
        )
    # The record is evidence only under a stop finish.
    with pytest.raises(ToolContractError):
        enforce_tool_contract(
            body, [], finish_reason="tool_calls", stop_truncated_tool_call=CUT
        )
    # A cut call that had already named an undeclared function is the
    # model's violation, as a completed call to it would be.
    with pytest.raises(ToolContractError, match="undeclared"):
        enforce_tool_contract(
            body, [], finish_reason="stop", stop_truncated_tool_call={"function": "news"}
        )


def test_contract_holds_a_cut_call_to_the_named_choice():
    two = TOOLS + [{"type": "function", "function": {"name": "news", "parameters": {}}}]
    body = {"tools": two, "tool_choice": NAMED}
    enforce_tool_contract(
        body, [], finish_reason="stop", stop_truncated_tool_call={"function": "weather"}
    )
    enforce_tool_contract(body, [], finish_reason="stop", stop_truncated_tool_call=CUT)
    # The partial call named a declared function other than the one selected.
    with pytest.raises(ToolContractError, match="exclusively"):
        enforce_tool_contract(
            body, [], finish_reason="stop", stop_truncated_tool_call={"function": "news"}
        )
    # Under ``auto`` the cut call is simply dropped, but an undeclared name
    # still fails, as a completed undeclared call does.
    enforce_tool_contract(
        {"tools": two}, [], finish_reason="stop", stop_truncated_tool_call={"function": "news"}
    )
    with pytest.raises(ToolContractError, match="undeclared"):
        enforce_tool_contract(
            {"tools": two}, [], finish_reason="stop", stop_truncated_tool_call={"function": "x"}
        )


def test_partial_function_name_recovers_markup_and_json_forms():
    # Each grammar reads the committed name by its own parser rules
    # (integ-api group); the record is built from what it reads.
    qwen = parse_tool_call.partial_function_names
    assert stop_cut_record(qwen, "\n<function=weather>\n<parameter=city>", TOOLS) == {
        "function": "weather"
    }
    assert stop_cut_record(qwen, "\n<function=weat", TOOLS) == CUT
    assert stop_cut_record(qwen, "", TOOLS) == CUT
    from mlx2.adapters.xing_output import parse_tool_block as xing

    assert stop_cut_record(
        xing.partial_function_names, '{"name": "weather", "arguments": {"ci', TOOLS
    ) == {"function": "weather"}


CALL = "<tool_call>\n<function=weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"


def _drive(parser, text, finish=None, chunk=4):
    for start in range(0, len(text), chunk):
        piece = text[start:start + chunk]
        if finish and start + chunk >= len(text):
            parser.finish(piece, finish)
        else:
            parser.push(piece)
        if parser.stopped:
            break
    return parser


def test_output_parser_records_only_a_call_the_stop_cut():
    def parser(stops):
        return OutputParser(
            chat=True, tools=TOOLS, parse_tool=parse_tool_call, stops=stops,
            constrained_tools=True,
        )

    # The stop lands right after the opener: recorded, no name yet.
    cut = _drive(parser(["\n"]), CALL)
    assert cut.stopped and cut.tool_count == 0
    assert cut.stop_truncated_tool_call == CUT
    # The stop lands inside the buffered call body: the name is recorded.
    named = _drive(parser(["Paris"]), CALL)
    assert named.stopped and named.tool_count == 0
    assert named.stop_truncated_tool_call == {"function": "weather"}
    # ...whatever it names; the contract judges it.
    other = _drive(parser(["Paris"]), CALL.replace("weather", "news"))
    assert other.stop_truncated_tool_call == {"function": "news"}
    # The stop lands in prose before any call: nothing was cut.
    prose = _drive(parser(["Paris"]), "The city is Paris.")
    assert prose.stopped and prose.stop_truncated_tool_call is None
    # max_tokens inside the call: a length cut, not a stop cut.
    length = _drive(parser(()), CALL[:40], finish="length")
    assert not length.stopped and length.stop_truncated_tool_call is None
    # No stop matched and the call completed: nothing recorded.
    whole = _drive(parser(["\n\n"]), CALL)
    assert whole.tool_count == 1 and whole.stop_truncated_tool_call is None


ATEM = (
    "to=functions.weather<|message|><atem:function_calls>"
    '<atem:invoke name="functions.weather"><atem:parameter name="city">Paris'
    "</atem:parameter></atem:invoke></atem:function_calls><|eot|>"
)
MUSE_TOOLS = [
    {"type": "function", "function": {"name": "functions.weather", "parameters": {}}}
]


def test_muse_parser_records_only_a_call_the_stop_cut():
    # The same drop path as OutputParser, the same record.
    cut = _drive(MuseOutputParser(chat=True, tools=MUSE_TOOLS, stops=["Paris"]), ATEM)
    assert cut.stopped and cut.tool_count == 0
    assert cut.stop_truncated_tool_call == {"function": "functions.weather"}
    # Right after the opener, nothing buffered: recorded without a name.
    opener = _drive(MuseOutputParser(chat=True, tools=MUSE_TOOLS, stops=["<atem:invoke"]), ATEM)
    assert opener.stopped and opener.tool_count == 0
    assert opener.stop_truncated_tool_call == CUT
    prose = _drive(MuseOutputParser(chat=True, tools=MUSE_TOOLS, stops=["Paris"]), "Paris.")
    assert prose.stopped and prose.stop_truncated_tool_call is None
    length = _drive(MuseOutputParser(chat=True, tools=MUSE_TOOLS), ATEM[:80], finish="length")
    assert not length.stopped and length.stop_truncated_tool_call is None
    whole = _drive(MuseOutputParser(chat=True, tools=MUSE_TOOLS, stops=["zzz"]), ATEM)
    assert whole.tool_count == 1 and whole.stop_truncated_tool_call is None


@pytest.mark.parametrize("choice", ["required", NAMED])
@pytest.mark.parametrize("stream", [False, True])
def test_chat_reports_a_stop_cut_required_call_as_stop(endpoint, choice, stream):
    engine, base = endpoint
    engine.stop_sequence = "\n"
    engine.stop_truncated_tool_call = CUT
    body = {
        "model": "fixture",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": TOOLS,
        "tool_choice": choice,
        "stop": ["\n"],
        "stream": stream,
    }
    status, wire = _post(base, "/v1/chat/completions", body)
    assert status == 200, wire
    if stream:
        finishes = [
            choice["finish_reason"]
            for chunk in _chunks(wire)
            for choice in chunk.get("choices", ())
            if choice.get("finish_reason")
        ]
        assert finishes == ["stop"]
    else:
        payload = json.loads(wire)
        assert payload["choices"][0]["finish_reason"] == "stop"
        assert "tool_calls" not in payload["choices"][0]["message"]
        assert payload["mlx2"]["stop_sequence"] == "\n"

    # The cut call had named an undeclared function: the model's violation.
    engine.stop_truncated_tool_call = {"function": "news"}
    status, wire = _post(base, "/v1/chat/completions", body)
    assert status == 502, wire
    # The stop matched in prose (no call was cut): the model failed to call.
    engine.stop_truncated_tool_call = None
    status, wire = _post(base, "/v1/chat/completions", body)
    assert status == 502, wire
    # No stop matched at all: still the model's violation.
    engine.stop_sequence = None
    status, wire = _post(base, "/v1/chat/completions", body)
    assert status == 502, wire


@pytest.mark.parametrize("stream", [False, True])
def test_messages_reports_a_stop_cut_required_call_as_stop_sequence(endpoint, stream):
    engine, base = endpoint
    engine.stop_sequence = "\n"
    engine.stop_truncated_tool_call = CUT
    body = {
        "model": "fixture",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
        "stop_sequences": ["\n"],
        "tools": ANTHROPIC_TOOLS,
        "tool_choice": {"type": "any"},
        "stream": stream,
    }
    status, wire = _post(base, "/v1/messages", body)
    assert status == 200, wire
    if stream:
        deltas = [
            json.loads(line[len("data: "):])
            for line in wire.splitlines()
            if line.startswith("data: {")
        ]
        [stop] = [
            event["delta"] for event in deltas if event["type"] == "message_delta"
        ]
        assert stop == {"stop_reason": "stop_sequence", "stop_sequence": "\n"}
        assert "event: error" not in wire
    else:
        payload = json.loads(wire)
        assert payload["stop_reason"] == "stop_sequence"
        assert payload["stop_sequence"] == "\n"

    for engine.stop_truncated_tool_call, engine.stop_sequence in (
        ({"function": "news"}, "\n"),  # the cut call named an undeclared function
        (None, "\n"),  # stop matched in prose: no call was cut
        (None, None),  # no stop at all
    ):
        status, wire = _post(base, "/v1/messages", body)
        assert (status == 502 if not stream else "event: error" in wire), wire


@pytest.mark.parametrize("stream", [False, True])
def test_responses_reports_a_stop_cut_required_call(endpoint, stream):
    engine, base = endpoint
    engine.stop_sequence = "\n"
    engine.stop_truncated_tool_call = CUT
    body = {
        "model": "fixture",
        "input": "hi",
        "tools": RESPONSES_TOOLS,
        "tool_choice": "required",
        "stream": stream,
    }
    status, wire = _post(base, "/v1/responses", body)
    assert status == 200, wire
    if stream:
        assert "response.completed" in wire and "response.failed" not in wire
        assert '"type": "error"' not in wire
    else:
        payload = json.loads(wire)
        assert payload["status"] == "completed"
        assert not [item for item in payload["output"] if item["type"] == "function_call"]
    for engine.stop_truncated_tool_call in ({"function": "news"}, None):
        status, wire = _post(base, "/v1/responses", body)
        if stream:
            assert status != 200 or "response.completed" not in wire, wire
        else:
            assert status == 502, wire


def test_collector_reads_the_record_from_the_terminal_receipt():
    body = {"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS,
            "tool_choice": "required"}

    def job_with(record):
        job = Job(body)
        job.prompt_tokens, job.completion_tokens = 5, 2
        job.events.put({"text": ""})
        receipt = {"cache": "apcv2", "stop_sequence": "\n"}
        if record is not None:
            receipt["stop_truncated_tool_call"] = record
        job.events.put({"finish_reason": "stop", "receipt": receipt})
        return job

    choice, _usage, receipt = collect_nonstream_job(job_with(CUT), body, chat=True)
    assert choice["finish_reason"] == "stop" and "tool_calls" not in choice["message"]
    assert receipt["stop_truncated_tool_call"] == CUT
    for record in ({"function": "news"}, None):
        with pytest.raises(ToolContractError):
            collect_nonstream_job(job_with(record), body, chat=True)


# Review round 2: the HTTP cases above manufacture the receipt record in
# ``StopCutEngine``; this drives a real ``ServingEngine`` loop (scripted
# model, real OutputParser, real receipt) so the parser-to-receipt bridge in
# serving.py is itself under test, end to end through the HTTP contract.
from test_structured_deferral import (  # noqa: E402,F401 - shared fixture
    EOS,
    HELLO,
    TOOL_CALL,
    _collect,
    scripted_engine,
)

SUM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "sum",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
        },
    }
]


def test_serving_receipt_carries_the_parsers_stop_cut_record(scripted_engine):
    build, state = scripted_engine
    engine = build(declare_marker=True)
    request = {
        "messages": [{"role": "user", "content": "add"}],
        "tools": SUM_TOOLS,
        "tool_choice": "required",
        "max_tokens": 8,
        "temperature": 0,
    }
    # The model starts the call; the client's stop lands inside it.
    state["script"] = [TOOL_CALL, EOS]
    _, content, cut = _collect(engine.submit({**request, "stop": ["\n"]}))
    assert "error" not in cut, cut
    assert cut["finish_reason"] == "stop" and content == ""
    assert cut["receipt"]["stop_sequence"] == "\n"
    assert cut["receipt"]["stop_truncated_tool_call"] == CUT
    # Cut after the name: the receipt carries which call was in progress.
    _, _, named = _collect(engine.submit({**request, "stop": ["<parameter"]}))
    assert named["finish_reason"] == "stop"
    assert named["receipt"]["stop_truncated_tool_call"] == {"function": "sum"}
    # The same call, uncut: no record.
    _, _, whole = _collect(engine.submit({**request, "stop": ["zzz"]}))
    assert whole["finish_reason"] == "tool_calls"
    assert "stop_truncated_tool_call" not in whole["receipt"]
    # Prose that the stop matched: no call was cut, no record.
    state["script"] = [HELLO, HELLO, EOS]
    _, _, prose = _collect(engine.submit({**request, "stop": ["hello"]}))
    assert prose["finish_reason"] == "stop"
    assert prose["receipt"]["stop_sequence"] == "hello"
    assert "stop_truncated_tool_call" not in prose["receipt"]

    # And through the HTTP contract on the same engine.
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        state["script"] = [TOOL_CALL, EOS]
        status, wire = _post(
            base, "/v1/chat/completions", {**request, "model": "fake", "stop": ["\n"]}
        )
        assert status == 200, wire
        payload = json.loads(wire)
        assert payload["choices"][0]["finish_reason"] == "stop"
        assert payload["mlx2"]["stop_truncated_tool_call"] == CUT
        # A named choice: the cut call named it (200) or another tool (502).
        two = SUM_TOOLS + [{"type": "function", "function": {"name": "other", "parameters": {}}}]
        for name, expected in (("sum", 200), ("other", 502)):
            state["script"] = [TOOL_CALL, EOS]
            status, wire = _post(base, "/v1/chat/completions", {
                **request, "model": "fake", "tools": two, "stop": ["<parameter"],
                "tool_choice": {"type": "function", "function": {"name": name}},
            })
            assert status == expected, (name, wire)
        state["script"] = [HELLO, HELLO, EOS]
        status, wire = _post(
            base, "/v1/chat/completions", {**request, "model": "fake", "stop": ["hello"]}
        )
        assert status == 502, wire
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


# Review round 4: ``OutputParser`` also serves the Xing/Mellum plain-name
# grammar (``name<param_key>`` / ``name{``) and Laguna's (``name<arg_key>``);
# the name recovery covered only Qwen markup and JSON, so a cut call that had
# named another function under those grammars was recorded nameless and
# excused.  The name is bound once its delimiter is seen; mid-name cuts stay
# nameless.
from mlx2.adapters.xing_output import parse_tool_block  # noqa: E402
from mlx2.runtime.tool_parsers.laguna import parse_tool_call as laguna_parse  # noqa: E402

TWO_TOOLS = TOOLS + [{"type": "function", "function": {"name": "news", "parameters": {}}}]


@pytest.mark.parametrize(
    "parse_tool, call",
    [
        (parse_tool_block, "<tool_call>news<param_key>city</param_key><param_value>Paris</param_value></tool_call>"),
        (parse_tool_block, '<tool_call>{"name": "news", "arguments": {"city": "Paris"}}</tool_call>'),
        (laguna_parse, "<tool_call>news<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"),
        (parse_tool_call, "<tool_call>\n<function=news>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"),
    ],
    ids=["xing-plain", "xing-json", "laguna", "qwen"],
)
def test_every_grammar_binds_the_cut_calls_name(parse_tool, call):
    def parser(stops):
        return OutputParser(
            chat=True, tools=TWO_TOOLS, parse_tool=parse_tool, stops=stops,
            constrained_tools=True,
        )

    named = _drive(parser(["Paris"]), call)
    assert named.stopped and named.tool_count == 0
    assert named.stop_truncated_tool_call == {"function": "news"}
    # Held to the request: the cut call was not the one the choice selected.
    with pytest.raises(ToolContractError, match="exclusively"):
        enforce_tool_contract(
            {"tools": TWO_TOOLS, "tool_choice": NAMED}, [], finish_reason="stop",
            stop_truncated_tool_call=named.stop_truncated_tool_call,
        )
    # Cut inside the name: nothing bound (the name is not committed yet).
    mid = _drive(parser(["ew"]), call)
    assert mid.stopped and mid.stop_truncated_tool_call == CUT


def test_partial_name_needs_its_delimiter_and_ignores_nested_names():
    xing = parse_tool_block.partial_function_names
    laguna = laguna_parse.partial_function_names
    assert xing("news<param_key>city") == ["news"]
    assert laguna("news<arg_key>city") == ["news"]
    assert xing('news{"city": "Pa') == ["news"]
    assert xing("news") == []  # delimiter not seen yet
    assert laguna("news") == []
    assert xing("ne") == []
    # Only the top-level JSON "name" binds; an argument's nested "name" does
    # not, and the member is read wherever it appears.
    assert xing('{"arguments": {"name": "x"}, "name": "we') == []
    assert xing('{"arguments": {"name": "x"}, "name": "weather", "a') == ["weather"]
