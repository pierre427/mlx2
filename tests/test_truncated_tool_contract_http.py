"""Transport coverage for unfinished tool calls and terminal causes."""

import json
import threading
from copy import deepcopy
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from test_serving_contract import TOOLS, FakeEngine
from test_structured_deferral import scripted_engine  # noqa: F401 - shared fixture

from mlx2.server import handler_for

WEATHER = TOOLS[0]
NEWS = {
    "type": "function",
    "function": {"name": "news", "parameters": {"type": "object"}},
}
TWO_TOOLS = [WEATHER, NEWS]
ANTHROPIC_TOOLS = [
    {"name": tool["function"]["name"], "input_schema": tool["function"]["parameters"]}
    for tool in TWO_TOOLS
]
RESPONSES_TOOLS = [
    {
        "type": "function",
        "name": tool["function"]["name"],
        "parameters": tool["function"]["parameters"],
    }
    for tool in TWO_TOOLS
]
NAMED_CHAT = {"type": "function", "function": {"name": "weather"}}


class TruncationEngine(FakeEngine):
    """Fake the terminal receipt while preserving the HTTP serving paths."""

    truncated_tool_call = {"function": None, "cause": "max_tokens"}
    completed_calls = None

    def submit(self, request, *, tenant_id="default"):
        job = super().submit(request, tenant_id=tenant_id)
        events = []
        while not job.events.empty():
            event = job.events.get_nowait()
            if "text" in event and self.completed_calls is not None:
                event = {"delta": {"tool_calls": deepcopy(self.completed_calls)}}
            if "finish_reason" in event:
                event["finish_reason"] = "length"
                if self.truncated_tool_call is not None:
                    event["receipt"]["truncated_tool_call"] = deepcopy(
                        self.truncated_tool_call
                    )
            events.append(event)
        for event in events:
            job.events.put(event)
        return job


@pytest.fixture
def endpoint():
    engine = TruncationEngine()
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
    except HTTPError as error:
        return error.code, error.read().decode()


def _choice(surface, choice):
    if surface == "messages":
        return {
            "required": {"type": "any"},
            "named": {"type": "tool", "name": "weather"},
            "auto": {"type": "auto"},
            "none": {"type": "none"},
        }[choice]
    if surface == "responses":
        return {
            "required": "required",
            "named": {"type": "function", "name": "weather"},
            "auto": "auto",
            "none": "none",
        }[choice]
    return {
        "required": "required",
        "named": NAMED_CHAT,
        "auto": "auto",
        "none": "none",
    }[choice]


def _request(surface, choice, stream):
    if surface == "messages":
        return "/v1/messages", {
            "model": "fixture",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "tools": ANTHROPIC_TOOLS,
            "tool_choice": _choice(surface, choice),
            "stream": stream,
        }
    if surface == "responses":
        return "/v1/responses", {
            "model": "fixture",
            "input": "hi",
            "max_output_tokens": 8,
            "tools": RESPONSES_TOOLS,
            "tool_choice": _choice(surface, choice),
            "stream": stream,
        }
    return "/v1/chat/completions", {
        "model": "fixture",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
        "tools": TWO_TOOLS,
        "tool_choice": _choice(surface, choice),
        "stream": stream,
    }


def _data_events(wire):
    events = []
    for line in wire.splitlines():
        if line.startswith("data: {"):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


def _contains_record(value, expected):
    if isinstance(value, dict):
        if value == expected:
            return True
        return any(_contains_record(item, expected) for item in value.values())
    if isinstance(value, list):
        return any(_contains_record(item, expected) for item in value)
    return False


def _assert_incomplete(surface, stream, wire, expected_record):
    events = _data_events(wire) if stream else [json.loads(wire)]
    if surface == "chat":
        finishes = [
            choice["finish_reason"]
            for event in events
            for choice in event.get("choices", ())
            if choice.get("finish_reason")
        ]
        assert finishes == ["length"]
        if not stream:
            message = events[0]["choices"][0]["message"]
            assert "tool_calls" not in message
    elif surface == "responses":
        response = events[-1]["response"] if stream else events[0]
        assert response["status"] == "incomplete"
        assert response["incomplete_details"] == {"reason": "max_output_tokens"}
        assert not [
            item for item in response["output"] if item["type"] == "function_call"
        ]
        if stream:
            assert "response.incomplete" in wire and "response.completed" not in wire
    else:
        if stream:
            deltas = [
                event["delta"]
                for event in events
                if event.get("type") == "message_delta"
            ]
            assert deltas[-1]["stop_reason"] == "max_tokens"
            assert not any(
                event.get("type") == "content_block_start"
                and event.get("content_block", {}).get("type") == "tool_use"
                for event in events
            )
        else:
            assert events[0]["stop_reason"] == "max_tokens"
            assert not [
                block for block in events[0]["content"] if block["type"] == "tool_use"
            ]
    assert _contains_record(events, expected_record)


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
@pytest.mark.parametrize("choice", ["required", "named", "auto"])
def test_max_tokens_cut_is_a_truthful_incomplete_response(
    endpoint, surface, stream, choice
):
    _engine, base = endpoint
    path, body = _request(surface, choice, stream)
    status, wire = _post(base, path, body)
    assert status == 200, wire
    _assert_incomplete(surface, stream, wire, {"function": None, "cause": "max_tokens"})


_BAD_CUTS = [
    ("required", {"function": "ghost", "cause": "max_tokens"}),
    ("required", {"function": None, "name_unreadable": True, "cause": "max_tokens"}),
    ("named", {"function": "news", "cause": "max_tokens"}),
    ("named", {"function": None, "name_unreadable": True, "cause": "max_tokens"}),
    ("auto", {"function": "ghost", "cause": "max_tokens"}),
    ("auto", {"function": None, "name_unreadable": True, "cause": "max_tokens"}),
    ("none", {"function": "weather", "cause": "max_tokens"}),
]


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
@pytest.mark.parametrize("choice, record", _BAD_CUTS)
def test_contract_rejects_bad_truncated_names_on_each_http_surface(
    endpoint, surface, stream, choice, record
):
    engine, base = endpoint
    engine.truncated_tool_call = record
    path, body = _request(surface, choice, stream)
    status, wire = _post(base, path, body)
    if not stream:
        assert status == 502, wire
    else:
        assert status == 502 or any(
            marker in wire
            for marker in ("event: error", "response.failed", '"type": "api_error"')
        ), wire
        assert "response.completed" not in wire
        assert '"stop_reason": "max_tokens"' not in wire


def test_completed_call_is_still_checked_when_terminal_finish_is_length(endpoint):
    engine, base = endpoint
    engine.truncated_tool_call = None
    engine.completed_calls = [
        {
            "index": 0,
            "id": "call_weather",
            "type": "function",
            "function": {
                "name": "weather",
                "arguments": json.dumps({"city": "Paris"}),
            },
        }
    ]
    path, body = _request("chat", "named", False)
    status, wire = _post(base, path, body)
    assert status == 200, wire
    payload = json.loads(wire)
    assert payload["choices"][0]["finish_reason"] == "length"
    assert (
        payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
        == "weather"
    )

    engine.completed_calls[0]["function"]["name"] = "news"
    status, wire = _post(base, path, body)
    assert status == 502, wire


def test_serving_emits_xing_max_tokens_receipt_through_chat_http(
    monkeypatch, scripted_engine
):
    import test_structured_deferral as fixture_module

    from mlx2.adapters.xing_output import XingOutputParser

    build, state = scripted_engine
    # The scripted fixture's selected token now decodes to an open Xing block.
    monkeypatch.setattr(fixture_module, "PIECES", list(fixture_module.PIECES))
    fixture_module.PIECES[fixture_module.TOOL_CALL] = (
        "<tool_call>sum<param_key>x</param_key><param_value>1"
    )
    engine = build(declare_marker=True)
    engine.adapter.output_parser = lambda request: XingOutputParser(
        chat=True,
        thinking=False,
        tools=request.get("tools"),
        stops=request.get("stop", ()),
        parallel_tool_calls=request.get("parallel_tool_calls", True),
    )
    state["script"] = [fixture_module.TOOL_CALL]
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = {
            "model": "fake",
            "messages": [{"role": "user", "content": "add"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "sum", "parameters": {"type": "object"}},
                }
            ],
            "tool_choice": "required",
            "max_tokens": 1,
            "temperature": 0,
        }
        status, wire = _post(
            f"http://127.0.0.1:{server.server_port}", "/v1/chat/completions", body
        )
        assert status == 200, wire
        payload = json.loads(wire)
        assert payload["choices"][0]["finish_reason"] == "length"
        assert "tool_calls" not in payload["choices"][0]["message"]
        assert payload["mlx2"]["truncated_tool_call"] == {
            "function": "sum",
            "cause": "max_tokens",
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_completed_call_then_client_cut_keeps_stop_sequence_cause(scripted_engine):
    import test_structured_deferral as fixture_module
    from test_structured_deferral import EOS, TOOL_CALL, _collect

    build, state = scripted_engine
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(fixture_module, "PIECES", list(fixture_module.PIECES))
    fixture_module.PIECES[fixture_module.MALFORMED_TOOL_CALL] = (
        "<tool_call>\n<function=sum>\n<parameter=x>\n2"
    )
    try:
        engine = build(declare_marker=True)
        state["script"] = [TOOL_CALL, fixture_module.MALFORMED_TOOL_CALL, EOS]
        request = {
            "messages": [{"role": "user", "content": "add"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "sum", "parameters": {"type": "object"}},
                }
            ],
            "tool_choice": "required",
            "max_tokens": 8,
            "stop": ["<parameter=x>\n2"],
            "temperature": 0,
        }
        job = engine.submit(request)
        completed_calls = []
        while True:
            event = job.events.get(timeout=10)
            completed_calls.extend((event.get("delta") or {}).get("tool_calls", ()))
            if "finish_reason" in event or "error" in event:
                break
        assert "finish_reason" in event, event
        assert event["finish_reason"] == "stop"
        assert len(completed_calls) == 1
        assert event["receipt"]["truncated_tool_call"] == {
            "function": "sum",
            "cause": "stop_sequence",
        }
    finally:
        monkeypatch.undo()
