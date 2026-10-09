"""A silent stream gets SSE keepalives (FreeToken #572, Splash #208).

Node's fetch (undici) drops a connection after 300 s without bytes; a long
prefill sends none before the first token, and a buffered tool stream none
before its terminal contract.
"""

import json
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from mlx2.server import handler_for
from mlx2.serving import Job
from test_serving_contract import FakeEngine

STRICT_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
            "additionalProperties": False,
        },
    },
}


class SilentEngine(FakeEngine):
    """Admits (or not), stays silent for ``delay`` seconds, then answers."""

    delay = 0.9
    admitted = True
    events = ()
    spacing = 0.0
    tool_grammar_status = None

    def submit(self, request, *, tenant_id="default"):
        job = self.job = Job(request)
        job.tenant_id = tenant_id
        if self.admitted:
            # Counted, then attached to a lane: the keepalive's admission test.
            job.prompt_tokens, job.cached_tokens = 5, 0
            job.uid = 0
        job.tool_grammar_status = self.tool_grammar_status
        job.completion_tokens = 1

        def later():
            time.sleep(self.delay)
            for event in self.events:
                job.events.put(event)
                time.sleep(self.spacing)

        threading.Thread(target=later, daemon=True).start()
        return job


@pytest.fixture
def served():
    engine = SilentEngine()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), handler_for(engine, sse_keepalive_seconds=0.2)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield engine, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def post(base, path="/v1/chat/completions", **body):
    body.setdefault("model", "fixture")
    body.setdefault("messages", [{"role": "user", "content": "hi"}])
    return urlopen(
        Request(
            base + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        ),
        timeout=10,
    )


FINISH = {"finish_reason": "stop", "receipt": {"cache": "apcv2"}}


def test_silent_prefill_gets_keepalives_before_the_first_token(served):
    engine, base = served
    engine.events = ({"delta": {"content": "hello"}}, FINISH)
    with post(base, stream=True) as response:
        assert response.status == 200
        raw = response.read().decode()
    first_data = raw.index("data: ")
    assert raw[:first_data].count(": keep-alive\n\n") >= 2
    chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"]) == "hello"
    assert raw.rstrip().endswith("data: [DONE]")


def test_an_unadmitted_request_commits_nothing_so_errors_keep_their_status(served):
    engine, base = served
    engine.admitted = False
    engine.events = ({"error": "bad request", "status": 400},)
    with pytest.raises(HTTPError) as caught:
        post(base, stream=True)
    assert caught.value.code == 400


def test_buffered_tool_stream_keeps_alive_and_still_withholds_an_invalid_call(served):
    engine, base = served
    bad_call = {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"}}]}
    engine.events = ({"delta": bad_call}, {**FINISH, "finish_reason": "tool_calls"})
    with post(base, stream=True, tools=[STRICT_TOOL]) as response:
        assert response.status == 200
        raw = response.read().decode()
    assert raw.count(": keep-alive\n\n") >= 2
    # The call that fails its strict schema is never sent; the failure is an
    # SSE error event now that a keepalive has opened the stream.
    assert "tool_calls" not in raw
    errors = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
    assert errors and "error" in errors[-1]


def test_non_stream_requests_are_unchanged(served):
    engine, base = served
    engine.events = ({"delta": {"content": "hello"}}, FINISH)
    with post(base) as response:
        payload = json.load(response)
    assert payload["choices"][0]["message"]["content"] == "hello"


def test_keepalive_can_be_disabled():
    engine = SilentEngine()
    engine.events = ({"delta": {"content": "hello"}}, FINISH)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), handler_for(engine, sse_keepalive_seconds=None)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with post(f"http://127.0.0.1:{server.server_port}", stream=True) as response:
            raw = response.read().decode()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert ": keep-alive" not in raw


def test_buffered_tool_stream_sends_a_valid_call_after_keepalives(served):
    engine, base = served
    call = {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                            "function": {"name": "lookup", "arguments": '{"q": "x"}'}}]}
    engine.events = ({"delta": call}, {**FINISH, "finish_reason": "tool_calls"})
    with post(base, stream=True, tools=[STRICT_TOOL]) as response:
        raw = response.read().decode()
    assert raw.index(": keep-alive") < raw.index("data: {")
    chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
    sent = [c for chunk in chunks for choice in chunk["choices"]
            for c in choice["delta"].get("tool_calls", ())]
    assert [c["function"]["arguments"] for c in sent] == ['{"q": "x"}']


def test_anthropic_stream_opens_with_the_admitted_prompt_count(served):
    engine, base = served
    engine.events = ({"delta": {"content": "hello"}}, FINISH)
    with post(base, "/v1/messages", max_tokens=16, stream=True) as response:
        raw = response.read().decode()
    start = raw.index("event: message_start")
    assert raw.index(": keep-alive") > start  # the prologue first, then keepalives
    message_start = json.loads(raw[start:].split("data: ", 1)[1].split("\n", 1)[0])
    assert message_start["message"]["usage"]["input_tokens"] == 5
    assert "event: message_stop" in raw


def test_counted_but_unattached_request_commits_nothing(served):
    """A prompt counted but still waiting for memory or a LoRA slot is not
    admitted: an admission error must keep its HTTP status."""
    engine, base = served
    original = SilentEngine.submit

    def counted_only(self, request, *, tenant_id="default"):
        job = original(self, request, tenant_id=tenant_id)
        job.uid = None
        return job

    engine.submit = counted_only.__get__(engine)
    engine.events = ({"error": "no memory", "status": 503},)
    with pytest.raises(HTTPError) as caught:
        post(base, stream=True)
    assert caught.value.code == 503


def test_buffered_stream_keeps_alive_while_it_consumes_deltas_silently(served):
    """Keepalives follow bytes written, not engine events: a buffered tool
    stream receiving a delta every 0.1 s still writes nothing for seconds."""
    engine, base = served
    engine.delay, engine.spacing = 0.0, 0.1
    pieces = ['{"q": "', *"abcdefghij", '"}']
    engine.events = tuple(
        {"delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                   "function": {"name": "lookup", "arguments": piece}}]}}
        for piece in pieces
    ) + ({**FINISH, "finish_reason": "tool_calls"},)
    with post(base, stream=True, tools=[STRICT_TOOL]) as response:
        raw = response.read().decode()
    assert raw.count(": keep-alive\n\n") >= 2


def test_grammar_tool_streaming_still_engages_after_an_early_keepalive(served):
    engine, base = served
    engine.tool_grammar_status = "engaged"
    engine.status = lambda: {**FakeEngine.status(engine), "settings": {"tool_grammar_streaming": True}}
    call = {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                            "function": {"name": "lookup", "arguments": '{"q": "x"}'}}]}
    engine.events = ({"delta": call}, {**FINISH, "finish_reason": "tool_calls"})
    with post(base, stream=True, tools=[STRICT_TOOL]) as response:
        raw = response.read().decode()
    assert raw.index(": keep-alive") < raw.index("data: {")
    assert engine.counts["constrained_tool_grammar_streams"] == 1


def test_hosted_stream_keeps_alive_through_a_slow_tool_with_one_response_id():
    from test_serving_contract import TOOLS, post_response

    class Backend:
        def prepare(self, tools):
            return TOOLS, {"weather": "binding"}

        def execute(self, binding, arguments):
            time.sleep(0.9)  # a slow hosted tool writes nothing meanwhile
            return {"temperature": 21}

    class ToolEngine(SilentEngine):
        rounds = 0

        def submit(self, request, *, tenant_id="default"):
            self.rounds += 1
            if self.rounds == 1:
                self.events = (
                    {"delta": {"tool_calls": [{
                        "index": 0, "id": "call_weather", "type": "function",
                        "function": {"name": "weather", "arguments": '{"city":"T"}'},
                    }]}},
                    {"finish_reason": "tool_calls", "receipt": {}},
                )
            else:
                self.delay = 0.0
                self.events = ({"delta": {"content": "21 C."}}, {"finish_reason": "stop", "receipt": {}})
            return super().submit(request, tenant_id=tenant_id)

    engine = ToolEngine()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        handler_for(engine, tool_backend=Backend(), sse_keepalive_seconds=0.2),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with post_response(
            f"http://127.0.0.1:{server.server_port}",
            input="weather?",
            stream=True,
            tools=[{
                "type": "mcp", "server_label": "weather",
                "server_url": "https://example.invalid/mcp", "require_approval": "never",
            }],
        ) as response:
            raw = response.read().decode()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    # Keepalives before the first event and during the tool call.
    assert raw.count(": keep-alive\n\n") >= 6
    events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
    created = [e for e in events if e["type"] == "response.created"]
    completed = [e for e in events if e["type"] == "response.completed"]
    assert len(created) == 1 and len(completed) == 1
    assert created[0]["response"]["id"] == completed[0]["response"]["id"]


def test_a_queued_stream_waits_between_keepalive_attempts(monkeypatch):
    """A keepalive declines to write for a request not yet on a lane.  The
    wait then recomputed a deadline in the past and polled the socket with a
    zero timeout: one busy core per queued stream (~280k polls in 2 s)."""
    import mlx2.server as server_mod

    polls = {"n": 0}
    original = server_mod.client_disconnected

    def counting(connection):
        polls["n"] += 1
        return original(connection)

    monkeypatch.setattr(server_mod, "client_disconnected", counting)
    engine = SilentEngine()
    engine.admitted, engine.delay = False, 2.0
    engine.events = ({"delta": {"content": "hello"}}, FINISH)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), server_mod.handler_for(engine, sse_keepalive_seconds=0.2)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with post(f"http://127.0.0.1:{server.server_port}", stream=True) as response:
            raw = response.read().decode()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert "hello" in raw
    assert polls["n"] < 60, polls


@pytest.mark.parametrize("variant", ["tool_error", "engine_error"])
def test_a_failing_hosted_stream_still_sends_response_created_first(variant):
    """A keepalive opens a hosted-tool Responses stream without its prologue;
    a later failure wrote a bare response.failed with no response.created."""
    from mlx2.tool_backend import HostedToolError
    from test_serving_contract import TOOLS, post_response

    class Backend:
        def prepare(self, tools):
            return TOOLS, {"weather": "binding"}

        def execute(self, binding, arguments):
            time.sleep(0.6)
            raise HostedToolError("MCP server unreachable")

    class ToolEngine(SilentEngine):
        def submit(self, request, *, tenant_id="default"):
            self.events = (
                ({"error": "lane fault", "status": 503},)
                if variant == "engine_error"
                else (
                    {"delta": {"tool_calls": [{
                        "index": 0, "id": "call_weather", "type": "function",
                        "function": {"name": "weather", "arguments": '{"city":"T"}'},
                    }]}},
                    {"finish_reason": "tool_calls", "receipt": {}},
                )
            )
            return super().submit(request, tenant_id=tenant_id)

    engine = ToolEngine()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        handler_for(engine, tool_backend=Backend(), sse_keepalive_seconds=0.2),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with post_response(
            f"http://127.0.0.1:{server.server_port}", input="weather?", stream=True,
            tools=[{"type": "mcp", "server_label": "weather",
                    "server_url": "https://example.invalid/mcp", "require_approval": "never"}],
        ) as response:
            raw = response.read().decode()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert raw.count(": keep-alive") >= 1
    types = [json.loads(line[6:])["type"] for line in raw.splitlines() if line.startswith("data: {")]
    assert types[0] == "response.created" and types[-1] == "response.failed", types


@pytest.mark.parametrize("variant", ["overloaded", "admission_closed", "engine_down"])
def test_a_continuation_failure_on_an_open_hosted_stream_is_an_sse_failure(variant):
    """A slow hosted tool's keepalives commit ``200 text/event-stream``; the
    continuation round is then submitted again.  When that submit raised
    (slots taken while the tool ran, a drain timeout, a dead engine) the
    handler wrote a raw ``HTTP/1.1 429``/``503`` response into the event
    stream, or ended it with no terminal event and no HTTP metric."""
    from test_serving_contract import TOOLS, post_response

    from mlx2.batch_metrics import HttpRuntimeMetrics
    from mlx2.serving import AdmissionClosed, Overloaded

    failures = {
        "overloaded": lambda: Overloaded("maximum inflight requests reached"),
        "admission_closed": lambda: AdmissionClosed("draining", "generation"),
        "engine_down": lambda: RuntimeError("model is not ready"),
    }

    class Backend:
        def prepare(self, tools):
            return TOOLS, {"weather": "binding"}

        def execute(self, binding, arguments):
            time.sleep(0.6)  # long enough for keepalives to open the stream
            return {"temperature": 21}

    class ToolEngine(SilentEngine):
        rounds = 0

        def submit(self, request, *, tenant_id="default", **_):
            self.rounds += 1
            if self.rounds > 1:
                raise failures[variant]()
            self.events = (
                {"delta": {"tool_calls": [{
                    "index": 0, "id": "call_weather", "type": "function",
                    "function": {"name": "weather", "arguments": '{"city":"T"}'},
                }]}},
                {"finish_reason": "tool_calls", "receipt": {}},
            )
            return super().submit(request, tenant_id=tenant_id)

    engine = ToolEngine()
    engine.http_metrics = HttpRuntimeMetrics()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        handler_for(engine, tool_backend=Backend(), sse_keepalive_seconds=0.2),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with post_response(
            f"http://127.0.0.1:{server.server_port}", input="weather?", stream=True,
            tools=[{"type": "mcp", "server_label": "weather",
                    "server_url": "https://example.invalid/mcp", "require_approval": "never"}],
        ) as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "text/event-stream"
            raw = response.read().decode()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert engine.rounds == 2
    assert raw.count(": keep-alive") >= 1
    # No second HTTP response is written into the committed event stream.
    assert "HTTP/1.1" not in raw, raw[-300:]
    types = [json.loads(line[6:])["type"] for line in raw.splitlines() if line.startswith("data: {")]
    assert types and types[0] == "response.created" and types[-1] == "response.failed", types
    # Counted once, with the status the client actually saw.
    assert engine.http_metrics.prometheus_snapshot()["requests"] == {
        ("POST", "responses", "2xx"): 1,
    }
