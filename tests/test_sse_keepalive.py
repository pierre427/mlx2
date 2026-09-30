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

    def submit(self, request, *, tenant_id="default"):
        job = self.job = Job(request)
        job.tenant_id = tenant_id
        if self.admitted:
            job.prompt_tokens, job.cached_tokens = 5, 0
        job.completion_tokens = 1

        def later():
            time.sleep(self.delay)
            for event in self.events:
                job.events.put(event)

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
