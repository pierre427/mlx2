"""A Responses hosted (MCP) tool loop keeps the whole response within
``max_output_tokens``.

The loop resubmitted the request unchanged apart from its messages, so every
continuation round got the full cap again and the summed usage could reach
several times ``max_output_tokens``.  OpenAI defines the field as the bound for
the whole response.
"""

import json
import threading
from collections import Counter
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

import pytest

from mlx2.server import MIN_REQUEST_BODY_BYTES, handler_for
from mlx2.serving import Job

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

MCP_TOOL = {
    "type": "mcp",
    "server_label": "weather",
    "server_url": "https://example.invalid/mcp",
    "require_approval": "never",
}


class Backend:
    def __init__(self):
        self.executed = 0

    def prepare(self, tools):
        return TOOLS, {"weather": "binding"}

    def execute(self, binding, arguments):
        self.executed += 1
        return {"temperature": 21}


class BudgetEngine:
    """Each round decodes up to 4 tokens; rounds 1 and 2 call the hosted tool."""

    model_path = "fixture"
    max_context = 16384
    per_round = 4

    def __init__(self):
        self.lock = threading.Lock()
        self.counts = Counter()
        self.max_request_bytes = MIN_REQUEST_BODY_BYTES
        self.requests = []

    def status(self):
        return {"healthy": True, "error": None, "model": "fixture",
                "http": {"max_request_bytes": self.max_request_bytes}}

    def batching_status(self):
        return {"schema": "mlx2.batch-runtime.v1", "gauges": {"queue_depth": 0}}

    def submit(self, request, *, tenant_id="default", **_):
        self.requests.append(request)
        job = Job(request)
        job.tenant_id = tenant_id
        cap = request.get("max_tokens")
        produced = self.per_round if cap is None else min(self.per_round, cap)
        job.prompt_tokens, job.completion_tokens = 5, produced
        if len(self.requests) <= 2:
            job.events.put({"delta": {"tool_calls": [{
                "index": 0,
                "id": f"call_{len(self.requests)}",
                "type": "function",
                "function": {"name": "weather", "arguments": '{"city":"Toronto"}'},
            }]}})
            reason = "tool_calls"
        else:
            job.events.put({"delta": {"content": "It is 21 C."}})
            reason = "length" if cap is not None and produced >= cap else "stop"
        job.events.put({"finish_reason": reason, "receipt": {}})
        return job


def _respond(engine, backend, body):
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        handler_for(engine, tool_backend=backend, sse_keepalive_seconds=None),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(Request(base + "/v1/responses", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})) as response:
            raw = response.read().decode()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    if body.get("stream"):
        events = [json.loads(line[len("data: "):]) for line in raw.splitlines()
                  if line.startswith("data: {")]
        # Exactly one terminal event ends the stream, named after the object's
        # status: a truncated response ends with response.incomplete.
        terminal = [e["type"] for e in events if e["type"] in (
            "response.completed", "response.incomplete", "response.failed")]
        assert len(terminal) == 1 and events[-1]["type"] == terminal[0], terminal
        payload = events[-1]["response"]
        assert terminal[0] == "response." + payload["status"], terminal
        return payload
    return json.loads(raw)


@pytest.mark.parametrize("stream", [False, True])
def test_hosted_tool_rounds_share_one_max_output_tokens_budget(stream):
    engine = BudgetEngine()
    payload = _respond(engine, Backend(), {
        "model": "fixture", "input": "weather?", "max_output_tokens": 10,
        "tools": [MCP_TOOL], "stream": stream,
    })
    caps = [request.get("max_tokens") for request in engine.requests]
    # Round 1 spent 4 of 10, round 2 another 4: the continuations may only
    # ask for what is left of the response's budget.
    assert caps == [10, 6, 2]
    assert payload["usage"]["output_tokens"] <= 10, payload["usage"]
    assert payload["mlx2"]["hosted_tools"]["output_budget_exhausted"] is False
    # The last round stopped on what was left of the budget: the response
    # was cut at max_output_tokens.
    assert payload["status"] == "incomplete"
    assert payload["incomplete_details"] == {"reason": "max_output_tokens"}


@pytest.mark.parametrize("stream", [False, True])
def test_an_exhausted_budget_ends_the_loop_without_another_round(stream):
    engine, backend = BudgetEngine(), Backend()
    payload = _respond(engine, backend, {
        "model": "fixture", "input": "weather?", "max_output_tokens": 8,
        "tools": [MCP_TOOL], "stream": stream,
    })
    # Two rounds spend the whole budget; the second round's call is neither
    # executed (no round is left to read its output) nor handed to the client
    # as a function call it cannot run.
    assert [request.get("max_tokens") for request in engine.requests] == [8, 4]
    assert backend.executed == 1
    assert payload["usage"]["output_tokens"] == 8
    assert not [item for item in payload["output"] if item["type"] == "function_call"]
    assert payload["mlx2"]["hosted_tools"]["rounds"] == 1
    assert payload["mlx2"]["hosted_tools"]["output_budget_exhausted"] is True
    # The loop ended on the output budget: the response is incomplete, and
    # a stream ends with response.incomplete (see _respond).
    assert payload["status"] == "incomplete"
    assert payload["incomplete_details"] == {"reason": "max_output_tokens"}
    assert [item["status"] for item in payload["output"]
            if item["type"] == "message"] == ["incomplete"]


def test_an_unset_budget_keeps_the_per_round_default():
    engine = BudgetEngine()
    _respond(engine, Backend(), {"model": "fixture", "input": "weather?",
                                 "tools": [MCP_TOOL]})
    assert [request.get("max_tokens") for request in engine.requests] == [None] * 3
