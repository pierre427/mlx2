"""``/v1/messages/count_tokens`` must count the prompt ``/v1/messages`` serves.

docs/SERVING.md: ``/tokenize``'s ``count`` "equals /v1/messages/count_tokens
and the served prompt_tokens".  A named ``tool_choice`` narrows ``tools`` to
the selected function in ``validate_request`` (``normalize_tool_choice``), so
the served prompt renders one tool; ``count_tokens`` returned before that
validation, rendered every declared tool and accepted a choice naming an
undeclared tool that ``/v1/messages`` refuses.
"""

import json
import threading
from collections import Counter
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from mlx2.server import handler_for
from mlx2.serving import Job


def _tokens(request):
    """Deterministic stand-in for ``adapter.prompt_tokens``."""
    text = json.dumps(
        {"messages": request["messages"], "tools": request.get("tools", [])}
    )
    return [len(word) for word in text.split()]


class RenderingEngine:
    model_path = "fixture"
    max_context = 16384

    def __init__(self):
        self.counts = Counter()
        self.counted = []
        self.submitted = []

    def status(self):
        return {"healthy": True, "error": None, "model": "fixture"}

    def batching_status(self):
        return {}

    def render_prompt(self, request):
        return _tokens(request)

    def count_tokens(self, request):
        self.counted.append(request)
        return len(self.render_prompt(request))

    def submit(self, request, *, tenant_id="default", **_):
        self.submitted.append(request)
        job = Job(request)
        job.prompt_tokens = len(_tokens(request))
        job.cached_tokens, job.completion_tokens = 0, 1
        choice = request.get("tool_choice")
        name = choice["function"]["name"] if isinstance(choice, dict) else "tool_0"
        job.events.put(
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": name, "arguments": '{"z": "x"}'},
                        }
                    ]
                }
            }
        )
        job.events.put({"finish_reason": "tool_calls", "receipt": {"cache": "apcv2"}})
        return job


@pytest.fixture
def endpoint():
    engine = RenderingEngine()
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
        method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=20) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        return error.code, error.read().decode()


TOOLS = [
    {
        "name": f"tool_{index}",
        "description": f"tool number {index} does a thing",
        "input_schema": {
            "type": "object",
            "properties": {"z": {"type": "string"}},
            "required": ["z"],
        },
    }
    for index in range(4)
]
BODY = {
    "model": "fixture",
    "messages": [{"role": "user", "content": "hi"}],
    "tools": TOOLS,
    "tool_choice": {"type": "tool", "name": "tool_2"},
}


def test_named_tool_choice_count_matches_served_prompt(endpoint):
    engine, base = endpoint
    status, counted = _post(base, "/v1/messages/count_tokens", BODY)
    assert status == 200
    status, served = _post(base, "/v1/messages", {**BODY, "max_tokens": 32})
    assert status == 200
    assert [t["function"]["name"] for t in engine.submitted[-1]["tools"]] == ["tool_2"]
    assert [t["function"]["name"] for t in engine.counted[-1]["tools"]] == ["tool_2"]
    assert counted["input_tokens"] == served["usage"]["input_tokens"]


def test_count_tokens_counts_the_canonical_tool_schema(endpoint):
    # Equivalent schemas with members in another order render the served
    # prompt from the canonical copy; the count must use the same copy.
    engine, base = endpoint
    shuffled = [
        {
            "input_schema": {
                "required": ["z"],
                "properties": {"z": {"type": "string"}},
                "type": "object",
            },
            "description": tool["description"],
            "name": tool["name"],
        }
        for tool in TOOLS
    ]
    body = {**BODY, "tools": shuffled, "tool_choice": {"type": "auto"}}
    assert _post(base, "/v1/messages/count_tokens", body)[0] == 200
    assert _post(base, "/v1/messages", {**body, "max_tokens": 32})[0] == 200
    # Dict equality ignores member order; the rendered JSON does not.
    assert json.dumps(engine.counted[-1]["tools"]) == json.dumps(
        engine.submitted[-1]["tools"]
    )


def test_count_tokens_rejects_an_undeclared_named_tool(endpoint):
    _engine, base = endpoint
    bad = {**BODY, "tool_choice": {"type": "tool", "name": "missing"}}
    assert _post(base, "/v1/messages", {**bad, "max_tokens": 32})[0] == 400
    assert _post(base, "/v1/messages/count_tokens", bad)[0] == 400
