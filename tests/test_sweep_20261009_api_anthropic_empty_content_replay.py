"""Anthropic ``/v1/messages`` accepts a replay of its own empty answer.

A turn that produced no visible text, reasoning or tool call (stopped at once,
or ``max_tokens`` hit before any visible token) is answered with
``content: []``.  A client that appends that assistant message verbatim and
continues (the Anthropic SDK's own ``messages.append(response)`` idiom) gets a
400 "message content must be text or a nonempty block list" from
``_anthropic_message``, although the same validator accepts ``content: ""``.
The Responses API had the analogous defect fixed in 12c8c24d0 (empty text is
accepted for the assistant role only); this is the Anthropic half.

Style follows tests/test_anthropic_compat.py (AnthropicEngine + ThreadingHTTPServer)
and tests/test_responses_integrations.py::test_stateless_replay_accepts_the_servers_own_empty_answer.
"""

import json
import threading
from collections import Counter
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from mlx2.anthropic_compat import anthropic_request_to_chat
from mlx2.server import handler_for
from mlx2.serving import Job


class EmptyAnswerEngine:
    """First turn emits nothing visible; every turn records the request."""

    model_path = "fixture"

    def __init__(self, finish):
        self.counts = Counter()
        self.requests = []
        self.finish = finish

    def status(self):
        return {"healthy": True, "error": None, "model": "fixture"}

    def batching_status(self):
        return {}

    def submit(self, request, *, tenant_id="default"):
        self.requests.append(request)
        job = Job(request)
        job.tenant_id = tenant_id
        job.prompt_tokens = 5
        job.completion_tokens = 1
        job.events.put({"delta": {"content": ""}})
        job.events.put({"finish_reason": self.finish, "receipt": {}})
        return job


@pytest.fixture(params=["length", "stop"])
def endpoint(request):
    engine = EmptyAnswerEngine(request.param)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield engine, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def _post(base, body):
    return urlopen(
        Request(
            base + "/v1/messages",
            method="POST",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
    )


def _assemble_stream(raw):
    """Rebuild ``message.content`` the way the Anthropic SDK's accumulator does."""
    content = []
    for frame in raw.split("\n\n"):
        for line in frame.splitlines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event["type"] == "content_block_start":
                content.append(dict(event["content_block"]))
            elif event["type"] == "content_block_delta":
                delta = event["delta"]
                if delta["type"] == "text_delta":
                    content[event["index"]]["text"] += delta["text"]
    return content


@pytest.mark.parametrize("stream", [False, True])
def test_anthropic_replays_its_own_empty_answer(endpoint, stream):
    engine, base = endpoint
    first = {
        "model": "fixture",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}],
    }
    with _post(base, {**first, "stream": stream}) as response:
        assert response.status == 200
        raw = response.read().decode()
    content = _assemble_stream(raw) if stream else json.loads(raw)["content"]
    # The server's own answer for a turn without visible output.
    assert content == [] or content == [{"type": "text", "text": ""}]

    # ``messages.append({"role": "assistant", "content": response.content})``
    # then continue: the server must accept what it emitted.
    follow_up = {
        **first,
        "messages": [
            *first["messages"],
            {"role": "assistant", "content": content},
            {"role": "user", "content": "continue"},
        ],
    }
    try:
        with _post(base, follow_up) as response:
            assert response.status == 200
    except HTTPError as error:
        pytest.fail(
            f"replay of the server's own empty answer was refused: "
            f"{error.code} {error.read().decode()}"
        )
    history = engine.requests[-1]["messages"]
    assert [message["role"] for message in history] == ["user", "assistant", "user"]
    assert history[1]["content"] == ""


def test_empty_assistant_block_list_is_accepted_but_empty_user_list_is_not():
    # Symmetric with Responses 12c8c24d0: the assistant role may replay an
    # empty answer; user content stays required.
    base = {"model": "fixture", "max_tokens": 8}
    request = anthropic_request_to_chat(
        {
            **base,
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": []},
                {"role": "user", "content": "b"},
            ],
        }
    )
    assert [m["content"] for m in request["messages"]] == ["a", "", "b"]
    with pytest.raises(ValueError, match="nonempty"):
        anthropic_request_to_chat(
            {**base, "messages": [{"role": "user", "content": []}]}
        )
