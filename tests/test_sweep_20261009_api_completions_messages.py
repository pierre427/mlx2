"""``/v1/completions`` refuses ``messages``; ``/v1/chat/completions`` refuses ``prompt``.

The validator checked ``prompt`` for a Completions body and ``messages`` for a
Chat one but let the other route's field through, while every adapter decides
by ``"messages" in request``: a Completions body carrying both (a client that
reuses one request builder for both endpoints) was validated on ``prompt`` and
then rendered through the chat template with the chat parser, so the
``text_completion`` came from a prompt the caller never sent.  Each route now
owns its field: the other one is an unsupported field for that route.
"""

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from test_serving_contract import FakeEngine

from mlx2.server import handler_for, prompt_render_payload, validate_request

MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture
def endpoint():
    engine = FakeEngine()
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
            return response.status, json.load(response)
    except HTTPError as exc:
        return exc.code, json.load(exc)


def test_validator_refuses_the_other_routes_field():
    with pytest.raises(ValueError, match="messages"):
        validate_request({"prompt": "abc", "messages": MESSAGES}, False)
    with pytest.raises(ValueError, match="prompt"):
        validate_request({"prompt": "abc", "messages": MESSAGES}, True)
    # Each route still accepts its own field alone.
    assert validate_request({"prompt": "abc"}, False)["prompt"] == "abc"
    assert validate_request({"messages": MESSAGES}, True)["messages"] == MESSAGES


def test_completions_with_messages_is_a_400_and_never_reaches_the_engine(endpoint):
    engine, base = endpoint
    status, payload = _post(
        base, "/v1/completions", {"model": "fixture", "prompt": "abc", "messages": MESSAGES}
    )
    assert status == 400, payload
    assert "messages" in payload["error"]["message"]
    assert engine.job is None

    status, payload = _post(
        base, "/v1/chat/completions", {"model": "fixture", "prompt": "abc", "messages": MESSAGES}
    )
    assert status == 400, payload
    assert "prompt" in payload["error"]["message"]
    assert engine.job is None

    # The plain Completions body still serves.
    status, payload = _post(base, "/v1/completions", {"model": "fixture", "prompt": "abc"})
    assert status == 200 and payload["object"] == "text_completion"


def test_prompt_render_refuses_a_body_that_names_both():
    class RenderingEngine(FakeEngine):
        def render_prompt(self, request):
            return [7, 8, 9]

    engine = RenderingEngine()
    with pytest.raises(ValueError, match="prompt|messages"):
        prompt_render_payload(engine, {"prompt": "abc", "messages": MESSAGES}, "/tokenize")
