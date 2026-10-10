"""``POST /tokenize`` and ``POST /apply-template`` answered 404
"unknown model" for ``"model": null`` while the generation endpoints accept the
same body (7f16cb5b1 made a recognised null mean "absent", but
``prompt_render_payload`` resolves the model from the raw body before
``validate_request`` drops the null).

Style follows tests/test_progress_tokenize.py::test_tokenize_and_apply_template
and tests/test_sweep_20261008_api_null_fields.py.
"""

import json
import sys
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

sys.path.insert(0, "tests")
from test_serving_contract import FakeEngine  # noqa: E402

from mlx2.server import handler_for, prompt_render_payload  # noqa: E402

MESSAGES = [{"role": "user", "content": "hello"}]


class RenderingEngine(FakeEngine):
    def __init__(self):
        super().__init__()
        self.rendered = []

    def render_prompt(self, request):
        self.rendered.append(request)
        return [7, 8, 9]

    def apply_template(self, request):
        self.rendered.append(request)
        return "<user>hello</user>"


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
    return urlopen(
        Request(
            base + path,
            method="POST",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
    )


@pytest.mark.parametrize("path", ["/tokenize", "/apply-template"])
def test_prompt_inspection_treats_null_model_like_generation(endpoint, path):
    engine, base = endpoint
    body = {"messages": MESSAGES, "model": None}
    # Control: generation accepts the body (null model means the loaded model).
    with _post(base, "/v1/chat/completions", body) as response:
        assert response.status == 200
    try:
        with _post(base, path, body) as response:
            assert response.status == 200
            payload = json.load(response)
    except HTTPError as error:
        pytest.fail(f"{path} refused model=null: {error.code} {error.read().decode()}")
    if path == "/tokenize":
        assert payload["count"] == 3
    else:
        assert payload == {"prompt": "<user>hello</user>"}
    assert "model" not in engine.rendered[-1] or engine.rendered[-1]["model"] == "fixture"
    # An unknown model is still a 404, the LoRA-selection contract is kept.
    with pytest.raises(HTTPError) as error:
        _post(base, path, {"messages": MESSAGES, "model": "other"})
    assert error.value.code == 404


def test_prompt_render_payload_resolves_the_model_after_validation():
    engine = RenderingEngine()
    body = {"messages": MESSAGES, "model": None}
    assert prompt_render_payload(engine, body, "/tokenize")["count"] == 3
    assert prompt_render_payload(engine, body, "/apply-template") == {
        "prompt": "<user>hello</user>"
    }


def test_messages_rejects_null_model(endpoint):
    _engine, base = endpoint
    body = {
        "model": None,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hello"}],
    }
    with pytest.raises(HTTPError) as error:
        _post(base, "/v1/messages", body)
    assert error.value.code == 400
    assert json.load(error.value)["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize("path", ["/tokenize", "/apply-template"])
def test_prompt_inspection_treats_null_messages_like_completions(endpoint, path):
    # Review round 1: a null ``messages`` beside ``prompt`` is an absent one,
    # so the body is the Completions request /v1/completions serves.
    engine, base = endpoint
    body = {"prompt": "abc", "messages": None}
    with _post(base, "/v1/completions", body) as response:
        assert response.status == 200
    try:
        with _post(base, path, body) as response:
            assert response.status == 200
    except HTTPError as error:
        pytest.fail(f"{path} refused messages=null: {error.code} {error.read().decode()}")
    assert engine.rendered[-1]["prompt"] == "abc" and "messages" not in engine.rendered[-1]
    # And the mirror: a null ``prompt`` beside ``messages`` is a Chat body.
    with _post(base, path, {"messages": MESSAGES, "prompt": None}) as response:
        assert response.status == 200
    assert "prompt" not in engine.rendered[-1]
