"""A ``{"type": "text"}`` content part without a string ``text`` is a 400.

The part check validated only ``type``; ``flatten_text_parts`` requires a
string ``text`` and otherwise leaves the list alone, so a text part with
``text: null`` or no ``text`` at all survived as list content and was treated
as media: 501 "no qualified image, video, or audio encoder" on a text route,
and a silently dropped part (an empty user turn) on a VLM route.  The validator
and the flattener now agree on what a text part is.
"""

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from test_serving_contract import FakeEngine

from mlx2.server import handler_for, validate_request

BAD_PARTS = [
    [{"type": "text", "text": None}],
    [{"type": "text"}],
    [{"type": "text", "text": 1}],
    [{"type": "text", "text": "ok"}, {"type": "text"}],
    [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
        {"type": "text", "text": None},
    ],
]


class MultimodalEngine(FakeEngine):
    def supports_multimodal(self):
        return True


@pytest.fixture(params=[FakeEngine, MultimodalEngine], ids=["text-route", "vlm-route"])
def endpoint(request):
    engine = request.param()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield engine, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def _chat(base, content):
    request = Request(
        base + "/v1/chat/completions",
        data=json.dumps(
            {"model": "fixture", "messages": [{"role": "user", "content": content}]}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        return exc.code, json.load(exc)


@pytest.mark.parametrize("content", BAD_PARTS)
def test_validator_requires_a_text_string_on_text_parts(content):
    with pytest.raises(ValueError, match="text"):
        validate_request({"messages": [{"role": "user", "content": content}]}, True)


@pytest.mark.parametrize("content", BAD_PARTS)
def test_text_part_without_text_is_a_400_on_every_route(endpoint, content):
    engine, base = endpoint
    status, payload = _chat(base, content)
    assert status == 400, payload
    assert payload["error"]["type"] == "invalid_request_error"
    assert engine.job is None


def test_valid_text_parts_still_flatten_and_serve(endpoint):
    engine, base = endpoint
    status, payload = _chat(base, [{"type": "text", "text": "hi "}, {"type": "text", "text": "there"}])
    assert status == 200, payload
    assert engine.job.request["messages"] == [{"role": "user", "content": "hi there"}]
