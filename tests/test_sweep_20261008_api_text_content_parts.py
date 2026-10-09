"""OpenAI Chat accepts ``content`` as an array of text parts on every role.

A request made only of text parts asks for no media capability.  The chat
path counted any list-valued content as media: a text route answered 501 "no
qualified image, video, or audio encoder" for a user text-part array and 400
"multimodal content requires a nonempty user message" for a system or
assistant one, while ``/v1/responses`` and ``/v1/messages`` flatten the same
text and serve it.  A real image part on a text route must still fail closed.
"""

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from test_serving_contract import FakeEngine
from test_structured_deferral import _collect, scripted_engine  # noqa: F401 - shared fixture

from mlx2.server import handler_for, validate_request

PNG = "data:image/png;base64,iVBORw0KGgo="


@pytest.fixture
def text_server():
    engine = FakeEngine()  # no supports_multimodal: a text-only route
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield engine, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def _chat(base, messages):
    request = Request(
        base + "/v1/chat/completions",
        data=json.dumps({"model": "fixture", "messages": messages}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        return exc.code, json.load(exc)


def _parts(*texts):
    return [{"type": "text", "text": text} for text in texts]


@pytest.mark.parametrize(
    "messages, flattened",
    [
        (
            [{"role": "user", "content": _parts("hi ", "there")}],
            [{"role": "user", "content": "hi there"}],
        ),
        (
            [
                {"role": "system", "content": _parts("be brief")},
                {"role": "user", "content": "hi"},
            ],
            [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
            ],
        ),
        (
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": _parts("hello")},
                {"role": "user", "content": "again"},
            ],
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "again"},
            ],
        ),
    ],
    ids=["user", "system", "assistant"],
)
def test_text_part_arrays_are_text_on_a_text_route(text_server, messages, flattened):
    engine, base = text_server
    status, payload = _chat(base, messages)
    assert status == 200, payload
    assert engine.job is not None
    assert engine.job.request["messages"] == flattened
    assert engine.counts["multimodal_rejected"] == 0


def test_media_parts_still_fail_closed_on_a_text_route(text_server):
    engine, base = text_server
    content = [*_parts("describe"), {"type": "image_url", "image_url": {"url": PNG}}]
    status, payload = _chat(base, [{"role": "user", "content": content}])
    assert status == 501, payload
    assert "encoder" in payload["error"]["message"]
    assert engine.job is None
    assert engine.counts["multimodal_rejected"] == 1


def test_media_parts_still_require_a_nonempty_user_message():
    image = {"type": "image_url", "image_url": {"url": PNG}}
    for messages in (
        [{"role": "system", "content": [image]}, {"role": "user", "content": "hi"}],
        [{"role": "user", "content": []}],
    ):
        with pytest.raises(ValueError, match="nonempty user message"):
            validate_request({"messages": messages})


def test_engine_serves_a_text_part_array_without_a_media_hook(scripted_engine):
    build, state = scripted_engine
    engine = build(declare_marker=True)
    assert not engine.supports_multimodal()
    seen = []
    render = engine.adapter.prompt_tokens

    def prompt_tokens(request):
        seen.append(request["messages"])
        return render(request)

    engine.adapter.prompt_tokens = prompt_tokens
    state["script"] = []
    job = engine.submit(
        {
            "messages": [{"role": "user", "content": _parts("hi")}],
            "temperature": 0,
            "max_tokens": 2,
        }
    )
    _, _, final = _collect(job)
    assert "error" not in final, final
    assert seen and seen[-1] == [{"role": "user", "content": "hi"}]
