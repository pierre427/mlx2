"""A lone UTF-16 surrogate in request text is a client error (omlx #4253).

``json.loads`` turns ``"\\ud800"`` into a Python string no tokenizer can
encode; the HF fast tokenizer then raised ``TypeError`` inside prompt
rendering and the client got a 500 "chat template rendering failed".
"""

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from test_serving_contract import FakeEngine

from mlx2.server import _request_json, handler_for


@pytest.mark.parametrize(
    "raw",
    [
        b'{"messages":[{"role":"user","content":"hi \\ud800 there"}]}',
        b'{"messages":[{"role":"user","content":"hi \\uDC00"}]}',
        '{"prompt":"x"}'.encode().replace(b"x", b"\xed\xa0\x80"),
        b'{"\\ud800":1}',
    ],
)
def test_lone_surrogate_is_value_error(raw):
    with pytest.raises(ValueError, match="unpaired UTF-16 surrogate"):
        _request_json(raw)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"prompt":"\\ud83d\\ude00"}',  # a paired surrogate is one code point
        '{"prompt":"한글"}'.encode(),  # Hangul encodes with 0xED lead bytes
        b'{"prompt":"\\\\ud800"}',  # an escaped backslash, not an escape
    ],
)
def test_valid_text_still_parses(raw):
    assert _request_json(raw) == json.loads(raw)


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
def test_http_answers_400_invalid_request(path):
    engine = FakeEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = (
            b'{"model":"fixture","max_tokens":4,'
            b'"messages":[{"role":"user","content":"hi \\ud800"}]}'
        )
        request = Request(
            f"http://127.0.0.1:{server.server_port}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(HTTPError) as raised:
            urlopen(request)
        assert raised.value.code == 400
        payload = json.load(raised.value)
        kind = payload["error"]["type"]
        assert kind == "invalid_request_error", payload
        assert engine.job is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
