"""The lone-surrogate guard (omlx #4253) also covers UTF-16/UTF-32 bodies.

``json.loads(bytes)`` auto-detects UTF-16 and UTF-32, where a ``\\ud800``
escape is not the byte sequence ``b"\\\\ud"`` and a raw surrogate code unit is
not a ``0xED`` byte.  The byte scan in ``_request_json`` skipped those bodies,
so the lone surrogate reached the tokenizer and the client got a 500
"chat template rendering failed" instead of a 400.
"""

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from test_serving_contract import FakeEngine

from mlx2.server import _request_json, handler_for

_ESCAPED = json.dumps({"messages": [{"role": "user", "content": "x\ud800y"}]})
_RAW = json.dumps(
    {"messages": [{"role": "user", "content": "x\udc00y"}]}, ensure_ascii=False
)


@pytest.mark.parametrize(
    "raw",
    [
        _ESCAPED.encode("utf-16"),
        _ESCAPED.encode("utf-16-le"),
        _ESCAPED.encode("utf-16-be"),
        _ESCAPED.encode("utf-32"),
        _ESCAPED.encode("utf-32-le"),
        _RAW.encode("utf-16-le", "surrogatepass"),
        _RAW.encode("utf-32-le", "surrogatepass"),
    ],
    ids=[
        "utf16-bom-escape",
        "utf16le-escape",
        "utf16be-escape",
        "utf32-bom-escape",
        "utf32le-escape",
        "utf16le-raw-unit",
        "utf32le-raw-unit",
    ],
)
def test_non_utf8_lone_surrogate_is_value_error(raw):
    with pytest.raises(ValueError, match="unpaired UTF-16 surrogate"):
        _request_json(raw)


@pytest.mark.parametrize("encoding", ["utf-16", "utf-32"])
def test_http_non_utf8_lone_surrogate_answers_400(encoding):
    engine = FakeEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = json.dumps(
            {
                "model": "fixture",
                "max_tokens": 4,
                "messages": [{"role": "user", "content": "hi \ud800"}],
            }
        ).encode(encoding)
        request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(HTTPError) as raised:
            urlopen(request)
        assert raised.value.code == 400
        payload = json.load(raised.value)
        assert payload["error"]["type"] == "invalid_request_error", payload
        assert engine.job is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("encoding", ["utf-16", "utf-16-le", "utf-32"])
def test_valid_non_utf8_text_still_parses(encoding):
    text = json.dumps({"prompt": "\U0001f600 😀 한글"})
    assert _request_json(text.encode(encoding)) == json.loads(text)
