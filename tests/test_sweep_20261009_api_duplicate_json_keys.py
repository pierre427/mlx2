"""A request body that repeats an object key is a 400 naming the key.

``_request_json`` parsed bodies with plain ``json.loads``, which keeps the
last of duplicate keys silently: two ``messages`` or two ``max_tokens`` in
one body served the second as if the first had never been sent.  The
decisions server refuses duplicates (``decode_request_body`` /
``DuplicateKeyError``, 30b49cbd3); the main API now does the same through
its one body parser, for every JSON route.
"""

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from test_serving_contract import FakeEngine

from mlx2.server import DuplicateKeyError, _request_json, handler_for


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


def _post_raw(base, path, raw):
    request = Request(
        base + path, data=raw.encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urlopen(request) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        return exc.code, json.load(exc)


MESSAGE = '{"role": "user", "content": "hi"}'
CHAT = '{"model": "fixture", "messages": [' + MESSAGE + "]"


def test_request_json_refuses_duplicate_keys():
    with pytest.raises(DuplicateKeyError, match="'max_tokens'"):
        _request_json('{"max_tokens": 1, "max_tokens": 2}')
    with pytest.raises(ValueError, match="'role'"):  # nested objects too
        _request_json('{"messages": [{"role": "user", "role": "system"}]}')
    assert _request_json('{"max_tokens": 1}') == {"max_tokens": 1}


@pytest.mark.parametrize(
    "raw, key",
    [
        (CHAT + ', "messages": [' + MESSAGE + "]}", "messages"),
        (CHAT + ', "max_tokens": 1, "max_tokens": 2}', "max_tokens"),
        ('{"model": "fixture", "messages": [{"role": "user", "content": "a", "content": "b"}]}', "content"),
    ],
)
def test_chat_duplicate_key_is_a_400_naming_the_key(endpoint, raw, key):
    engine, base = endpoint
    status, payload = _post_raw(base, "/v1/chat/completions", raw)
    assert status == 400, payload
    assert payload["error"]["type"] == "invalid_request_error"
    assert f"duplicate object key '{key}'" in payload["error"]["message"]
    assert engine.job is None
    # The same body without the repeat serves.
    assert _post_raw(base, "/v1/chat/completions", CHAT + "}")[0] == 200


def test_every_json_route_shares_the_refusal(endpoint):
    engine, base = endpoint
    status, payload = _post_raw(
        base, "/v1/messages",
        '{"model": "fixture", "max_tokens": 4, "max_tokens": 8, "messages": [' + MESSAGE + "]}",
    )
    assert status == 400 and payload["type"] == "error", payload
    assert "duplicate object key 'max_tokens'" in payload["error"]["message"]
    status, payload = _post_raw(base, "/v1/completions", '{"prompt": "a", "prompt": "b"}')
    assert status == 400 and "'prompt'" in payload["error"]["message"]
    assert engine.job is None


def test_batch_rows_share_the_refusal(endpoint):
    # Review round 4: batch JSONL rows are another body entry point and were
    # still parsed last-wins.  A duplicate anywhere in the row fails that row
    # with the same message; siblings still run.
    from test_serving_contract import _run_batch

    engine, base = endpoint
    rows = [
        {"custom_id": "dup", "method": "POST", "url": "/v1/chat/completions",
         "body": {"model": "fixture", "messages": [{"role": "user", "content": "hi"}]}},
        {"custom_id": "ok", "method": "POST", "url": "/v1/chat/completions",
         "body": {"model": "fixture", "messages": [{"role": "user", "content": "hi"}]}},
    ]
    content = (
        json.dumps(rows[0])[:-2] + ', "max_tokens": 1, "max_tokens": 2}}\n'
        + json.dumps(rows[1]) + "\n"
    ).encode()
    from test_serving_contract import _json_post, upload_file

    with upload_file(base, content) as response:
        file_id = json.load(response)["id"]
    with _json_post(base, "/v1/batches", {
        "input_file_id": file_id, "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
    }) as response:
        batch = json.load(response)
    import time
    for _ in range(500):
        with urlopen(base + "/v1/batches/" + batch["id"]) as response:
            batch = json.load(response)
        if batch["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.01)
    assert batch["request_counts"] == {"total": 2, "completed": 1, "failed": 1}
    with urlopen(base + f"/v1/files/{batch['error_file_id']}/content") as response:
        errors = [json.loads(line) for line in response.read().splitlines()]
    assert len(errors) == 1 and "duplicate object key 'max_tokens'" in json.dumps(errors[0])
