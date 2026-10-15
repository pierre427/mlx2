"""A truncated Responses generation reports OpenAI's incomplete semantics.

A generation that ends on the output cap (``finish_reason:"length"``) is a
Responses object with ``status:"incomplete"`` and
``incomplete_details:{"reason":"max_output_tokens"}``; the item it cut off
(the last) is ``incomplete`` and earlier items ``completed``, and a stream ends with ``response.incomplete`` instead of
``response.completed``.  The live, streamed, stored and batch-row objects all
come from ``responses_payload`` and agree.  A normal stop is unchanged.
"""

import json
import threading
import time
from collections import Counter
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

import pytest

from mlx2.api_resources import ResponseStore
from mlx2.server import MIN_REQUEST_BODY_BYTES, handler_for
from mlx2.serving import Job

INCOMPLETE = {"reason": "max_output_tokens"}


class CapEngine:
    """Decodes ``per_request`` tokens, or stops early on a smaller cap."""

    model_path = "fixture"
    max_context = 16384
    per_request = 4

    def __init__(self, *, reasoning_only=False):
        self.lock = threading.Lock()
        self.counts = Counter()
        self.max_request_bytes = MIN_REQUEST_BODY_BYTES
        self.reasoning_only = reasoning_only
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
        produced = self.per_request if cap is None else min(self.per_request, cap)
        job.prompt_tokens, job.completion_tokens = 3, produced
        job.events.put({"delta": {"reasoning_content": "thinking"}})
        if not self.reasoning_only:
            job.events.put({"delta": {"content": "partial"}})
        reason = "length" if cap is not None and produced >= cap else "stop"
        job.events.put({"finish_reason": reason, "receipt": {}})
        return job


class _Served:
    def __init__(self, engine):
        self.engine = engine

    def __enter__(self):
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            handler_for(
                self.engine,
                response_store=ResponseStore(),
                sse_keepalive_seconds=None,
            ),
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}"

    def __exit__(self, *_exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def _post(base, path, body, content_type="application/json"):
    return urlopen(Request(base + path, data=body, headers={"Content-Type": content_type}))


def _respond(base, body):
    with _post(base, "/v1/responses", json.dumps(body).encode()) as response:
        raw = response.read().decode()
    if not body.get("stream"):
        return json.loads(raw), None, raw
    events = [json.loads(line[len("data: "):]) for line in raw.splitlines()
              if line.startswith("data: {")]
    return events[-1]["response"], events, raw


def _stored(base, response_identifier):
    with urlopen(base + "/v1/responses/" + response_identifier) as response:
        return json.load(response)


def _message(payload):
    (item,) = [item for item in payload["output"] if item["type"] == "message"]
    return item


@pytest.mark.parametrize("reasoning_only", [False, True], ids=["text", "reasoning-only"])
def test_a_truncated_response_is_incomplete_and_stored_that_way(reasoning_only):
    engine = CapEngine(reasoning_only=reasoning_only)
    with _Served(engine) as base:
        payload, _, _ = _respond(base, {
            "model": "fixture", "input": "hi", "max_output_tokens": 2,
        })
        stored = _stored(base, payload["id"])
    assert engine.requests[0]["max_tokens"] == 2
    assert payload["status"] == "incomplete"
    assert payload["incomplete_details"] == INCOMPLETE
    statuses = [(item["type"], item["status"]) for item in payload["output"]]
    if reasoning_only:
        # Cut inside reasoning: that item is incomplete, and so is the empty
        # answer item stateless replay round-trips.
        assert statuses == [("reasoning", "incomplete"), ("message", "incomplete")]
        assert _message(payload)["content"][0]["text"] == ""
    else:
        # The cut-off answer is an incomplete item; the reasoning before it ended.
        assert statuses == [("reasoning", "completed"), ("message", "incomplete")]
        assert _message(payload)["content"][0]["text"] == "partial"
    assert stored == payload


def test_a_truncated_stream_ends_with_response_incomplete():
    engine = CapEngine()
    with _Served(engine) as base:
        payload, events, raw = _respond(base, {
            "model": "fixture", "input": "hi", "max_output_tokens": 2,
            "stream": True,
        })
        stored = _stored(base, payload["id"])
    terminal = [event["type"] for event in events if event["type"] in (
        "response.completed", "response.incomplete", "response.failed")]
    assert terminal == ["response.incomplete"], terminal
    assert events[-1]["type"] == "response.incomplete"
    # The SSE event name is the terminal type too.
    assert "event: response.incomplete\n" in raw
    assert "event: response.completed\n" not in raw
    assert payload["status"] == "incomplete"
    assert payload["incomplete_details"] == INCOMPLETE
    assert _message(payload)["status"] == "incomplete"
    # The message item streamed live: it opened in progress and closed
    # incomplete, as the terminal object reports it.
    added = [event["item"] for event in events if event["type"] == "response.output_item.added"
             and event["item"]["type"] == "message"]
    done = [event["item"] for event in events if event["type"] == "response.output_item.done"
            and event["item"]["type"] == "message"]
    assert [item["status"] for item in added] == ["in_progress"]
    assert [item["status"] for item in done] == ["incomplete"]
    assert stored == payload


def test_a_truncated_batch_row_is_incomplete():
    engine = CapEngine()
    row = {"custom_id": "cut", "method": "POST", "url": "/v1/responses",
           "body": {"model": "fixture", "input": "hi", "max_output_tokens": 2}}
    boundary = "mlx2-test-boundary"
    upload = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="rows.jsonl"\r\n'
        "Content-Type: application/jsonl\r\n\r\n"
    ).encode() + json.dumps(row).encode() + f"\n\r\n--{boundary}--\r\n".encode()
    with _Served(engine) as base:
        with _post(base, "/v1/files", upload,
                   f"multipart/form-data; boundary={boundary}") as response:
            file_id = json.load(response)["id"]
        with _post(base, "/v1/batches", json.dumps({
            "input_file_id": file_id, "endpoint": "/v1/responses",
            "completion_window": "24h",
        }).encode()) as response:
            batch = json.load(response)
        for _ in range(500):
            with urlopen(base + "/v1/batches/" + batch["id"]) as response:
                batch = json.load(response)
            if batch["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.01)
        assert batch["request_counts"] == {"total": 1, "completed": 1, "failed": 0}
        with urlopen(base + f"/v1/files/{batch['output_file_id']}/content") as response:
            (result,) = [json.loads(line) for line in response.read().splitlines()]
        payload = result["response"]["body"]
        stored = _stored(base, payload["id"])
    assert payload["status"] == "incomplete"
    assert payload["incomplete_details"] == INCOMPLETE
    assert _message(payload)["status"] == "incomplete"
    assert stored == payload


@pytest.mark.parametrize("stream", [False, True])
def test_a_normal_stop_stays_completed(stream):
    engine = CapEngine()
    with _Served(engine) as base:
        payload, events, raw = _respond(base, {
            "model": "fixture", "input": "hi", "max_output_tokens": 64,
            "stream": stream,
        })
        stored = _stored(base, payload["id"])
    assert payload["status"] == "completed"
    assert "incomplete_details" not in payload
    assert [item["status"] for item in payload["output"]] == ["completed", "completed"]
    assert stored == payload
    if stream:
        terminal = [event["type"] for event in events if event["type"] in (
            "response.completed", "response.incomplete", "response.failed")]
        assert terminal == ["response.completed"], terminal
        assert "event: response.incomplete\n" not in raw


class _Job:
    id = "job1"
    created = 0
    request = {}


def _payload(message, finish_reason, last_output=None):
    from mlx2.openai_compat import responses_payload

    usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    job = _Job()
    if last_output is not None:
        job.responses_last_output = last_output
    return responses_payload(
        job=job, model="fixture",
        choice={"message": message, "finish_reason": finish_reason},
        usage=usage, receipt={}, metadata={},
    )


def _call(index):
    return {"id": f"call_{index}", "type": "function",
            "function": {"name": "f", "arguments": "{}"}}


def test_a_truncated_tool_call_is_the_incomplete_item():
    payload = _payload(
        {"content": "let me check", "reasoning_content": "plan",
         "tool_calls": [_call(0), _call(1)]},
        "length",
    )
    assert payload["status"] == "incomplete"
    assert [(item["type"], item["status"]) for item in payload["output"]] == [
        ("reasoning", "completed"),
        ("message", "completed"),
        ("function_call", "completed"),
        ("function_call", "incomplete"),
    ]
    finished = _payload({"content": "", "tool_calls": [_call(0)]}, "tool_calls")
    assert [item["status"] for item in finished["output"]] == ["completed"]


@pytest.mark.parametrize("status", ["incomplete", "completed"])
def test_agent_compat_custom_tool_call_keeps_the_item_status(status):
    from mlx2.agent_compat import (
        AgentCompatPolicy,
        rewrite_responses_output,
        translate_responses_tools,
    )

    compat = AgentCompatPolicy("auto", "off").resolve(
        {"X-MLX2-Agent-Compat": "on"}, "default", "/v1/responses"
    )
    _, tool_map = translate_responses_tools(
        [{"type": "custom", "name": "t"}], compat, Counter()
    )
    payload = {
        "output": [{
            "type": "function_call", "id": "fc_1", "call_id": "call_1",
            "status": status, "name": "t",
            "arguments": json.dumps({"input": "x"}),
        }],
        "mlx2": {},
    }
    rewrite_responses_output(payload, tool_map, compat, Counter())
    (item,) = payload["output"]
    assert (item["type"], item["status"]) == ("custom_tool_call", status)


def test_the_cut_item_is_marked_again_after_a_stream_reorders_output():
    # Codex review: a grammar call streamed before the text that was cut;
    # the stream sorts output into arrival order after responses_payload.
    from mlx2.openai_compat import mark_truncated_output

    payload = _payload(
        {"content": "trailing text", "tool_calls": [_call(0)]}, "length"
    )
    assert [item["type"] for item in payload["output"]] == ["message", "function_call"]
    payload["output"].reverse()  # arrival order: the call opened first
    mark_truncated_output(payload)
    assert [(item["type"], item["status"]) for item in payload["output"]] == [
        ("function_call", "completed"),
        ("message", "incomplete"),
    ]


def test_marking_leaves_other_item_statuses_alone():
    from mlx2.openai_compat import mark_truncated_output

    payload = {"status": "incomplete", "output": [
        {"type": "web_search_call", "status": "failed"},
        {"type": "message", "status": "completed", "content": [{"text": "x"}]},
    ]}
    mark_truncated_output(payload)
    assert [item["status"] for item in payload["output"]] == ["failed", "incomplete"]


def test_a_non_streamed_cut_marks_the_kind_that_arrived_last():
    # Codex review round 2: a call completed, then trailing text hit the cap.
    # The payload lists the message first, so position alone marks the call.
    message = {"content": "trailing text", "tool_calls": [_call(0)]}
    cut_text = _payload(message, "length", last_output="message")
    assert [(item["type"], item["status"]) for item in cut_text["output"]] == [
        ("message", "incomplete"), ("function_call", "completed"),
    ]
    cut_call = _payload(message, "length", last_output="function_call")
    assert [(item["type"], item["status"]) for item in cut_call["output"]] == [
        ("message", "completed"), ("function_call", "incomplete"),
    ]


def test_the_output_kind_is_recorded_from_deltas():
    from mlx2.openai_compat import note_output_kind

    job = _Job()
    for delta, kind in (
        ({"reasoning_content": "r"}, "reasoning"),
        ({"content": "a"}, "message"),
        ({"tool_calls": [_call(0)]}, "function_call"),
        ({"content": ""}, "function_call"),
    ):
        note_output_kind(job, delta)
        assert job.responses_last_output == kind
