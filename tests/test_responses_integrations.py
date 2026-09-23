import json
import threading
from collections import Counter
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib.request import Request, urlopen

import pytest

from mlx2.api_resources import ResponseStore
from mlx2.openai_compat import responses_payload, responses_to_chat_request
from mlx2.reasoning_signatures import ReasoningSigner
from mlx2.server import handler_for
from mlx2.serving import Job


def test_responses_signed_unicode_reasoning_and_logprobs_serialize_on_main_payload():
    signer = ReasoningSigner(b"shared-secret")
    text = "raisonnement café 漢字"
    token = signer.sign_responses(model="fixture", tenant="tenant-a", text=text)
    request, options = responses_to_chat_request(
        {
            "model": "fixture",
            "input": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": text}],
                    "encrypted_content": token,
                },
                {"role": "user", "content": "continue"},
            ],
            "include": [
                "reasoning.encrypted_content",
                "message.output_text.logprobs",
            ],
            "top_logprobs": 1,
            "reasoning": {"summary": "auto"},
            "text": {"verbosity": "low"},
            "user": "client",
            "prompt_cache_key": "hint",
            "truncation": "disabled",
            "service_tier": "default",
            "stream_options": {"include_usage": True},
        },
        signer=signer,
        tenant_id="tenant-a",
        model="fixture",
    )
    assert request["messages"][0]["reasoning_content"] == text
    assert options.get("reasoning_signature_rejections", 0) == 0
    job = SimpleNamespace(
        id="job", created=1, cached_tokens=0,
        request={"parallel_tool_calls": True, "tool_choice": "auto"},
    )
    payload = responses_payload(
        job=job,
        model="fixture",
        choice={
            "finish_reason": "length",
            "message": {"role": "assistant", "content": " hi", "reasoning_content": text},
            "logprobs": {"content": [{"token": "▁hi", "logprob": -0.1, "bytes": list("▁hi".encode()), "top_logprobs": []}]},
        },
        usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        receipt={},
        metadata={},
        signer=signer,
        tenant_id="tenant-a",
        include=options["include"],
    )
    assert payload["status"] == "completed"
    assert "incomplete_details" not in payload
    assert signer.verify_responses(
        payload["output"][0]["encrypted_content"],
        model="fixture",
        tenant="tenant-a",
        expected_text=text,
    ) == text
    assert payload["output"][1]["content"][0]["logprobs"][0]["bytes"] == list("▁hi".encode())


@pytest.mark.parametrize("extra", [
    {"include": [{}]},
    {"service_tier": {}},
])
def test_responses_reject_unhashable_option_shapes_as_validation_errors(extra):
    with pytest.raises(ValueError):
        responses_to_chat_request({"input": "hi", **extra})


class ContinuationEngine:
    model_path = "fixture"

    def __init__(self):
        self.counts = Counter()
        self.reasoning_signer = ReasoningSigner(b"shared-secret")
        self.requests = []

    def status(self):
        return {"healthy": True, "error": None, "model": "fixture"}

    def batching_status(self):
        return {}

    def submit(self, request, *, tenant_id="default"):
        self.requests.append(request)
        job = Job(request)
        job.prompt_tokens = 2
        job.completion_tokens = 2
        # The engine emits each token's logprob before that token's deltas,
        # reasoning tokens included.
        if request.get("logprobs"):
            job.events.put(
                {
                    "logprob": {
                        "id": 6,
                        "token": "trusted",
                        "logprob": -0.5,
                        "bytes": list(b"trusted"),
                        "top_logprobs": [],
                    }
                }
            )
        job.events.put({"delta": {"reasoning_content": "trusted thought"}})
        if request.get("logprobs"):
            job.events.put(
                {
                    "logprob": {
                        "id": 7,
                        "token": "answer",
                        "logprob": -0.25,
                        "bytes": list(b"answer"),
                        "top_logprobs": [],
                    }
                }
            )
        job.events.put({"delta": {"content": "answer"}})
        job.events.put({"finish_reason": "stop", "receipt": {}})
        return job


def _post(base, body):
    return urlopen(Request(
        base + "/v1/responses",
        method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-Tenant-ID": "tenant-a"},
    ))


def test_server_held_continuation_omits_reasoning_and_store_failures_surface():
    engine = ContinuationEngine()
    store = ResponseStore()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine, response_store=store))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with _post(base, {"model": "fixture", "input": "first", "include": ["reasoning.encrypted_content"]}) as response:
            first = json.load(response)
        with _post(base, {"model": "fixture", "input": "second", "previous_response_id": first["id"]}) as response:
            assert response.status == 200
        assert any(
            message.get("reasoning_content") == "trusted thought"
            for message in engine.requests[-1]["messages"]
        )
        assert engine.counts["reasoning_signature_rejections"] == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_streaming_responses_preserve_requested_reasoning_logprobs_and_store_context():
    engine = ContinuationEngine()
    store = ResponseStore()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), handler_for(engine, response_store=store)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with _post(
            base,
            {
                "model": "fixture",
                "input": "first",
                "stream": True,
                "include": [
                    "reasoning.encrypted_content",
                    "message.output_text.logprobs",
                ],
            },
        ) as response:
            wire = response.read().decode()
        assert "chat.completion.chunk" not in wire
        events = [
            json.loads(line.removeprefix("data: "))
            for line in wire.splitlines()
            if line.startswith("data: {")
        ]
        logprob_events = [
            event
            for event in events
            if event["type"] == "response.output_text.delta"
            and event.get("logprobs")
        ]
        assert logprob_events[0]["logprobs"][0]["bytes"] == list(b"answer")
        completed = next(
            event["response"]
            for event in events
            if event["type"] == "response.completed"
        )
        assert [item["type"] for item in completed["output"]] == [
            "reasoning",
            "message",
        ]
        reasoning = completed["output"][0]
        assert engine.reasoning_signer.verify_responses(
            reasoning["encrypted_content"],
            model="fixture",
            tenant="tenant-a",
            expected_text="trusted thought",
        ) == "trusted thought"
        assert completed["output"][1]["content"][0]["logprobs"][0][
            "token"
        ] == "answer"

        with _post(
            base,
            {
                "model": "fixture",
                "input": "second",
                "previous_response_id": completed["id"],
            },
        ) as response:
            assert response.status == 200
        assert any(
            message.get("reasoning_content") == "trusted thought"
            for message in engine.requests[-1]["messages"]
        )

        with _post(base, {"model": "fixture", "input": "plain"}) as response:
            plain = json.load(response)
        plain_reasoning = next(
            item for item in plain["output"] if item["type"] == "reasoning"
        )
        assert "encrypted_content" not in plain_reasoning
        assert not any(
            "reasoning_content" in message
            for message in store.context("tenant-a", plain["id"])
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    class BrokenStore(ResponseStore):
        def put(self, *_args, **_kwargs):
            raise OSError("disk unavailable")

    engine = ContinuationEngine()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), handler_for(engine, response_store=BrokenStore())
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(Exception):
            _post(
                f"http://127.0.0.1:{server.server_port}",
                {"model": "fixture", "input": "store failure surfaces"},
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class _Served:
    """Serve one engine on an ephemeral port for the duration of a test."""

    def __init__(self, engine, **kwargs):
        self.engine = engine
        self.kwargs = kwargs

    def __enter__(self):
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), handler_for(self.engine, **self.kwargs)
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}"

    def __exit__(self, *_exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def test_reasoning_output_item_round_trips_and_its_content_is_never_trusted():
    # The standard SDK loop is ``input += response.output``; this server's own
    # reasoning items carry ``content``, so turn two must accept it.  Only the
    # signed ``encrypted_content`` may reach the prompt: client-edited
    # ``content`` must change nothing.
    engine = ContinuationEngine()
    with _Served(engine, response_store=ResponseStore()) as base:
        with _post(base, {
            "model": "fixture",
            "input": "first",
            "include": ["reasoning.encrypted_content"],
        }) as response:
            first = json.load(response)
        reasoning = first["output"][0]
        assert reasoning["type"] == "reasoning" and "content" in reasoning

        def turn_two(output):
            with _post(base, {
                "model": "fixture",
                "input": [
                    {"role": "user", "content": "first"},
                    *output,
                    {"role": "user", "content": "second"},
                ],
            }) as response:
                assert response.status == 200
            return engine.requests[-1]["messages"]

        echoed = turn_two(first["output"])
        tampered_item = {
            **reasoning,
            "content": [{"type": "reasoning_text", "text": "INJECTED"}],
        }
        tampered = turn_two([tampered_item, *first["output"][1:]])
        unsigned = turn_two(
            [{k: v for k, v in tampered_item.items() if k != "encrypted_content"},
             *first["output"][1:]]
        )
    assert echoed == tampered
    assert echoed[1]["reasoning_content"] == "trusted thought"
    assert "INJECTED" not in json.dumps(tampered + unsigned)
    assert not any("reasoning_content" in message for message in unsigned)


class ToolTurnEngine(ContinuationEngine):
    """First turn: reasoning, commentary text and two parallel calls."""

    def submit(self, request, *, tenant_id="default"):
        self.requests.append(request)
        job = Job(request)
        job.prompt_tokens = 2
        job.completion_tokens = 2
        if len(self.requests) == 1:
            events = [
                {"delta": {"reasoning_content": "plan"}},
                {"delta": {"content": "Checking."}},
                *(
                    {"delta": {"tool_calls": [{
                        "index": index,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }]}}
                    for index, call_id in enumerate(("call_a", "call_b"))
                ),
                {"finish_reason": "tool_calls", "receipt": {}},
            ]
        else:
            events = [
                {"delta": {"content": "done"}},
                {"finish_reason": "stop", "receipt": {}},
            ]
        for event in events:
            job.events.put(event)
        return job


def test_stateless_replay_renders_the_same_history_as_previous_response_id():
    # One model turn (reasoning, text, two parallel calls) must replay as the
    # single assistant message the model produced, whether the client echoes
    # ``response.output`` or the server holds the context, with agent-compat
    # off as well as on.
    tools = [{"type": "function", "name": "f", "parameters": {"type": "object"}}]
    outputs = [
        {"type": "function_call_output", "call_id": call_id, "output": "ok"}
        for call_id in ("call_a", "call_b")
    ]
    engine = ToolTurnEngine()
    with _Served(engine, response_store=ResponseStore()) as base:
        with _post(base, {
            "model": "fixture",
            "input": "go",
            "tools": tools,
            "include": ["reasoning.encrypted_content"],
        }) as response:
            first = json.load(response)
        assert [item["type"] for item in first["output"]] == [
            "reasoning", "message", "function_call", "function_call"
        ]
        with _post(base, {
            "model": "fixture",
            "input": outputs,
            "tools": tools,
            "previous_response_id": first["id"],
        }) as response:
            assert response.status == 200
        held = engine.requests[-1]["messages"]
        with _post(base, {
            "model": "fixture",
            "input": [{"role": "user", "content": "go"}, *first["output"], *outputs],
            "tools": tools,
            "include": ["reasoning.encrypted_content"],
        }) as response:
            assert response.status == 200
        stateless = engine.requests[-1]["messages"]
    assert stateless == held
    assert [message["role"] for message in stateless] == [
        "user", "assistant", "tool", "tool"
    ]
    assert stateless[1]["content"] == "Checking."
    assert stateless[1]["reasoning_content"] == "plan"
    assert [call["id"] for call in stateless[1]["tool_calls"]] == [
        "call_a", "call_b"
    ]
    assert engine.counts["reasoning_signature_rejections"] == 0


def test_responses_render_one_leading_system_message_without_agent_compat():
    # ``instructions`` plus a developer item (directly or held through
    # ``previous_response_id``) used to reach the template as two system
    # messages, which Qwen3.5/3.6 templates reject with a 500.
    engine = ContinuationEngine()
    with _Served(engine, response_store=ResponseStore()) as base:
        with _post(base, {
            "model": "fixture",
            "instructions": "You are terse.",
            "input": [
                {"role": "developer", "content": "Answer in French."},
                {"role": "system", "content": "No emoji."},
                {"role": "user", "content": "hi"},
            ],
        }) as response:
            assert response.status == 200
        direct = engine.requests[-1]["messages"]
        with _post(base, {
            "model": "fixture",
            "input": [
                {"role": "developer", "content": "Answer in French."},
                {"role": "user", "content": "hi"},
            ],
        }) as response:
            first = json.load(response)
        with _post(base, {
            "model": "fixture",
            "instructions": "You are terse.",
            "previous_response_id": first["id"],
            "input": [
                {"role": "developer", "content": "Now answer in German."},
                {"role": "user", "content": "more"},
            ],
        }) as response:
            assert response.status == 200
        held = engine.requests[-1]["messages"]
    assert [message["role"] for message in direct] == ["system", "user"]
    assert direct[0]["content"] == (
        "You are terse.\n\nAnswer in French.\n\nNo emoji."
    )
    assert held[0] == {
        "role": "system", "content": "You are terse.\n\nAnswer in French."
    }
    # A mid-conversation developer item keeps its position as user text.
    assert [message["role"] for message in held[1:]] == [
        "user", "assistant", "user", "user"
    ]
    assert [message["content"] for message in held[-2:]] == [
        "Now answer in German.", "more"
    ]
    assert engine.counts["agent_compat_system_folded"] == 0


def _logprob(token):
    return {
        "logprob": {
            "token": token,
            "logprob": -0.1,
            "bytes": list(token.encode()),
            "top_logprobs": [],
        }
    }


class ScriptedEngine(ContinuationEngine):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def submit(self, request, *, tenant_id="default"):
        self.requests.append(request)
        job = Job(request)
        job.prompt_tokens = 2
        job.completion_tokens = 2
        for event in self.events:
            job.events.put(event)
        return job


_WEATHER_CALL = {"delta": {"tool_calls": [{
    "index": 0,
    "id": "call_1",
    "type": "function",
    "function": {"name": "weather", "arguments": "{}"},
}]}}


@pytest.mark.parametrize("events, expected_types, text_tokens", [
    (
        [
            _logprob("p"), {"delta": {"reasoning_content": "plan"}},
            _logprob("a"), {"delta": {"content": "answer"}},
            _logprob("<eos>"),
            {"finish_reason": "stop", "receipt": {}},
        ],
        ["reasoning", "message"],
        ["a", "<eos>"],
    ),
    (
        [
            _logprob("<tc>"), _WEATHER_CALL, _logprob("<eos>"),
            {"finish_reason": "tool_calls", "receipt": {}},
        ],
        ["function_call"],
        None,
    ),
    (
        [
            _logprob("he"), {"delta": {"content": "he"}},
            _logprob("<hold>"), _logprob("llo"), {"delta": {"content": "llo"}},
            {"finish_reason": "stop", "receipt": {}},
        ],
        ["message"],
        ["he", "<hold>", "llo"],
    ),
])
def test_streamed_responses_items_match_nonstream_with_logprobs(
    events, expected_types, text_tokens
):
    # Each token's logprob precedes its delta, so a logprob must wait for the
    # delta that names its item: opening the message on it streamed
    # [message, reasoning] and left a message item that never closed.
    tools = [{"type": "function", "name": "weather", "parameters": {"type": "object"}}]
    body = {
        "model": "fixture",
        "input": "hi",
        "tools": tools,
        "include": ["message.output_text.logprobs"],
    }
    engine = ScriptedEngine(events)
    with _Served(engine) as base:
        with _post(base, body) as response:
            nonstream = json.load(response)
        with _post(base, {**body, "stream": True}) as response:
            wire = response.read().decode()
    records = [
        json.loads(line.removeprefix("data: "))
        for line in wire.splitlines()
        if line.startswith("data: {")
    ]
    completed = next(
        record["response"] for record in records
        if record["type"] == "response.completed"
    )

    def shape(output):
        return [
            {key: value for key, value in item.items() if key != "id"}
            for item in output
        ]

    assert [item["type"] for item in nonstream["output"]] == expected_types
    assert shape(completed["output"]) == shape(nonstream["output"])
    added = [
        (record["output_index"], record["item"]["type"])
        for record in records if record["type"] == "response.output_item.added"
    ]
    done = [
        (record["output_index"], record["item"]["type"])
        for record in records if record["type"] == "response.output_item.done"
    ]
    assert added == done == list(enumerate(expected_types))
    streamed = [
        value["token"]
        for record in records
        if record["type"] == "response.output_text.delta"
        for value in record.get("logprobs", ())
    ]
    if text_tokens is None:
        assert streamed == []
    else:
        message = next(
            item for item in nonstream["output"] if item["type"] == "message"
        )
        assert [
            value["token"] for value in message["content"][0]["logprobs"]
        ] == text_tokens
        assert streamed == text_tokens


def test_batch_collection_attributes_responses_logprobs_like_the_live_path():
    from mlx2.server import collect_nonstream_job

    def collect(responses):
        job = Job({"messages": [{"role": "user", "content": "hi"}]})
        for event in (
            _logprob("p"), {"delta": {"reasoning_content": "plan"}},
            _logprob("a"), {"delta": {"content": "answer"}},
            {"finish_reason": "stop", "receipt": {}},
        ):
            job.events.put(event)
        choice, _usage, _receipt = collect_nonstream_job(
            job, {"logprobs": True}, chat=True, responses=responses
        )
        return [value["token"] for value in choice["logprobs"]["content"]]

    assert collect(responses=True) == ["a"]
    # Chat Completions keeps every generated token's logprob.
    assert collect(responses=False) == ["p", "a"]


def test_responses_echo_the_callers_named_tool_choice():
    tools = [{"type": "function", "name": "weather", "parameters": {"type": "object"}}]
    sent = {"type": "function", "name": "weather"}
    engine = ScriptedEngine([
        _WEATHER_CALL, {"finish_reason": "tool_calls", "receipt": {}},
    ])
    body = {"model": "fixture", "input": "hi", "tools": tools, "tool_choice": sent}
    with _Served(engine, response_store=ResponseStore()) as base:
        with _post(base, body) as response:
            nonstream = json.load(response)
        with _post(base, {**body, "stream": True}) as response:
            wire = response.read().decode()
    completed = next(
        json.loads(line.removeprefix("data: "))["response"]
        for line in wire.splitlines()
        if line.startswith("data: {") and '"response.completed"' in line
    )
    assert nonstream["tool_choice"] == sent
    assert completed["tool_choice"] == sent
    # The engine still received the translated chat shape.
    assert engine.requests[-1]["tool_choice"] == {
        "type": "function", "function": {"name": "weather"}
    }
