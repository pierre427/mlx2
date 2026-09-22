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
