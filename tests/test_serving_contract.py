import base64
import argparse
import json
import threading
import time
from collections import Counter
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
import numpy as np

from mlx2.server import (
    MAX_REQUEST_BODY_BYTES,
    MIN_REQUEST_BODY_BYTES,
    build_parser,
    default_max_tokens_arg,
    handler_for,
    request_body_limit,
    validate_speech_request,
    validate_request,
)
from mlx2.adapters.base import AudioOutput
from mlx2.openai_compat import enforce_tool_contract, responses_to_chat_request
from mlx2.reasoning_signatures import ReasoningSigner
from mlx2.request_limits import (
    ContextLengthExceeded,
    DEFAULT_OUTPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    resolve_output_limit,
)
from mlx2.serving import Job, Overloaded, ServingEngine, minimum_tokens_processor


def test_server_accepts_coalescing_and_cohort_deadlines():
    args = build_parser().parse_args([
        "--model", "fixture",
        "--coalesce-window-ms", "4",
        "--batch-cohort-timeout-ms", "1500",
    ])
    assert args.coalesce_window_ms == 4
    assert args.batch_cohort_timeout_ms == 1500


def test_default_max_tokens_flag_accepts_exact_bounds_and_rejects_invalid_values():
    parser = build_parser()
    assert (
        parser.parse_args(["--model", "fixture"]).default_max_tokens
        == DEFAULT_OUTPUT_TOKENS
    )
    assert parser.parse_args(
        ["--model", "fixture", "--default-max-tokens", "1"]
    ).default_max_tokens == 1
    assert parser.parse_args(
        ["--model", "fixture", "--default-max-tokens", str(MAX_OUTPUT_TOKENS)]
    ).default_max_tokens == MAX_OUTPUT_TOKENS
    for value in ("0", "-1", str(MAX_OUTPUT_TOKENS + 1), "true"):
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["--model", "fixture", "--default-max-tokens", value]
            )
    for value in (True, False, 1.0):
        with pytest.raises(argparse.ArgumentTypeError):
            default_max_tokens_arg(value)


@pytest.mark.parametrize(
    "value", [True, False, 0, -1, MAX_OUTPUT_TOKENS + 1, 1.0, "512"]
)
def test_serving_engine_rejects_invalid_default_max_tokens(value):
    with pytest.raises(ValueError, match="default_max_tokens"):
        ServingEngine("unused", default_max_tokens=value)


def test_speech_contract_is_bounded_and_format_explicit():
    assert validate_speech_request(
        {"input": "hello", "voice": "default", "response_format": "wav", "speed": 1.25}
    ) == {
        "model": None,
        "input": "hello",
        "voice": "default",
        "instructions": None,
        "response_format": "wav",
        "speed": 1.25,
        "stream_format": "audio",
    }
    assert validate_speech_request(
        {"input": "hello", "voice": {"id": "voice_fixture"}, "instructions": "calm"}
    )["voice"] == {"id": "voice_fixture"}
    with pytest.raises(ValueError, match="response_format"):
        validate_speech_request({"input": "hello", "response_format": "raw"})
    with pytest.raises(ValueError, match="4096"):
        validate_speech_request({"input": "x" * 4097})


@pytest.mark.parametrize(
    "extra",
    [
        {"temperature": float("nan")},
        {"max_tokens": -1},
        {"max_tokens": True},
        {"top_k": -1},
        {"n": 9},
        {"seed": -1},
        {"response_format": {"type": "yaml"}},
        {"frequency_penalty": 3},
        {"max_completion_tokens": 0},
        {"max_tokens": 63, "min_tokens": 64},
        {"min_tokens": -1},
        {"batch_cohort": {"id": "x"}},
        {"batch_cohort": {"id": "", "size": 2}},
        {"batch_cohort": {"id": "x", "size": True}},
        {"batch_cohort": {"id": "x", "size": 2}, "n": 2},
    ],
)
def test_reject_invalid_or_unqualified_request(extra):
    with pytest.raises(ValueError):
        validate_request({"messages": [{"role": "user", "content": "hello"}], **extra})


def test_minimum_tokens_processor_masks_eos_until_exact_boundary():
    processor = minimum_tokens_processor(np, {1, 4}, prompt_tokens=3, minimum=2)
    logits = np.arange(6, dtype=np.float32)[None, :]
    before_boundary = processor(np.array([10, 11, 12, 13]), logits)
    assert np.isneginf(before_boundary[0, 1])
    assert np.isneginf(before_boundary[0, 4])
    assert before_boundary[0, 5] == logits[0, 5]
    at_boundary = processor(np.array([10, 11, 12, 13, 14]), logits)
    np.testing.assert_array_equal(at_boundary, logits)


def test_minimum_tokens_is_preserved_as_validated_request_control():
    request = validate_request({
        "messages": [{"role": "user", "content": "hello"}],
        "min_tokens": 64,
        "max_tokens": 64,
    })
    assert request["min_tokens"] == request["max_tokens"] == 64


def test_output_token_limits_accept_boundaries_and_reject_above_ceiling():
    base = {"messages": [{"role": "user", "content": "hello"}]}
    assert validate_request({**base, "max_tokens": 1})["max_tokens"] == 1
    assert (
        validate_request({**base, "max_tokens": MAX_OUTPUT_TOKENS})["max_tokens"]
        == MAX_OUTPUT_TOKENS
    )
    assert (
        validate_request({**base, "max_completion_tokens": MAX_OUTPUT_TOKENS})[
            "max_tokens"
        ]
        == MAX_OUTPUT_TOKENS
    )
    assert (
        validate_request(
            {**base, "max_tokens": MAX_OUTPUT_TOKENS, "min_tokens": MAX_OUTPUT_TOKENS}
        )["min_tokens"]
        == MAX_OUTPUT_TOKENS
    )
    with pytest.raises(ValueError, match="max_tokens"):
        validate_request({**base, "max_tokens": MAX_OUTPUT_TOKENS + 1})
    with pytest.raises(ValueError, match="max_tokens"):
        validate_request({**base, "max_completion_tokens": MAX_OUTPUT_TOKENS + 1})
    with pytest.raises(ValueError, match="min_tokens"):
        validate_request({**base, "min_tokens": MAX_OUTPUT_TOKENS + 1})


def test_hermes_output_alias_uses_the_same_two_million_ceiling():
    base = {"messages": [{"role": "user", "content": "hello"}]}
    assert validate_request(
        {**base, "options": {"num_predict": MAX_OUTPUT_TOKENS}}
    )["max_tokens"] == MAX_OUTPUT_TOKENS
    with pytest.raises(ValueError, match="max_tokens"):
        validate_request(
            {**base, "options": {"num_predict": MAX_OUTPUT_TOKENS + 1}}
        )


def test_prompt_aware_default_clamps_at_context_edge_but_explicit_overflow_fails():
    assert resolve_output_limit(
        {}, prompt_tokens=2_048, effective_context=262_144
    ) == (65_536, True)
    assert resolve_output_limit(
        {}, prompt_tokens=250_000, effective_context=262_144
    ) == (12_144, True)
    assert resolve_output_limit(
        {},
        prompt_tokens=250_000,
        effective_context=262_144,
        default_max_tokens=512,
    ) == (512, True)
    with pytest.raises(ContextLengthExceeded) as failure:
        resolve_output_limit(
            {"max_tokens": 12_145},
            prompt_tokens=250_000,
            effective_context=262_144,
        )
    assert failure.value.code == "context_length_exceeded"


def test_every_openai_and_hermes_surface_preserves_output_limit_omission():
    chat = validate_request({"messages": [{"role": "user", "content": "hi"}]})
    completion = validate_request({"prompt": "hi"}, chat=False)
    hermes = validate_request(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"temperature": 0},
        }
    )
    responses, _ = responses_to_chat_request({"input": "hi"})
    for request in (chat, completion, hermes, validate_request(responses)):
        assert "max_tokens" not in request


@pytest.mark.parametrize(
    ("max_tokens", "thinking_budget"),
    [(512, 1_024), (512, 512), (8_192, 8_192)],
)
@pytest.mark.parametrize("chat", [True, False])
def test_openai_thinking_budget_is_independent_of_output_limit(
    chat, max_tokens, thinking_budget
):
    input_field = (
        {"messages": [{"role": "user", "content": "hello"}]}
        if chat
        else {"prompt": "hello"}
    )
    accepted = validate_request(
        {
            **input_field,
            "max_tokens": max_tokens,
            "thinking_budget": thinking_budget,
        },
        chat=chat,
    )
    assert accepted["thinking_budget"] == thinking_budget


def test_thinking_budget_range_and_prompt_edge_are_independent_of_output_limit():
    base = {"messages": [{"role": "user", "content": "hello"}]}
    accepted = validate_request(
        {**base, "max_tokens": 512, "thinking_budget": MAX_OUTPUT_TOKENS}
    )
    assert accepted["thinking_budget"] == MAX_OUTPUT_TOKENS
    with pytest.raises(ValueError, match="thinking_budget"):
        validate_request(
            {**base, "thinking_budget": MAX_OUTPUT_TOKENS + 1}
        )
    assert resolve_output_limit(
        {"thinking_budget": 2_048},
        prompt_tokens=262_144 - 600,
        effective_context=262_144,
    ) == (600, True)


def test_thinking_budget_is_independent_across_compatibility_surfaces():
    base = {"messages": [{"role": "user", "content": "hello"}]}
    responses, _ = responses_to_chat_request(
        {"input": "hello", "max_output_tokens": 512, "thinking_budget": 1_024}
    )
    requests = (
        {**base, "max_tokens": 512, "thinking_budget": 1_024},
        {"prompt": "hello", "max_tokens": 512, "thinking_budget": 1_024},
        responses,
        {
            **base,
            "options": {"num_predict": 512, "thinking_budget": 1_024},
        },
    )
    assert validate_request(requests[0])["thinking_budget"] == 1_024
    assert validate_request(requests[1], chat=False)["thinking_budget"] == 1_024
    assert validate_request(requests[2])["thinking_budget"] == 1_024
    assert validate_request(requests[3])["thinking_budget"] == 1_024


def test_batch_cohort_is_preserved_as_validated_request_control():
    request = validate_request({
        "messages": [{"role": "user", "content": "hello"}],
        "batch_cohort": {"id": "cell-123", "size": 20},
    })
    assert request["batch_cohort"] == {"id": "cell-123", "size": 20}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]


def test_required_named_and_strict_tool_controls_are_validated():
    base = {"messages": [{"role": "user", "content": "weather?"}], "tools": TOOLS}
    assert validate_request({**base, "tool_choice": "required"})["tool_choice"] == "required"
    named = {"type": "function", "function": {"name": "weather"}}
    assert validate_request({**base, "tool_choice": named})["tool_choice"] == named
    with pytest.raises(ValueError, match="declared function"):
        validate_request(
            {**base, "tool_choice": {"type": "function", "function": {"name": "clock"}}}
        )
    with pytest.raises(ValueError, match="requires function tools"):
        validate_request(
            {"messages": base["messages"], "parallel_tool_calls": False}
        )
    with pytest.raises(ValueError, match="streaming strict"):
        validate_request({**base, "stream": True})


def test_strict_tool_result_and_parallel_choice_fail_closed():
    body = validate_request(
        {
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": TOOLS,
            "tool_choice": "required",
            "parallel_tool_calls": False,
        }
    )
    valid = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "weather", "arguments": '{"city":"Toronto"}'},
    }
    enforce_tool_contract(body, [valid])
    with pytest.raises(ValueError, match="required tool call"):
        enforce_tool_contract(body, [])
    invalid = {
        **valid,
        "function": {"name": "weather", "arguments": '{"city":7}'},
    }
    with pytest.raises(ValueError, match="strict tool type"):
        enforce_tool_contract(body, [invalid])
    with pytest.raises(ValueError, match="parallel calls"):
        enforce_tool_contract(body, [valid, valid])


def test_text_only_responses_request_maps_to_chat_without_silent_drops():
    request, options = responses_to_chat_request(
        {
            "model": "fixture",
            "instructions": "Be concise.",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
            "max_output_tokens": 17,
            "metadata": {"case": "cpu"},
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                }
            },
        }
    )
    assert request["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "hello"},
    ]
    assert request["max_tokens"] == 17
    assert request["response_format"]["type"] == "json_schema"
    assert options == {
        "metadata": {"case": "cpu"},
        "store": False,
        "previous_response_id": None,
        "context_messages": [{"role": "user", "content": "hello"}],
        "tool_executors": {},
    }
    multimodal, _ = responses_to_chat_request(
        {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "x"}],
                }
            ]
        }
    )
    assert multimodal["messages"][0]["content"] == [
        {"type": "input_image", "image_url": "x"}
    ]


def test_responses_output_limit_maps_ceiling_and_preserves_omission():
    request, _ = responses_to_chat_request(
        {"input": "hello", "max_output_tokens": MAX_OUTPUT_TOKENS}
    )
    assert validate_request(request)["max_tokens"] == MAX_OUTPUT_TOKENS
    omitted, _ = responses_to_chat_request({"input": "hello"})
    assert "max_tokens" not in validate_request(omitted)
    with pytest.raises(ValueError, match="max_tokens"):
        validate_request(
            responses_to_chat_request(
                {"input": "hello", "max_output_tokens": MAX_OUTPUT_TOKENS + 1}
            )[0]
        )


def test_responses_function_tools_and_named_choice_map_to_chat_contract():
    request, _ = responses_to_chat_request(
        {
            "input": "weather?",
            "tools": [
                {
                    "type": "function",
                    "name": "weather",
                    "parameters": TOOLS[0]["function"]["parameters"],
                    "strict": True,
                }
            ],
            "tool_choice": {"type": "function", "name": "weather"},
            "parallel_tool_calls": False,
        }
    )
    validated = validate_request(request)
    assert validated["tool_choice"] == {
        "type": "function",
        "function": {"name": "weather"},
    }


def test_skip_writing_prefix_cache_requires_boolean_and_is_preserved():
    request = validate_request({
        "messages": [{"role": "user", "content": "hello"}],
        "skip_writing_prefix_cache": True,
    })
    assert request["skip_writing_prefix_cache"] is True
    with pytest.raises(ValueError, match="skip_writing_prefix_cache must be boolean"):
        validate_request({
            "messages": [{"role": "user", "content": "hello"}],
            "skip_writing_prefix_cache": 1,
        })


class FakeEngine:
    model_path = "fixture"
    max_context = 16384

    def __init__(self):
        self.job = None
        self.overloaded = False
        self.lock = threading.Lock()
        self.counts = Counter()
        self.max_request_bytes = MIN_REQUEST_BODY_BYTES
        self.stop_sequence = None

    def status(self):
        return {
            "healthy": True,
            "error": None,
            "model": "fixture",
            "http": {"max_request_bytes": self.max_request_bytes},
        }

    def batching_status(self):
        return {"schema": "mlx2.batch-runtime.v1", "gauges": {"queue_depth": 0}}

    def submit(self, request, *, tenant_id="default"):
        if self.overloaded:
            raise Overloaded("full")
        self.job = Job(request)
        self.job.tenant_id = tenant_id
        self.job.prompt_tokens, self.job.completion_tokens = 5, 2
        if request.get("logprobs") or request.get("top_logprobs"):
            for token in (5, 6):
                self.job.events.put({"logprob": {"id": token, "token": str(token),
                                                "logprob": -0.5,
                                                "top_logprobs": [{"id": token, "token": str(token), "logprob": -0.5}]}})
        self.job.events.put({"text": "hello"})
        receipt = {"cache": "apcv2"}
        if self.stop_sequence is not None:
            receipt["stop_sequence"] = self.stop_sequence
        self.job.events.put({"finish_reason": "stop", "receipt": receipt})
        return self.job


@pytest.fixture
def http_engine():
    engine = FakeEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield engine, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def post(base, **extra):
    body = {
        "model": "fixture",
        "messages": [{"role": "user", "content": "hi"}],
        **extra,
    }
    return urlopen(
        Request(
            base + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
    )


def post_response(base, **extra):
    body = {"model": "fixture", "input": "hi", **extra}
    return urlopen(
        Request(
            base + "/v1/responses",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
    )


def test_http_accepts_advertised_lora_model_and_preserves_identity():
    class LoRAEngine(FakeEngine):
        def status(self):
            return {**super().status(), "multi_lora": {"registered": ["sql"]}}

    engine = LoRAEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(base + "/v1/models") as response:
            assert "sql" in {entry["id"] for entry in json.load(response)["data"]}
        with post(base, model="sql") as response:
            assert json.load(response)["model"] == "sql"
        assert engine.job.request["model"] == "sql"
        with _json_post(base, "/v1/messages", {
            "model": "sql",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
        }) as response:
            assert json.load(response)["model"] == "sql"
        with post_response(base, model="sql") as response:
            assert json.load(response)["model"] == "sql"
        row = {
            "custom_id": "lora-row", "method": "POST", "url": "/v1/responses",
            "body": {"model": "sql", "input": "hi", "store": False},
        }
        with upload_file(base, (json.dumps(row) + "\n").encode()) as response:
            file_id = json.load(response)["id"]
        with _json_post(base, "/v1/batches", {
            "input_file_id": file_id,
            "endpoint": "/v1/responses",
            "completion_window": "24h",
        }) as response:
            batch = json.load(response)
        for _ in range(100):
            with urlopen(base + "/v1/batches/" + batch["id"]) as response:
                batch = json.load(response)
            if batch["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.01)
        assert batch["request_counts"] == {"total": 1, "completed": 1, "failed": 0}
        with urlopen(base + f"/v1/files/{batch['output_file_id']}/content") as response:
            assert json.loads(response.read())["response"]["body"]["model"] == "sql"
        with pytest.raises(HTTPError) as error:
            post(base, model="unknown")
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_keepalive_counts_each_request():
    from http.client import HTTPConnection
    from mlx2.batch_metrics import HttpRuntimeMetrics

    engine = FakeEngine()
    engine.http_metrics = HttpRuntimeMetrics()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        for path in ("/health", "/v1/models", "/health"):
            client.request("GET", path)
            response = client.getresponse()
            assert response.status == 200
            response.read()
        assert sum(engine.http_metrics.prometheus_snapshot()["requests"].values()) == 3
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join()


def upload_file(base, content, *, filename="input.jsonl", purpose="batch", content_type="application/jsonl"):
    boundary = "mlx2-test-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="purpose"\r\n\r\n'
        f"{purpose}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    return urlopen(
        Request(
            base + "/v1/files",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
    )


def test_nonstreaming_receipt_and_usage(http_engine):
    engine, base = http_engine
    engine.stop_sequence = "STOP"
    with post(base) as response:
        data = json.load(response)
    assert data["choices"][0]["message"]["content"] == "hello"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["mlx2"] == {"cache": "apcv2", "stop_sequence": "STOP"}
    assert data["usage"]["total_tokens"] == 7


def test_nonstreaming_failure_preserves_qualification_receipt():
    engine = FakeEngine()
    receipt = {
        "structured_output_failure": {
            "schema": "mlx2.structured-output-dead-end.v1",
            "generated_tokens": 1,
        }
    }

    def submit(request, *, tenant_id="default"):
        job = Job(request)
        job.tenant_id = tenant_id
        job.events.put({
            "error": "structured output failed closed",
            "status": 502,
            "mlx2": receipt,
        })
        return job

    engine.submit = submit
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(HTTPError) as failure:
            post(f"http://127.0.0.1:{server.server_port}")
        assert failure.value.code == 502
        assert json.load(failure.value)["mlx2"] == receipt
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_audio_speech_returns_adapter_owned_binary_contract():
    class AudioEngine(FakeEngine):
        def synthesize_speech(
            self, text, *, voice, instructions, response_format, speed
        ):
            assert (text, voice, instructions, response_format, speed) == (
                "hello", "alloy", None, "wav", 1.0
            )
            return AudioOutput(b"RIFFfixture", "audio/wav", sample_rate=24_000, channels=1)

    engine = AudioEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/audio/speech",
            data=json.dumps(
                {"model": "fixture", "input": "hello", "voice": "alloy", "response_format": "wav"}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request) as response:
            assert response.headers["Content-Type"] == "audio/wav"
            assert response.read() == b"RIFFfixture"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_nonstreaming_responses_api_text_shape_and_metadata(http_engine):
    engine, base = http_engine
    with post_response(base, instructions="Be brief.", metadata={"case": "cpu"}) as response:
        data = json.load(response)
    assert data["object"] == "response" and data["status"] == "completed"
    assert data["output"][0]["type"] == "message"
    assert data["output"][0]["content"] == [
        {"type": "output_text", "text": "hello", "annotations": []}
    ]
    assert data["usage"] == {
        "input_tokens": 5,
        "output_tokens": 2,
        "total_tokens": 7,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }
    assert data["metadata"] == {"case": "cpu"}
    assert engine.job.request["messages"][0] == {
        "role": "system",
        "content": "Be brief.",
    }


def test_streaming_responses_api_uses_typed_events(http_engine):
    _, base = http_engine
    with post_response(base, stream=True) as response:
        wire = response.read().decode()
    records = [
        json.loads(line[6:])
        for line in wire.splitlines()
        if line.startswith("data: ")
    ]
    assert [record["type"] for record in records] == [
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert records[3]["delta"] == "hello"
    assert [record["sequence_number"] for record in records] == list(
        range(len(records))
    )
    assert records[-1]["response"]["usage"]["total_tokens"] == 7
    assert "[DONE]" not in wire


class ReasoningEngine(FakeEngine):
    def submit(self, request, *, tenant_id="default"):
        self.job = Job(request)
        self.job.tenant_id = tenant_id
        self.job.prompt_tokens = 5
        self.job.completion_tokens = 4
        self.job.reasoning_tokens = 2
        self.job.events.put({"delta": {"reasoning_content": "plan"}})
        self.job.events.put({"delta": {"content": "answer"}})
        self.job.events.put(
            {"finish_reason": "stop", "receipt": {"cache": "apcv2"}}
        )
        return self.job


@pytest.fixture
def reasoning_http_engine():
    engine = ReasoningEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield engine, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def test_nonstreaming_responses_api_preserves_reasoning(reasoning_http_engine):
    _, base = reasoning_http_engine
    with post_response(base) as response:
        data = json.load(response)
    assert [item["type"] for item in data["output"]] == ["reasoning", "message"]
    reasoning = data["output"][0]
    assert reasoning["summary"] == [{"type": "summary_text", "text": "plan"}]
    assert reasoning["content"] == [{"type": "reasoning_text", "text": "plan"}]
    assert data["output"][1]["content"][0]["text"] == "answer"
    assert data["usage"]["output_tokens_details"] == {"reasoning_tokens": 2}


def test_streaming_responses_api_preserves_reasoning(reasoning_http_engine):
    _, base = reasoning_http_engine
    with post_response(base, stream=True) as response:
        wire = response.read().decode()
    records = [
        json.loads(line[6:])
        for line in wire.splitlines()
        if line.startswith("data: ")
    ]
    assert [record["type"] for record in records] == [
        "response.created",
        "response.output_item.added",
        "response.reasoning_summary_text.delta",
        "response.reasoning_text.delta",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.reasoning_summary_text.done",
        "response.reasoning_text.done",
        "response.output_item.done",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert records[2]["delta"] == records[3]["delta"] == "plan"
    assert records[7]["text"] == records[8]["text"] == "plan"
    assert [item["type"] for item in records[-1]["response"]["output"]] == [
        "reasoning",
        "message",
    ]


def test_context_length_error_has_machine_readable_code():
    class ContextErrorEngine(FakeEngine):
        def submit(self, request, *, tenant_id="default"):
            self.job = Job(request)
            error = ContextLengthExceeded(
                prompt_tokens=15,
                max_output_tokens=4,
                context_limit=16,
            )
            self.job.events.put(
                {"error": str(error), "status": 400, "code": error.code}
            )
            return self.job

    engine = ContextErrorEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with pytest.raises(HTTPError) as raised:
            post_response(base)
        assert raised.value.code == 400
        payload = json.load(raised.value)
        assert payload["error"]["type"] == "invalid_request_error"
        assert payload["error"]["code"] == "context_length_exceeded"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_responses_named_strict_function_call_is_rendered_as_output_item():
    class ToolEngine(FakeEngine):
        def submit(self, request, *, tenant_id="default"):
            self.job = Job(request)
            self.job.tenant_id = tenant_id
            self.job.prompt_tokens, self.job.completion_tokens = 8, 3
            self.job.events.put(
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_weather",
                                "type": "function",
                                "function": {
                                    "name": "weather",
                                    "arguments": '{"city":"Toronto"}',
                                },
                            }
                        ]
                    }
                }
            )
            self.job.events.put(
                {"finish_reason": "tool_calls", "receipt": {"cache": "apcv2"}}
            )
            return self.job

    engine = ToolEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with post_response(
            base,
            input="weather?",
            tools=[
                {
                    "type": "function",
                    "name": "weather",
                    "parameters": TOOLS[0]["function"]["parameters"],
                    "strict": True,
                }
            ],
            tool_choice={"type": "function", "name": "weather"},
            parallel_tool_calls=False,
        ) as response:
            data = json.load(response)
        assert data["output"] == [
            {
                "id": "call_weather",
                "type": "function_call",
                "status": "completed",
                "call_id": "call_weather",
                "name": "weather",
                "arguments": '{"city":"Toronto"}',
            }
        ]
        assert data["status"] == "completed"
        with post_response(
            base,
            input="weather?",
            stream=True,
            tools=[
                {
                    "type": "function",
                    "name": "weather",
                    "parameters": TOOLS[0]["function"]["parameters"],
                    "strict": True,
                }
            ],
            tool_choice={"type": "function", "name": "weather"},
            parallel_tool_calls=False,
            store=False,
        ) as response:
            wire = response.read().decode()
        records = [
            json.loads(line[6:])
            for line in wire.splitlines()
            if line.startswith("data: ")
        ]
        assert [record["type"] for record in records] == [
            "response.created",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ]
        assert records[2]["delta"] == '{"city":"Toronto"}'
        assert records[-1]["response"]["output"][0]["type"] == "function_call"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_chat_strict_tool_stream_is_validated_before_first_chunk():
    class ToolEngine(FakeEngine):
        def submit(self, request, *, tenant_id="default"):
            self.job = Job(request)
            self.job.events.put(
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_weather",
                                "type": "function",
                                "function": {
                                    "name": "weather",
                                    "arguments": '{"city":"Toronto"}',
                                },
                            }
                        ]
                    }
                }
            )
            self.job.events.put(
                {"finish_reason": "tool_calls", "receipt": {"cache": "apcv2"}}
            )
            return self.job

    engine = ToolEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with post(
            base,
            stream=True,
            tools=TOOLS,
            tool_choice="required",
            parallel_tool_calls=False,
        ) as response:
            records = [
                json.loads(line[6:])
                for line in response.read().decode().splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
        assert records[0]["choices"][0]["delta"]["tool_calls"][0]["function"] == {
            "name": "weather",
            "arguments": '{"city":"Toronto"}',
        }
        assert records[-1]["choices"][0]["finish_reason"] == "tool_calls"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_responses_allowlisted_mcp_backend_executes_and_resumes_model():
    class Backend:
        def prepare(self, tools):
            assert tools[0]["type"] == "mcp"
            return TOOLS, {"weather": "binding"}

        def execute(self, binding, arguments):
            assert binding == "binding"
            assert arguments == {"city": "Toronto"}
            return {"temperature": 21}

    class ToolEngine(FakeEngine):
        def __init__(self):
            super().__init__()
            self.requests = []

        def submit(self, request, *, tenant_id="default"):
            self.requests.append(request)
            self.job = Job(request)
            if len(self.requests) == 1:
                self.job.events.put(
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_weather",
                                    "type": "function",
                                    "function": {
                                        "name": "weather",
                                        "arguments": '{"city":"Toronto"}',
                                    },
                                }
                            ]
                        }
                    }
                )
                reason = "tool_calls"
            else:
                self.job.events.put({"text": "It is 21 C."})
                reason = "stop"
            self.job.events.put(
                {"finish_reason": reason, "receipt": {"cache": "apcv2"}}
            )
            return self.job

    engine = ToolEngine()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), handler_for(engine, tool_backend=Backend())
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with post_response(
            base,
            input="weather?",
            tools=[
                {
                    "type": "mcp",
                    "server_label": "weather",
                    "server_url": "https://example.invalid/mcp",
                    "require_approval": "never",
                }
            ],
        ) as response:
            payload = json.load(response)
        assert payload["output"][0]["content"][0]["text"] == "It is 21 C."
        assert engine.requests[1]["messages"][-1] == {
            "role": "tool",
            "tool_call_id": "call_weather",
            "content": '{"temperature":21}',
        }
        assert payload["mlx2"]["hosted_tools"]["rounds"] == 1
        with post_response(
            base,
            input="What did the tool observe?",
            previous_response_id=payload["id"],
        ) as response:
            assert response.status == 200
        replay = engine.requests[2]["messages"]
        assert [message["role"] for message in replay] == [
            "user", "assistant", "tool", "assistant", "user"
        ]
        assert replay[2]["tool_call_id"] == "call_weather"
        assert replay[2]["content"] == '{"temperature":21}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_hosted_tool_stream_gives_each_output_item_one_index():
    class Backend:
        def prepare(self, tools):
            return TOOLS, {"weather": "binding"}

        def execute(self, binding, arguments):
            return {"temperature": 21}

    class ToolEngine(FakeEngine):
        rounds = 0

        def submit(self, request, *, tenant_id="default"):
            self.rounds += 1
            self.job = Job(request)
            if self.rounds == 1:
                self.job.events.put({"delta": {"tool_calls": [{
                    "index": 0,
                    "id": "call_weather",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city":"T"}'},
                }]}})
                reason = "tool_calls"
            else:
                self.job.events.put({"delta": {"reasoning_content": "thinking"}})
                self.job.events.put({"delta": {"content": "It is 21 C."}})
                reason = "stop"
            self.job.events.put({"finish_reason": reason, "receipt": {}})
            return self.job

    engine = ToolEngine()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), handler_for(engine, tool_backend=Backend())
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with post_response(
            base,
            input="weather?",
            stream=True,
            tools=[{
                "type": "mcp",
                "server_label": "weather",
                "server_url": "https://example.invalid/mcp",
                "require_approval": "never",
            }],
        ) as response:
            events = [
                json.loads(line[len("data: "):])
                for line in response.read().decode().splitlines()
                if line.startswith("data: {")
            ]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    added = {
        event["item"]["id"]: event["output_index"]
        for event in events
        if event["type"] == "response.output_item.added"
    }
    done = {
        event["item"]["id"]: event["output_index"]
        for event in events
        if event["type"] == "response.output_item.done"
    }
    assert added == done
    assert sorted(added.values()) == list(range(len(added)))
    for event in events:
        if "item_id" in event:
            assert event["output_index"] == added[event["item_id"]], event["type"]
    completed = next(e for e in events if e["type"] == "response.completed")
    assert [item["id"] for item in completed["response"]["output"]] == sorted(
        added, key=added.get
    )
    text = "".join(
        e["delta"] for e in events if e["type"] == "response.output_text.delta"
    )
    assert text == "It is 21 C."


def test_required_tool_model_violation_is_server_error(http_engine):
    _, base = http_engine
    with pytest.raises(HTTPError) as error:
        post(
            base,
            tools=TOOLS,
            tool_choice="required",
        )
    assert error.value.code == 502


def test_responses_unsupported_multimodal_fields_fail_closed(http_engine):
    _, base = http_engine
    for extra in (
        {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "https://invalid"}],
                }
            ]
        },
    ):
        with pytest.raises(HTTPError) as error:
            post_response(base, **extra)
        assert error.value.code == 501


def test_responses_hosted_tools_fail_as_unavailable_not_silently_ignored(http_engine):
    _, base = http_engine
    with pytest.raises(HTTPError) as error:
        post_response(
            base,
            tools=[{"type": "web_search"}],
        )
    assert error.value.code == 501


def test_responses_store_retrieve_continue_and_delete_are_tenant_scoped(http_engine):
    engine, base = http_engine
    with post_response(base, input="first") as response:
        first = json.load(response)
    assert first["store"] is True
    store = engine.api_resources["responses"]
    _size, stored = store._entries[("default", first["id"])]
    assert set(stored) == {"payload", "context_messages"}
    with urlopen(base + "/v1/responses/" + first["id"]) as response:
        assert json.load(response)["id"] == first["id"]
    with urlopen(base + f"/v1/responses/{first['id']}/input_items") as response:
        input_items = json.load(response)
    assert input_items["object"] == "list"
    assert input_items["has_more"] is False
    assert input_items["first_id"] == input_items["last_id"]
    assert input_items["data"] == [
        {
            "id": input_items["first_id"],
            "type": "message",
            "status": "completed",
            "role": "user",
            "content": [{"type": "input_text", "text": "first"}],
        }
    ]

    with post_response(
        base,
        input="second",
        instructions="new policy",
        previous_response_id=first["id"],
        store=False,
    ) as response:
        second = json.load(response)
    assert second["previous_response_id"] == first["id"]
    assert engine.job.request["messages"] == [
        {"role": "system", "content": "new policy"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "second"},
    ]
    request = Request(base + "/v1/responses/" + first["id"], method="DELETE")
    with urlopen(request) as response:
        assert json.load(response)["deleted"] is True
    with pytest.raises(HTTPError) as error:
        urlopen(base + "/v1/responses/" + first["id"])
    assert error.value.code == 404


def test_responses_continuation_with_null_instructions_keeps_turn_order(
    http_engine,
):
    engine, base = http_engine
    with post_response(base, input="first") as response:
        first = json.load(response)
    with post_response(
        base, input="second", instructions=None, previous_response_id=first["id"]
    ):
        pass
    assert engine.job.request["messages"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "second"},
    ]


def test_responses_input_items_derive_old_records_and_fail_closed(http_engine):
    engine, base = http_engine
    store = engine.api_resources["responses"]
    payload = {"id": "resp_old", "object": "response", "model": "fixture"}
    store.put(
        "default",
        payload,
        [
            {"role": "user", "content": "old input"},
            {"role": "assistant", "content": "old output"},
        ],
    )
    with urlopen(base + "/v1/responses/resp_old/input_items") as response:
        page = json.load(response)
    assert page["data"] == [
        {
            "id": page["first_id"],
            "type": "message",
            "status": "completed",
            "role": "user",
            "content": [{"type": "input_text", "text": "old input"}],
        }
    ]

    store.put("default", {"id": "resp_empty", "object": "response"}, [])
    with pytest.raises(HTTPError) as error:
        urlopen(base + "/v1/responses/resp_empty/input_items")
    assert error.value.code == 409
    assert "cannot be derived" in json.load(error.value)["error"]["message"]


def test_responses_input_items_do_not_persist_raw_input_and_validate_include(
    http_engine,
):
    engine, base = http_engine
    raw_file_data = base64.b64encode(b"private file bytes").decode()
    rejected_reasoning = "rejected-encrypted-content"
    with post_response(
        base,
        input=[
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "untrusted"}],
                "encrypted_content": rejected_reasoning,
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": "private.txt",
                        "file_data": raw_file_data,
                    }
                ],
            },
        ],
    ) as response:
        payload = json.load(response)
    store = engine.api_resources["responses"]
    _size, record = store._entries[("default", payload["id"])]
    serialized = json.dumps(record)
    assert set(record) == {"payload", "context_messages"}
    assert raw_file_data not in serialized
    assert rejected_reasoning not in serialized
    with urlopen(base + f"/v1/responses/{payload['id']}/input_items") as response:
        page = json.load(response)
    assert "private file bytes" in page["data"][0]["content"][0]["text"]
    assert raw_file_data not in json.dumps(page)
    assert rejected_reasoning not in json.dumps(page)

    with pytest.raises(HTTPError) as error:
        urlopen(base + f"/v1/responses/{payload['id']}/input_items?include=unknown")
    assert error.value.code == 400


def test_responses_input_items_reasoning_include_is_derived_not_stored(http_engine):
    engine, base = http_engine
    engine.reasoning_signer = ReasoningSigner(b"input-items-test")
    store = engine.api_resources["responses"]
    store.put(
        "tenant-a",
        {"id": "resp_reasoning", "object": "response", "model": "fixture"},
        [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "trusted prior thought",
            },
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": "answer"},
        ],
    )
    request = Request(
        base
        + "/v1/responses/resp_reasoning/input_items"
        + "?include[]=reasoning.encrypted_content",
        headers={"X-Tenant-ID": "tenant-a"},
    )
    with urlopen(request) as response:
        page = json.load(response)
    reasoning = next(item for item in page["data"] if item["type"] == "reasoning")
    assert engine.reasoning_signer.verify_responses(
        reasoning["encrypted_content"],
        model="fixture",
        tenant="tenant-a",
        expected_text="trusted prior thought",
    ) == "trusted prior thought"
    _size, record = store._entries[("tenant-a", "resp_reasoning")]
    assert "encrypted_content" not in json.dumps(record)


def test_responses_input_items_image_url_requires_include(http_engine):
    engine, base = http_engine
    store = engine.api_resources["responses"]
    store.put(
        "default",
        {"id": "resp_image", "object": "response", "model": "fixture"},
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": "https://example.test/private.png",
                        "detail": "low",
                    }
                ],
            },
            {"role": "assistant", "content": "answer"},
        ],
    )
    endpoint = base + "/v1/responses/resp_image/input_items"
    with urlopen(endpoint) as response:
        page = json.load(response)
    assert page["data"][0]["content"] == [
        {
            "type": "input_text",
            "text": "[input image omitted from stored item view]",
        }
    ]

    with urlopen(endpoint + "?include=message.input_image.image_url") as response:
        page = json.load(response)
    assert page["data"][0]["content"] == [
        {
            "type": "input_image",
            "image_url": "https://example.test/private.png",
            "detail": "low",
        }
    ]


def test_responses_storage_keeps_main_eviction_capacity():
    from mlx2.api_resources import ResponseStore

    store = ResponseStore(max_entries=128, max_bytes=10_000)
    engine = FakeEngine()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), handler_for(engine, response_store=store)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        response_ids = []
        for suffix in ("one", "two"):
            with post_response(base, input="x" * 4_000 + suffix) as response:
                response_ids.append(json.load(response)["id"])
        assert store.context("default", response_ids[0])[0]["content"].endswith(
            "one"
        )
        assert store.status()["entries"] == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_missing_or_cross_tenant_response_continuation_is_not_found(http_engine):
    _, base = http_engine
    with post_response(base) as response:
        first = json.load(response)
    request = Request(
        base + "/v1/responses",
        data=json.dumps(
            {"model": "fixture", "input": "next", "previous_response_id": first["id"]}
        ).encode(),
        headers={"Content-Type": "application/json", "X-Tenant-ID": "other"},
    )
    with pytest.raises(HTTPError) as error:
        urlopen(request)
    assert error.value.code == 404


def test_files_api_and_responses_utf8_file_input(http_engine):
    engine, base = http_engine
    with upload_file(
        base,
        b"alpha beta",
        filename="notes.txt",
        purpose="user_data",
        content_type="text/plain",
    ) as response:
        uploaded = json.load(response)
    assert uploaded["object"] == "file" and uploaded["bytes"] == 10
    with urlopen(base + f"/v1/files/{uploaded['id']}/content") as response:
        assert response.read() == b"alpha beta"
    with post_response(
        base,
        store=False,
        input=[
            {
                "role": "user",
                "content": [{"type": "input_file", "file_id": uploaded["id"]}],
            }
        ],
    ):
        pass
    assert "<file name=\"notes.txt\">" in engine.job.request["messages"][0]["content"]
    assert "alpha beta" in engine.job.request["messages"][0]["content"]


def test_embeddings_and_rerank_are_capability_gated_and_standard_shaped():
    class AuxiliaryEngine(FakeEngine):
        def embed(self, inputs, *, dimensions=None):
            width = dimensions or 3
            return [[float(index + 1)] * width for index, _ in enumerate(inputs)], 7

        def rerank(self, query, documents):
            assert query == "q"
            return [0.1, 0.9, 0.2]

    engine = AuxiliaryEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(
            Request(
                base + "/v1/embeddings",
                data=json.dumps(
                    {"model": "fixture", "input": ["a", "b"], "dimensions": 2}
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
        ) as response:
            embedded = json.load(response)
        assert embedded == {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [1.0, 1.0], "index": 0},
                {"object": "embedding", "embedding": [2.0, 2.0], "index": 1},
            ],
            "model": "fixture",
            "usage": {"prompt_tokens": 7, "total_tokens": 7},
        }
        with urlopen(
            Request(
                base + "/v1/rerank",
                data=json.dumps(
                    {
                        "model": "fixture",
                        "query": "q",
                        "documents": ["a", "b", "c"],
                        "top_n": 2,
                        "return_documents": True,
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
        ) as response:
            reranked = json.load(response)
        assert [item["index"] for item in reranked["results"]] == [1, 2]
        assert reranked["results"][0]["document"] == {"text": "b"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    engine = FakeEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        request = Request(
            base + "/v1/embeddings",
            data=json.dumps({"model": "fixture", "input": "a"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 501
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_files_backed_batch_lifecycle_executes_jsonl_requests(http_engine):
    _, base = http_engine
    row = {
        "custom_id": "request-1",
        "method": "POST",
        "url": "/v1/responses",
        "body": {"model": "fixture", "input": "hello", "store": False},
    }
    with upload_file(base, json.dumps(row).encode() + b"\n") as response:
        file_object = json.load(response)
    with urlopen(
        Request(
            base + "/v1/batches",
            data=json.dumps(
                {
                    "input_file_id": file_object["id"],
                    "endpoint": "/v1/responses",
                    "completion_window": "24h",
                    "metadata": {"case": "cpu"},
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
    ) as response:
        batch = json.load(response)
    for _ in range(100):
        with urlopen(base + "/v1/batches/" + batch["id"]) as response:
            batch = json.load(response)
        if batch["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.01)
    assert batch["status"] == "completed"
    assert batch["request_counts"] == {"total": 1, "completed": 1, "failed": 0}
    with urlopen(base + f"/v1/files/{batch['output_file_id']}/content") as response:
        result = json.loads(response.read())
    assert result["custom_id"] == "request-1"
    assert result["response"]["status_code"] == 200
    assert result["response"]["body"]["object"] == "response"


def test_batch_row_waits_out_a_momentarily_full_engine():
    class BusyOnceEngine(FakeEngine):
        busy = 1

        def submit(self, request, *, tenant_id="default"):
            if self.busy:
                self.busy -= 1
                raise Overloaded("maximum inflight requests reached")
            return super().submit(request, tenant_id=tenant_id)

    engine = BusyOnceEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        row = {
            "custom_id": "busy-row",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "fixture", "messages": [{"role": "user", "content": "hi"}]},
        }
        with upload_file(base, json.dumps(row).encode() + b"\n") as response:
            file_id = json.load(response)["id"]
        with _json_post(base, "/v1/batches", {
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        }) as response:
            batch = json.load(response)
        for _ in range(200):
            with urlopen(base + "/v1/batches/" + batch["id"]) as response:
                batch = json.load(response)
            if batch["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.01)
        assert batch["request_counts"] == {"total": 1, "completed": 1, "failed": 0}
        assert batch["error_file_id"] is None
        with urlopen(base + f"/v1/files/{batch['output_file_id']}/content") as response:
            assert json.loads(response.read())["response"]["status_code"] == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_vllm_lora_lifecycle_contract_is_engine_owned_and_fail_closed():
    class LoRAEngine(FakeEngine):
        def load_lora_adapter(self, name, path, *, base_model_name=None):
            return {
                "status": "loaded",
                "path": path,
                "base_model_name": base_model_name,
                "revision": "adapter-r1",
            }

        def unload_lora_adapter(self, name, *, lora_int_id=None):
            return {"status": "unloaded", "lora_int_id": lora_int_id}

    engine = LoRAEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(
            Request(
                base + "/v1/load_lora_adapter",
                data=json.dumps(
                    {
                        "lora_name": "sql",
                        "lora_path": "/models/sql",
                        "base_model_name": "fixture",
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
        ) as response:
            loaded = json.load(response)
        assert loaded["status"] == "loaded" and loaded["revision"] == "adapter-r1"
        with urlopen(
            Request(
                base + "/v1/unload_lora_adapter",
                data=json.dumps({"lora_name": "sql"}).encode(),
                headers={"Content-Type": "application/json"},
            )
        ) as response:
            assert json.load(response)["status"] == "unloaded"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    engine = FakeEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        request = Request(
            base + "/v1/load_lora_adapter",
            data=json.dumps({"lora_name": "sql", "lora_path": "/models/sql"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 501
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_hermes_custom_provider_wire_request(http_engine):
    engine, base = http_engine
    with post(base, options={"num_ctx": 262144}, reasoning_effort="none", think=False,
              enable_thinking=False) as response:
        assert response.status == 200
    assert engine.job.request["enable_thinking"] is False
    assert engine.job.request["context_limit"] == 262144
    assert "options" not in engine.job.request


def test_nonstreaming_logprobs_are_in_token_order(http_engine):
    _, base = http_engine
    with post(base, logprobs=True, top_logprobs=1,
              response_format={"type": "text"}) as response:
        data = json.load(response)
    content = data["choices"][0]["logprobs"]["content"]
    assert [item["id"] for item in content] == [5, 6]
    assert len(content) == data["usage"]["completion_tokens"]
    assert data["choices"][0]["message"]["content"] == "hello"


def test_streaming_logprobs_are_emitted_once_without_duplicate_text(http_engine):
    _, base = http_engine
    with post(base, stream=True, logprobs=True, top_logprobs=1) as response:
        records = [json.loads(line[6:]) for line in response.read().decode().splitlines()
                   if line.startswith("data: ") and line != "data: [DONE]"]
    content = [item for record in records for choice in record["choices"]
               for item in choice.get("logprobs", {}).get("content", [])]
    assert [item["id"] for item in content] == [5, 6]
    assert "".join(choice.get("delta", {}).get("content", "")
                   for record in records for choice in record["choices"]) == "hello"


@pytest.mark.parametrize("extra,thinking", [
    ({"think": True}, True), ({"reasoning_effort": "high"}, True),
    ({"reasoning_effort": "none"}, False),
    ({"reasoning_effort": "high", "enable_thinking": False}, False),
    ({"chat_template_kwargs": {"enable_thinking": True}}, True),
])
def test_thinking_aliases(extra, thinking):
    request = {"messages": [{"role": "user", "content": "hi"}], **extra}
    normalized = validate_request(request)
    assert normalized["enable_thinking"] is thinking
    assert request == {"messages": [{"role": "user", "content": "hi"}], **extra}


@pytest.mark.parametrize("extra", [
    {"options": []}, {"options": {"num_ctx": -1}},
    {"options": {"num_ctx": True}}, {"options": {"unknown": 1}},
    {"think": "yes"}, {"reasoning_effort": "bogus"},
    {"think": True, "enable_thinking": False},
    {"options": {"temperature": .5}, "temperature": .7},
    {"chat_template_kwargs": {"arbitrary_template": "bad"}},
])
def test_invalid_client_options_fail_closed(extra):
    with pytest.raises(ValueError):
        validate_request({"messages": [{"role": "user", "content": "hi"}], **extra})


def test_nested_sampling_controls():
    request = validate_request({"messages": [{"role": "user", "content": "hi"}],
                                "options": {"temperature": 0, "num_predict": 32}})
    assert request["temperature"] == 0 and request["max_tokens"] == 32


def test_streaming_has_terminal_and_receipt(http_engine):
    engine, base = http_engine
    with post(base, stream=True) as response:
        data = response.read().decode()
    assert '"content": "hello"' in data
    assert '"finish_reason": "stop"' in data
    assert data.endswith("data: [DONE]\n\n")


@pytest.mark.parametrize("include_usage", [None, False, True])
def test_chat_stream_sends_role_and_honors_include_usage(http_engine, include_usage):
    _engine, base = http_engine
    extra = {"stream": True}
    if include_usage is not None:
        extra["stream_options"] = {"include_usage": include_usage}
    with post(base, **extra) as response:
        wire = response.read().decode()
    chunks = [
        json.loads(line[len("data: "):])
        for line in wire.split("\n\n")
        if line.startswith("data: {")
    ]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": "hello"}
    assert all(
        "role" not in chunk["choices"][0]["delta"]
        for chunk in chunks[1:]
        if chunk["choices"]
    )
    # Usage keeps riding on the finish chunk for existing readers.
    finish = next(c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"])
    assert finish["usage"]["total_tokens"] == 7
    usage_only = [chunk for chunk in chunks if not chunk["choices"]]
    if include_usage:
        assert chunks[-1] is usage_only[0] and len(usage_only) == 1
        assert usage_only[0]["object"] == "chat.completion.chunk"
        assert usage_only[0]["usage"]["total_tokens"] == 7
    else:
        assert usage_only == []
    assert wire.endswith("data: [DONE]\n\n")


def test_parallel_sample_usage_counts_the_shared_prompt_once():
    class ParallelEngine(FakeEngine):
        def admit_parallel_samples(self, count):
            return {"samples": count}

        def submit_many(self, requests, *, tenant_id="default", **_kwargs):
            jobs = []
            for index, request in enumerate(requests):
                job = self.submit(request, tenant_id=tenant_id)
                job.prompt_tokens = 1000
                job.completion_tokens = 10 + index
                job.cached_tokens = 600 if index == 0 else 1000
                jobs.append(job)
            return jobs

    engine = ParallelEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with post(f"http://127.0.0.1:{server.server_port}", n=4) as response:
            data = json.load(response)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert len(data["choices"]) == 4
    assert data["usage"] == {
        "prompt_tokens": 1000,
        "completion_tokens": 10 + 11 + 12 + 13,
        "total_tokens": 1046,
        "prompt_tokens_details": {"cached_tokens": 600},
    }


def test_parallel_samples_above_lane_capacity_are_a_client_error():
    class LaneEngine(FakeEngine):
        max_lanes = 4
        admit_parallel_samples = ServingEngine.admit_parallel_samples

    engine = LaneEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(HTTPError) as error:
            post(f"http://127.0.0.1:{server.server_port}", n=5)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    # No retry can ever admit it: not a 429 with Retry-After.
    assert error.value.code == 400
    assert "Retry-After" not in error.value.headers
    assert "lane capacity" in json.load(error.value)["error"]["message"]
    assert engine.job is None


def test_overload_is_http_429(http_engine):
    engine, base = http_engine
    engine.overloaded = True
    with pytest.raises(HTTPError) as error:
        post(base)
    assert error.value.code == 429
    assert error.value.headers["Retry-After"] == "1"
    assert json.load(error.value)["error"]["type"] == "server_error"


def test_schema_reference_validation_failures_are_counted(http_engine):
    engine, base = http_engine
    schema = {
        "type": "json_schema",
        "json_schema": {
            "strict": True,
            "schema": {
                "$defs": {"loop": {"$ref": "#/$defs/loop"}},
                "$ref": "#/$defs/loop",
            },
        },
    }
    with pytest.raises(HTTPError) as error:
        post(base, response_format=schema)
    assert error.value.code == 400
    assert engine.counts["schema_ref_failures"] == 1


_CHAT_HI = [{"role": "user", "content": "hi"}]


def _recorded_http_requests(engine, expected=1):
    # The handler records the metric just after the response bytes are
    # written, so a fast client can read the response before it lands.
    deadline = time.monotonic() + 2
    while True:
        count = sum(engine.http_metrics.prometheus_snapshot()["requests"].values())
        if count >= expected or time.monotonic() > deadline:
            return count
        time.sleep(0.005)


@pytest.mark.parametrize(
    "path, body",
    [
        ("/v1/responses", []),
        ("/v1/messages", None),
        ("/v1/messages/count_tokens", [1]),
        ("/v1/chat/completions", {"messages": [{"role": ["user"], "content": "hi"}]}),
        ("/v1/chat/completions", {"messages": _CHAT_HI, "thinking_budget_mode": []}),
        ("/v1/responses", {"input": [{"type": ["message"]}]}),
        ("/v1/responses", {"input": "hi", "text": {"verbosity": {}}}),
        (
            "/v1/messages",
            {"max_tokens": 8, "messages": _CHAT_HI, "tool_choice": {"type": []}},
        ),
        (
            "/v1/batches",
            {"endpoint": [], "completion_window": "24h", "input_file_id": "x"},
        ),
        ("/v1/embeddings", {"input": "x", "encoding_format": []}),
        ("/v1/audio/speech", {"input": "x", "response_format": []}),
        ("/tokenize", {"messages": _CHAT_HI, "thinking_budget_mode": {}}),
    ],
)
def test_wrongly_typed_json_fields_are_a_counted_400(path, body):
    from http.client import HTTPConnection
    from mlx2.batch_metrics import HttpRuntimeMetrics

    engine = FakeEngine()
    engine.http_metrics = HttpRuntimeMetrics()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        client.request(
            "POST",
            path,
            body=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        response = client.getresponse()
        payload = json.loads(response.read())
        assert response.status == 400
        assert payload["error"]["message"]
        assert engine.job is None
        assert _recorded_http_requests(engine) == 1
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_unexpected_engine_fault_is_a_counted_500_not_a_client_error():
    from http.client import HTTPConnection
    from mlx2.batch_metrics import HttpRuntimeMetrics

    class BrokenEngine(FakeEngine):
        def submit(self, request, *, tenant_id="default"):
            raise TypeError("engine bug")

    engine = BrokenEngine()
    engine.http_metrics = HttpRuntimeMetrics()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        client.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps({"model": "fixture", "messages": _CHAT_HI}).encode(),
            headers={"Content-Type": "application/json"},
        )
        response = client.getresponse()
        payload = json.loads(response.read())
        assert response.status == 500
        assert payload["error"]["type"] == "server_error"
        assert "engine bug" not in payload["error"]["message"]
        assert _recorded_http_requests(engine) == 1
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_batching_status_endpoint_and_tenant_header(http_engine):
    engine, base = http_engine
    request = Request(
        base + "/v1/chat/completions",
        data=json.dumps({
            "model": "fixture",
            "messages": [{"role": "user", "content": "hi"}],
        }).encode(),
        headers={"Content-Type": "application/json", "X-Tenant-ID": "tenant-a"},
    )
    with urlopen(request) as response:
        assert response.status == 200
    assert engine.job.tenant_id == "tenant-a"
    with urlopen(base + "/v1/status/batching") as response:
        status = json.load(response)
    assert status["schema"] == "mlx2.batch-runtime.v1"


def test_request_body_limit_scales_for_north_500k_and_keeps_small_default():
    assert request_body_limit(16384) == MIN_REQUEST_BODY_BYTES
    assert request_body_limit(500000) > len(
        json.dumps(
            {
                "model": "fixture",
                "messages": [
                    {"role": "user", "content": "data " * (500000 - 256)}
                ],
                "max_tokens": 64,
            }
        ).encode()
    )
    assert request_body_limit(10**9) == MAX_REQUEST_BODY_BYTES
    assert request_body_limit(500000, 3 << 20) == 3 << 20
    with pytest.raises(ValueError):
        request_body_limit(500000, MAX_REQUEST_BODY_BYTES + 1)


def test_north_500k_sized_body_is_accepted_under_derived_cap():
    engine = FakeEngine()
    engine.max_context = 500000
    engine.max_request_bytes = request_body_limit(engine.max_context)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        handler_for(engine, max_request_bytes=engine.max_request_bytes),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        content = "data " * (500000 - 256)
        with post(f"http://127.0.0.1:{server.server_port}", messages=[
            {"role": "user", "content": content}
        ]) as response:
            assert response.status == 200
            assert json.load(response)["mlx2"] == {"cache": "apcv2"}
        with urlopen(f"http://127.0.0.1:{server.server_port}/v1/status") as response:
            status = json.load(response)
        assert status["http"] == {
            "max_request_bytes": request_body_limit(500000)
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_request_body_over_configured_cap_is_rejected_before_submit():
    engine = FakeEngine()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), handler_for(engine, max_request_bytes=256)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(HTTPError) as error:
            post(f"http://127.0.0.1:{server.server_port}", messages=[
                {"role": "user", "content": "x" * 1024}
            ])
        assert error.value.code == 413
        assert json.load(error.value)["error"]["type"] == "invalid_request_error"
        assert engine.job is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("content_length, expected", [(None, 413), ("bogus", 400)])
def test_missing_or_malformed_content_length_behavior_is_preserved(
    http_engine, content_length, expected
):
    import http.client

    _, base = http_engine
    host = base.removeprefix("http://")
    connection = http.client.HTTPConnection(host)
    connection.putrequest("POST", "/v1/chat/completions")
    if content_length is not None:
        connection.putheader("Content-Length", content_length)
    connection.endheaders()
    response = connection.getresponse()
    assert response.status == expected
    response.read()
    connection.close()


_SMUGGLED = b"GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nX-Smuggled: 1\r\n\r\n"


@pytest.mark.parametrize(
    "head",
    [
        # Unknown route: rejected before the body is read.
        b"POST /v1/unknown HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        b"Content-Length: %d\r\n\r\n" % len(_SMUGGLED),
        # APCv2 session route on an engine without sessions.
        b"POST /v1/apc/sessions/s1/park HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        b"Content-Length: %d\r\n\r\n" % len(_SMUGGLED),
        # Admin control body over its 64 KiB cap.
        b"POST /v1/admin/quiesce HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        b"Content-Length: 70000\r\n\r\n",
        # Content-Length that is not a decimal byte count.
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        b"Content-Length: 12abc\r\n\r\n",
        # int() accepts a sign and underscores; HTTP does not.
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        b"Content-Length: +2\r\n\r\n{}",
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        b"Content-Length: 0_2\r\n\r\n{}",
    ],
)
def test_reply_before_reading_the_body_closes_the_connection(http_engine, head):
    import re
    import socket

    _, base = http_engine
    host, port = base.removeprefix("http://").split(":")
    with socket.create_connection((host, int(port)), timeout=5) as client:
        client.sendall(head + _SMUGGLED)
        received = b""
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            received += chunk
    statuses = re.findall(rb"HTTP/1\.[01] (\d{3})", received)
    assert len(statuses) == 1 and statuses[0].startswith(b"4")
    assert b"Connection: close" in received
    assert b'"object": "list"' not in received


def test_expect_100_continue_keeps_the_connection_alive(http_engine):
    import socket

    _, base = http_engine
    host, port = base.removeprefix("http://").split(":")
    body = json.dumps({"model": "fixture", "messages": [{"role": "user", "content": "hi"}]})
    head = (
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        "Content-Type: application/json\r\nExpect: 100-continue\r\n"
        f"Content-Length: {len(body)}\r\n\r\n"
    ).encode()
    with socket.create_connection((host, int(port)), timeout=5) as client:
        client.sendall(head)
        interim = client.recv(65536)
        # The interim 100 promises the body will be read; it closes nothing.
        assert interim.startswith(b"HTTP/1.1 100")
        assert b"Connection: close" not in interim
        client.sendall(body.encode())
        client.sendall(b"GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        received = b""
        while b'"object": "list"' not in received:
            chunk = client.recv(65536)
            if not chunk:
                break
            received += chunk
    assert b"Connection: close" not in received
    assert b'"object": "list"' in received


def test_runtime_has_no_legacy_apc_or_unified_import():
    import ast
    from pathlib import Path

    root = Path(__file__).parents[1] / "src" / "mlx2"
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("mlx_lm")
            if isinstance(node, ast.Import):
                assert all(not n.name.startswith("mlx_lm") for n in node.names)
            if isinstance(node, ast.ClassDef):
                assert node.name not in {"AutomaticPrefixCache", "LRUPromptCache"}


def test_tool_history_arguments_are_normalized_without_mutation():
    from mlx2.adapters.flash_next import FlashNextAdapter
    from types import SimpleNamespace

    captured = []
    adapter = object.__new__(FlashNextAdapter)
    adapter.tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **kw: captured.append(messages) or [1]
    )
    request = {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "weather", "arguments": '{"city":"Toronto"}'}}
                ],
            }
        ]
    }
    adapter.prompt_tokens(request)
    assert captured[0][0]["tool_calls"][0]["function"]["arguments"] == {
        "city": "Toronto"
    }
    assert isinstance(
        request["messages"][0]["tool_calls"][0]["function"]["arguments"], str
    )


def test_flash_next_diagnostics_do_not_rewalk_a_mutating_model_tree():
    from types import SimpleNamespace

    from mlx2.adapters.flash_next import FlashNextAdapter

    class MutatingModel:
        def named_modules(self):
            raise RuntimeError("OrderedDict mutated during iteration")

    moe = SimpleNamespace(
        fused_gate_up=True,
        fused_expert_kernel_mode="tile4",
        fused_expert_dispatches={"scalar": 1, "tile4": 2},
        fused_expert_fallbacks=3,
        moe_router_calls=4,
    )
    adapter = object.__new__(FlashNextAdapter)
    adapter.model = MutatingModel()
    adapter.policy = SimpleNamespace(as_dict=dict)
    adapter._tables = []
    adapter._diagnostic_modules = (moe,)

    diagnostics = adapter.diagnostics()

    assert diagnostics["moe"] == {
        "fused_gate_up_layers": 1,
        "expert_modes": ["tile4"],
        "dispatches": {"scalar": 1, "tile4": 2},
        "fallbacks": 3,
        "router_calls": 4,
    }


def test_trickled_request_body_is_bounded_by_a_whole_body_deadline(monkeypatch):
    """A per-recv socket timeout never fires for a client that sends one byte
    every few seconds; the handler must bound the whole body read instead."""
    import socket
    import time

    engine = FakeEngine()
    Handler = handler_for(engine)
    monkeypatch.setattr(Handler, "REQUEST_BODY_DEADLINE_SECONDS", 0.5)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = socket.create_connection(("127.0.0.1", server.server_port))
        client.settimeout(5)
        client.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
            b"Content-Type: application/json\r\nContent-Length: 4000\r\n\r\n"
        )
        started = time.monotonic()
        client.sendall(b"{")
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = client.recv(4096)
            if not chunk:
                break
            response += chunk
        elapsed = time.monotonic() - started
        assert b" 408 " in response.split(b"\r\n", 1)[0], response[:80]
        assert elapsed < 4, f"handler waited {elapsed:.1f}s for a trickled body"
        client.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_nonstreaming_client_disconnect_cancels_the_job():
    """A non-streaming request that blocks on the event queue must notice the
    client closing its connection and cancel its job instead of decoding to
    max_tokens with nobody listening."""
    import socket
    import time
    from collections import Counter

    class SilentEngine(FakeEngine):
        def __init__(self):
            super().__init__()
            self.counts = Counter()

        def submit(self, request, *, tenant_id="default"):
            self.job = Job(request)  # never emits an event
            return self.job

    engine = SilentEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = json.dumps({"model": "fixture", "messages": [{"role": "user", "content": "hi"}]}).encode()
        client = socket.create_connection(("127.0.0.1", server.server_port))
        client.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode() + body
        )
        deadline = time.monotonic() + 5
        while engine.job is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert engine.job is not None
        time.sleep(0.3)
        assert not engine.job.cancelled.is_set()
        client.close()  # client gives up
        assert engine.job.cancelled.wait(5), "job was not cancelled after the client disconnected"
        assert engine.counts["client_disconnects"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_streaming_client_disconnect_is_counted_like_a_nonstreaming_one():
    """A streaming client that goes away mid-stream ends in the write's
    BrokenPipe/ConnectionReset branch.  It recorded the 499 but not the
    request_finish client_disconnect counter the ClientGone branch keeps."""
    import socket
    import struct

    from mlx2.batch_metrics import HttpRuntimeMetrics

    class EndlessEngine(FakeEngine):
        def __init__(self):
            super().__init__()
            self.http_metrics = HttpRuntimeMetrics()

        def submit(self, request, *, tenant_id="default"):
            self.job = job = Job(request)
            job.prompt_tokens = 5

            def produce():
                while not job.cancelled.is_set():
                    try:
                        job.events.put({"delta": {"content": "x" * 512}}, timeout=0.05)
                    except Exception:  # noqa: BLE001 - a full queue just retries
                        pass

            threading.Thread(target=produce, daemon=True).start()
            return job

    engine = EndlessEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = json.dumps({"model": "fixture", "stream": True,
                           "messages": [{"role": "user", "content": "hi"}]}).encode()
        client = socket.create_connection(("127.0.0.1", server.server_port))
        client.settimeout(5)
        client.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode() + body
        )
        assert client.recv(64).startswith(b"HTTP/1.")
        # Abort with a reset, as a crashed or killed client would.
        client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        client.close()
        assert engine.job.cancelled.wait(5)
        deadline = time.monotonic() + 5
        while (
            not engine.http_metrics.prometheus_snapshot()["requests"]
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        requests = engine.http_metrics.prometheus_snapshot()["requests"]
        assert list(requests.items()) == [(("POST", "chat_completions", "4xx"), 1)]
        assert engine.counts["client_disconnects"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class SingleToolCallEngine(FakeEngine):
    """An engine whose parser honoured ``parallel_tool_calls:false``.

    The over-calling model behaviour is absorbed by ``OutputParser`` (see
    ``mlx2.output.within_parallel_bound``), so what reaches the wire is one
    call.  This pins that every surface -- Chat, Responses and Anthropic
    Messages, buffered and streamed -- answers it as an ordinary success, the
    way it could not when the parser raised and the lane finished 502.
    """

    def submit(self, request, *, tenant_id="default"):
        job = Job(request)
        job.tenant_id = tenant_id
        job.prompt_tokens, job.completion_tokens = 5, 2
        job.events.put({
            "delta": {
                "content": "",
                "tool_calls": [{
                    "index": 0,
                    "id": "call_weather",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": '{"city":"Toronto"}',
                    },
                }],
            }
        })
        job.events.put({"finish_reason": "tool_calls", "receipt": {"cache": "apcv2"}})
        return job


@pytest.fixture
def single_tool_call_endpoint():
    engine = SingleToolCallEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield engine, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def _json_post(base, path, body):
    return urlopen(
        Request(
            base + path,
            method="POST",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
    )


def test_parallel_bound_answer_is_a_success_on_every_surface(single_tool_call_endpoint):
    _, base = single_tool_call_endpoint

    chat = {
        "model": "fixture",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": TOOLS,
        "parallel_tool_calls": False,
    }
    with _json_post(base, "/v1/chat/completions", chat) as response:
        payload = json.load(response)
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    assert len(payload["choices"][0]["message"]["tool_calls"]) == 1

    with _json_post(base, "/v1/chat/completions", {**chat, "stream": True}) as response:
        wire = response.read().decode()
    assert '"error"' not in wire
    assert wire.endswith("data: [DONE]\n\n")

    responses_body = {
        "model": "fixture",
        "input": "hi",
        "tools": [
            {
                "type": "function",
                "name": tool["function"]["name"],
                "parameters": tool["function"]["parameters"],
                "strict": True,
            }
            for tool in TOOLS
        ],
        "parallel_tool_calls": False,
    }
    with _json_post(base, "/v1/responses", responses_body) as response:
        payload = json.load(response)
    assert payload["status"] == "completed"
    assert [item["type"] for item in payload["output"]] == ["function_call"]

    with _json_post(base, "/v1/responses", {**responses_body, "stream": True}) as response:
        wire = response.read().decode()
    assert "response.failed" not in wire
    assert "response.completed" in wire

    messages_body = {
        "model": "fixture",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "name": tool["function"]["name"],
                "input_schema": tool["function"]["parameters"],
            }
            for tool in TOOLS
        ],
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
    }
    with _json_post(base, "/v1/messages", messages_body) as response:
        payload = json.load(response)
    assert payload["stop_reason"] == "tool_use"
    assert [block["type"] for block in payload["content"]] == ["tool_use"]

    with _json_post(base, "/v1/messages", {**messages_body, "stream": True}) as response:
        wire = response.read().decode()
    assert "event: error" not in wire
    assert "event: message_stop" in wire


class HostedBackend:
    """A hosted (MCP) backend fake: ``weather`` runs server-side."""

    def __init__(self, tools=None):
        self.tools = tools or TOOLS
        self.executed = []

    def prepare(self, tools):
        return self.tools, {"weather": "binding"}

    def execute(self, binding, arguments):
        self.executed.append(arguments)
        return {"temperature": 21}


def _hosted_call(call_id="call_weather", name="weather", index=0):
    return {
        "index": index,
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": '{"city":"T"}'},
    }


class HostedEngine(FakeEngine):
    """Round scripts keyed by round number; the last one repeats."""

    def __init__(self, rounds):
        super().__init__()
        self.rounds = rounds
        self.requests = []
        self.jobs = []

    def submit(self, request, *, tenant_id="default"):
        self.requests.append(request)
        self.job = Job(request)
        self.jobs.append(self.job)
        script = self.rounds[min(len(self.requests), len(self.rounds)) - 1]
        for event in script(self.job):
            self.job.events.put(event)
        return self.job


def _round_calls(*calls):
    def script(_job):
        return [
            {"delta": {"tool_calls": list(calls)}},
            {"finish_reason": "tool_calls", "receipt": {}},
        ]
    return script


def _round_text(text):
    def script(_job):
        return [{"text": text}, {"finish_reason": "stop", "receipt": {}}]
    return script


MCP_TOOL = {
    "type": "mcp",
    "server_label": "weather",
    "server_url": "https://example.invalid/mcp",
    "require_approval": "never",
}


def _serve_hosted(engine, backend):
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), handler_for(engine, tool_backend=backend)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def close():
        server.shutdown()
        server.server_close()
        thread.join()

    return f"http://127.0.0.1:{server.server_port}", close


@pytest.mark.parametrize("choice", [
    "required", {"type": "function", "name": "weather"},
])
def test_hosted_tool_choice_binds_only_the_first_round(choice):
    # The executed call satisfies ``required``/a named choice; round two must
    # be free to answer instead of 502ing on its text (or looping to the
    # round limit under a forced tool grammar).
    engine = HostedEngine([_round_calls(_hosted_call()), _round_text("It is 21 C.")])
    base, close = _serve_hosted(engine, HostedBackend())
    try:
        with post_response(base, input="weather?", tools=[MCP_TOOL], tool_choice=choice) as response:
            payload = json.load(response)
    finally:
        close()
    assert payload["output"][0]["content"][0]["text"] == "It is 21 C."
    assert payload["tool_choice"] == choice
    assert engine.requests[0]["tool_choice"] != "auto"
    assert engine.requests[1]["tool_choice"] == "auto"


def test_hosted_required_tool_choice_still_fails_when_no_round_calls_a_tool():
    engine = HostedEngine([_round_text("No tool needed.")])
    base, close = _serve_hosted(engine, HostedBackend())
    try:
        with pytest.raises(HTTPError) as error:
            post_response(base, input="weather?", tools=[MCP_TOOL], tool_choice="required")
    finally:
        close()
    assert error.value.code == 502


def _run_batch(base, rows, endpoint="/v1/responses"):
    content = "".join(json.dumps(row) + "\n" for row in rows).encode()
    with upload_file(base, content) as response:
        file_id = json.load(response)["id"]
    with _json_post(base, "/v1/batches", {
        "input_file_id": file_id,
        "endpoint": endpoint,
        "completion_window": "24h",
    }) as response:
        batch = json.load(response)
    for _ in range(500):
        with urlopen(base + "/v1/batches/" + batch["id"]) as response:
            batch = json.load(response)
        if batch["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.01)

    def lines(file_id):
        if file_id is None:
            return []
        with urlopen(base + f"/v1/files/{file_id}/content") as response:
            return [json.loads(line) for line in response.read().splitlines()]

    return batch, lines(batch.get("output_file_id")), lines(batch.get("error_file_id"))


def test_batch_rows_with_hosted_tools_fail_closed():
    # A batch row cannot run the hosted loop; it used to return the internal
    # ``weather`` call unexecuted and count as completed.
    engine = HostedEngine([
        lambda job: (_round_calls(_hosted_call()) if job.request.get("tools")
                     else _round_text("hello"))(job)
    ])
    backend = HostedBackend()
    base, close = _serve_hosted(engine, backend)
    try:
        batch, outputs, errors = _run_batch(base, [
            {"custom_id": "hosted", "method": "POST", "url": "/v1/responses",
             "body": {"model": "fixture", "input": "weather?", "tools": [MCP_TOOL]}},
            {"custom_id": "plain", "method": "POST", "url": "/v1/responses",
             "body": {"model": "fixture", "input": "hi"}},
        ])
    finally:
        close()
    assert batch["status"] == "completed"
    assert batch["request_counts"] == {"total": 2, "completed": 1, "failed": 1}
    assert [row["custom_id"] for row in outputs] == ["plain"]
    assert [row["custom_id"] for row in errors] == ["hosted"]
    assert "hosted (MCP) tools" in errors[0]["error"]["message"]
    assert backend.executed == []
    assert len(engine.requests) == 1


@pytest.mark.parametrize("stream", [False, True])
def test_mixed_hosted_and_client_calls_fail_closed(stream):
    # Executing only the hosted call has no replayable transcript, and
    # returning it hands the client an internal call it never declared.
    notify = {
        "type": "function",
        "function": {"name": "notify", "parameters": {"type": "object"}},
    }
    engine = HostedEngine([_round_calls(
        _hosted_call(),
        _hosted_call(call_id="call_notify", name="notify", index=1),
    )])
    backend = HostedBackend(tools=[*TOOLS, notify])
    base, close = _serve_hosted(engine, backend)
    tools = [
        MCP_TOOL,
        {"type": "function", "name": "notify", "parameters": {"type": "object"}},
    ]
    try:
        # A hosted stream is buffered until its last round, so it fails
        # before committing any bytes, like the non-stream request.
        with pytest.raises(HTTPError) as error:
            post_response(base, input="go", tools=tools, stream=stream)
        assert error.value.code == 502
        assert "mixed" in json.load(error.value)["error"]["message"]
    finally:
        close()
    assert backend.executed == []


@pytest.mark.parametrize("fault", [
    "call_bad_json", "call_http_500", "call_rpc_error", "list_bad_json",
    "list_not_a_list",
])
def test_mcp_transport_and_protocol_failures_are_bad_gateway(fault):
    # A malformed MCP reply is the upstream's fault, not the caller's: it
    # used to surface as 400 (JSONDecodeError is a ValueError) or 500.
    from http.server import BaseHTTPRequestHandler

    from mlx2.tool_backend import ConfiguredToolBackend

    class MCP(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, status, data):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            message = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if "id" not in message:
                return self.reply(202, b"")
            method = message["method"]
            result = {}
            if method == "tools/list":
                if fault == "list_bad_json":
                    return self.reply(200, b"{truncated")
                result = {"tools": "weather" if fault == "list_not_a_list" else [
                    {"name": "weather", "inputSchema": {"type": "object"}}
                ]}
            elif method == "tools/call":
                if fault == "call_bad_json":
                    return self.reply(200, b"{truncated")
                if fault == "call_http_500":
                    return self.reply(500, b"")
                if fault == "call_rpc_error":
                    return self.reply(200, json.dumps({
                        "jsonrpc": "2.0", "id": message["id"],
                        "error": {"code": -32000, "message": "boom"},
                    }).encode())
            self.reply(200, json.dumps(
                {"jsonrpc": "2.0", "id": message["id"], "result": result}
            ).encode())

    mcp = ThreadingHTTPServer(("127.0.0.1", 0), MCP)
    mcp_thread = threading.Thread(target=mcp.serve_forever, daemon=True)
    mcp_thread.start()
    url = f"http://127.0.0.1:{mcp.server_port}/mcp"
    engine = HostedEngine([_round_calls(_hosted_call(name="mcp__w__weather"))])
    base, close = _serve_hosted(
        engine, ConfiguredToolBackend({"servers": {"w": {"server_url": url}}})
    )
    try:
        with pytest.raises(HTTPError) as error:
            post_response(base, input="weather?", tools=[{
                "type": "mcp",
                "server_label": "w",
                "server_url": url,
                "require_approval": "never",
            }])
        assert error.value.code == 502
        assert "MCP" in json.load(error.value)["error"]["message"]
    finally:
        close()
        mcp.shutdown()
        mcp.server_close()
        mcp_thread.join()


@pytest.mark.parametrize("stream", [False, True])
def test_hosted_usage_sums_cached_and_reasoning_tokens_over_rounds(stream):
    def counted(script, prompt, cached, reasoning):
        def run(job):
            job.prompt_tokens, job.completion_tokens = prompt, 10
            job.cached_tokens, job.reasoning_tokens = cached, reasoning
            return script(job)
        return run

    engine = HostedEngine([
        counted(_round_calls(_hosted_call()), 100, 50, 7),
        counted(_round_text("It is 21 C."), 200, 90, 0),
    ])
    base, close = _serve_hosted(engine, HostedBackend())
    try:
        with post_response(base, input="weather?", tools=[MCP_TOOL], stream=stream) as response:
            if stream:
                usage = next(
                    json.loads(line[len("data: "):])["response"]["usage"]
                    for line in response.read().decode().splitlines()
                    if line.startswith("data: {") and '"response.completed"' in line
                )
            else:
                usage = json.load(response)["usage"]
    finally:
        close()
    assert usage == {
        "input_tokens": 300,
        "output_tokens": 20,
        "total_tokens": 320,
        "input_tokens_details": {"cached_tokens": 140},
        "output_tokens_details": {"reasoning_tokens": 7},
    }
