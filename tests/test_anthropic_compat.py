import json
import threading
from collections import Counter
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from mlx2.anthropic_compat import (
    AnthropicStreamTranslator,
    ModelOutputError,
    anthropic_request_to_chat,
    chat_result_to_anthropic,
)
from mlx2.reasoning_signatures import ReasoningSigner
from mlx2.server import handler_for, validate_request
from mlx2.serving import HostPromptCache, Job


def test_anthropic_omitted_disabled_and_enabled_thinking_translation():
    base = {
        "model": "fixture",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 64,
    }
    omitted = anthropic_request_to_chat(base)
    disabled = anthropic_request_to_chat(
        {**base, "thinking": {"type": "disabled"}}
    )
    enabled = anthropic_request_to_chat(
        {**base, "thinking": {"type": "enabled", "budget_tokens": 16}}
    )

    assert omitted["enable_thinking"] is False
    assert omitted == disabled
    assert HostPromptCache.key(omitted) == HostPromptCache.key(disabled)
    assert HostPromptCache.key(omitted) != HostPromptCache.key(enabled)
    assert enabled["enable_thinking"] is True
    assert enabled["thinking_budget"] == 16
    assert enabled["thinking_budget_mode"] == "history"


def test_anthropic_translation_maps_thinking_tools_and_signed_history():
    signer = ReasoningSigner(b"shared-secret")
    signature = signer.sign_anthropic(
        model="fixture", tenant="tenant-a", text="prior thought"
    )
    metadata = {}
    request = anthropic_request_to_chat(
        {
            "model": "fixture",
            "system": [{"type": "text", "text": "be precise", "cache_control": {"type": "ephemeral"}}],
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "prior thought", "signature": signature},
                        {"type": "text", "text": "prior answer"},
                    ],
                },
                {"role": "user", "content": "continue"},
            ],
            "max_tokens": 64,
            "thinking": {"type": "enabled", "budget_tokens": 16},
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "tool", "name": "lookup", "disable_parallel_tool_use": True},
        },
        signer=signer,
        tenant_id="tenant-a",
        model="fixture",
        translation_metadata=metadata,
    )
    assert request["messages"][1]["reasoning_content"] == "prior thought"
    assert request["thinking_budget_mode"] == "history"
    assert request["parallel_tool_calls"] is False
    assert metadata["reasoning_signature_rejections"] == 0


def test_anthropic_thinking_budget_can_reach_one_below_output_limit():
    request = anthropic_request_to_chat(
        {
            "model": "fixture",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 32_000,
            "thinking": {"type": "enabled", "budget_tokens": 31_999},
        }
    )
    assert validate_request(request)["thinking_budget"] == 31_999
    with pytest.raises(ValueError, match="less than max_tokens"):
        anthropic_request_to_chat(
            {
                "model": "fixture",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32_000,
                "thinking": {"type": "enabled", "budget_tokens": 32_000},
            }
        )


def test_anthropic_output_and_stream_sign_reasoning_and_fail_closed_tool_json():
    signer = ReasoningSigner(b"shared-secret")
    result = {
        "id": "chatcmpl-1",
        "model": "fixture",
        "choices": [{"finish_reason": "stop", "message": {"content": "answer", "reasoning_content": "thought"}}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 4}},
        "mlx2": {"stop_sequence": "STOP"},
    }
    payload = chat_result_to_anthropic(
        result, {}, signer=signer, tenant_id="tenant-a"
    )
    thinking = payload["content"][0]
    assert signer.verify_anthropic(
        thinking["signature"], model="fixture", tenant="tenant-a", text="thought"
    )
    assert payload["usage"]["input_tokens"] == 5
    assert payload["stop_reason"] == "stop_sequence"
    assert payload["stop_sequence"] == "STOP"

    translator = AnthropicStreamTranslator(
        message_id="1", model="fixture", signer=signer, tenant_id="tenant-a"
    )
    events = translator.start()
    events += translator.delta({"reasoning_content": "thought"})
    events += translator.delta({"content": "answer"})
    events += translator.finish(
        "stop", {"completion_tokens": 2}, {"stop_sequence": "STOP"}
    )
    assert [event["type"] for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[3]["delta"]["type"] == "signature_delta"
    assert events[-2]["delta"] == {
        "stop_reason": "stop_sequence",
        "stop_sequence": "STOP",
    }

    bad = {
        **result,
        "choices": [{"finish_reason": "tool_calls", "message": {"content": "", "tool_calls": [{"function": {"name": "x", "arguments": "["}}]}}],
    }
    with pytest.raises(ModelOutputError):
        chat_result_to_anthropic(bad)


def test_anthropic_sdk_output_blocks_round_trip_nullable_public_fields():
    request = anthropic_request_to_chat(
        {
            "model": "fixture",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "lookup",
                            "input": {"city": "Toronto"},
                            "caller": None,
                        },
                        {"type": "text", "text": "checking", "citations": None},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "sunny",
                        }
                    ],
                },
            ],
            "max_tokens": 64,
        }
    )
    assert request["messages"][0]["tool_calls"][0]["id"] == "call_1"
    assert request["messages"][0]["content"] == "checking"
    direct = anthropic_request_to_chat(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "lookup",
                            "input": {},
                            "caller": {"type": "direct"},
                        }
                    ],
                },
                {"role": "user", "content": "continue"},
            ],
            "max_tokens": 8,
        }
    )
    assert direct["messages"][0]["tool_calls"][0]["id"] == "call_1"
    with pytest.raises(ValueError, match="non-direct tool_use caller"):
        anthropic_request_to_chat(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "call_1",
                                "name": "lookup",
                                "input": {},
                                "caller": {
                                    "type": "code_execution_20260120",
                                    "tool_id": "srvtoolu_1",
                                },
                            }
                        ],
                    },
                    {"role": "user", "content": "continue"},
                ],
                "max_tokens": 8,
            }
        )


class AnthropicEngine:
    model_path = "fixture"

    def __init__(self):
        self.counts = Counter()
        self.reasoning_signer = ReasoningSigner(b"shared-secret")
        self.invalid_tool = False
        self.last_request = None
        self.thinking_enabled = None

    def status(self):
        return {"healthy": True, "error": None, "model": "fixture"}

    def batching_status(self):
        return {}

    def count_tokens(self, request):
        assert request["messages"][-1]["content"] == "hello"
        return 7

    def submit(self, request, *, tenant_id="default"):
        self.last_request = request
        # Simulate North/Xing: the adapter defaults thinking on unless the
        # translated request contains an explicit false control.
        self.thinking_enabled = request.get("enable_thinking", True)
        job = Job(request)
        job.tenant_id = tenant_id
        job.prompt_tokens = 9
        job.cached_tokens = 4
        job.completion_tokens = 2
        if self.invalid_tool:
            job.events.put({"delta": {"tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "["}}]}})
            finish = "tool_calls"
        else:
            job.events.put({"delta": {"reasoning_content": "thought"}})
            job.events.put({"delta": {"content": "answer"}})
            finish = "stop"
        job.events.put({"finish_reason": finish, "receipt": {"cache": "apcv2"}})
        return job


@pytest.fixture
def anthropic_endpoint():
    engine = AnthropicEngine()
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
            headers={"Content-Type": "application/json", "X-Tenant-ID": "tenant-a"},
        )
    )


def test_anthropic_http_nonstream_count_stream_errors_and_invalid_tool(anthropic_endpoint):
    engine, base = anthropic_endpoint
    body = {
        "model": "fixture",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 64,
        "thinking": {"type": "enabled", "budget_tokens": 16},
    }
    with _post(base, "/v1/messages/count_tokens", {k: v for k, v in body.items() if k != "max_tokens"}) as response:
        assert json.load(response) == {"input_tokens": 7}
    with _post(base, "/v1/messages", body) as response:
        payload = json.load(response)
    assert [block["type"] for block in payload["content"]] == ["thinking", "text"]
    assert payload["usage"]["input_tokens"] == 5

    with _post(base, "/v1/messages", {**body, "stream": True}) as response:
        wire = response.read().decode()
    assert "event: message_start" in wire
    assert "event: signature_delta" not in wire
    assert '"type": "signature_delta"' in wire
    assert wire.index("thinking_delta") < wire.index("text_delta")

    with pytest.raises(HTTPError) as error:
        _post(base, "/v1/messages", {"model": "fixture", "messages": []})
    assert error.value.code == 400
    assert json.load(error.value)["type"] == "error"

    with pytest.raises(HTTPError) as error:
        _post(
            base,
            "/v1/messages",
            {
                **body,
                "max_tokens": 32_000,
                "thinking": {"type": "enabled", "budget_tokens": 32_000},
            },
        )
    assert error.value.code == 400
    assert "less than max_tokens" in json.load(error.value)["error"]["message"]

    engine.invalid_tool = True
    with pytest.raises(HTTPError) as error:
        _post(
            base,
            "/v1/messages",
            {
                **body,
                "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
                "tool_choice": {"type": "any"},
            },
        )
    assert error.value.code == 502

    with _post(
        base,
        "/v1/messages",
        {
            **body,
            "stream": True,
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "any"},
        },
    ) as response:
        wire = response.read().decode()
    assert 'event: error' in wire
    assert '"type": "error"' in wire
    assert '"type": "api_error"' in wire
    assert 'chat.completion.chunk' not in wire


def test_anthropic_http_omitted_thinking_disables_thinking_default(
    anthropic_endpoint,
):
    engine, base = anthropic_endpoint
    with _post(
        base,
        "/v1/messages",
        {
            "model": "fixture",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 64,
        },
    ) as response:
        assert response.status == 200
    assert engine.last_request["enable_thinking"] is False
    assert engine.thinking_enabled is False


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/messages/count_tokens"])
def test_anthropic_unknown_model_uses_anthropic_not_found_envelope(
    anthropic_endpoint, path
):
    _engine, base = anthropic_endpoint
    body = {
        "model": "not-loaded",
        "messages": [{"role": "user", "content": "hello"}],
    }
    if path == "/v1/messages":
        body["max_tokens"] = 8
    with pytest.raises(HTTPError) as error:
        _post(base, path, body)
    assert error.value.code == 404
    assert json.load(error.value) == {
        "type": "error",
        "error": {"type": "not_found_error", "message": "unknown model"},
    }
