"""Invalid JSON values must remain request errors, without a model runtime."""

from copy import deepcopy

import pytest
from test_server_quiesce import AdminEngine

from mlx2.agent_compat import AgentCompat, translate_responses_tools
from mlx2.anthropic_compat import anthropic_request_to_chat
from mlx2.api_resources import BatchManager, FileStore
from mlx2.openai_compat import responses_to_chat_request
from mlx2.server import (
    embeddings_payload,
    validate_quiesce_body,
    validate_request,
    validate_speech_request,
)

CHAT = {"messages": [{"role": "user", "content": "hello"}]}
ANTHROPIC = {**CHAT, "max_tokens": 20}
COMPAT = AgentCompat(enabled=True)


@pytest.mark.parametrize("bad", [[], {}])
@pytest.mark.parametrize(
    "kind, extra",
    [
        ("chat", {"messages": [{"role": None, "content": "hello"}]}),
        ("chat", {"messages": [{"role": "user", "content": [{"type": None}]}]}),
        ("chat", {"thinking_budget_mode": None}),
        ("speech", {"response_format": None}),
        ("speech", {"stream_format": None}),
        ("embedding", {"encoding_format": None}),
        ("responses", {"service_tier": None}),
        ("responses", {"include": [None]}),
        ("responses", {"reasoning": {"summary": None}}),
        ("responses", {"text": {"verbosity": None}}),
        ("responses", {"input": [{"type": None}]}),
        ("responses", {"input": [{"role": None, "content": "hello"}]}),
        ("responses", {"input": [{"role": "user", "content": [{"type": None}]}]}),
        ("responses", {"input": [{"role": "assistant", "phase": None, "content": "hello"}]}),
        ("responses", {"tool_choice": {"type": None, "name": "tool"}}),
        ("responses", {"tools": [{"type": None}]}),
        ("responses", {"tools": [{"type": "function", "name": None}]}),
        ("responses", {"tools": [{"type": "custom", "name": "tool", "format": {"type": "grammar", "syntax": None, "definition": "x"}}]}),
        ("responses", {"input": [{"type": "function_call_output", "call_id": "call", "output": [{"type": None}]}]}),
        ("anthropic", {"service_tier": None}),
        ("anthropic", {"messages": [{"role": None, "content": "hello"}]}),
        ("anthropic", {"messages": [{"role": "user", "content": [{"type": None}]}]}),
        ("anthropic", {"tool_choice": {"type": None}}),
        ("anthropic", {"thinking": {"type": "adaptive", "display": None}}),
        ("anthropic", {"output_config": {"effort": None}}),
        ("anthropic", {"system": [{"type": "text", "text": "system", "cache_control": {"type": "ephemeral", "ttl": None}}]}),
        ("batch", {"endpoint": None}),
    ],
)
def test_container_values_for_scalar_fields_raise_value_error(kind, extra, bad):
    def replace(value):
        if value is None:
            return deepcopy(bad)
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        return value

    extra = replace(extra)
    with pytest.raises(ValueError):
        if kind == "chat":
            validate_request({**CHAT, **extra})
        elif kind == "speech":
            validate_speech_request({"input": "hello", **extra})
        elif kind == "embedding":
            embeddings_payload(AdminEngine(), {"input": "hello", **extra})
        elif kind == "responses":
            responses_to_chat_request({"input": "hello", **extra}, agent_compat=COMPAT)
        elif kind == "anthropic":
            anthropic_request_to_chat({**ANTHROPIC, **extra}, agent_compat=COMPAT)
        else:
            BatchManager(FileStore(), None).create("tenant", extra)


@pytest.mark.parametrize("name", [None, "", [], {}, 123])
def test_namespace_members_require_an_original_function_name(name):
    member = {"type": "function", "name": name}
    with pytest.raises(ValueError):
        translate_responses_tools(
            [{"type": "namespace", "name": "group", "tools": [member]}], COMPAT
        )


def test_named_choice_of_dropped_hosted_tool_is_a_request_error():
    with pytest.raises(ValueError, match="tool_choice"):
        responses_to_chat_request(
            {
                "input": "hello",
                "tools": [{"type": "web_search"}],
                "tool_choice": {"type": "function", "name": "search"},
            },
            agent_compat=COMPAT,
        )


@pytest.mark.parametrize("value", [10**400, -(10**400)])
def test_oversized_json_numbers_are_request_errors(value):
    for name in (
        "temperature", "top_p", "min_p", "repetition_penalty",
        "presence_penalty", "frequency_penalty",
    ):
        with pytest.raises(ValueError):
            validate_request({**CHAT, name: value})
    with pytest.raises(ValueError):
        validate_request({**CHAT, "logit_bias": {"1": value}})
    with pytest.raises(ValueError):
        validate_speech_request({"input": "hello", "speed": value})
    with pytest.raises(ValueError):
        validate_quiesce_body({"drain_timeout_seconds": value})
