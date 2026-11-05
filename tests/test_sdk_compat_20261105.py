"""Request shapes typed by anthropic-python 2.5.0 and openai-python 3.46.0.

Each shape is either accepted (echo-only metadata or a cache hint APCv2 makes
redundant) or refused with a typed 400 naming what is unsupported.
"""

import pytest

from mlx2.anthropic_compat import anthropic_request_to_chat
from mlx2.openai_compat import responses_to_chat_request

BASE = {
    "model": "fixture",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "hi"}],
}


def _messages(**extra):
    return anthropic_request_to_chat({**BASE, **extra})


def test_anthropic_top_level_cache_control_is_a_validated_hint():
    plain = _messages()
    for marker in ({"type": "ephemeral"}, {"type": "ephemeral", "ttl": "1h"}, None):
        assert _messages(cache_control=marker) == plain
    with pytest.raises(ValueError, match="cache_control ttl"):
        _messages(cache_control={"type": "ephemeral", "ttl": "2h"})
    with pytest.raises(ValueError, match="cache_control type"):
        _messages(cache_control={"type": "persistent"})


def test_anthropic_speed_standard_is_a_noop_and_fast_is_refused():
    plain = _messages()
    assert _messages(speed="standard") == plain
    assert _messages(speed=None) == plain
    with pytest.raises(ValueError, match="fast mode is unsupported"):
        _messages(speed="fast")


def test_anthropic_enabled_thinking_display():
    enabled = {"type": "enabled", "budget_tokens": 16}
    plain = _messages(thinking=enabled)
    assert _messages(thinking={**enabled, "display": "summarized"}) == plain
    assert _messages(thinking={**enabled, "display": None}) == plain
    with pytest.raises(ValueError, match="display omitted"):
        _messages(thinking={**enabled, "display": "omitted"})


def test_anthropic_between_tools_thinking_is_refused_by_name():
    with pytest.raises(ValueError, match="between_tools is unsupported"):
        _messages(thinking={"type": "between_tools"})


@pytest.mark.parametrize("field", ["diagnostics", "inference_geo", "temperatures"])
def test_anthropic_other_new_fields_stay_refused_by_name(field):
    with pytest.raises(ValueError, match=f"unsupported Anthropic request fields: {field}"):
        _messages(**{field: {"previous_message_id": "msg_1"}})


def _call(**extra):
    return {
        "type": "function_call",
        "call_id": "call_1",
        "name": "weather",
        "arguments": '{"city":"Toronto"}',
        **extra,
    }


def _output(**extra):
    return {"type": "function_call_output", "call_id": "call_1", "output": "21 C", **extra}


def _responses(input_items, **extra):
    request, _ = responses_to_chat_request(
        {"model": "fixture", "input": input_items, **extra}
    )
    return request


def test_responses_direct_caller_and_sync_call_metadata_round_trip():
    user = {"role": "user", "content": "weather?"}
    plain = _responses([user, _call(), _output()])
    echoed = _responses(
        [
            user,
            _call(caller={"type": "direct"}, **{"async": False}),
            _output(caller={"type": "direct"}, name="weather"),
        ]
    )
    assert echoed["messages"] == plain["messages"]


@pytest.mark.parametrize(
    "item, message",
    [
        (_call(caller={"type": "program", "caller_id": "p1"}), "programmatic tool calling"),
        (_call(caller={"type": "direct", "extra": 1}), "caller must be direct"),
        (_call(**{"async": True}), "async is unsupported"),
        (_output(caller={"type": "program", "caller_id": "p1"}), "programmatic tool calling"),
        (_output(name=""), "name must be nonempty"),
    ],
)
def test_responses_programmatic_and_async_calls_fail_closed(item, message):
    items = [{"role": "user", "content": "weather?"}]
    items += [item] if item["type"] == "function_call" else [_call(), item]
    with pytest.raises(ValueError, match=message):
        _responses(items)


def test_responses_prompt_cache_options_are_validated_hints():
    plain = _responses("hi")
    for options in (
        {"mode": "implicit"},
        {"mode": "explicit", "ttl": "12h"},
        {"ttl": "30m"},
        None,
    ):
        assert _responses("hi", prompt_cache_options=options) == plain
    for bad, message in (
        ({"mode": "always"}, "mode must be"),
        ({"ttl": "1h"}, "ttl must be"),
        ({"prewarm": True}, "unsupported prompt_cache_options fields: prewarm"),
        ({"comparison_response_id": "resp_1"}, "comparison_response_id"),
        ("explicit", "must be an object"),
    ):
        with pytest.raises(ValueError, match=message):
            _responses("hi", prompt_cache_options=bad)


def test_responses_prompt_cache_breakpoint_is_stripped_from_content_parts():
    plain = _responses([{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}])
    marked = _responses(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "hi",
                        "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "30m"},
                    }
                ],
            }
        ]
    )
    assert marked["messages"] == plain["messages"]
    with pytest.raises(ValueError, match="prompt_cache_breakpoint mode must be explicit"):
        _responses(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "hi",
                            "prompt_cache_breakpoint": {"mode": "implicit"},
                        }
                    ],
                }
            ]
        )
    image = _responses(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,AA==",
                        "prompt_cache_breakpoint": {"mode": "explicit"},
                    }
                ],
            }
        ]
    )
    assert "prompt_cache_breakpoint" not in image["messages"][0]["content"][0]


def test_review_round_null_and_type_edges():
    user = {"role": "user", "content": "weather?"}
    plain = _responses([user, _call(), _output()])
    # Nullable SDK fields are absent; async must be a real bool.
    assert _responses([user, _call(), _output(name=None, namespace=None)])["messages"] == plain["messages"]
    assert _responses([user, _call(), _output(namespace="tools")])["messages"] == plain["messages"]
    for value in (0, 0.0, "false"):
        with pytest.raises(ValueError, match="async is unsupported"):
            _responses([user, _call(**{"async": value})])
    # Options that ask for nothing are accepted; asking still fails closed.
    assert _responses("hi", prompt_cache_options={"prewarm": False, "comparison_response_id": None}) == _responses("hi")
    with pytest.raises(ValueError, match="prewarm"):
        _responses("hi", prompt_cache_options={"prewarm": True})
    # A breakpoint requires its mode.
    with pytest.raises(ValueError, match="mode must be explicit"):
        _responses([{"role": "user", "content": [{"type": "input_text", "text": "hi", "prompt_cache_breakpoint": {}}]}])
    # A bad display gets the generic message; adaptive accepts null.
    with pytest.raises(ValueError, match="display must be summarized or omitted"):
        _messages(thinking={"type": "enabled", "budget_tokens": 16, "display": "x"})


def test_adaptive_thinking_accepts_null_display_under_compat():
    from types import SimpleNamespace

    compat = SimpleNamespace(enabled=True)
    request = anthropic_request_to_chat(
        {**BASE, "thinking": {"type": "adaptive", "display": None}}, agent_compat=compat
    )
    assert request["enable_thinking"] is True
