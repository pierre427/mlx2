"""An explicit JSON null in a nullable request field means "unset".

openai-python serializes a keyword passed as ``None`` as ``null``
(``create(..., max_tokens=None, stop=None)`` sends ``"stop": null``), and the
OpenAI schema declares these fields nullable.  The validator treated null as
absent only for ``response_format`` and ``grammar`` and answered 400 for every
other field; chat also refused ``user``, which Responses validates and ignores.
"""

import pytest

from mlx2.anthropic_compat import anthropic_request_to_chat
from mlx2.openai_compat import responses_to_chat_request
from mlx2.server import validate_request

CHAT = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
COMPLETION = {"model": "m", "prompt": "hi"}

NULLABLE_COMMON = (
    "stop",
    "max_tokens",
    "seed",
    "temperature",
    "top_p",
    "n",
    "logprobs",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "stream",
    "stream_options",
    "user",
)
NULLABLE_CHAT = NULLABLE_COMMON + (
    "max_completion_tokens",
    "top_logprobs",
    "tool_choice",
    "tools",
    "parallel_tool_calls",
    "reasoning_effort",
    "enable_thinking",
    "top_k",
    "min_p",
)


@pytest.mark.parametrize("field", NULLABLE_CHAT)
def test_chat_explicit_null_means_absent(field):
    assert validate_request({**CHAT, field: None}, True) == validate_request(CHAT, True)


@pytest.mark.parametrize("field", NULLABLE_COMMON)
def test_completions_explicit_null_means_absent(field):
    assert validate_request({**COMPLETION, field: None}, False) == validate_request(
        COMPLETION, False
    )


def test_openai_python_none_keywords_are_accepted():
    # The body openai-python sends for chat.completions.create(temperature=None,
    # top_p=None, max_tokens=None, stop=None, seed=None).
    body = {
        **CHAT,
        "max_tokens": None,
        "seed": None,
        "stop": None,
        "temperature": None,
        "top_p": None,
    }
    request = validate_request(body)
    assert not any(value is None for value in request.values())
    # The server default applies, so the receipt reports max_tokens_defaulted.
    assert "max_tokens" not in request


@pytest.mark.parametrize(
    "pair", [("max_tokens", "max_completion_tokens"), ("max_completion_tokens", "max_tokens")]
)
def test_null_beside_a_value_is_not_a_disagreement(pair):
    null, value = pair
    request = validate_request({**CHAT, null: None, value: 7})
    assert request["max_tokens"] == 7


def test_null_options_member_means_absent():
    request = validate_request({**CHAT, "options": {"temperature": None, "num_predict": 9}})
    assert "temperature" not in request and request["max_tokens"] == 9


def test_null_is_not_a_wildcard_for_wrong_types():
    # Only null means "unset"; a wrongly typed value is still a 400.
    for field, value in (("stop", 3), ("max_tokens", "8"), ("stream", "yes"), ("user", 4)):
        with pytest.raises(ValueError):
            validate_request({**CHAT, field: value}, True)


def test_null_required_fields_still_fail():
    with pytest.raises(ValueError):
        validate_request({"messages": None})
    with pytest.raises(ValueError):
        validate_request({"prompt": None}, chat=False)


def test_chat_accepts_and_ignores_user_like_responses():
    assert validate_request({**CHAT, "user": "end-user-42"}) == validate_request(CHAT)
    assert validate_request({**COMPLETION, "user": "u"}, False) == validate_request(
        COMPLETION, False
    )


@pytest.mark.parametrize(
    "field",
    [
        "temperature",
        "top_p",
        "max_output_tokens",
        "tool_choice",
        "parallel_tool_calls",
        "stream",
        "stream_options",
        "user",
        "store",
        "include",
        "service_tier",
        "truncation",
        "top_logprobs",
        "prompt_cache_key",
        "reasoning",
        "tools",
    ],
)
def test_responses_null_means_absent(field):
    translated, options = responses_to_chat_request({"input": "hello", field: None})
    expected, expected_options = responses_to_chat_request({"input": "hello"})
    assert translated == expected and options == expected_options
    assert validate_request(translated) == validate_request(expected)


def test_responses_null_effort_is_absent_after_validation():
    translated, _ = responses_to_chat_request(
        {"input": "hello", "reasoning": {"effort": None}}
    )
    assert "reasoning_effort" not in validate_request(translated)


def test_anthropic_null_max_tokens_is_still_missing():
    # Anthropic requires max_tokens; a null must not fall through to the
    # server default now that a forwarded null means "absent".
    body = {
        "model": "fixture",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": None,
    }
    with pytest.raises(ValueError, match="max_tokens is required"):
        anthropic_request_to_chat(body)
    counted = anthropic_request_to_chat(body, count_tokens=True)
    assert "max_tokens" not in validate_request(counted)


# Review round 1: "null means unset" covers recognised fields only.  An unknown
# field is refused whatever its value, so a null must not slip it past the
# unsupported-field check of any entry point.


@pytest.mark.parametrize("body", [CHAT, COMPLETION])
def test_null_unknown_field_is_still_refused(body):
    with pytest.raises(ValueError, match="unsupported request fields: unsupported_capability"):
        validate_request({**body, "unsupported_capability": None}, "messages" in body)


def test_responses_null_unknown_field_is_still_refused():
    with pytest.raises(ValueError, match="unsupported Responses fields: unsupported_capability"):
        responses_to_chat_request({"input": "hello", "unsupported_capability": None})


def test_null_unknown_options_member_is_still_refused():
    with pytest.raises(ValueError, match="unsupported options: unsupported_option"):
        validate_request({**CHAT, "options": {"unsupported_option": None}})


@pytest.mark.parametrize("field", ["options", "chat_template_kwargs", "think"])
def test_null_client_option_container_means_absent(field):
    # The local-client controls that normalize_client_options folds in are
    # recognised fields too: their null stays "unset".
    assert validate_request({**CHAT, field: None}) == validate_request(CHAT)
