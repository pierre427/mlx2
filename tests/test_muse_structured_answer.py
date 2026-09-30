"""Muse replies open with a recipient header; a client grammar waits for the user's."""
from types import SimpleNamespace as NS

import pytest

from mlx2.adapters.muse_glimmer import MuseGlimmerAdapter
from mlx2.serving import structured_answer_token_ids, thinking_enabled

HEADER = (328, 76976, 200023)  # " to", "=user", "<|message|>" in the Muse vocabulary
TOOLS = [{"type": "function", "function": {"name": "weather", "parameters": {}}}]
CHAT = {"messages": [{"role": "user", "content": "hi"}], "response_format": {"type": "json_object"}}


def _adapter():
    def encode(text, **_kw):
        assert text == " to=user<|message|>"
        return list(HEADER)

    adapter = NS(tokenizer=NS(encode=encode))
    adapter.structured_answer_token_ids = (
        lambda request: MuseGlimmerAdapter.structured_answer_token_ids(adapter, request)
    )
    adapter.thinking_enabled = MuseGlimmerAdapter.thinking_enabled
    return adapter


@pytest.mark.parametrize(
    ("extra", "deferred"),
    [
        # The prompt already writes the user's header: bind from the first token.
        ({}, False),
        ({"tools": TOOLS, "tool_choice": "none"}, False),
        # The model writes the header: after a tool's, or after its reasoning.
        ({"tools": TOOLS}, True),
        ({"tools": TOOLS, "tool_choice": "required"}, True),
        ({"tools": TOOLS, "tool_choice": {"type": "function", "function": {"name": "weather"}}}, True),
        ({"enable_thinking": True}, True),
        ({"reasoning_effort": "high"}, True),
    ],
)
def test_client_grammar_waits_for_the_users_header(extra, deferred):
    ids = structured_answer_token_ids(_adapter(), {**CHAT, **extra})
    assert ids == (HEADER if deferred else None)


def test_raw_completion_has_no_header():
    assert structured_answer_token_ids(_adapter(), {"prompt": "x"}) is None


def test_serving_reads_reasoning_effort_as_thinking_like_the_template():
    # Serving read enable_thinking alone and took this for a direct answer, so
    # "structured output requires thinking to be disabled" never fired.
    adapter = _adapter()
    assert thinking_enabled(adapter, {**CHAT, "reasoning_effort": "high"}) is True
    assert thinking_enabled(adapter, {**CHAT, "reasoning_effort": "none"}) is False
    assert thinking_enabled(adapter, {**CHAT, "reasoning_effort": "high", "enable_thinking": False}) is False
    assert thinking_enabled(adapter, CHAT) is False
    assert thinking_enabled(adapter, {"prompt": "x", "reasoning_effort": "high"}) is False


def test_header_ids_match_the_real_tokenizer():
    from pathlib import Path

    path = Path.home() / "mlx-models" / "Muse-Glimmer-30B-mlx-4bit"
    if not (path / "tokenizer.json").is_file():
        pytest.skip("Muse tokenizer not present")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    header = tuple(tokenizer.encode(" to=user<|message|>", add_special_tokens=False).ids)
    assert header == HEADER
    # The template renders an answer after reasoning as <|eom|><|start|>assistant
    # to=user<|message|>; the header keeps its ids there.
    ids = tokenizer.encode(
        " to=self<|message|>think<|eom|><|start|>assistant to=user<|message|>{}",
        add_special_tokens=False,
    ).ids
    assert any(tuple(ids[i : i + 3]) == HEADER for i in range(len(ids)))
