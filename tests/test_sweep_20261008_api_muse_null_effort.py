"""Muse treats a null ``reasoning_effort`` (and ``enable_thinking``) as absent.

``request.get("reasoning_effort", default)`` applies the default only to an
absent key, so an explicit null reached the strengths lookup and failed the
request with 400 "Unsupported Muse reasoning_effort", and ``thinking_enabled``
answered True for null but False for an omitted field.  The HTTP validator now
drops top-level nulls, but the engine is reachable without it, and North,
Xing and GPT-OSS already read None as absent.
"""

import pytest

from mlx2.adapters.muse_glimmer import MuseGlimmerAdapter


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.strength = kwargs["reasoning_strength"]
        return "<|start|>assistant"

    def encode(self, prompt, **kwargs):
        self.prompt = prompt
        return [1, 2]


def _adapter():
    adapter = MuseGlimmerAdapter.__new__(MuseGlimmerAdapter)
    adapter.tokenizer = _Tokenizer()
    return adapter


def _render(request):
    adapter = _adapter()
    adapter.prompt_tokens(request)
    return adapter.tokenizer.strength, adapter.tokenizer.prompt


MESSAGES = [{"role": "user", "content": "hi"}]


def _same(request, omitted):
    assert _render(request) == _render(omitted)
    assert MuseGlimmerAdapter.thinking_enabled(request) is (
        MuseGlimmerAdapter.thinking_enabled(omitted)
    )


@pytest.mark.parametrize("toggle", [None, False, True])
def test_null_effort_renders_like_an_omitted_one(toggle):
    omitted = {"messages": MESSAGES}
    if toggle is not None:
        omitted["enable_thinking"] = toggle
    _same({**omitted, "reasoning_effort": None}, omitted)


@pytest.mark.parametrize("effort", [None, "none", "medium"])
def test_null_toggle_renders_like_an_omitted_one(effort):
    omitted = {"messages": MESSAGES}
    if effort is not None:
        omitted["reasoning_effort"] = effort
    _same({**omitted, "enable_thinking": None}, omitted)
    # An explicit effort alone still opens reasoning.
    assert MuseGlimmerAdapter.thinking_enabled(omitted) is (effort == "medium")


def test_unknown_effort_still_fails():
    with pytest.raises(ValueError, match="Unsupported Muse reasoning_effort"):
        _render({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "x"})
