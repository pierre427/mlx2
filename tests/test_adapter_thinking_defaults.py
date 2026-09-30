"""Adapters that declare no reasoning render their prompts with thinking off."""

from types import SimpleNamespace

import pytest


def _capturing_tokenizer(calls):
    def apply_chat_template(messages, **kwargs):
        calls.append(kwargs)
        return [1] if kwargs.get("tokenize") else "text"

    return SimpleNamespace(apply_chat_template=apply_chat_template, has_thinking=True)


def test_ordinary_text_never_opens_a_think_channel():
    """TokenizerWrapper defaults enable_thinking to has_thinking; the ordinary
    adapter (Agnes 3 Flash) passed none, so its prompt ended in "<think>" and
    the reasoning leaked into content, whatever the client asked for."""
    from mlx2.adapters.ordinary_text import OrdinaryTextAdapter

    calls = []
    adapter = object.__new__(OrdinaryTextAdapter)
    adapter.tokenizer = _capturing_tokenizer(calls)
    request = {"messages": [{"role": "user", "content": "hi"}]}
    adapter.prompt_tokens(request)
    adapter.render_prompt(request)
    adapter.prompt_tokens({**request, "enable_thinking": False})
    assert [call["enable_thinking"] for call in calls] == [False, False, False]


@pytest.mark.parametrize("model_type", ["qwen3", "qwen3_moe", "llama"])
def test_standard_decoder_defaults_thinking_off(model_type):
    """No REASONING is declared, but the default followed has_thinking: Qwen3
    prompts rendered in thinking mode and the parser opened a reasoning
    channel that serving's thinking_enabled() reported closed."""
    from mlx2.adapters.standard_decoder import StandardDecoderAdapter
    from mlx2.contracts import Capability
    from mlx2.serving import thinking_enabled

    calls = []
    adapter = object.__new__(StandardDecoderAdapter)
    adapter.tokenizer = _capturing_tokenizer(calls)
    adapter.config = {"model_type": model_type}
    request = {"messages": [{"role": "user", "content": "2+2?"}]}
    adapter.prompt_tokens(request)
    adapter.render_prompt(request)
    assert [call["enable_thinking"] for call in calls] == [False, False]
    assert adapter.output_parser(request).channel == "content"
    assert thinking_enabled(adapter, request) is False
