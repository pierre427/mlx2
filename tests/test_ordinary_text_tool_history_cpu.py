"""Synthetic HF-template history contracts; no model, weights or GPU required."""
import copy
from types import SimpleNamespace

import pytest
from jinja2 import Environment

from mlx2.adapters.ordinary_text import OrdinaryTextAdapter


@pytest.mark.parametrize('arguments', ['{"goal":"ok"}', {'goal': 'ok'}])
def test_ordinary_template_decodes_history_arguments_without_changing_wire_request(arguments):
    template = Environment().from_string(
        '{% for message in messages %}{% for call in message.tool_calls|default([]) %}'
        '{% for key, value in call.function.arguments|items %}{{key}}={{value}}{% endfor %}'
        '{% endfor %}{% endfor %}')
    seen = []

    def apply(messages, **options):
        seen.append(options)
        rendered = template.render(messages=messages)
        return [1, 2] if options['tokenize'] else rendered

    adapter = OrdinaryTextAdapter()
    adapter.tokenizer = SimpleNamespace(apply_chat_template=apply)
    request = {'messages': [{'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'historical-call', 'type': 'function', 'function': {'name': 'example', 'arguments': arguments}}]},
        {'role': 'tool', 'tool_call_id': 'historical-call', 'content': 'ok'},
        {'role': 'user', 'content': 'Continue'}]}
    original = copy.deepcopy(request)
    assert adapter.render_prompt(request) == 'goal=ok'
    assert adapter.prompt_tokens(request) == [1, 2]
    assert request == original
    assert all(options['enable_thinking'] is False for options in seen)


@pytest.mark.parametrize('arguments', ['[]', 'null', '"string"', 'invalid-json'])
def test_ordinary_template_refuses_nonobject_historical_arguments(arguments):
    adapter = OrdinaryTextAdapter()
    adapter.tokenizer = SimpleNamespace(apply_chat_template=lambda *_args, **_kwargs: pytest.fail('Invalid history reached template'))
    request = {'messages': [{'role': 'assistant', 'tool_calls': [{'function': {'name': 'example', 'arguments': arguments}}]}]}
    with pytest.raises((ValueError, TypeError)):
        adapter.render_prompt(request)
