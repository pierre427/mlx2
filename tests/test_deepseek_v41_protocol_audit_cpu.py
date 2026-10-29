"""Protocol-only regressions; no model artifact is loaded or qualified."""
import copy
import json
from types import SimpleNamespace

import pytest
from test_structured_deferral import (
    scripted_engine,  # noqa: F401 - shared bounded CPU worker fixture
)

from mlx2.adapters.deepseek_v41 import DeepSeekV41Adapter
from mlx2.adapters.deepseek_v41_protocol import CALLS_CLOSE, CALLS_OPEN
from mlx2.openai_compat import ToolContractError, enforce_tool_contract
from mlx2.server import validate_request


def adapter():
    target = object.__new__(DeepSeekV41Adapter)
    target.draft_model = None
    target.tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(map(ord, text)))
    return target


def tool(name='f', **extra):
    return {'type': 'function', 'function': {'name': name, 'parameters': {'type': 'object'}}, **extra}


def call(name='f', parameter=''):
    return CALLS_OPEN + f'\n<｜DSML｜ invoke name="{name}">' + parameter + '</｜DSML｜ invoke>\n' + CALLS_CLOSE


@pytest.mark.parametrize('name', ['missing', 'other::f'])
def test_truncated_undeclared_identity_is_not_exempt_from_terminal_contract(name):
    request = {'messages': [], 'tools': [tool()], 'tool_choice': 'required'}
    parser = adapter().output_parser(request)
    parser.finish(CALLS_OPEN + f'\n<｜DSML｜ invoke name="{name}">', 'length')
    assert parser.truncated_tool_call['function'] == name
    with pytest.raises(ToolContractError, match='undeclared'):
        enforce_tool_contract(request, [], finish_reason='length', truncated_tool_call=parser.truncated_tool_call)


def test_partial_reader_skips_invoke_spellings_inside_a_raw_parameter():
    request = {'messages': [], 'tools': [tool()], 'tool_choice': 'required'}
    parser = adapter().output_parser(request)
    partial = CALLS_OPEN + '\n<｜DSML｜ invoke name="f"><｜DSML｜ parameter name="x" string="true">literal <｜DSML｜ invoke name="missing">'
    parser.finish(partial, 'length')
    assert parser.truncated_tool_call == {'function': 'f', 'cause': 'max_tokens'}
    enforce_tool_contract(request, [], finish_reason='length', truncated_tool_call=parser.truncated_tool_call)


@pytest.mark.parametrize('namespace', ['weather', {'name': 'weather', 'description': 'Weather tools'}])
def test_namespace_metadata_is_canonical_across_prompt_parser_choice_and_history(namespace):
    request = {'messages': [{'role': 'user', 'content': 'go'},
        {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'c', 'namespace': 'weather', 'function': {'name': 'f', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'c', 'content': 'ok'}, {'role': 'user', 'content': 'again'}],
        'tools': [tool(namespace=namespace)], 'tool_choice': {'type': 'function', 'function': {'name': 'f'}}}
    original = copy.deepcopy(request)
    target = adapter()
    normalized = target.normalize_request(validate_request(request))
    assert normalized['tools'][0]['function']['name'] == 'weather::f'
    assert normalized['tool_choice']['function']['name'] == 'weather::f'
    assert target.normalize_request(normalized) == normalized
    assert '<｜DSML｜ invoke name="weather::f">' in target.render_prompt(normalized)
    events = target.output_parser(normalized).push(call('weather::f'), final=True)
    calls = [item for event in events for item in event.get('tool_calls', [])]
    enforce_tool_contract(normalized, calls)
    assert request == original


@pytest.mark.parametrize('tools', [
    [tool('weather::f', namespace='other')],
    [tool('f', namespace='weather'), tool('weather::f')],
    [tool('a::b::f')],
])
def test_namespace_conflicts_and_canonical_collisions_fail_closed(tools):
    with pytest.raises(ValueError, match='namespace|unique'):
        adapter().normalize_request({'tools': tools})


@pytest.mark.parametrize('choice', [
    {'type': 'invalid', 'function': {'name': 'f'}},
    {'type': 'function', 'function': {'name': 'f'}, 'extra': True},
    {'type': 'function', 'function': {'name': 'f', 'extra': True}},
])
def test_adapter_normalization_does_not_erase_invalid_choice_controls(choice):
    with pytest.raises(ValueError, match='tool_choice'):
        validate_request({'messages': [{'role': 'user', 'content': 'go'}],
                          'tools': [tool()], 'tool_choice': choice}, adapter=adapter())


@pytest.mark.parametrize('arguments', ['{"x": NaN}', '{"x": Infinity}', '{"x": 1e999}', '{"x": 1, "x": 2}', '{"x": {"nested": 1, "nested": 2}}', {'x': float('nan')}, {1: 'value'}, {'x': (1, 2)}])
def test_history_arguments_must_be_unambiguous_json_native_objects(arguments):
    request = {'messages': [{'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'c', 'type': 'function', 'function': {'name': 'f', 'arguments': arguments}}]},
        {'role': 'tool', 'tool_call_id': 'c', 'content': 'ok'}, {'role': 'user', 'content': 'go'}]}
    with pytest.raises(ValueError, match='JSON'):
        adapter().render_prompt(request)


def test_valid_history_preserves_json_native_value_types():
    arguments = {'s': 'true', 'b': True, 'i': 2, 'a': [None, False, 1.5]}
    request = {'messages': [{'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'c', 'function': {'name': 'f', 'arguments': json.dumps(arguments)}}]},
        {'role': 'tool', 'tool_call_id': 'c', 'content': 'ok'}, {'role': 'user', 'content': 'go'}]}
    text = adapter().render_prompt(request)
    assert 'name="s" string="true">true<' in text
    assert 'name="b" string="false">true<' in text
    assert 'name="i" string="false">2<' in text


@pytest.mark.parametrize('stream', [False, True, None], ids=['nonstream', 'stream', 'worker-collector'])
@pytest.mark.parametrize('choice', [None, 'f', 'weather::f'])
def test_real_worker_namespaces_survive_http_terminal_validation(scripted_engine, monkeypatch, stream, choice):  # noqa: F811 - fixture injection
    import re
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    import test_structured_deferral as fixture_module

    from mlx2.server import collect_nonstream_job, handler_for

    fragments = [part for part in re.split('(｜DSML｜)', call('weather::f')) if part]
    pieces = list(fixture_module.PIECES)
    for piece in ['｜DSML｜', '<think>', *fragments]:
        if piece not in pieces:
            pieces.append(piece)
    monkeypatch.setattr(fixture_module, 'PIECES', pieces)
    build, state = scripted_engine
    engine = build(declare_marker=True, max_lanes=1)
    target = adapter()
    target.tokenizer = engine.adapter.tokenizer
    for name in ('normalize_request', 'output_parser', 'tool_constraint', 'thinking_enabled', 'thinking_close_token_ids'):
        setattr(engine.adapter, name, getattr(target, name))
    engine.adapter.reasoning_effort_semantics = target.reasoning_effort_semantics
    state['script'] = [pieces.index(part) for part in fragments] + [0]
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = {'model': 'fake', 'messages': [{'role': 'user', 'content': 'go'}],
                'tools': [tool(namespace='weather')] + ([tool(namespace='other')] if choice != 'f' else []),
                'tool_choice': {'type': 'function', 'function': {'name': choice}} if choice else 'auto',
                'stream': bool(stream), 'max_tokens': 64, 'temperature': 0}
        if stream is None:
            # Submit the original aliases directly so the collector must use
            # the normalized worker request rather than the caller's body.
            job = engine.submit(body)
            result, _usage, _receipt = collect_nonstream_job(job, body, chat=True)
            assert result['message']['tool_calls'][0]['function']['name'] == 'weather::f'
            return
        request = Request(f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                          data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        try:
            with urlopen(request, timeout=10) as response:
                status, wire = response.status, response.read().decode()
        except HTTPError as exc:
            status, wire = exc.code, exc.read().decode()
        assert status == 200, wire
        if stream:
            assert '"name": "weather::f"' in wire
            assert '[DONE]' in wire and '"error"' not in wire
        else:
            choices = json.loads(wire)['choices']
            assert len(choices) == 1
            assert all(choice['message']['tool_calls'][0]['function']['name'] == 'weather::f' for choice in choices)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
