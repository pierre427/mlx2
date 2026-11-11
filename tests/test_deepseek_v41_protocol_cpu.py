"""Public V4.1 protocol probes: no artifact/model load or qualification claim."""
import copy
import json
from types import SimpleNamespace as NS

import pytest
import regex

from mlx2.adapters.deepseek_v41 import DESCRIPTOR, DeepSeekV41Adapter
from mlx2.adapters.deepseek_v41_protocol import (
    CALLS_CLOSE,
    CALLS_OPEN,
    constrained_tool_grammar,
    messages_for_encoding,
)
from mlx2.contracts import Capability
from mlx2.server import apply_ast_invocation, normalize_client_options, validate_request

TOOLS = [{'type': 'function', 'function': {'name': 'weather::lookup', 'strict': True,
    'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}, 'days': {'type': 'integer'}},
                   'required': ['city', 'days'], 'additionalProperties': False}}}]
CALL = (CALLS_OPEN + '\n<｜DSML｜ invoke name="weather::lookup">\n'
        '<｜DSML｜ parameter name="city" string="true">Montréal</｜DSML｜ parameter>\n'
        '<｜DSML｜ parameter name="days" string="false">2</｜DSML｜ parameter>\n'
        '</｜DSML｜ invoke>\n' + CALLS_CLOSE)


def adapter():
    result = object.__new__(DeepSeekV41Adapter)
    result.draft_model = None
    result.tokenizer = NS(encode=lambda value, **kw: [ord(c) for c in value])
    return result


@pytest.mark.parametrize('effort,numeric', [('minimal', 50), ('low', 50), ('medium', 75), ('high', 75), ('xhigh', 75), ('max', 100), ('ultra', 100), (1, 1), (100, 100)])
def test_http_and_direct_strength_share_prompt_and_effective_request(effort, numeric):
    direct = {'messages': [{'role': 'user', 'content': 'hello'}], 'reasoning_effort': effort}
    http = validate_request(normalize_client_options(copy.deepcopy(direct)))
    target = adapter()
    assert target.render_prompt(http) == target.render_prompt(direct)
    assert f'Reasoning Effort: {numeric}' in target.render_prompt(http)
    normalized = target.normalize_request(http)
    assert normalized['reasoning_effort'] == numeric
    assert normalized['_mlx2_reasoning_effort_requested'] == effort
    assert target.normalize_request(normalized) == normalized
    assert target.reasoning_effort_semantics == 'numeric_strength_1_100'


@pytest.mark.parametrize('effort', [0, 101, True, 1.5])
def test_http_rejects_invalid_numeric_strength(effort):
    with pytest.raises(ValueError, match='reasoning_effort'):
        normalize_client_options({'reasoning_effort': effort})


def test_none_default_and_explicit_thinking_controls():
    target = adapter()
    request = {'messages': [{'role': 'user', 'content': 'hi'}], 'reasoning_effort': 'none'}
    assert target.render_prompt(request).endswith('</think>')
    assert 'Reasoning Effort:' not in target.render_prompt(request)
    assert 'Reasoning Effort: 75' in target.render_prompt({**request, 'reasoning_effort': None, 'think': True})
    assert target.normalize_request({**request, 'enable_thinking': True})['reasoning_effort'] == 75
    with pytest.raises(ValueError, match='toggles disagree'):
        target.normalize_request({'enable_thinking': True, 'chat_template_kwargs': {'enable_thinking': False}})


def test_tools_and_history_render_vendored_wire_without_mutating_caller():
    request = {'messages': [{'role': 'user', 'content': 'weather'},
        {'role': 'assistant', 'content': None, 'reasoning_content': 'use tool', 'tool_calls': [
            {'id': 'c1', 'type': 'function', 'function': {'name': 'weather::lookup', 'arguments': '{"city":"Paris","days":2}'}}]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'sunny'},
        {'role': 'system', 'content': 'be concise'}, {'role': 'user', 'content': 'continue'}], 'tools': TOOLS}
    original = copy.deepcopy(request)
    text = adapter().render_prompt(request)
    assert CALLS_OPEN in text and '<｜DSML｜ invoke name="weather::lookup">' in text
    assert '<tool_result>sunny</tool_result>' in text
    assert '<｜System｜>be concise' in text
    assert request == original


@pytest.mark.parametrize('messages', [
    [{'role': 'tool', 'tool_call_id': 'unknown', 'content': 'x'}],
    [{'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'x', 'function': {'name': 'f', 'arguments': '{}'}}]}],
])
def test_tool_history_refuses_unassociated_or_unresolved_results(messages):
    with pytest.raises(ValueError, match='tool'):
        messages_for_encoding({'messages': messages})


def test_incremental_spaced_dsml_every_split_typed_namespaced_and_reasoning():
    target = adapter()
    request = {'messages': [], 'tools': TOOLS, 'enable_thinking': True}
    text = 'consider ' + CALL + '</think>\n' + CALL
    for cut in range(len(text) + 1):
        parser = target.output_parser(request)
        events = parser.push(text[:cut]) + parser.push(text[cut:], final=True)
        calls = [call for event in events for call in event.get('tool_calls', [])]
        assert len(calls) == 1
        assert calls[0]['function']['name'] == 'weather::lookup'
        assert json.loads(calls[0]['function']['arguments']) == {'city': 'Montréal', 'days': 2}
        assert ''.join(e.get('reasoning_content', '') for e in events) == 'consider ' + CALL


def test_json_quoted_closer_is_data_and_strict_schema_is_checked():
    tools = [{'type': 'function', 'function': {'name': 'f', 'parameters': {'type': 'object'}}}]
    raw = CALLS_OPEN + '\n<｜DSML｜ invoke name="f"><｜DSML｜ parameter name="x" string="false">' + json.dumps({'s': CALLS_CLOSE}) + '</｜DSML｜ parameter></｜DSML｜ invoke>\n' + CALLS_CLOSE
    parser = adapter().output_parser({'messages': [], 'tools': tools})
    events = []
    for character in raw:
        events += parser.push(character)
    events += parser.push('', final=True)
    assert json.loads(events[0]['tool_calls'][0]['function']['arguments'])['x'] == {'s': CALLS_CLOSE}
    parser = adapter().output_parser({'messages': [], 'tools': TOOLS})
    with pytest.raises(ValueError, match='schema'):
        parser.push(CALL.replace('>2</', '>true</'), final=True)


def test_truncation_does_not_emit_partial_call_and_code_examples_are_text():
    parser = adapter().output_parser({'messages': [], 'tools': TOOLS})
    assert not any('tool_calls' in e for e in parser.finish(CALL[:-20], 'length'))
    assert parser.truncated_tool_call is not None
    parser = adapter().output_parser({'messages': [], 'tools': TOOLS})
    events = parser.push('```\n' + CALL + '\n```', final=True)
    assert not any('tool_calls' in e for e in events)


def test_constrained_wire_schema_parallel_and_thinking_deferral_hooks():
    pattern = constrained_tool_grammar(TOOLS, 'required')
    assert regex.fullmatch(pattern, CALL)
    assert not regex.fullmatch(pattern, CALL.replace('>2</', '>true</'))
    second_invoke = CALL[len(CALLS_OPEN) + 1:-len(CALLS_CLOSE) - 1]
    parallel = CALL[:-len(CALLS_CLOSE) - 1] + '\n' + second_invoke + '\n' + CALLS_CLOSE
    assert regex.fullmatch(pattern, parallel)
    assert not regex.fullmatch(constrained_tool_grammar(TOOLS, 'required', parallel_tool_calls=False), parallel)
    assert adapter().thinking_close_token_ids() == tuple(map(ord, '</think>'))
    assert {Capability.TOOLS, Capability.GRAMMAR} <= DESCRIPTOR.capabilities


def test_speculative_routes_fail_closed_in_direct_hooks_and_capability_receipts():
    target = adapter()
    target.draft_model = object()
    for request in ({'tools': TOOLS}, {'grammar': 'yes'}, {'response_format': {'type': 'json_object'}}):
        with pytest.raises(ValueError, match='ordinary B1'):
            target.normalize_request(request)
    for route in ('external_draft', 'native_mtp', 'prompt_lookup'):
        assert not {Capability.TOOLS, Capability.GRAMMAR} & target.effective_route_capabilities(DESCRIPTOR.capabilities, route=route, max_lanes=1)


def test_ast_lowers_v41_copy_to_external_adapter_contract():
    args = NS(model='/target', ast=True, ast_drafter=None, ast_topology=None, ast_graft_pld=False,
              draft_model=None, external_draft=False, native_mtp=False, prompt_lookup=False)
    resolution = NS(adapter_type=DeepSeekV41Adapter, artifact={}, default_route='ordinary')
    policy = apply_ast_invocation(args, None, resolution)
    assert policy == {'draft_model': 'prompt_lookup', 'ast_prompt_lookup': True}
    assert args.external_draft and not args.prompt_lookup


def test_generated_plain_token_spellings_are_data_including_raw_parameter_closer():
    import re
    special = {'｜DSML｜': 90001, '<think>': 90002, '</think>': 90003}
    target = adapter()
    target.tokenizer = NS(encode=lambda value, **kw: [special[value]] if value in special else list(map(ord, value)))
    parser = target.output_parser({'messages': [], 'tools': TOOLS})
    # A whole call spelled by plain tokens remains answer text.
    parser.record_output_token(CALL, 99999)
    events = parser.push(CALL, final=True)
    assert not any('tool_calls' in event for event in events)
    assert ''.join(event.get('content', '') for event in events) == CALL
    # In a real call, a parameter closer spelled by plain tokens stays in
    # its raw string; only the dedicated DSML token closes the parameter.
    text = CALL.replace('Montréal', 'before </｜DSML｜ parameter> after')
    plain_start = text.index('before')
    plain_end = text.index('after') + len('after')
    parser = target.output_parser({'messages': [], 'tools': TOOLS})
    events, cursor = [], 0
    for match in re.finditer('｜DSML｜|<think>|</think>', text):
        if cursor < match.start():
            plain = text[cursor:match.start()]
            parser.record_output_token(plain, 99999)
            events += parser.push(plain)
        token = 99999 if plain_start <= match.start() < plain_end else special[match[0]]
        parser.record_output_token(match[0], token)
        events += parser.push(match[0])
        cursor = match.end()
    parser.record_output_token(text[cursor:], 99999)
    events += parser.push(text[cursor:], final=True)
    call = next(event['tool_calls'][0] for event in events if 'tool_calls' in event)
    assert json.loads(call['function']['arguments'])['city'] == 'before </｜DSML｜ parameter> after'
    assert parser._special_spans == set()


def test_plain_thinking_marker_does_not_switch_generated_channel():
    target = adapter()
    target.tokenizer = NS(encode=lambda value, **kw: [90000 + ['｜DSML｜', '<think>', '</think>'].index(value)] if value in ('｜DSML｜', '<think>', '</think>') else list(map(ord, value)))
    parser = target.output_parser({'messages': []})
    parser.record_output_token('<think>literal</think>', 99999)
    events = parser.push('<think>literal</think>', final=True)
    assert ''.join(e.get('content', '') for e in events) == '<think>literal</think>'
    assert not any('reasoning_content' in e for e in events)


def test_engine_direct_admission_normalizes_effort_before_grammar_and_render():
    from collections import Counter

    from mlx2.serving import ServingEngine
    engine = object.__new__(ServingEngine)
    engine.adapter = adapter()
    engine.route_capabilities = DESCRIPTOR.capabilities
    engine.qualification_mode = False
    engine.apc_reuse_disabled_reason = None
    engine.counts = Counter()
    engine.snapshot = {'settings': {'route': 'ordinary'}}
    engine._prepare_structured_automata = lambda request: ('strength', request['reasoning_effort'])
    engine._prerender_prompt = lambda request, **kwargs: [request['reasoning_effort']]
    job = engine._prepare_job({'messages': [{'role': 'user', 'content': 'hi'}], 'reasoning_effort': 'minimal'}, tenant_id='test')
    assert job.request['enable_thinking'] is True
    assert job.request['reasoning_effort'] == 50
    assert job.request['_mlx2_reasoning_effort_requested'] == 'minimal'
    assert job.structured_automata == ('strength', 50)
    assert job.admission_tokens == [50]


def test_v41_json_and_regex_processors_defer_until_actual_adapter_close():
    import mlx.core as mx
    import numpy as np
    from test_structured_deferral import PIECES, _tokenizer

    from mlx2.structured_output import make_structured_processor, structured_receipt
    target = adapter()
    target.tokenizer = _tokenizer()
    close = target.thinking_close_token_ids()
    assert close == (5,)
    for constraint in ({'response_format': {'type': 'json_object'}}, {'grammar': 'yes|no'}):
        processor = make_structured_processor(target.tokenizer, 2, defer_until=close, **constraint)
        logits = mx.zeros((1, len(PIECES)))
        before = processor(mx.array([1, 1, 2], dtype=mx.uint32), logits)
        assert np.isfinite(np.array(before)).all()
        after = processor(mx.array([1, 1, 2, 5], dtype=mx.uint32), logits)
        assert not np.isfinite(np.array(after)).all()
        receipt = structured_receipt(processor, 2)
        assert receipt['deferred'] is True


def anthropic_deferred(messages, *, tools=None):
    from mlx2.agent_compat import AgentCompat
    from mlx2.anthropic_compat import anthropic_request_to_chat
    return anthropic_request_to_chat({'model': 'fixture', 'max_tokens': 100,
        'tools': tools or [
            {'name': 'ToolSearch', 'input_schema': {'type': 'object', 'properties': {'query': {'type': 'string'}}}},
            {'name': 'weather::lookup', 'defer_loading': True, 'input_schema': TOOLS[0]['function']['parameters']}],
        'messages': messages}, agent_compat=AgentCompat(enabled=True), preserve_tool_schema_history=True)


def test_client_tool_search_activates_namespaced_definitions_inline_after_associated_result():
    history = [
        {'role': 'user', 'content': 'weather'},
        {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'search1', 'name': 'ToolSearch', 'input': {'query': 'weather'}}]},
        {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'search1', 'content': [{'type': 'tool_reference', 'tool_name': 'weather::lookup'}]}]},
        {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'weather1', 'name': 'weather::lookup', 'input': {'city': 'Paris', 'days': 1}}]},
        {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'weather1', 'content': 'sunny'}]},
    ]
    request = validate_request(anthropic_deferred(history))
    rendered = adapter().render_prompt(request)
    prefix, update = rendered.split('### Updated Tool Schemas', 1)
    assert '"name": "weather::lookup"' not in prefix
    assert '"name": "weather::lookup"' in update
    assert '<｜System｜>' in rendered
    assert rendered.index('<tool_result></tool_result>') < rendered.index('### Updated Tool Schemas')
    assert [tool['function']['name'] for tool in request['tools']] == ['ToolSearch', 'weather::lookup']
    # Rendering an extended history cannot move the deferred schema into its
    # old prompt prefix, which protects APCv2's exact-prefix contract.
    initial = validate_request(anthropic_deferred([history[0]]))
    assert '"name": "weather::lookup"' not in adapter().render_prompt(initial)


def test_deferred_tool_calls_fail_before_activation_and_after_removal():
    use = {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'c', 'name': 'weather::lookup', 'input': {}}]}
    result = {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'c', 'content': 'x'}]}
    with pytest.raises(ValueError, match='before activation'):
        anthropic_deferred([{'role': 'user', 'content': 'go'}, use, result])
    addition = {'role': 'system', 'content': [{'type': 'tool_addition', 'tool': {'type': 'tool_reference', 'name': 'weather::lookup'}}]}
    removal = {'role': 'system', 'content': [{'type': 'tool_removal', 'tool': {'type': 'tool_reference', 'name': 'weather::lookup'}}]}
    with pytest.raises(ValueError, match='after removal'):
        anthropic_deferred([{'role': 'user', 'content': 'go'}, addition, removal, use, result])
    with pytest.raises(ValueError, match='undeclared'):
        anthropic_deferred([{'role': 'user', 'content': 'go'}, {'role': 'system', 'content': [{'type': 'tool_addition', 'tool': {'type': 'tool_reference', 'name': 'missing'}}]}])


def test_numeric_strength_is_refused_for_other_adapters_at_generation_and_prompt_boundaries():
    from mlx2.serving import normalize_adapter_request, render_prompt_tokens
    other = NS(prompt_tokens=lambda request: [1])
    with pytest.raises(ValueError, match='not supported by the selected adapter'):
        normalize_adapter_request(other, {'reasoning_effort': 50})
    with pytest.raises(ValueError, match='not supported by the selected adapter'):
        render_prompt_tokens(other, {'reasoning_effort': 50})
    assert normalize_adapter_request(other, {'reasoning_effort': 'low'}) == {'reasoning_effort': 'low'}


from test_structured_deferral import (
    scripted_engine,  # noqa: F401 - register the shared pytest fixture
)


def test_real_serving_loop_wires_numeric_receipt_token_parser_and_constrained_dsml(scripted_engine, monkeypatch):  # noqa: F811 - pytest fixture injection
    import re

    import test_structured_deferral as scripted
    pieces = list(scripted.PIECES)
    fragments = [part for part in re.split('(｜DSML｜)', CALL) if part]
    for piece in ['｜DSML｜', '<think>', *fragments]:
        if piece not in pieces:
            pieces.append(piece)
    monkeypatch.setattr(scripted, 'PIECES', pieces)
    build, state = scripted_engine
    engine = build(declare_marker=True, execution_policy={"constrained_tool_grammar": True}, max_lanes=1)
    target = adapter()
    target.tokenizer = engine.adapter.tokenizer
    # Preserve the fixture's bounded CPU model/backend and exercise the real
    # V4.1 public adapter hooks in the actual worker and terminal receipts.
    for name in ('normalize_request', 'output_parser', 'tool_constraint', 'thinking_enabled', 'thinking_close_token_ids'):
        setattr(engine.adapter, name, getattr(target, name))
    engine.adapter.reasoning_effort_semantics = target.reasoning_effort_semantics
    state['script'] = [2, 5] + [pieces.index(part) for part in fragments] + [0]
    request = {'messages': [{'role': 'user', 'content': 'weather'}], 'reasoning_effort': 'minimal',
               'tools': TOOLS, 'tool_choice': 'required', 'max_tokens': 64, 'temperature': 0}
    job = engine.submit(request)
    events = []
    while True:
        event = job.events.get(timeout=10)
        events.append(event)
        if 'finish_reason' in event or 'error' in event:
            break
    assert 'error' not in events[-1], events[-1]
    calls = [call for event in events for call in event.get('delta', {}).get('tool_calls', [])]
    assert len(calls) == 1 and calls[0]['function']['name'] == 'weather::lookup'
    assert json.loads(calls[0]['function']['arguments']) == {'city': 'Montréal', 'days': 2}
    controls = events[-1]['receipt']['request_controls']
    assert controls['reasoning_effort'] == 50
    assert controls['requested_reasoning_effort'] == 'minimal'
    assert controls['effort_semantics'] == 'numeric_strength_1_100'
    assert controls['structured_output']['enforced'] is True
    assert controls['structured_output']['deferred'] is True


def test_template_control_metadata_can_be_serialized_by_status_endpoint():
    from mlx2.serving import validate_template_request

    target = adapter()
    controls = target.chat_template_options()
    assert json.loads(json.dumps({'chat_template_options': controls})) == {
        'chat_template_options': {'preserve_thinking': {'implemented': True, 'default': False}}}
    validate_template_request(target, {'preserve_thinking': True})


def _tool_loop(reasoning):
    assistant = {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': 'c1', 'type': 'function', 'function': {'name': 'weather::lookup', 'arguments': '{"city":"Paris","days":2}'}}]}
    if reasoning is not _ABSENT:
        assistant['reasoning_content'] = reasoning
    return {'messages': [{'role': 'user', 'content': 'weather'}, assistant,
                         {'role': 'tool', 'tool_call_id': 'c1', 'content': 'sunny'}],
            'tools': TOOLS, 'think': True}


_ABSENT = object()


@pytest.mark.parametrize('blank', [' ', ''])
def test_blank_echoed_reasoning_renders_as_absent_reasoning(blank):
    """antirez/ds4#1296: DeepSeek clients echo a reasoning-free turn as " ".
    After the last user message the encoder keeps that turn's think block, so
    a verbatim space is an unsampled token that breaks the cached prefix and
    re-prefills the rest of the conversation."""
    from mlx2.runtime.models.deepseek_v41_encoding import encode_messages

    target = adapter()
    absent = target.render_prompt(_tool_loop(_ABSENT))
    assert target.render_prompt(_tool_loop(blank)) == absent
    # The hazard is real: the verbatim encoder would have rendered the pad.
    raw = _tool_loop(' ')['messages']
    assert encode_messages(copy.deepcopy(raw), thinking_mode='thinking') != encode_messages(
        [m for m in copy.deepcopy(_tool_loop(_ABSENT)['messages'])], thinking_mode='thinking')
    # Real reasoning is still rendered.
    assert 'Need to run it.' in target.render_prompt(_tool_loop('Need to run it.'))
    assert 'Need to run it.' not in absent


@pytest.mark.parametrize('sampled', ['\n', '  ', ' \n'])
def test_other_whitespace_reasoning_is_rendered_as_sampled(sampled):
    """Whitespace other than the client pad may be what the model sampled and
    mlx2 returned verbatim; its exact echo keeps the cached prefix."""
    from mlx2.runtime.models.deepseek_v41_encoding import encode_messages

    rendered = adapter().render_prompt(_tool_loop(sampled))
    assert rendered != adapter().render_prompt(_tool_loop(_ABSENT))
    assert sampled + '</think>' in rendered
    assert rendered == encode_messages(messages_for_encoding(_tool_loop(sampled)), thinking_mode='thinking')
