"""CPU contracts for exact Kimi training targets and authored behavioral eval."""
import copy
import json
from pathlib import Path
import random

import pytest

from scripts.research import kimi_lora_data as data


class Tokenizer:
    markers = {'<|im_end|>': data.END_ID}

    def encode(self, text):
        tokens = []
        while text:
            marker = next((m for m in self.markers if text.startswith(m)), None)
            if marker:
                tokens.append(self.markers[marker]); text = text[len(marker):]
            else:
                tokens.append(ord(text[0])); text = text[1:]
        return tokens

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False):
        text = '<|im_system|>tool_declare<|im_middle|>' + json.dumps(tools) + '<|im_end|>' if tools else ''
        for message in messages:
            role = message['role']; marker = role if role in ('user', 'assistant') else 'system'
            text += f'<|im_{marker}|>' + (message.get('name') or role) + '<|im_middle|>'
            text += message.get('content', '')
            if message.get('tool_calls'):
                text += json.dumps(message['tool_calls'])
            text += '<|im_end|>'
        return self.encode(text) if tokenize else text


def record(messages):
    return {'id': 'test', 'messages': messages, 'metadata': {'split': 'train', 'split_group': 'whole-task',
            'source_id': 'verified-fixture', 'quality_tier': 'source_curated', 'curriculum_domain_bucket': 'coding_reasoning'}}


def test_reasoning_preserved_and_only_assistant_targets_receive_loss():
    original = record([{'role': 'user', 'content': 'USER_ONLY'},
                       {'role': 'assistant', 'content': 'The result is 7.', 'reasoning_content': 'ORIGINAL_REASONING'}])
    before = copy.deepcopy(original)
    result = data.prepare_record(original, Tokenizer())
    assert original == before and result['messages'] == original['messages']
    assert len(result['input_ids']) == len(result['loss_mask'])
    supervised = ''.join(chr(t) for t, m in zip(result['input_ids'], result['loss_mask']) if m and t != data.END_ID)
    assert supervised == 'Reasoning:\nORIGINAL_REASONING\n\nAnswer:\nThe result is 7.'
    assert result['loss_mask'][-1] == 1 and result['input_ids'][-1] == data.END_ID
    assert all(result['loss_mask'][s['start']:s['end']] for s in result['serialization']['assistant_spans'])


def test_tool_name_id_mapping_and_observation_mask():
    messages = [{'role': 'user', 'content': 'Inspect the source.'},
                {'role': 'assistant', 'content': '', 'reasoning_content': 'Read the file first.',
                 'tool_calls': [{'id': 'call_x', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{"path":"x.py"}'}}]},
                {'role': 'tool', 'tool_call_id': 'call_x', 'content': 'OBSERVATION_ONLY'},
                {'role': 'assistant', 'content': 'The file is empty.'}]
    normalized, pending = data.training_messages(messages)
    assert not pending
    assert normalized[1]['tool_calls'][0]['id'] == 'functions.read_file:0'
    assert normalized[2]['tool_call_id'] == 'functions.read_file:0'
    result = data.prepare_record(record(messages), Tokenizer())
    supervised = ''.join(chr(t) for t, m in zip(result['input_ids'], result['loss_mask']) if m and t != data.END_ID)
    assert 'OBSERVATION_ONLY' not in supervised and 'functions.read_file:0' in supervised


def test_native_calls_and_json_encoded_tool_schemas_preserve_semantics():
    messages = [{'role': 'user', 'content': 'Inspect.'},
                {'role': 'assistant', 'content': '', 'tool_calls': [
                    {'name': 'read_file', 'arguments': '{"path":"x"}', 'id': 'old'}]},
                {'role': 'tool', 'tool_call_id': 'old', 'content': 'empty'},
                {'role': 'assistant', 'content': 'The file is empty.'}]
    r = record(messages)
    schema = {'name': 'read_file', 'parameters': {'type': 'object'}}
    r['tools'] = [json.dumps(schema)]
    prepared = data.prepare_record(r, Tokenizer())
    assert prepared['structured_tool_calls'] == 1
    assert prepared['messages'] == messages and prepared['tools'] == r['tools']
    normalized, _ = data.training_messages(messages)
    assert normalized[1]['tool_calls'][0]['function'] == {'name': 'read_file', 'arguments': '{"path":"x"}'}
    assert data.training_tools(r['tools']) == [{'type': 'function', 'function': schema}]
    with pytest.raises(ValueError): data.training_tools(['not json'])


def test_reject_holdout_review_truncation_and_unanswered_calls():
    r = record([{'role': 'user', 'content': 'x'}, {'role': 'assistant', 'content': 'y'}])
    with pytest.raises(ValueError, match='exceeds_limit'):
        data.prepare_record(r, Tokenizer(), max_tokens=2)
    for field, value in [('split', 'test'), ('quality_tier', 'heuristic_candidate')]:
        bad = copy.deepcopy(r); bad['metadata'][field] = value
        with pytest.raises(ValueError): data.prepare_record(bad, Tokenizer())
    bad = record([{'role': 'user', 'content': 'x'}, {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'x', 'function': {'name': 'submit', 'arguments': '{}'}}]}])
    with pytest.raises(ValueError, match='completion receipt'): data.prepare_record(bad, Tokenizer())


def test_frozen_eval_counts_disjoint_prompts_and_no_overwrite(tmp_path):
    receipt = data.build_eval_suites(tmp_path)
    dev = [json.loads(line) for line in (tmp_path/'dev.jsonl').read_text().splitlines()]
    final = [json.loads(line) for line in (tmp_path/'final.jsonl').read_text().splitlines()]
    assert len(dev) == 48 and len(final) == 144
    assert not ({x['prompt_sha256'] for x in dev} & {x['prompt_sha256'] for x in final})
    assert receipt['model_results_seen'] is False
    with pytest.raises(FileExistsError): data.build_eval_suites(tmp_path)


def test_exact_oracles_capped_and_unsupported_completion_are_separate():
    case = data._operational_case('dev', 2)
    expected = case['oracle']['expected']
    result = data.grade_case(case, {'text': json.dumps(expected), 'finish_reason': 'length'})
    assert result['correct'] and result['capped'] and not result['unsupported_completion']
    bad = {**expected, 'status': 'complete'}
    result = data.grade_case(case, json.dumps(bad))
    assert not result['correct'] and result['unsupported_completion'] and not result['capped']
    control = data._control_case('dev', 0)
    assert data.grade_case(control, control['oracle']['expected'])['correct']
    assert not data.grade_case(control, 'Reasoning: ' + control['oracle']['expected'])['correct']


def test_interactive_oracle_requires_actual_current_patch_checks_and_finish():
    case = data._episode('test-repair', 'dev', random.Random(31))
    state = data.episode_initial_state(case)
    premature, _, done = data.episode_step(case, state, {'name': 'finish', 'arguments': {'complete': True}})
    assert done and data.grade_case(case, {'episode_state': premature})['unsupported_completion']
    correct = next(key for key in case['oracle']['patches'] if all(data._checks(case['oracle'], key)))
    for name, args in [('read_file', {}), ('apply_patch', {'patch_id': correct}), ('run_checks', {}), ('finish', {'complete': True})]:
        state, result, done = data.episode_step(case, state, {'name': name, 'arguments': args})
    graded = data.grade_case(case, {'episode_state': state})
    assert graded['correct'] and not graded['unsupported_completion']
    assert len(state['events']) == 4
    state = data.episode_initial_state(case)
    for name, args in [('read_file', {}), ('apply_patch', {'patch_id': correct}), ('run_checks', {}), ('apply_patch', {'patch_id': correct}), ('finish', {'complete': True})]:
        state, _, _ = data.episode_step(case, state, {'name': name, 'arguments': args})
    assert data.grade_case(case, {'episode_state': state})['unsupported_completion']


def test_exact_staged_q8_tokenizer_serialization_cpu():
    path = Path.home() / 'Models/Kimi-Linear-48B-A3B-Instruct-8bit'
    if not (path/'tiktoken.model').exists(): pytest.skip('exact local tokenizer artifact unavailable')
    tokenizer = data.load_tokenizer(path)
    r = record([{'role': 'user', 'content': 'Return the sum of 2 and 3.'},
                {'role': 'assistant', 'content': '5', 'reasoning_content': 'Adding the two integers gives five.'}])
    prepared = data.prepare_record(r, tokenizer, 512)
    assert prepared['input_ids'][-1] == 163586
    assert len(prepared['input_ids']) == len(prepared['loss_mask'])
    selected = [t for t, m in zip(prepared['input_ids'], prepared['loss_mask']) if m]
    assert 'Reasoning:' in tokenizer.decode(selected) and 'Answer:' in tokenizer.decode(selected)
