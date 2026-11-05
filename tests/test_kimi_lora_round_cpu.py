"""Host-only guards for the bounded Kimi LoRA experiment runner."""
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts/research'))

SPEC = importlib.util.spec_from_file_location('kimi_lora_round', Path(__file__).parents[1] / 'scripts/research/kimi_lora_round.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_original_and_nested_outputs_are_protected(tmp_path):
    spec = {'model': str(tmp_path / 'copy'), 'original_model': str(tmp_path / 'original'),
            'out': str(tmp_path / 'round'), 'mode': 'probe', 'data': 'fixture.jsonl'}
    runner.validate_spec(spec)
    with pytest.raises(ValueError, match='separate base copy'):
        runner.validate_spec({**spec, 'model': spec['original_model']})
    with pytest.raises(ValueError, match='outside both'):
        runner.validate_spec({**spec, 'out': str(tmp_path / 'copy' / 'training')})


def test_final_requires_candidate_selection(tmp_path):
    spec = {'model': str(tmp_path / 'copy'), 'original_model': str(tmp_path / 'original'),
            'out': str(tmp_path / 'round'), 'mode': 'evaluate', 'eval_split': 'final'}
    with pytest.raises(ValueError, match='frozen candidate'):
        runner.validate_spec(spec)


@pytest.mark.parametrize('change', [
    {'input_ids': [0]}, {'loss_mask': [0, 0]}, {'loss_mask': [1, 0]},
    {'loss_mask': [0, float('nan')]}, {'loss_mask': [0, .5]},
    {'input_ids': [0, 163840]}, {'input_ids': [0, True]}, {'id': ''},
])
def test_invalid_training_rows_rejected_before_model_load(change):
    with pytest.raises(ValueError):
        runner.validate_training_rows([{'id': 'source:1', 'split': 'train', 'input_ids': [0, 2], 'loss_mask': [0, 1], **change}])


def test_training_rows_have_causal_assistant_targets():
    runner.validate_training_rows([{'id': 'source:1', 'split': 'train', 'input_ids': [0, 2], 'loss_mask': [0, 1]}])
    with pytest.raises(ValueError, match='empty'):
        runner.validate_training_rows([])


@pytest.mark.parametrize('split', ['dev', 'final', None])
def test_held_out_training_rows_rejected(split):
    with pytest.raises(ValueError, match='training split'):
        runner.validate_training_rows([{'id': 'heldout', 'split': split, 'input_ids': [0, 2], 'loss_mask': [0, 1]}])


def test_evaluation_split_mixing_and_duplicate_ids_rejected():
    case = {'id': 'dev-1', 'split': 'dev', 'evaluation_only': True}
    runner.validate_evaluation_cases([case], 'dev')
    for cases in ([{**case, 'split': 'final'}], [{**case, 'evaluation_only': False}], [case, case]):
        with pytest.raises(ValueError):
            runner.validate_evaluation_cases(cases, 'dev')


def episode_fixture():
    sys.path.insert(0, str(Path(runner.__file__).parent))
    from kimi_lora_data import _episode, _checks
    case = _episode('dev-fixture', 'dev', random.Random(13))
    patch = next(key for key in case['oracle']['patches'] if all(_checks(case['oracle'], key)))
    return case, patch


def call(name, arguments=None, index=0):
    return ('<|tool_call_begin|>functions.' + name + ':' + str(index)
            + '<|tool_call_argument_begin|>' + json.dumps(arguments or {}) + '<|tool_call_end|>')


def episode_run(tmp_path, monkeypatch, answers, *, max_turns=8):
    case, patch = episode_fixture()
    case['max_turns'] = max_turns
    outputs = iter(answers(patch))
    def fake_generate(*args, **kwargs):
        return {'text': next(outputs), 'finish_reason': 'stop', 'tokens': []}
    monkeypatch.setattr(runner, 'generate', fake_generate)
    path = tmp_path / 'eval.jsonl'
    result = runner.evaluate(None, [case], path, guard=lambda: None)
    return result, json.loads(path.read_text())


def successful_batch(patch):
    return (call('read_file', index=0) + call('apply_patch', {'patch_id': patch}, 1)
            + call('run_checks', index=2) + call('finish', {'complete': True}, 3))


def test_complete_tool_episode_requires_plain_final_turn(tmp_path, monkeypatch):
    result, row = episode_run(tmp_path, monkeypatch, lambda patch: [successful_batch(patch), 'The patch is fixed and verified.'])
    assert result['correct'] == 1 and result['protocol_failed'] == 0
    assert row['grade']['final_assistant_received']
    calls = [c['id'] for m in row['messages'] for c in m.get('tool_calls', [])]
    results = [m['tool_call_id'] for m in row['messages'] if m['role'] == 'tool']
    assert calls == results


@pytest.mark.parametrize('after', ['extra_call', 'empty', 'turn_budget'])
def test_environment_success_does_not_hide_invalid_terminal_protocol(tmp_path, monkeypatch, after):
    def answers(patch):
        return [successful_batch(patch), call('read_file') if after == 'extra_call' else '']
    result, row = episode_run(tmp_path, monkeypatch, answers, max_turns=1 if after == 'turn_budget' else 8)
    assert result['correct'] == 0 and result['protocol_failed'] == 1
    assert row['grade']['environment_success'] is True
    assert row['grade']['format_ok'] is False


@pytest.mark.parametrize('bad', ['malformed', 'after_finish', 'unknown_tool', 'invalid_schema', 'duplicate_id'])
def test_invalid_batch_has_no_partial_environment_effect(tmp_path, monkeypatch, bad):
    def answers(patch):
        first = successful_batch(patch)
        if bad == 'malformed':
            return [first + '<|tool_call_begin|>functions.read_file:9']
        if bad == 'after_finish':
            return [first + call('read_file', index=9)]
        if bad == 'unknown_tool':
            return [call('read_file') + call('unknown', index=1)]
        if bad == 'invalid_schema':
            return [call('read_file') + call('finish', {'complete': 'true'}, 1)]
        return [call('read_file') + call('read_file')]
    result, row = episode_run(tmp_path, monkeypatch, answers)
    assert result['correct'] == 0 and result['protocol_failed'] == 1
    assert row['episode_state']['calls'] == 0


def test_final_candidate_binding_rejects_changed_bytes_and_mount(tmp_path):
    import numpy as np
    from mlx2.experimental.weight_space.programs import create_program, export_program
    from mlx2.runtime.lora import read_lora_config, inspect_lora_program, lora_fingerprint
    checkpoint = 'fixture@' + 'a' * 64
    path = tmp_path / 'adapter'
    program = create_program({'model.layers.0.attention.q_proj': (np.ones((4, 2), np.float32), np.ones((2, 4), np.float32))},
                             checkpoint=checkpoint, source_revision='b' * 40,
                             construction='synthetic_test', provenance={'test': True})
    export_program(program, path)
    _, config = read_lora_config(path)
    payload = (path / 'adapters.safetensors').read_bytes()
    manifest = inspect_lora_program(path, config, base_checkpoint=checkpoint, payload=payload)
    fingerprint = lora_fingerprint(payload, config['rank'], config['scale'], config['keys'], manifest)
    dev = tmp_path / 'dev.json'
    dev.write_text(json.dumps({'status': 'completed', 'adapter_fingerprint': fingerprint, 'checkpoint': checkpoint,
                               'spec': {'eval_split': 'dev'}, 'evaluations': {'adapter': {'cases': 4, 'correct': 4}}}))
    selected_path = tmp_path / 'selected.json'
    selected = {'schema': 'mlx2.kimi-lora-candidate-selection.v1', 'adapter_path': str(path),
                'adapter_fingerprint': fingerprint, 'checkpoint': checkpoint, 'program_id': program.program_id,
                'source_revision': 'b' * 40, 'development_receipt': {'path': str(dev), 'sha256': runner.sha(dev)}}
    selected_path.write_text(json.dumps(selected))
    spec = {'eval_split': 'final', 'artifact_name': 'fixture', 'base_fingerprint': 'a' * 64,
            'adapter': str(path), 'selected_candidate_receipt': str(selected_path),
            'selected_candidate_receipt_sha256': runner.sha(selected_path)}
    got = runner.candidate_selection(spec)
    runner.verify_selected_mount(got, SimpleNamespace(fingerprint=fingerprint))
    with pytest.raises(ValueError, match='mounted adapter changed'):
        runner.verify_selected_mount(got, SimpleNamespace(fingerprint='changed'))
    dev.write_text(dev.read_text() + '\n')
    with pytest.raises(ValueError, match='development receipt hash'):
        runner.candidate_selection(spec)
    selected_path.write_text(selected_path.read_text() + '\n')
    with pytest.raises(ValueError, match='selection receipt hash'):
        runner.candidate_selection(spec)


def toy_adapter():
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    class Model:
        def __init__(self): self.caches = []; self.calls = 0
        def make_cache(self):
            cache = []; self.caches.append(cache); return cache
        def __call__(self, ids, *, cache):
            self.calls += 1
            cache.append(ids.shape[-1])
            # Nondegenerate deterministic logits, no target model weights.
            logits = mx.array([float(i % 7) / 3 for i in range(64)])
            return mx.broadcast_to(logits, (1, ids.shape[-1], 64))
    return SimpleNamespace(model=Model(), prompt_tokens=lambda request: [7, 8, 9],
        tokenizer=SimpleNamespace(eos_token_ids={63}, decode=lambda ids: ','.join(map(str, ids))))


def test_generate_legacy_greedy_exact_and_fresh_cache():
    import mlx.core as mx
    from kimi_lora_sampling import CaseSampler, GREEDY_PROFILE
    mx.set_default_device(mx.cpu)
    adapter = toy_adapter()
    legacy = runner.generate(adapter, [{'role': 'user', 'content': 'fixture'}], max_tokens=5, prefill_chunk=2)
    explicit = runner.generate(adapter, [{'role': 'user', 'content': 'fixture'}], max_tokens=5, prefill_chunk=2,
                              sampler=CaseSampler(GREEDY_PROFILE, suite_seed=3, case_id='fixture'))
    # Tied maxima: raw argmax takes first index6, exactly as original runner.
    assert legacy['tokens'] == explicit['tokens'] == [6] * 5
    assert legacy['text'] == explicit['text'] == '6,6,6,6,6'
    assert legacy['finish_reason'] == explicit['finish_reason'] == 'length'
    assert legacy['prompt_tokens'] == explicit['prompt_tokens'] == 3
    assert len(adapter.model.caches) == 2 and adapter.model.caches[0] is not adapter.model.caches[1]
    assert adapter.model.calls == 14  # Two prefill chunks + five decode calls, per run.


def test_generate_seeded_replay_preserves_global_rng_and_stops_on_eos():
    import mlx.core as mx
    import numpy as np
    from kimi_lora_sampling import CaseSampler
    mx.set_default_device(mx.cpu)
    mx.random.seed(17)
    before = [np.array(key) for key in mx.random.state]
    adapter = toy_adapter()
    a = runner.generate(adapter, [], max_tokens=20, sampler=CaseSampler(suite_seed=7, case_id='same'))
    b = runner.generate(adapter, [], max_tokens=20, sampler=CaseSampler(suite_seed=7, case_id='same'))
    assert a['tokens'] == b['tokens']
    assert all(np.array_equal(x, np.array(y)) for x, y in zip(before, mx.random.state))
    adapter.tokenizer.eos_token_ids = {6}
    stopped = runner.generate(adapter, [], max_tokens=20)
    assert stopped['tokens'] == [6] and stopped['finish_reason'] == 'stop'


def test_one_seeded_sampler_per_interactive_case_across_turns(tmp_path, monkeypatch):
    import mlx.core as mx
    from kimi_lora_sampling import expected_case_receipt, generation_profile
    mx.set_default_device(mx.cpu)
    case, patch = episode_fixture()
    answers = iter([successful_batch(patch), 'The patch is fixed and verified.'])
    samplers = []
    def fake_generate(*args, **kwargs):
        sampler = kwargs['sampler'];samplers.append(sampler)
        token = sampler.draw(mx.array([float(i) / 20 for i in range(64)]))
        return {'text': next(answers), 'finish_reason': 'stop', 'tokens': [token]}
    monkeypatch.setattr(runner, 'generate', fake_generate)
    spec = {'sampling_profile': {'temperature': .7}, 'sampling_seed': 17, 'replicate': 2}
    path = tmp_path / 'sampled.jsonl'
    runner.evaluate(None, [case], path, guard=lambda: None, sampling_spec=spec)
    row = json.loads(path.read_text())
    assert samplers[0] is samplers[1]
    assert row['sampling'] == expected_case_receipt(generation_profile(spec), case['id'], 2)


def test_actual_execution_is_resolved_snapshot_not_mutable_requested_policy():
    adapter = SimpleNamespace(policy={'kda': 'kernel'}, environment={'numerics': 'value'}, profile='fixture',
                              descriptor=SimpleNamespace(cache_layout='resolved-layout', metadata={}))
    result = runner.actual_execution(adapter)
    adapter.policy['kda'] = 'changed'
    assert result == {'policy': {'kda': 'kernel'}, 'environment': {'numerics': 'value'}, 'profile': 'fixture',
                      'cache_layout': 'resolved-layout', 'sampling_contract': None}


@pytest.mark.parametrize('extra', [
    {'sampling_seed': 7}, {'replicate': 1}, {'sampling_profile': {}},
    {'sampling_profile': None, 'sampling_seed': 1},
    {'sampling_profile': {}, 'sampling_seed': True},
])
def test_sampling_spec_rejected_before_model_load(tmp_path, extra):
    spec = {'model': str(tmp_path / 'copy'), 'original_model': str(tmp_path / 'original'),
            'out': str(tmp_path / 'out'), 'mode': 'baseline', **extra}
    with pytest.raises(ValueError):runner.validate_spec(spec)


def test_minimum_row_length_is_enforced_when_requested():
    row = {'id': 'source:1', 'split': 'train', 'input_ids': list(range(9)), 'loss_mask': [0] + [1] * 8}
    runner.validate_training_rows([row])
    with pytest.raises(ValueError):
        runner.validate_training_rows([row], min_tokens=10)
