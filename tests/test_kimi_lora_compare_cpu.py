"""CPU receipt validation and paired-statistic oracles."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/research'))

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/research/kimi_lora_compare.py'
spec = importlib.util.spec_from_file_location('kimi_lora_compare', SCRIPT)
compare_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_module)


def fixture(tmp_path, outcomes=((False, True), (False, True), (False, True), (True, False)), split='dev'):
    case_path = tmp_path / 'cases.jsonl'
    cases = [{'id': f'case-{i}', 'domain': 'logic' if i < 2 else 'tools', 'split': split,
              'evaluation_only': True, 'mode': 'single_turn', 'max_tokens': 16,
              'messages': [{'role': 'user', 'content': f'Fixture{i}'}],
              'oracle': {'kind': 'exact_text', 'expected': 'yes'}} for i in range(len(outcomes))]
    case_path.write_text(''.join(json.dumps(case) + '\n' for case in cases))
    paths = []
    for index, arm in enumerate(('base', 'adapter')):
        folder = tmp_path / arm
        folder.mkdir()
        rows = [{'id': case['id'], 'domain': case['domain'], 'split': split,
                 'turns': [{'text': 'yes', 'tokens': [1], 'finish_reason': 'stop'}], 'protocol_errors': [],
                 'grade': {'correct': outcomes[i][index], 'format_ok': i != 0,
                           'capped': False, 'unsupported_completion': False, 'checks': {}}}
                for i, case in enumerate(cases)]
        result = folder / f'{arm}.jsonl'
        result.write_text(''.join(json.dumps(row) + '\n' for row in reversed(rows)))
        receipt = {'status': 'completed', 'checkpoint': 'fixture@abc', 'source_revision': 'source123',
                   'spec': {'artifact_name': 'fixture', 'base_fingerprint': 'abc',
                            'eval_cases': str(case_path), 'eval_split': split,
                            'execution_policy': {'mla': 'expanded'}},
                   'eval_cases_sha256': compare_module.sha(case_path),
                   'code_sha256': {key: 'a' * 64 for key in compare_module.EVALUATION_CODE},
                   'evaluations': {arm: {'cases': len(cases), 'correct': sum(row['grade']['correct'] for row in rows),
                                        'sha256': compare_module.sha(result)}}}
        path = folder / 'receipt.json'
        path.write_text(json.dumps(receipt))
        paths.append(path)
    return paths


def rewrite(path, fn):
    value = json.loads(path.read_text())
    fn(value)
    path.write_text(json.dumps(value))


def test_paired_case_ids_domains_metrics_and_reproducible_intervals(tmp_path):
    paths = fixture(tmp_path)
    a = compare_module.compare(*paths, samples=200, seed=7)
    b = compare_module.compare(*paths, samples=200, seed=7)
    assert a == b
    accuracy = a['overall']['metrics']['correct']
    assert accuracy['base_count'] == 1 and accuracy['adapter_count'] == 3
    assert accuracy['delta_rate'] == .5
    assert accuracy['exact_mcnemar_two_sided_p'] == .625
    assert accuracy['paired_case_bootstrap_95'][0] <= .5 <= accuracy['paired_case_bootstrap_95'][1]
    assert a['wins'] == ['case-0', 'case-1', 'case-2'] and a['losses'] == ['case-3']
    assert a['per_domain']['logic']['cases'] == 2
    assert a['overall']['metrics']['correct_and_format']['adapter_count'] == 2
    assert a['selection_eligible'] and not a['reasoning_capability_claim']


def test_exact_mcnemar_and_degenerate_interval_are_not_equivalence(tmp_path):
    assert compare_module.exact_mcnemar(0, 0) == 1.
    assert compare_module.exact_mcnemar(10, 0) == 2 / 1024
    assert compare_module.exact_mcnemar(0, 10) == 2 / 1024
    paths = fixture(tmp_path, outcomes=((True, True), (False, False)))
    report = compare_module.compare(*paths, samples=100)
    metric = report['overall']['metrics']['correct']
    assert metric['paired_case_bootstrap_95'] == [0., 0.]
    assert any('not proof of behavioral equivalence' in text for text in report['limitations'])


@pytest.mark.parametrize('change', ['policy', 'source', 'code', 'checkpoint'])
def test_mismatched_runtime_bindings_refused(tmp_path, change):
    paths = fixture(tmp_path)
    def mutate(receipt):
        if change == 'policy': receipt['spec']['execution_policy']['mla'] = 'auto'
        elif change == 'source': receipt['source_revision'] = 'other'
        elif change == 'code': receipt['code_sha256'][compare_module.EVALUATION_CODE[0]] = 'b' * 64
        else: receipt['checkpoint'] = 'other@abc'
    rewrite(paths[1], mutate)
    with pytest.raises(ValueError, match='differ'):
        compare_module.compare(*paths, samples=100)


def test_case_budget_change_and_unbound_payload_refused(tmp_path):
    paths = fixture(tmp_path)
    result = paths[1].parent / 'adapter.jsonl'
    result.write_text(result.read_text() + '\n')
    with pytest.raises(ValueError, match='payload'):
        compare_module.compare(*paths, samples=100)
    rewrite(paths[1], lambda receipt: receipt['evaluations']['adapter'].update(sha256=compare_module.sha(result)))
    cases = tmp_path / 'cases.jsonl'
    cases.write_text(cases.read_text().replace('"max_tokens": 16', '"max_tokens": 32'))
    with pytest.raises(ValueError, match='cases do not match'):
        compare_module.compare(*paths, samples=100)


def test_duplicate_or_missing_cases_refused_even_if_rehashed(tmp_path):
    paths = fixture(tmp_path)
    result = paths[1].parent / 'adapter.jsonl'
    lines = result.read_text().splitlines()
    result.write_text('\n'.join(lines[:-1] + [lines[0]]) + '\n')
    rewrite(paths[1], lambda receipt: receipt['evaluations']['adapter'].update(sha256=compare_module.sha(result)))
    with pytest.raises(ValueError, match='unique'):
        compare_module.compare(*paths, samples=100)


def test_final_comparison_is_not_selection_evidence_and_output_immutable(tmp_path):
    paths = fixture(tmp_path, split='final')
    report = compare_module.compare(*paths, samples=100)
    assert not report['selection_eligible']
    output = tmp_path / 'comparison.json'
    compare_module.write_report(output, report)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        compare_module.write_report(output, {})
    assert output.read_bytes() == before


def test_incomplete_round_is_not_eligible(tmp_path):
    paths = fixture(tmp_path)
    rewrite(paths[1], lambda receipt: receipt.update(status='starting'))
    with pytest.raises(ValueError, match='completed'):
        compare_module.compare(*paths, samples=100)


def bind_new_sampling_receipt(path, *, seed=17, replicate=0, temperature=.7):
    from kimi_lora_sampling import generation_profile, expected_case_receipt, SAMPLING_CODE
    receipt = json.loads(path.read_text())
    receipt['spec'].update(sampling_profile={'temperature': temperature}, sampling_seed=seed, replicate=replicate)
    receipt['generation_profile'] = generation_profile(receipt['spec'])
    receipt['actual_execution'] = {'policy': {'kda': 'kernel'}, 'environment': {'mode': 'fixed'},
                                   'profile': 'fixture-profile', 'cache_layout': 'fixture-layout',
                                   'sampling_contract': None}
    receipt['code_sha256'].update({key: 'd' * 64 for key in SAMPLING_CODE})
    arm = next(iter(receipt['evaluations']))
    result = path.parent / f'{arm}.jsonl'
    rows = [json.loads(line) for line in result.read_text().splitlines()]
    for row in rows:
        draws = sum(len(turn['tokens']) for turn in row['turns']) if temperature else 0
        row['sampling'] = expected_case_receipt(receipt['generation_profile'], row['id'], draws)
    result.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    receipt['evaluations'][arm]['sha256'] = compare_module.sha(result)
    path.write_text(json.dumps(receipt))


def test_matching_seeded_profiles_compare_with_actual_policy_binding(tmp_path):
    paths = fixture(tmp_path)
    for path in paths:bind_new_sampling_receipt(path)
    report = compare_module.compare(*paths, samples=100)
    generation = report['binding']['generation']
    assert generation['sampling'] == 'seeded_categorical'
    assert generation['profile']['top_p'] == .8 and generation['profile']['top_k'] == 20
    assert generation['seed_strategy']['suite_seed'] == 17
    assert report['binding']['actual_execution']['cache_layout'] == 'fixture-layout'


@pytest.mark.parametrize('kind', ['seed', 'replicate', 'temperature', 'law', 'environment', 'cache_layout', 'sampling_code'])
def test_sampled_profile_and_resolved_runtime_mismatches_refused(tmp_path, kind):
    paths = fixture(tmp_path)
    for path in paths:bind_new_sampling_receipt(path)
    if kind in ('seed', 'replicate', 'temperature'):
        bind_new_sampling_receipt(paths[1], **{kind: {'seed': 18, 'replicate': 1, 'temperature': .6}[kind]})
    else:
        def change(r):
            if kind == 'law': r['generation_profile']['sampling'] = 'invented_law'
            elif kind == 'sampling_code': r['code_sha256']['src/mlx2/runtime/sample_utils.py'] = 'e' * 64
            else:r['actual_execution'][kind] = 'different'
        rewrite(paths[1], change)
    with pytest.raises(ValueError, match='differ'):
        compare_module.compare(*paths, samples=100)


@pytest.mark.parametrize('field', ['derived_seed', 'rng_draws', 'seed_derivation'])
def test_rehashed_case_sampling_receipt_tampering_refused(tmp_path, field):
    paths = fixture(tmp_path)
    for path in paths:bind_new_sampling_receipt(path)
    result = paths[1].parent / 'adapter.jsonl';rows = [json.loads(line) for line in result.read_text().splitlines()]
    rows[0]['sampling'][field] = 'bad'
    result.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    rewrite(paths[1], lambda r: r['evaluations']['adapter'].update(sha256=compare_module.sha(result)))
    with pytest.raises(ValueError, match='sampling receipt'):
        compare_module.compare(*paths, samples=100)


def test_explicit_sampling_spec_without_receipt_is_not_historical_greedy(tmp_path):
    paths = fixture(tmp_path)
    rewrite(paths[1], lambda r: r['spec'].update(sampling_profile={'temperature': .7}, sampling_seed=1))
    with pytest.raises(ValueError, match='lacks bound'):
        compare_module.compare(*paths, samples=100)
