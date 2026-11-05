#!/usr/bin/env python3
"""Receipt-bound paired evaluation summaries; CPU-only and no model imports.

Inference is descriptive for this authored suite. An output rationale is not
evidence of a faithful internal reasoning process or broad reasoning gains.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile

METRICS = ('correct', 'format_ok', 'correct_and_format', 'capped',
           'unsupported_completion', 'protocol_failed')
EVALUATION_CODE = ('scripts/research/kimi_lora_round.py', 'scripts/research/kimi_lora_data.py',
                   'src/mlx2/adapters/kimi_linear.py', 'src/mlx2/runtime/models/kimi_linear.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def read_rows(path):
    with Path(path).open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    keys = [row.get('id') for row in rows]
    if not rows or any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError('case IDs must be nonempty and unique')
    return rows


def metrics(row):
    grade = row['grade']
    for key in ('correct', 'format_ok', 'capped', 'unsupported_completion'):
        if type(grade.get(key)) is not bool:
            raise ValueError('evaluation requires explicit boolean grade: ' + key)
    return {key: grade[key] for key in ('correct', 'format_ok', 'capped', 'unsupported_completion')} | {
        'correct_and_format': grade['correct'] and grade['format_ok'],
        'protocol_failed': bool(row.get('protocol_errors', []))}


def load_arm(receipt_path, arm):
    if arm not in {'base', 'adapter'}:
        raise ValueError('arm must be base or adapter')
    path = Path(receipt_path).resolve()
    receipt = json.loads(path.read_text())
    if receipt.get('status') != 'completed':
        raise ValueError('comparison requires completed round receipts')
    from kimi_lora_sampling import generation_profile, expected_case_receipt, SAMPLING_CODE
    spec = receipt['spec']
    generation = generation_profile(spec)
    explicit_receipt = 'generation_profile' in receipt
    if explicit_receipt and receipt['generation_profile'] != generation:
        raise ValueError('recorded generation profile differs from actual spec')
    if not explicit_receipt and 'sampling_profile' in spec:
        raise ValueError('sampled receipt lacks bound generation profile')
    result_path = path.parent / f'{arm}.jsonl'
    summary = receipt.get('evaluations', {}).get(arm)
    if not summary or not result_path.is_file() or sha(result_path) != summary.get('sha256'):
        raise ValueError('evaluation payload does not match its receipt hash')
    case_path = Path(spec['eval_cases'])
    if sha(case_path) != receipt.get('eval_cases_sha256'):
        raise ValueError('evaluation cases do not match their receipt hash')
    cases = read_rows(case_path)
    if spec.get('eval_limit') is not None:
        if type(spec['eval_limit']) is not int or spec['eval_limit'] < 1:
            raise ValueError('invalid evaluation limit')
        cases = cases[:spec['eval_limit']]
    cases = {case['id']: case for case in cases}
    rows = {row['id']: row for row in read_rows(result_path)}
    if set(rows) != set(cases) or len(rows) != summary.get('cases'):
        raise ValueError('evaluation case set/count mismatch')
    split = spec.get('eval_split', 'dev')
    for identity, row in rows.items():
        case = cases[identity]
        if (case.get('evaluation_only') is not True or case.get('split') != split
                or row.get('split') != split or row.get('domain') != case.get('domain')):
            raise ValueError('evaluation split/domain/eligibility mismatch')
        turns = row.get('turns')
        if not isinstance(turns, list) or not turns:
            raise ValueError('evaluation requires recorded turns')
        max_turns = int(case.get('max_turns', 8)) if case.get('mode') == 'interactive' else 1
        if len(turns) > max_turns or any(len(turn.get('tokens', [])) > int(case.get('max_tokens', 384)) for turn in turns):
            raise ValueError('recorded generation exceeds case budget')
        if explicit_receipt:
            draws = sum(len(turn.get('tokens', [])) for turn in turns) if generation['sampling'] == 'seeded_categorical' else 0
            if row.get('sampling') != expected_case_receipt(generation, identity, draws):
                raise ValueError('case sampling receipt differs from profile, seed or draw count')
        metrics(row)
    for key in ('correct', 'format_ok', 'capped', 'unsupported_completion', 'protocol_failed'):
        observed = sum(metrics(row)[key] for row in rows.values())
        if key in summary and summary[key] != observed:
            raise ValueError('receipt summary does not match recorded grades: ' + key)
    code = receipt.get('code_sha256', {})
    evaluation_code = (*EVALUATION_CODE, *SAMPLING_CODE) if explicit_receipt else EVALUATION_CODE
    if any(not isinstance(code.get(key), str) or len(code[key]) != 64 for key in evaluation_code):
        raise ValueError('receipt lacks exact evaluation code bindings')
    actual = receipt.get('evaluation_execution', {}).get(arm, receipt.get('actual_execution'))
    if explicit_receipt and (not isinstance(actual, dict) or not {'policy', 'environment', 'profile', 'cache_layout', 'sampling_contract'} <= set(actual)):
        raise ValueError('receipt lacks actual execution policy/environment/profile/cache layout')
    checkpoint = receipt.get('checkpoint')
    if checkpoint != spec['artifact_name'] + '@' + spec['base_fingerprint']:
        raise ValueError('receipt checkpoint differs from actual spec')
    return {'receipt': receipt, 'receipt_path': str(path), 'receipt_sha256': sha(path),
            'result_path': str(result_path), 'result_sha256': sha(result_path), 'arm': arm,
            'rows': rows, 'cases': cases,
            'binding': {'checkpoint': checkpoint, 'source_revision': receipt['source_revision'],
                        'evaluation_code_sha256': {key: code[key] for key in evaluation_code},
                        'eval_cases_sha256': receipt['eval_cases_sha256'], 'eval_split': split,
                        'execution_policy': spec.get('execution_policy'),
                        'actual_execution': actual, 'runtime': receipt.get('runtime'),
                        'case_sha256': hashlib.sha256(canonical(cases).encode()).hexdigest(),
                        'generation': generation}}


def exact_mcnemar(wins, losses):
    """Two-sided exact conditional binomial test on discordant paired cases."""
    n = wins + losses
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n)


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def ci(draws):
    return [percentile(draws, .025), percentile(draws, .975)]


def paired_summary(pairs, *, samples, seed, clusters=True):
    """Paired case bootstrap; cluster draws retain every case in each domain."""
    rng = random.Random(seed)
    domains = defaultdict(list)
    for case, base, adapter in pairs:
        domains[case['domain']].append((metrics(base), metrics(adapter)))
    names = sorted(domains)
    n = len(pairs)
    vectors = {key: [int(metrics(b)[key]) - int(metrics(a)[key]) for _, a, b in pairs] for key in METRICS}
    case_draws = {key: [] for key in METRICS}
    cluster_draws = {key: [] for key in METRICS}
    macro_draws = {key: [] for key in METRICS}
    totals = {key: [(sum(int(b[key]) - int(a[key]) for a, b in domains[name]), len(domains[name]))
                    for name in names] for key in METRICS}
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        for key in METRICS:
            case_draws[key].append(sum(vectors[key][i] for i in indices) / n)
        if clusters:
            selected = [rng.randrange(len(names)) for _ in names]
            for key in METRICS:
                picked = [totals[key][i] for i in selected]
                cluster_draws[key].append(sum(x for x, _ in picked) / sum(count for _, count in picked))
                macro_draws[key].append(sum(x / count for x, count in picked) / len(picked))
    result = {'cases': n, 'domains': len(names), 'metrics': {}}
    for key in METRICS:
        base = sum(metrics(a)[key] for _, a, _ in pairs)
        adapter = sum(metrics(b)[key] for _, _, b in pairs)
        delta = vectors[key]
        wins, losses = delta.count(1), delta.count(-1)
        metric = {'base_count': base, 'adapter_count': adapter, 'base_rate': base / n, 'adapter_rate': adapter / n,
                  'delta_rate': (adapter - base) / n, 'false_to_true': wins, 'true_to_false': losses,
                  'paired_case_bootstrap_95': ci(case_draws[key]), 'exact_mcnemar_two_sided_p': exact_mcnemar(wins, losses)}
        if clusters:
            metric.update(domain_cluster_bootstrap_95=ci(cluster_draws[key]),
                          equal_domain_mean_delta=sum(x / count for x, count in totals[key]) / len(names),
                          equal_domain_cluster_bootstrap_95=ci(macro_draws[key]))
        result['metrics'][key] = metric
    return result


def compare(base_receipt, adapter_receipt, *, base_arm='base', adapter_arm='adapter', samples=10000, seed=1371701):
    if type(samples) is not int or not 100 <= samples <= 100000:
        raise ValueError('bootstrap samples must be between100 and100000')
    base, adapter = load_arm(base_receipt, base_arm), load_arm(adapter_receipt, adapter_arm)
    if base['binding'] != adapter['binding'] or set(base['rows']) != set(adapter['rows']):
        raise ValueError('paired arms differ in source, checkpoint, cases, budgets, split, sampling or execution policy')
    pairs = [(base['cases'][key], base['rows'][key], adapter['rows'][key]) for key in sorted(base['rows'])]
    by_domain = defaultdict(list)
    details = []
    for case, left, right in pairs:
        by_domain[case['domain']].append((case, left, right))
        a, b = metrics(left), metrics(right)
        details.append({'id': case['id'], 'domain': case['domain'], 'base': a, 'adapter': b,
                        'correctness_result': 'win' if b['correct'] and not a['correct'] else
                            'loss' if a['correct'] and not b['correct'] else 'tie',
                        'generated_tokens_identical': [t['tokens'] for t in left['turns']] == [t['tokens'] for t in right['turns']],
                        'base_checks': left['grade'].get('checks'), 'adapter_checks': right['grade'].get('checks')})
    bindings = {key: value for key, value in base['binding'].items()}
    return {'schema': 'mlx2.kimi-lora-paired-comparison.v1', 'binding': bindings,
            'inputs': {name: {key: arm[key] for key in ('receipt_path', 'receipt_sha256', 'result_path', 'result_sha256', 'arm')}
                       for name, arm in (('base', base), ('adapter', adapter))},
            'selection_eligible': bindings['eval_split'] == 'dev', 'reasoning_capability_claim': False,
            'bootstrap': {'samples': samples, 'seed': seed, 'interval': 'percentile95',
                          'case_unit': 'paired case IDs', 'cluster_unit': 'whole domains resampled with replacement'},
            'overall': paired_summary(pairs, samples=samples, seed=seed),
            'per_domain': {name: paired_summary(group, samples=samples, seed=seed + index + 1, clusters=False)
                           for index, (name, group) in enumerate(sorted(by_domain.items()))},
            'wins': [row['id'] for row in details if row['correctness_result'] == 'win'],
            'losses': [row['id'] for row in details if row['correctness_result'] == 'loss'], 'paired_cases': details,
            'limitations': [
                'Authored small suite: estimates describe these cases; they do not establish broad reasoning gains.',
                'Oracle correctness is preserved as recorded. Numerically correct prose may fail a requested JSON answer; format is reported separately.',
                'Ambiguous oracle wording, including cause_proven judgments, requires separate adjudication and a new suite version, not post-hoc score changes.',
                'Rationale text and answer accuracy do not establish faithful internal reasoning or a learned search procedure.',
                'Case bootstrap assumes exchangeable cases; domain-cluster intervals use few heterogeneous clusters and are exploratory.',
                'If all paired outcomes agree, bootstrap intervals collapse to zero; this is not proof of behavioral equivalence.',
                'McNemar p values are unadjusted descriptive tests; repeated development tuning and multiple domains/metrics invalidate confirmatory selection claims.',
                'Negative deltas are favorable for capped, unsupported_completion and protocol_failed; positive deltas are favorable for accuracy and format.',
                'Candidate selection must use development cases only. Final held-out comparisons cannot select a candidate.']}


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        # link publishes the complete file atomically and refuses all existing outputs.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-receipt', type=Path, required=True)
    parser.add_argument('--adapter-receipt', type=Path, required=True)
    parser.add_argument('--base-arm', choices=('base', 'adapter'), default='base')
    parser.add_argument('--adapter-arm', choices=('base', 'adapter'), default='adapter')
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=1371701)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.base_receipt, args.adapter_receipt, base_arm=args.base_arm,
                     adapter_arm=args.adapter_arm, samples=args.samples, seed=args.seed)
    report['analysis_code_sha256'] = sha(__file__)
    write_report(args.output, report)
    print(json.dumps({'output': str(args.output), 'sha256': sha(args.output), 'correctness': report['overall']['metrics']['correct'],
                      'wins': report['wins'], 'losses': report['losses']}))
