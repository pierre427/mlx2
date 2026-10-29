# SPDX-License-Identifier: MIT
"""Resumable serial DS4.1 diagnostic arms using the existing owned GPU runner.

Preparation and assessment are CPU only. ``--run`` explicitly launches the
local harness, whose CPG/paired locks, frozen source, smoke and finally cleanup
remain authoritative. This wrapper neither runs preflight nor qualifies a
route or selects production defaults. Provenance: ds41-options-campaign.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

SCHEMA = 'mlx2.ds41-options-campaign.v1'
SETTING_ENV = {'prompt_tokens':'MLX2_BENCH_PROMPT_TOKENS','prefill_step':'MLX2_BENCH_PREFILL_STEP','apc_bytes':'MLX2_BENCH_APC_BYTES','context_cap':'MLX2_BENCH_CONTEXT_CAP','read_workers':'MLX2_BENCH_READ_WORKERS','expert_bytes':'MLX2_EXPERT_BUDGET_BYTES','preheat':'MLX2_BENCH_PREHEAT','persist':'MLX2_BENCH_PERSIST'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_arm(arm):
    if not isinstance(arm, dict) or not re.fullmatch('[a-z0-9][a-z0-9-]{0,70}', arm.get('id', '')):
        raise ValueError('Each arm needs a stable lowercase id of at most71 characters')
    if arm.get('apc_mode') not in ('exact', 'disabled'):
        raise ValueError('Declare apc_mode exact or disabled explicitly')
    if arm['apc_mode'] == 'disabled' and not arm.get('apc_reason'):
        raise ValueError('Disabled APC requires an explicit profile reason')
    if not isinstance(arm.get('policy', {}), dict):
        raise ValueError('Arm policy must be an object')
    if type(arm.get('expert_math', 'reference')) is not str:
        raise ValueError('Expert math must be a policy string')
    # Expert residency keys reach run.py through MLX2_BENCH_EXPERT_RESIDENCY;
    # they are validated again by the Source LFU policy parser at load.
    if not isinstance(arm.get('expert_residency', {}), dict):
        raise ValueError('Arm expert_residency must be an object')
    if 'hotset' in arm and (type(arm['hotset']) is not str or not Path(arm['hotset']).is_absolute()):
        raise ValueError('Arm hotset must be an absolute hot-map path')
    if arm.get('token_parity', 'required') not in ('required', 'informational'):
        raise ValueError('token_parity must be required or informational')
    if (arm.get('expert_residency', {}).get('residency') == 'dynamic_precision_v1'
            and (arm.get('token_parity') != 'informational' or arm['apc_mode'] != 'disabled')):
        raise ValueError('dynamic_precision_v1 arms are approximate: declare token_parity informational and apc_mode disabled')
    if 'expected_apc_cached' in arm and (type(arm['expected_apc_cached']) is not int
            or arm['expected_apc_cached'] < 0 or arm['apc_mode'] != 'exact'):
        raise ValueError('expected_apc_cached is a nonnegative int for an exact-APC arm')
    if 'advanced_case' in arm:
        sys.path.insert(0, str(Path(__file__).parent))
        from ds41_modality_options import case_by_id
        case = case_by_id(arm['advanced_case'])
        if any(arm.get('policy', {}).get(k) != v for k, v in case['policy'].items()):
            raise ValueError('Advanced arm policy must carry its case policy')
    result = dict(arm)
    result['policy'] = dict(arm.get('policy', {}))
    if 'deepseek_source_expert_math' in result['policy']:
        declared = result['policy'].pop('deepseek_source_expert_math')
        if 'expert_math' in arm and arm['expert_math'] != declared:
            raise ValueError('Conflicting expert-math declarations')
        result['expert_math'] = declared
    return result


def plan():
    # The existing research hooks require max_tokens on EVERY step.
    return [dict(label=label, mode='control', clear_apc=clear, max_tokens=32)
            for label, clear in [('cold', True), ('warm', True), ('apc_warm', False)]]


def _flatten(value, prefix=''):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = prefix + '.' + str(key) if prefix else str(key)
            result.update(_flatten(item, name))
    elif type(value) in (int, float):
        result[prefix] = value
    return result


def observations(row):
    before, after = (_flatten(row.get('status_' + arm, {})) for arm in ('before', 'after'))
    return {key: value - before[key] for key, value in after.items()
            if key in before and ('scheduled_' in key or 'counters' in key)
            and value != before[key]}


def counting(text):
    if not isinstance(text, str) or not re.fullmatch(r'[\s\d.,)\-]+', text):
        return False
    values = [int(x) for x in re.findall(r'\d+', text)]
    return len(values) >= 5 and values == list(range(1, len(values) + 1))


def assess_data(spec, result, release, arm):
    """Profile-aware correctness gates; a declared disabled APC is not a hit."""
    arm = validate_arm(arm)
    gates = {
        'completed': result.get('status') == 'completed' and result.get('passed') is True,
        'ownership_released': release.get('owned_release_verified', release.get('owners_absent')) is True,
        'source_unchanged': release.get('source_unchanged') is True,
        'checkpoint_named': bool(re.fullmatch(r'.+@[0-9a-f]{64}', spec.get('checkpoint', ''))),
        'source_revision_named': bool(re.fullmatch('[0-9a-f]{40}', spec.get('source_revision', ''))),
        'checkpoint_matches_result': result.get('checkpoint') == spec.get('checkpoint'),
        'source_revision_matches_result': result.get('source', {}).get('commit') == spec.get('source_revision'),
        'source_snapshot_matches_result': bool(spec.get('source_snapshot_sha256')) and result.get('source', {}).get('source_snapshot_sha256') == spec['source_snapshot_sha256'],
        'policy_bound': spec.get('extra_source_policy', {}) == arm.get('policy', {}),
        'math_bound': spec.get('source_expert_math', 'reference') == arm.get('expert_math', 'reference'),
        'ordinary_reference': result.get('ordinary_reference', {}).get('passed') is True,
    }
    if arm.get('advanced_case'):
        return assess_advanced(spec, result, arm, gates)
    requests = result.get('requests', {})
    ids = {}
    cells = {}
    for label, expected_cached in [('cold', 0), ('warm', 0),
                                   ('apc_warm', arm.get('expected_apc_cached', spec.get('prompt_tokens',100)-1)
                                    if arm['apc_mode'] == 'exact' else 0)]:
        row = requests.get(label, {})
        response = row.get('response', {})
        receipt = response.get('receipt', {})
        trace = response.get('token_trace', {})
        tokens = trace.get('token_ids')
        ids[label] = tokens
        checks = {
            'present': bool(row),
            'functional_counting': counting(response.get('text')),
            'ordinary_route': receipt.get('route') == 'ordinary' and bool(receipt.get('route_receipt')),
            'prompt_length': receipt.get('prompt_tokens') == spec.get('prompt_tokens',100),
            'output_length': receipt.get('completion_tokens') == 32,
            'cache_frontier': receipt.get('cached_tokens') == expected_cached,
            'token_trace': isinstance(tokens, list) and len(tokens) == 32,
            'physical_width_one': bool(trace.get('execution_widths')) and set(trace['execution_widths']) == {1},
        }
        if arm['apc_mode'] == 'exact':
            checks['APCv2_engine'] = receipt.get('cache') == 'apcv2'
        gates.update({label + '_' + key: value for key, value in checks.items()})
        elapsed, ttft = receipt.get('elapsed_seconds'), receipt.get('ttft_seconds')
        seconds = elapsed - ttft if type(elapsed) in (int, float) and type(ttft) in (int, float) else None
        cells[label] = dict(text=response.get('text'), cached_tokens=receipt.get('cached_tokens'),
            TTFT_seconds=ttft, elapsed_seconds=elapsed,
            decode_tokens_per_second=31 / seconds if seconds is not None and seconds > 0 else None,
            scheduled_counter_deltas=observations(row),
            checkpoint_resources=receipt.get('checkpoint_resources'),
            environment_before=row.get('environment_before'), environment_after=row.get('environment_after'))
    parity = {'ordinary_repeat_token_parity': ids['cold'] is not None and ids['cold'] == ids['warm'],
              'followup_token_parity': ids['cold'] is not None and ids['cold'] == ids['apc_warm']}
    if arm.get('token_parity', 'required') == 'required':
        gates.update(parity)
    expected_residency = arm.get('expert_residency', {})
    if expected_residency:
        recorded = spec.get('expert_policy', {})
        gates['expert_residency_bound'] = all(recorded.get(k) == v for k, v in expected_residency.items())
    return dict(schema=SCHEMA, arm=arm, checkpoint=spec.get('checkpoint'),
        source_revision=spec.get('source_revision'), source_snapshot_sha256=spec.get('source_snapshot_sha256'),
        gates=gates, passed=all(gates.values()), failed_gates=[k for k, v in gates.items() if not v],
        cells=cells, selected_policy=arm.get('policy', {}),
        expert_residency_policy=arm.get('expert_residency', {}), informational_token_parity=parity,
        observed_scope='Positive scheduled counter deltas prove dispatch scheduling, not GPU completion.',
        apc_mode=arm['apc_mode'], apc_reason=arm.get('apc_reason'),
        expert_residency='Inspect resource receipts and miss/read/upload deltas; this wrapper does not pin allocations.',
        qualification=False, performance_qualified=False, repetitions=1, thermal_controlled=False)


def assess_advanced(spec, result, arm, gates):
    """Advanced (draft/baseline) arms: the case's own oracles plus timings."""
    advanced = result.get('advanced') or {}
    gates['advanced_case_bound'] = (spec.get('advanced_case') == arm['advanced_case']
                                    and advanced.get('case', {}).get('id') == arm['advanced_case'])
    gates['advanced_passed'] = advanced.get('passed') is True
    cells = {}
    for row in advanced.get('rows', []):
        receipt = row['response'].get('receipt', {})
        elapsed, ttft = receipt.get('elapsed_seconds'), receipt.get('ttft_seconds')
        produced = receipt.get('completion_tokens')
        seconds = elapsed - ttft if type(elapsed) in (int, float) and type(ttft) in (int, float) else None
        speculation = receipt.get('speculation') or {}
        cells[row['label']] = dict(text=row['response'].get('text'), completion_tokens=produced,
            TTFT_seconds=ttft, elapsed_seconds=elapsed,
            decode_tokens_per_second=(produced - 1) / seconds if seconds and produced else None,
            proposed=speculation.get('proposed'), accepted=speculation.get('accepted'),
            external_rounds=speculation.get('external_rounds'),
            oracle_passed=advanced.get('assessment', {}).get('gates', {}).get(row['label'] + '_quality'))
    return dict(schema=SCHEMA, arm=arm, checkpoint=spec.get('checkpoint'),
        source_revision=spec.get('source_revision'), source_snapshot_sha256=spec.get('source_snapshot_sha256'),
        gates=gates, passed=all(gates.values()), failed_gates=[k for k, v in gates.items() if not v],
        cells=cells, selected_policy=arm.get('policy', {}),
        draft_observations=advanced.get('assessment', {}).get('draft_observations'),
        apc_mode=arm['apc_mode'], apc_reason=arm.get('apc_reason'),
        qualification=False, performance_qualified=False, repetitions=1, thermal_controlled=False)


def assess_unit(unit, arm):
    unit = Path(unit)
    from mlx2.desktop_campaign_release import release_evidence
    return assess_data(read(unit/'spec.json'), read(unit/'results.json'), release_evidence(unit), arm)


def reassess_prior(prior, arm):
    """CPU-only resume decision from current raw receipts, never stale status.

    A formerly passing unit must still pass; corruption fails closed. A failed
    or incomplete summary can be recovered only by a complete strict raw-data
    assessment. Promotion updates this mutable summary and keeps its old verdict.
    """
    if not isinstance(prior, dict) or not prior.get('unit'):
        return False
    was_passing = prior.get('assessment', {}).get('passed') is True
    try:
        if validate_arm(prior.get('arm', {})) != validate_arm(arm):
            raise ValueError('Prior arm binding differs from requested arm')
        check = assess_unit(prior['unit'], arm)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        if was_passing:
            raise ValueError('Previously completed raw receipt cannot be revalidated') from error
        return False
    if check.get('passed') is not True:
        if was_passing:
            raise ValueError('Previously completed raw receipt no longer passes')
        return False
    if prior.get('status') == 'passed' and prior.get('assessment') == check:
        return True
    unit = Path(prior['unit'])
    names = ('spec.json', 'results.json', 'release-verified.json', 'release-supplement.json')
    evidence = {name: hashlib.sha256((unit/name).read_bytes()).hexdigest()
                for name in names if (unit/name).is_file()}
    now = time.time()
    prior.setdefault('audit_history', []).append(dict(event='raw_receipt_reassessment', at=now,
        unit=prior['unit'], previous_status=prior.get('status'),
        previous_assessment=prior.get('assessment'), previous_error=prior.get('error')))
    prior.update(status='passed', assessment=check, reassessment=dict(at=now,
        reason='Current strict raw assessment passes; no GPU retry needed',
        evidence_sha256=evidence, supplemental_release='release-supplement.json' in evidence,
        executed_gpu=False))
    prior.pop('error', None)
    return True


def reconcile_prior_summary(receipt_path, arm):
    """Explicit CPU-only reconciliation; never dispatch a new attempt."""
    receipt_path = Path(receipt_path)
    report = read(receipt_path)
    prior = report.get('arms', {}).get(arm['id'])
    passed = reassess_prior(prior, arm)
    if passed:
        atomic(receipt_path, report)
    return dict(passed=passed, arm=arm['id'], executed_gpu=False,
                reassessment=prior.get('reassessment') if prior else None)


def stop(process):
    if process.poll() is not None:
        return
    # The owned controller receives TERM and runs its child-group finally.
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run_campaign(root, manifest, output, *, python, atlas_seed=None, only=()):
    root, output = Path(root).resolve(), Path(output).resolve()
    arms = [validate_arm(a) for a in manifest['arms']]
    if len({a['id'] for a in arms}) != len(arms):
        raise ValueError('Duplicate arm id')
    if only:
        if set(only) - {a['id'] for a in arms}:
            raise ValueError('Unknown requested arm')
        arms = [a for a in arms if a['id'] in only]
    output.mkdir(parents=True, exist_ok=True)
    manifest_identity = digest(manifest)
    receipt_path = output/'campaign.json'
    report = read(receipt_path) if receipt_path.exists() else dict(schema=SCHEMA,
        manifest_sha256=manifest_identity, arms={}, qualification=False, preflight=False)
    if report.get('manifest_sha256') != manifest_identity:
        raise ValueError('Campaign manifest changed; use a new output directory')
    plan_path = output/'plan.json'
    atomic(plan_path, plan())
    hotset = root/'.local-source-lfu/hot60-map.json'
    for arm in arms:
        prior = report['arms'].get(arm['id'])
        if reassess_prior(prior, arm):
            # A hash-bound supplement can repair a stale failed summary without
            # consuming another GPU window. Unit evidence stays untouched.
            atomic(receipt_path, report)
            continue
        attempt = 1
        stem = 'ds41-options-' + arm['id'] + '-' + manifest_identity[:8]
        while (root/'qualification/runs'/(stem + '-r' + str(attempt))).exists():
            attempt += 1
        name = stem + '-r' + str(attempt)
        unit = root/'qualification/runs'/name
        atlas = output/(name + '-atlas.json')
        if atlas_seed is not None:
            # Pin the seed bytes; each arm gets its own mutable frequency log.
            seed = Path(atlas_seed).read_bytes()
            with atlas.open('xb') as stream:
                stream.write(seed)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(('MLX2_BENCH_', 'MLX2_SOURCE_EXTRA_', 'MLX2_SOURCE_EXPERT_'))}
        env.update(PYTHONPATH=str(root/'src'), MLX2_BENCH_PLAN=str(plan_path),
            MLX2_BENCH_ATLAS=str(atlas), MLX2_SOURCE_EXTRA_POLICY=json.dumps(arm.get('policy', {})),
            MLX2_SOURCE_EXPERT_MATH=arm.get('expert_math', 'reference'),
            MLX2_BENCH_EXPERT_RESIDENCY=json.dumps(arm.get('expert_residency', {})))
        env.update({SETTING_ENV[k]:str(int(v)) for k,v in arm.get('settings',{}).items()})
        if arm.get('advanced_case'):
            env['MLX2_BENCH_ADVANCED_CASE'] = arm['advanced_case']
        entry = dict(unit=str(unit), status='running', arm=arm, started_at=time.time(),
                     atlas_seed_sha256=hashlib.sha256(seed).hexdigest() if atlas_seed is not None else None)
        report['arms'][arm['id']] = entry
        atomic(receipt_path, report)
        print(json.dumps(dict(event='arm_started', **entry)), flush=True)
        process = None
        try:
            with (output/(name + '-controller.log')).open('xb') as log:
                process = subprocess.Popen([str(python), str(root/'.local-source-lfu/run.py'),
                    name, 'mixed', str(arm.get('hotset', hotset))], cwd=root, env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                while process.poll() is None:
                    time.sleep(1)
            entry['exit_code'] = process.returncode
            entry['assessment'] = assess_unit(unit, arm)
            entry['status'] = 'passed' if entry['assessment']['passed'] else 'failed'
        except Exception as error:
            entry.update(status='failed', error=dict(type=type(error).__name__, message=str(error)))
            if isinstance(error, InterruptedError):
                entry['status'] = 'interrupted'
                raise
        finally:
            if process is not None:
                stop(process)
            entry['finished_at'] = time.time()
            atomic(receipt_path, report)
        print(json.dumps(dict(event='arm_finished', id=arm['id'], status=entry['status'],
            failed_gates=entry.get('assessment', {}).get('failed_gates'), error=entry.get('error'))), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--atlas-seed', type=Path)
    parser.add_argument('--arm', action='append', default=[])
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--assess-unit', type=Path)
    args = parser.parse_args()
    def interrupted(signum, _frame):
        raise InterruptedError('Campaign controller interrupted: ' + str(signum))
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    manifest = read(args.manifest)
    arms = [validate_arm(a) for a in manifest['arms']]
    if args.assess_unit:
        if len(args.arm) != 1:
            parser.error('--assess-unit needs exactly one --arm')
        arm = next(a for a in arms if a['id'] == args.arm[0])
        atomic(args.out, assess_unit(args.assess_unit, arm))
    elif args.run:
        run_campaign(args.root, manifest, args.out, python=args.python, atlas_seed=args.atlas_seed, only=args.arm)
    else:
        print(json.dumps(dict(schema=SCHEMA, validated_arms=len(arms), plan=plan(), executed=False)))


if __name__ == '__main__':
    main()
