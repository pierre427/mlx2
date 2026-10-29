# SPDX-License-Identifier: MIT
"""Interleaved GPU A/B for three opt-in DS4.1 Source streaming mechanisms.

Mechanisms (each against the current default it modifies):

* ``prefill-residency``: ``deepseek_source_experts.prefill_residency``
  ``protect_decode_v1`` versus today's eviction, 4,096-token prompt, default
  1,024-row prefill. Read: warm decode tok/s (decode after a 4K prompt; was
  2.8-7.7 versus 15.6 warm), warm TTFT, expert bytes and shield counters.
* ``readahead``: ``deepseek_source_prefill_readahead: layer_ahead_v1`` on the
  ``layer_major_v1`` 4,096-row wave versus layer-major without read-ahead.
  Read: warm TTFT, ``readahead_*`` counters (used/ready/wasted bytes).
* ``miss-gate``: embedded DSpark serial d2 with
  ``deepseek_dspark_admission: expert_miss_cost_v1`` versus plain DSpark d2
  and ordinary decode. Read: decode tok/s, handoffs, oracle pass.

Arms reuse the recorded 2026-10-28 measurement arms
(``ds41_measure_ab_manifest.json``) so every other policy is the selected
default, and run in counterbalanced interleaved order (A B, B A, ...). The
existing owned runner (``ds41_options_campaign.run_campaign`` ->
``.local-source-lfu/run.py``) holds the CPG/paired locks, writes the per-unit
checkpoint, source revision and snapshot, runs the ordinary-reference smoke and
always cleans up. This wrapper adds: a clean-tree check, the mlx2 commit, the
expected checkpoint, and a refusal to summarize a unit whose recorded
checkpoint differs. Without ``--run`` it only writes and prints the plan.
Exploratory and unqualified; one host, no thermal control.

    PYTHONPATH=src .venv/bin/python scripts/research/ds41_streaming_ab.py \\
        --mechanism prefill-residency --out output/ds41-prefill-residency-ab --run
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ('DeepSeek-V4.1-Flash-1017-Source-Hot60GB-Q1g64@'
              '927325be8884e00a7bf60b9aa7d540b167a5efec569b9c7c30b1d34c2793d000')
BASE_MANIFEST = Path(__file__).with_name('ds41_measure_ab_manifest.json')
SCHEMA = 'mlx2.ds41-streaming-ab.v1'


def _arm(base, arm_id, *, policy=None, residency=None):
    arm = copy.deepcopy(base)
    arm['id'] = arm_id
    arm['policy'].update(policy or {})
    arm['expert_residency'].update(residency or {})
    return arm


def mechanism_arms(name, bases):
    """Return [(label, arm)] for one repetition: control(s) first."""
    if name == 'prefill-residency':
        base = bases['pf-ref']
        return [('control', _arm(base, 'pres-evict')),
                ('treatment', _arm(base, 'pres-protect', residency={'prefill_residency': 'protect_decode_v1'}))]
    if name == 'readahead':
        base = bases['pf-layer-major']
        return [('control', _arm(base, 'ra-layer-major')),
                ('treatment', _arm(base, 'ra-layer-ahead',
                                   policy={'deepseek_source_prefill_readahead': 'layer_ahead_v1'}))]
    if name == 'miss-gate':
        return [('ordinary', _arm(bases['dsp-ordinary'], 'gate-ordinary')),
                ('control', _arm(bases['dsp-serial-d2'], 'gate-serial-d2')),
                ('treatment', _arm(bases['dsp-serial-d2'], 'gate-serial-d2-cost',
                                   policy={'deepseek_dspark_admission': 'expert_miss_cost_v1'}))]
    raise ValueError('unknown mechanism: ' + name)


def build_manifest(name, repetitions):
    if type(repetitions) is not int or not 1 <= repetitions <= 9:
        raise ValueError('repetitions must be 1..9')
    bases = {arm['id']: arm for arm in json.loads(BASE_MANIFEST.read_text())['arms']}
    arms, order = [], []
    for repetition in range(1, repetitions + 1):
        row = mechanism_arms(name, bases)
        if repetition % 2 == 0:
            row = row[::-1]                      # counterbalance position bias
        for label, arm in row:
            arm = dict(arm, id=f'{arm["id"]}-r{repetition}')
            arms.append(arm)
            order.append(dict(id=arm['id'], label=label, repetition=repetition))
    return dict(purpose=f'DS4.1 Source {name} interleaved A/B; exploratory, unqualified, one host',
                arms=arms), order


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def summarize(report, order):
    rows = []
    for item in order:
        entry = report.get('arms', {}).get(item['id'], {})
        assessment = entry.get('assessment', {})
        recorded = assessment.get('checkpoint')
        if recorded is not None and recorded != CHECKPOINT:
            raise ValueError(f'unit {item["id"]} ran checkpoint {recorded}, expected {CHECKPOINT}')
        rows.append(dict(item, status=entry.get('status'), checkpoint=recorded,
                         source_revision=assessment.get('source_revision'),
                         failed_gates=assessment.get('failed_gates'), cells=assessment.get('cells')))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mechanism', required=True, choices=('prefill-residency', 'readahead', 'miss-gate'))
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--atlas-seed', type=Path)
    parser.add_argument('--run', action='store_true', help='launch the owned GPU runner (GPU owner only)')
    args = parser.parse_args()
    manifest, order = build_manifest(args.mechanism, args.repetitions)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()) and not (out / 'ab.json').exists():
        raise SystemExit('refusing a nonempty evidence directory that is not this A/B')
    commit = git('rev-parse', 'HEAD')
    dirty = bool(git('status', '--porcelain', '--untracked-files=no'))
    plan = dict(schema=SCHEMA, mechanism=args.mechanism, expected_checkpoint=CHECKPOINT,
                mlx2_commit=commit, tree_dirty=dirty, repetitions=args.repetitions, order=order,
                qualification=False, performance_qualified=False, thermal_controlled=False)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    if not args.run:
        print(json.dumps(dict(plan, executed=False), indent=1))
        return
    if dirty:
        raise SystemExit('commit or discard tracked changes first: units must name a clean commit')
    spec = importlib.util.spec_from_file_location('ds41_options_campaign',
                                                  Path(__file__).with_name('ds41_options_campaign.py'))
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    (out / 'ab.json').write_text(json.dumps(dict(plan, status='running'), indent=1) + '\n')
    report = campaign.run_campaign(ROOT, manifest, out / 'campaign', python=args.python,
                                   atlas_seed=args.atlas_seed)
    result = dict(plan, status='finished', rows=summarize(report, order),
                  mlx2_commit_after=git('rev-parse', 'HEAD'))
    (out / 'ab.json').write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps(dict(status='finished', out=str(out)), indent=1))


if __name__ == '__main__':
    main()
