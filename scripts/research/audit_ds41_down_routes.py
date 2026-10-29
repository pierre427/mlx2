# SPDX-License-Identifier: MIT
"""CPU census of actual PP DOWN rows and synchronized diagnostic events."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ds41_down_actual_capture import band


def audit(unit, boundary='boundary-0004.json'):
    unit = Path(unit)
    spec = json.loads((unit/'spec.json').read_text())
    plan_file = unit/'resident-plan.json'
    plan = json.loads(plan_file.read_text())
    profile = json.loads((unit/boundary).read_text())
    if hashlib.sha256(plan_file.read_bytes()).hexdigest() != spec['resident_plan_sha256']:
        raise ValueError('DOWN census plan file seal mismatch')
    if plan['checkpoint'] != spec['checkpoint'] or plan['route_source_snapshot_sha256'] != spec['source_snapshot_sha256']:
        raise ValueError('DOWN census plan checkpoint/source binding mismatch')
    identity = profile['capture_identity']
    for key in ('checkpoint', 'source_revision', 'source_snapshot_sha256'):
        if identity[key] != spec[key]:
            raise ValueError('DOWN census capture identity mismatch')
    if (identity['input_rows'], identity['frontier'], identity['synchronized_diagnostic'], identity['rate_eligible']) != (99, 0, True, False):
        raise ValueError('DOWN census requires untimed synchronized PP99/frontier0')
    if plan['budget_bytes'] != 75_000_000_000 or plan['hot_bytes'] != 60_000_000_000:
        raise ValueError('DOWN census budget changed')
    hot = frozenset(map(tuple, plan['hot_experts']))
    jobs = []
    layers = set()
    for route in profile['routing']:
        layer = int(route['layer'].split('.')[1])
        ids = np.asarray(route['expert_ids'])
        if layer in layers or ids.shape != (99, 6) or np.any(ids < 0) or np.any(ids > 383) or np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
            raise ValueError('DOWN census malformed/duplicate routed layer')
        layers.add(layer)
        for expert, rows in sorted(Counter(map(int, ids.ravel())).items()):
            jobs.append((layer, expert, 'original' if (layer, expert) in hot else 'q1', rows))
    if layers != set(range(40)):
        raise ValueError('DOWN census needs all40 layers')
    events = [e for e in profile['events'] if e['category'] == 'expert_compute' and e['name'] in ('qmm:mxfp4:down', 'q1_g64_matmul:down')]
    if len(events) != len(jobs):
        raise ValueError('DOWN job/event count mismatch')
    strata = {}
    kinds = {}
    histogram = {'original': Counter(), 'q1': Counter()}
    for (layer, expert, kind, rows), event in zip(jobs, events):
        expected = 'qmm:mxfp4:down' if kind == 'original' else 'q1_g64_matmul:down'
        if event['layer'] != layer or event['name'] != expected:
            raise ValueError('DOWN sequential provider job/event identity order mismatch')
        histogram[kind][rows] += 1
        label = band(kind, rows)
        for target, key in ((strata, label), (kinds, kind)):
            row = target.setdefault(key, dict(calls=0, activation_rows=0, synchronized_exclusive_seconds=0., input_barrier_seconds=0.,
                                             logical_dot_FLOPs=0, unique_projection_coefficient_bytes=0, logical_input_bytes=0, logical_output_bytes=0))
            row['calls'] += 1
            row['activation_rows'] += rows
            row['synchronized_exclusive_seconds'] += event['exclusive_seconds']
            row['input_barrier_seconds'] += event['input_barrier_seconds']
            row['logical_dot_FLOPs'] += 2*rows*2304*5120
            row['unique_projection_coefficient_bytes'] += 6_266_880 if kind == 'original' else 2_211_840
            row['logical_input_bytes'] += rows*2304*4
            row['logical_output_bytes'] += rows*5120*4
    return dict(schema='mlx2.ds41-down-route-census.v1', checkpoint=spec['checkpoint'], source_revision=spec['source_revision'],
        source_snapshot_sha256=spec['source_snapshot_sha256'], unit=str(unit.resolve()),
        bindings={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (unit/'spec.json', plan_file, unit/boundary)},
        plan_fingerprint=plan['fingerprint'], original_artifact_fingerprint=plan['source_binding'],
        q1_bank_binding=plan['q1_binding'], hot_map_binding=plan['hotmap_sha256'],
        expert_budget_bytes=plan['budget_bytes'], original_hot_bytes=plan['hot_bytes'],
        scope='Actual PP99 jobs matched positionally to ordinary per-expert DOWN events, asserted by layer+kind and frozen provider order',
        timing_scope='Forced input/output eval synchronized wall, not ordinary critical path or benchmark rate; input barrier already inside total wall',
        traffic_scope='Logical unique projection payload and operation counts only; not physical GPU traffic, achieved FLOPs or bandwidth',
        observed_jobs=len(jobs), observed_activation_rows=sum(j[3] for j in jobs), kinds=kinds, strata=strata,
        row_histogram={k: dict(sorted(v.items())) for k, v in histogram.items()}, qualified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = audit(args.unit)
    with Path(args.out).open('x') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
