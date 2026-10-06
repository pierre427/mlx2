"""Bounded paired same-service complete-request tree15/ordinary measurements.

B1 compares tree15 with its same-native-law serial reference. B2 auto and
forced ordinary both use shared ordinary kernels; that comparison measures
width-policy behavior including any B1 edges, not a B2 tree/ordinary crossover. No default is promoted.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import platform
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import threading
import time

import tree15_owned_transition_gate as gate
ROOT = gate.ROOT
SCRIPT = 'scripts/research/tree15_owned_request_perf_gate.py'


def host_snapshot():
    values = {'platform': platform.platform(), 'machine': platform.machine()}
    for label, command in [('swapusage', ['sysctl', 'vm.swapusage']),
                           ('thermal', ['pmset', '-g', 'therm'])]:
        try:
            values[label] = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=3).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            values[label] = {'unavailable': repr(exc)}
    return values


def run_arm(base, bodies, width, route):
    """Wall covers complete cohort admission/prefill/output, not next() time."""
    barrier = threading.Barrier(width + 1)
    def request(body):
        barrier.wait(timeout=5)
        return gate.post(base, body, route)
    with concurrent.futures.ThreadPoolExecutor(max_workers=width) as pool:
        futures = [pool.submit(request, b) for b in bodies[:width]]
        started = time.monotonic()
        barrier.wait(timeout=5)
        rows = [f.result(timeout=120) for f in futures]
        wall = time.monotonic() - started
    return {'route_requested': route, 'width': width, 'requests': rows,
            'complete_cohort_wall_seconds': wall,
            'aggregate_completion_tokens': sum(r['response']['usage']['completion_tokens'] for r in rows),
            'aggregate_complete_request_tokens_per_second':
                sum(r['response']['usage']['completion_tokens'] for r in rows) / wall,
            'rate_definition': 'sum actual completion tokens / complete cohort monotonic wall; includes prefill',
            'per_request_metrics': [{
                'complete_http_wall_seconds': r['wall_seconds'],
                'completion_tokens': r['response']['usage']['completion_tokens'],
                'native_prefill_to_first_token_seconds': r['response']['mlx2']['timing']['prefill_to_first_token_seconds'],
                'native_ttft_from_router_enqueue_seconds': r['response']['mlx2']['timing']['first_token_offset_seconds'],
                'native_decode_seconds_including_peer_prefill_pauses': r['response']['mlx2']['timing']['decode_seconds'],
                'native_decode_tokens_per_second_including_peer_prefill_pauses': r['response']['mlx2']['timing']['decode_tokens_per_second'],
                'peer_prefill_seconds_overlap_not_subtracted': [other['response']['mlx2']['timing']['prefill_to_first_token_seconds'] for other in rows if other is not r],
            } for r in rows],
            'b1_tree_edges_observed': any(r['response']['mlx2']['active_width_modes'].get('b1_tree_eligible', 0) > 0 for r in rows),
            'idle_after': gate.wait_idle(base)}


def check_arm(arm, identity, caps):
    if len(arm['requests']) != arm['width'] or len(caps) != arm['width']:
        raise RuntimeError('complete cohort request count differs from width')
    for row, cap in zip(arm['requests'], caps):
        receipt = row['response']['mlx2']
        if not gate.isolated(row) or not gate.valid_completion(row, cap) or receipt['identity'] != identity:
            raise RuntimeError('request identity/cache/finish contract failed')
        timing = receipt['timing']
        if (timing['clock'] != 'host_monotonic_ns' or timing['cached_prompt_tokens'] != 0
                or timing['prompt_tokens'] != row['response']['usage']['prompt_tokens']
                or timing['completion_tokens'] != row['response']['usage']['completion_tokens']
                or timing['prefill_to_first_token_seconds'] <= 0
                or timing['first_token_offset_seconds'] is None):
            raise RuntimeError('actual native token/timing boundaries failed')
        mode = ('b2plus_shared_ordinary' if arm['width'] == 2 else
                'b1_tree_eligible' if arm['route_requested'] == 'auto' else 'b1_serial_ordinary')
        if receipt['active_width_modes'].get(mode, 0) < 1:
            raise RuntimeError('requested mechanism not physically observed')
        if arm['width'] == 2:
            if receipt['active_width_counts'].get('2', 0) < 1:
                raise RuntimeError('B2 must observe shared ordinary width2')
            if receipt['drafted'] != 0 and (arm['route_requested'] == 'serial'
                    or receipt['active_width_modes'].get('b1_tree_eligible', 0) < 1):
                raise RuntimeError('drafts require an observed auto B1 edge')
        elif arm['route_requested'] == 'auto':
            if receipt['drafted'] <= 0: raise RuntimeError('B1 tree did not draft')
        elif receipt['drafted'] != 0:
            raise RuntimeError('forced ordinary unexpectedly drafted')


def pair_summary(pairs):
    result = {}
    for width in (1, 2):
        rows = [p for p in pairs if p['width'] == width]
        ratios = [p['auto']['complete_cohort_wall_seconds'] / p['serial']['complete_cohort_wall_seconds'] for p in rows]
        result[str(width)] = {
            'pairs': len(rows), 'auto_over_serial_complete_wall_ratios': ratios,
            'median_auto_over_serial_complete_wall_ratio': statistics.median(ratios),
            'auto_complete_wall_seconds': [p['auto']['complete_cohort_wall_seconds'] for p in rows],
            'serial_complete_wall_seconds': [p['serial']['complete_cohort_wall_seconds'] for p in rows],
            'comparison': 'actual B1 tree15 versus same-native ordinary' if width == 1 else
                          'same shared ordinary at width2; auto may include measured B1 tree edges',
            'b1_edges_observed': [p['auto']['b1_tree_edges_observed'] for p in rows],
        }
    return result


def measured_pairs(base, bodies, identity, pairs=3, *, rows=None, persist=lambda: None):
    if rows is None: rows = []
    # Interleave widths; alternate order within each width. Three fixed pairs,
    # no adaptive early stopping or selecting only favorable observations.
    for repeat in range(pairs):
        for width in (1, 2):
            order = ('serial', 'auto') if repeat % 2 == 0 else ('auto', 'serial')
            row = {'repeat': repeat, 'width': width, 'order': list(order), 'host_before': host_snapshot()}
            rows.append(row)
            selected = bodies[:1] if width == 1 else [{**b, 'max_tokens': 64} for b in bodies]
            for route in order:
                arm = run_arm(base, selected, width, route)
                check_arm(arm, identity, [192] if width == 1 else [64, 64])
                row[route] = arm
                persist()
            row['exact_native_serial'] = all(gate.exact(a, b) for a, b in
                                            zip(row['auto']['requests'], row['serial']['requests']))
            if not row['exact_native_serial']:
                raise RuntimeError('paired exact native serial output mismatch')
            row['host_after'] = host_snapshot()
            persist()
    return rows


def child(args):
    sys.path.insert(0, str(gate.HELPERS))
    from tensorfold_owned_spomin_cell import source_cell, make_body, FROZEN_CORPUS_SHA256
    from mlx2.runtime.tensorfold_owned_worker import source_identity, mlx_lm_identity, artifact_identity, profile_identity
    identities = {'tensorfold': source_identity(args.source), 'mlx_lm': mlx_lm_identity(args.mlx_lm_source),
                  'target': artifact_identity(args.target), 'drafter': artifact_identity(args.drafter)}
    identity = profile_identity(identities['tensorfold'], identities['target'], identities['drafter'], identities['mlx_lm'])
    domain, _, prepared, system = source_cell(0, args.target, args.prepared_report, args.prepared_sha)
    bodies = [make_body(system, prepared[0], 'full', 192), make_body(system, prepared[1], 'full', 64)]
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    command = [sys.executable, '-m', 'mlx2.server', '--model', args.target, '--ordinary',
               '--host', '127.0.0.1', '--port', str(port), '--max-lanes', '8', '--max-inflight', '8',
               '--max-context', '8192', '--cache-bytes', str(512 << 20), '--tensorfold-owned-live',
               '--tensorfold-owned-source', args.source, '--tensorfold-owned-mlx-lm-source', args.mlx_lm_source,
               '--tensorfold-owned-drafter', args.drafter]
    binding = gate.source_binding(args.source_commit); binding['files'][SCRIPT] = gate.digest(ROOT / SCRIPT)
    output = {'schema': 'mlx2.tree15-owned-request-perf-gate.v1', 'status': 'running',
              'source': binding, 'lease': gate.lease_identity(), 'identities': identities,
              'expected_identity': identity, 'command': command, 'domain': domain,
              'corpus_sha256': FROZEN_CORPUS_SHA256, 'prepared_report_sha256': args.prepared_sha,
              'cases': [p['receipt'] for p in prepared[:2]],
              'request_body_sha256': [hashlib.sha256(json.dumps(b, sort_keys=True).encode()).hexdigest() for b in bodies],
              'qualified': False, 'default_off': True, 'scope': 'three alternating complete-request pairs per width1/2',
              'b2_comparison': 'both execute shared ordinary at actual width2; measured auto B1 edges may contribute to complete wall',
              'completion_caps': {'B1': [192], 'B2': [64, 64]},
              'no_b2_tree_or_b3plus_crossover_claim': True,
              'ttft_definition': 'native first token ready offset from router enqueue; nonstreaming HTTP completion is separate',
              'decode_rate_definition': 'per request native tokens after first / native first-to-final ready wall, including peer prefill pauses; not steady decode',
              'host_before': host_snapshot(), 'pairs': []}
    persist = lambda: Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True) + '\n')
    persist()
    service = subprocess.Popen(command, cwd=ROOT, stdout=sys.stderr, stderr=sys.stderr)
    base = f'http://127.0.0.1:{port}'
    try:
        deadline = time.monotonic() + 60
        while True:
            if service.poll() is not None: raise RuntimeError('service exited during startup')
            try: gate.get(base, '/health'); break
            except Exception:
                if time.monotonic() > deadline: raise TimeoutError('service startup timeout')
                time.sleep(.1)
        warm = [{**b, 'max_tokens': 8} for b in bodies]
        output['warmup'] = [run_arm(base, warm, 1, 'serial'), run_arm(base, warm, 1, 'auto'),
                            run_arm(base, warm, 2, 'serial')]
        persist()
        measured_pairs(base, bodies, identity, rows=output['pairs'], persist=persist)
        output['summary'] = pair_summary(output['pairs'])
        output['host_after'] = host_snapshot(); output['idle_after'] = gate.wait_idle(base)
        output['status'] = 'completed'
    except Exception as exc:
        output['status'] = 'failed'; output['error'] = repr(exc)
    finally:
        service.terminate()
        try: service.wait(timeout=5)
        except subprocess.TimeoutExpired: service.kill(); service.wait(timeout=3)
        Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True) + '\n')
    return 0 if output['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'target', 'drafter', 'mlx-lm-source', 'prepared-report', 'prepared-sha', 'source-commit', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--deadline-seconds', type=float, default=360)
    parser.add_argument('--rss-gib', type=float, default=48)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    raise SystemExit(child(args) if args.child else gate.supervise(args, entrypoint=__file__, extra_source_paths=(SCRIPT,), max_deadline_seconds=420))

if __name__ == '__main__': main()
