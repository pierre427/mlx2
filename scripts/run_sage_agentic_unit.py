#!/usr/bin/env python3
"""One bounded owned unit for Sage's new agentic request features.

Does not mutate or resume any existing qualification campaign. The caller must
queue through run_with_gpu_locks + CPG/radio + run_qualification_owned. All
subprocesses inherit this source, and the server group is retired in finally.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from check_sage_agentic import probe
from run_single_model_smoke_ladder import (
    atomic_json,
    stop_server,
    thermal_sample,
    validate_campaign_ownership,
    wait_health,
)

from mlx2.adapters.qwen36_35b import inspect_artifact
from mlx2.apple_telemetry import TemperatureSampler

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source-revision', required=True); p.add_argument('--artifact-fingerprint', required=True)
    p.add_argument('--session', required=True); p.add_argument('--label', required=True)
    p.add_argument('--cpg-task', required=True); p.add_argument('--cpg-owner-lock', type=Path, required=True)
    p.add_argument('--generation', type=int, required=True); p.add_argument('--port', type=int, default=8439)
    p.add_argument('--route', choices=('ordinary', 'native_mtp'), default='native_mtp')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    report = {'schema': 'mlx2.sage-agentic-owned-unit.v1', 'source_revision': args.source_revision,
        'checkpoint': args.model.name + '@sha256:' + args.artifact_fingerprint,
        'scope': 'targeted feature smoke, not a replacement qualification', 'route': args.route, 'started': time.time(), 'status': 'running'}
    server = None
    def save(): atomic_json(args.out / 'unit.json', report)
    def ownership():
        return validate_campaign_ownership(args.session, args.label, cpg_task=args.cpg_task,
            cpg_owner_lock=args.cpg_owner_lock, cpg_generation=args.generation)
    def interrupted(signum, frame): raise KeyboardInterrupt(f'signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP): signal.signal(sig, interrupted)
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        if revision != args.source_revision: raise RuntimeError('Source commit changed')
        artifact = inspect_artifact(args.model)
        if artifact['identity']['fingerprint'] != args.artifact_fingerprint: raise RuntimeError('Artifact changed')
        report['ownership_start'] = ownership()
        inventory = subprocess.check_output(['ps', '-axo', 'pid,ppid,etime,command'], text=True)
        report['process_inventory'] = inventory
        foreign = [line for line in inventory.splitlines() if '-m mlx2.server ' in line and '/bin/zsh' not in line]
        if foreign: raise RuntimeError('Foreign serving process exists; do not disturb it')
        report['services'] = subprocess.check_output(['launchctl', 'list'], text=True)
        report['vm_stat_before'] = subprocess.check_output(['vm_stat'], text=True)
        report['pmset_before'] = subprocess.check_output(['pmset', '-g', 'therm'], text=True)
        report['thermal_policy'] = {'purpose': 'functional feature smoke, not performance qualification',
            'accepted_thermal_states': [0, 1], 'consecutive_samples': 3,
            'sample_interval_seconds': 15, 'die_temperature_gating': False}
        temperatures = TemperatureSampler(); samples = []
        try:
            for index in range(3):
                ownership()
                sample = {**thermal_sample(), **temperatures.die_summary(), 'recorded_at': time.time()}
                samples.append(sample); report['thermal_admission'] = samples; save()
                if sample['thermal_state'] not in (0, 1):
                    raise RuntimeError('Thermal admission pending; release ownership and retry in new evidence')
                if index < 2: time.sleep(15)
        finally: temperatures.close()
        with (args.out / 'reference.log').open('w') as log:
            subprocess.run([sys.executable, str(ROOT / 'scripts/smoke_local_model.py'), '--model', str(args.model),
                '--out', str(args.out / 'reference.json'), '--max-tokens', '16', '--i-own-gpu',
                '--cpg-lease', f'{args.cpg_task}:{args.generation}'], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        ownership()
        command = [sys.executable, '-m', 'mlx2.server', '--model', str(args.model), '--host', '127.0.0.1',
            '--port', str(args.port), '--max-context', '262144', '--max-lanes', '16', '--max-inflight', '32',
            '--cache-bytes', str(16 * 1024**3), '--cache-dir', str(args.out / 'cache'),
            '--prefill-step', '256', '--native-mtp' if args.route == 'native_mtp' else '--ordinary', '--fabric', 'off']
        report['server_command'] = command; save()
        with (args.out / 'server.log').open('w') as log:
            server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        report['server_pid'] = server.pid; save()
        status = wait_health(f'http://127.0.0.1:{args.port}', server, 600)
        report['initial_status'] = status
        result = probe(f'http://127.0.0.1:{args.port}', args.out / 'features', source_revision=args.source_revision,
                       expected_checkpoint=report['checkpoint'])
        report['status'] = 'passed' if result['passed'] else 'failed'
    except BaseException as error:
        report['status'] = 'pending_admission' if 'pending;' in str(error) else 'failed'
        report['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        stop_server(server)
        report['server_stopped'] = server is None or server.poll() is not None
        try:
            report['ownership_end'] = ownership()
        except Exception as error:  # noqa: BLE001 - retain cleanup evidence after lease loss
            report['ownership_end_error'] = type(error).__name__ + ': ' + str(error)
        report['finished'] = time.time(); save()


if __name__ == '__main__': main()
