#!/usr/bin/env python3
"""Owned, ephemeral ordinary-serving check for one selected Kimi weight program.

Checks base cold/warm, adapter cold/warm, unload and base cold/warm restoration.
No service default is promoted. No model is loaded in this supervisor; only its
owned loopback server child loads the read-only copy. Run via kimi_lora_owned.py.
"""
from __future__ import annotations

import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAX_METAL_BYTES = 100 << 30
DEFAULT_SYSTEM_RESERVE_BYTES = 20 << 30


def atomic(path, value):
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def readonly_manifest(path):
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError('serving model must be a real read-only copy directory')
    if path.stat().st_mode & 0o222:
        raise ValueError('model copy has a writable root directory')
    rows = []
    for file in sorted(path.rglob('*')):
        if file.is_symlink():
            raise ValueError('read-only copy must not contain symlinks')
        if file.is_dir() and file.stat().st_mode & 0o222:
            raise ValueError('model copy has a writable directory')
        if file.is_file():
            stat = file.stat()
            if stat.st_mode & 0o222:
                raise ValueError('model copy contains a writable file: ' + str(file.relative_to(path)))
            rows.append({'path': str(file.relative_to(path)), 'size': stat.st_size,
                         'mtime_ns': stat.st_mtime_ns, 'mode': stat.st_mode, 'inode': stat.st_ino})
    if not rows or not (path / 'config.json').is_file():
        raise ValueError('read-only model copy is incomplete')
    return {'file_count': len(rows), 'metadata_sha256': hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest(), 'files': rows}


def validate_spec(spec):
    for field in ('model', 'original_model', 'artifact_name', 'base_fingerprint', 'source_revision',
                  'out', 'job_id', 'adapter', 'selected_candidate_receipt', 'selected_candidate_receipt_sha256'):
        if not isinstance(spec.get(field), str) or not spec[field]:
            raise ValueError('missing serving spec field: ' + field)
    model, original = Path(spec['model']).resolve(), Path(spec['original_model']).resolve()
    if model == original or original in model.parents or model in original.parents:
        raise ValueError('serving requires a separate model copy')
    if model.name != original.name or model.name != spec['artifact_name']:
        raise ValueError('copy basename must match the exact original checkpoint name')
    out = Path(spec['out']).resolve()
    if any(out == base or base in out.parents for base in (model, original)):
        raise ValueError('receipts must be outside both model directories')
    if not isinstance(spec.get('execution_policy'), dict) or not spec['execution_policy']:
        raise ValueError('serving requires an explicit frozen execution_policy')
    for key, default, maximum in (('max_tokens', 64, 256), ('max_context', 4096, 32768),
                                  ('prefill_step', 128, 2048), ('cache_bytes', 1 << 30, 4 << 30)):
        value = spec.get(key, default)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError('invalid bounded serving option: ' + key)
    for key, default in (('max_metal_bytes', DEFAULT_MAX_METAL_BYTES),
                         ('system_reserve_bytes', DEFAULT_SYSTEM_RESERVE_BYTES)):
        value = spec.get(key, default)
        if type(value) is not int or value <= 0:
            raise ValueError('invalid serving resource bound: ' + key)
    if spec.get('max_metal_bytes', DEFAULT_MAX_METAL_BYTES) > DEFAULT_MAX_METAL_BYTES:
        raise ValueError('serving Metal bound may not exceed 100 GiB')
    if spec.get('system_reserve_bytes', DEFAULT_SYSTEM_RESERVE_BYTES) < DEFAULT_SYSTEM_RESERVE_BYTES:
        raise ValueError('serving system reserve may not be below 20 GiB')
    return model, out


def resource_evidence(status, spec, *, available_bytes=None):
    """Sample host availability and the worker's last published Metal peak.

    This is a lifecycle-boundary guard, not continuous allocation enforcement:
    the server snapshot can lag the worker, and an HTTP request is synchronous.
    """
    if available_bytes is None:
        import psutil
        available_bytes = psutil.virtual_memory().available
    peak = status.get('metal_peak_bytes')
    if type(peak) is not int or peak < 0 or type(available_bytes) is not int or available_bytes < 0:
        raise ValueError('serving resource sample is missing or invalid')
    ceiling = spec.get('max_metal_bytes', DEFAULT_MAX_METAL_BYTES)
    reserve = spec.get('system_reserve_bytes', DEFAULT_SYSTEM_RESERVE_BYTES)
    evidence = {'metal_peak_bytes_last_published': peak, 'host_available_bytes': available_bytes,
                'max_metal_bytes': ceiling, 'system_reserve_bytes': reserve,
                'sampled_at': time.time(), 'sampling': 'lifecycle_boundary_not_continuous'}
    if peak > ceiling:
        raise MemoryError('serving Metal peak exceeded experiment bound')
    if available_bytes < reserve:
        raise MemoryError('serving system memory reserve exhausted')
    return evidence


def server_command(spec, directory, port, token_path):
    return [sys.executable, '-u', '-m', 'mlx2.server', '--model', str(Path(spec['model']).resolve()),
            '--host', '127.0.0.1', '--port', str(port), '--max-lanes', '1', '--max-inflight', '1',
            '--ordinary', '--no-auto-route', '--no-thinking-auto-calibration',
            '--max-context', str(spec.get('max_context', 4096)),
            '--default-max-tokens', str(spec.get('max_tokens', 64)),
            '--prefill-step', str(spec.get('prefill_step', 128)),
            '--cache-bytes', str(spec.get('cache_bytes', 1 << 30)), '--cache-dir', str(directory / 'cache'),
            '--lora-dir', str(Path(spec['adapter']).resolve()), '--max-loras', '0',
            '--admin-token-file', str(token_path), '--execution-policy', str(directory / 'policy.json')]


class Client:
    def __init__(self, port, token):
        self.base, self.token = f'http://127.0.0.1:{port}', token

    def call(self, path, body=None, *, timeout=180):
        headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.token}
        request = Request(self.base + path, data=None if body is None else json.dumps(body).encode(), headers=headers)
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
        return json.loads(payload) if path != '/metrics' else payload.decode()


def wait_ready(client, server, *, timeout=240, clock=time.monotonic, sleep=time.sleep):
    deadline = clock() + timeout
    while clock() < deadline:
        if server.poll() is not None:
            raise RuntimeError('owned server exited before readiness')
        try:
            status = client.call('/v1/status', timeout=3)
            if status.get('state') == 'ready' and status.get('healthy') is True:
                return status
            if status.get('error'):
                raise RuntimeError('owned server reports startup error: ' + str(status['error']))
        except (URLError, TimeoutError, OSError):
            pass
        sleep(.25)
    raise TimeoutError('owned server readiness timed out')


def wait_idle(client, *, timeout=20, clock=time.monotonic, sleep=time.sleep):
    deadline = clock() + timeout
    while clock() < deadline:
        status = client.call('/v1/status', timeout=3)
        if (status.get('healthy') is True and status.get('inflight') == 0 and status.get('queue_depth') == 0
                and status.get('accepted_lifecycles') == 0
                and (status.get('quiesce') or {}).get('terminal_cleanup_pending') == 0):
            return status
        sleep(.1)
    raise TimeoutError('owned server did not reach complete request quiescence')


def stable_identity(status):
    execution, settings = status.get('execution') or {}, status.get('settings') or {}
    apc = status.get('apcv2') or {}
    if (execution.get('route') != 'ordinary' or settings.get('route') != 'ordinary'
            or settings.get('max_lanes') != 1 or settings.get('max_inflight') != 1
            or (status.get('multi_lora') or {}).get('enabled') is not False):
        raise ValueError('server did not select the requested single-lane ordinary route')
    if apc.get('version') != 2 or not apc.get('layout_name'):
        raise ValueError('server did not expose exact APCv2 layout identity')
    return {key: status[key] for key in ('model', 'artifact', 'base_artifact', 'runtime', 'build_runtime', 'profile')} | {
        'execution': {key: execution.get(key) for key in ('route', 'policy', 'artifact')},
        'settings': settings, 'apcv2': {'version': 2, 'layout_name': apc['layout_name']},
    }


def first_divergence(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def response_evidence(response, *, expected, program=None):
    from mlx2.generation_trace import read_token_trace
    receipt = response.get('mlx2') or {}
    if receipt.get('cache') != 'apcv2' or receipt.get('route') != 'ordinary':
        raise ValueError('response did not execute ordinary APCv2 serving')
    trace = read_token_trace(receipt, expected_tokens=response.get('usage', {}).get('completion_tokens'))
    if not trace['token_ids'] or any(width != 1 for width in trace['execution_widths']):
        raise ValueError('response token trace is empty or not width one')
    cached = (response.get('usage', {}).get('prompt_tokens_details') or {}).get('cached_tokens')
    prompt_tokens = response.get('usage', {}).get('prompt_tokens')
    if (type(cached) is not int or type(prompt_tokens) is not int or not 0 <= cached <= prompt_tokens
            or cached != receipt.get('cached_tokens')):
        raise ValueError('cached-token usage and APCv2 receipt disagree')
    observed = receipt.get('weight_program')
    if program is None:
        if observed is not None:
            raise ValueError('base response unexpectedly selected a weight program')
    elif (not isinstance(observed, dict) or observed.get('program_id') != program['program_id']
          or observed.get('checkpoint') != program['checkpoint'] or observed.get('source_revision') != program['source_revision']
          or observed.get('observed_used') is not True or observed.get('selected') is not True
          or observed.get('qualified') is not False or receipt.get('qualification') != 'unqualified'):
        raise ValueError('response did not observe the exact unqualified weight program')
    choice = response['choices'][0]
    content = choice['message'].get('content') or ''
    # The identifier oracle permits only the project-approved Unicode dash normalization.
    normalized = content.strip().translate(str.maketrans({c: '-' for c in '\u2010\u2011\u2012\u2013\u2014\u2212'}))
    return {'response': response, 'tokens': trace['token_ids'], 'cached_tokens': cached,
            'oracle_passed': normalized == expected, 'finish_reason': choice.get('finish_reason')}


def compare_replay(cold, warm):
    divergence = first_divergence(cold['tokens'], warm['tokens'])
    return {'cold_miss': cold['cached_tokens'] == 0, 'warm_hit': warm['cached_tokens'] > 0,
            'exact_token_parity': divergence is None, 'first_divergence_zero_based': divergence,
            'classification': 'exact' if divergence is None else 'unclassified',
            'oracle_passed': cold['oracle_passed'] and warm['oracle_passed']}


def verify_transition(before, after, *, operation, active, program=None):
    if stable_identity(before) != stable_identity(after):
        raise ValueError('artifact/runtime/profile/policy/settings changed during LoRA lifecycle')
    if after.get('model_revision') != before.get('model_revision', -1) + 1:
        raise ValueError('LoRA lifecycle did not invalidate the model revision exactly once')
    if (after.get('lora') or {}).get('active') != active:
        raise ValueError('active adapter status differs from lifecycle action')
    matches = [row for row in after.get('recent_operation_receipts', [])
               if row.get('operation') == operation and row.get('status') == 'completed'
               and row.get('model_revision') == after['model_revision']]
    if not matches:
        raise ValueError('missing revision-bound completed lifecycle receipt')
    observed = (after.get('lora') or {}).get('weight_program')
    if program is not None:
        if (not isinstance(observed, dict) or any(observed.get(key) != program[key]
                for key in ('program_id', 'checkpoint', 'source_revision'))
                or observed.get('adapter_delta_applications') != 0 or observed.get('observed_used') is not False):
            raise ValueError('new adapter did not start with exact identity and zero applications')
    elif observed is not None:
        raise ValueError('unloaded base retained a weight-program identity')
    return {'operation': operation, 'revision_before': before['model_revision'], 'revision_after': after['model_revision'],
            'operation_receipt': matches[-1], 'apcv2_contract': 'single-session state invalidation; verified by subsequent cold miss'}


def run_lifecycle(client, spec, selected, *, ownership_check, persist=lambda: None, record=None,
                  resource_check=lambda status, stage: None):
    record = {} if record is None else record
    name = spec['job_id'] + '-adapter'
    expected = 'KL-7319'
    # A unique request avoids inherited disk entries; the fresh per-unit cache is also empty.
    nonce = hashlib.sha256(spec['job_id'].encode()).hexdigest()[:12]
    prompt = ('Audit ' + nonce + '.\n' + '\n'.join(f'Item {i:03d} has status recorded.' for i in range(80))
              + '\nThe shipment identifier is ' + expected + '. Return only the shipment identifier.')
    request = {'model': spec['artifact_name'], 'messages': [{'role': 'user', 'content': prompt}],
               'temperature': 0, 'max_tokens': spec.get('max_tokens', 64), 'stream': False, 'return_token_trace': True}
    startup = wait_idle(client)
    reference_identity = stable_identity(startup)
    if startup.get('artifact') != spec['base_fingerprint'] or startup.get('model') != spec['artifact_name']:
        raise ValueError('server loaded the wrong exact base checkpoint')
    if (startup.get('execution') or {}).get('policy') != spec['execution_policy']:
        raise ValueError('server execution policy differs from the frozen spec')
    if (startup.get('lora') or {}).get('active') is not None:
        raise ValueError('fresh server already has an active adapter')
    record.update(startup_status=startup, request=request, arms=[], transitions=[])
    persist()
    resource_check(startup, 'startup')

    def pair(label, program=None):
        before = wait_idle(client)
        resource_check(before, label + '_before')
        rows = []
        arm = {'name': label, 'rows': rows, 'before': before}
        record['arms'].append(arm)
        for phase in ('cold', 'warm'):
            ownership_check()
            response = client.call('/v1/chat/completions', request)
            row = {'response': response, 'phase': phase}
            rows.append(row)
            persist()
            row.update(response_evidence(response, expected=expected, program=program))
            resource_check(wait_idle(client), label + '_' + phase + '_after')
        after = wait_idle(client)
        arm['after'] = after
        persist()
        if stable_identity(after) != reference_identity:
            raise ValueError('server identity changed during a replay pair')
        replay = compare_replay(*rows)
        arm['replay'] = replay
        if program is not None:
            left = (before.get('lora') or {}).get('weight_program') or {}
            right = (after.get('lora') or {}).get('weight_program') or {}
            start, finish = left.get('adapter_delta_applications'), right.get('adapter_delta_applications')
            if type(start) is not int or type(finish) is not int or finish <= start:
                raise ValueError('native adapter application counter did not increase')
            arm['adapter_application_delta'] = finish - start
        persist()
        return arm

    base = pair('base')
    ownership_check()
    before = wait_idle(client)
    loaded = client.call('/v1/load_lora_adapter', {'lora_name': name, 'lora_path': spec['adapter'], 'base_model_name': spec['artifact_name']})
    record['load_response'] = loaded
    after = wait_idle(client)
    record['load_after_status'] = after
    persist()
    resource_check(after, 'adapter_load_after')
    record['transitions'].append(verify_transition(before, after, operation='lora_load', active=name, program=selected))
    persist()
    adapted = pair('adapter', selected)
    ownership_check()
    before = wait_idle(client)
    record['unload_response'] = client.call('/v1/unload_lora_adapter', {'lora_name': name})
    after = wait_idle(client)
    record['unload_after_status'] = after
    persist()
    resource_check(after, 'adapter_unload_after')
    record['transitions'].append(verify_transition(before, after, operation='lora_unload', active=None))
    persist()
    restored = pair('restored_base')
    record['base_restored_exactly'] = base['rows'][0]['tokens'] == restored['rows'][0]['tokens']
    record['ordinary_reference'] = base['rows'][0]['response']
    record['passed'] = (record['base_restored_exactly'] and all(all(arm['replay'][key]
        for key in ('cold_miss', 'warm_hit', 'exact_token_parity', 'oracle_passed')) for arm in (base, adapted, restored)))
    persist()
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args(argv)
    spec_bytes = args.spec.read_bytes()
    spec = json.loads(spec_bytes.decode())
    spec_sha256 = hashlib.sha256(spec_bytes).hexdigest()
    model, out = validate_spec(spec)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if revision != spec['source_revision']:
        raise ValueError('serving worktree revision differs from spec')
    # All ownership checks precede any server process/model load.
    from mlx2.desktop_campaign_worker import check_ownership
    from mlx2.desktop_campaign_process import spawn
    ownership = check_ownership(spec)
    from kimi_lora_round import candidate_selection
    selected = candidate_selection({**spec, 'eval_split': 'final'})
    files_before = readonly_manifest(model)
    directory = out / 'serving'
    directory.mkdir(mode=0o700)
    token = secrets.token_hex(32)
    token_path = directory / 'admin-token'
    token_path.write_text(token); token_path.chmod(0o600)
    atomic(directory / 'policy.json', spec['execution_policy'])
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    command = server_command(spec, directory, port, token_path)
    record = {'schema': 'mlx2.kimi-lora-serving-check.v1', 'spec': spec, 'spec_sha256': spec_sha256,
              'checkpoint': spec['artifact_name'] + '@' + spec['base_fingerprint'],
              'source_revision': revision, 'source_files': {name: sha(ROOT / name) for name in (
                  'scripts/research/kimi_lora_serving.py', 'scripts/research/kimi_lora_round.py',
                  'src/mlx2/serving.py', 'src/mlx2/server.py', 'src/mlx2/runtime/lora.py')},
              'ownership': ownership, 'selected_candidate': selected, 'readonly_copy_before': files_before,
              'command': command, 'port': port, 'started_at': time.time(), 'passed': False,
              'qualified': False, 'default_promoted': False, 'performance_claim': False}
    receipt_path = directory / 'receipt.json'
    persist = lambda: atomic(receipt_path, record)
    persist()
    def check_resources(status, stage):
        import psutil
        available = psutil.virtual_memory().available
        try:
            evidence = resource_evidence(status, spec, available_bytes=available)
        except (ValueError, MemoryError) as error:
            record.setdefault('resource_checks', []).append({'stage': stage, 'passed': False,
                'sampled_at': time.time(), 'metal_peak_bytes_last_published': status.get('metal_peak_bytes'),
                'host_available_bytes': available,
                'max_metal_bytes': spec.get('max_metal_bytes', DEFAULT_MAX_METAL_BYTES),
                'system_reserve_bytes': spec.get('system_reserve_bytes', DEFAULT_SYSTEM_RESERVE_BYTES),
                'sampling': 'lifecycle_boundary_not_continuous',
                'error': f'{type(error).__name__}: {error}'})
            persist()
            raise
        record.setdefault('resource_checks', []).append({'stage': stage, 'passed': True, **evidence})
        persist()
    server = None
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f'signal {signum}')
    handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        with (directory / 'server.log').open('x') as log:
            server = spawn(command, cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'PYTHONUNBUFFERED': '1'},
                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record['server_pid'] = server.pid; persist()
            client = Client(port, token)
            wait_ready(client, server)
            listeners = subprocess.check_output(['/usr/sbin/lsof', '-nP', f'-iTCP:{port}', '-sTCP:LISTEN', '-Fp'], text=True)
            if {int(row[1:]) for row in listeners.splitlines() if row.startswith('p')} != {server.pid}:
                raise RuntimeError('loopback listener does not belong exclusively to the owned server')
            run_lifecycle(client, spec, selected, ownership_check=lambda: check_ownership(spec), persist=persist, record=record,
                          resource_check=check_resources)
            # Revalidate selection bytes after the server loaded and used them.
            if candidate_selection({**spec, 'eval_split': 'final'}) != selected:
                raise ValueError('selected adapter/source receipt changed during serving check')
            (directory / 'metrics.prom').write_text(client.call('/metrics'))
    except BaseException as error:
        record.update(passed=False, error=f'{type(error).__name__}: {error}')
    finally:
        try:
            if server is not None:
                server.stop(timeout=20)
                record['server_exit_code'] = server.returncode
                record['server_stopped'] = not server.live()
        except BaseException as error:
            record.update(passed=False, cleanup_error=f'{type(error).__name__}: {error}')
        try:
            record['readonly_copy_after'] = readonly_manifest(model)
            record['readonly_copy_unchanged'] = files_before == record['readonly_copy_after']
            if not record['readonly_copy_unchanged']:
                record['passed'] = False
        except BaseException as error:
            record.update(passed=False, copy_check_error=f'{type(error).__name__}: {error}')
        token_path.unlink(missing_ok=True)
        record['finished_at'] = time.time(); persist()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
