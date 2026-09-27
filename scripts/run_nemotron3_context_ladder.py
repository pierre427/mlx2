#!/usr/bin/env python3
"""One candidate ordinary/MTP3 run per exact Nemotron context cell.

This is a direct-model experiment, not a serving qualification. It writes each
cell atomically before moving to the next so a stopped ladder remains auditable.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

from qualify_nemotron3_super import ARTIFACT, mtp, source_hash

LOCKS = (Path('/tmp/gpu.lock/owner.json'), Path('/Users/Shared/mlxuag/gpu.lock/owner.json'))
EXPECTED = {
    'session_id': '3c65a7827e83456a89fffdeefadec196',
    'lease_id': '2362c6c845d244eea73b27334add9154',
    'resource_key': 'gpu',
}


def require_locks():
    rows = [json.loads(path.read_text()) for path in LOCKS]
    if (rows[0] != rows[1]
            or rows[0].get('owner') != os.environ.get('MLX2_NEMOTRON_GPU_OWNER', 'codex-nemotron3-super-context-ladder')
            or rows[0].get('generation') != int(os.environ.get('MLX2_NEMOTRON_GPU_GENERATION', '363'))
            or any(row.get(key) != value for key, value in EXPECTED.items() for row in rows)):
        raise RuntimeError('Nemotron ladder does not own both GPU locks')
    return rows[0]


def device_state():
    therm = subprocess.run(['pmset', '-g', 'therm'], capture_output=True, text=True, check=True).stdout
    swap = subprocess.run(['sysctl', 'vm.swapusage'], capture_output=True, text=True, check=True).stdout
    match = re.search(r'used\s*=\s*([\d.]+)([MG])', swap)
    if match is None:
        raise RuntimeError(f'cannot parse swap usage: {swap!r}')
    used_mb = float(match.group(1)) * (1024 if match.group(2) == 'G' else 1)
    nominal = all('No ' in line for line in therm.splitlines() if 'warning level' in line)
    return {'thermal_nominal': nominal, 'thermal_raw': therm, 'swap_used_mb': used_mb}


def ordinary(model, prompt, count, mx, *, chunk_size=256):
    cache = model.make_cache()
    prefill_start = time.monotonic()
    for offset in range(0, int(prompt.size), chunk_size):
        logits = model(prompt[None, offset:offset + chunk_size], cache=cache)
        mx.eval(logits)
    prefill_s = time.monotonic() - prefill_start
    current = int(mx.argmax(logits[0, -1]).item())
    tokens = [current]
    decode_start = time.monotonic()
    while len(tokens) < count and current not in (2, 11):
        logits = model(mx.array([[current]], mx.uint32), cache=cache)
        mx.eval(logits)
        current = int(mx.argmax(logits[0, -1]).item())
        tokens.append(current)
    return {'tokens': tokens, 'prefill_s': prefill_s,
            'decode_s': time.monotonic() - decode_start, 'target_forwards': len(tokens)}


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-context', type=int, default=262016)
    parser.add_argument('--output-tokens', type=int, default=32)
    parser.add_argument('--resume', action='store_true',
                        help='append only unattempted context cells to an identity-matching receipt')
    parser.add_argument('--continue-on-mismatch', action='store_true',
                        help='diagnostic sweep only; retain failures and continue to later cells')
    args = parser.parse_args()
    if not 1 <= args.output_tokens <= 128:
        raise ValueError('output-tokens must be 1..128')
    lease = require_locks()
    os.environ.update(MLX_ENABLE_TF32='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import mlx.core as mx

    from mlx2.adapters.nemotron3_super import Nemotron3SuperAdapter

    prompt_rows = json.loads(args.prompts.read_text())
    planned = [row for row in prompt_rows if row['target_tokens'] <= args.max_context]
    if not planned:
        raise ValueError('no context cells selected')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fresh_result = {
        'schema': 'mlx2.nemotron3-super-context-ladder-candidate.v1',
        'status': 'running', 'qualification': 'direct-model-candidate-only',
        'source_sha256': source_hash(), 'gpu_lease': lease,
        'prompt_manifest': str(args.prompts.resolve()),
        'prompt_manifest_sha256': hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        'planned_contexts': [row['target_tokens'] for row in planned],
        'output_tokens': args.output_tokens, 'runs_per_cell': 1,
        'prefill_chunk_tokens': 256, 'cells': [],
    }
    if args.resume:
        result = json.loads(args.output.read_text())
        if str(result.get('stop_reason', '')).startswith('device '):
            raise RuntimeError('cannot resume after a thermal or swap guard stop')
        for key in ('source_sha256', 'prompt_manifest_sha256', 'planned_contexts',
                    'output_tokens', 'runs_per_cell', 'prefill_chunk_tokens'):
            if result.get(key) != fresh_result[key]:
                raise RuntimeError(f'cannot resume changed ladder identity: {key}')
        if len({cell['context_tokens'] for cell in result['cells']}) != len(result['cells']):
            raise RuntimeError('duplicate context cells in resume receipt')
        result['status'] = 'running'
        result['stop_reason'] = None
    else:
        result = fresh_result
    atomic_json(args.output, result)
    load_start = time.monotonic()
    adapter = Nemotron3SuperAdapter(str(ARTIFACT))
    model = adapter.model
    if args.resume and result['artifact']['fingerprint'] != adapter.identity['fingerprint']:
        raise RuntimeError('cannot resume changed model artifact')
    result['artifact'] = adapter.identity
    result['load_s'] = time.monotonic() - load_start
    if args.resume:
        result.setdefault('resume_baselines', []).append(device_state())
    else:
        result['baseline'] = device_state()
    atomic_json(args.output, result)
    baseline_swap = (result['resume_baselines'][-1] if args.resume else result['baseline'])['swap_used_mb']
    stop = None
    for index, row in enumerate(planned):
        target = row['target_tokens']
        if args.resume and any(cell['context_tokens'] == target for cell in result['cells']):
            continue
        source = Path(row['path'])
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            stop = f'prompt hash changed at {target}'
            break
        text = data.decode()
        ids = adapter.prompt_tokens({'messages': [{'role': 'user', 'content': text}],
                                     'enable_thinking': False})
        if len(ids) != target:
            stop = f'prompt token mismatch at {target}: {len(ids)}'
            break
        prompt = mx.array(ids, mx.uint32)
        cell = {'context_tokens': target, 'prompt_sha256': row['sha256'],
                'actual_prompt_tokens': len(ids), 'arms': {}, 'status': 'running'}
        result['cells'].append(cell)
        atomic_json(args.output, result)
        order = ('ordinary', 'mtp3') if index % 2 == 0 else ('mtp3', 'ordinary')
        for route in order:
            require_locks()
            before = device_state()
            if not before['thermal_nominal'] or before['swap_used_mb'] - baseline_swap > 2048:
                stop = f'device preflight failed at {target}/{route}'
                break
            mx.clear_cache(); mx.reset_peak_memory()
            print(f'start {target} {route}', flush=True)
            try:
                arm = ordinary(model, prompt, args.output_tokens, mx) if route == 'ordinary' else (
                    mtp(model, prompt, args.output_tokens, 3, mx)
                )
                mx.synchronize()
                arm['active_memory_bytes'] = mx.get_active_memory()
                arm['peak_memory_bytes'] = mx.get_peak_memory()
                arm['cache_memory_bytes'] = mx.get_cache_memory()
                arm['decoded'] = adapter.tokenizer.decode(arm['tokens'], skip_special_tokens=True)
                arm['ladder_ready'] = arm['decoded'].lstrip().startswith('LADDER_READY')
                arm['decode_tokens_per_s'] = max(0, len(arm['tokens']) - 1) / arm['decode_s'] if arm['decode_s'] else None
                if route == 'mtp3' and arm['draft_proposed'] <= 0:
                    raise RuntimeError('MTP mechanism was not engaged')
                after = device_state()
                arm['device_before'] = before
                arm['device_after'] = after
                cell['arms'][route] = arm
                atomic_json(args.output, result)
                print(f'done {target} {route}: prefill {arm["prefill_s"]:.2f}s, '
                      f'decode {arm["decode_s"]:.2f}s, ready={arm["ladder_ready"]}, '
                      f'peak={arm["peak_memory_bytes"] / 2**30:.2f}GiB', flush=True)
                if not after['thermal_nominal'] or after['swap_used_mb'] - baseline_swap > 2048:
                    stop = f'device postflight failed at {target}/{route}'
                    break
            except Exception as exc:  # noqa: BLE001 - retain failed GPU cell before stopping
                cell['arms'][route] = {'error': repr(exc), 'device_before': before}
                atomic_json(args.output, result)
                stop = f'{target}/{route}: {exc!r}'
                break
            finally:
                gc.collect(); mx.clear_cache()
        if stop:
            cell['status'] = 'stopped'
            break
        ordinary_row, mtp_row = cell['arms']['ordinary'], cell['arms']['mtp3']
        cell['greedy_parity'] = ordinary_row['tokens'] == mtp_row['tokens']
        cell['quality_pass'] = ordinary_row['ladder_ready'] and mtp_row['ladder_ready']
        cell['status'] = 'passed' if cell['greedy_parity'] and cell['quality_pass'] else 'failed'
        atomic_json(args.output, result)
        if cell['status'] == 'failed' and not args.continue_on_mismatch:
            stop = f'correctness mismatch at {target}'
            break
    result['status'] = ('stopped' if stop else
                        'complete_with_failures' if any(cell['status'] != 'passed' for cell in result['cells'])
                        else 'complete')
    result['stop_reason'] = stop
    result['final_device'] = device_state()
    atomic_json(args.output, result)
    print('result', result['status'], stop, args.output, flush=True)


if __name__ == '__main__':
    main()
