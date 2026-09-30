#!/usr/bin/env python3
"""Exact-artifact greedy ordinary versus native-MTP direct-model gate.

Run only while holding the shared GPU task and matching filesystem locks.
This records mechanism engagement and end-to-end direct-model timings; it does
not issue a production serving qualification receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = Path('~/mlx-models/Nemotron-3-Super-120B-A12B-5bit-MTP')
LOCKS = (Path('/tmp/gpu.lock/owner.json'), Path('/Users/Shared/mlxuag/gpu.lock/owner.json'))


def require_locks():
    receipts = [json.loads(path.read_text()) for path in LOCKS]
    keys = ('campaign_id', 'session_id', 'lease_id', 'generation', 'owner')
    if any(tuple(row.get(key) for key in keys) != tuple(receipts[0].get(key) for key in keys) for row in receipts):
        raise RuntimeError('GPU lock receipts disagree')
    if receipts[0].get('resource_key') != 'gpu' or receipts[0].get('owner') != 'codex-nemotron3-super-mtp':
        raise RuntimeError('GPU lock is not owned by this qualification')
    return receipts[0]


def source_hash():
    digest = hashlib.sha256()
    for path in sorted((ROOT / 'src/mlx2').rglob('*.py')):
        digest.update(str(path.relative_to(ROOT)).encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def ordinary(model, prompt, count, mx):
    cache = model.make_cache()
    start = time.monotonic()
    logits = model(prompt[None], cache=cache)
    mx.eval(logits)
    prefill_s = time.monotonic() - start
    current = int(mx.argmax(logits[0, -1]).item())
    tokens = [current]
    decode_start = time.monotonic()
    while len(tokens) < count:
        logits = model(mx.array([[current]], mx.uint32), cache=cache)
        mx.eval(logits)
        current = int(mx.argmax(logits[0, -1]).item())
        tokens.append(current)
    return {'tokens': tokens, 'prefill_s': prefill_s,
            'decode_s': time.monotonic() - decode_start,
            'target_forwards': count}


def mtp(model, prompt, count, depth, mx):
    from mlx2.runtime.hybrid_speculative import (
        attach_self_mtp_lanes,
        commit_batched_self_mtp,
        prepare_self_mtp_lane,
        propose_batched_self_mtp,
    )

    start = time.monotonic()
    detached, first = prepare_self_mtp_lane(
        prompt, model, uid=1, max_tokens=count, prompt_cache=None,
        mtp_state=None, lane_rng=None, num_draft=depth, sampling_temp=0,
        sampling_top_p=1, sampling_top_k=0, sampling_min_p=0,
        accept_rule='exact', logits_processors=[], prefill_step_size=256,
        share_qsa_indices=False,
    )
    prefill_s = time.monotonic() - start
    batch = attach_self_mtp_lanes(model, None, [detached])
    tokens = [first.token]
    decode_start = time.monotonic()
    cycles = 0
    accepted = 0
    proposed = 0
    while len(tokens) < count:
        proposal = propose_batched_self_mtp(model, batch)
        row = proposal.outputs[0]
        take = min(len(row), count - len(tokens))
        commit_batched_self_mtp(
            batch, proposal, emitted_counts=[take],
            terminal=[take < len(row) or len(tokens) + take >= count],
        )
        tokens.extend(item.token for item in row[:take])
        cycles += 1
        accepted += proposal.accepted_lengths[0]
        proposed += proposal.draft_depths[0]
    return {'tokens': tokens, 'prefill_s': prefill_s,
            'decode_s': time.monotonic() - decode_start,
            'cycles': cycles, 'draft_proposed': proposed,
            'draft_accepted': accepted}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 4 <= args.tokens <= 128:
        raise ValueError('tokens must be 4..128')
    lock = require_locks()
    os.environ['MLX_ENABLE_TF32'] = '0'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import mlx.core as mx

    from mlx2.adapters.nemotron3_super import Nemotron3SuperAdapter

    print('loading exact 5-bit artifact', flush=True)
    load_start = time.monotonic()
    adapter = Nemotron3SuperAdapter(str(ARTIFACT))
    load_s = time.monotonic() - load_start
    prompt = mx.array(adapter.prompt_tokens({'messages': [
        {'role': 'user', 'content': 'Explain why 17 times 19 equals 323 in two short sentences.'}
    ]}), mx.uint32)
    mx.eval(prompt)
    print(f'loaded in {load_s:.2f}s, prompt={prompt.size} tokens', flush=True)
    model = adapter.model
    # Discard one short cycle per path to amortize first kernel/graph setup.
    ordinary(model, prompt, 4, mx)
    mtp(model, prompt, 4, 2, mx)
    mx.synchronize()
    arms = {}
    for name, fn in (
        ('ordinary_a', lambda: ordinary(model, prompt, args.tokens, mx)),
        ('mtp2_a', lambda: mtp(model, prompt, args.tokens, 2, mx)),
        ('mtp3', lambda: mtp(model, prompt, args.tokens, 3, mx)),
        ('mtp2_b', lambda: mtp(model, prompt, args.tokens, 2, mx)),
        ('ordinary_b', lambda: ordinary(model, prompt, args.tokens, mx)),
    ):
        mx.clear_cache()
        arm = fn()
        arm['decode_tokens_per_s'] = (args.tokens - 1) / arm['decode_s']
        arm['active_memory_bytes'] = mx.get_active_memory()
        arm['peak_memory_bytes'] = mx.get_peak_memory()
        arms[name] = arm
        print(name, {k: v for k, v in arm.items() if k != 'tokens'}, flush=True)
    reference = arms['ordinary_a']['tokens']
    parity = {name: arm['tokens'] == reference for name, arm in arms.items()}
    result = {
        'schema': 'mlx2.nemotron3-super-mtp-candidate.v1',
        'source_sha256': source_hash(), 'artifact': adapter.identity,
        'gpu_lease': lock, 'mlx_version': mx.__version__ if hasattr(mx, '__version__') else None,
        'prompt_tokens': int(prompt.size), 'generated_tokens': args.tokens,
        'load_s': load_s, 'rss_peak_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'arms': arms, 'greedy_parity': parity,
        'qualification': 'direct-model-candidate-only',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('parity', parity, 'result', args.output, flush=True)
    if not all(parity.values()) or any(arms[name]['draft_proposed'] <= 0 for name in ('mtp2_a','mtp3','mtp2_b')):
        raise SystemExit('MTP parity or engagement gate failed')


if __name__ == '__main__':
    main()
