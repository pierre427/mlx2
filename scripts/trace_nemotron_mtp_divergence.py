#!/usr/bin/env python3
"""Diagnostic comparison of Nemotron ordinary and self-MTP target state."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from qualify_nemotron3_super import ARTIFACT, source_hash

LOCKS = (Path('/tmp/gpu.lock/owner.json'), Path('/Users/Shared/mlxuag/gpu.lock/owner.json'))


def require_locks():
    rows = [json.loads(path.read_text()) for path in LOCKS]
    keys = ('owner', 'session_id', 'lease_id', 'generation', 'resource_key')
    if (rows[0] != rows[1] or rows[0]['owner'] != 'codex-nemotron3-super-debug'
            or rows[0]['resource_key'] != 'gpu'
            or any(key not in rows[0] for key in keys)):
        raise RuntimeError('GPU lock owner does not match Nemotron debugger')
    return rows[0]


def cache_diff(reference, candidate, mx):
    rows = []
    for index, (left, right) in enumerate(zip(reference, candidate)):
        right = right.extract(0)
        if hasattr(left, 'cache'):
            entries = []
            for a, b in zip(left.cache, right.cache):
                entries.append(None if a is None or b is None else
                               mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))
            mx.eval(*(value for value in entries if value is not None))
            rows.append({'layer': index, 'type': 'mamba',
                         'conv_max_abs': None if entries[0] is None else float(entries[0].item()),
                         'state_max_abs': None if entries[1] is None else float(entries[1].item())})
        else:
            left_keys, left_values = left.keys_and_values()
            right_keys, right_values = right.keys_and_values()
            dk = mx.max(mx.abs(left_keys.astype(mx.float32) - right_keys.astype(mx.float32)))
            dv = mx.max(mx.abs(left_values.astype(mx.float32) - right_values.astype(mx.float32)))
            mx.eval(dk, dv)
            rows.append({'layer': index, 'type': 'attention',
                         'reference_offset': left.offset, 'candidate_offset': right.offset,
                         'keys_max_abs': float(dk.item()), 'values_max_abs': float(dv.item())})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--split-last', action='store_true')
    args = parser.parse_args()
    lease = require_locks()
    os.environ.update(MLX_ENABLE_TF32='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import mlx.core as mx

    from mlx2.adapters.nemotron3_super import Nemotron3SuperAdapter
    from mlx2.runtime.hybrid_speculative import (
        attach_self_mtp_lanes,
        commit_batched_self_mtp,
        prepare_self_mtp_lane,
        propose_batched_self_mtp,
    )

    adapter = Nemotron3SuperAdapter(str(ARTIFACT))
    model = adapter.model
    prompt_text = args.prompt.read_text()
    prompt_ids = adapter.prompt_tokens({'messages': [{'role': 'user', 'content': prompt_text}],
                                        'enable_thinking': False})
    prompt = mx.array(prompt_ids, mx.uint32)
    reference_cache = model.make_cache()
    prefix_len = len(prompt_ids) - 1 if args.split_last else len(prompt_ids)
    for offset in range(0, prefix_len, 256):
        logits = model(prompt[None, offset:min(offset + 256, prefix_len)], cache=reference_cache)
        mx.eval(logits)
    if args.split_last:
        logits = model(prompt[None, -1:], cache=reference_cache)
        mx.eval(logits)
    reference_first = int(mx.argmax(logits[0, -1]).item())
    detached, first = prepare_self_mtp_lane(
        prompt, model, uid=1, max_tokens=args.tokens, prompt_cache=None,
        mtp_state=None, lane_rng=None, num_draft=3, sampling_temp=0,
        sampling_top_p=1, sampling_top_k=0, sampling_min_p=0,
        accept_rule='exact', logits_processors=[], prefill_step_size=256,
        share_qsa_indices=False,
    )
    batch = attach_self_mtp_lanes(model, None, [detached])
    result = {
        'schema': 'mlx2.nemotron3-mtp-divergence-trace.v1',
        'source_sha256': source_hash(), 'artifact': adapter.identity,
        'gpu_lease': lease, 'prompt_sha256': hashlib.sha256(prompt_text.encode()).hexdigest(),
        'prompt_tokens': len(prompt_ids), 'reference_first': reference_first,
        'split_last': args.split_last,
        'mtp_first': first.token, 'cycles': [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    if first.token != reference_first:
        print('first token differs', flush=True)
        return
    reference_current = reference_first
    emitted_total = 1
    for cycle_index in range(args.tokens):
        if emitted_total >= args.tokens:
            break
        before = cache_diff(reference_cache, batch.caches.target, mx)
        began = time.monotonic()
        proposal = propose_batched_self_mtp(model, batch)
        outputs = proposal.outputs[0]
        target_law = proposal._logprobs[0]
        emitted = min(len(outputs), args.tokens - emitted_total)
        comparisons = []
        mismatch = None
        for position, item in enumerate(outputs[:emitted]):
            logits = model(mx.array([[reference_current]], mx.uint32), cache=reference_cache)[0, -1]
            target_logprobs = target_law[position]
            mx.eval(logits, target_logprobs)
            ordinary_next = int(mx.argmax(logits).item())
            mtp_target_next = int(mx.argmax(target_logprobs).item())
            row = {'output_index': emitted_total + position,
                   'input_token': reference_current,
                   'ordinary_next': ordinary_next, 'mtp_next': item.token,
                   'mtp_target_next': mtp_target_next,
                   'mtp_from_draft': item.from_draft}
            comparisons.append(row)
            if mismatch is None and ordinary_next != item.token:
                mismatch = row
            reference_current = ordinary_next
        commit_batched_self_mtp(
            batch, proposal, emitted_counts=[emitted],
            terminal=[emitted < len(outputs) or emitted_total + emitted >= args.tokens],
        )
        emitted_total += emitted
        cycle = {'cycle': cycle_index, 'elapsed_s': time.monotonic() - began,
                 'accepted': proposal.accepted_lengths[0],
                 'drafts': list(proposal._drafts[0]),
                 'cache_before': before, 'outputs': comparisons}
        result['cycles'].append(cycle)
        args.output.write_text(json.dumps(result, indent=2, default=lambda value: value.tolist()) + '\n')
        max_mamba = max((max(row['conv_max_abs'] or 0, row['state_max_abs'] or 0)
                         for row in before if row['type'] == 'mamba'), default=0)
        max_kv = max((max(row['keys_max_abs'], row['values_max_abs'])
                      for row in before if row['type'] == 'attention'), default=0)
        print('cycle', cycle_index, 'output_start', comparisons[0]['output_index'],
              'accepted', proposal.accepted_lengths[0],
              'mamba_max', max_mamba, 'kv_max', max_kv,
              'mismatch', mismatch, flush=True)
        if mismatch is not None:
            result['first_mismatch'] = mismatch
            args.output.write_text(json.dumps(result, indent=2, default=lambda value: value.tolist()) + '\n')
            break


if __name__ == '__main__':
    main()
