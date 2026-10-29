# SPDX-License-Identifier: MIT
"""GPU quality + speed A/B: native vs APPROXIMATE resident-biased decode routing.

GPU OWNER ONLY. Loads DeepSeek-V4.1-Flash-1017-Source (Hot60 + Q1 g64) once
with ``deepseek_source_route_bias``. For each context: one exact prefill of a
distinct span of realistic tracked prose (never repeated filler; prefill is
never biased), a physical cache seed clone, then per repetition, interleaved
ABBA, ``--continuation`` teacher-forced B1 decode steps on a fresh clone with
the bias suspended (native control) or active. Teacher forcing feeds both
arms the same reference tokens, so KL measures the per-step distribution
shift including KV drift caused by biased experts.

Recorded: checkpoint ``<name>@<fingerprint>``, mlx2 commit (+dirty flag),
policy, owner receipt, expert cache budget, per-arm decode tok/s and expert
bytes/misses, route-bias change counters, and on repetition 0 mean/max
KL(native||biased), top-1 agreement and perplexity of both arms.

Expected-benefit caveat: at the 78 GB automatic maximum a warm B1 decode
reads ~0 expert bytes; rerun with ``--expert-budget-bytes`` lowered (memory
pressure) or right after a long prefill to see any speed effect. The arm run
second inherits residency from the first; ABBA ordering is recorded.
Exploratory only: not qualification and not a default change.

Usage (example):
  PYTHONPATH=$PWD/src .venv/bin/python scripts/research/ds41_route_bias_quality_speed.py \
      --owner-receipt /Users/Shared/mlxuag/gpu.lock/owner.json --out qualification/runs/ds41-route-bias-<date> \
      --mode resident_guarded_v1 --lambda 0.35 --contexts 2048,8192 --expert-budget-bytes 40000000000
"""
import argparse
import json
import time

from ds41_approx_common import (
    add_arguments,
    assert_varied,
    atomic,
    checkpoint,
    close_cache,
    compare,
    corpus_tokens,
    delta,
    load,
    logprobs,
    prepare_out,
    source_revision,
    stream_counters,
)


def clone(model, cache):
    from ds41_decode_ablation import clone_cache_physical
    return clone_cache_physical(model, cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    parser.add_argument('--contexts', default='2048,8192')
    parser.add_argument('--mode', default='resident_guarded_v1', choices=('resident_v1', 'resident_guarded_v1'))
    parser.add_argument('--lambda', dest='lam', type=float, default=0.35)
    parser.add_argument('--adaptive', default='off', choices=('off', 'margin_v1'))
    parser.add_argument('--margin', default=None, help='minimum,maximum,hard_gap,easy_gap for margin_v1')
    args = parser.parse_args()
    out = prepare_out(args.out)
    contexts = [int(v) for v in args.contexts.split(',')]
    if any(not 2 <= c < 16384 - args.continuation for c in contexts):
        raise SystemExit('route-bias contexts must stay below the 16384 clone frontier')
    from mlx2.adapters.deepseek_v41_source_approx import (
        ROUTE_ADAPTIVE,
        ROUTE_KEY,
        ROUTE_LAMBDA,
        ROUTE_MARGIN,
    )
    extra = {ROUTE_KEY: args.mode, ROUTE_ADAPTIVE: args.adaptive}
    if args.adaptive == 'off':
        extra[ROUTE_LAMBDA] = args.lam
    elif args.margin:
        low, high, hard, easy = (float(v) for v in args.margin.split(','))
        extra[ROUTE_MARGIN] = dict(minimum=low, maximum=high, hard_gap=hard, easy_gap=easy)
    record = dict(schema='mlx2.ds41-route-bias-quality-speed.v1', status='starting', qualification=False,
                  performance_claim=False, approximate=True, owner_receipt=json.loads(args.owner_receipt.read_text()),
                  source_revision=source_revision(), expert_budget_bytes=args.expert_budget_bytes,
                  **checkpoint(args), started_at=time.time(), cells=[])
    atomic(out / 'results.json', record)
    import mlx.core as mx
    import numpy as np
    adapter = None
    try:
        adapter, policy = load(args, extra)
        record.update(policy=policy, profile=adapter.profile, numeric_profile=adapter.numeric_profile,
                      approximate_receipt=adapter.diagnostics()['source_approximate_operations'])
        model, bias = adapter.model, adapter.model.source_route_bias
        stream, corpus = corpus_tokens(adapter, sum(contexts) + len(contexts) * args.continuation + 1)
        record['corpus'] = corpus
        atomic(out / 'results.json', record)
        offset = 0
        for context in contexts:
            prompt = stream[offset:offset + context]
            continuation = stream[offset + context:offset + context + args.continuation]
            offset += context + args.continuation
            assert_varied(prompt)
            seed = model.make_cache()
            for start in range(0, len(prompt) - 1, 1024):
                piece = prompt[start:min(len(prompt) - 1, start + 1024)]
                value = model(mx.array([piece], mx.uint32), seed, logits_to_keep=1)
                mx.eval(value, [row.state for row in seed])
            cell = dict(context=context, continuation=len(continuation), runs=[])
            arrays = {}
            for rep in range(args.reps):
                order = ('native', 'biased') if rep % 2 == 0 else ('biased', 'native')
                for arm in order:
                    cache = clone(model, seed)
                    before = stream_counters(adapter)
                    changed_before = bias.receipt()['recent_changed_layer_rows']
                    feed = [prompt[-1]] + continuation[:-1]
                    rows = []
                    started = time.perf_counter()
                    with (bias.suspend() if arm == 'native' else _null()):
                        for token in feed:
                            logits = model(mx.array([[token]], mx.uint32), cache)
                            mx.eval(logits)
                            if rep == 0:
                                rows.append(logprobs(logits))
                    seconds = time.perf_counter() - started
                    run = dict(rep=rep, arm=arm, decode_tokens=len(feed), seconds=seconds,
                               tokens_per_second=len(feed) / seconds, expert=delta(before, stream_counters(adapter)),
                               changed_layer_rows_window=bias.receipt()['recent_changed_layer_rows'] - changed_before
                               if arm == 'biased' else 0)
                    if rep == 0:
                        arrays[arm] = np.concatenate(rows)
                    cell['runs'].append(run)
                    close_cache(cache)
                    print(json.dumps(dict(context=context, **run)), flush=True)
            close_cache(seed)
            cell['quality'] = compare(arrays['native'], arrays['biased'], np.asarray(continuation))
            rate = {arm: [r['tokens_per_second'] for r in cell['runs'] if r['arm'] == arm] for arm in ('native', 'biased')}
            cell['decode_mean_tokens_per_second'] = {k: sum(v) / len(v) for k, v in rate.items()}
            record['cells'].append(cell)
            record['route_bias_receipt'] = bias.receipt()
            atomic(out / 'results.json', record)
        record['status'] = 'completed'
    except BaseException as error:
        record.update(status='failed', error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        if adapter is not None:
            try:
                adapter.close()
                record['adapter_closed'] = True
            except BaseException as error:  # noqa: BLE001 - keep evidence
                record['cleanup_error'] = repr(error)
        record['finished_at'] = time.time()
        atomic(out / 'results.json', record)


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == '__main__':
    main()
