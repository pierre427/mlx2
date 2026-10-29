# SPDX-License-Identifier: MIT
"""GPU quality + speed A/B: exact Source prefill vs APPROXIMATE yoco_v1.

GPU OWNER ONLY. Loads DeepSeek-V4.1-Flash-1017-Source (Hot60 + Q1 g64) once
with ``deepseek_source_prefill_approx=yoco_v1``. For each context it slices a
distinct span of realistic tracked prose (never repeated filler), then per
repetition runs, interleaved ABBA, the exact control (ordinary chunk-major
sequence on the same model: ``exact_reference_prefill``) and the admitted
YOCO wave. Both arms then teacher-force the same reference continuation
(the corpus' next ``--continuation`` tokens) through the ordinary path.

Recorded: checkpoint ``<name>@<fingerprint>``, mlx2 commit (+dirty flag),
policy, owner receipt, per-arm prefill seconds and expert bytes read, YOCO
wave receipt, and on repetition 0: mean/max KL(exact||yoco), top-1
agreement and perplexity of both arms on the reference continuation.
Exploratory, one checkpoint/shape; not qualification, not a default change.

Usage (example):
  PYTHONPATH=$PWD/src .venv/bin/python scripts/research/ds41_yoco_quality_speed.py \
      --owner-receipt /Users/Shared/mlxuag/gpu.lock/owner.json --out qualification/runs/ds41-yoco-<date> \
      --contexts 8192,16384,32768 --keep-rows 140 --layers 21-39
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
    timed,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    parser.add_argument('--contexts', default='8192,16384,32768')
    parser.add_argument('--keep-rows', type=int, default=140)
    parser.add_argument('--layers', default='21-39')
    parser.add_argument('--min-rows', type=int, default=4096)
    args = parser.parse_args()
    out = prepare_out(args.out)
    contexts = [int(v) for v in args.contexts.split(',')]
    if any(c < args.min_rows or c + args.continuation > 65536 for c in contexts):
        raise SystemExit('contexts must be >= --min-rows and fit the 65536 reference context')
    from mlx2.adapters.deepseek_v41_source_approx import (
        YOCO_KEEP,
        YOCO_KEY,
        YOCO_LAYERS,
        YOCO_MIN_ROWS,
    )
    record = dict(schema='mlx2.ds41-yoco-quality-speed.v1', status='starting', qualification=False,
                  performance_claim=False, approximate=True, owner_receipt=json.loads(args.owner_receipt.read_text()),
                  source_revision=source_revision(), **checkpoint(args), started_at=time.time(), cells=[])
    atomic(out / 'results.json', record)
    import mlx.core as mx

    from mlx2.adapters.deepseek_v41_source_yoco import exact_reference_prefill
    adapter = None
    try:
        adapter, policy = load(args, {YOCO_KEY: 'yoco_v1', YOCO_LAYERS: args.layers,
                                      YOCO_KEEP: args.keep_rows, YOCO_MIN_ROWS: args.min_rows})
        record.update(policy=policy, profile=adapter.profile, numeric_profile=adapter.numeric_profile,
                      approximate_receipt=adapter.diagnostics()['source_approximate_operations'])
        model = adapter.model
        stream, corpus = corpus_tokens(adapter, sum(contexts) + len(contexts) * args.continuation + 1)
        record['corpus'] = corpus
        atomic(out / 'results.json', record)
        offset = 0
        for context in contexts:
            prompt = stream[offset:offset + context]
            continuation = stream[offset + context:offset + context + args.continuation]
            offset += context + args.continuation
            assert_varied(prompt)
            cell = dict(context=context, continuation=len(continuation), runs=[])
            arrays = {}
            for rep in range(args.reps):
                order = ('exact', 'yoco') if rep % 2 == 0 else ('yoco', 'exact')
                for arm in order:
                    mx.clear_cache()
                    cache = model.make_cache()
                    ids = mx.array([prompt], mx.uint32)
                    before = stream_counters(adapter)

                    def prefill(arm=arm, ids=ids, cache=cache):
                        if arm == 'exact':
                            value = exact_reference_prefill(model, ids, cache)
                        else:
                            value = model(ids, cache)
                        mx.eval(value, [row.state for row in cache])
                        return value
                    last, seconds = timed(prefill)
                    run = dict(rep=rep, arm=arm, prefill_seconds=seconds, expert=delta(before, stream_counters(adapter)),
                               yoco=dict(model.source_prefill_approx_last or {}) if arm == 'yoco' else None)
                    if rep == 0:
                        tail = model(mx.array([continuation[:-1]], mx.uint32), cache)
                        mx.eval(tail)
                        arrays[arm] = (logprobs(last[:, -1:]), logprobs(tail))
                    cell['runs'].append(run)
                    close_cache(cache)
                    print(json.dumps(dict(context=context, **run)), flush=True)
            import numpy as np
            exact = np.concatenate(arrays['exact'])
            approx = np.concatenate(arrays['yoco'])
            cell['quality'] = compare(exact, approx, np.asarray(continuation))
            times = {arm: [r['prefill_seconds'] for r in cell['runs'] if r['arm'] == arm] for arm in ('exact', 'yoco')}
            cell['prefill_mean_seconds'] = {k: sum(v) / len(v) for k, v in times.items()}
            cell['prefill_time_reduction'] = 1.0 - cell['prefill_mean_seconds']['yoco'] / cell['prefill_mean_seconds']['exact']
            record['cells'].append(cell)
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


if __name__ == '__main__':
    main()
