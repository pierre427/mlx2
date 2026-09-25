"""A/B: retiring committed GDN rollback records (40e2ea5b) on Flash-Next.

Arms: default (retire after each committed batched self-MTP cycle) and
--no-retire (restores the pre-fix behaviour: records live to the 64-token
window). The adapter is constructed before any runtime module is imported,
as the import-order guard requires. Records peak memory, active memory at the
end of decode (the retained records show up there), decode speed and tokens.

  PYTHONPATH=src .venv/bin/python scripts/bench_rollback_retention.py --i-own-the-gpu \
      --model ~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP --prompt-file p.txt \
      --context 16000 --batch 4 --gen 512 [--no-retire] --out run.json
"""

import argparse
import hashlib
import json
import time

import mlx.core as mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--context", type=int, default=16000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--num-draft", type=int, default=2)
    ap.add_argument("--no-retire", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")

    from mlx2.adapters.registry import resolve_adapter

    adapter = resolve_adapter(a.model, mtp=True)(a.model)
    mx.eval(adapter.model.parameters())
    from mlx2.runtime import generate as G
    from mlx2.runtime import hybrid_speculative as HS
    from mlx2.runtime.models.cache import ArraysCache
    from mlx2.runtime.sample_utils import LaneRNG

    retired = {"calls": 0, "dropped": 0}
    if a.no_retire:
        HS._retire_committed_rollbacks = lambda caches: None
    else:
        original = ArraysCache.retire_rollbacks

        def counted(self, keep=1):
            dropped = original(self, keep)
            retired["calls"] += 1
            retired["dropped"] += dropped
            return dropped

        ArraysCache.retire_rollbacks = counted
    print("LOADED", flush=True)

    ids = list(adapter.tokenizer.encode(open(a.prompt_file).read()))
    prompts = [ids[i * a.context : (i + 1) * a.context] for i in range(a.batch)]
    gen = G.BatchGenerator(
        adapter.model, completion_batch_size=a.batch, prefill_batch_size=1,
        prefill_step_size=2048,
        self_mtp={"num_draft": a.num_draft, "persistent": True, "rate_gate": False,
                  "prefill_step_size": 2048},
    )
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    uids = gen.insert(prompts, max_tokens=[a.gen] * a.batch,
                      lane_rngs=[LaneRNG(1 + i) for i in range(a.batch)],
                      self_mtp_configs=[{"sampling_temp": 0.0}] * a.batch)
    tokens, done, started = {}, set(), set()
    t_first = None
    emitted = 0
    active_samples = []
    step = 0
    try:
        while len(done) < a.batch:
            _p, responses = gen.next()
            now = time.perf_counter()
            started.update(r.uid for r in responses)
            if responses and t_first is None and started >= set(uids):
                t_first = now
                prefill_peak = mx.get_peak_memory()
                mx.reset_peak_memory()
            for r in responses:
                tokens.setdefault(r.uid, []).append(int(r.token))
                if r.finish_reason:
                    done.add(r.uid)
            if t_first is not None:
                emitted += len(responses)
                step += 1
                if step % 16 == 0:
                    active_samples.append(mx.get_active_memory() / 2**30)
        t_end = time.perf_counter()
        active_end = mx.get_active_memory() / 2**30
        decode_peak = mx.get_peak_memory() / 2**30
    finally:
        gen.close()
    rec = {
        "label": a.label, "no_retire": a.no_retire, "batch": a.batch, "context": a.context,
        "prefill_s": t_first - t0, "decode_s": t_end - t_first,
        "decode_tps": emitted / (t_end - t_first),
        "prefill_peak_gib": prefill_peak / 2**30, "decode_peak_gib": decode_peak,
        "active_end_gib": active_end, "active_samples_gib": active_samples,
        "retired": retired,
        "tokens_sha": [hashlib.sha256(json.dumps(tokens.get(u, [])).encode()).hexdigest()[:16] for u in uids],
        "mlx": mx.__version__,
    }
    json.dump(rec, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != "active_samples_gib"}))


if __name__ == "__main__":
    main()
