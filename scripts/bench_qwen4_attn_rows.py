"""In-process decode A/B for the fused Qwen4 attention rows (omlx #4052 port).

One model load.  For each (route, context) config every arm runs once per rep
after a discarded warm-up (two arms alternate ABBA; more arms rotate).  Each run prefills ``context`` tokens from an empty
cache through the served BatchGenerator (B1), then decodes ``--gen`` greedy
tokens; decode tok/s is measured from the first decoded token.  Token
identity against the first arm is recorded per run.

Arms: ``stock`` (fused rows off, the shipped path) or ``fused`` (policy
attn_fused_rows), optionally ``@<tokens>`` to move the one-token indexed-QSA
threshold (MLX_QWEN4_QSA_INDEXED_AUTO_MIN_CONTEXT_M1, default 65536) for the
run, ``@inf`` keeping one-token steps on the masked arm at every context
(omlx #4062's idea).  e.g. ``stock fused fused@inf fused@16384``.

  PYTHONPATH=src MLX_ENABLE_TF32=0 python scripts/bench_qwen4_attn_rows.py \
      --i-own-the-gpu --model ~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP \
      --prompt-file p.txt --configs ordinary:1024 mtp:1024 ordinary:32768 \
      --arms stock fused --out ab.json
"""

import argparse
import hashlib
import json
import statistics
import time

import mlx.core as mx


# Polls without any response before a run is declared stuck (prefill polls
# return nothing).  A lane the generator drops (a non-finite sampled row) never
# emits a finish_reason, so a loop waiting for every lane would spin forever;
# same rule as scripts/paired_direct_ab.py.
IDLE_POLL_LIMIT = 4096


def _drain_poll(gen, idle):
    """One ``gen.next()``: raise on a dropped lane or a stuck generator."""
    _p, responses = gen.next()
    take = getattr(gen, "take_lane_failures", None)
    lost = take() if take is not None else []
    if lost:
        raise RuntimeError("generator dropped lane(s): " + "; ".join(str(f) for f in lost))
    idle = 0 if responses else idle + 1
    if idle > IDLE_POLL_LIMIT:
        raise RuntimeError(f"no response in {IDLE_POLL_LIMIT} polls")
    return responses, idle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--configs", nargs="+", default=["ordinary:1024", "mtp:1024"])
    ap.add_argument("--arms", nargs="+", default=["stock", "fused"])
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--prefill-step", type=int, default=8192)
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime import generate as G
    from mlx2.runtime.models import qwen4_attn_rows as R
    from mlx2.runtime.models import qwen4_qsa_indexed as QI
    from mlx2.runtime.sample_utils import LaneRNG

    adapter = resolve_adapter(a.model, mtp=True)(a.model)
    model = adapter.model
    mx.eval(model.parameters())
    mx.set_cache_limit(4 << 30)
    ids = list(adapter.tokenizer.encode(open(a.prompt_file).read()))
    default_m1 = QI._AUTO_MIN_CONTEXT_M1
    print("LOADED", f"tokens={len(ids)}", f"active={mx.get_active_memory() / 2**30:.1f}GiB",
          f"m1_threshold={default_m1}", flush=True)

    def configure(arm):
        base, _, threshold = arm.partition("@")
        if base not in ("stock", "fused"):
            raise ValueError(f"unknown arm {arm!r}")
        R.set_enabled(base == "fused")
        if not threshold:
            QI._AUTO_MIN_CONTEXT_M1 = default_m1
        else:
            QI._AUTO_MIN_CONTEXT_M1 = 2**31 - 1 if threshold == "inf" else int(threshold)

    def run(route, context):
        prompt = ids[:context]
        kwargs = {}
        if route == "mtp":
            kwargs["self_mtp"] = adapter.execution_config(max_lanes=1, prefill_step=a.prefill_step)
        gen = G.BatchGenerator(model, completion_batch_size=1, prefill_batch_size=1,
                               prefill_step_size=a.prefill_step, **kwargs)
        insert = {"max_tokens": [a.gen], "lane_rngs": [LaneRNG(1)]}
        if route == "mtp":
            insert["self_mtp_configs"] = [{"sampling_temp": 0.0}]
        gen.insert([prompt], **insert)
        tokens = []
        t_first = t_end = None
        steps = 0
        try:
            done, idle = False, 0
            while not done:
                responses, idle = _drain_poll(gen, idle)
                now = time.perf_counter()
                if responses:
                    if t_first is None:
                        t_first = now
                    else:
                        steps += 1
                        t_end = now
                for r in responses:
                    tokens.append(int(r.token))
                    if r.finish_reason:
                        done = True
        finally:
            gen.close()
        mx.clear_cache()
        emitted = len(tokens) - 1
        tps = emitted / (t_end - t_first) if t_end else 0.0
        sha = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()[:16]
        return {"tps": tps, "sha": sha, "tokens": tokens, "steps": steps,
                "tokens_per_step": emitted / max(1, steps),
                "ms_per_token": 1e3 * (t_end - t_first) / max(1, emitted) if t_end else None}

    results = {}
    for config in a.configs:
        route, context = config.split(":")
        context = int(context)
        configure(a.arms[0])
        run(route, min(context, 2048))  # warm-up, discarded
        per_arm = {arm: [] for arm in a.arms}
        for rep in range(a.reps):
            if len(a.arms) == 2:
                # ABBA: A B, B A, A B, ...
                order = a.arms if rep % 2 == 0 else a.arms[::-1]
            else:
                # each arm takes each position once every len(arms) reps
                order = a.arms[rep % len(a.arms):] + a.arms[: rep % len(a.arms)]
            for arm in order:
                configure(arm)
                R.status(reset=True)
                res = run(route, context)
                res["attn_rows_counts"] = R.status()["counts"]
                per_arm[arm].append(res)
                print(f"{config} rep{rep} {arm} {res['tps']:.2f} tok/s "
                      f"{res['tokens_per_step']:.3f} tok/step sha={res['sha']} "
                      f"counts={ {k: v for k, v in res['attn_rows_counts'].items() if not k.startswith('fallback')} }",
                      flush=True)
        ref = per_arm[a.arms[0]][0]["tokens"]
        summary = {}
        base = statistics.median(r["tps"] for r in per_arm[a.arms[0]])
        for arm, runs in per_arm.items():
            tps = [r["tps"] for r in runs]
            diverge = None
            for r in runs:
                k = next((i for i, (x, y) in enumerate(zip(ref, r["tokens"])) if x != y), None)
                if k is not None:
                    diverge = k if diverge is None else min(diverge, k)
            summary[arm] = {
                "median_tps": statistics.median(tps), "min_tps": min(tps), "max_tps": max(tps),
                "delta_vs_first_pct": 100 * (statistics.median(tps) / base - 1),
                "paired_delta_pct": [100 * (x["tps"] / y["tps"] - 1)
                                     for x, y in zip(runs, per_arm[a.arms[0]])],
                "tokens_identical_to_first_arm": all(r["tokens"] == ref for r in runs),
                "first_divergence": diverge,
                "mean_tokens_per_step": statistics.mean(r["tokens_per_step"] for r in runs),
            }
        for runs in per_arm.values():
            for r in runs:
                r.pop("tokens")
        results[config] = {"runs": per_arm, "summary": summary}
        print(config, json.dumps(summary, indent=1), flush=True)
        json.dump({"gen": a.gen, "reps": a.reps, "arms": a.arms, "results": results,
                   "peak_gib": mx.get_peak_memory() / 2**30, "mlx": mx.__version__},
                  open(a.out, "w"), indent=1)
    configure("stock")


if __name__ == "__main__":
    main()
