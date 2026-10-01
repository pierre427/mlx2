"""In-process long-context decode A/B for the TensorFold 0.6.1 Flash-Next intake.

One model load, the served Flash-Next policy (default execution policy).
Each config ``route:lanes:context`` prefills ``context`` real tokens per lane
(slices of ``--prompt-file`` at rotating offsets, one distinct slice per lane)
through the served BatchGenerator, then decodes ``--gen`` greedy tokens; decode
tok/s is measured from the first decoded token.  Arms toggle the intake's
runtime switches; every arm runs once per rep after a discarded warm-up, in
rotated order (two arms: ABBA).  Records tok/s, ms/step, tokens/step, host CPU
ms per step, engagement counters, and greedy-token identity per run against
the first arm on the same prompt.

Arms: ``base`` (all intake switches off: main's route), ``scores``
(qsa_fused_scores), ``ple`` (ple_early_dispatch), ``all`` (both).

  MLX_ENABLE_TF32=0 PYTHONPATH=src python scripts/bench_tf_longctx.py \\
      --i-own-the-gpu --model ~/mlx-models/Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP \\
      --prompt-file prompt.txt --configs mtp:1:65536 ordinary:1:65536 --out ab.json
"""

import argparse
import hashlib
import json
import statistics
import time

import mlx.core as mx

ARMS = {
    "base": {"scores": False, "ple": False},
    "scores": {"scores": True, "ple": False},
    "ple": {"scores": False, "ple": True},
    "all": {"scores": True, "ple": True},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--configs", nargs="+", default=["mtp:1:65536"])
    ap.add_argument("--arms", nargs="+", default=["base", "all"])
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--prefill-step", type=int, default=8192)
    ap.add_argument("--prompt-offsets", type=int, nargs="+", default=[0, 300000, 600000, 850000])
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    for arm in a.arms:
        if arm not in ARMS:
            ap.error(f"unknown arm {arm!r}")

    from mlx2.adapters.flash_next import FlashNextAdapter

    # The adapter pins its environment before any runtime model module is
    # imported (import_env guard), so construct it first.
    adapter = FlashNextAdapter(a.model)
    from mlx2.runtime import generate as G
    from mlx2.runtime.models import qwen4_exp as Q
    from mlx2.runtime.models import qwen4_qsa_scores as S
    from mlx2.runtime.sample_utils import LaneRNG

    model = adapter.model
    mx.eval(model.parameters())
    mx.set_cache_limit(4 << 30)
    ids = list(adapter.tokenizer.encode(open(a.prompt_file).read()))
    print("LOADED", f"tokens={len(ids)}", f"active={mx.get_active_memory() / 2**30:.1f}GiB",
          "policy", adapter.policy.as_dict(), flush=True)

    def configure(arm):
        S.set_enabled(ARMS[arm]["scores"])
        Q.set_ple_early_dispatch(ARMS[arm]["ple"])

    def counters():
        sc = S.status()["counts"]
        ple = dict(Q._PLE_EARLY_STATS)
        return {"scores_engaged": sc.get("engaged", 0),
                "scores_fallbacks": sum(v for k, v in sc.items() if k.startswith("fallback_")),
                "scores_fallback_reasons": {k: v for k, v in sc.items() if k.startswith("fallback_")},
                "ple_dispatches": ple.get("dispatches", 0), "ple_device_ids": ple.get("device_ids", 0)}

    def run(route, lanes, context, offset, gen):
        prompts = [ids[offset + i * context: offset + (i + 1) * context] for i in range(lanes)]
        assert all(len(p) == context for p in prompts), "prompt file too short"
        kwargs = dict(completion_batch_size=lanes, prefill_batch_size=1, prefill_step_size=a.prefill_step)
        insert = {"max_tokens": [gen] * lanes, "lane_rngs": [LaneRNG(1 + i) for i in range(lanes)]}
        if route == "mtp":
            kwargs["self_mtp"] = adapter.execution_config(max_lanes=lanes, prefill_step=a.prefill_step)
            insert["self_mtp_configs"] = [{"sampling_temp": 0.0}] * lanes
        S.status(reset=True)
        Q._PLE_EARLY_STATS.clear()
        g = G.BatchGenerator(model, **kwargs)
        uids = g.insert(prompts, **insert)
        tokens, done, started = {}, set(), set()
        t_first = t_end = None
        c_first = None
        emitted = steps = 0
        try:
            while len(done) < lanes:
                _p, responses = g.next()
                now = time.perf_counter()
                if t_first is not None and responses:
                    steps += 1
                started.update(r.uid for r in responses)
                if responses and t_first is None and started >= set(uids):
                    t_first, c_first = now, time.process_time()
                    emitted -= len(responses)
                for r in responses:
                    tokens.setdefault(r.uid, []).append(int(r.token))
                    if r.finish_reason:
                        done.add(r.uid)
                if t_first is not None:
                    emitted += len(responses)
                    t_end = now
            cpu = time.process_time() - c_first
        finally:
            g.close()
        mx.clear_cache()
        lane_tokens = [tokens.get(u, []) for u in uids]
        sha = hashlib.sha256(json.dumps(lane_tokens).encode()).hexdigest()[:16]
        wall = t_end - t_first
        return {"tps": emitted / wall, "ms_per_step": 1e3 * wall / max(1, steps),
                "cpu_ms_per_step": 1e3 * cpu / max(1, steps),
                "tokens_per_step": emitted / max(1, steps), "steps": steps, "sha": sha,
                "offset": offset, "counters": counters()}, lane_tokens

    results = {}
    for config in a.configs:
        route, lanes, context = config.split(":")
        lanes, context = int(lanes), int(context)
        configure(a.arms[0])
        run(route, lanes, min(context, 4096), 0, 32)  # warm-up, discarded
        for arm in a.arms[1:]:  # build each arm's kernels before timing
            configure(arm)
            run(route, lanes, min(context, 4096), 0, 32)
        per_arm = {arm: [] for arm in a.arms}
        lanes_by = {arm: [] for arm in a.arms}
        for rep in range(a.reps):
            if len(a.arms) == 2:
                order = a.arms if rep % 2 == 0 else a.arms[::-1]
            else:
                k = rep % len(a.arms)
                order = a.arms[k:] + a.arms[:k]
            offset = a.prompt_offsets[rep % len(a.prompt_offsets)]
            for arm in order:
                configure(arm)
                rec, lane_tokens = run(route, lanes, context, offset, a.gen)
                per_arm[arm].append(rec)
                lanes_by[arm].append(lane_tokens)
                print(f"{config} rep{rep} off{offset} {arm} {rec['tps']:.2f} tok/s {rec['ms_per_step']:.2f} ms/step "
                      f"{rec['tokens_per_step']:.3f} tok/step cpu {rec['cpu_ms_per_step']:.2f} ms/step "
                      f"sha={rec['sha']} {rec['counters']}", flush=True)
        base_arm = a.arms[0]
        summary = {}
        base_med = statistics.median(r["tps"] for r in per_arm[base_arm])
        for arm, runs in per_arm.items():
            tps = [r["tps"] for r in runs]
            diverge = None
            identical = True
            if arm != base_arm:
                for ref_rep, got_rep in zip(lanes_by[base_arm], lanes_by[arm]):
                    for lane_ref, lane in zip(ref_rep, got_rep):
                        if lane_ref != lane:
                            identical = False
                        k = next((i for i, (x, y) in enumerate(zip(lane_ref, lane)) if x != y), None)
                        if k is not None:
                            diverge = k if diverge is None else min(diverge, k)
            paired = None if arm == base_arm else [
                100 * (x["tps"] / y["tps"] - 1) for x, y in zip(runs, per_arm[base_arm])]
            summary[arm] = {
                "median_tps": statistics.median(tps), "min_tps": min(tps), "max_tps": max(tps),
                "median_delta_pct": 100 * (statistics.median(tps) / base_med - 1),
                "paired_delta_pct": paired,
                "paired_delta_median_pct": None if paired is None else statistics.median(paired),
                "median_ms_per_step": statistics.median(r["ms_per_step"] for r in runs),
                "median_cpu_ms_per_step": statistics.median(r["cpu_ms_per_step"] for r in runs),
                "mean_tokens_per_step": statistics.mean(r["tokens_per_step"] for r in runs),
                "tokens_identical_to_base": None if arm == base_arm else identical,
                "first_divergence": diverge,
                "counters_first_run": runs[0]["counters"],
            }
        results[config] = {"runs": per_arm, "summary": summary}
        print(config, json.dumps(summary, indent=1), flush=True)
        json.dump({"partial": True, "results": results}, open(a.out, "w"), indent=1)
    configure("base")
    json.dump({"model": a.model, "gen": a.gen, "reps": a.reps, "arms": a.arms,
               "prompt_offsets": a.prompt_offsets, "policy": adapter.policy.as_dict(),
               "results": results, "peak_gib": mx.get_peak_memory() / 2**30, "mlx": mx.__version__},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
