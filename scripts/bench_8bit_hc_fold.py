"""In-process decode A/B for the 8-bit HC kernels and 8-bit shared-expert fold.

Mixed-precision Flash-Next artifacts (the uncensored conversion) quantize the
hyper-connection projections and the shared expert at 8 bits.  Before 8-bit
support, both mechanisms declined every call there.  One model load, default
execution policy; per (route, batch) config both arms run per rep, order
alternated (ABBA), after a discarded warm-up:

  main   main's defaults as of a3f07334: the 8-bit admissions refused
         exactly as there (the HC layout check runs, then reports
         ``down: bits``; the shared-fold admission runs, then reports the
         4-bit-only format refusal) and 2..8-row HC calls served with the
         qmv_wide law (multi-row on)
  tree   this tree's defaults (8-bit HC rows and fold served; HC multi-row
         path off: 2..8-row calls stay composed)

Records greedy-token hashes (identity vs base), HC calls/declines, shared
fold calls/fallbacks, ms per generation step.  Prompts are token slices of a
real text file at the given offsets (content-varied reps).

  MLX_ENABLE_TF32=0 PYTHONPATH=src .venv/bin/python scripts/bench_8bit_hc_fold.py \\
      --i-own-the-gpu --model ~/mlx-models/Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP \\
      --prompt-file docs.txt --configs ordinary:1 mtp:1 ordinary:4 --out ab.json
"""

import argparse
import hashlib
import json
import statistics
import time

import mlx.core as mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--arms", nargs="+", default=["main", "tree"])
    ap.add_argument("--configs", nargs="+", default=["ordinary:1", "mtp:1", "ordinary:4"])
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--num-draft", type=int, default=2)
    ap.add_argument("--prefill-step", type=int, default=8192)
    ap.add_argument("--prompt-offsets", type=int, nargs="+", default=[0, 40000, 80000, 120000])
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")

    from mlx2.adapters.flash_next import FlashNextAdapter

    # The adapter pins its environment before any runtime model module is
    # imported (import_env guard), so construct it first.
    adapter = FlashNextAdapter(a.model)
    from mlx2.runtime import generate as G
    from mlx2.runtime.models import qwen3_next as QN
    from mlx2.runtime.models import qwen4_hc_decode as HCD
    from mlx2.runtime.models import qwen4_routed_decode as RD
    from mlx2.runtime.sample_utils import LaneRNG

    model = adapter.model
    mx.eval(model.parameters())
    mx.set_cache_limit(8 << 30)
    print("LOADED", f"active={mx.get_active_memory() / 2**30:.1f}GiB", "policy", adapter.policy.as_dict(),
          flush=True)
    all_ids = list(adapter.tokenizer.encode(open(a.prompt_file).read()))
    blocks = [m for _, m in model.named_modules() if isinstance(m, QN.Qwen3NextSparseMoeBlock)]

    real_plan = HCD._cached_plan
    real_admit = RD.admit_shared_fold

    def base_plan(module):
        reason, plan = real_plan(module)
        down, up = module.input_mix_weight_down, module.input_mix_weight_up
        if getattr(down, "bits", 4) != 4:
            return "down: bits", None
        if getattr(up, "bits", 4) != 4:
            return "up: bits", None
        return reason, plan

    def base_admit(shared, hidden, inter):
        admission = real_admit(shared, hidden, inter)
        for name in ("gate_proj", "up_proj", "down_proj"):
            layer = shared.get(name) if shared is not None else None
            bits = getattr(layer, "bits", 4)
            if admission.accepted and bits != 4:
                return RD.RoutedDecodeAdmission(False, f"shared {name}: format b{bits}g64 != b4g64")
        return admission

    tree_multi_row = HCD.set_hc_multi_row_enabled(False)
    HCD.set_hc_multi_row_enabled(tree_multi_row)

    def configure(arm):
        main = arm in ("main", "base")
        HCD._cached_plan = base_plan if main else real_plan
        RD.admit_shared_fold = base_admit if main else real_admit
        HCD.set_hc_multi_row_enabled(True if main else tree_multi_row)

    def counters():
        status = HCD.hc_decode_status()
        return (status["calls"], sum(status["declines"].values()),
                sum(b.shared_fold_calls for b in blocks), sum(b.shared_fold_fallbacks for b in blocks))

    def run(route, batch, offset=0):
        prompts = [all_ids[offset + i * a.context: offset + (i + 1) * a.context] for i in range(batch)]
        kwargs = dict(completion_batch_size=batch, prefill_batch_size=1, prefill_step_size=a.prefill_step)
        insert = {"max_tokens": [a.gen] * batch, "lane_rngs": [LaneRNG(1 + i) for i in range(batch)]}
        if route == "mtp":
            kwargs["self_mtp"] = adapter.policy.batch_config(max_lanes=batch, prefill_step=a.prefill_step)
            kwargs["self_mtp"]["num_draft"] = a.num_draft
            insert["self_mtp_configs"] = [{"sampling_temp": 0.0}] * batch
        gen = G.BatchGenerator(model, **kwargs)
        uids = gen.insert(prompts, **insert)
        tokens, done, started = {}, set(), set()
        t_first = t_end = None
        emitted = steps = 0
        try:
            while len(done) < batch:
                _p, responses = gen.next()
                now = time.perf_counter()
                if t_first is not None and responses:
                    steps += 1
                started.update(r.uid for r in responses)
                if responses and t_first is None and started >= set(uids):
                    t_first = now
                    emitted -= len(responses)
                for r in responses:
                    tokens.setdefault(r.uid, []).append(int(r.token))
                    if r.finish_reason:
                        done.add(r.uid)
                if t_first is not None:
                    emitted += len(responses)
                    t_end = now
        finally:
            gen.close()
        mx.clear_cache()
        lanes = [tokens.get(u, []) for u in uids]
        sha = hashlib.sha256(json.dumps(lanes).encode()).hexdigest()[:16]
        info = {"steps": steps, "ms_per_step": 1e3 * (t_end - t_first) / max(1, steps),
                "tokens_per_step": emitted / max(1, steps), "emitted": sum(map(len, lanes))}
        return emitted / (t_end - t_first), sha, lanes, info

    results = {}
    for config in a.configs:
        route, batch = config.split(":")
        batch = int(batch)
        configure(a.arms[0])
        run(route, batch)  # warm-up, discarded
        keys = ("tps", "sha", "ms_per_step", "tokens_per_step", "offset", "hc_calls", "hc_declines",
                "fold_calls", "fold_fallbacks", "emitted")
        per_arm = {arm: {k: [] for k in keys} for arm in a.arms}
        lanes_by = {arm: [] for arm in a.arms}
        for rep in range(a.reps):
            order = list(a.arms) if rep % 2 == 0 else list(a.arms)[::-1]
            offset = a.prompt_offsets[rep % len(a.prompt_offsets)]
            for arm in order:
                configure(arm)
                before = counters()
                tps, sha, lanes, info = run(route, batch, offset)
                after = counters()
                d = [x - y for x, y in zip(after, before)]
                rec = per_arm[arm]
                for k, v in (("tps", tps), ("sha", sha), ("offset", offset), ("hc_calls", d[0]),
                             ("hc_declines", d[1]), ("fold_calls", d[2]), ("fold_fallbacks", d[3])):
                    rec[k].append(v)
                for k in ("ms_per_step", "tokens_per_step", "emitted"):
                    rec[k].append(info[k])
                lanes_by[arm].append(lanes)
                print(f"{config} rep{rep} off{offset} {arm} {tps:.2f} tok/s {info['ms_per_step']:.2f} ms/step "
                      f"{info['tokens_per_step']:.3f} tok/step hc={d[0]}/{d[1]}decl fold={d[2]}/{d[3]}fb "
                      f"sha={sha}", flush=True)
        base_arm = a.arms[0]
        base = statistics.median(per_arm[base_arm]["tps"])
        summary = {}
        for arm, v in per_arm.items():
            med = statistics.median(v["tps"])
            diverge = None
            if arm != base_arm:
                for ref_rep, got_rep in zip(lanes_by[base_arm], lanes_by[arm]):
                    for lane_ref, lane in zip(ref_rep, got_rep):
                        k = next((i for i, (x, y) in enumerate(zip(lane_ref, lane)) if x != y), None)
                        if k is not None:
                            diverge = k if diverge is None else min(diverge, k)
            summary[arm] = {
                "median_tps": med, "min_tps": min(v["tps"]), "max_tps": max(v["tps"]),
                "median_ms_per_step": statistics.median(v["ms_per_step"]),
                "mean_tokens_per_step": statistics.mean(v["tokens_per_step"]),
                "delta_vs_base_pct": 100 * (med / base - 1),
                "hc_calls_per_token": v["hc_calls"][0] / max(1, v["emitted"][0]),
                "hc_declines_per_token": v["hc_declines"][0] / max(1, v["emitted"][0]),
                "fold_calls_per_run": v["fold_calls"][0], "fold_fallbacks_per_run": v["fold_fallbacks"][0],
                "tokens_identical_to_base": None if arm == base_arm else all(
                    s == r for s, r in zip(v["sha"], per_arm[base_arm]["sha"])),
                "first_divergence": diverge,
                "paired_tps_delta_pct": None if arm == base_arm else [
                    100 * (x / y - 1) for x, y in zip(v["tps"], per_arm[base_arm]["tps"])],
                "paired_ms_per_step_delta_pct": None if arm == base_arm else [
                    100 * (x / y - 1) for x, y in zip(v["ms_per_step"], per_arm[base_arm]["ms_per_step"])],
            }
        results[config] = {"runs": per_arm, "summary": summary}
        print(config, json.dumps(summary, indent=1), flush=True)
        json.dump({"partial": True, "results": results}, open(a.out, "w"), indent=1)
    configure("tree")
    json.dump({"model": a.model, "context": a.context, "gen": a.gen, "reps": a.reps,
               "num_draft": a.num_draft, "prompt_offsets": a.prompt_offsets,
               "policy": adapter.policy.as_dict(), "results": results,
               "peak_gib": mx.get_peak_memory() / 2**30, "mlx": mx.__version__},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
