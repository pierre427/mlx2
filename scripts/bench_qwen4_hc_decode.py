"""In-process decode A/B for the two-launch HC decode (omlx #4038 port) on Flash-Next.

One model load.  For each (route, batch) config, runs both arms per rep,
alternating the order (ABBA) after a discarded warm-up run.  Arms:

  off  the composed GatedResidual ops (stock)
  on   qwen4_hc_decode's two launches for one-row HC calls (verify rows and
       anything else not admitted stay composed and are counted)

Each run prefills ``--context`` tokens per lane from an empty cache and decodes
``--gen`` greedy tokens per lane; decode tok/s is measured once every lane is
decoding.  Records greedy-token hashes (token identity vs off), the HC kernel
call/decline counts per run, and ms per generation step.

  PYTHONPATH=src .venv/bin/python scripts/bench_qwen4_hc_decode.py --i-own-the-gpu \
      --model ~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP --prompt-file p.txt --out ab.json
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
    ap.add_argument("--arms", nargs="+", default=["off", "on"])
    ap.add_argument("--configs", nargs="+", default=["ordinary:1", "mtp:1"])
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--prefill-step", type=int, default=8192)
    ap.add_argument("--prompt-offsets", type=int, nargs="+", default=None,
                    help="rep r uses the prompt at offset[r %% n] (content-varied reps)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.models import qwen4_hc_decode as HCD

    adapter = resolve_adapter(a.model, mtp=True)(a.model)
    model = adapter.model
    mx.eval(model.parameters())
    mx.set_cache_limit(8 << 30)
    print("LOADED", f"active={mx.get_active_memory() / 2**30:.1f}GiB", flush=True)
    all_ids = list(adapter.tokenizer.encode(open(a.prompt_file).read()))

    def configure(arm):
        HCD.set_hc_decode_enabled(arm == "on")

    def calls():
        status = HCD.hc_decode_status()
        return status["calls"], sum(status["declines"].values()), status["inject_calls"]

    from mlx2.runtime import generate as G
    from mlx2.runtime.sample_utils import LaneRNG

    def run(route, batch, offset=0):
        prompts = [all_ids[offset + i * a.context : offset + (i + 1) * a.context] for i in range(batch)]
        kwargs = {}
        if route == "mtp":
            kwargs["self_mtp"] = {"num_draft": 2, "persistent": True, "rate_gate": False,
                                  "prefill_step_size": a.prefill_step}
        gen = G.BatchGenerator(model, completion_batch_size=batch, prefill_batch_size=1,
                               prefill_step_size=a.prefill_step, **kwargs)
        insert = {"max_tokens": [a.gen] * batch,
                  "lane_rngs": [LaneRNG(1 + i) for i in range(batch)]}
        if route == "mtp":
            insert["self_mtp_configs"] = [{"sampling_temp": 0.0}] * batch
        uids = gen.insert(prompts, **insert)
        tokens, done, started = {}, set(), set()
        t_first = t_end = None
        emitted = 0
        steps = 0
        try:
            while len(done) < batch:
                _p, responses = gen.next()
                now = time.perf_counter()
                if t_first is not None and responses:
                    steps += 1
                started.update(r.uid for r in responses)
                if responses and t_first is None and started >= set(uids):
                    t_first = now
                    emitted -= sum(1 for _ in responses)
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
        sha = hashlib.sha256(json.dumps([tokens.get(u, []) for u in uids]).encode()).hexdigest()[:16]
        run.last = {"steps": steps, "ms_per_step": 1e3 * (t_end - t_first) / max(1, steps),
                    "tokens_per_step": emitted / max(1, steps),
                    "emitted": sum(len(tokens.get(u, [])) for u in uids)}
        return emitted / (t_end - t_first), sha, [tokens.get(u, []) for u in uids]

    results = {}
    for config in a.configs:
        route, batch = config.split(":")
        batch = int(batch)
        configure(a.arms[0])
        run(route, batch)  # warm-up, discarded
        per_arm = {arm: {"tps": [], "sha": [], "calls": [], "declines": [], "inject_calls": [],
                         "emitted": [], "ms_per_step": [], "tokens_per_step": [], "offset": []}
                   for arm in a.arms}
        first_tokens = {}
        for rep in range(a.reps):
            # ABBA: even reps run the arms in order, odd reps reversed.
            order = list(a.arms) if rep % 2 == 0 else list(a.arms)[::-1]
            for arm in order:
                configure(arm)
                before = calls()
                offset = a.prompt_offsets[rep % len(a.prompt_offsets)] if a.prompt_offsets else 0
                tps, sha, toks = run(route, batch, offset)
                after = calls()
                per_arm[arm]["ms_per_step"].append(run.last["ms_per_step"])
                per_arm[arm]["tokens_per_step"].append(run.last["tokens_per_step"])
                per_arm[arm]["offset"].append(offset)
                per_arm[arm]["tps"].append(tps)
                per_arm[arm]["sha"].append(sha)
                per_arm[arm]["calls"].append(after[0] - before[0])
                per_arm[arm]["declines"].append(after[1] - before[1])
                per_arm[arm]["inject_calls"].append(after[2] - before[2])
                per_arm[arm]["emitted"].append(run.last["emitted"])
                first_tokens.setdefault(arm, toks)
                print(f"{config} rep{rep} off{offset} {arm} {tps:.2f} tok/s {run.last['ms_per_step']:.2f} ms/step "
                      f"{run.last['tokens_per_step']:.3f} tok/step hc_calls={after[0] - before[0]} hc_declines={after[1] - before[1]} sha={sha}", flush=True)
        base = statistics.median(per_arm["off"]["tps"]) if "off" in per_arm else None
        summary = {}
        for arm, v in per_arm.items():
            med = statistics.median(v["tps"])
            diverge = None
            if "off" in first_tokens and arm != "off":
                for lane_ref, lane in zip(first_tokens["off"], first_tokens[arm]):
                    k = next((i for i, (x, y) in enumerate(zip(lane_ref, lane)) if x != y), None)
                    if k is not None:
                        diverge = k if diverge is None else min(diverge, k)
            summary[arm] = {"median_tps": med,
                            "median_ms_per_step": statistics.median(v["ms_per_step"]),
                            "mean_tokens_per_step": statistics.mean(v["tokens_per_step"]), "min_tps": min(v["tps"]), "max_tps": max(v["tps"]),
                            "delta_vs_off_pct": None if base is None else 100 * (med / base - 1),
                            "hc_calls_per_run": v["calls"][0],
                            "hc_declines_per_run": v["declines"][0],
                            "hc_calls_per_token": v["calls"][0] / max(1, v["emitted"][0]),
                            "hc_declines_per_token": v["declines"][0] / max(1, v["emitted"][0]),
                            "tokens_identical_to_off": None if arm == "off" else all(
                                s == r for s, r in zip(v["sha"], per_arm["off"]["sha"])),
                            "first_divergence": diverge,
                            "paired_tps_delta_pct": None if arm == "off" else [
                                100 * (x / y - 1) for x, y in zip(v["tps"], per_arm["off"]["tps"])],
                            "paired_ms_per_step_delta_pct": None if arm == "off" else [
                                100 * (x / y - 1) for x, y in zip(v["ms_per_step"], per_arm["off"]["ms_per_step"])]}
        results[config] = {"runs": per_arm, "summary": summary}
        print(config, json.dumps(summary, indent=1), flush=True)
    configure("off")
    json.dump({"phase": "speed", "context": a.context, "hc_status": HCD.hc_decode_status(), "gen": a.gen, "reps": a.reps,
               "prefill_step": a.prefill_step, "results": results,
               "peak_gib": mx.get_peak_memory() / 2**30, "mlx": mx.__version__},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
