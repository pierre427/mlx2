"""In-process decode A/B for the omlx #3912 port on Flash-Next (one model load).

``--phase speed``: for each (route, batch) config, runs every arm (the MoE
routed-decode mode: off | gate_up | two_launch) per rep, rotating the order
(ABBA-style: rep r starts at arm r mod n, odd reps reversed) after a discarded
warm-up run. Each run prefills ``--context`` tokens per lane from an empty
cache and decodes ``--gen`` greedy tokens per lane; decode tok/s is measured
once every lane is decoding. Records greedy-token hashes and the routed-decode
call count per run.

``--phase quality``: teacher-forced gate. Prefill ``--context`` tokens (the
routed kernels cannot engage on prefill rows), then feed the next
``--score-tokens`` ground-truth tokens one at a time (M=1, where they do
engage) and collect logprobs; per arm report KL(off || arm), top-1 agreement
with off, and the mean NLL of the ground-truth tokens.

  PYTHONPATH=src .venv/bin/python scripts/bench_fn_routed_decode.py --i-own-the-gpu \
      --model ~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP --prompt-file p.txt \
      --phase speed --out speed.json
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
    ap.add_argument("--phase", choices=("speed", "quality"), required=True)
    ap.add_argument("--knob", choices=("moe_routed", "gdn_batch"), default="moe_routed",
                    help="gdn_batch: arms are fused GDN batched-decode modes (off | row_exact)")
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--configs", nargs="+", default=["ordinary:1", "mtp:1", "ordinary:4", "mtp:4"])
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--gen", type=int, default=384)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--prefill-step", type=int, default=8192)
    ap.add_argument("--score-tokens", type=int, default=256)
    ap.add_argument("--offsets", type=int, nargs="+", default=[0, 30000, 60000],
                    help="quality: prompt offsets; speed: with --prompt-offsets, per-rep prompts")
    ap.add_argument("--prompt-offsets", type=int, nargs="+", default=None,
                    help="speed: rep r uses the prompt at offset[r %% n] (content-varied reps)")
    ap.add_argument("--cache-limit-gib", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if a.arms is None:
        a.arms = ["off", "row_exact"] if a.knob == "gdn_batch" else ["off", "gate_up", "two_launch"]
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")

    from mlx2.adapters.registry import resolve_adapter

    adapter = resolve_adapter(a.model, mtp=True)(a.model)
    model = adapter.model
    mx.eval(model.parameters())
    mx.set_cache_limit(a.cache_limit_gib << 30)
    blocks = [m for _, m in model.named_modules() if hasattr(m, "set_moe_routed_decode_mode")]
    print("LOADED", len(blocks), "moe blocks", f"active={mx.get_active_memory() / 2**30:.1f}GiB", flush=True)
    all_ids = list(adapter.tokenizer.encode(open(a.prompt_file).read()))

    default_expert_mode = blocks[0].fused_expert_kernel_mode

    gdn_layers = [m for _, m in model.named_modules() if hasattr(m, "set_fused_gdn_batch_decode_mode")]

    def configure(arm):
        if a.knob == "gdn_batch":
            # Arms: the fused GDN batched one-token decode mode (off | row_exact).
            for layer in gdn_layers:
                layer.set_fused_gdn_batch_decode_mode(arm)
            return
        # Arms: a routed-decode mode (off | gate_up | two_launch), or
        # "stock_down": routed off and the fused-expert tile4 down replaced by
        # the stock gather_qmm + weighted sum (the numerics two_launch matches).
        for b in blocks:
            b.set_moe_routed_decode_mode("off" if arm == "stock_down" else arm)
            b.set_fused_expert_kernel_mode("stock" if arm == "stock_down" else default_expert_mode)

    def calls():
        if a.knob == "gdn_batch":
            return sum(m.fused_gdn_batch_decode_calls for m in gdn_layers), sum(
                m.fused_gdn_batch_decode_fallbacks for m in gdn_layers)
        return sum(b.switch_mlp.routed_decode_calls for b in blocks), sum(
            b.switch_mlp.routed_decode_fallbacks for b in blocks)

    if a.phase == "quality":
        from mlx.utils import tree_flatten

        from mlx2.runtime.models.cache import make_prompt_cache

        def score(ids):
            prompt = mx.array(ids[: a.context], mx.uint32)[None]
            tail = ids[a.context : a.context + a.score_tokens + 1]
            cache = make_prompt_cache(model)
            pos = 0
            while pos < prompt.shape[1]:
                out = model(prompt[:, pos : pos + a.prefill_step], cache=cache)
                mx.eval(out, [v for _, v in tree_flatten([getattr(c, "state", None) for c in cache]) if isinstance(v, mx.array)])
                pos += a.prefill_step
            rows = []
            for t in tail[:-1]:
                logits = model(mx.array([[t]], mx.uint32), cache=cache)[0, -1].astype(mx.float32)
                lp = logits - mx.logsumexp(logits)
                mx.eval(lp)
                rows.append(lp)
            del cache
            mx.clear_cache()
            return mx.stack(rows), mx.array(tail[1:])

        report = {}
        for offset in a.offsets:
            ids = all_ids[offset:]
            ref = None
            for arm in a.arms:
                configure(arm)
                before = calls()
                lp, truth = score(ids)
                after = calls()
                nll = -mx.take_along_axis(lp, truth[:, None], axis=-1).mean().item()
                entry = {"offset": offset, "nll": nll, "routed_calls": after[0] - before[0],
                         "routed_fallbacks": after[1] - before[1]}
                if arm == "off":
                    ref = lp
                if ref is not None:
                    kl = (mx.exp(ref) * (ref - lp)).sum(axis=-1)
                    entry.update({
                        "kl_mean_vs_off": kl.mean().item(), "kl_max_vs_off": kl.max().item(),
                        "top1_agree_vs_off": (mx.argmax(lp, -1) == mx.argmax(ref, -1)).astype(mx.float32).mean().item(),
                        "bit_identical_vs_off": bool(mx.array_equal(lp, ref).item()),
                    })
                report.setdefault(arm, []).append(entry)
                print(offset, arm, json.dumps(entry), flush=True)
        configure("off")
        summary = {arm: {
            "nll_mean": statistics.mean(e["nll"] for e in v),
            "kl_mean_vs_off": statistics.mean(e.get("kl_mean_vs_off", 0.0) for e in v),
            "kl_max_vs_off": max(e.get("kl_max_vs_off", 0.0) for e in v),
            "top1_agree_vs_off": statistics.mean(e.get("top1_agree_vs_off", 1.0) for e in v),
        } for arm, v in report.items()}
        json.dump({"phase": "quality", "context": a.context, "score_tokens": a.score_tokens,
                   "offsets": a.offsets, "arms": report, "summary": summary,
                   "peak_gib": mx.get_peak_memory() / 2**30, "mlx": mx.__version__},
                  open(a.out, "w"), indent=1)
        print(json.dumps(summary, indent=1))
        return

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
                    "tokens_per_step": emitted / max(1, steps)}
        return emitted / (t_end - t_first), sha, [tokens.get(u, []) for u in uids]

    results = {}
    for config in a.configs:
        route, batch = config.split(":")
        batch = int(batch)
        configure(a.arms[0])
        run(route, batch)  # warm-up, discarded
        per_arm = {arm: {"tps": [], "sha": [], "calls": [], "fallbacks": [], "ms_per_step": [],
                         "tokens_per_step": [], "offset": []} for arm in a.arms}
        first_tokens = {}
        for rep in range(a.reps):
            order = a.arms[rep % len(a.arms):] + a.arms[: rep % len(a.arms)]
            if rep % 2 and len(a.arms) > 2:
                # With two arms the rotation alone alternates AB/BA; reversing
                # too would restore AB on every rep (position bias).
                order = order[::-1]
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
                per_arm[arm]["fallbacks"].append(after[1] - before[1])
                first_tokens.setdefault(arm, toks)
                print(f"{config} rep{rep} off{offset} {arm} {tps:.2f} tok/s {run.last['ms_per_step']:.2f} ms/step "
                      f"{run.last['tokens_per_step']:.3f} tok/step calls={after[0] - before[0]} sha={sha}", flush=True)
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
                            "routed_calls_per_run": v["calls"][0],
                            "fallbacks_per_run": v["fallbacks"],
                            "lanes_identical_to_off": None if arm == "off" or "off" not in first_tokens else [
                                x == y for x, y in zip(first_tokens["off"], first_tokens[arm])],
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
    json.dump({"phase": "speed", "knob": a.knob, "context": a.context, "gen": a.gen, "reps": a.reps,
               "prefill_step": a.prefill_step, "results": results,
               "peak_gib": mx.get_peak_memory() / 2**30, "mlx": mx.__version__},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
