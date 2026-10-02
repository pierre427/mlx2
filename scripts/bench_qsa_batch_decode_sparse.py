"""In-process A/B of the batched one-token sparse QSA arm (oMLX #4070 port).

One model load, the served Flash-Next policy.  Each config ``route:lanes:context``
prefills ``context`` real tokens per lane (one distinct slice of
``--prompt-file`` per lane) through the served BatchGenerator and decodes
``--gen`` greedy tokens.  Route ``mtp`` runs native MTP with the served
MTP->ordinary handoff (max MTP width 3), route ``ordinary`` plain decode.

Per config, after a discarded short warm-up per arm:
  * one full run per arm (``base`` first, then each ``--arms`` arm): greedy
    tokens for identity against base, aggregate decode tok/s;
  * one ALTERNATING run: the arm switches every ``--window`` response polls
    (base, arm, base, arm, ...; the first ``--settle`` polls of each window are
    dropped as possibly dispatched under the previous arm), giving paired
    per-window tok/s on the same rows at the same depth.  Only with exactly
    one non-base arm.
Before a config runs, the serving memory admission
(``SelfMTPLaneAdmissionController`` with the adapter's cache estimator, live
headroom) is asked to seat the cohort atomically; a config it would not seat
is recorded as refused and skipped.  A run of a non-base arm whose mechanism
counter stayed 0 is marked invalid.

  MLX_ENABLE_TF32=0 PYTHONPATH=src python scripts/bench_qsa_batch_decode_sparse.py \\
      --i-own-the-gpu --model ~/mlx-models/Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP \\
      --prompt-file prompt.txt --configs ordinary:4:32768 --arms gather --out ab.json
"""

import argparse
import hashlib
import json
import statistics
import time

import mlx.core as mx

IDLE_POLL_LIMIT = 4096


def _drain_poll(gen, idle):
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
    ap.add_argument("--configs", nargs="+", default=["ordinary:4:32768"])
    ap.add_argument("--arms", nargs="+", default=["gather"], choices=["gather", "indexed"])
    ap.add_argument("--min-context", type=int, default=16384)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--alt-gen", type=int, default=384)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--settle", type=int, default=2)
    ap.add_argument("--no-full-runs", action="store_true")
    ap.add_argument("--prefill-step", type=int, default=8192)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--handoff-width", type=int, default=3)
    ap.add_argument("--admission-table", action="store_true",
                    help="print what admission seats at 4/8 lanes x 16K..64K first")
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")

    from mlx2.adapters.flash_next import FlashNextAdapter

    adapter = FlashNextAdapter(a.model)
    from mlx2.memory import execution_headroom, host_memory_gib, metal_advisory_gib
    from mlx2.runtime import generate as G
    from mlx2.runtime.adaptive_policy import MTPOrdinaryHandoffPolicy
    from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController
    from mlx2.runtime.models import qwen4_exp as Q
    from mlx2.runtime.models import qwen4_qsa_indexed as I
    from mlx2.runtime.sample_utils import LaneRNG

    model = adapter.model
    mx.eval(model.parameters())
    mx.set_cache_limit(4 << 30)
    ids = list(adapter.tokenizer.encode(open(a.prompt_file).read()))
    print("LOADED", f"tokens={len(ids)}", f"active={mx.get_active_memory() / 2**30:.1f}GiB",
          "policy", adapter.policy.as_dict(), flush=True)

    budget = adapter.cache_budget(mtp=True)
    controller = SelfMTPLaneAdmissionController(
        host_memory_gib=host_memory_gib(), advisory_gib=metal_advisory_gib(),
        cache_estimator=budget.project,
        transient_gib_per_lane=getattr(budget, "transient_gib_per_lane",
                                       SelfMTPLaneAdmissionController.K2_TRANSIENT_GIB_PER_LANE))

    def admission(route, lanes, context):
        mx.clear_cache()
        free = execution_headroom() / 2**30
        decision = controller.decide([context] * lanes, free, atomic_cohort=True,
                                     max_draft=2 if route == "mtp" else 0)
        return {"free_gib": round(free, 2), "stage": decision.stage,
                "modes": list(decision.modes), "estimated_gib": round(decision.estimated_gib, 2),
                "usable_gib": round(decision.usable_gib, 2),
                "seated": all(m != "queue" for m in decision.modes)}

    def configure(arm):
        Q.set_qsa_batch_decode_sparse("off" if arm == "base" else arm, min_context=a.min_context)

    def indexed_widths():
        status = I.qsa_indexed_status()
        return {"counts": dict(status["counts"]), "widths": json.loads(json.dumps(status["query_width_counts"]))}

    def run(route, lanes, context, gen, arm, alternate=None):
        prompts = [ids[a.offset + i * context: a.offset + (i + 1) * context] for i in range(lanes)]
        assert all(len(p) == context for p in prompts), "prompt file too short"
        kwargs = dict(completion_batch_size=lanes, prefill_batch_size=1, prefill_step_size=a.prefill_step)
        insert = {"max_tokens": [gen] * lanes, "lane_rngs": [LaneRNG(1 + i) for i in range(lanes)]}
        if route == "mtp":
            kwargs["self_mtp"] = adapter.execution_config(max_lanes=lanes, prefill_step=a.prefill_step)
            kwargs["mtp_ordinary_handoff"] = MTPOrdinaryHandoffPolicy(
                enabled=True, max_mtp_width=a.handoff_width)
            insert["self_mtp_configs"] = [{"sampling_temp": 0.0}] * lanes
        configure(arm if alternate is None else "base")
        Q.qsa_batch_decode_sparse_status(reset=True)
        before = indexed_widths()
        g = G.BatchGenerator(model, **kwargs)
        uids = g.insert(prompts, **insert)
        tokens, done, started = {}, set(), set()
        t_first = t_end = None
        emitted = steps = 0
        windows = []  # (arm, tokens, seconds)
        polls_in_window = 0
        window_arm = "base"
        window_tokens = 0
        window_start = None
        try:
            idle = 0
            while len(done) < lanes:
                responses, idle = _drain_poll(g, idle)
                now = time.perf_counter()
                if t_first is not None and responses:
                    steps += 1
                started.update(r.uid for r in responses)
                first_now = False
                if responses and t_first is None and started >= set(uids):
                    t_first = now
                    emitted -= len(responses)
                    first_now = True
                for r in responses:
                    tokens.setdefault(r.uid, []).append(int(r.token))
                    if r.finish_reason:
                        done.add(r.uid)
                if t_first is not None:
                    emitted += len(responses)
                    t_end = now
                if alternate is not None and t_first is not None and responses and not first_now:
                    polls_in_window += 1
                    if polls_in_window == a.settle:
                        window_start, window_tokens = now, 0
                    elif polls_in_window > a.settle:
                        window_tokens += len(responses)
                    if polls_in_window == a.window:
                        if window_start is not None and len(done) == 0:
                            windows.append((window_arm, window_tokens, now - window_start))
                        window_arm = alternate if window_arm == "base" else "base"
                        configure(window_arm)
                        polls_in_window, window_start = 0, None
        finally:
            g.close()
            configure("base")
        mx.clear_cache()
        lane_tokens = [tokens.get(u, []) for u in uids]
        sha = hashlib.sha256(json.dumps(lane_tokens).encode()).hexdigest()[:16]
        wall = t_end - t_first
        after = indexed_widths()
        sparse = Q.qsa_batch_decode_sparse_status()
        rec = {"arm": arm if alternate is None else f"alt:{alternate}", "tps": emitted / wall,
               "ms_per_step": 1e3 * wall / max(1, steps), "tokens_per_step": emitted / max(1, steps),
               "steps": steps, "sha": sha, "batch_decode_sparse": sparse,
               "indexed_counts_delta": {k: v - before["counts"].get(k, 0) for k, v in after["counts"].items()
                                        if v - before["counts"].get(k, 0)},
               "handoff": {k: v for k, v in (getattr(g, "scheduler_stats", {}) or {}).items()
                           if "handoff" in str(k)}}
        if alternate is not None:
            rec["windows"] = windows
        engaged_arm = arm if alternate is None else alternate
        if engaged_arm != "base" and sparse["engagements"] == 0:
            rec["INVALID"] = "mechanism counter stayed 0"
        return rec, lane_tokens

    results = {}
    if a.admission_table:
        table = {}
        for route in ("ordinary", "mtp"):
            for lanes in (4, 8):
                for context in (16384, 24576, 32768, 49152, 65536):
                    adm = admission(route, lanes, context)
                    table[f"{route}:{lanes}:{context}"] = adm
                    print("ADMISSION", route, lanes, context, adm["stage"], adm["seated"],
                          adm["estimated_gib"], adm["usable_gib"], adm["free_gib"], flush=True)
        results["admission_table"] = table
    for config in a.configs:
        route, lanes, context = config.split(":")
        lanes, context = int(lanes), int(context)
        adm = admission(route, lanes, context)
        print(config, "admission", adm, flush=True)
        if not adm["seated"]:
            results[config] = {"admission": adm, "refused": True}
            json.dump({"partial": True, "results": results}, open(a.out, "w"), indent=1)
            continue
        for arm in ["base", *a.arms]:  # warm-up: build each arm's kernels
            run(route, lanes, 4096, 32, arm)
        runs = {}
        lane_tokens = {}
        if not a.no_full_runs:
            for arm in ["base", *a.arms]:
                rec, toks = run(route, lanes, context, a.gen, arm)
                runs[arm], lane_tokens[arm] = rec, toks
                print(f"{config} {arm} {rec['tps']:.2f} tok/s {rec['ms_per_step']:.2f} ms/step "
                      f"sha={rec['sha']} sparse={rec['batch_decode_sparse']['counts']} "
                      f"indexed={rec['indexed_counts_delta']} handoff={rec['handoff']}"
                      + (f" INVALID={rec['INVALID']}" if "INVALID" in rec else ""), flush=True)
        identity = {}
        for arm in a.arms:
            if arm not in lane_tokens:
                continue
            ref, got = lane_tokens["base"], lane_tokens[arm]
            first = None
            lanes_equal = 0
            for lane_ref, lane in zip(ref, got):
                k = next((i for i, (x, y) in enumerate(zip(lane_ref, lane)) if x != y), None)
                if k is None and len(lane_ref) == len(lane):
                    lanes_equal += 1
                elif k is not None:
                    first = k if first is None else min(first, k)
            identity[arm] = {"identical": lanes_equal == len(ref), "lanes_identical": lanes_equal,
                             "lanes": len(ref), "first_divergence": first}
        alt = None
        if len(a.arms) == 1 and a.alt_gen > 0:
            alt, _ = run(route, lanes, context, a.alt_gen, "base", alternate=a.arms[0])
            per = {"base": [], a.arms[0]: []}
            for arm, n, sec in alt["windows"]:
                per[arm].append(n / sec)
            pairs = [100 * (x / y - 1) for x, y in zip(per[a.arms[0]], per["base"])]
            alt["paired_window_delta_pct"] = {
                "n": len(pairs), "median": statistics.median(pairs) if pairs else None,
                "min": min(pairs) if pairs else None, "max": max(pairs) if pairs else None,
                "faster": sum(p > 0 for p in pairs)}
            alt["window_tps_median"] = {k: statistics.median(v) if v else None for k, v in per.items()}
            print(f"{config} ALT {alt['paired_window_delta_pct']} medians={alt['window_tps_median']} "
                  f"sparse={alt['batch_decode_sparse']['counts']}"
                  + (f" INVALID={alt['INVALID']}" if "INVALID" in alt else ""), flush=True)
        results[config] = {"admission": adm, "runs": runs, "identity": identity, "alternating": alt}
        print(config, "identity", identity, flush=True)
        json.dump({"partial": True, "results": results}, open(a.out, "w"), indent=1)
    json.dump({"model": a.model, "gen": a.gen, "alt_gen": a.alt_gen, "window": a.window,
               "settle": a.settle, "arms": a.arms, "min_context": a.min_context,
               "policy": adapter.policy.as_dict(), "results": results,
               "peak_gib": mx.get_peak_memory() / 2**30, "mlx": mx.__version__},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
