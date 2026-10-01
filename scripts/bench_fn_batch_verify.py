"""In-process A/B of the batched fused GDN verify on Flash-Next (one model load).

Arms are the GatedDeltaNet batched-verify mode (``off`` = the served stock
multi-row chain for B>1 verify blocks, ``row_exact`` = the batched fused
verify).  For each ``mtp:B`` config the served batched self-MTP route runs
``B`` lanes (``adapter.execution_config`` as the server builds it, the
route's copy-draft policy, greedy), every lane its own real chat prompt, and
decodes ``--gen`` tokens per lane.  Arms alternate AB/BA per rep after a
discarded warm-up of BOTH arms (kernel compilation stays out of the reps).
Decode tok/s is measured once every lane is decoding.  Every run records
per-lane tokens, the batched-verify engagement counters and the GDN verify
fallback reasons.

``--row-exact-phase B``: afterwards install the opt-in row-exact verify route
and run ``B`` lanes once per arm, reporting the route's window counters and
failure reasons, and every lane's tokens vs that lane's MTP-off B=1 decode.

  MLX_ENABLE_TF32=0 PYTHONPATH=src .venv/bin/python scripts/bench_fn_batch_verify.py \\
      --i-own-the-gpu --model ~/mlx-models/Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP \\
      --configs mtp:1 mtp:2 mtp:4 mtp:8 mtp:16 --out ab.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_DOC = (ROOT / "docs" / "QUALIFICATION.md")

TASKS = [
    "Explain how a database transaction works, in numbered sections.",
    "Write a Python function that parses an ISO 8601 duration string such as "
    "P3DT4H12M into total seconds, with a docstring and three doctests.",
    "A train leaves at 09:40 and travels 212 km at 84 km/h, then waits 11 minutes "
    "and travels 95 km at 76 km/h. When does it arrive? Show the arithmetic step by step.",
    "Write a short story (about 300 words) about a lighthouse keeper who finds a "
    "message in a bottle written in her own handwriting.",
    "Compare TCP and UDP for a multiplayer game server. Give a table and a recommendation.",
    "Implement a thread-safe LRU cache in Rust with get and put, and explain the "
    "ownership choices.",
    "What were the main causes of the French Revolution? Answer in five bullet points "
    "with one sentence each.",
    "Translate into French and then explain three grammatical choices you made: "
    "'The committee postponed its decision until the auditors had finished their review.'",
    "Derive the closed form of the sum 1^2 + 2^2 + ... + n^2 by induction.",
    "Write a SQL query that finds, for every customer, their three most recent orders, "
    "and explain how the window function works.",
    "Give a beginner-friendly explanation of how a transformer language model "
    "generates text, without equations.",
    "Draft a polite email declining a meeting invitation and proposing two alternative times.",
    "Write a bash script that finds the ten largest files under a directory, "
    "handling spaces in names, and explain each line.",
    "List the planets of the solar system with one distinctive fact each, then "
    "rank them by average density.",
    "Explain the difference between a mutex and a semaphore with a concrete example in C.",
    "Plan a three-day walking itinerary for Lisbon for someone who likes architecture "
    "and food, with times.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--configs", nargs="*", default=["mtp:1", "mtp:2", "mtp:4", "mtp:8", "mtp:16"])
    ap.add_argument("--arms", nargs="+", default=["off", "row_exact"])
    ap.add_argument("--gen", type=int, default=192)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--prompt-offset", type=int, default=0,
                    help="lane i gets task (offset + i) mod len(TASKS)")
    ap.add_argument("--with-document", action="store_true",
                    help="prefix every other lane's task with a real repository document")
    ap.add_argument("--row-exact-phase", type=int, default=0)
    ap.add_argument("--policy", default=None,
                    help="JSON execution policy (e.g. '{\"row_exact_verify\": true}')")
    ap.add_argument("--max-swapout-pages", type=int, default=20000)
    ap.add_argument("--cache-limit-gib", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")

    # The adapter pins the serving environment; construct it before any
    # mlx2.runtime model module is imported (import_env guard).
    from mlx2.adapters.registry import resolve_adapter

    policy = json.loads(a.policy) if a.policy else None
    adapter = resolve_adapter(a.model, mtp=True)(a.model, execution_policy=policy)
    import mlx.core as mx

    from mlx2.runtime import generate as G
    from mlx2.runtime.models import qwen4_exp
    from mlx2.runtime.sample_utils import LaneRNG

    model = adapter.model
    mx.eval(model.parameters())
    mx.set_cache_limit(a.cache_limit_gib << 30)

    def swapouts():
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        line = next(l for l in out.splitlines() if l.startswith("Swapouts"))
        return int(line.split(":")[1].strip().rstrip("."))

    swap0 = swapouts()
    gdn = [m for _, m in model.named_modules() if isinstance(m, qwen4_exp.GatedDeltaNet)]
    print("LOADED", len(gdn), "GDN layers", f"active={mx.get_active_memory() / 2**30:.1f}GiB",
          flush=True)
    copy_policy = adapter.default_route_execution_policy["native_mtp"]["self_mtp_copy_draft"]
    document = _DOC.read_text()[:6000] if a.with_document else None

    def prompt(i):
        task = TASKS[(a.prompt_offset + i) % len(TASKS)]
        if document is not None and i % 2:
            task = f"Read this document:\n\n{document}\n\nThen: {task}"
        return list(adapter.prompt_tokens({"messages": [{"role": "user", "content": task}]}))

    def configure(arm):
        for layer in gdn:
            layer.set_fused_gdn_batch_verify_mode(arm)

    def counters():
        stats = qwen4_exp.qwen4_fused_gdn_stats(model, modules=gdn)
        batch = stats.get("batch_verify", {})
        return {
            "verify_calls": stats["verify_calls"],
            "batch_calls": batch.get("calls", 0),
            "batch_ragged_calls": batch.get("ragged_calls", 0),
            "batch_rollback_calls": batch.get("rollback_calls", 0),
            "batch_fallback_reasons": batch.get("fallback_reasons", {}),
            "verify_fallback_reasons": stats["verify_fallback_reasons"],
        }

    def run(batch, mtp=True):
        prompts = [prompt(i) for i in range(batch)]
        kwargs = {}
        if mtp:
            kwargs["self_mtp"] = adapter.execution_config(max_lanes=batch,
                                                          prefill_step=adapter.prefill_step_default())
            kwargs["copy_draft"] = copy_policy
        stats = {}
        gen = G.BatchGenerator(model, completion_batch_size=batch, prefill_batch_size=1,
                               prefill_step_size=adapter.prefill_step_default(),
                               scheduler_stats=stats, **kwargs)
        insert = {"max_tokens": [a.gen] * batch, "lane_rngs": [LaneRNG(1 + i) for i in range(batch)]}
        if mtp:
            insert["self_mtp_configs"] = [{"sampling_temp": 0.0}] * batch
        qwen4_exp.qwen4_fused_gdn_stats(model, modules=gdn, reset=True)
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
        return {
            "tps": emitted / max(1e-9, t_end - t_first),
            "ms_per_step": 1e3 * (t_end - t_first) / max(1, steps),
            "tokens_per_step": emitted / max(1, steps),
            "sha": hashlib.sha256(json.dumps(lanes).encode()).hexdigest()[:16],
            "lanes": lanes,
            "counters": counters(),
            "prompt_tokens": [len(p) for p in prompts],
        }

    def check_swap():
        grew = swapouts() - swap0
        if grew > a.max_swapout_pages:
            raise SystemExit(f"aborting: Swapouts grew by {grew} pages since load")

    def first_div(x, y):
        return next((i for i, (p, q) in enumerate(zip(x, y)) if p != q),
                    None if len(x) == len(y) else min(len(x), len(y)))

    results = {}
    for config in a.configs:
        route, batch = config.split(":")
        batch = int(batch)
        for arm in a.arms:  # warm-up, both arms, discarded
            configure(arm)
            run(batch)
        per_arm = {arm: [] for arm in a.arms}
        for rep in range(a.reps):
            order = a.arms if rep % 2 == 0 else a.arms[::-1]
            for arm in order:
                configure(arm)
                result = run(batch)
                check_swap()
                per_arm[arm].append(result)
                c = result["counters"]
                print(f"{config} rep{rep} {arm} {result['tps']:.2f} tok/s "
                      f"{result['ms_per_step']:.2f} ms/step {result['tokens_per_step']:.2f} tok/step "
                      f"batch_calls={c['batch_calls']} ragged={c['batch_ragged_calls']} "
                      f"b1_verify={c['verify_calls']} sha={result['sha']}", flush=True)
        summary = {}
        off = per_arm.get("off")
        for arm, runs in per_arm.items():
            tps = [r["tps"] for r in runs]
            entry = {
                "median_tps": statistics.median(tps),
                "min_tps": min(tps), "max_tps": max(tps),
                "median_ms_per_step": statistics.median(r["ms_per_step"] for r in runs),
                "mean_tokens_per_step": statistics.mean(r["tokens_per_step"] for r in runs),
                "deterministic_across_reps": len({r["sha"] for r in runs}) == 1,
                "batch_calls_per_run": [r["counters"]["batch_calls"] for r in runs],
                "batch_ragged_calls_per_run": [r["counters"]["batch_ragged_calls"] for r in runs],
                "batch_fallback_reasons": runs[0]["counters"]["batch_fallback_reasons"],
                "verify_fallback_reasons": runs[0]["counters"]["verify_fallback_reasons"],
            }
            if off is not None and arm != "off":
                entry["paired_tps_delta_pct"] = [
                    100 * (r["tps"] / o["tps"] - 1) for r, o in zip(runs, off)]
                entry["median_delta_vs_off_pct"] = 100 * (
                    entry["median_tps"] / statistics.median(o["tps"] for o in off) - 1)
                entry["lanes_identical_to_off"] = [
                    x == y for x, y in zip(runs[0]["lanes"], off[0]["lanes"])]
                entry["first_divergence_vs_off"] = [
                    first_div(x, y) for x, y in zip(off[0]["lanes"], runs[0]["lanes"])]
            summary[arm] = entry
        results[config] = {"summary": summary,
                           "runs": {arm: [{k: v for k, v in r.items() if k != "lanes"}
                                          for r in runs] for arm, runs in per_arm.items()},
                           "lanes": {arm: runs[0]["lanes"] for arm, runs in per_arm.items()}}
        print(config, json.dumps(summary, indent=1), flush=True)

    row_exact = None
    if a.row_exact_phase:
        from mlx2.runtime.models import qwen4_row_exact

        batch = a.row_exact_phase
        configure("off")
        solo = []
        for i in range(batch):  # each lane's MTP-off B=1 decode: the row-exact oracle
            gen = G.BatchGenerator(model, completion_batch_size=1, prefill_batch_size=1,
                                   prefill_step_size=adapter.prefill_step_default())
            gen.insert([prompt(i)], max_tokens=[a.gen], lane_rngs=[LaneRNG(1 + i)])
            out, done = [], False
            try:
                while not done:
                    for r in gen.next()[1]:
                        out.append(int(r.token))
                        done = done or bool(r.finish_reason)
            finally:
                gen.close()
            solo.append(out)
            mx.clear_cache()
        handle = qwen4_row_exact.install(model)
        handle.enable(True)
        row_exact = {"lanes": batch, "arms": {}}
        try:
            for arm in a.arms:
                configure(arm)
                before = handle.status()
                result = run(batch)
                after = handle.status()
                check_swap()
                windows = after["windows"] - before["windows"]
                failures = {k: v - before["failures"].get(k, 0)
                            for k, v in after["failures"].items()
                            if v - before["failures"].get(k, 0)}
                gdn_stage = {k: v - before["stages"].get("gdn", {}).get(k, 0)
                             for k, v in after["stages"].get("gdn", {}).items()}
                row_exact["arms"][arm] = {
                    "tps": result["tps"],
                    "windows": windows,
                    "windows_row_exact": after["windows_row_exact"] - before["windows_row_exact"],
                    "windows_not_exact": after["windows_not_exact"] - before["windows_not_exact"],
                    "failures": failures,
                    "gdn_stage": gdn_stage,
                    "lanes_identical_to_mtp_off_b1": [x == y for x, y in zip(result["lanes"], solo)],
                    "first_divergence_vs_mtp_off_b1": [first_div(y, x) for x, y in zip(result["lanes"], solo)],
                    "counters": result["counters"],
                }
                print("row-exact", arm, json.dumps(row_exact["arms"][arm]), flush=True)
        finally:
            handle.remove()
            configure("off")

    json.dump({"model": a.model, "policy": adapter.policy.as_dict(), "gen": a.gen,
               "reps": a.reps, "copy_draft": copy_policy, "results": results,
               "row_exact_phase": row_exact,
               "swapouts_delta_pages": swapouts() - swap0,
               "peak_gib": mx.get_peak_memory() / 2**30, "mlx": mx.__version__},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
