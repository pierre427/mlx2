#!/usr/bin/env python3
"""End-to-end NAX QSA prefill A/B on the served Flash-Next artifact.

One process, one model load (served policy).  Arms switch the NAX prefill
route in process through the module selections qwen4_exp reads per call:

  off        MLX_QWEN4_QSA_NAX_KERNEL=0 (masked SDPA everywhere)
  x<N>       auto with crossover N (``qsa_nax_min_physical_kv``)
  bx<N>      auto, crossover N, batched admission on (``qsa_nax_batched``)

--mode single: one real-document prompt per context, B=1.
  TTFT through the served generator (native MTP, ``max_tokens`` 1),
  interleaved arms, one discarded warm-up per context.  Numerics: a direct
  trunk prefill in 8192-row chunks (the served chunking), last-position
  logits (fp32) max-diff against ``off``, then greedy decode and the first
  diverging token.

--mode batched: N concurrent ragged prompts through the served generator
  (prefill batches of 2), time until every lane has its first token, decode
  tokens per lane against the first arm.

Each arm records the NAX admission counts it produced (qsa_nax_status).

  gpuq.sh qsa-nax-b env PYTHONPATH=src MLX_ENABLE_TF32=0 .venv/bin/python \\
      scripts/qsa_nax_e2e_probe.py --i-own-the-gpu --mode single --out b.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from flash_next_options_sweep import MODEL, Harness, swapouts  # noqa: E402


def set_arm(QE, arm):
    QE._QSA_NAX_AUTO_BATCHED = False
    QE._QSA_NAX_AUTO_MIN_PHYSICAL_KV = 16384
    if arm == "off":
        QE._QSA_NAX_KERNEL = False
        return
    QE._QSA_NAX_KERNEL = None
    if arm.startswith("bx"):
        QE._QSA_NAX_AUTO_BATCHED = True
        QE._QSA_NAX_AUTO_MIN_PHYSICAL_KV = int(arm[2:])
    elif arm.startswith("x"):
        QE._QSA_NAX_AUTO_MIN_PHYSICAL_KV = int(arm[1:])
    else:
        raise ValueError(arm)


def doc_prompt(h, ids, tokens, offset):
    body = h.adapter.tokenizer.decode(ids[offset: offset + tokens])
    request = {"messages": [{"role": "user", "content": "Here is a project document:\n\n" + body
                             + "\n\nIn five bullet points, what are the main components it describes?"}]}
    return list(h.adapter.prompt_tokens(request))


def timed_run(h, prompts, *, max_tokens, mtp):
    """Harness.run, plus the time until every lane has its first token."""
    from mlx2.runtime.sample_utils import LaneRNG

    mx = h.mx
    gen, _ = h.generator(lanes=len(prompts), mtp=mtp)
    insert = {"max_tokens": [max_tokens] * len(prompts),
              "lane_rngs": [LaneRNG(1 + i) for i in range(len(prompts))]}
    if mtp:
        insert["self_mtp_configs"] = [{"sampling_temp": 0.0}] * len(prompts)
    t0 = time.perf_counter()
    uids = gen.insert([list(p) for p in prompts], **insert)
    tokens, done, firsts = {}, set(), {}
    try:
        while len(done) < len(prompts):
            _p, responses = gen.next()
            now = time.perf_counter()
            for r in responses:
                firsts.setdefault(r.uid, now - t0)
                tokens.setdefault(r.uid, []).append(int(r.token))
                if r.finish_reason:
                    done.add(r.uid)
    finally:
        gen.close()
    wall = time.perf_counter() - t0
    mx.clear_cache()
    return {"lanes": [tokens.get(u, []) for u in uids],
            "first_token_s": [firsts.get(u) for u in uids],
            "all_first_s": max(firsts.values()), "wall_s": wall}


def numerics(h, QE, prompt, *, step, decode):
    """Direct trunk prefill in ``step`` chunks: last logits, greedy tokens, ms."""
    mx = h.mx
    model = h.adapter.model
    cache = model.make_cache()
    mx.synchronize()
    t0 = time.perf_counter()
    for start in range(0, len(prompt), step):
        hidden = model.model(mx.array([prompt[start: start + step]]), cache)
        mx.eval(hidden)
    mx.synchronize()
    prefill_s = time.perf_counter() - t0
    logits = model.language_model.logits(hidden[:, -1:]).astype(mx.float32)[0, -1]
    mx.eval(logits)
    first = logits
    out = []
    for _ in range(decode):
        tok = int(mx.argmax(logits).item())
        out.append(tok)
        logits = model(mx.array([[tok]]), cache=cache)[0, -1].astype(mx.float32)
        mx.eval(logits)
    del cache
    mx.clear_cache()
    return first, out, prefill_s


def nax_counts(QE):
    return dict(QE.qsa_nax_status(reset=True)["counts"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--mode", choices=("single", "batched"), required=True)
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--contexts", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--cohorts", nargs="+", default=["16384,24576", "12288,16384,20480,24576"],
                    help="batched mode: comma-separated prompt token counts per cohort")
    ap.add_argument("--max-swapout-pages", type=int, default=20000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("Metal run: pass --i-own-the-gpu under the GPU lock")
    arms = a.arms or (["off", "x4096", "x8192", "x16384", "x32768"] if a.mode == "single"
                      else ["x16384", "bx16384", "bx8192"])

    h = Harness("default", model=a.model)
    mx = h.mx
    from mlx2.runtime.models import qwen4_exp as QE
    from mlx2.runtime.models import qwen4_qsa_nax as NAX

    swap0 = swapouts()
    corpus = "\n\n".join(p.read_text() for p in sorted((ROOT / "docs").glob("*.md")))
    ids = list(h.adapter.tokenizer.encode(corpus))
    out = {"model": a.model, "mlx": mx.__version__, "mode": a.mode, "arms": arms,
           "device": mx.device_info().get("device_name"), "load_s": h.load_s,
           "nax_kernel_available": bool(NAX.nax_kernel_available()),
           "prefill_step": h.adapter.policy.prefill_step,
           "policy": h.adapter.policy.as_dict(), "results": []}
    print("LOADED", f"{h.load_s:.1f}s", "corpus", len(ids), flush=True)

    def save():
        set_arm(QE, "x16384")
        QE._QSA_NAX_KERNEL = None
        out["swapouts_delta"] = swapouts() - swap0
        Path(a.out).write_text(json.dumps(out, indent=1))

    if a.mode == "single":
        for T in a.contexts:
            prompt = doc_prompt(h, ids, T, (T * 7) % max(1, len(ids) - T))
            P = len(prompt)
            ctx_arms = [x for x in arms if x in ("off", "x16384") or not x.startswith("x")
                        or int(x[1:]) <= P]
            rec = {"context": T, "prompt_tokens": P, "arms": ctx_arms,
                   "ttft_s": {x: [] for x in ctx_arms}, "counts": {}}
            for rep in range(a.reps + 1):
                k = len(ctx_arms)
                order = ctx_arms[rep % k:] + ctx_arms[: rep % k]
                if rep % 2:
                    order = order[::-1]
                for arm in order:
                    set_arm(QE, arm)
                    nax_counts(QE)
                    r = timed_run(h, [prompt], max_tokens=1, mtp=True)
                    counts = nax_counts(QE)
                    if rep:
                        rec["ttft_s"][arm].append(r["all_first_s"])
                    rec["counts"].setdefault(arm, counts)
                    print(f"T={T} rep{rep} {arm:7s} ttft={r['all_first_s']:.3f}s {counts}", flush=True)
            base = statistics.median(rec["ttft_s"]["x16384"])
            rec["median_ttft_s"] = {x: statistics.median(v) for x, v in rec["ttft_s"].items()}
            rec["ttft_vs_x16384_pct"] = {x: 100 * (v / base - 1) for x, v in rec["median_ttft_s"].items()}
            # Numerics against the masked arm.
            ref_logits = ref_tokens = None
            rec["numerics"] = {}
            for arm in ctx_arms:
                set_arm(QE, arm)
                nax_counts(QE)
                logits, toks, prefill_s = numerics(h, QE, prompt, step=h.adapter.policy.prefill_step,
                                                   decode=a.decode)
                entry = {"prefill_s": prefill_s, "counts": nax_counts(QE), "tokens": toks}
                if arm == "off":
                    ref_logits, ref_tokens = logits, toks
                else:
                    entry["logit_max_abs_diff"] = float(mx.max(mx.abs(logits - ref_logits)).item())
                    entry["argmax_same"] = int(mx.argmax(logits).item()) == int(mx.argmax(ref_logits).item())
                    entry["first_divergence"] = next(
                        (i for i, (x, y) in enumerate(zip(toks, ref_tokens)) if x != y), None)
                rec["numerics"][arm] = entry
                print(f"T={T} numerics {arm:7s} prefill={prefill_s:.3f}s "
                      f"maxdiff={entry.get('logit_max_abs_diff')} "
                      f"first_div={entry.get('first_divergence')}", flush=True)
            out["results"].append(rec)
            save()
            if swapouts() - swap0 > a.max_swapout_pages:
                out["aborted"] = f"swapouts at T={T}"
                break
    else:
        for cohort in a.cohorts:
            sizes = [int(s) for s in cohort.split(",")]
            prompts = [doc_prompt(h, ids, n, (i * 9973 + n) % max(1, len(ids) - n))
                       for i, n in enumerate(sizes)]
            rec = {"cohort": sizes, "prompt_tokens": [len(p) for p in prompts],
                   "all_first_s": {x: [] for x in arms}, "counts": {}, "tokens": {}}
            for rep in range(a.reps + 1):
                k = len(arms)
                order = arms[rep % k:] + arms[: rep % k]
                if rep % 2:
                    order = order[::-1]
                for arm in order:
                    set_arm(QE, arm)
                    nax_counts(QE)
                    r = timed_run(h, prompts, max_tokens=a.decode, mtp=True)
                    counts = nax_counts(QE)
                    if rep:
                        rec["all_first_s"][arm].append(r["all_first_s"])
                        rec["tokens"].setdefault(arm, r["lanes"])
                    rec["counts"].setdefault(arm, counts)
                    print(f"cohort={sizes} rep{rep} {arm:8s} all_first={r['all_first_s']:.3f}s {counts}",
                          flush=True)
            base_arm = arms[0]
            base = statistics.median(rec["all_first_s"][base_arm])
            rec["median_all_first_s"] = {x: statistics.median(v) for x, v in rec["all_first_s"].items()}
            rec["vs_first_arm_pct"] = {x: 100 * (v / base - 1) for x, v in rec["median_all_first_s"].items()}
            ref = rec["tokens"][base_arm]
            rec["first_divergence_per_lane"] = {
                x: [next((i for i, (p, q) in enumerate(zip(lane, rlane)) if p != q), None)
                    for lane, rlane in zip(rec["tokens"][x], ref)]
                for x in arms}
            out["results"].append(rec)
            save()
            if swapouts() - swap0 > a.max_swapout_pages:
                out["aborted"] = f"swapouts at cohort {sizes}"
                break
    save()
    print("DONE", a.out, "swap", out["swapouts_delta"], flush=True)


if __name__ == "__main__":
    main()
