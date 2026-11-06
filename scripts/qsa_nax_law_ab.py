#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Model test flow for the NAX QSA prefill kernel laws on Flash-Next.

One process, one model load (the served adapter policy).  Arms switch the
kernel law in process through ``qwen4_qsa_nax._LAW`` (what
``nax_qsa_attention`` and the route receipts read per call):

  off      MLX_QWEN4_QSA_NAX_KERNEL off: masked SDPA (the reference path)
  v1       NAX law v1 (today's default)
  v2       NAX law v2, bf16 P
  v2_hilo  NAX law v2, fp16 hi+lo P

Per context (default 16K and 32K), one real-document prompt with a needle
sentence at ~40% depth and a recall question at the end:

* route receipts: ``qsa_nax_status`` counts per arm, engagements, law;
* TTFT through the served generator (native MTP, ``max_tokens`` 1),
  interleaved arms, one discarded warm-up round;
* numerics: a direct trunk prefill in the served chunking, then greedy
  decode; per step the top-``WINDOW`` log-probabilities (fp32).  Functional
  oracle: the continuation contains the needle code (Unicode dashes
  normalized).  For every arm pair (v2/v2_hilo against v1 and off) the first
  divergent token is classified with the AGENTS.md ``near_tie_equivalent``
  contract (standard, exact-tie and multi-way clauses) from the two arms'
  windows at that position only.

Usage: ``qsa_nax_law_ab.py owned OUT_DIR`` (CPG lease + paired GPU locks) or
``qsa_nax_law_ab.py unit OUT_DIR`` (already owns the GPU).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CHECKPOINT = (
    "Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP@sha256:"
    "25c2da42836b90dd13fc63e1b0d49aa765512de8ff07a22384595bf5b3fd5248"
)
ARMS = ("off", "v1", "v2", "v2_hilo")
WINDOW = 8
CEILING = 0.5
NEEDLE_CODE = "KV-7342-QMR"
NEEDLE = (f"\n\nArchive note: the access code for the cold-storage archive is "
          f"{NEEDLE_CODE}. Keep it with the release checklist.\n\n")
QUESTIONS = {
    # Short: the code alone (a few tokens).
    "recall": ("\n\nWhat is the access code for the cold-storage archive mentioned in the "
               "document? Reply with the code only."),
    # Long: the code, then a summary, so the continuation is long enough for
    # divergence to show (Claude review 2026-11-05, finding 3).
    "recall_summary": ("\n\nFirst write the access code for the cold-storage archive "
                       "mentioned in the document. Then summarize the document in three "
                       "bullet points."),
}
DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")


def source():
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout
    diff = git("diff", "HEAD", "--", "src", "scripts", "tests")
    files = ["src/mlx2/runtime/models/qwen4_qsa_nax.py", "src/mlx2/runtime/models/qwen4_exp.py",
             "scripts/qsa_nax_law_ab.py"]
    return {
        "commit": git("rev-parse", "HEAD").strip(),
        "src_dirty": bool(diff),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
        "untracked": [p for p in git("ls-files", "--others", "--exclude-standard",
                                     "src", "scripts", "tests").split() if p],
        "files": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files},
    }


def set_arm(QE, NAX, arm):
    if arm == "off":
        QE._QSA_NAX_KERNEL = False
        NAX._LAW = "v1"
        return
    QE._QSA_NAX_KERNEL = None
    NAX._LAW = NAX.resolve_law(arm)


def needle_prompt(h, ids, tokens, offset, question="recall"):
    tok = h.adapter.tokenizer
    body = ids[offset: offset + tokens]
    cut = int(len(body) * 0.4)
    text = tok.decode(body[:cut]) + NEEDLE + tok.decode(body[cut:])
    request = {"messages": [{"role": "user",
                             "content": "Here is a project document:\n\n" + text + QUESTIONS[question]}]}
    return list(h.adapter.prompt_tokens(request))


def oracle(text):
    return NEEDLE_CODE in text.translate(DASHES)


def timed_first_token(h, prompt):
    from mlx2.runtime.sample_utils import LaneRNG

    gen, _ = h.generator(lanes=1, mtp=True)
    insert = {"max_tokens": [1], "lane_rngs": [LaneRNG(1)],
              "self_mtp_configs": [{"sampling_temp": 0.0}]}
    t0 = time.perf_counter()
    gen.insert([list(prompt)], **insert)
    first, done = None, False
    try:
        while not done:
            _p, responses = gen.next()
            for r in responses:
                if first is None:
                    first = time.perf_counter() - t0
                done = done or bool(r.finish_reason)
    finally:
        gen.close()
    h.mx.clear_cache()
    return first


def greedy(h, prompt, *, step, decode):
    """Direct trunk prefill in ``step`` chunks, greedy decode, top-k windows."""
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
    first_lp = logits - mx.logsumexp(logits)
    mx.eval(first_lp)
    tokens, windows = [], []
    eos = set(getattr(h.adapter.tokenizer, "eos_token_ids", None) or [])
    for _ in range(decode):
        lp = logits - mx.logsumexp(logits)
        top = mx.argpartition(-lp, WINDOW - 1)[:WINDOW]
        vals = lp[top]
        mx.eval(top, vals)
        window = sorted(zip(top.tolist(), vals.tolist()), key=lambda x: -x[1])
        tok = int(window[0][0])
        tokens.append(tok)
        windows.append([[int(t), float(v)] for t, v in window])
        if tok in eos:
            break
        logits = model(mx.array([[tok]]), cache=cache)[0, -1].astype(mx.float32)
    del cache
    mx.clear_cache()
    return {"prefill_s": prefill_s, "tokens": tokens, "windows": windows,
            "text": h.adapter.tokenizer.decode(tokens), "_first_lp": first_lp}


def first_step_distance(mx, a, b):
    """Shared-prefix (prompt-only) next-token distributions of two arms."""
    la, lb = a["_first_lp"], b["_first_lp"]
    pa = mx.exp(la)
    kl = mx.sum(pa * (la - lb))
    mad = mx.max(mx.abs(la - lb) * (mx.maximum(la, lb) > -20))
    mx.eval(kl, mad)
    return {"kl_nats": float(kl.item()), "max_abs_dlogprob_above_m20": float(mad.item())}


def classify(a, b):
    """First divergence of arms ``a`` and ``b`` under the near-tie contract."""
    ta, tb = a["tokens"], b["tokens"]
    i = next((n for n, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
    out = {"first_divergence": i, "oracle_a": a["oracle"], "oracle_b": b["oracle"]}
    if i is None:
        out["verdict"] = "exact" if len(ta) == len(tb) else "exact_prefix_length_differs"
        return out
    wa, wb = dict(map(tuple, a["windows"][i])), dict(map(tuple, b["windows"][i]))
    sa, sb = ta[i], tb[i]
    top_a, top_b = max(wa.values()), max(wb.values())
    floor_a, floor_b = min(wa.values()), min(wb.values())
    out.update(selected=[sa, sb], window_a=a["windows"][i], window_b=b["windows"][i])
    if sb not in wa or sa not in wb:
        # Missing from a window: fails only if the window proves it too far.
        if (sb not in wa and top_a - floor_a > CEILING) or (sa not in wb and top_b - floor_b > CEILING):
            out["verdict"] = "high_margin_divergence"
        else:
            out["verdict"] = "unclassified"
        return out
    margin_a = top_a - wa[sb]          # A's selected (its top) vs B's selection
    margin_b = top_b - wb[sa]
    out.update(margin_a=margin_a, margin_b=margin_b)
    ranked = lambda w: sorted(w.items(), key=lambda x: -x[1])  # noqa: E731
    top2 = lambda w: {t for t, _v in ranked(w)[:2]}  # noqa: E731
    own = lambda w: ranked(w)[0][1] - ranked(w)[1][1]  # noqa: E731  selected vs its runner-up

    def exact_tie(w_tied, w_other, selected_other):
        # One arm's top two are exactly tied; the other arm selected one of
        # them, with its own selected-vs-alternate margin within the ceiling.
        return own(w_tied) == 0.0 and selected_other in top2(w_tied) and own(w_other) <= CEILING

    clause = None
    if margin_a <= CEILING and margin_b <= CEILING and top2(wa) == top2(wb):
        clause = "standard"
    elif exact_tie(wa, wb, sb) or exact_tie(wb, wa, sa):
        clause = "exact_tie"
    elif margin_a <= CEILING and margin_b <= CEILING:
        clause = "multi_way"
    if clause is None:
        out["verdict"] = "high_margin_divergence"
    elif a["oracle"] and b["oracle"]:
        out.update(verdict="near_tie_equivalent", clause=clause)
    else:
        out.update(verdict="near_tie_oracle_failed", clause=clause)
    return out


def swapouts():
    from flash_next_options_sweep import swapouts as sw

    return sw()


def atomic(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)


def unit(out_dir: Path, contexts, reps, decode, max_swapout_pages, questions=("recall",),
         ttft=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    def log(msg):
        print(msg, flush=True)
        with log_path.open("a") as fh:
            fh.write(msg + "\n")

    from flash_next_options_sweep import Harness

    h = Harness("default")
    mx = h.mx
    from mlx2.runtime.models import qwen4_exp as QE
    from mlx2.runtime.models import qwen4_qsa_nax as NAX

    swap0 = swapouts()
    corpus = "\n\n".join(p.read_text() for p in sorted((ROOT / "docs").glob("*.md")))
    ids = list(h.adapter.tokenizer.encode(corpus))
    report = {
        "checkpoint": CHECKPOINT, "source": source(), "mlx": mx.__version__,
        "device": mx.device_info().get("device_name"), "load_s": h.load_s,
        "policy": h.adapter.policy.as_dict(), "prefill_step": h.adapter.policy.prefill_step,
        "min_physical_kv": int(QE._QSA_NAX_AUTO_MIN_PHYSICAL_KV),
        "nax_kernel_available": bool(NAX.nax_kernel_available()),
        "arms": list(ARMS), "window": WINDOW, "ceiling_nats": CEILING,
        "needle_code": NEEDLE_CODE, "questions": {q: QUESTIONS[q] for q in questions},
        "decode": decode, "ttft": ttft, "scope": "not a qualification run; sanity/correctness "
        "and an unthermal TTFT A/B (no thermal holds)", "results": [],
    }
    log(f"LOADED {h.load_s:.1f}s corpus {len(ids)} tokens")
    try:
        for T, question in [(T, q) for T in contexts for q in questions]:
            prompt = needle_prompt(h, ids, T, (T * 7) % max(1, len(ids) - T), question)
            rec = {"context": T, "question": question, "prompt_tokens": len(prompt),
                   "ttft_s": {x: [] for x in ARMS}, "receipts": {}}
            for rep in range(reps + 1 if ttft else 0):
                order = list(ARMS[rep % len(ARMS):] + ARMS[: rep % len(ARMS)])
                if rep % 2:
                    order.reverse()
                for arm in order:
                    set_arm(QE, NAX, arm)
                    QE.qsa_nax_status(reset=True)
                    ttft = timed_first_token(h, prompt)
                    status = QE.qsa_nax_status(reset=True)
                    receipt = {k: status[k] for k in ("engagements", "fallbacks", "counts", "law",
                                                      "law_id", "last_receipt")}
                    rec["receipts"].setdefault(arm, receipt)
                    if rep:
                        rec["ttft_s"][arm].append(ttft)
                    log(f"T={T} rep{rep} {arm:7s} ttft={ttft:.3f}s eng={status['engagements']} "
                        f"law={status['law']}")
            if ttft:
                rec["median_ttft_s"] = {x: statistics.median(v) for x, v in rec["ttft_s"].items()}
                base = rec["median_ttft_s"]["v1"]
                rec["ttft_vs_v1_pct"] = {x: 100 * (v / base - 1)
                                         for x, v in rec["median_ttft_s"].items()}
            runs = {}
            for arm in ARMS:
                set_arm(QE, NAX, arm)
                QE.qsa_nax_status(reset=True)
                r = greedy(h, prompt, step=h.adapter.policy.prefill_step, decode=decode)
                status = QE.qsa_nax_status(reset=True)
                r["oracle"] = oracle(r["text"])
                r["engagements"] = status["engagements"]
                r["law"] = status["law"]
                runs[arm] = r
                log(f"T={T} greedy {arm:7s} prefill={r['prefill_s']:.3f}s oracle={r['oracle']} "
                    f"eng={status['engagements']} text={r['text'][:60]!r}")
            pairs = (("v1", "off"), ("v2", "v1"), ("v2_hilo", "v1"),
                     ("v2", "off"), ("v2_hilo", "off"), ("v2_hilo", "v2"))
            rec["pairs"] = {}
            for a, b in pairs:
                verdict = classify(runs[a], runs[b])
                verdict["first_step"] = first_step_distance(mx, runs[a], runs[b])
                rec["pairs"][f"{a}~{b}"] = verdict
            for r in runs.values():
                r.pop("_first_lp", None)
            rec["greedy"] = runs
            for name, verdict in rec["pairs"].items():
                log(f"T={T} {question} pair {name:12s} {verdict['verdict']} at "
                    f"{verdict['first_divergence']} {verdict.get('clause', '')} "
                    f"kl={verdict['first_step']['kl_nats']:.2e} "
                    f"maxd={verdict['first_step']['max_abs_dlogprob_above_m20']:.3f}")
            report["results"].append(rec)
            report["swapouts_delta"] = swapouts() - swap0
            atomic(out_dir / "report.partial.json", report)
            if report["swapouts_delta"] > max_swapout_pages:
                report["aborted"] = f"swapouts at T={T}"
                break
    finally:
        set_arm(QE, NAX, "v1")
        report["swapouts_delta"] = swapouts() - swap0
        atomic(out_dir / "report.json", report)
    log(f"DONE swap {report['swapouts_delta']}")
    return 0


def owned(out_dir: Path, job_id: str, extra):
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    with DesktopCPGLease(out_dir, job_id, checkpoint=CHECKPOINT,
                         source_revision=source()["commit"]) as lease:
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                   "--session", lease.owner["cpg_session"], "--label", job_id,
                   "--desktop-cpg-lease", str(lease.path),
                   "--receipt", str(out_dir / "paired-locks.json"), "--",
                   sys.executable, "-u", str(Path(__file__)), "unit", str(out_dir), *extra]
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), MLX_ENABLE_TF32="0")
        child = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
        try:
            while child.poll() is None:
                lease.check()
                time.sleep(1)
            lease.success = child.returncode == 0
            return child.returncode
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=120)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("owned", "unit"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--contexts", type=int, nargs="+", default=[16384, 32768])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--decode", type=int, default=32)
    parser.add_argument("--max-swapout-pages", type=int, default=20000)
    parser.add_argument("--questions", nargs="+", default=["recall"], choices=sorted(QUESTIONS))
    parser.add_argument("--no-ttft", action="store_true", help="numerics only")
    parser.add_argument("--job-id", default="qsa-nax-law-v2-ab-20261105")
    args = parser.parse_args()
    if args.mode == "unit":
        return unit(args.out, args.contexts, args.reps, args.decode, args.max_swapout_pages,
                    tuple(args.questions), not args.no_ttft)
    extra = ["--contexts", *map(str, args.contexts), "--reps", str(args.reps),
             "--decode", str(args.decode), "--max-swapout-pages", str(args.max_swapout_pages),
             "--questions", *args.questions, *(["--no-ttft"] if args.no_ttft else [])]
    return owned(args.out, args.job_id, extra)


if __name__ == "__main__":
    raise SystemExit(main())
