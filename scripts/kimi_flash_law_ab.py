#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Model test flow for the Kimi MLA flash192 kernel laws (v1, v2).

One process, one load of Kimi-Linear-48B-A3B-Instruct-8bit with the served
policy (mla auto, mla_flash all).  Arms switch ``flash_law`` in place on the
MLA layers (what ``MultiLatentAttention`` passes to ``flash192`` and counts).

Per context (default 16K, 32K, 64K), one real-document prompt with a needle
at ~40% depth, for each question (code only; code then summary):

* route receipts: the MLA counters per arm (``expanded_flash192_law_*``,
  ``flash_declined``, latent-flash rows);
* prefill time: a direct trunk prefill in the served 8192-row chunks,
  interleaved arms, one discarded warm-up round (no thermal holds);
* numerics: greedy decode with per-step top-8 log-probabilities; functional
  oracle (needle code present, Unicode dashes normalized); first-divergence
  classification under the AGENTS.md near_tie_equivalent contract (shared
  with ``qsa_nax_law_ab.classify``); first-step KL between arms.

Usage: ``kimi_flash_law_ab.py owned OUT_DIR`` or ``... unit OUT_DIR``.
"""

from __future__ import annotations

import argparse
import os
import signal
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from qsa_nax_law_ab import (  # noqa: E402
    NEEDLE, NEEDLE_CODE, QUESTIONS, WINDOW, CEILING, atomic, classify, first_step_distance, oracle,
)

MODEL = Path.home() / "Models/Kimi-Linear-48B-A3B-Instruct-8bit"
LAWS = ("v1", "v2")
STEP = 8192


def source():
    import hashlib

    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True,  # noqa: E731
                                    text=True).stdout
    diff = git("diff", "HEAD", "--", "src", "scripts", "tests")
    files = ["src/mlx2/runtime/models/kimi_linear_mla_flash.py",
             "src/mlx2/runtime/models/kimi_linear.py", "scripts/kimi_flash_law_ab.py"]
    return {"commit": git("rev-parse", "HEAD").strip(), "src_dirty": bool(diff),
            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
            "files": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files}}


def swapouts():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    line = next(l for l in out.splitlines() if l.startswith("Swapouts"))
    return int(line.split(":")[1].strip().rstrip("."))


def mla_layers(model):
    return [layer.attention for layer in model.model.layers if layer.is_global]


def set_law(model, law):
    for attn in mla_layers(model):
        attn.flash_law = law
        attn.calls.clear()


def counters(model):
    total = Counter()
    for attn in mla_layers(model):
        total.update(attn.calls)
    return dict(total)


def build_prompt(adapter, ids, tokens, offset, question):
    tok = adapter.tokenizer
    body = ids[offset: offset + tokens]
    cut = int(len(body) * 0.4)
    text = tok.decode(body[:cut]) + NEEDLE + tok.decode(body[cut:])
    request = {"messages": [{"role": "user",
                             "content": "Here is a project document:\n\n" + text + QUESTIONS[question]}]}
    return list(adapter.prompt_tokens(request))


def prefill(mx, model, prompt):
    from mlx2.runtime.models import moe_nax_gather

    cache = model.make_cache()
    mx.synchronize()
    t0 = time.perf_counter()
    for start in range(0, len(prompt), STEP):
        chunk = mx.array([prompt[start: start + STEP]])
        with moe_nax_gather.forward_scope(chunk, cache):
            hidden = model.model(chunk, cache)
        mx.eval(hidden)
    mx.synchronize()
    seconds = time.perf_counter() - t0
    logits = model.lm_head(hidden[:, -1:]).astype(mx.float32)[0, -1]
    mx.eval(logits)
    return cache, logits, seconds


def greedy(mx, adapter, prompt, decode):
    model = adapter.model
    cache, logits, seconds = prefill(mx, model, prompt)
    first_lp = logits - mx.logsumexp(logits)
    mx.eval(first_lp)
    eos = set(adapter.tokenizer.eos_token_ids)
    tokens, windows = [], []
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
        logits = model(mx.array([[tok]]), cache)[0, -1].astype(mx.float32)
    del cache
    mx.clear_cache()
    return {"prefill_s": seconds, "tokens": tokens, "windows": windows,
            "text": adapter.tokenizer.decode(tokens), "_first_lp": first_lp}


def unit(out_dir, contexts, questions, reps, decode, max_swap):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    def log(msg):
        print(msg, flush=True)
        with log_path.open("a") as fh:
            fh.write(msg + "\n")

    import mlx.core as mx
    from mlx2.adapters.kimi_linear import KimiLinearAdapter, checkpoint_name

    t0 = time.perf_counter()
    adapter = KimiLinearAdapter(str(MODEL))
    load_s = time.perf_counter() - t0
    mx.set_cache_limit(8 << 30)
    model = adapter.model
    swap0 = swapouts()
    corpus = "\n\n".join(p.read_text() for p in sorted((ROOT / "docs").glob("*.md")))
    ids = list(adapter.tokenizer.encode(corpus))
    from mlx2.runtime.models import kimi_linear_mla_flash as fl

    report = {
        "checkpoint": checkpoint_name(str(MODEL)) + ":" + adapter.identity["fingerprint"],
        "source": source(), "mlx": mx.__version__, "device": mx.device_info().get("device_name"),
        "load_s": load_s, "policy": adapter.policy, "prefill_step": STEP, "laws": list(LAWS),
        "probes": {k: fl.available(k) for k in ("expanded", "expanded_v2", "absorbed")},
        "needle_code": NEEDLE_CODE, "questions": {q: QUESTIONS[q] for q in questions},
        "window": WINDOW, "ceiling_nats": CEILING, "decode": decode,
        "scope": "not a qualification run; correctness + unthermal prefill A/B", "results": [],
    }
    log(f"LOADED {load_s:.1f}s {report['checkpoint']} probes={report['probes']}")
    try:
        for T in contexts:
            offset = (T * 7) % max(1, len(ids) - T)
            prompts = {q: build_prompt(adapter, ids, T, offset, q) for q in questions}
            rec = {"context": T, "prompt_tokens": {q: len(p) for q, p in prompts.items()},
                   "prefill_s": {law: [] for law in LAWS}, "receipts": {}}
            timing_prompt = prompts[questions[0]]
            for rep in range(reps + 1):
                for law in (LAWS if rep % 2 == 0 else LAWS[::-1]):
                    set_law(model, law)
                    cache, _logits, seconds = prefill(mx, model, timing_prompt)
                    del cache
                    mx.clear_cache()
                    rec["receipts"].setdefault(law, counters(model))
                    if rep:
                        rec["prefill_s"][law].append(seconds)
                    log(f"T={T} rep{rep} {law} prefill={seconds:.3f}s {counters(model)}")
            rec["median_prefill_s"] = {k: statistics.median(v) for k, v in rec["prefill_s"].items()}
            rec["prefill_vs_v1_pct"] = {k: 100 * (v / rec["median_prefill_s"]["v1"] - 1)
                                        for k, v in rec["median_prefill_s"].items()}
            rec["numerics"] = {}
            for q, prompt in prompts.items():
                runs = {}
                for law in LAWS:
                    set_law(model, law)
                    r = greedy(mx, adapter, prompt, decode)
                    r["oracle"] = oracle(r["text"])
                    r["counters"] = counters(model)
                    runs[law] = r
                    log(f"T={T} {q} {law} oracle={r['oracle']} n={len(r['tokens'])} "
                        f"text={r['text'][:70]!r}")
                verdict = classify(runs["v2"], runs["v1"])
                verdict["first_step"] = first_step_distance(mx, runs["v2"], runs["v1"])
                for r in runs.values():
                    r.pop("_first_lp", None)
                rec["numerics"][q] = {"runs": runs, "v2~v1": verdict}
                log(f"T={T} {q} v2~v1 {verdict['verdict']} at {verdict['first_divergence']} "
                    f"{verdict.get('clause', '')} kl={verdict['first_step']['kl_nats']:.2e}")
            report["results"].append(rec)
            report["swapouts_delta"] = swapouts() - swap0
            atomic(out_dir / "report.partial.json", report)
            if report["swapouts_delta"] > max_swap:
                report["aborted"] = f"swapouts at T={T}"
                break
    finally:
        set_law(model, "v1")
        report["swapouts_delta"] = swapouts() - swap0
        atomic(out_dir / "report.json", report)
    log(f"DONE swap {report['swapouts_delta']}")
    return 0


def owned(out_dir, job_id, extra):
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    with DesktopCPGLease(out_dir, job_id,
                         checkpoint="Kimi-Linear-48B-A3B-Instruct-8bit@553a7a5673491df2b2d38f7380188a9d8e9f9a66",
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("owned", "unit"))
    ap.add_argument("out", type=Path)
    ap.add_argument("--contexts", type=int, nargs="+", default=[16384, 32768, 65536])
    ap.add_argument("--questions", nargs="+", default=["recall", "recall_summary"],
                    choices=sorted(QUESTIONS))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--decode", type=int, default=64)
    ap.add_argument("--max-swapout-pages", type=int, default=20000)
    ap.add_argument("--job-id", default="kimi-flash-law-ab-20261105")
    a = ap.parse_args()
    if a.mode == "unit":
        return unit(a.out, a.contexts, tuple(a.questions), a.reps, a.decode, a.max_swapout_pages)
    extra = ["--contexts", *map(str, a.contexts), "--questions", *a.questions, "--reps",
             str(a.reps), "--decode", str(a.decode), "--max-swapout-pages", str(a.max_swapout_pages)]
    return owned(a.out, a.job_id, extra)


if __name__ == "__main__":
    raise SystemExit(main())
