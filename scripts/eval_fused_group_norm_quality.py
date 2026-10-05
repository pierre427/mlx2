#!/usr/bin/env python3
"""Does the fused GroupRMSNorm change answer quality? Perplexity + generation A/B.

The fused kernel is 8.06x faster than the eager path at T=16384 but is not
bit-exact: 2-4 bf16 ULP on about 2.6 elements per million of the norm output.
Per-op deltas cannot answer whether that matters. rm15 is the precedent for why:
a 1 bf16 ULP perturbation at GDN layer 0 amplified through depth to
max|delta logit| 7.55 and top-1 agreement 0.984 at 16K. So the question has to
be asked at the output, on the real model, on real long-context prompts.

Arms differ ONLY in the ``MLX_QWEN4_FUSED_GROUP_NORM`` lever, flipped live via
``set_fused_group_norm_enabled``. Everything else -- artifact, weights,
tokenizer, prompt, cache construction -- is identical, because both arms run in
one process against one loaded model.

Two phases answer different questions:

  Phase A -- ANSWER QUALITY.  Prefill (long context + a question that can only
  be answered from that context), then greedy-generate.  The text is recorded
  verbatim for both arms so quality can be read rather than inferred from a
  scalar.  Divergence is measured only over the COMMON PREFIX: once the arms
  pick different tokens their caches hold different contexts, so comparing
  distributions past that point would compare different questions.

  Phase B -- PERPLEXITY.  Prefill disjoint source-only windows, then
  teacher-force continuations held out from each window's prefill token by
  token.  The scoring windows may overlap phase A's source documents; they
  do not constitute an independent evaluation corpus.  Both arms score the
  SAME tokens, so perplexity, KL, max|delta logit| and top-1 agreement are
  directly comparable.  Scoring one token at a time bounds memory: a full
  [context, vocab] logprob matrix would be ~8 GB at vocab 248320.

Both phases prefill exactly ``--context-tokens`` rows, because admission is
locked to a candidate row-count list; a prompt one token wider declines and the
arm silently measures the eager path.  The harness raises if fused engagement
is absent or eager engagement occurs.  The generation budget is a ceiling:
completed answers require an EOS and a nonempty final answer after </think>.

Engagement is proven, not asserted: ``fused_group_norm_stats()`` is recorded per
arm, and the harness fails loudly if the fused arm shows zero kernel calls or
the eager arm shows any.

``--passes 2`` optionally runs the set twice with the arm order swapped
(AB then BA); the default is one pass because greedy outputs were identical
across the two passes in the previous run.

GREEDY ONLY.  Sampling would add variance that swamps the effect being measured
and would make divergence unattributable.

Loads the full artifact (~97 GB resident weights, ~77 GB peak Metal) and needs
the GPU exclusively for several minutes.  --dry-run prints the plan and exits.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SCHEMA = "mlx2.fused-group-norm-quality.v2"

PROMPT_SPECS = (
    ("ARCHITECTURE.md", "Summarise how APCv2 owns state and why ordinary decode remains a reference route."),
    ("SERVING.md", "What does the qualification gate require before a route is selectable?"),
    ("FLASHNEXT-PARITY.md", "Which Flash-Next mechanisms are on by default and which are off?"),
    ("RESULTS.md", "What do the historical results establish and what do they not establish?"),
    ("RESUME.md", "What is the current Qwen3.6 decision and what evidence is missing?"),
)
CORPUS_DOCS = (
    "SERVING.md", "ARCHITECTURE.md", "FLASHNEXT-PARITY.md", "RESULTS.md",
    "RESUME.md", "API-PARITY.md", "METRICS.md", "QUALIFICATION-EXPERIMENTS.md",
    "PROVENANCE.md", "ports/QWEN38-27B.md", "ports/QWEN36-35B-A3B.md",
    "ports/MUSE-GLIMMER.md", "ports/XING4-0.md",
)


def encode(tok, text):
    try:
        return list(tok.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tok.encode(text))


def corpus_tokens(tok, names=CORPUS_DOCS):
    parts = []
    for name in names:
        path = ROOT / "docs" / name
        parts.append((name, encode(tok, f"\n\n# {name}\n" + path.read_text(encoding="utf-8"))))
    return parts


def build_prompt(tok, context_tokens, doc_name, question, corpus):
    """Construct exactly one candidate-width prefill without repeating a document."""
    q = encode(tok, "\n\n" + question + "\n")
    ctx_len = context_tokens - len(q)
    if ctx_len < 1:
        raise ValueError(f"context_tokens {context_tokens} too small for the question")
    ordered = sorted(corpus, key=lambda part: part[0] != doc_name)
    if not ordered or ordered[0][0] != doc_name:
        raise ValueError(f"missing question document {doc_name}")
    if len(ordered[0][1]) < min(ctx_len // 4, 1024):
        raise ValueError(f"question document {doc_name} is too short")
    context = []
    sources = []
    for name, ids in ordered:
        take = ids[:ctx_len - len(context)]
        context.extend(take)
        if take:
            sources.append({"document": name, "tokens": len(take)})
        if len(context) == ctx_len:
            break
    if len(context) != ctx_len:
        raise ValueError(f"unique context has {len(context)} tokens, needs {ctx_len}")
    prompt = context + q
    return {"name": doc_name, "question": question, "prompt": prompt,
            "question_ids": q, "sources": sources}


def build_scoring_windows(corpus, context_tokens, holdout, count):
    """Nonoverlapping windows within the scoring phase, without wraparound."""
    if count < 1 or holdout < 1 or context_tokens < 1:
        raise ValueError("scoring window count, context and holdout must be positive")
    width = context_tokens + holdout
    windows = []
    for name, ids in corpus:
        for start in range(0, len(ids) - width + 1, width):
            span = ids[start:start + width]
            windows.append({"name": name, "start": start,
                            "prompt": span[:context_tokens],
                            "continuation": span[context_tokens:]})
            if len(windows) == count:
                return windows
    raise ValueError(f"only {len(windows)} disjoint scoring windows; need {count}")


def stop_token_ids(tok):
    ids = set(getattr(tok, "eos_token_ids", ()) or ())
    if getattr(tok, "eos_token_id", None) is not None:
        ids.add(tok.eos_token_id)
    return ids


def generation_status(tok, ids, text):
    return {"stopped": bool(ids and ids[-1] in stop_token_ids(tok)),
            "closed_thinking": "</think>" in text,
            "final_answer": text.split("</think>", 1)[-1].split("<|im_end|>", 1)[0].strip()
            if "</think>" in text else ""}


def phase_a_generate(mx, model, prompt_ids, gen_tokens, stop_ids=()):
    """Greedy generation, keeping each step's full logprob vector.

    Returns (token_ids, logprobs [S,V] fp32).  Memory is S x V, not context x V.
    """
    from mlx2.runtime.models.cache import make_prompt_cache

    cache = list(make_prompt_cache(model))
    logits = model(mx.array([prompt_ids], dtype=mx.uint32), cache=cache)
    lps = []
    ids = []
    lg = logits[0, -1].astype(mx.float32)
    for step in range(gen_tokens):
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        mx.eval(lp)
        lps.append(lp)
        nxt = int(mx.argmax(lp, axis=-1).item())
        ids.append(nxt)
        if nxt in stop_ids:
            break
        if step + 1 < gen_tokens:
            logits = model(mx.array([[nxt]], dtype=mx.uint32), cache=cache)
            lg = logits[0, -1].astype(mx.float32)
    del cache
    return ids, mx.stack(lps)


def phase_b_score(mx, model, context_ids, continuation_ids):
    """Teacher-force a fixed continuation after a context-only prefill.

    Alignment matters: the prefill's last position predicts continuation[0], and
    feeding continuation[i] yields the distribution that predicts
    continuation[i+1].  Pairing the i-th returned distribution with
    continuation[i] instead is an off-by-one that scores every token against the
    wrong context and produces an NLL worse than uniform.
    """
    from mlx2.runtime.models.cache import make_prompt_cache

    cache = list(make_prompt_cache(model))
    logits = model(mx.array([context_ids], dtype=mx.uint32), cache=cache)
    lg = logits[0, -1].astype(mx.float32)
    lps = []
    last = len(continuation_ids) - 1
    for i, tok_id in enumerate(continuation_ids):
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        mx.eval(lp)
        lps.append(lp)
        if i < last:
            logits = model(mx.array([[tok_id]], dtype=mx.uint32), cache=cache)
            lg = logits[0, -1].astype(mx.float32)
    del cache
    out = mx.stack(lps)
    mx.eval(out)
    return out


def run_arm(mx, model, spec, gen_tokens, lever, fgn, *, score=False, stop_ids=()):
    fgn.set_fused_group_norm_enabled(lever)
    fgn.reset_fused_group_norm_stats()
    try:
        t0 = time.perf_counter()
        if score:
            values = phase_b_score(mx, model, spec["prompt"], spec["continuation"])
        else:
            values = phase_a_generate(mx, model, spec["prompt"], gen_tokens, stop_ids)
        elapsed = time.perf_counter() - t0
        stats = fgn.fused_group_norm_stats()
    finally:
        fgn.set_fused_group_norm_enabled(False)
    if (lever and stats.get("calls", 0) == 0) or (not lever and stats.get("calls", 0)):
        raise RuntimeError(f"unexpected fused kernel engagement: lever={lever}, {stats}")
    return {"values": values, "seconds": elapsed, "counters": stats}


def divergence(mx, lp_a, lp_b, n):
    """KL and logit deltas over the first n positions, where both arms share a
    context.  Beyond a divergence the two caches hold different tokens, so
    comparing distributions there would compare different questions."""
    if n <= 0:
        return {"compared_positions": 0}
    a = lp_a[:n]
    b = lp_b[:n]
    mx.eval(a, b)
    pa = mx.exp(a)
    kl = mx.maximum(mx.sum(pa * (a - b), axis=-1), 0)
    kl_rev = mx.maximum(mx.sum(mx.exp(b) * (b - a), axis=-1), 0)
    mx.eval(kl, kl_rev)
    agree = (mx.argmax(a, axis=-1) == mx.argmax(b, axis=-1)).astype(mx.float32)
    mx.eval(agree)
    return {
        "compared_positions": n,
        "kl_mean": float(mx.mean(kl).item()),
        "kl_max": float(mx.max(kl).item()),
        "kl_reverse_mean": float(mx.mean(kl_rev).item()),
        "max_abs_logit_delta": float(mx.max(mx.abs(b - a)).item()),
        "top1_agreement": float(mx.mean(agree).item()),
    }


def perplexity(mx, lp, targets):
    """lp [S,V] fp32 log-probs; targets the S token ids actually scored."""
    idx = mx.array(targets, dtype=mx.int32)[:, None]
    taken = mx.take_along_axis(lp, idx, axis=-1)
    mx.eval(taken)
    nll = float(-mx.mean(taken).item())
    return math.exp(nll), nll


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",
                   default=str(Path.home() / "mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP"))
    p.add_argument("--context-tokens", type=int, default=8192)
    p.add_argument("--gen-tokens", type=int, default=1024)
    p.add_argument("--score-tokens", type=int, default=256,
                   help="held-out continuation length for the perplexity phase")
    p.add_argument("--score-windows", type=int, default=5)
    p.add_argument("--passes", type=int, default=1)
    p.add_argument("--limit", type=int, default=0,
                   help="run only the first N prompts (0 = all); for smoke tests")
    p.add_argument("--out", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--i-own-the-gpu", action="store_true")
    a = p.parse_args()
    if min(a.context_tokens, a.gen_tokens, a.score_tokens, a.score_windows, a.passes) < 1:
        p.error("context, generation, scoring, window count and passes must be positive")
    if a.limit < 0:
        p.error("limit must be nonnegative")

    plan = {
        "schema": SCHEMA,
        "model": a.model,
        "context_tokens": a.context_tokens,
        "gen_tokens": a.gen_tokens,
        "score_tokens": a.score_tokens,
        "score_windows": a.score_windows,
        "passes": a.passes,
        "prompts": [{"doc": d, "question": q} for d, q in PROMPT_SPECS],
        "corpus_docs": CORPUS_DOCS,
        "arms": {"eager": "MLX_QWEN4_FUSED_GROUP_NORM off (production)",
                 "fused": "MLX_QWEN4_FUSED_GROUP_NORM on"},
        "sampling": "greedy only",
        "difference_between_arms": "the lever only; one process, one loaded model",
        "phases": {
            "A": "answer quality: unique source context+question, greedy generate",
            "B": ("perplexity: nonoverlapping source-only scoring windows "
                  "with continuations held out from each prefill; source "
                  "documents may overlap phase A"),
        },
        "divergence_scope": ("phase A compares only the common generated prefix; "
                             "phase B compares all scored positions (same tokens)"),
    }
    if a.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2))
        return
    if not a.i_own_the_gpu:
        raise SystemExit("refusing Metal without --i-own-the-gpu")

    import mlx.core as mx

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.models import qwen4_fused_group_norm as fgn

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise SystemExit("this harness needs the GPU queue")

    t0 = time.perf_counter()
    adapter = resolve_adapter(a.model)(a.model)
    model, tok = adapter.model, adapter.tokenizer
    load_s = time.perf_counter() - t0
    print(f"model loaded in {load_s:.1f}s; peak Metal "
          f"{mx.get_peak_memory()/1e9:.2f} GB", flush=True)

    def decode(ids):
        try:
            return tok.decode(ids)
        except Exception:  # noqa: BLE001 -- tokenizer APIs differ across versions
            return "".join(tok.decode([i]) for i in ids)

    corpus = corpus_tokens(tok)
    specs = [build_prompt(tok, a.context_tokens, doc, q, corpus)
             for doc, q in (PROMPT_SPECS[:a.limit] if a.limit else PROMPT_SPECS)]
    windows = build_scoring_windows(corpus, a.context_tokens, a.score_tokens,
                                    a.score_windows)
    for s in specs:
        print(f"  {s['name']}: {s['sources']} + {len(s['question_ids'])} question "
              f"= {len(s['prompt'])} prompt", flush=True)
    print(f"  held-out scoring: {[(w['name'], w['start']) for w in windows]}",
          flush=True)

    report = dict(plan)
    report["mlx_version"] = mx.__version__
    report["device"] = str(mx.default_device())
    report["model_load_seconds"] = load_s
    report["host_started"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    report["passes"] = []

    for p_index in range(a.passes):
        order = ("eager", "fused") if p_index % 2 == 0 else ("fused", "eager")
        print(f"\n=== pass {p_index + 1}/{a.passes} order={order} ===", flush=True)
        entry = {"pass": p_index + 1, "order": list(order),
                 "prompts": [], "scoring_windows": []}
        for spec in specs:
            got = {arm: run_arm(mx, model, spec, a.gen_tokens, arm == "fused", fgn,
                                stop_ids=stop_token_ids(tok)) for arm in order}
            e, f = got["eager"], got["fused"]
            ge, le = e["values"]
            gf, lf = f["values"]
            prefix = 0
            while prefix < min(len(ge), len(gf)) and ge[prefix] == gf[prefix]:
                prefix += 1
            phase_a = divergence(mx, le, lf, prefix)
            phase_a.update({"common_prefix_tokens": prefix,
                            "generation_identical": ge == gf,
                            "first_divergent_token": None if ge == gf else prefix})
            text_e, text_f = decode(ge), decode(gf)
            entry["prompts"].append({
                "name": spec["name"], "question": spec["question"],
                "sources": spec["sources"], "prompt_tokens": len(spec["prompt"]),
                "phase_a_generation": phase_a,
                "generation_status": {"eager": generation_status(tok, ge, text_e),
                                      "fused": generation_status(tok, gf, text_f)},
                "counters": {arm: got[arm]["counters"] for arm in order},
                "seconds": {arm: got[arm]["seconds"] for arm in order},
                "answer_eager": text_e, "answer_fused": text_f,
            })
            print(json.dumps({"prompt": spec["name"], "phase_a": phase_a},
                             indent=2), flush=True)
            del got, e, f, le, lf
            mx.clear_cache()
        for window in windows:
            got = {arm: run_arm(mx, model, window, 0, arm == "fused", fgn,
                                score=True) for arm in order}
            e, f = got["eager"]["values"], got["fused"]["values"]
            cont = window["continuation"]
            ppl_e, nll_e = perplexity(mx, e, cont)
            ppl_f, nll_f = perplexity(mx, f, cont)
            phase_b = divergence(mx, e, f, len(cont))
            phase_b.update({"perplexity_eager": ppl_e, "perplexity_fused": ppl_f,
                            "perplexity_delta": ppl_f - ppl_e,
                            "perplexity_rel_delta": (ppl_f - ppl_e) / ppl_e,
                            "nll_eager": nll_e, "nll_fused": nll_f})
            entry["scoring_windows"].append({
                "name": window["name"], "start": window["start"],
                "prompt_tokens": len(window["prompt"]),
                "phase_b_perplexity": phase_b,
                "counters": {arm: got[arm]["counters"] for arm in order},
                "seconds": {arm: got[arm]["seconds"] for arm in order},
            })
            del got, e, f
            mx.clear_cache()
        entry["peak_mem_gb"] = mx.get_peak_memory() / 1e9
        mx.reset_peak_memory()
        report["passes"].append(entry)

    def collect(key, phase, group):
        vals = [pr[phase][key] for p_ in report["passes"] for pr in p_[group]
                if pr[phase].get(key) is not None]
        if not vals:
            return None
        return {"median": statistics.median(vals), "min": min(vals),
                "max": max(vals), "n": len(vals)}

    agg = {
        "phase_b_perplexity_rel_delta": collect("perplexity_rel_delta", "phase_b_perplexity", "scoring_windows"),
        "phase_b_kl_mean": collect("kl_mean", "phase_b_perplexity", "scoring_windows"),
        "phase_b_kl_max": collect("kl_max", "phase_b_perplexity", "scoring_windows"),
        "phase_b_max_abs_logit_delta": collect("max_abs_logit_delta", "phase_b_perplexity", "scoring_windows"),
        "phase_b_top1_agreement": collect("top1_agreement", "phase_b_perplexity", "scoring_windows"),
        "phase_a_common_prefix_tokens": collect("common_prefix_tokens", "phase_a_generation", "prompts"),
        "phase_a_kl_mean_over_prefix": collect("kl_mean", "phase_a_generation", "prompts"),
        "phase_a_max_abs_logit_delta": collect("max_abs_logit_delta", "phase_a_generation", "prompts"),
        "all_generations_identical": all(
            pr["phase_a_generation"]["generation_identical"]
            for p_ in report["passes"] for pr in p_["prompts"]),
        "completed_answers": sum(bool(pr["generation_status"][arm]["stopped"]
                                      and pr["generation_status"][arm]["final_answer"])
                                 for p_ in report["passes"] for pr in p_["prompts"]
                                 for arm in ("eager", "fused")),
    }
    comparisons = [pr for p_ in report["passes"]
                   for group in ("prompts", "scoring_windows") for pr in p_[group]]
    fused_calls = [pr["counters"]["fused"].get("calls", 0) for pr in comparisons]
    eager_calls = [pr["counters"]["eager"].get("calls", 0) for pr in comparisons]
    agg["fused_kernel_calls"] = {"min": min(fused_calls), "max": max(fused_calls)}
    agg["eager_arm_fused_kernel_calls_must_be_zero"] = max(eager_calls)
    agg["engagement_verified"] = (min(fused_calls) > 0 and max(eager_calls) == 0)
    report["aggregate"] = agg
    report["host_finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(f"\n=== aggregate ===\n{json.dumps(agg, indent=2)}", flush=True)
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
