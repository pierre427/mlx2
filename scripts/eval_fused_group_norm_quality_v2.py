#!/usr/bin/env python3
"""Source-bound, long-context quality gate for default-off fused GroupRMSNorm.

Twenty questions use distinct 16384-row prompts made from each source document
at most once. Their final answers need human rubric review; keyword hints are
diagnostic only. A separate experiments corpus supplies 20 non-overlapping
held-out continuations for teacher-forced perplexity. This is direct-model
evidence, not a serving latency or route-qualification receipt.

Run ``--dry-run`` before acquiring the GPU. A real run requires exclusive GPU
ownership and ``--i-own-the-gpu``. It leaves the production lever default-off.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import eval_fused_group_norm_quality as base

SCHEMA = "mlx2.fused-group-norm-quality.v2"
QA_DOCS = (
    "docs/ARCHITECTURE.md", "docs/QUALIFICATION.md", "docs/METRICS.md",
    "docs/FLASHNEXT-PARITY.md", "docs/RESUME.md", "docs/API-PARITY.md",
    "docs/PLD-POLICY.md", "docs/RESULTS.md", "docs/PROVENANCE.md",
    "docs/generative-media-adapters.md", "docs/ECOSYSTEM-PROBES-2026-09-19.md",
    "docs/ADVANCED_IMPORTS_REPORT_2026-09-18.md",
)
PPL_DOCS = tuple(str(p.relative_to(ROOT)) for p in
                 sorted((ROOT / "docs" / "experiments").glob("*.md")))


@dataclass(frozen=True)
class Case:
    source: str
    question: str
    rubric_hints: tuple[str, ...]


CASES = (
    Case(QA_DOCS[0], "Which prefix-cache engine owns serving state?", ("APCv2",)),
    Case(QA_DOCS[0], "Which four identities prevent cross-artifact prefix reuse?",
         ("revision", "model", "tokenizer", "layout")),
    Case(QA_DOCS[0], "What does ordinary decode publish at the prompt boundary?",
         ("checkpoint", "target")),
    Case(QA_DOCS[0], "Where may approximate state be published?",
         ("request-private", "APCv2")),
    Case(QA_DOCS[1], "What three questions does route qualification answer?",
         ("load", "default", "regression")),
    Case(QA_DOCS[1], "What does runtime_identity bind in a qualification receipt?",
         ("source", "MLX", "artifact", "settings")),
    Case(QA_DOCS[1], "Does a passing smoke test by itself qualify a route? Why?",
         ("smoke", "qualif")),
    Case(QA_DOCS[1], "Which GPU queue and file lock are required before a run?",
         ("gpuq.sh", "gpu.lock")),
    Case(QA_DOCS[2], "Does scraping /metrics synchronize or clear device state?",
         ("scrape", "synchron")),
    Case(QA_DOCS[2], "What expression is the canonical aggregate generation rate?",
         ("rate(mlx2_generation_tokens_total[1m])",)),
    Case(QA_DOCS[2], "Where does detailed request history remain available?",
         ("/v1/status",)),
    Case(QA_DOCS[2], "Which metric family distinguishes selected capability from engagement?",
         ("mlx2_capability", "engagement")),
    Case(QA_DOCS[3], "Name the four separate states of a mechanism.",
         ("implemented", "qualified", "selected", "observed")),
    Case(QA_DOCS[3], "Which four mechanisms did the optimized launcher explicitly disable?",
         ("whole-decode", "megakernel", "shared suffix", "rate gat")),
    Case(QA_DOCS[3], "What proves that a selected mechanism actually executed?",
         ("counter", "receipt")),
    Case(QA_DOCS[3], "Which prefix-cache engine is in the selected Flash-Next profile?",
         ("APCv2",)),
    Case(QA_DOCS[4], "What Qwen3.6 route is the candidate to select?",
         ("ordinary", "APCv2")),
    Case(QA_DOCS[4], "How did native MTP compare with ordinary for Qwen3.6?",
         ("0.35", "0.63")),
    Case(QA_DOCS[4], "Can the preserved Qwen3.6 evidence claim an MTP uplift?",
         ("no", "MTP")),
    Case(QA_DOCS[4], "Why does the preserved earlier-source evidence not select a route?",
         ("source", "qualif")),
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_bytes(path: str) -> bytes:
    return (ROOT / path).read_bytes()


def prepare_sources():
    names = tuple(dict.fromkeys((*QA_DOCS, *PPL_DOCS)))
    return {name: source_bytes(name) for name in names}


def build_qa(tok, context_tokens: int, cases=CASES, sources=None):
    """One copy of each file per prompt; question is included in exact row count."""
    sources = prepare_sources() if sources is None else sources
    out = []
    for index, case in enumerate(cases):
        q = base.encode(tok, "\n\nQuestion: " + case.question +
                        "\nAnswer in at most 80 words, then stop.\n")
        budget = context_tokens - len(q)
        if budget <= 8192:
            raise ValueError("question leaves <=8192 context tokens")
        order = (case.source,) + tuple(n for n in QA_DOCS if n != case.source)
        chunks = []
        used = []
        for name in order:
            header = f"\n\n=== Source: {name} ===\n"
            ids = base.encode(tok, header + sources[name].decode("utf-8"))
            take = min(len(ids), budget - len(chunks))
            if take <= 0:
                break
            chunks.extend(ids[:take])
            used.append({"path": name, "sha256": sha(sources[name]),
                         "tokens_used": take, "tokens_available": len(ids)})
        if len(chunks) != budget:
            raise ValueError(f"case {index}: only {len(chunks)}/{budget} source tokens")
        prompt = chunks + q
        out.append({"index": index, "source": case.source,
                    "question": case.question, "rubric_hints": case.rubric_hints,
                    "prompt": prompt, "prompt_sha256": sha(bytes_for_ids(prompt)),
                    "source_files": used, "source_tokens": len(chunks)})
    hashes = [v["prompt_sha256"] for v in out]
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate QA prompts")
    return out


def bytes_for_ids(ids):
    return b"".join(int(i).to_bytes(4, "little", signed=False) for i in ids)


def build_ppl(tok, context_tokens: int, score_tokens: int, count=20,
              sources=None):
    """Held-out files are disjoint from QA files; scored spans never overlap."""
    sources = prepare_sources() if sources is None else sources
    ids = []
    files = []
    for name in PPL_DOCS:
        part = base.encode(tok, f"\n\n=== Source: {name} ===\n" +
                           sources[name].decode("utf-8"))
        ids.extend(part)
        files.append({"path": name, "sha256": sha(sources[name]),
                      "tokens": len(part)})
    available = len(ids) - context_tokens - score_tokens
    if count < 2 or available < (count - 1) * score_tokens:
        raise ValueError("held-out corpus too small for distinct scored spans")
    starts = [i * available // (count - 1) for i in range(count)]
    if any(b - a < score_tokens for a, b in pairwise(starts)):
        raise ValueError("held-out scored continuations overlap")
    return [{"index": i, "context": ids[start:start + context_tokens],
             "continuation": ids[start + context_tokens:
                                 start + context_tokens + score_tokens],
             "context_sha256": sha(bytes_for_ids(ids[start:start + context_tokens])),
             "continuation_sha256": sha(bytes_for_ids(ids[start + context_tokens:
                                            start + context_tokens + score_tokens])),
             "corpus_offset": start, "source_files": files}
            for i, start in enumerate(starts)]


def stop_ids(tok):
    ids = set()
    ids.update(v for v in getattr(tok, "eos_token_ids", ())
               if isinstance(v, int) and v >= 0)
    raw = getattr(tok, "eos_token_id", None)
    if isinstance(raw, int) and raw >= 0:
        ids.add(raw)
    elif isinstance(raw, (list, tuple, set)):
        ids.update(v for v in raw if isinstance(v, int) and v >= 0)
    convert = getattr(tok, "convert_tokens_to_ids", None)
    if callable(convert):
        unknown = getattr(tok, "unk_token_id", None)
        for spelling in ("<|im_end|>", "<|endoftext|>"):
            value = convert(spelling)
            if isinstance(value, int) and value >= 0 and value != unknown:
                ids.add(value)
    return ids


def generate(mx, model, tok, prompt, cap: int):
    from mlx2.runtime.models.cache import make_prompt_cache

    cache = list(make_prompt_cache(model))
    t0 = time.perf_counter()
    logits = model(mx.array([prompt], dtype=mx.uint32), cache=cache)
    mx.eval(logits)
    prefill_s = time.perf_counter() - t0
    lg = logits[0, -1].astype(mx.float32)
    ids = []
    terminal = stop_ids(tok)
    reason = "token_cap"
    t0 = time.perf_counter()
    for _ in range(cap):
        nxt = int(mx.argmax(lg, axis=-1).item())
        if nxt in terminal:  # stop before appending chat control tokens
            reason = "stop_token"
            break
        ids.append(nxt)
        logits = model(mx.array([[nxt]], dtype=mx.uint32), cache=cache)
        lg = logits[0, -1].astype(mx.float32)
    mx.eval(lg)
    decode_s = time.perf_counter() - t0
    del cache
    raw = tok.decode(ids)
    think_closed = "</think>" in raw
    final = raw.rsplit("</think>", 1)[-1].strip() if think_closed else ""
    return {"ids": ids, "text": raw, "final_answer": final,
            "think_closed": think_closed,
            "answer_complete": reason == "stop_token" and think_closed and len(final) >= 20,
            "stop_reason": reason, "prefill_seconds": prefill_s,
            "decode_seconds": decode_s}


def run_arm(mx, model, tok, case, ppl, lever, fgn, gen_cap):
    fgn.set_fused_group_norm_enabled(lever)
    fgn.reset_fused_group_norm_stats()
    try:
        qa = generate(mx, model, tok, case["prompt"], gen_cap)
        t0 = time.perf_counter()
        lp = base.phase_b_score(mx, model, ppl["context"], ppl["continuation"])
        score_s = time.perf_counter() - t0
        stats = fgn.fused_group_norm_stats()
    finally:
        fgn.set_fused_group_norm_enabled(False)
    if lever and stats.get("calls", 0) < 2:
        raise RuntimeError(f"fused arm did not engage both prefills: {stats}")
    if not lever and stats.get("calls", 0):
        raise RuntimeError(f"eager arm launched fused kernel: {stats}")
    return qa, lp, score_s, stats


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(Path.home() / "mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP"))
    p.add_argument("--context-tokens", type=int, default=16384)
    p.add_argument("--score-tokens", type=int, default=128)
    p.add_argument("--gen-cap", type=int, default=768)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--out", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--i-own-the-gpu", action="store_true")
    args = p.parse_args(argv)
    if args.context_tokens != 16384 or args.limit < 20 or args.limit > len(CASES):
        p.error("quality gate requires 20 cases at exactly 16384 rows")
    if args.score_tokens < 32 or args.gen_cap < 256:
        p.error("score-tokens must be >=32 and gen-cap >=256")

    # Dry run can load only the tokenizer when available; no model/Metal import.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False,
                                        local_files_only=True)
    sources = prepare_sources()
    cases = build_qa(tok, args.context_tokens, CASES[:args.limit], sources)
    ppl = build_ppl(tok, args.context_tokens, args.score_tokens, args.limit, sources)
    plan = {"schema": SCHEMA, "source_revision": None,
            "model": args.model, "context_tokens": args.context_tokens,
            "gen_cap": args.gen_cap, "score_tokens": args.score_tokens,
            "case_count": len(cases), "heldout_corpus": "docs/experiments/*.md",
            "qa_sources": list(QA_DOCS), "source_file_hashes":
            {name: sha(data) for name, data in sources.items()},
            "cases": [{k: v for k, v in case.items() if k != "prompt"}
                      for case in cases],
            "ppl_windows": [{k: v for k, v in window.items()
                             if k not in ("context", "continuation", "source_files")}
                            for window in ppl],
            "scope": "direct model A/B; neither serving timing nor qualification"}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.i_own_the_gpu:
        p.error("refusing Metal without --i-own-the-gpu")

    import mlx.core as mx

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.models import qwen4_fused_group_norm as fgn

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise SystemExit("this harness needs the exclusive Metal GPU queue")
    if fgn.fused_group_norm_enabled():
        raise SystemExit("start with production fused GroupRMSNorm lever off")
    plan["source_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    plan["harness_sha256"] = sha(source_bytes("scripts/eval_fused_group_norm_quality_v2.py"))
    plan["kernel_sha256"] = sha(source_bytes("src/mlx2/runtime/models/qwen4_fused_group_norm.py"))
    plan["model_config_sha256"] = sha((Path(args.model) / "config.json").read_bytes())
    model_adapter = resolve_adapter(args.model)(args.model)
    model = model_adapter.model
    tok = model_adapter.tokenizer
    # Rebuild with the exact tokenizer used by the loaded model.
    cases = build_qa(tok, args.context_tokens, CASES[:args.limit], sources)
    ppl = build_ppl(tok, args.context_tokens, args.score_tokens, args.limit, sources)
    plan["mlx_version"] = mx.__version__
    plan["device"] = str(mx.default_device())
    plan["results"] = []
    for case, window in zip(cases, ppl):
        order = ("eager", "fused") if case["index"] % 2 == 0 else ("fused", "eager")
        arms = {}
        lps = {}
        for arm in order:
            qa, lp, score_s, counters = run_arm(
                mx, model, tok, case, window, arm == "fused", fgn, args.gen_cap)
            arms[arm] = {**qa, "score_seconds": score_s, "counters": counters}
            lps[arm] = lp
        target = window["continuation"]
        eager_ppl, eager_nll = base.perplexity(mx, lps["eager"], target)
        fused_ppl, fused_nll = base.perplexity(mx, lps["fused"], target)
        numeric = base.divergence(mx, lps["eager"], lps["fused"], len(target))
        eids, fids = arms["eager"]["ids"], arms["fused"]["ids"]
        prefix = 0
        while prefix < min(len(eids), len(fids)) and eids[prefix] == fids[prefix]:
            prefix += 1
        hints = tuple(s.lower() for s in case["rubric_hints"])
        for arm in arms.values():
            arm["rubric_hint_hits"] = [s for s in hints if s in arm["final_answer"].lower()]
            arm.pop("ids")
        item = {"index": case["index"], "question": case["question"],
                "source": case["source"], "prompt_sha256": case["prompt_sha256"],
                "continuation_sha256": window["continuation_sha256"],
                "order": order, "arms": arms,
                "generation_common_prefix_tokens": prefix,
                "generation_identical": eids == fids,
                "heldout_perplexity": {"eager": eager_ppl, "fused": fused_ppl,
                                        "relative_delta": (fused_ppl - eager_ppl) / eager_ppl,
                                        "eager_nll": eager_nll, "fused_nll": fused_nll,
                                        **numeric},
                "human_rubric_review": "pending"}
        plan["results"].append(item)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(plan, indent=2))
        print(f"case {case['index'] + 1}/{len(cases)}: "
              f"complete={arms['eager']['answer_complete']}/"
              f"{arms['fused']['answer_complete']} "
              f"ppl delta={(fused_ppl - eager_ppl) / eager_ppl:+.3%}", flush=True)
        del lps
        mx.clear_cache()
    deltas = [r["heldout_perplexity"]["relative_delta"] for r in plan["results"]]
    plan["aggregate"] = {"ppl_relative_delta_median": statistics.median(deltas),
                         "ppl_relative_delta_max_abs": max(abs(x) for x in deltas),
                         "completed_answer_pairs": sum(
                             all(a["answer_complete"] for a in r["arms"].values())
                             for r in plan["results"]),
                         "rubric_review": "pending_human_review",
                         "qualification": "not_qualified"}
    plan["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    Path(args.out).write_text(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
