#!/usr/bin/env python3
"""Offline CPU replay: would copy-drafts inside self-MTP pay on real text?

No model runs.  Each recorded greedy completion is re-tokenized with the
Qwen tokenizer and replayed round by round through the *runtime's own*
``CopyDraftState`` (index, sizer, gate):

* a copy round's accepted length is the exact match against the recorded
  continuation (exact for greedy text);
* a head round is modelled as K Bernoulli(alpha) positions (geometric
  prefix), alpha taken from measured self-MTP acceptance;
* cost units follow the policy's cost model (head: 1 + (draft+row)*K,
  copy: 1 + row*W), so the answer is "tokens per unit verify cost relative
  to head-only self-MTP", swept over row costs because the NAX verify-width
  curve is a GPU measurement.

Corpora are recorded lab outputs (paths below); nothing is copied into mlx2.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlx2.runtime.copy_draft import CopyDraftPolicy, CopyDraftState  # noqa: E402

LAB = Path("~/Desktop/mlx-uag/results")
TOKENIZER = Path(
    "~/mlx-models/"
    "Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-oQ4e-mtp"
)


def _answer(text):
    """The answer the user received: recorded rows can run past the stop."""
    return text.split("<|im_end|>")[0].split("<|endoftext|>")[0]


def _load_35b_workloads(tok):
    sys.path.insert(0, str(LAB))
    source = (LAB / "pld-vs-compiled-35b-20260905.py").read_text()
    namespace = {}
    # Evaluate only the literal prompt material (DOC/CODE), not the driver.
    start = source.index("DOC = (")
    end = source.index("WORKLOADS = {")
    exec(source[start:end], namespace)  # noqa: S102 - lab-owned literals
    e2e_src = (LAB / "compiled-serving-e2e-20260905.py").read_text()
    p0 = e2e_src.index("PROMPT = (")
    p1 = e2e_src.index("SHORT_PROMPT")
    exec(e2e_src[p0:p1], namespace)  # noqa: S102
    prompts = {
        "retrieval": f"Here is a document:\n\n{namespace['DOCUMENT']}\n\nReproduce the document above verbatim, word for word.",
        "code_edit": f"Here is a function:\n\n```python\n{namespace['CODE']}```\n\nReturn the same function unchanged except rename `ids` to `token_ids` everywhere. Output only the code.",
        "no_retrieval": namespace["PROMPT"],
    }
    kinds = {"retrieval": "copy-heavy", "code_edit": "code", "no_retrieval": "prose"}
    data = json.loads((LAB / "pld-vs-compiled-35b-rerun-20260905.json").read_text())
    rows = []
    for name, workload in data["workloads"].items():
        for rep, run in enumerate(workload["runs"]["compiled"]):
            rows.append(
                {
                    "corpus": f"35b-{name}",
                    "kind": kinds[name],
                    "prompt": _chat(tok, prompts[name]),
                    "completion": tok.encode(_answer(run["text"]), add_special_tokens=False),
                    "id": f"{name}.{rep}",
                }
            )
    return rows


def _load_prose_10x10(tok, limit):
    corpus = json.loads((LAB / "qwen4-publication-20260912/domain-10x10.json").read_text())
    data = json.loads(
        (LAB / "qwen4-unified-10x10-fullopt-native-mtp-20260912T2340.json").read_text()
    )
    rows = []
    for row in data["rows"][:limit]:
        rows.append(
            {
                "corpus": "flashnext-10x10",
                "kind": "prose",
                "prompt": _chat(tok, row["question"], system=corpus["system"]),
                # Rows ran to the length cap past the stop token; replay only
                # the answer the user received.
                "completion": tok.encode(_answer(row["text"]), add_special_tokens=False),
                "id": row["case_id"],
            }
        )
    return rows


def _load_agnes(tok):
    rows = []
    for directory, kind in (
        ("agnes-pld-q6-code-greedy-fixed-20260912", "code"),
        ("agnes-pld-q6-document-greedy-20260912", "document"),
    ):
        base = LAB / directory
        formatted = next(base.glob("*.formatted.txt")).read_text()
        for path in sorted(base.glob("*-plain.txt")):
            rows.append(
                {
                    "corpus": f"agnes-{kind}",
                    "kind": kind,
                    "prompt": tok.encode(formatted, add_special_tokens=False),
                    "completion": tok.encode(_answer(path.read_text()), add_special_tokens=False),
                    "id": path.name,
                }
            )
    return rows


def _chat(tok, user, system=None):
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user}
    ]
    try:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except Exception:
        text = (system + "\n" if system else "") + user
    return tok.encode(text, add_special_tokens=False)


def replay(row, policy, *, alpha, depth, seed, copy_enabled):
    rng = random.Random(seed)
    stream = row["completion"]
    if len(stream) < 2:
        return None
    state = CopyDraftState(policy, list(row["prompt"]) + [stream[0]]) if copy_enabled else None
    t, cost, rounds = 1, 0.0, 0
    copy_rounds = copy_prop = copy_acc = declines = 0
    while t < len(stream):
        cap = len(stream) - t - 1
        span, decision = (state.plan(head_depth=depth, cap=cap) if state else ([], "off"))
        rounds += 1
        if decision == "declined":
            declines += 1
        if span:
            accepted = 0
            while accepted < len(span) and span[accepted] == stream[t + accepted]:
                accepted += 1
            emitted = min(accepted + 1, len(stream) - t)
            cost += policy.copy_cost(len(span))
            copy_rounds += 1
            copy_prop += len(span)
            copy_acc += accepted
            state.record(
                copy_span=len(span), head_depth=0, accepted=accepted,
                emitted=emitted, committed=stream[t : t + emitted],
            )
        else:
            k = min(depth, max(cap, 0))
            accepted = 0
            while accepted < k and rng.random() < alpha:
                accepted += 1
            emitted = min(accepted + 1, len(stream) - t)
            cost += policy.head_cost(k)
            if state:
                state.record(
                    copy_span=0, head_depth=k, accepted=accepted,
                    emitted=emitted, committed=stream[t : t + emitted],
                )
        t += emitted
    return {
        "tokens": len(stream) - 1,
        "cost": cost,
        "rounds": rounds,
        "copy_rounds": copy_rounds,
        "copy_proposed": copy_prop,
        "copy_accepted": copy_acc,
        "declines": declines,
    }


def evaluate(rows, policy, *, alpha, depth, seeds):
    by_corpus = {}
    for row in rows:
        for seed in range(seeds):
            base = replay(row, policy, alpha=alpha, depth=depth, seed=seed, copy_enabled=False)
            arm = replay(row, policy, alpha=alpha, depth=depth, seed=seed, copy_enabled=True)
            if base is None:
                continue
            slot = by_corpus.setdefault(row["corpus"], {"kind": row["kind"], "ratios": [], "arm": [], "base": []})
            slot["ratios"].append((arm["tokens"] / arm["cost"]) / (base["tokens"] / base["cost"]))
            slot["arm"].append(arm)
            slot["base"].append(base)
    summary = {}
    for corpus, slot in by_corpus.items():
        arm = slot["arm"]
        copy_rounds = sum(a["copy_rounds"] for a in arm)
        summary[corpus] = {
            "kind": slot["kind"],
            "samples": len(slot["ratios"]),
            "rel_tokens_per_cost_mean": round(statistics.mean(slot["ratios"]), 4),
            "rel_tokens_per_cost_min": round(min(slot["ratios"]), 4),
            "copy_round_fraction": round(copy_rounds / max(sum(a["rounds"] for a in arm), 1), 4),
            "copy_acceptance": round(
                sum(a["copy_accepted"] for a in arm) / max(sum(a["copy_proposed"] for a in arm), 1), 4
            ),
            "mean_accepted_per_copy_round": round(
                sum(a["copy_accepted"] for a in arm) / max(copy_rounds, 1), 3
            ),
            "tokens_per_round_arm": round(
                sum(a["tokens"] for a in arm) / max(sum(a["rounds"] for a in arm), 1), 3
            ),
            "tokens_per_round_head_only": round(
                sum(b["tokens"] for b in slot["base"]) / max(sum(b["rounds"] for b in slot["base"]), 1), 3
            ),
            "gate_declines": sum(a["declines"] for a in arm),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--prose-limit", type=int, default=100)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER))
    rows = _load_35b_workloads(tok) + _load_prose_10x10(tok, args.prose_limit) + _load_agnes(tok)
    corpora = {}
    for row in rows:
        c = corpora.setdefault(row["corpus"], {"rows": 0, "completion_tokens": 0, "prompt_tokens": 0})
        c["rows"] += 1
        c["completion_tokens"] += len(row["completion"])
        c["prompt_tokens"] += len(row["prompt"])
    grid = []
    for alpha in (0.74, 0.55):
        for row_cost in (0.05, 0.1, 0.25):
            for max_span, gated in ((8, True), (16, True), (31, True), (8, False)):
                policy = CopyDraftPolicy(
                    enabled=True, max_span=max_span, verify_row_cost=row_cost,
                    min_yield_ratio=1.0 if gated else 0.0,
                )
                grid.append(
                    {
                        "alpha": alpha, "depth": 2, "verify_row_cost": row_cost,
                        "max_span": max_span, "gated": gated,
                        "summary": evaluate(rows, policy, alpha=alpha, depth=2, seeds=args.seeds),
                    }
                )
                print(json.dumps({k: v for k, v in grid[-1].items() if k != "summary"}))
                for corpus, s in grid[-1]["summary"].items():
                    print(f"  {corpus:18s} rel={s['rel_tokens_per_cost_mean']:.3f} (min {s['rel_tokens_per_cost_min']:.3f}) copy_frac={s['copy_round_fraction']:.3f} acc={s['copy_acceptance']:.3f} acc/round={s['mean_accepted_per_copy_round']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"schema": "mlx2.copy-mtp-replay-probe.v1", "corpora": corpora, "grid": grid}, indent=2))


if __name__ == "__main__":
    main()
