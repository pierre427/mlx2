# SPDX-License-Identifier: MIT
"""GLM-5.3 prefill law A/B through the wired model path (round 2, WP5).

Arms switch only ``model.sparse_prefill`` (validated with the adapter's rules);
every other policy key stays the loaded policy, so the decode law is the same
in both arms and only the prompt law differs.

``breadth``  correctness: varied realistic prompts (scripts/realistic_haystack
             filler at distinct offsets, six task templates with checkable
             answers) per context; each arm runs independently (its own cache,
             its own natural greedy continuation, bounded top-k window at every
             step) and every candidate is classified against the reference
             under the AGENTS.md near-tie contract
             (glm53_policy_ab.classify_divergence: same-top-two, exact-tie and
             multi-way clauses, 0.5-nat ceiling, functional oracle on both
             continuations).
``ttft``     timing: one realistic needle prompt per context, arms strictly
             alternating A B A B ..., TTFT (sum of chunk times), Metal peak and
             the engagement counters per run. Not thermally gated: A/B signal,
             not publishable statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts"))

COUNTERS = ("mla_prompt_expanded_rows", "mla_prompt_expanded_calls",
            "sparse_attention_fused_v2", "sparse_attention_fused_v2_rows", "sparse_attention_fused_v2_dense_rows",
            "sparse_prefill_fused_v2_blocks", "sparse_prefill_gathered_blocks", "sparse_prefill_dense_pieces",
            "sparse_attention_fused_f16", "sparse_attention_fused_f16_rows", "sparse_prefill_fused_f16_blocks",
            "sparse_attention_gathered", "sparse_attention_mask", "mla_fused", "mla_fused_declined")

# Six task templates; each buries facts in varied filler and asks a question
# whose answer a functional oracle checks (dash-normalized, case-sensitive).
TEMPLATES = (
    {"name": "code_early", "facts": [(0.04, "The archive access code for the Lisbon office is QX-7194-LM.")],
     "question": "What is the archive access code for the Lisbon office? Reply with the code only.",
     "answers": ["QX-7194-LM"]},
    {"name": "code_middle", "facts": [(0.5, "The vault override phrase for the Kyoto lab is RT-5531-ZV.")],
     "question": "What is the vault override phrase for the Kyoto lab? Reply with the phrase only.",
     "answers": ["RT-5531-ZV"]},
    {"name": "code_late", "facts": [(0.93, "The badge number issued to the visiting auditor is 88-4127-KD.")],
     "question": "Which badge number was issued to the visiting auditor? Answer with the number only.",
     "answers": ["88-4127-KD"]},
    {"name": "two_facts", "facts": [(0.2, "The Oslo depot ticket reads OS-2290-PX."),
                                    (0.75, "The Porto depot ticket reads PT-6618-WQ.")],
     "question": "List the Oslo depot ticket and the Porto depot ticket, separated by a comma.",
     "answers": ["OS-2290-PX", "PT-6618-WQ"]},
    {"name": "arithmetic", "facts": [(0.3, "Warehouse North holds 1284 crates of spare fans."),
                                     (0.65, "Warehouse South holds 2371 crates of spare fans.")],
     "question": "How many crates of spare fans do Warehouse North and Warehouse South hold together? "
                 "Answer with the number only.",
     "answers": ["3655"], "patterns": [r"(?<![\w.,+-])3,?655(?![\w]|[.,]\d)"]},
    {"name": "attribute", "facts": [(0.45, ("Dr. Amara Okafor chairs the cache-coherence review board, "
                                             "which meets every second Thursday in room B-17."))],
     "question": "In which room does the cache-coherence review board meet? Reply with the room only.",
     "answers": ["B-17"]},
)


def oracle(text, answers):
    from glm53_policy_ab import normalize_dashes

    text = normalize_dashes(text)
    return all(re.search(a, text) for a in answers)


def patterns(template):
    """Oracle regexes: the exact answers (codes stay case-sensitive) unless a
    template lists its own (a thousands separator in a number)."""
    return template.get("patterns") or [r"(?<![\w-])" + re.escape(a) + r"(?![\w-])" for a in template["answers"]]


def render_prompt(adapter, template, tokens, offset):
    """Chat ids of filler (from ``offset``) + buried facts + question, as close to
    ``tokens`` as the filler granularity allows and never above it."""
    import realistic_haystack

    exclude = tuple(template["answers"])
    corpus = len(realistic_haystack.artifact_corpus(exclude=exclude))
    # Wrap the start so the filler (at most tokens * 6 characters) stays inside the corpus.
    offset %= corpus - tokens * 6

    def render(chars):
        text = realistic_haystack.haystack(chars, exclude=exclude, offset=offset)
        for fraction, fact in sorted(template["facts"], reverse=True):
            text = realistic_haystack.bury(text, fact, fraction)
        return adapter.prompt_tokens({"messages": [{"role": "user", "content": text + "\n\n" + template["question"]}],
                                      "enable_thinking": False})

    lo, hi, best = 0, tokens * 6, None
    for _ in range(26):
        if hi - lo <= 1:
            break
        mid = (lo + hi) // 2
        ids = render(mid)
        if len(ids) <= tokens:
            lo = mid
            if best is None or len(ids) > len(best):
                best = ids
            if len(ids) == tokens:
                break
        else:
            hi = mid
    if best is None or len(best) < tokens - 16:
        raise SystemExit(f"cannot render {template['name']} at {tokens} tokens")
    return best


# Which policy the arms switch (--law-key): the prompt attention law, or the
# prompt-phase decoded-weight GEMM law (glm53_prompt_decoded), whose engagement
# is read from the weights' route counters instead of the model's.
LAW_KEY = "sparse_prefill"
WEIGHT_COUNTERS = ("decoded_f32_grouped", "decoded_f32_dense", "decoded_f16_grouped", "decoded_f16_dense")


def set_law(adapter, value):
    if LAW_KEY == "mla_prompt":
        from mlx2.runtime.models.glm53_gguf import MLA_PROMPT_POLICIES

        if value not in MLA_PROMPT_POLICIES:
            raise ValueError(f"unknown glm53_mla_prompt value {value!r}")
        adapter.model.mla_prompt = value
        return
    if LAW_KEY == "prompt_decoded":
        from mlx2.runtime.gguf.weights import _PROMPT_DECODED_POLICIES

        if value not in ("reference",) + _PROMPT_DECODED_POLICIES:
            raise ValueError(f"unknown glm53_prompt_decoded value {value!r}")
        # The linear memo keys carry the policy, so a switch rebuilds them.
        adapter.weights.prompt_decoded_policy = value
        return
    from mlx2.adapters.glm53_gguf import _check_sparse_combination

    policy = {**adapter.kernel_policy, "glm53_sparse_prefill": value}
    _check_sparse_combination(policy)
    adapter.model.sparse_prefill = value


def weight_counters(adapter):
    w = adapter.weights
    return {k: w.scheduled_grouped_calls.get(k, 0) + w.scheduled_dense_calls.get(k, 0) for k in WEIGHT_COUNTERS}


def run_one(adapter, ids, decode, window, mx):
    from glm53_ordinary_profile import run_pass
    from glm53_policy_ab import decode_text, served

    model = adapter.model
    before = {k: model.calls.get(k, 0) for k in COUNTERS}
    before_weights = weight_counters(adapter)
    mx.clear_cache()
    mx.reset_peak_memory()
    peak = {}

    def on_phase(phase):
        if phase == "decode" and "prefill" not in peak:
            peak["prefill"] = mx.get_peak_memory()

    chunks, steps, out = run_pass(model, ids, 2048, decode, mx, on_phase=on_phase, window=window)
    text, eos_at = decode_text(adapter, out)
    return {"ttft_s": sum(chunks), "chunks_s": chunks,
            "decode_step_p50_ms": statistics.median(steps) * 1e3 if steps else None,
            "prefill_peak_bytes": peak.get("prefill", mx.get_peak_memory()), "tokens": out, "eos_at": eos_at,
            "served": served(out, eos_at), "windows": run_pass.windows, "finite": run_pass.finite, "text": text,
            "prefill_logits_sha256": run_pass.prefill_logits_sha256,
            "counters": {k: model.calls.get(k, 0) - before[k] for k in COUNTERS},
            "weight_counters": {k: v - before_weights[k] for k, v in weight_counters(adapter).items()},
            "sparse_prefill": model.sparse_prefill}


MLA_LAYERS = 11


def engaged(row, law, tokens):
    """Each arm's prompt law engaged on exactly the rows its positions select."""
    from mlx2.runtime.models import glm53_sparse_attention as sa

    if LAW_KEY == "mla_prompt":
        # Every prompt row below the loaded gathered threshold, over the 11
        # MLA layers; none on the reference.
        threshold = sa.PREFILL_THRESHOLDS[row["sparse_prefill"]]
        expanded = MLA_LAYERS * (tokens if threshold is None else min(tokens, threshold - 1))
        want = 0 if law == "reference" else expanded
        return row["counters"]["mla_prompt_expanded_rows"] == want
    if LAW_KEY == "prompt_decoded":
        # Per 2048-row chunk of at least 16 rows (prompt_rows_active): 11 MLA
        # layers x (k_b, v_b) grouped calls and 34 KDA layers x (ssm_f_b,
        # ssm_g_b) dense calls; none on the reference.
        chunks = tokens // 2048 + (1 if tokens % 2048 >= 16 else 0)
        w = row["weight_counters"]
        grouped = w["decoded_f32_grouped"] + w.get("decoded_f16_grouped", 0)
        dense = w["decoded_f32_dense"] + w.get("decoded_f16_dense", 0)
        if law == "reference":
            return grouped == 0 and dense == 0
        # The arm's own dtype must be the one that ran.
        own = "decoded_f16" if law == "decoded_f16_v1" else "decoded_f32"
        return (w.get(own + "_grouped", 0) == MLA_LAYERS * 2 * chunks == grouped
                and w.get(own + "_dense", 0) == 34 * 2 * chunks == dense)
    c = row["counters"]
    threshold = sa.PREFILL_THRESHOLDS[law]
    if threshold is None:  # dense_reference: masked pieces only
        return bool(c["sparse_attention_fused_v2"] == 0 and c["sparse_attention_fused_f16"] == 0
                    and c["sparse_prefill_gathered_blocks"] == 0
                    and (c["sparse_prefill_dense_pieces"] > 0) == (tokens > sa.TOPK))
    gathered_rows = MLA_LAYERS * max(0, tokens - (threshold - 1)) if tokens > sa.TOPK else 0
    masked = tokens > sa.TOPK and threshold - 1 > sa.TOPK
    fused_law = sa.prefill_law(law)
    if fused_law == "gathered_v1":
        ok = c["sparse_attention_fused_v2"] == 0 and c["sparse_attention_fused_f16"] == 0
        ok &= (c["sparse_prefill_gathered_blocks"] > 0) == (gathered_rows > 0)
        return bool(ok and (c["sparse_prefill_dense_pieces"] > 0) == masked)
    # Prompt rows below the threshold are the reference dense pieces (2026-11-01):
    # no fused dense rows at prompt time; the fused rows are the selected ones,
    # all of them under the arm's own kernel (fp32 v2 or fp16 operands).
    own, other = ("fused_f16", "fused_v2") if fused_law == "fused_f16" else ("fused_v2", "fused_f16")
    ok = c["sparse_attention_fused_v2_dense_rows"] == 0
    ok &= c[f"sparse_attention_{own}_rows"] == gathered_rows
    ok &= c[f"sparse_attention_{other}_rows"] == 0
    ok &= c["sparse_prefill_gathered_blocks"] == 0
    # _mla_masked counts dense pieces only when the chunk selects (total >
    # 2048): a prompt of <= 2048 tokens, or a threshold whose last dense row
    # lies inside the first 2048-row chunk, counts none.
    return bool(ok and (c["sparse_prefill_dense_pieces"] > 0) == masked)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("command", choices=("breadth", "ttft"))
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--contexts", default="8192,16384,32768")
    parser.add_argument("--reference", default=None, help="reference prompt law (default: the loaded policy's)")
    parser.add_argument("--candidate", default="gathered_v2_from_5120")
    parser.add_argument("--law-key", choices=("sparse_prefill", "prompt_decoded", "mla_prompt"), default="sparse_prefill",
                        help="which policy the arms switch: glm53_sparse_prefill, glm53_prompt_decoded or glm53_mla_prompt values")
    parser.add_argument("--templates", default=",".join(t["name"] for t in TEMPLATES))
    parser.add_argument("--offset-base", type=int, default=0, help="filler offset seed (distinct per job)")
    parser.add_argument("--control", default=None,
                        help="breadth: an extra arm (e.g. dense_reference) classified against the reference the "
                             "same way, as an ordinary-vs-ordinary sensitivity control; never gates the verdict")
    parser.add_argument("--decode", type=int, default=160)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--reps", type=int, default=2, help="ttft: A/B pairs per context")
    parser.add_argument("--allow-dirty-src", action="store_true",
                        help="record uncommitted src/ paths in the receipt instead of refusing (a shared checkout "
                             "with another session's unrelated edits); the receipt names them")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to reuse a nonempty evidence directory")
    args.out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx
    from glm53_decode_replay import load
    from glm53_ordinary_profile import CHECKPOINT
    from glm53_policy_ab import classify_divergence, host_record, source_record

    source = source_record(args.source_revision, allow_dirty=args.allow_dirty_src)
    global LAW_KEY
    LAW_KEY = args.law_key
    adapter, policy = load(args.policy)
    reference = args.reference or {"sparse_prefill": adapter.model.sparse_prefill,
                                   "prompt_decoded": adapter.weights.prompt_decoded_policy,
                                   "mla_prompt": adapter.model.mla_prompt}[LAW_KEY]
    arms = [("reference", reference), ("candidate", args.candidate)]
    if args.control and args.command == "breadth":
        arms.append(("control", args.control))
    for _, law in arms:
        set_law(adapter, law)
    contexts = [int(c) for c in args.contexts.split(",")]
    names = args.templates.split(",")
    known = {t["name"] for t in TEMPLATES}
    if not names or any(n not in known for n in names):
        raise SystemExit(f"--templates must name some of {sorted(known)}")
    templates = [t for t in TEMPLATES if t["name"] in names]
    record = {"schema": f"mlx2.glm53-v2-prefill-ab.{args.command}.v1", "checkpoint": CHECKPOINT, **source,
              "law_key": LAW_KEY,
              "host": host_record(mx), "policy": policy, "kernel_policy": adapter.kernel_policy,
              "arms": dict(arms), "contexts": contexts, "decode": args.decode, "window": args.window,
              "thermal": "not gated: correctness / A-B signal, not publishable statistics",
              "apc": "not applicable: direct model, fresh cache per run", "runs": [], "status": "running"}
    out_path = args.out / f"{args.command}.json"
    out_path.write_text(json.dumps(record, indent=1) + "\n")
    try:
        # Warm both laws' kernels (first-use gates, compiles) on a short prompt; not recorded.
        warm = render_prompt(adapter, TEMPLATES[0], 2600, 0)
        for _, law in arms:
            set_law(adapter, law)
            run_one(adapter, warm, 1, 0, mx)
        if args.command == "breadth":
            breadth(args, adapter, arms, contexts, templates, record, out_path, mx, classify_divergence)
        else:
            ttft(args, adapter, arms, contexts, record, out_path, mx)
        record["status"] = "done"
    except BaseException as error:
        record.update(status="error", error=repr(error))
        raise
    finally:
        set_law(adapter, reference)
        record["host_after"] = host_record(mx)
        out_path.write_text(json.dumps(record, indent=1) + "\n")
        adapter.close()
    print(json.dumps(record.get("summary"), indent=1), flush=True)
    return 0 if record.get("summary", {}).get("passed") else 3


def breadth(args, adapter, arms, contexts, templates, record, out_path, mx, classify_divergence):
    counts = {"exact": 0, "near_tie_equivalent": 0, "failed": 0, "unclassified": 0}
    control_counts = dict(counts)
    problems = []
    for ci, context in enumerate(contexts):
        for ti, template in enumerate(templates):
            # Distinct filler per (context, template): offsets 300K characters apart.
            offset = args.offset_base + (ci * len(TEMPLATES) + TEMPLATES.index(template)) * 300_000
            ids = render_prompt(adapter, template, context, offset)
            cell = {"context": context, "template": template["name"], "offset": offset, "prompt_tokens": len(ids),
                    "prompt_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(), "arms": {}}
            for name, law in (arms if ti % 2 == 0 else arms[::-1]):
                set_law(adapter, law)
                row = run_one(adapter, ids, args.decode, args.window, mx)
                row["oracle"] = oracle(row["text"], patterns(template))
                row["engaged"] = engaged(row, law, len(ids))
                cell["arms"][name] = row
            ref, cand = cell["arms"]["reference"], cell["arms"]["candidate"]
            cell["classification"] = classify_divergence(ref["served"], cand["served"], ref["windows"],
                                                         cand["windows"], ref["oracle"], cand["oracle"])
            if "control" in cell["arms"]:
                ctl = cell["arms"]["control"]
                cell["control_classification"] = classify_divergence(
                    ref["served"], ctl["served"], ref["windows"], ctl["windows"], ref["oracle"], ctl["oracle"])
                control_counts[cell["control_classification"]["verdict"]] += 1
            verdict = cell["classification"]["verdict"]
            counts[verdict] += 1
            if verdict not in ("exact", "near_tie_equivalent"):
                problems.append(f"{context} {template['name']}: {verdict}")
            for name, row in cell["arms"].items():
                # Numerical parity never excuses a wrong answer: each arm must pass.
                if not row["oracle"] and name != "control":
                    problems.append(f"{context} {template['name']} {name}: functional oracle failed")
                if not row["engaged"]:
                    problems.append(f"{context} {template['name']} {name}: engagement")
                if not row["finite"]:
                    problems.append(f"{context} {template['name']} {name}: nonfinite")
            record["runs"].append(cell)
            out_path.write_text(json.dumps(record, indent=1) + "\n")
            print(json.dumps({"context": context, "template": template["name"], "verdict": verdict,
                              "control": cell.get("control_classification", {}).get("verdict"),
                              "first_divergence": cell["classification"].get("first_divergence"),
                              "oracle": [ref["oracle"], cand["oracle"]],
                              "ttft_s": [round(r["ttft_s"], 2) for r in cell["arms"].values()],
                              "text": [ref["text"][:60], cand["text"][:60]]}), flush=True)
    record["summary"] = {"counts": counts, "control_counts": control_counts if len(arms) > 2 else None,
                         "problems": problems, "passed": not problems}


def ttft(args, adapter, arms, contexts, record, out_path, mx):
    summary = {}
    for context in contexts:
        ids = render_prompt(adapter, TEMPLATES[1], context, args.offset_base)
        for rep in range(args.reps):
            for name, law in arms:  # strictly A B A B
                set_law(adapter, law)
                t0 = time.perf_counter()
                row = run_one(adapter, ids, 4, 0, mx)
                row.update(context=context, rep=rep, arm=name, law=law, wall_s=time.perf_counter() - t0,
                           prompt_tokens=len(ids), engaged=engaged(row, law, len(ids)),
                           oracle=oracle(row["text"], patterns(TEMPLATES[1])))
                row.pop("windows")
                record["runs"].append(row)
                out_path.write_text(json.dumps(record, indent=1) + "\n")
                print(json.dumps({k: row[k] for k in ("context", "rep", "arm", "ttft_s", "prefill_peak_bytes",
                                                     "engaged", "counters")}), flush=True)
        for name, _ in arms:
            values = [r["ttft_s"] for r in record["runs"] if r["context"] == context and r["arm"] == name]
            summary.setdefault(str(context), {})[name] = {"ttft_s": values, "median": statistics.median(values)}
        s = summary[str(context)]
        s["delta_pct"] = 100 * (s["candidate"]["median"] / s["reference"]["median"] - 1)
    problems = [f"{r['context']} {r['arm']} rep {r['rep']}: engagement" for r in record["runs"] if not r["engaged"]]
    record["summary"] = {"ttft": summary, "problems": problems, "passed": not problems}


if __name__ == "__main__":
    sys.exit(main())
