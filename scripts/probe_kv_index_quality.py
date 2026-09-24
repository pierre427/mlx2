#!/usr/bin/env python3
"""KV quantization vs quantized-key token selection: speed, perplexity, answers.

One round = one model at one context length. Every arm in the round runs in
turn on a fresh cache with the same prompt:

* ``dense`` (run first and last, so timing drift is visible), ``kv_q8`` and
  ``kv_k8v4``: the serving KV-quantization conversions, applied after prefill.
* ``idx{8,4}_b{N}``: an fp16 cache plus a quantized key copy that picks each
  layer's own top-N rows (plus 128 recent) at decode
  (``runtime/quantized_key_index.py``). ``exact_b4096`` picks with the fp16
  keys instead: the selection-quality upper bound, which saves nothing.

Each arm gets the same three measurements:

1. **Speed:** median decode ms/token over a teacher-forced text segment,
   after warm-up steps.
2. **Perplexity:** on that same segment, KL(dense || arm), top-1 agreement,
   and added perplexity, against the round's first dense arm.
3. **Answers:** greedy answers to questions about facts planted in the
   prompt: single-hop access codes, plus two-hop "code of X's partner".
   They are scored by exact match, losses against dense, and text agreement
   with dense.

``--cpu-tiny`` validates the harness on a random-weight model, without
questions; its report is non-evidence. GPU mode refuses without
``--i-own-the-gpu``. ``--dry-run`` prints the plan.
Pre-registered gates: docs/experiments/KV-INDEX-2026-09-23.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REPORT_SCHEMA = "mlx2.kv-index-quality.v1"

ARMS = {
    "dense": {},
    "kv_q8": {"quantize": {"key_bits": 8, "value_bits": 8}},
    "kv_k8v4": {"quantize": {"key_bits": 8, "value_bits": 4}},
    "idx8_b4096": {"index": {"bits": 8, "budget": 4096}},
    "idx4_b4096": {"index": {"bits": 4, "budget": 4096}},
    "idx4_b8192": {"index": {"bits": 4, "budget": 8192}},
    "exact_b4096": {"index": {"exact_scores": True, "budget": 4096}},
}
DEFAULT_ARMS = "dense,kv_q8,kv_k8v4,idx8_b4096,idx4_b4096,idx4_b8192,exact_b4096,dense"
CANDIDATES = ("idx8_b4096", "idx4_b4096", "idx4_b8192")

# Pre-registered 2026-09-23, before any GPU data (see the experiment doc).
GATES = {
    "speed_context": 131072,
    "speedup_vs_dense_min": 1.10,
    "speedup_vs_kv_q8_min": 1.05,
    "kl_mean_max": 0.02,
    "kl_p99_max": 0.2,
    "top1_min": 0.97,
    "added_ppl_pct_max": 1.0,
    "answer_losses_max": 0,
}

NAMES = ("Orion", "Vega", "Lyra", "Draco", "Cygnus", "Perseus", "Aquila", "Hydra",
         "Carina", "Pavo", "Fornax", "Tucana")


# -------------------------------------------------------------------- inputs


def tiny_model():
    import mlx.core as mx

    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel

    args = TextModelArgs(
        model_type="qwen3_5", hidden_size=64, intermediate_size=64,
        num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, vocab_size=128, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=3, full_attention_interval=2,
        mtp_num_hidden_layers=0, partial_rotary_factor=0.5,
        rope_parameters=None, max_position_embeddings=8192,
    )
    mx.random.seed(7)
    model = TextModel(args)
    model.eval()
    for layer in model.model.layers:
        attention = getattr(layer, "self_attn", None)
        if attention is not None:
            attention.v_proj.weight = attention.v_proj.weight * 8.0
            attention.o_proj.weight = attention.o_proj.weight * 8.0
    mx.eval(model.parameters())
    return model


def build_workload(encode, filler, context, n_needles, n_twohop, tf_tokens, rng):
    """Prompt of exactly ``context`` tokens with planted facts, a text segment, questions."""
    names = rng.sample(NAMES, n_needles + n_twohop)
    needles = {n: f"{rng.randrange(10**5, 10**6)}" for n in names[:n_needles]}
    partners = dict(zip(names[n_needles:], rng.sample(list(needles), n_twohop)))
    facts = [f" The access code for {n} is {c}. " for n, c in needles.items()]
    facts += [f" {a}'s partner is {b}. " for a, b in partners.items()]
    rng.shuffle(facts)
    fact_ids = [list(encode(f)) for f in facts]
    body = list(filler[: context - sum(map(len, fact_ids))])
    depths = [int(len(body) * (0.04 + 0.92 * i / max(1, len(facts) - 1))) for i in range(len(facts))]
    for depth, ids in sorted(zip(depths, fact_ids), reverse=True):
        body[depth:depth] = ids
    if len(body) != context:
        raise ValueError("filler too short for context")
    text = list(filler[context : context + tf_tokens])
    questions = [{"kind": "single", "name": n, "code": c,
                  "prompt": f"\n\nQuestion: What is the access code for {n}?\n"
                            f"Answer: The access code for {n} is"}
                 for n, c in needles.items()]
    # Two-hop is a two-turn chain: ask for the partner, then for the code of
    # whichever name the model gave. Correct only if both hops are right. (A
    # one-step "code of X's partner" question was unanswerable even for dense
    # 9B, 0/3 in the 2026-09-23 validity round, so it measured nothing.)
    questions += [{"kind": "twohop", "name": a, "partner": b, "code": needles[b],
                   "prompt": f"\n\nQuestion: Who is {a}'s partner?\nAnswer: {a}'s partner is"}
                  for a, b in partners.items()]
    rng.shuffle(questions)
    return body, text, questions


def named_partner(text):
    """First word of a "X's partner is ..." answer, punctuation stripped."""
    words = re.findall(r"[A-Za-z]+", text.split("\n", 1)[0])
    return words[0] if words else ""


def follow_up_prompt(name):
    return (f"\n\nQuestion: What is the access code for {name}?\n"
            f"Answer: The access code for {name} is")


def score_answer(question, text, hop1=None):
    first_line = text.split("\n", 1)[0]
    codes = re.findall(r"\d{6}", first_line.replace(" ", ""))
    code_ok = bool(codes) and codes[0] == question["code"]
    if question["kind"] == "single":
        return code_ok
    return code_ok and named_partner(hop1 or "") == question["partner"]


# ---------------------------------------------------------------------- arms


def run_arm(model, arm, prompt, text, questions, *, encode, decode, eos, args):
    import mlx.core as mx

    from mlx2.runtime.models.cache import make_prompt_cache
    from mlx2.runtime.quantized_key_index import install_index, quantize_full_attention

    spec = ARMS[arm]
    mx.clear_cache()
    mx.reset_peak_memory()
    cache = make_prompt_cache(model)
    indexed = []
    if "index" in spec:
        cache, indexed = install_index(cache, window=args.window, **spec["index"])
    logits = None
    for start in range(0, len(prompt), args.prefill_step):
        logits = model(mx.array([prompt[start : start + args.prefill_step]], dtype=mx.uint32),
                       cache=cache)
        mx.eval(logits)
    converted = 0
    if "quantize" in spec:
        cache, converted = quantize_full_attention(cache, **spec["quantize"])
        if not converted:
            raise SystemExit(f"{arm}: no full-attention cache converted; refusing to report")
    for c in indexed:
        c.armed = True

    rows, step_ms = [logits[0, -1]], []
    for token in text[:-1]:
        started = time.perf_counter()
        out = model(mx.array([[token]], dtype=mx.uint32), cache=cache)[0, -1]
        mx.eval(out)
        step_ms.append((time.perf_counter() - started) * 1e3)
        rows.append(out)
    timed = step_ms[args.warmup_steps:]

    margins = []

    def pick(logits):
        """Greedy token; records the top-2 log-prob margin of every choice."""
        logp = logits.astype(mx.float32) - mx.logsumexp(logits.astype(mx.float32))
        top2 = mx.argpartition(-logp, kth=1)[:2]
        vals = logp[top2]
        first, second = (0, 1) if vals[0].item() >= vals[1].item() else (1, 0)
        token = int(top2[first].item())
        margins.append({"token": token, "runner_up": int(top2[second].item()),
                        "margin": float(vals[first].item() - vals[second].item())})
        return token

    def generate(prompt_text, cache=cache):
        out = model(mx.array([list(encode(prompt_text))], dtype=mx.uint32), cache=cache)
        token = pick(out[0, -1])
        generated = []
        for _ in range(args.answer_tokens):
            if token in eos:
                break
            generated.append(token)
            if "\n" in decode(generated):
                break
            out = model(mx.array([[token]], dtype=mx.uint32), cache=cache)
            token = pick(out[0, -1])
        return decode(generated)

    answers = []
    for q in questions:
        if q["kind"] == "single":
            answer = generate(q["prompt"])
            answers.append({"name": q["name"], "kind": "single", "answer": answer,
                            "correct": score_answer(q, answer)})
        else:
            hop1 = generate(q["prompt"])
            answer = generate(follow_up_prompt(named_partner(hop1) or "them"))
            answers.append({"name": q["name"], "kind": "twohop", "hop1": hop1,
                            "answer": answer, "hop1_correct": named_partner(hop1) == q["partner"],
                            "correct": score_answer(q, answer, hop1)})

    record = {
        "arm": arm,
        "decode_ms": statistics.median(timed) if timed else None,
        "decode_ms_p90": sorted(timed)[int(0.9 * (len(timed) - 1))] if timed else None,
        "timed_steps": len(timed),
        "converted_caches": converted,
        "indexed_calls": sum(c.counts["indexed"] for c in indexed),
        "index_layers": len(indexed),
        "index_bytes": sum(c.index_nbytes() for c in indexed),
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "answers": answers,
        "answer_margins": margins,
    }
    if indexed:
        expected = len(indexed) * (len(text) - 1)
        if record["indexed_calls"] < expected:
            raise SystemExit(f"{arm}: {record['indexed_calls']} indexed calls over the text "
                             f"segment, expected at least {expected}; refusing to report")
    del cache
    mx.clear_cache()
    return record, mx.stack(rows)


def compare(dense_rows, rows, text, dense_record, record):
    import mlx.core as mx

    from mlx2.runtime.kv_quant_fidelity import summarize, token_metrics

    target = mx.array(text, dtype=mx.uint32)
    m = token_metrics(dense_rows, rows, target)
    logp = rows.astype(mx.float32) - mx.logsumexp(rows.astype(mx.float32), axis=-1, keepdims=True)
    dense_logp = (dense_rows.astype(mx.float32)
                  - mx.logsumexp(dense_rows.astype(mx.float32), axis=-1, keepdims=True))
    nll = -mx.take_along_axis(logp, target[:, None], axis=-1).mean()
    dense_nll = -mx.take_along_axis(dense_logp, target[:, None], axis=-1).mean()
    mx.eval(m, nll, dense_nll)
    out = summarize({k: [float(x) for x in v.tolist()] for k, v in m.items()})
    out["ppl"] = math.exp(float(nll.item()))
    out["added_ppl_pct"] = (math.exp(float(nll.item()) - float(dense_nll.item())) - 1) * 100
    ref = {a["name"]: a for a in dense_record["answers"]}
    out["answer_accuracy"] = (sum(a["correct"] for a in record["answers"]) / len(record["answers"])
                              if record["answers"] else None)
    out["answer_losses"] = [a["name"] for a in record["answers"]
                            if ref[a["name"]]["correct"] and not a["correct"]]
    out["answer_gains"] = [a["name"] for a in record["answers"]
                           if a["correct"] and not ref[a["name"]]["correct"]]
    out["answer_text_agreement"] = (sum(a["answer"] == ref[a["name"]]["answer"]
                                        for a in record["answers"]) / len(record["answers"])
                                    if record["answers"] else None)
    return out


# ----------------------------------------------------------------------- run


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model")
    p.add_argument("--cpu-tiny", action="store_true")
    p.add_argument("--i-own-the-gpu", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--context", type=int, default=None, help="default GPU 65536, CPU 192")
    p.add_argument("--arms", default=DEFAULT_ARMS)
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--tf-tokens", type=int, default=256)
    p.add_argument("--warmup-steps", type=int, default=16)
    p.add_argument("--needles", type=int, default=8)
    p.add_argument("--twohop", type=int, default=3)
    p.add_argument("--answer-tokens", type=int, default=24)
    p.add_argument("--prefill-step", type=int, default=2048)
    p.add_argument("--cache-limit-gb", type=float, default=4.0)
    p.add_argument("--corpus", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path)
    p.add_argument("--save-reference", type=Path,
                   help="dense arm only: save its logits and answers for later arm processes")
    p.add_argument("--reference", type=Path,
                   help="compare against a saved dense reference instead of an in-process dense arm")
    args = p.parse_args(argv)
    if not args.cpu_tiny and not args.model:
        raise SystemExit("--model is required unless --cpu-tiny")
    arms = args.arms.split(",")
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}")
    if args.reference is None and arms[0] != "dense":
        raise SystemExit("the first arm must be dense unless --reference is given")
    if args.save_reference is not None and arms != ["dense"]:
        raise SystemExit("--save-reference runs the dense arm alone")
    if args.cpu_tiny:
        args.context = args.context or 192
        args.window, args.tf_tokens, args.warmup_steps = 8, 24, 2
        for spec in ARMS.values():  # scale budgets to the tiny context
            if "index" in spec:
                spec["index"] = dict(spec["index"], budget=spec["index"]["budget"] // 256)
    args.context = args.context or 65536
    if args.dry_run:
        print(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "arms": arms,
                          "gates": GATES}, indent=2))
        return 0
    if not args.cpu_tiny and not args.i_own_the_gpu:
        raise SystemExit("refusing Metal execution without --i-own-the-gpu")

    import mlx.core as mx

    mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    rng = random.Random(args.seed)
    if args.cpu_tiny:
        mx.set_default_device(mx.cpu)
        model = tiny_model()
        stream = [random.Random(args.seed + 1).randrange(1, 120)
                  for _ in range(args.context + args.tf_tokens)]
        prompt, text, questions = stream[: args.context], stream[args.context :], []
        encode = decode = None
        eos = set()
        identity = {"model": "cpu-tiny-qwen38-architecture"}
    else:
        from mlx2.adapters.registry import resolve_adapter

        adapter = resolve_adapter(args.model)(args.model)
        model, tokenizer = adapter.model, adapter.tokenizer

        def encode(s):
            try:
                return tokenizer.encode(s, add_special_tokens=False)
            except TypeError:
                return tokenizer.encode(s)

        decode = tokenizer.decode
        raw_eos = getattr(tokenizer, "eos_token_id", None)
        eos = set(raw_eos) if isinstance(raw_eos, (list, tuple, set)) else (
            {raw_eos} if raw_eos is not None else set())
        corpus = (args.corpus.read_text(encoding="utf-8") if args.corpus else
                  "\n\n".join(q.read_text(encoding="utf-8")
                              for q in sorted((ROOT / "docs").glob("*.md"))))
        filler = list(encode(corpus))
        while len(filler) < args.context + args.tf_tokens + 1:
            filler += filler
        prompt, text, questions = build_workload(encode, filler, args.context, args.needles,
                                                 args.twohop, args.tf_tokens, rng)
        identity = {"model": str(args.model), "fingerprint": adapter.identity.get("fingerprint"),
                    "corpus_sha256": hashlib.sha256(corpus.encode()).hexdigest()}

    report = {
        "schema": REPORT_SCHEMA, "evidence": not args.cpu_tiny,
        "qualification": False, "route_selected": False,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_revision": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       capture_output=True, text=True).stdout.strip(),
        "platform": platform.platform(), "mlx_version": getattr(mx, "__version__", None),
        "identity": identity, "context": args.context, "arms_run": arms,
        "questions": [{k: q[k] for k in ("kind", "name", "code")} | (
            {"partner": q["partner"]} if "partner" in q else {}) for q in questions],
        "gates": GATES, "arms": [],
    }
    # One arm per process keeps long-lived allocate/free cycles (which made
    # macOS write other apps' compressed memory to swap, 2026-09-23) out of
    # the run; the dense reference then travels between processes on disk.
    workload_sha = hashlib.sha256(json.dumps([prompt, text, [q["prompt"] for q in questions]]
                                             ).encode()).hexdigest()
    report["workload_sha256"] = workload_sha
    dense_rows = dense_record = None
    if args.reference is not None:
        meta = json.loads(args.reference.with_suffix(".json").read_text())
        if meta["workload_sha256"] != workload_sha:
            raise SystemExit("reference was made on a different prompt/text/questions; refusing")
        dense_rows = mx.load(str(args.reference.with_suffix(".npz")))["rows"]
        dense_record = meta["record"]
        report["reference"] = {"path": str(args.reference), "decode_ms": dense_record["decode_ms"]}
    for i, arm in enumerate(arms):
        started = time.perf_counter()
        record, rows = run_arm(model, arm, prompt, text, questions, encode=encode,
                               decode=decode, eos=eos, args=args)
        record["wall_s"] = time.perf_counter() - started
        if dense_rows is None:
            dense_rows, dense_record = rows, record
        else:
            record["vs_dense"] = compare(dense_rows, rows, text, dense_record, record)
        record["label"] = arm if arm != "dense" or (i == 0 and args.reference is None) else "dense_end"
        report["arms"].append(record)
        brief = {"arm": record["label"], "ms": round(record["decode_ms"], 2),
                 "peak_gb": round(record["peak_memory_gb"], 1), "wall_s": round(record["wall_s"], 1)}
        if "vs_dense" in record:
            v = record["vs_dense"]
            brief |= {"kl": round(v["kl_mean"], 4), "top1": round(v["top1_agreement"], 3),
                      "ppl_pct": round(v["added_ppl_pct"], 2), "acc": v["answer_accuracy"],
                      "losses": v["answer_losses"]}
        elif record["answers"]:
            brief["acc"] = sum(a["correct"] for a in record["answers"]) / len(record["answers"])
        print(json.dumps(brief), flush=True)

    if args.save_reference is not None:
        args.save_reference.parent.mkdir(parents=True, exist_ok=True)
        mx.savez(str(args.save_reference.with_suffix(".npz")), rows=dense_rows)
        args.save_reference.with_suffix(".json").write_text(json.dumps(
            {"workload_sha256": workload_sha, "record": dense_record}, default=str))
    dense_ms = dense_record["decode_ms"]
    q8 = next((r for r in report["arms"] if r["arm"] == "kv_q8"), None)
    for r in report["arms"]:
        r["speedup_vs_dense"] = dense_ms / r["decode_ms"]
        if q8:
            r["speedup_vs_kv_q8"] = q8["decode_ms"] / r["decode_ms"]
    text_out = json.dumps(report, indent=2, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text_out + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
