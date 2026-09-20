#!/usr/bin/env python3
"""Measure approximate-KV fidelity and cost: exact KV vs kv_q8 / kv_k8v4.

Per operation and context length this records, against the exact cache of the
same model on the same token stream (runtime/kv_quant_fidelity.py):

* teacher-forced per-token KL(exact || quant), top-1 agreement, top-5 overlap,
  exact-token log-prob delta (mean / p99 / max);
* needle retrieval: facts planted at several depths, greedy answers on both
  arms; ``quant_losses`` counts needles the exact arm retrieved and the
  quantized arm did not;
* attention-plane bytes (measured ``nbytes``, not estimated) and the ratio;
* decode tok/s for both arms, interleaved A B A B (back-to-back drift ~10 %).

The quantized arm is produced by the adapter-declared operation through
``KVQuantizationOperation.apply`` (the serving path).  An arm whose mechanism
count is wrong (0 quantized planes, or quantized planes on the exact arm) is
refused, not reported.

Modes:

* ``--cpu-tiny``: tiny random-weight Qwen3.8-architecture model on CPU.  It
  validates the harness and the mechanism counters; its numbers are not
  evidence about real models, and its report can never pass the gate (device
  and contexts are wrong by construction).
* GPU (``--model PATH``): Metal.  Refuses to run without ``--i-own-the-gpu``;
  run it under the lab GPU lock wrapper.  ``--dry-run`` prints the plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NEEDLE_NAMES = (
    "Orion", "Vega", "Lyra", "Draco", "Cygnus", "Perseus", "Aquila", "Hydra",
    "Carina", "Pavo", "Fornax", "Tucana",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", help="Model artifact path (GPU mode)")
    p.add_argument("--cpu-tiny", action="store_true", help="CPU harness self-check")
    p.add_argument("--i-own-the-gpu", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--operations", default="kv_q8,kv_k8v4")
    p.add_argument("--contexts", default=None,
                   help="Comma list (default GPU 4096,16384,32768,65536; CPU 64,192)")
    p.add_argument("--score-tokens", type=int, default=None,
                   help="Teacher-forced positions per context (default GPU 256, CPU 48)")
    p.add_argument("--prefill-step", type=int, default=2048)
    p.add_argument("--needles", type=int, default=None,
                   help="Needles per context (default GPU 9, CPU 3); 0 disables")
    p.add_argument("--needle-contexts", default=None,
                   help="Contexts that get needle checks (default: all)")
    p.add_argument("--decode-tokens", type=int, default=None,
                   help="Greedy decode tokens per timed repeat (default GPU 128, CPU 8)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--corpus", type=Path,
                   help="UTF-8 text corpus (default: this repository's docs/*.md)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tiny-attention-gain", type=float, default=8.0,
                   help="CPU only: scale attention v/o so quantization is visible")
    p.add_argument("--out", type=Path, required=False)
    return p.parse_args(argv)


def plan(args):
    contexts = [int(x) for x in args.contexts.split(",")]
    return {
        "mode": "cpu-tiny" if args.cpu_tiny else "gpu",
        "model": args.model,
        "operations": args.operations.split(","),
        "contexts": contexts,
        "score_tokens": args.score_tokens,
        "needles": args.needles,
        "needle_contexts": args.needle_contexts,
        "decode_tokens": args.decode_tokens,
        "repeats": args.repeats,
        "prefill_step": args.prefill_step,
        "out": str(args.out) if args.out else None,
    }


# --------------------------------------------------------------------- models


def tiny_model(gain):
    import mlx.core as mx

    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel

    args = TextModelArgs(
        model_type="qwen3_5", hidden_size=64, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=64, vocab_size=128, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=3, full_attention_interval=2,
        mtp_num_hidden_layers=0, partial_rotary_factor=0.5,
        rope_parameters=None, max_position_embeddings=4096,
    )
    mx.random.seed(7)
    model = TextModel(args)
    model.eval()
    if gain != 1.0:
        for layer in model.model.layers:
            attention = getattr(layer, "self_attn", None)
            if attention is not None:
                attention.v_proj.weight = attention.v_proj.weight * gain
                attention.o_proj.weight = attention.o_proj.weight * gain
    mx.eval(model.parameters())
    return model


def load_gpu_adapter(path):
    from mlx2.adapters.registry import resolve_adapter

    adapter_type = resolve_adapter(path)
    return adapter_type(path)


# --------------------------------------------------------------------- corpus


def corpus_text(path):
    if path is not None:
        return path.read_text(encoding="utf-8")
    parts = [p.read_text(encoding="utf-8") for p in sorted((ROOT / "docs").glob("*.md"))]
    return "\n\n".join(parts)


def token_stream(encode, text, needed):
    ids = list(encode(text))
    if not ids:
        raise ValueError("corpus encodes to no tokens")
    while len(ids) < needed:
        ids = ids + ids
    return ids[:needed]


# ------------------------------------------------------------------- measures


def greedy(model, cache, prompt_ids, steps, prefill_step):
    import mlx.core as mx

    ids = mx.array([list(prompt_ids)], dtype=mx.uint32)
    logits = None
    for start in range(0, ids.shape[1], prefill_step):
        logits = model(ids[:, start : start + prefill_step], cache=cache)
        mx.eval(logits)
    out = []
    token = mx.argmax(logits[0, -1])
    for _ in range(steps):
        out.append(int(token.item()))
        logits = model(token.reshape(1, 1), cache=cache)
        token = mx.argmax(logits[0, -1])
    return out


def needle_answer_len(encode, code):
    """Greedy tokens needed to emit the whole code after "... is".

    Qwen-family tokenizers split digits one per token and emit the leading
    space as its own token, so " 042917." is 8 tokens; a fixed 6 truncated
    the last digit and the exact arm retrieved nothing (a vacuous needle
    check, 2026-09-19 27B run).  Two tokens of margin for punctuation.
    """
    return len(list(encode(f" {code}."))) + 2


def needle_checks(model, make_cache, operation, *, context, count, encode, decode,
                  filler_ids, prefill_step, rng):
    """Plant ``count`` facts at spread depths; ask for each on both arms."""
    from mlx2.runtime.approximate_kv import LaneKVState

    results = []
    depths = [(i + 1) / (count + 1) for i in range(count)]
    for index, depth in enumerate(depths):
        name = NEEDLE_NAMES[index % len(NEEDLE_NAMES)]
        if decode is None:  # CPU tiny: id-pattern needle
            key = [100 + index % 20, 120 + index % 8]
            value = [rng.randrange(1, 90) for _ in range(3)]
            fact, question, answer_len = key + value, key, 3
        else:
            code = f"{rng.randrange(0, 10**6):06d}"
            fact = list(encode(f"\nThe pass code for {name} is {code}.\n"))
            question = list(encode(
                f"\n\nQuestion: What is the pass code for {name}?\n"
                f"Answer: The pass code for {name} is"
            ))
            value, answer_len = code, needle_answer_len(encode, code)
        body = context - len(fact) - len(question)
        if body < 8:
            raise ValueError(f"context {context} too short for needles")
        cut = int(body * depth)
        prompt = filler_ids[:cut] + fact + filler_ids[cut:body] + question
        arms, answers = {}, {}
        for arm in ("exact", "quant"):
            cache = list(make_cache())
            if arm == "quant":
                cache = list(operation.apply(
                    LaneKVState(operation.revision, tuple(cache))).planes)
            answer = greedy(model, cache, prompt, answer_len, prefill_step)
            answers[arm] = answer
            if decode is None:
                arms[arm] = answer == value
            else:
                arms[arm] = value in decode(answer).replace(" ", "")
        results.append({
            "depth": round(depth, 3), "name": name, **arms,
            "answers_agree": answers["exact"] == answers["quant"],
            "expected": value if decode is not None else None,
            "answers": (None if decode is None
                        else {arm: decode(ids) for arm, ids in answers.items()}),
        })
    exact_hits = sum(r["exact"] for r in results)
    quant_hits = sum(r["quant"] for r in results)
    losses = sum(1 for r in results if r["exact"] and not r["quant"])
    return {
        "total": len(results),
        "exact_hits": exact_hits,
        "quant_hits": quant_hits,
        "quant_losses": losses,
        # Greedy answer identity is informative even where neither arm
        # retrieves (the CPU tiny model cannot).
        "answers_agree": sum(r["answers_agree"] for r in results),
        "detail": results,
    }


def decode_rates(model, make_cache, operation, *, prompt_ids, decode_tokens,
                 repeats, prefill_step):
    """Interleaved A B A B greedy decode tok/s after one shared-length prefill."""
    import mlx.core as mx

    from mlx2.runtime.approximate_kv import LaneKVState

    caches = {"exact": list(make_cache())}
    caches["quant"] = list(operation.apply(
        LaneKVState(operation.revision, tuple(make_cache()))).planes)
    tokens = {}
    ids = mx.array([list(prompt_ids)], dtype=mx.uint32)
    for arm, cache in caches.items():
        logits = None
        for start in range(0, ids.shape[1], prefill_step):
            logits = model(ids[:, start : start + prefill_step], cache=cache)
            mx.eval(logits)
        tokens[arm] = mx.argmax(logits[0, -1])
    rates = {"exact": [], "quant": []}
    for _ in range(repeats):
        for arm in ("exact", "quant"):
            token = tokens[arm]
            mx.synchronize()
            start = time.perf_counter()
            for _ in range(decode_tokens):
                logits = model(token.reshape(1, 1), cache=caches[arm])
                token = mx.argmax(logits[0, -1])
                mx.eval(token)
            mx.synchronize()
            rates[arm].append(decode_tokens / (time.perf_counter() - start))
            tokens[arm] = token
    median = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
    return {
        "decode_tokens": decode_tokens,
        "repeats": repeats,
        "exact_tok_s": rates["exact"],
        "quant_tok_s": rates["quant"],
        "exact_tok_s_median": median(rates["exact"]),
        "quant_tok_s_median": median(rates["quant"]),
        "quant_over_exact": median(rates["quant"]) / median(rates["exact"]),
    }


# ----------------------------------------------------------------------- main


def run(args):
    import mlx.core as mx

    from mlx2.runtime.approximate_kv import KVQuantizationOperation, declared_operations
    from mlx2.runtime.kv_quant_fidelity import (
        BUNDLE_SCHEMA,
        REPORT_SCHEMA,
        evaluate_fidelity_report,
        measure_teacher_forced,
    )
    from mlx2.runtime.models.cache import make_prompt_cache

    rng = random.Random(args.seed)
    contexts = [int(x) for x in args.contexts.split(",")]
    needle_contexts = (
        set(contexts) if args.needle_contexts is None
        else {int(x) for x in args.needle_contexts.split(",")}
    )
    if args.cpu_tiny:
        mx.set_default_device(mx.cpu)
        from mlx2.runtime.approximate_kv import standard_kv_quantization_operations

        model = tiny_model(args.tiny_attention_gain)
        fingerprint = "cpu-tiny-qwen38-architecture"
        descriptors = standard_kv_quantization_operations(group_size=64)
        operations = {
            name: KVQuantizationOperation(name, d, adapter_fingerprint=fingerprint)
            for name, d in descriptors.items()
        }
        encode = decode = None
        stream_rng = random.Random(args.seed + 1)
        needed = max(contexts) + args.score_tokens + 1
        stream = [stream_rng.randrange(1, 100) for _ in range(needed)]
        corpus_sha = hashlib.sha256(json.dumps(stream).encode()).hexdigest()
        device = "cpu"
    else:
        adapter = load_gpu_adapter(args.model)
        model = adapter.model
        fingerprint = adapter.identity["fingerprint"]
        operations = declared_operations(adapter, adapter_fingerprint=fingerprint)
        tokenizer = adapter.tokenizer
        def encode(text):
            try:
                return tokenizer.encode(text, add_special_tokens=False)
            except TypeError:  # wrapper without the HF keyword
                return tokenizer.encode(text)

        decode = tokenizer.decode
        text = corpus_text(args.corpus)
        corpus_sha = hashlib.sha256(text.encode()).hexdigest()
        stream = token_stream(encode, text, max(contexts) + args.score_tokens + 1)
        device = "gpu"
    make_cache = lambda: make_prompt_cache(model)  # noqa: E731
    selected = args.operations.split(",")
    missing = [name for name in selected if name not in operations]
    if missing:
        raise SystemExit(f"adapter does not declare operations {missing}")

    bundle = {
        "schema": BUNDLE_SCHEMA,
        "harness": {
            "name": "scripts/measure_kv_quant_fidelity.py",
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "plan": plan(args),
        "platform": platform.platform(),
        "mlx_version": getattr(mx, "__version__", None),
        "reports": {},
        "verdicts": {},
    }
    for name in selected:
        operation = operations[name]
        entries = []
        for context in contexts:
            started = time.perf_counter()
            entry = measure_teacher_forced(
                model, stream, context=context, score_tokens=args.score_tokens,
                operation=operation, prefill_step=args.prefill_step,
                make_cache=make_cache,
            )
            if args.needles and context in needle_contexts:
                entry["needles"] = needle_checks(
                    model, make_cache, operation, context=context,
                    count=args.needles, encode=encode, decode=decode,
                    filler_ids=stream, prefill_step=args.prefill_step, rng=rng,
                )
            if args.decode_tokens:
                entry["decode"] = decode_rates(
                    model, make_cache, operation, prompt_ids=stream[:context],
                    decode_tokens=args.decode_tokens, repeats=args.repeats,
                    prefill_step=args.prefill_step,
                )
            entry["wall_s"] = time.perf_counter() - started
            entries.append(entry)
            print(json.dumps({"operation": name, **{k: v for k, v in entry.items()
                              if k not in ("needles", "decode")}}), flush=True)
        report = {
            "schema": REPORT_SCHEMA,
            "device": device,
            "adapter_fingerprint": fingerprint,
            "operation": name,
            "operation_revision": operation.revision,
            "descriptor": operation.descriptor.as_dict(),
            "corpus_sha256": corpus_sha,
            "contexts": entries,
        }
        bundle["reports"][name] = report
        bundle["verdicts"][name] = evaluate_fidelity_report(
            report, operation=name, adapter_fingerprint=fingerprint
        )
    return bundle


def main(argv=None):
    args = parse_args(argv)
    if not args.cpu_tiny and not args.model:
        raise SystemExit("--model is required unless --cpu-tiny")
    gpu = not args.cpu_tiny
    defaults = (
        ("4096,16384,32768,65536", 256, 9, 128) if gpu else ("64,192", 48, 3, 8)
    )
    args.contexts = args.contexts or defaults[0]
    if gpu and args.needle_contexts is None:
        args.needle_contexts = "4096,32768"  # 18 long prefills per context otherwise
    args.score_tokens = args.score_tokens or defaults[1]
    args.needles = defaults[2] if args.needles is None else args.needles
    args.decode_tokens = defaults[3] if args.decode_tokens is None else args.decode_tokens
    if args.dry_run:
        print(json.dumps(plan(args), indent=2))
        return 0
    if gpu and not args.i_own_the_gpu:
        raise SystemExit("refusing Metal execution without --i-own-the-gpu")
    bundle = run(args)
    text = json.dumps(bundle, indent=2, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(json.dumps({"verdicts": {k: {"passed": v["passed"], "failures": v["failures"][:6]}
                                   for k, v in bundle["verdicts"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
