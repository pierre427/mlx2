#!/usr/bin/env python3
"""bf16-head vs fp32-head logits: near-tie flips and decode cost (GPU).

Loads one Qwen3.8 text adapter (27B or Flash-Next) in-process, then:

Agreement (teacher-forced, fixed prompt set): N sequences of L tokens from this
checkout's docs and sources.  One trunk forward per sequence gives the final
hidden states; on those same hiddens it compares greedy argmax of
  bf16   the shipped quantized head (bf16 store),
  fp32   the same head with fp32 scales/biases (fp32 store, opt-in switch),
against two references:
  head_ref   fp32 matmul of the fp32 hidden with the dequantized head (fp32
             weights), i.e. the head's exact answer for this trunk;
  fp32_trunk the whole trunk re-run with fp32 activations (``--fp32-trunk``;
             every floating parameter widened, quantized weights unchanged)
             and the fp32 head: a teacher-forced fp32 reference.
Near ties are positions whose head_ref top-1/top-2 gap is under two bf16 ulps
of the top-1 logit.

Cost: ABBA-interleaved blocks of B=1 ordinary decode steps (ms/step), plus
the head alone at M=1 and M=3 rows (native-MTP d2 verify width).

Design input: Inco Splash (Apache-2.0, f786bed) PR #141; no code used.
Refuses to run without ``--i-own-the-gpu``.
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


def corpus_tokens(tokenizer, count: int, length: int):
    parts = []
    for pattern in ("docs/*.md", "src/mlx2/*.py", "src/mlx2/runtime/*.py"):
        for path in sorted(ROOT.glob(pattern)):
            parts.append(path.read_text(errors="ignore"))
    text = "\n\n".join(parts)
    stride = max(1, (len(text) - length * 8) // count)
    out = []
    for index in range(count):
        chunk = text[index * stride : index * stride + length * 8]
        ids = tokenizer.encode(chunk)
        out.append(ids[:length])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--sequences", type=int, default=16)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--vocab-chunk", type=int, default=32768)
    parser.add_argument("--decode-blocks", type=int, default=8)
    parser.add_argument("--decode-steps", type=int, default=24)
    parser.add_argument("--fp32-trunk", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()
    if not args.i_own_the_gpu:
        print("refusing: pass --i-own-the-gpu under the GPU lock wrapper", file=sys.stderr)
        return 2

    import mlx.core as mx

    from mlx2.adapters.registry import resolve_adapter

    mx.set_cache_limit(4 << 30)
    t0 = time.perf_counter()
    adapter = resolve_adapter(args.model)(args.model)
    load_s = time.perf_counter() - t0
    lm = adapter.model.language_model
    head = lm.lm_head
    tokenizer = adapter.tokenizer
    sequences = corpus_tokens(tokenizer, args.sequences, args.length)

    scales16, biases16 = head["scales"], head.get("biases")
    scales32 = scales16.astype(mx.float32)
    biases32 = None if biases16 is None else biases16.astype(mx.float32)
    mx.eval(scales32, *(() if biases32 is None else (biases32,)))

    def use(variant):
        head.scales = scales32 if variant == "fp32" else scales16
        if biases16 is not None:
            head.biases = biases32 if variant == "fp32" else biases16

    vocab = head["weight"].shape[0]

    def head_ref_top2(h32):
        """Top-2 of the fp32 dequantized head, chunked over the vocabulary."""
        best_v, best_i = None, None
        for r0 in range(0, vocab, args.vocab_chunk):
            r1 = min(vocab, r0 + args.vocab_chunk)
            w = mx.dequantize(
                head["weight"][r0:r1], scales32[r0:r1],
                None if biases32 is None else biases32[r0:r1],
                group_size=head.group_size, bits=head.bits, mode=head.mode,
            ).astype(mx.float32)
            logits = h32 @ w.T  # [L, r]
            idx = mx.argpartition(-logits, 1, axis=-1)[:, :2]
            vals = mx.take_along_axis(logits, idx, axis=-1)
            idx = idx + r0
            if best_v is None:
                best_v, best_i = vals, idx
            else:
                cv = mx.concatenate([best_v, vals], axis=-1)
                ci = mx.concatenate([best_i, idx], axis=-1)
                order = mx.argsort(-cv, axis=-1)[:, :2]
                best_v = mx.take_along_axis(cv, order, axis=-1)
                best_i = mx.take_along_axis(ci, order, axis=-1)
            mx.eval(best_v, best_i)
        order = mx.argsort(-best_v, axis=-1)
        return mx.take_along_axis(best_v, order, axis=-1), mx.take_along_axis(best_i, order, axis=-1)

    def trunk(tokens):
        cache = adapter.model.make_cache()
        h = lm.model(mx.array(tokens)[None], cache)
        mx.eval(h)
        return h[0]

    rows = []
    stored = []
    for index, tokens in enumerate(sequences):
        h = trunk(tokens)
        use("bf16")
        lb = head(h)
        arg_b = mx.argmax(lb, axis=-1)
        use("fp32")
        l32 = head(h)
        arg_32 = mx.argmax(l32, axis=-1)
        mx.eval(arg_b, arg_32)
        dtypes = (str(lb.dtype), str(l32.dtype))
        del lb, l32
        top_v, top_i = head_ref_top2(h.astype(mx.float32))
        top1 = top_v[:, 0]
        gap = top_v[:, 0] - top_v[:, 1]
        # bf16 ulp at |x|: 2^(floor(log2|x|) - 7)
        ulp = mx.power(2.0, mx.floor(mx.log2(mx.maximum(mx.abs(top1), 1e-6))) - 7)
        near = gap < 2 * ulp
        ref = top_i[:, 0]
        b, f, r, n = (x.tolist() for x in (arg_b, arg_32, ref, near))
        stored.append((b, f))
        rows.append({
            "sequence": index, "positions": len(b), "dtypes": dtypes,
            "bf16_vs_head_ref": sum(int(x == y) for x, y in zip(b, r)),
            "fp32_vs_head_ref": sum(int(x == y) for x, y in zip(f, r)),
            "bf16_vs_fp32": sum(int(x == y) for x, y in zip(b, f)),
            "near_ties": sum(int(x) for x in n),
            "bf16_flips_at_near_ties": sum(int(x != y and t) for x, y, t in zip(b, r, n)),
            "fp32_flips_at_near_ties": sum(int(x != y and t) for x, y, t in zip(f, r, n)),
        })
        print(json.dumps(rows[-1]), flush=True)

    def total(key):
        return sum(row[key] for row in rows)

    positions = total("positions")
    agreement = {
        "positions": positions,
        "bf16_top1_vs_head_ref": total("bf16_vs_head_ref") / positions,
        "fp32_top1_vs_head_ref": total("fp32_vs_head_ref") / positions,
        "bf16_vs_fp32_top1": total("bf16_vs_fp32") / positions,
        "near_ties": total("near_ties"),
        "bf16_flips_at_near_ties": total("bf16_flips_at_near_ties"),
        "fp32_flips_at_near_ties": total("fp32_flips_at_near_ties"),
    }

    # Cost: head alone, then B=1 ordinary decode, ABBA blocks.
    cost = {"head_only_ms": {}, "decode_ms_per_step": {}}
    for m in (1, 3):
        x = mx.random.normal((1, m, lm.args.hidden_size)).astype(mx.bfloat16)
        mx.eval(x)
        samples = {"bf16": [], "fp32": []}
        for block in range(16):
            for variant in (("bf16", "fp32") if block % 2 == 0 else ("fp32", "bf16")):
                use(variant)
                mx.eval(head(x))
                start = time.perf_counter()
                for _ in range(20):
                    mx.eval(head(x))
                samples[variant].append((time.perf_counter() - start) / 20 * 1e3)
        cost["head_only_ms"][f"M{m}"] = {k: statistics.median(v) for k, v in samples.items()}

    prompt = mx.array(sequences[0])[None]
    cache = adapter.model.make_cache()
    logits = adapter.model(prompt, cache)
    token = mx.argmax(logits[:, -1:], axis=-1)
    mx.eval(token)
    samples = {"bf16": [], "fp32": []}
    for block in range(args.decode_blocks):
        for variant in (("bf16", "fp32") if block % 2 == 0 else ("fp32", "bf16")):
            use(variant)
            for _ in range(2):  # warm the variant's kernels
                token = mx.argmax(adapter.model(token, cache)[:, -1:], axis=-1)
                mx.eval(token)
            start = time.perf_counter()
            for _ in range(args.decode_steps):
                token = mx.argmax(adapter.model(token, cache)[:, -1:], axis=-1)
                mx.eval(token)
            samples[variant].append((time.perf_counter() - start) / args.decode_steps * 1e3)
    cost["decode_ms_per_step"] = {
        k: {"median": statistics.median(v), "blocks": v} for k, v in samples.items()
    }
    med = {k: v["median"] for k, v in cost["decode_ms_per_step"].items()}
    cost["decode_delta_pct"] = (med["fp32"] / med["bf16"] - 1.0) * 100
    del cache

    trunk_ref = None
    if args.fp32_trunk:
        try:
            use("fp32")
            lm.model.set_dtype(mx.float32)
            agree_b = agree_f = count = 0
            for (b, f), tokens in zip(stored, sequences):
                h = trunk(tokens)
                ref = mx.argmax(head(h), axis=-1).tolist()
                agree_b += sum(int(x == y) for x, y in zip(b, ref))
                agree_f += sum(int(x == y) for x, y in zip(f, ref))
                count += len(ref)
            trunk_ref = {
                "positions": count,
                "bf16_top1_vs_fp32_trunk": agree_b / count,
                "fp32_top1_vs_fp32_trunk": agree_f / count,
            }
        except Exception as error:  # noqa: BLE001 - recorded, not fatal
            trunk_ref = {"error": repr(error)}

    result = {
        "model": args.model,
        "load_s": load_s,
        "sequences": args.sequences,
        "length": args.length,
        "head": {"bits": head.bits, "group_size": head.group_size, "mode": head.mode,
                 "vocab": vocab, "fp32_extra_bytes": int(scales32.nbytes - scales16.nbytes)
                 + (0 if biases16 is None else int(biases32.nbytes - biases16.nbytes))},
        "agreement": agreement,
        "fp32_trunk_reference": trunk_ref,
        "cost": cost,
        "peak_memory_gib": mx.get_peak_memory() / (1 << 30),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("agreement", "fp32_trunk_reference", "cost")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
