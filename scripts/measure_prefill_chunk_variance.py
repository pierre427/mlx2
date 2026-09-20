#!/usr/bin/env python3
"""rm15: does the prefill chunk size change a model's exact outputs?

rm07 found that an exact-path control arm differing ONLY in prefill chunk
size (4096 vs 2048) agreed with the exact reference on 0.891 of top-1 tokens
at 16K on Qwen3.8-27B, with max |delta logit| up to 6.4, while a bit-for-bit
repeat of the same chunk size agreed 1.000 with KL exactly 0.0.  This harness
reproduces that on demand and localises it.

Arms at every context C (each gets its OWN fresh cache; arms run one at a
time so only one cache is resident):

  full   one chunk of C           -- the UNCHUNKED GROUND TRUTH
  <n>    chunks of n tokens
  <n>r   a bit-for-bit repeat of arm <n>  -- the determinism control

Everything is measured against ``full``, so the question "is ONE chunk size
right and the other wrong, or do they merely differ?" is answerable and not
assumed.  Three measurements per arm:

  1. first-divergence: each decoder layer's output at the LAST prefill
     position, so the first layer that differs is named with its magnitude;
  2. the prefill cache state (KV keys/values, GDN conv + recurrent state),
     diffed field by field;
  3. teacher-forced decode over the following window: top-1 agreement,
     mean KL and max |delta logit|.

Metal: run under the GPU queue only.  --dry-run prints the plan and exits.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SCHEMA = "mlx2.rm15.prefill-chunk-variance.v1"


def corpus_text() -> str:
    parts = [p.read_text(encoding="utf-8") for p in sorted((ROOT / "docs").glob("*.md"))]
    if not parts:
        raise SystemExit("no corpus: docs/*.md is empty")
    return "\n\n".join(parts)


def token_stream(encode, text, needed):
    ids = list(encode(text))
    if not ids:
        raise ValueError("corpus encodes to no tokens")
    while len(ids) < needed:
        ids = ids + ids
    return ids[:needed]


def parse_arms(spec, ctx):
    """'full,2048,2048r,4096' -> [(label, step, is_repeat), ...]."""
    arms = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if raw == "full":
            arms.append((raw, ctx, False))
        elif raw.endswith("r"):
            arms.append((raw, int(raw[:-1]), True))
        else:
            arms.append((raw, int(raw), False))
    return arms


# ------------------------------------------------------------------ capture


def layer_names(model):
    layers = model.language_model.model.layers
    out = []
    for i, layer in enumerate(layers):
        kind = "gdn" if getattr(layer, "is_linear", False) else "attn"
        out.append(f"L{i}.{kind}")
    return out


def instrument(model, capture):
    """Patch the decoder-layer class so each layer records its last-position
    output.  Special methods resolve on the type, so an instance attribute
    would never be seen."""
    layers = model.language_model.model.layers
    index = {id(l): i for i, l in enumerate(layers)}
    cls = type(layers[0])
    if getattr(cls, "_rm15_patched", False):
        return
    orig = cls.__call__

    def wrapped(self, x, mask=None, cache=None):
        out = orig(self, x, mask=mask, cache=cache)
        i = index.get(id(self))
        if i is not None:
            capture[i] = out[:, -1]
        return out

    wrapped._rm15_patched = True
    cls.__call__ = wrapped
    cls._rm15_patched = True


def cache_fields(entry, tail):
    """{name: array} for one cache entry, trimmed to what is actually used."""
    import mlx.core as mx

    out = {}
    if hasattr(entry, "keys") and getattr(entry, "keys", None) is not None:
        k, v = entry.keys_and_values()
        out["keys_tail"] = k[..., -tail:, :]
        out["values_tail"] = v[..., -tail:, :]
        out["offset"] = mx.array([float(entry.offset)])
    else:
        for i, arr in enumerate(getattr(entry, "cache", []) or []):
            if isinstance(arr, mx.array):
                out[f"arr{i}"] = arr
        off = getattr(entry, "offset", None)
        if off is not None:
            out["offset"] = mx.array([float(off)])
    return out


# --------------------------------------------------------------------- arms


def run_arm(model, stream, ctx, step, score_tokens, capture, names, tail):
    import mlx.core as mx
    from mlx2.runtime.models.cache import make_prompt_cache

    cache = list(make_prompt_cache(model))
    t0 = time.perf_counter()
    for start in range(0, ctx, step):
        capture.clear()
        mx.eval(model(stream[:, start : min(start + step, ctx)], cache=cache))
    hiddens = {}
    for i, val in capture.items():
        v = val.astype(mx.float32)
        mx.eval(v)
        hiddens[names[i]] = v
    state = []
    for i, entry in enumerate(cache):
        fields = {}
        for name, arr in cache_fields(entry, tail).items():
            a = arr.astype(mx.float32)
            mx.eval(a)
            fields[name] = a
        kind = "gdn" if getattr(model.language_model.model.layers[i], "is_linear", False) else "attn"
        state.append((kind, fields))
    prefill_s = time.perf_counter() - t0

    logits = []
    for j in range(score_tokens):
        lg = model(stream[:, ctx + j : ctx + j + 1], cache=cache)[0, -1]
        lg = lg.astype(mx.float32)
        mx.eval(lg)
        logits.append(lg)
    out = mx.stack(logits)
    mx.eval(out)
    del cache
    mx.clear_cache()
    return {"hiddens": hiddens, "state": state, "logits": out,
            "prefill_s": prefill_s}


def compare(arm, ref):
    import mlx.core as mx

    first = None
    per_layer = []
    for name, val in ref["hiddens"].items():
        other = arm["hiddens"].get(name)
        if other is None or other.shape != val.shape:
            per_layer.append({"layer": name, "note": "shape mismatch"})
            continue
        d = float(mx.max(mx.abs(other - val)).item())
        per_layer.append({"layer": name, "max_abs_delta": d})
        if first is None and d > 0.0:
            first = {"layer": name, "max_abs_delta": d}

    state_rows = []
    first_state = None
    for i, (kind, fields) in enumerate(ref["state"]):
        other = arm["state"][i][1] if i < len(arm["state"]) else {}
        for name, val in fields.items():
            o = other.get(name)
            if o is None or o.shape != val.shape:
                state_rows.append({"layer": i, "kind": kind, "field": name,
                                   "note": "shape mismatch"})
                continue
            d = float(mx.max(mx.abs(o - val)).item())
            if d != 0.0:
                row = {"layer": i, "kind": kind, "field": name,
                       "max_abs_delta": d}
                state_rows.append(row)
                if first_state is None:
                    first_state = dict(row)

    a, r = arm["logits"], ref["logits"]
    lr = r - mx.logsumexp(r, axis=-1, keepdims=True)
    la = a - mx.logsumexp(a, axis=-1, keepdims=True)
    pr = mx.exp(lr)
    kl = mx.maximum(mx.sum(pr * (lr - la), axis=-1), 0)
    agree = (mx.argmax(a, axis=-1) == mx.argmax(r, axis=-1)).astype(mx.float32)
    mx.eval(kl, agree)
    return {
        "first_divergent_layer": first,
        "first_divergent_cache_field": first_state,
        "n_differing_cache_fields": len(state_rows),
        "differing_cache_fields": state_rows[:40],
        "per_layer_last_pos_max_abs_delta": per_layer,
        "decode_top1_agreement": float(mx.mean(agree).item()),
        "decode_kl_mean": float(mx.mean(kl).item()),
        "decode_kl_max": float(mx.max(kl).item()),
        "decode_max_abs_logit_delta": float(mx.max(mx.abs(a - r)).item()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--contexts", default="4096,16384")
    p.add_argument("--arms", default="full,2048,2048r,4096",
                   help="'full' is the unchunked ground truth; a trailing 'r' "
                        "repeats that chunk size as a determinism control")
    p.add_argument("--score-tokens", type=int, default=256)
    p.add_argument("--cache-tail", type=int, default=16,
                   help="how many trailing KV positions to diff")
    p.add_argument("--out", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--i-own-the-gpu", action="store_true")
    a = p.parse_args()

    contexts = [int(x) for x in a.contexts.split(",")]
    plan = {"schema": SCHEMA, "model": a.model, "contexts": contexts,
            "arms": a.arms, "score_tokens": a.score_tokens,
            "cache_tail": a.cache_tail, "out": a.out,
            "reference_arm": "full (one chunk of the whole context)"}
    if a.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2))
        return
    if not a.i_own_the_gpu:
        raise SystemExit("refusing Metal without --i-own-the-gpu")

    import mlx.core as mx
    from mlx2.adapters.registry import resolve_adapter

    adapter = resolve_adapter(a.model)(a.model)
    model, tok = adapter.model, adapter.tokenizer

    def enc(t):
        try:
            return tok.encode(t, add_special_tokens=False)
        except TypeError:
            return tok.encode(t)

    ids = token_stream(enc, corpus_text(), max(contexts) + a.score_tokens + 1)
    stream = mx.array([ids], dtype=mx.uint32)

    capture = {}
    names = layer_names(model)
    instrument(model, capture)

    report = dict(plan)
    report["mlx_version"] = mx.__version__
    report["n_layers"] = len(names)
    report["results"] = []
    for ctx in contexts:
        arms = parse_arms(a.arms, ctx)
        ref = None
        entry = {"context": ctx, "arms": {}}
        t0 = time.perf_counter()
        for label, step, _repeat in arms:
            r = run_arm(model, stream, ctx, step, a.score_tokens, capture,
                        names, a.cache_tail)
            if ref is None:
                ref = r
                entry["arms"][label] = {"prefill_step": step,
                                        "n_chunks": (ctx + step - 1) // step,
                                        "reference": True,
                                        "prefill_s": r["prefill_s"]}
                continue
            cmp = compare(r, ref)
            cmp["prefill_step"] = step
            cmp["n_chunks"] = (ctx + step - 1) // step
            cmp["prefill_s"] = r["prefill_s"]
            entry["arms"][label] = cmp
            del r
            mx.clear_cache()
        entry["wall_s"] = time.perf_counter() - t0
        report["results"].append(entry)
        print(json.dumps({k: v for k, v in entry.items() if k != "arms"}
                         | {"arms": {k: {kk: vv for kk, vv in v.items()
                                         if kk != "per_layer_last_pos_max_abs_delta"
                                         and kk != "differing_cache_fields"}
                                     for k, v in entry["arms"].items()}}),
              flush=True)
        del ref
        mx.clear_cache()

    Path(a.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
