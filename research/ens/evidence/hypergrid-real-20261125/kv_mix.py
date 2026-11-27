"""Encode two related papers, mix their hidden states (KV caches), inject into Qwen3-0.6B, ask 10
questions about how the papers' concepts relate. Greedy decoding, thinking off.

Arms (the question is always encoded live on top of the injected cache):
  none      no papers (what the model already knows)
  text      both papers as ordinary text in one prompt (reference: the model reads them jointly)
  parallel  each paper encoded ALONE (it never attends to the other), KV caches concatenated;
            paper 2's keys are re-rotated (RoPE) to sit after paper 1. Mixing happens only when
            the question attends over both.
  pooled    each paper's KV compressed to M slots (chunk means, keys re-rotated to slot positions),
            slots of paper 1 then paper 2 (about 10x compression)
  blended   slot j = average of paper-1 slot j and paper-2 slot j: one fused M-slot memory

Run under mlx-uag venv: python kv_mix.py --model ... --out kv_mix_answers.json
"""

import argparse
import json
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import KVCache

QUESTIONS = [
    "Both papers discuss autoencoders and proper orthogonal decomposition (POD). What does each paper say about how a linear autoencoder relates to POD?",
    "Which earlier study is cited by both papers for the role of nonlinear activation functions in autoencoders, and what flow did that study consider?",
    "Both papers cite Srinivasan et al. What did that work do, and why does each paper mention it?",
    "Paper 2 describes Wang et al. predicting POD coefficients with an LSTM. How does that relate to the CNN-LSTM reduced order model of Hasegawa et al. described in Paper 1?",
    "Paper 2 is about generalization of neural networks for fluid flows. Which findings reported in Paper 1 bear on generalization, and what limitations did Paper 1 point out?",
    "Who performed the first autoencoder application to fluid dynamics, what did they compare, and does Paper 2 refer to that work too?",
    "Why are CNN-based autoencoders popular for fluid data according to Paper 1, and in which other tasks do CNNs appear in Paper 2?",
    "Paper 1 asks open questions about autoencoder-based order reduction. How could the hierarchical autoencoder mentioned in Paper 2 relate to those questions?",
    "Both papers contrast linear methods with neural networks. Which linear tools does Paper 1 name, and what advantage of neural networks do both papers emphasize?",
    "What drag reductions did Tang et al. report in Paper 2, and how does the laminar-only limitation mentioned there echo a challenge raised in Paper 1?",
]

SYSTEM = "You are a careful research assistant. Answer only from the papers you were given."


def rope_shift(x, delta, theta, D):
    """Rotate RoPE-encoded keys by per-slot position deltas (non-traditional / rotate-half)."""
    half = D // 2
    freqs = theta ** (-mx.arange(0, half, dtype=mx.float32) * 2 / D)
    ang = mx.array(delta, dtype=mx.float32)[:, None] * freqs[None, :]          # [L, half]
    c, s = mx.cos(ang), mx.sin(ang)
    xf = x.astype(mx.float32)
    x1, x2 = xf[..., :half], xf[..., half:]
    return mx.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1).astype(x.dtype)


def prefill(model, ids):
    n = len(model.model.layers)
    cache = [KVCache() for _ in range(n)]
    model(mx.array(ids)[None], cache=cache)
    mx.eval([c.keys for c in cache])
    return [(c.keys[..., :c.offset, :], c.values[..., :c.offset, :]) for c in cache]


def make_cache(kv):
    out = []
    for k, v in kv:
        c = KVCache()
        c.state = (k, v)
        out.append(c)
    return out


def generate(model, tok, kv, suffix_ids, max_tokens, stop_ids):
    cache = make_cache(kv)
    logits = model(mx.array(suffix_ids)[None], cache=cache)[0, -1]
    out = []
    for _ in range(max_tokens):
        t = int(mx.argmax(logits).item())
        if t in stop_ids:
            break
        out.append(t)
        logits = model(mx.array([[t]]), cache=cache)[0, -1]
    return tok.decode(out)


def pool(kv, start, end, M, slot0, theta, D):
    """Chunk-mean the span [start, end) into M slots; re-rotate keys to positions slot0..slot0+M-1."""
    edges = np.linspace(start, end, M + 1).round().astype(int)
    centers = (edges[:-1] + edges[1:] - 1) / 2
    delta = (slot0 + np.arange(M)) - centers
    out = []
    for k, v in kv:
        ks = mx.stack([k[:, :, a:b].mean(axis=2) for a, b in zip(edges[:-1], edges[1:])], axis=2)
        vs = mx.stack([v[:, :, a:b].mean(axis=2) for a, b in zip(edges[:-1], edges[1:])], axis=2)
        out.append((rope_shift(ks, delta, theta, D), vs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--paper1", required=True)
    ap.add_argument("--paper2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slots", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=220)
    args = ap.parse_args()
    t0 = time.time()
    model, tok = load(args.model)
    cfg = model.args
    theta, D = float(cfg.rope_theta), int(cfg.head_dim)
    p1, p2 = open(args.paper1).read().strip(), open(args.paper2).read().strip()

    head = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\nBelow are two research papers.\n\n"
    head_none = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n"
    t1 = f"Paper 1:\n{p1}\n\n"
    t2 = f"Paper 2:\n{p2}\n\n"
    suffix = lambda q: f"Question: {q}\nAnswer in a few sentences.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    h, a, b = enc(head), enc(t1), enc(t2)
    stop = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    print(f"tokens: head {len(h)} paper1 {len(a)} paper2 {len(b)}", flush=True)

    # self-check: manual RoPE shift matches mx.fast.rope composition
    x = mx.random.normal((1, 2, 5, D))
    ref = mx.fast.rope(x, D, traditional=False, base=theta, scale=1.0, offset=10)
    mine = rope_shift(mx.fast.rope(x, D, traditional=False, base=theta, scale=1.0, offset=3),
                      [7] * 5, theta, D)
    err = mx.max(mx.abs(ref - mine)).item()
    print(f"rope self-check max err {err:.2e}", flush=True)
    assert err < 1e-3

    kv_text = prefill(model, h + a + b)
    kv_ha = prefill(model, h + a)
    kv_hb = prefill(model, h + b)
    H, A, B = len(h), len(a), len(b)
    # parallel: paper 2's span shifted by +A so it sits after paper 1
    shift = [A] * B
    kv_par = [(mx.concatenate([ka, rope_shift(kb[:, :, H:], shift, theta, D)], axis=2),
               mx.concatenate([va, vb[:, :, H:]], axis=2)) for (ka, va), (kb, vb) in zip(kv_ha, kv_hb)]
    M = args.slots
    pa = pool(kv_ha, H, H + A, M, H, theta, D)
    pb = pool(kv_hb, H, H + B, M, H + M, theta, D)
    kv_pool = [(mx.concatenate([ka[:, :, :H], kpa, kpb], axis=2), mx.concatenate([va[:, :, :H], vpa, vpb], axis=2))
               for (ka, va), (kpa, vpa), (kpb, vpb) in zip(kv_ha, pa, pb)]
    pb0 = pool(kv_hb, H, H + B, M, H, theta, D)   # same slot positions as paper 1's slots
    kv_blend = [(mx.concatenate([ka[:, :, :H], (kpa + kpb) / 2], axis=2),
                 mx.concatenate([va[:, :, :H], (vpa + vpb) / 2], axis=2))
                for (ka, va), (kpa, vpa), (kpb, vpb) in zip(kv_ha, pa, pb0)]
    kv_none = prefill(model, enc(head_none))
    arms = {"none": kv_none, "text": kv_text, "parallel": kv_par, "pooled": kv_pool, "blended": kv_blend}
    print("injected cache lengths:", {k: int(v[0][0].shape[2]) for k, v in arms.items()}, flush=True)

    res = {"model": args.model, "slots": M, "tokens": {"head": H, "paper1": A, "paper2": B}, "answers": []}
    for qi, q in enumerate(QUESTIONS):
        row = {"q": q}
        for name, kv in arms.items():
            row[name] = generate(model, tok, kv, enc(suffix(q)), args.max_tokens, stop)
        res["answers"].append(row)
        print(f"q{qi + 1} done {time.time() - t0:.0f}s", flush=True)
        json.dump(res, open(args.out, "w"), indent=1)
    res["total_seconds"] = time.time() - t0
    res["peak_gib"] = mx.get_peak_memory() / 2 ** 30
    json.dump(res, open(args.out, "w"), indent=1)
    print("done", res["total_seconds"], flush=True)


if __name__ == "__main__":
    main()
