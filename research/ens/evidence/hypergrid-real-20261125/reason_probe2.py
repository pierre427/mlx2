"""Thinking probe 2 (Qwen3, greedy, mlx-uag venv): run-on thinking controls, controls for "is it just
more compute", neuron ablations vs random, attention at the answer token, thinking selectivity.

Phase A  think generation with mlx2's ThinkingGuard (budget, run-on CUSUM alarm, ramped close bias).
Phase B  forced-answer readouts ("The answer is ") on prefixes built from each trace:
         full, empty think, true no-think, dots filler (same length), shuffled sentences,
         another problem's trace, trace cut at 25/50/75 %, and the full trace with MLP neurons
         zero-ablated (think-selective sel64/sel256, high-firing act64, random rnd64 x3 / rnd256).
Phase C  full-trace capture: attention of the answer-choice position (per layer, mean over heads)
         on prompt / trace / the trace positions holding the gold number; MLP selectivity during
         thinking (participation ratio, top-1 % sets, digit vs word tokens).
Phase D  thinking GENERATED with sel64 vs rnd64 ablated (first problems, as time allows).
Each phase is time-capped and saved when done.
"""

import argparse
import importlib.util
import json
import os
import re
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.activations import swiglu


def first_num(t):
    m = re.findall(r"-?\d[\d,]*(?:\.\d+)?", t)
    if not m:
        return None
    try:
        v = float(m[0].replace(",", ""))
        return int(v) if v == int(v) else v
    except ValueError:
        return None


def last_num(s):
    m = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    if not m:
        return None
    v = float(m[-1])
    return int(v) if v == int(v) else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gsm", required=True)
    ap.add_argument("--guard", required=True, help="path to mlx2 src/mlx2/thinking_guard.py")
    ap.add_argument("--sets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--budget", type=int, default=2400)
    ap.add_argument("--caps", default="540,360,180,240", help="seconds for phases A,B,C,D")
    ap.add_argument("--d-problems", type=int, default=20)
    args = ap.parse_args()
    if os.environ.get("PROBE_CPU"):
        mx.set_default_device(mx.cpu)
    capA, capB, capC, capD = [float(x) for x in args.caps.split(",")]
    spec = importlib.util.spec_from_file_location("thinking_guard", args.guard)
    tg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tg)
    t0 = time.time()
    model, tok = load(args.model)
    inner = model.model
    nL = len(inner.layers)
    I = inner.layers[0].mlp.gate_proj.weight.shape[0]
    att0 = inner.layers[0].self_attn
    H, KV, HD = att0.n_heads, att0.n_kv_heads, inner.args.head_dim if hasattr(inner, "args") else model.args.head_dim
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    stop = {im_end, tok.eos_token_id}
    think_end = tok.convert_tokens_to_ids("</think>")
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    F_T, F_N = enc("\n\nThe answer is "), enc("The answer is ")
    probs = [json.loads(l) for l in open(args.gsm)][:args.n]
    S = json.load(open(args.sets))

    def mask_for(name):
        if name is None:
            return None
        m = {}
        for l, idx in S["sets"][name].items():
            v = np.ones(I, np.float32)
            v[idx] = 0.0
            m[int(l)] = mx.array(v).astype(mx.bfloat16)
        return m

    def fwd(ids, cache, ablate=None, hook=None, want_all=False):
        x = mx.array(ids)[None]
        h = inner.embed_tokens(x)
        mask = create_attention_mask(h, cache[0])
        for li, layer in enumerate(inner.layers):
            xn = layer.input_layernorm(h)
            hh = h + layer.self_attn(xn, mask, cache[li])
            m_in = layer.post_attention_layernorm(hh)
            a = swiglu(layer.mlp.gate_proj(m_in), layer.mlp.up_proj(m_in))
            if ablate is not None and li in ablate:
                a = a * ablate[li]
            if hook is not None:
                hook(li, a[0], xn)
            h = hh + layer.mlp.down_proj(a)
        hn = inner.norm(h if want_all else h[:, -1:])
        lg = inner.embed_tokens.as_linear(hn) if model.args.tie_word_embeddings else model.lm_head(hn)
        return lg[0].astype(mx.float32)

    newcache = lambda: [KVCache() for _ in range(nL)]

    # self-check: manual forward == model()
    ids0 = enc("<|im_start|>user\nWhat is 12 plus 30?<|im_end|>\n<|im_start|>assistant\n")
    ref = model(mx.array(ids0)[None])[0, -1].astype(mx.float32)
    mine = fwd(ids0, newcache())[-1]
    err = mx.max(mx.abs(ref - mine)).item()
    print(f"self-check max |logit diff| {err:.3g}", flush=True)
    assert err <= 0.25 and int(mx.argmax(ref).item()) == int(mx.argmax(mine).item()), "manual forward disagrees"

    def prompt(q, think):
        p = f"<|im_start|>user\n{q}\nGive only the final answer as a number.<|im_end|>\n<|im_start|>assistant\n<think>\n"
        return enc(p if think else p + "\n</think>\n\n")

    def gen_think(P, ablate=None, hard=None):
        guard = tg.ThinkingGuard(len(P), [think_end], budget=args.budget)
        cache = newcache()
        logits = fwd(P, cache, ablate)[-1]
        out = []
        hard = hard or args.budget + 200
        for _ in range(hard):
            lg = guard(mx.array(P + out), logits)
            t = int(mx.argmax(lg).item())
            out.append(t)
            if t == think_end or t in stop:
                break
            logits = fwd([t], cache, ablate)[-1]
        guard._observe(mx.array(P + out))
        info = {"trip_reason": guard.trip_reason, "forced": bool(guard.forced),
                "tripped_at": guard._tripped_at, "think_tokens": guard.think_tokens,
                "natural_close": (out[-1] == think_end) and guard._tripped_at is None}
        return out, info

    def readout(prefix, gold, ablate=None, n=12):
        cache = newcache()
        logits = fwd(prefix, cache, ablate)[-1]
        lp = logits - mx.logsumexp(logits)
        p = mx.exp(lp)
        ent = float(-(p * lp).sum().item())
        top1 = float(p.max().item())
        gid = enc(str(gold))[0] if gold is not None else None
        glp = float(lp[gid].item()) if gid is not None else None
        toks = []
        for _ in range(n):
            t = int(mx.argmax(logits).item())
            if t in stop:
                break
            toks.append(t)
            logits = fwd([t], cache, ablate)[-1]
        text = tok.decode(toks)
        pred = first_num(text)
        ok = pred is not None and gold is not None and abs(pred - gold) < 1e-6
        return {"pred": pred, "correct": ok, "entropy": ent, "top1": top1, "gold_first_lp": glp, "text": text[:40]}

    res = {"model": args.model, "budget": args.budget, "rows": [], "readouts": {}, "attention": [],
           "selectivity": [], "ablated_gen": []}
    save = lambda: json.dump(res, open(args.out + ".json", "w"))

    # ---------------- A: thinking with run-on controls
    ta = time.time()
    traces = []
    for i, pr in enumerate(probs):
        gold = last_num(pr["answer"].split("####")[-1])
        P = prompt(pr["question"], True)
        g, info = gen_think(P)
        T = g[:-1] if g and g[-1] in (think_end, im_end) else g
        traces.append({"i": i, "gold": gold, "P": P, "T": T, "q": pr["question"]})
        res["rows"].append({"i": i, "gold": gold, "think_len": len(T), **info})
        print(f"A {i} len {len(T)} trip {info['trip_reason']} natural {info['natural_close']} t={time.time() - t0:.0f}s", flush=True)
        if time.time() - ta > capA:
            print("phase A cap", flush=True)
            break
    save()

    # ---------------- B: readouts
    tb = time.time()
    ab_names = ["sel64", "rnd64_s0", "act64", "rnd64_s1", "rnd64_s2", "sel256", "rnd256_s0"]
    masks = {n: mask_for(n) for n in ab_names}
    dots = enc(" .")
    close = [think_end]
    for j, tr in enumerate(traces):
        P, T, gold = tr["P"], tr["T"], tr["gold"]
        other = traces[(j + 1) % len(traces)]["T"]
        txt = tok.decode(T)
        sents = re.split(r"(?<=[.!?\n])\s+", txt)
        rng = np.random.default_rng(j)
        shuf = enc(" ".join(rng.permutation(sents).tolist()))
        arms = {
            "full": P + T + close + F_T,
            "empty": P + close + F_T,
            "nothink": prompt(tr["q"], False) + F_N,
            "dots": P + (dots * max(1, len(T) // len(dots)))[:len(T)] + close + F_T,
            "shuffled": P + shuf + close + F_T,
            "other_trace": P + other + close + F_T,
            "cut25": P + T[:len(T) // 4] + close + F_T,
            "cut50": P + T[:len(T) // 2] + close + F_T,
            "cut75": P + T[:3 * len(T) // 4] + close + F_T,
        }
        out = {name: readout(ids, gold) for name, ids in arms.items()}
        for name in ab_names:
            out["abl_" + name] = readout(arms["full"], gold, masks[name])
        res["readouts"][str(tr["i"])] = out
        print(f"B {tr['i']} full {out['full']['correct']} sel64 {out['abl_sel64']['correct']} rnd64 {out['abl_rnd64_s0']['correct']} t={time.time() - t0:.0f}s", flush=True)
        if time.time() - tb > capB:
            print("phase B cap", flush=True)
            break
    save()

    # ---------------- C: attention + selectivity on the full trace
    tc = time.time()
    digit_tok = np.array([bool(re.fullmatch(r"\s*[\d+\-*/=×÷.,%$]+\s*", tok.decode([k])))
                          for k in range(int(model.args.vocab_size))])
    print("digit/operator tokens in vocab:", int(digit_tok.sum()), flush=True)
    for tr in traces:
        P, T, gold = tr["P"], tr["T"], tr["gold"]
        ids = P + T + [think_end] + F_T
        p = len(ids) - 1
        # gold-number positions in the trace (last occurrence of its digit tokens)
        gtok = enc(str(gold)) if gold is not None else []
        gpos = []
        for s in range(len(T) - len(gtok), -1, -1):
            if gtok and T[s:s + len(gtok)] == gtok:
                gpos = [len(P) + s + k for k in range(len(gtok))]
                break
        stats = {"prompt": [], "trace": [], "gold_pos": [], "force": [], "max_head_gold": []}
        sel = {"pr_think": [], "pr_prompt": [], "top1pct_think": [], "top1pct_prompt": []}
        dig = np.array([bool(digit_tok[t]) for t in T]) if digit_tok is not None else np.zeros(len(T), bool)
        dig_means, word_means = [], []
        cache = newcache()
        xs = {}

        def hook(li, a, xn):
            xs[li] = xn[:, -1:]
            A = mx.abs(a.astype(mx.float32))
            tseg = A[len(P):len(P) + len(T)]
            pseg = A[:len(P)]
            mt, mp = tseg.mean(0), pseg.mean(0)
            dm = tseg[mx.array(np.where(dig)[0])].mean(0) if dig.any() else mx.zeros(I)
            wm = tseg[mx.array(np.where(~dig)[0])].mean(0) if (~dig).any() else mx.zeros(I)
            mx.eval(mt, mp, dm, wm)
            mt_, mp_ = np.array(mt), np.array(mp)
            sel["pr_think"].append(float(mt_.sum() ** 2 / (I * (mt_ ** 2).sum())))
            sel["pr_prompt"].append(float(mp_.sum() ** 2 / (I * (mp_ ** 2).sum())))
            sel["top1pct_think"].append(np.argsort(-mt_)[:I // 100].tolist())
            sel["top1pct_prompt"].append(np.argsort(-mp_)[:I // 100].tolist())
            dig_means.append(np.array(dm).astype(np.float16))
            word_means.append(np.array(wm).astype(np.float16))
        fwd(ids, cache, None, hook)
        for li, layer in enumerate(inner.layers):
            att = layer.self_attn
            q = att.q_proj(xs[li]).reshape(1, 1, H, -1)
            q = att.q_norm(q).transpose(0, 2, 1, 3)
            q = att.rope(q, offset=p)
            k = cache[li].keys[..., :p + 1, :]
            k = mx.repeat(k, H // KV, axis=1)
            w = mx.softmax((q * att.scale) @ k.transpose(0, 1, 3, 2), axis=-1)[0, :, 0, :].astype(mx.float32)  # [H, p+1]
            mx.eval(w)
            wn = np.array(w)
            stats["prompt"].append(float(wn[:, :len(P)].sum(1).mean()))
            stats["trace"].append(float(wn[:, len(P):len(P) + len(T)].sum(1).mean()))
            stats["force"].append(float(wn[:, len(P) + len(T):].sum(1).mean()))
            g = wn[:, gpos].sum(1) if gpos else np.zeros(H)
            stats["gold_pos"].append(float(g.mean()))
            stats["max_head_gold"].append(float(g.max()))
        res["attention"].append({"i": tr["i"], "gold_in_trace": bool(gpos), **stats})
        res["selectivity"].append({"i": tr["i"], "pr_think": sel["pr_think"], "pr_prompt": sel["pr_prompt"],
                                   "top1pct_think": sel["top1pct_think"], "top1pct_prompt": sel["top1pct_prompt"],
                                   "n_digit": int(dig.sum()), "n_word": int((~dig).sum())})
        np.savez_compressed(f"{args.out}.dw{tr['i']}.npz", digit=np.stack(dig_means), word=np.stack(word_means))
        print(f"C {tr['i']} t={time.time() - t0:.0f}s", flush=True)
        if time.time() - tc > capC:
            print("phase C cap", flush=True)
            break
    save()

    # ---------------- D: thinking generated with neurons ablated
    td = time.time()
    for tr in traces[:args.d_problems]:
        row = {"i": tr["i"]}
        for name in ("sel64", "rnd64_s0"):
            g, info = gen_think(tr["P"], masks[name])
            T = g[:-1] if g and g[-1] in (think_end, im_end) else g
            r = readout(tr["P"] + T + [think_end] + F_T, tr["gold"], masks[name])
            row[name] = {"think_len": len(T), "trip": info["trip_reason"], **r}
        res["ablated_gen"].append(row)
        print(f"D {tr['i']} sel64 {row['sel64']['correct']} ({row['sel64']['think_len']}) rnd64 {row['rnd64_s0']['correct']} ({row['rnd64_s0']['think_len']}) t={time.time() - t0:.0f}s", flush=True)
        save()
        if time.time() - td > capD:
            print("phase D cap", flush=True)
            break
    res["total_seconds"] = time.time() - t0
    res["peak_gib"] = mx.get_peak_memory() / 2 ** 30
    save()
    print(f"done {res['total_seconds']:.0f}s peak {res['peak_gib']:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
