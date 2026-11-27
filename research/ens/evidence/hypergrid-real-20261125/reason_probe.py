"""Thinking vs no-thinking on GSM8K: logits and neuron firing (Qwen3, greedy). Run under mlx-uag venv.

Phase A: generate both modes for N problems. Same user message ("Give only the final answer as a
number."); the ONLY difference is the empty think block (no-think) vs free thinking (think).
Phase B: teacher-forced re-run of every prompt+generation with capture (manual forward, checked
against model()):
  per position: next-token entropy, top-1 prob, top1-top2 margin; per layer: mean |MLP act|,
  residual norm, top-32 most active MLP neurons (act = silu(gate) * up)
  per problem x phase (prompt / think / answer): mean |act| vector per layer
  at the answer-choice position (first answer digit): full MLP act + residual per layer,
  and a logit lens (final norm + LM head on each layer's residual) for the correct first token.
"""

import argparse
import json
import re
import time

import os

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache


def num(s):
    s = s.replace(",", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        v = float(m[-1])
        return int(v) if v == int(v) else v
    except ValueError:
        return None


def gen(model, ids, max_tokens, stop):
    n = len(model.model.layers)
    cache = [KVCache() for _ in range(n)]
    logits = model(mx.array(ids)[None], cache=cache)[0, -1]
    out = []
    for _ in range(max_tokens):
        t = int(mx.argmax(logits).item())
        out.append(t)
        if t in stop:
            break
        logits = model(mx.array([[t]]), cache=cache)[0, -1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gsm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--think-max", type=int, default=1800)
    ap.add_argument("--gen-seconds", type=float, default=480)
    ap.add_argument("--reuse", default=None, help="phaseA json with saved greedy thinking traces")
    args = ap.parse_args()
    if os.environ.get("PROBE_CPU"):
        mx.set_default_device(mx.cpu)
    t0 = time.time()
    model, tok = load(args.model)
    inner = model.model
    nL = len(inner.layers)
    stop = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    think_end = tok.convert_tokens_to_ids("</think>")
    probs = [json.loads(l) for l in open(args.gsm)][:args.n]

    def prompt(q, think):
        p = f"<|im_start|>user\n{q}\nGive only the final answer as a number.<|im_end|>\n<|im_start|>assistant\n<think>\n"
        return tok.encode(p if think else p + "\n</think>\n\n", add_special_tokens=False)

    # ---------------- phase A: both modes must commit right after a forced "The answer is "
    reuse = {}
    if args.reuse:
        for x in json.load(open(args.reuse)):
            reuse[x["i"]] = x["think"]["gen"]
    FORCE_T = tok.encode("\n\nThe answer is ", add_special_tokens=False)
    FORCE_N = tok.encode("The answer is ", add_special_tokens=False)

    def first_num(t):
        m = re.findall(r"-?\d[\d,]*(?:\.\d+)?", t)
        if not m:
            return None
        try:
            v = float(m[0].replace(",", ""))
            return int(v) if v == int(v) else v
        except ValueError:
            return None
    rows = []
    for i, pr in enumerate(probs):
        gold = num(pr["answer"].split("####")[-1])
        r = {"i": i, "gold": gold}
        P = prompt(pr["question"], False)
        g = gen(model, P + FORCE_N, 12, stop)
        r["nothink"] = {"prompt": P, "pre": FORCE_N, "n_think": 0, "gen": g, "closed": True}
        P = prompt(pr["question"], True)
        th = reuse.get(i) or gen(model, P, args.think_max, stop)
        closed = think_end in th
        th = th[:th.index(think_end) + 1] if closed else th + [think_end]   # budget-forced close
        pre = th + FORCE_T
        g = gen(model, P + pre, 12, stop)
        r["think"] = {"prompt": P, "pre": pre, "n_think": len(th), "gen": g, "closed": closed}
        for mode in ("nothink", "think"):
            text = tok.decode(r[mode]["gen"])
            r[mode]["text"] = text
            r[mode]["pred"] = first_num(text)
            r[mode]["correct"] = r[mode]["pred"] is not None and gold is not None and abs(r[mode]["pred"] - gold) < 1e-6
        rows.append(r)
        print(f"A {i} gold {gold} nothink {r['nothink']['pred']} think {r['think']['pred']} closed {closed} "
              f"thinklen {len(th)} t={time.time() - t0:.0f}s", flush=True)
        if time.time() - t0 > args.gen_seconds:
            print("generation time cap reached", flush=True)
            break

    json.dump([{"i": r["i"], "gold": r["gold"], **{m: r[m] for m in ("nothink", "think")}} for r in rows],
              open(args.out + ".phaseA.json", "w"))
    print("phase A saved", flush=True)
    # ---------------- phase B
    gold_first = {}

    def forward_capture(ids):
        x = mx.array(ids)[None]
        h = inner.embed_tokens(x)
        mask = create_attention_mask(h, None)
        acts, resid = [], []
        for layer in inner.layers:
            r_ = layer.self_attn(layer.input_layernorm(h), mask, None)
            hh = h + r_
            m_in = layer.post_attention_layernorm(hh)
            a = mx.sigmoid(layer.mlp.gate_proj(m_in)) * layer.mlp.gate_proj(m_in) * layer.mlp.up_proj(m_in)
            h = hh + layer.mlp.down_proj(a)
            acts.append(a[0])
            resid.append(h[0])
        hn = inner.norm(h)
        logits = (inner.embed_tokens.as_linear(hn) if model.args.tie_word_embeddings else model.lm_head(hn))[0]
        return acts, resid, logits.astype(mx.float32)

    # self-check against model()
    ids0 = rows[0]["nothink"]["prompt"] + rows[0]["nothink"]["gen"]
    _, _, lg = forward_capture(ids0)
    ref = model(mx.array(ids0)[None])[0].astype(mx.float32)
    err = mx.max(mx.abs(ref - lg)).item()
    print(f"capture self-check max |logit diff| {err:.3g}", flush=True)
    assert err < 0.05

    def lens(res_vec):
        hn = inner.norm(res_vec)
        lg = (inner.embed_tokens.as_linear(hn) if model.args.tie_word_embeddings else model.lm_head(hn)).astype(mx.float32)
        return lg - mx.logsumexp(lg, axis=-1, keepdims=True)

    rec = {"pos": [], "key": []}
    big_mean, big_rn = [], []
    phase_means = []   # (problem, mode, phase) -> [nL, I]
    meta = []
    for r in rows:
        gid = tok.encode(str(r["gold"]), add_special_tokens=False)[0] if r["gold"] is not None else None
        for mode in ("nothink", "think"):
            P = r[mode]["prompt"]
            pre, nth = r[mode]["pre"], r[mode]["n_think"]
            G = pre + r[mode]["gen"]
            ids = P + G
            acts, resid, logits = forward_capture(ids)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            pr_ = mx.exp(lp)
            ent = -(pr_ * lp).sum(-1)
            top2 = mx.sort(mx.topk(lp, 2, axis=-1), axis=-1)[:, ::-1]
            npos = len(P)
            span = list(range(npos - 1, len(ids) - 1))
            phase = ["think" if k < nth else "forced" if k < len(pre) else "answer" for k in range(len(G))]
            ans_k = len(pre)
            for k in range(len(pre), len(G)):
                if re.search(r"\d", tok.decode([G[k]])):
                    ans_k = k
                    break
            A = mx.stack([mx.abs(a) for a in acts])       # [nL, T, I]
            mean_abs = A.mean(-1)                           # [nL, T]
            rn = mx.stack([mx.linalg.norm(x_.astype(mx.float32), axis=-1) for x_ in resid])
            mx.eval(ent, top2, mean_abs, rn)
            sp = np.array(span)
            ent_n, top2_n = np.array(ent.astype(mx.float32))[sp], np.array(top2.astype(mx.float32))[sp]
            rec["pos"].append({
                "problem": r["i"], "mode": mode, "correct": r[mode]["correct"],
                "phase": phase, "ans_k": ans_k,
                "entropy": ent_n.astype(np.float32).round(4).tolist(),
                "lp_top2": top2_n.astype(np.float32).round(4).tolist(),
                "offset": int(sum(len(x) for x in big_mean)),
            })
            big_mean.append(np.array(mean_abs.astype(mx.float32))[:, sp].T.astype(np.float16))
            big_rn.append(np.array(rn.astype(mx.float32))[:, sp].T.astype(np.float16))
            # phase means: prompt tokens, think tokens, answer tokens (positions predicting them)
            for ph in ("prompt", "think", "answer"):
                if ph == "prompt":
                    sel = list(range(0, npos - 1))
                else:
                    sel = [npos - 1 + k for k in range(len(G)) if phase[k] == ph]
                if sel:
                    v = A[:, mx.array(sel)].mean(1)
                    phase_means.append(np.array(v.astype(mx.float32)).astype(np.float16))
                    meta.append({"problem": r["i"], "mode": mode, "phase": ph, "n": len(sel),
                                 "correct": r[mode]["correct"]})
            # key position
            if ans_k is not None:
                p = npos - 1 + ans_k
                full_act = mx.stack([a[p] for a in acts])
                full_res = mx.stack([x_[p] for x_ in resid])
                lensrow = mx.stack([lens(x_[p]) for x_ in resid])     # [nL, V]
                emitted = G[ans_k]
                lr = {}
                for name, t_ in (("gold_first", gid), ("emitted", emitted)):
                    if t_ is None:
                        continue
                    col = lensrow[:, t_]
                    rank = (lensrow > col[:, None]).sum(-1)
                    mx.eval(col, rank)
                    lr[name] = {"token": int(t_), "logprob": np.array(col.astype(mx.float32)).tolist(), "rank": np.array(rank).tolist()}
                mx.eval(full_act, full_res)
                rec["key"].append({"problem": r["i"], "mode": mode, "correct": r[mode]["correct"],
                                   "lens": lr, "act": np.array(full_act.astype(mx.float32)).astype(np.float16),
                                   "res": np.array(full_res.astype(mx.float32)).astype(np.float16)})
        print(f"B {r['i']} t={time.time() - t0:.0f}s", flush=True)

    np.savez_compressed(args.out + ".key.npz",
                        act=np.stack([k["act"] for k in rec["key"]]), res=np.stack([k["res"] for k in rec["key"]]),
                        phase_means=np.stack(phase_means),
                        pos_mean_abs=np.concatenate(big_mean), pos_resid_norm=np.concatenate(big_rn))
    json.dump({"model": args.model, "rows": [{"i": r["i"], "gold": r["gold"],
                                              **{m: {k: v for k, v in r[m].items() if k != "prompt"} for m in ("nothink", "think")}}
                                             for r in rows],
               "pos": rec["pos"], "key_meta": [{k: v for k, v in kk.items() if k not in ("act", "res")} for kk in rec["key"]],
               "phase_meta": meta, "total_seconds": time.time() - t0},
              open(args.out + ".json", "w"))
    print(f"done {time.time() - t0:.0f}s peak {mx.get_peak_memory() / 2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
