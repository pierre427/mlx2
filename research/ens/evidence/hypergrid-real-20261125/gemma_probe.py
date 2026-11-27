"""Thinking probe on a second family: Gemma 4 (dense GeGLU MLP + 128-expert MoE per layer).

Same questions as reason_probe2.py, adapted:
  * hooks instead of a hand-written forward (Gemma 4 shares KV across layers, mixes sliding/global
    attention): each decoder layer's dense `mlp` and `router` are wrapped in place, so model() runs
    its own code; the wrappers record activations / expert choices and apply ablations.
  * thinking guard = budget only (tau disabled), so the run-on alarm does not cut traces.
  * neuron/expert sets are SELECTED on even problems and ABLATED on odd problems (held out).
  * no attention-head analysis (shared-KV attention is not re-derived here).
Phases: A think generation, S selection capture, B readouts (controls all problems, ablations on
odd problems), C selectivity capture. Time-capped, saved per phase.
"""

import argparse
import importlib.util
import itertools
import json
import os
import re
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.gemma4_text import geglu
from mlx_lm.models.cache import make_prompt_cache


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
    v = float(m[-1])
    return int(v) if v == int(v) else v


class Rec:
    on = False
    ranges = {}          # name -> (start, end) positions in the current forward
    dense = {}           # (name, layer) -> sum |a| vector, count
    experts = {}         # (name, layer) -> counts[E]
    dense_ablate = {}    # layer -> mx mask
    expert_ablate = {}   # layer -> mx bool [E]


class HookMLP(nn.Module):
    def __init__(self, inner, li):
        super().__init__()
        self.inner = inner
        self.li = li

    def __call__(self, x):
        a = geglu(self.inner.gate_proj(x), self.inner.up_proj(x))
        m = Rec.dense_ablate.get(self.li)
        if m is not None:
            a = a * m
        if Rec.on and x.shape[1] > 1:
            A = mx.abs(a[0].astype(mx.float32))
            for name, (s, e) in Rec.ranges.items():
                if e > s:
                    v = A[s:e].sum(0)
                    mx.eval(v)
                    k = (name, self.li)
                    old = Rec.dense.get(k)
                    Rec.dense[k] = (np.array(v) + (old[0] if old else 0), (old[1] if old else 0) + (e - s))
        return self.inner.down_proj(a)


class HookRouter(nn.Module):
    def __init__(self, inner, li):
        super().__init__()
        self.inner = inner
        self.li = li

    def __call__(self, x):
        idx, w = self.inner(x)
        ab = Rec.expert_ablate.get(self.li)
        if ab is not None:
            w = mx.where(ab[idx], mx.zeros_like(w), w)
        if Rec.on and x.shape[1] > 1:
            ii = np.array(idx[0])
            E = self.inner.proj.weight.shape[0]
            for name, (s, e) in Rec.ranges.items():
                if e > s:
                    c = np.bincount(ii[s:e].ravel(), minlength=E)
                    k = (name, self.li)
                    Rec.experts[k] = Rec.experts.get(k, 0) + c
        return idx, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gsm", required=True)
    ap.add_argument("--guard", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--budget", type=int, default=1600)
    ap.add_argument("--caps", default="540,60,420,120")
    ap.add_argument("--reuse", default=None, help="<out>.traces.json from an earlier run: skip phase A")
    ap.add_argument("--nothink-control", default=None,
                    help="<run>.json whose sets_info.sets to reuse: only no-think readouts, unablated and per set, then exit")
    args = ap.parse_args()
    if os.environ.get("PROBE_CPU"):
        mx.set_default_device(mx.cpu)
    capA, capS, capB, capC = [float(x) for x in args.caps.split(",")]
    spec = importlib.util.spec_from_file_location("thinking_guard", args.guard)
    tg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tg)
    t0 = time.time()
    model, tok = load(args.model)
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    layers = model.layers
    nL = len(layers)
    for li, layer in enumerate(layers):
        layer.mlp = HookMLP(layer.mlp, li)
        if getattr(layer, "enable_moe", False):
            layer.router = HookRouter(layer.router, li)
    I = layers[0].mlp.inner.gate_proj.weight.shape[0]
    E = layers[0].router.inner.proj.weight.shape[0] if getattr(layers[0], "enable_moe", False) else 0
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    close_id = tok.convert_tokens_to_ids("<channel|>")
    open_ids = enc("<|channel>thought\n")
    stop = set(tok.eos_token_ids) if hasattr(tok, "eos_token_ids") else {tok.eos_token_id}
    stop |= {1, 106, 50}
    F = enc("The answer is ")
    print(f"layers {nL} dense I {I} experts {E} close {close_id} open {open_ids} stop {sorted(stop)}", flush=True)
    probs = [json.loads(l) for l in open(args.gsm)][:args.n]

    def prompt(q, think):
        msgs = [{"role": "user", "content": f"{q}\nGive only the final answer as a number."}]
        s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=think)
        return enc(s)

    def step_logits(ids, cache):
        return model(mx.array(ids)[None], cache=cache)[0, -1].astype(mx.float32)

    def gen_think(P):
        guard = tg.ThinkingGuard(len(P), [close_id], budget=args.budget, tau=1e9)
        cache = make_prompt_cache(model)
        logits = step_logits(P, cache)
        out = []
        for _ in range(args.budget + 200):
            t = int(mx.argmax(guard(mx.array(P + out), logits)).item())
            out.append(t)
            if t == close_id or t in stop:
                break
            logits = step_logits([t], cache)
        guard._observe(mx.array(P + out))
        return out, {"forced": bool(guard.forced), "trip_reason": guard.trip_reason,
                     "closed_naturally": out[-1] == close_id and not guard.forced}

    def readout(prefix, gold, n=12):
        cache = make_prompt_cache(model)
        logits = step_logits(prefix, cache)
        lp = logits - mx.logsumexp(logits)
        p = mx.exp(lp)
        ent = float(-(p * lp).sum().item())
        top1 = float(p.max().item())
        toks = []
        for _ in range(n):
            t = int(mx.argmax(logits).item())
            if t in stop:
                break
            toks.append(t)
            logits = step_logits([t], cache)
        text = tok.decode(toks)
        pred = first_num(text)
        return {"pred": pred, "correct": pred is not None and abs(pred - gold) < 1e-6,
                "entropy": ent, "top1": top1, "text": text[:40], "ids": toks}

    if args.nothink_control:
        # Control: does ablating a set hurt answering WITHOUT any thinking? Sets come from an earlier run
        # (chosen there on even problems' thinking); odd problems are the held-out ones.
        src = json.load(open(args.nothink_control))
        S = {k: {int(l): np.array(v) for l, v in d.items()} for k, d in src["sets_info"]["sets"].items()}
        out_all = {"model": args.model, "sets_from": args.nothink_control, "readouts": {}, "rows": []}
        for i, pr in enumerate(probs):
            gold = last_num(pr["answer"].split("####")[-1])
            nt = prompt(pr["question"], False) + F
            Rec.dense_ablate, Rec.expert_ablate = {}, {}
            o = {"nothink": readout(nt, gold)}
            for name, d in S.items():
                Rec.dense_ablate, Rec.expert_ablate = {}, {}
                for l, idx in d.items():
                    if name.startswith("e"):
                        v = np.zeros(E, bool); v[idx] = True
                        Rec.expert_ablate[l] = mx.array(v)
                    else:
                        v = np.ones(I, np.float32); v[idx] = 0
                        Rec.dense_ablate[l] = mx.array(v).astype(mx.bfloat16)
                o["ntabl_" + name] = readout(nt, gold)
            Rec.dense_ablate, Rec.expert_ablate = {}, {}
            for v in o.values():
                v.pop("ids", None)
            out_all["readouts"][str(i)] = o
            out_all["rows"].append({"i": i, "gold": gold})
            json.dump(out_all, open(args.out + ".json", "w"))
            print(f"N {i} nothink {o['nothink']['correct']} act64 {o['ntabl_act64']['correct']} t={time.time() - t0:.0f}s", flush=True)
        out_all["total_seconds"] = time.time() - t0
        json.dump(out_all, open(args.out + ".json", "w"))
        return

    res = {"model": args.model, "budget": args.budget, "rows": [], "readouts": {}, "selectivity": {}}
    save = lambda: json.dump(res, open(args.out + ".json", "w"))

    # -------- A
    ta = time.time()
    traces = []
    if args.reuse:
        old = json.load(open(args.reuse))
        traces, res["rows"] = old["traces"], old["rows"]
        print(f"reused {len(traces)} traces from {args.reuse}", flush=True)
    for i, pr in enumerate(probs if not args.reuse else []):
        gold = last_num(pr["answer"].split("####")[-1])
        P = prompt(pr["question"], True)
        g, info = gen_think(P)
        body = g[:-1] if g and (g[-1] == close_id or g[-1] in stop) else g
        if body[:len(open_ids)] == open_ids:
            body = body[len(open_ids):]
        traces.append({"i": i, "gold": gold, "P": P, "B": body, "q": pr["question"]})
        res["rows"].append({"i": i, "gold": gold, "think_len": len(body), **info})
        print(f"A {i} len {len(body)} forced {info['forced']} t={time.time() - t0:.0f}s", flush=True)
        if time.time() - ta > capA:
            print("phase A cap", flush=True)
            break
    save()
    json.dump({"traces": traces, "rows": res["rows"]}, open(args.out + ".traces.json", "w"))

    def full_ids(tr, body=None):
        return tr["P"] + open_ids + (tr["B"] if body is None else body) + [close_id] + F

    # -------- S: select sets on even problems
    ts = time.time()
    Rec.dense, Rec.experts = {}, {}
    for tr in traces:
        if tr["i"] % 2:
            continue
        ids = full_ids(tr)
        s = len(tr["P"]) + len(open_ids)
        Rec.ranges = {"think": (s, s + len(tr["B"]))}
        Rec.on = True
        model(mx.array(ids)[None])
        Rec.on = False
        nt = prompt(tr["q"], False) + F
        r = readout(nt, tr["gold"])
        ids2 = nt + r["ids"]
        Rec.ranges = {"answer": (len(nt) - 1, len(ids2))}
        Rec.on = True
        model(mx.array(ids2)[None])
        Rec.on = False
        if time.time() - ts > capS:
            break
    rng = np.random.default_rng(7)
    lay = list(range(int(nL * 0.4), nL))
    sets = {}
    Tm = {l: Rec.dense[("think", l)][0] / Rec.dense[("think", l)][1] for l in lay}
    Am = {l: Rec.dense[("answer", l)][0] / Rec.dense[("answer", l)][1] for l in lay}
    sets["sel64"] = {l: np.argsort(-np.log((Tm[l] + 1e-3) / (Am[l] + 1e-3)))[:64] for l in lay}
    sets["act64"] = {l: np.argsort(-Tm[l])[:64] for l in lay}
    for sd in range(2):
        sets[f"rnd64_s{sd}"] = {l: rng.choice(np.setdiff1d(np.arange(I), sets["sel64"][l]), 64, replace=False) for l in lay}
    esets = {}
    if E:
        Te = {l: Rec.experts[("think", l)] / Rec.experts[("think", l)].sum() for l in lay}
        Ae = {l: Rec.experts[("answer", l)] / Rec.experts[("answer", l)].sum() for l in lay}
        esets["esel8"] = {l: np.argsort(-np.log((Te[l] + 1e-4) / (Ae[l] + 1e-4)))[:8] for l in lay}
        esets["eact8"] = {l: np.argsort(-Te[l])[:8] for l in lay}
        esets["ernd8"] = {l: rng.choice(np.setdiff1d(np.arange(E), esets["esel8"][l]), 8, replace=False) for l in lay}
    ov = np.mean([len(set(sets["sel64"][l]) & set(sets["act64"][l])) / 64 for l in lay])
    res["sets_info"] = {"layers_0idx": lay, "sel64_act64_overlap": float(ov),
                        "sets": {k: {str(l): v.tolist() for l, v in d.items()} for k, d in {**sets, **esets}.items()}}
    print(f"S done: sel64/act64 overlap {ov:.3f} t={time.time() - t0:.0f}s", flush=True)
    save()

    def set_ablation(name):
        Rec.dense_ablate, Rec.expert_ablate = {}, {}
        if name in sets:
            for l, idx in sets[name].items():
                v = np.ones(I, np.float32)
                v[idx] = 0
                Rec.dense_ablate[l] = mx.array(v).astype(mx.bfloat16)
        if name in esets:
            for l, idx in esets[name].items():
                v = np.zeros(E, bool)
                v[idx] = True
                Rec.expert_ablate[l] = mx.array(v)

    # -------- B: readouts
    tb = time.time()
    dots = enc(" .")
    for j, tr in enumerate(traces):
        B, gold = tr["B"], tr["gold"]
        other = traces[(j + 1) % len(traces)]["B"]
        sents = re.split(r"(?<=[.!?\n])\s+", tok.decode(B))
        shuf = enc(" ".join(np.random.default_rng(j).permutation(sents).tolist())) if B else []
        arms = {
            "full": full_ids(tr), "empty": full_ids(tr, []), "nothink": prompt(tr["q"], False) + F,
            "dots": full_ids(tr, (dots * max(1, len(B)))[:len(B)]), "shuffled": full_ids(tr, shuf),
            "other_trace": full_ids(tr, other), "cut25": full_ids(tr, B[:len(B) // 4]),
            "cut50": full_ids(tr, B[:len(B) // 2]), "cut75": full_ids(tr, B[:3 * len(B) // 4]),
        }
        out = {k: readout(v, gold) for k, v in arms.items()}
        if tr["i"] % 2:
            for name in list(sets) + list(esets):
                set_ablation(name)
                out["abl_" + name] = readout(arms["full"], gold)
                out["ntabl_" + name] = readout(arms["nothink"], gold)   # same ablation, no thinking at all
            set_ablation(None)
        for v in out.values():
            v.pop("ids", None)
        res["readouts"][str(tr["i"])] = out
        save()
        print(f"B {tr['i']} full {out['full']['correct']} nothink {out['nothink']['correct']} t={time.time() - t0:.0f}s", flush=True)
        if time.time() - tb > capB:
            print("phase B cap", flush=True)
            break
    save()

    # -------- C: selectivity on all problems (dense PR, top-1% overlap; expert usage)
    tc = time.time()
    per = []
    for tr in traces:
        Rec.dense, Rec.experts = {}, {}
        ids = full_ids(tr)
        s = len(tr["P"]) + len(open_ids)
        Rec.ranges = {"think": (s, s + len(tr["B"])), "prompt": (0, len(tr["P"]))}
        Rec.on = True
        model(mx.array(ids)[None])
        Rec.on = False
        row = {"i": tr["i"], "pr": {}, "top1pct": {}, "expert_entropy": {}, "expert_top8": {}}
        for ph in ("think", "prompt"):
            prs, tops, ents, et = [], [], [], []
            for l in range(nL):
                if (ph, l) not in Rec.dense:
                    continue
                v = Rec.dense[(ph, l)][0] / Rec.dense[(ph, l)][1]
                prs.append(float(v.sum() ** 2 / (len(v) * (v ** 2).sum())))
                tops.append(np.argsort(-v)[:max(1, len(v) // 100)].tolist())
                if E and (ph, l) in Rec.experts:
                    c = Rec.experts[(ph, l)] / Rec.experts[(ph, l)].sum()
                    ents.append(float(-(c[c > 0] * np.log(c[c > 0])).sum()))
                    et.append(np.argsort(-c)[:8].tolist())
            row["pr"][ph], row["top1pct"][ph], row["expert_entropy"][ph], row["expert_top8"][ph] = prs, tops, ents, et
        per.append(row)
        if time.time() - tc > capC:
            break
    res["selectivity"] = per
    res["total_seconds"] = time.time() - t0
    res["peak_gib"] = mx.get_peak_memory() / 2 ** 30
    save()
    print(f"done {res['total_seconds']:.0f}s peak {res['peak_gib']:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
