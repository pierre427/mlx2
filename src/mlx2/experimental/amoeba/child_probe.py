"""Two model-loading checks on probe windows (child process).

corecheck  mapping-workflow step 2.2: score held probes with the core alone against the full model, per
           language (non-English should be poor) and on held English (should stay near the full model).
gates      step 5: per box, a detector on the model's own early-layer hidden state (the residual stream
           entering layer --layer), trained on fit probes (the box's topic against everything else), with
           the offline check on held probes: false-open rate on unrelated content, miss rate on matching.

Windows follow the data rules: --window-tokens tokens at a seeded random offset across the document
(rule 1, 4), code documents cut at their first line of code (rule 3); held windows are only scored.

Detector law (the one src/mlx2/runtime/amoeba_tiles.py applies per token on a layer input): logit =
x . w + b on the raw hidden state x (the standardisation is folded into w and b). A window (a stretch of
the stream) opens the box when the mean per-token probability sigmoid(logit) reaches the box's threshold,
chosen on FIT windows only (false-open <= --fit-false-open on fit negatives, never below 0.5).

    python -m mlx2.experimental.amoeba.child_probe corecheck --model M --core NPZ --probes C --english E --out J
    python -m mlx2.experimental.amoeba.child_probe gates --model M --boxset DIR --english E --layer L --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .progress import child_event


def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().split("\n") if x.strip()]


def window(host, text: str, W: int, key: str, seed: int, code: bool, min_len: int = 32):
    from research.amoeba.carve.corpus import cut_at_first_code
    if code:
        text = cut_at_first_code(text)
    ids = host.encode(text)
    if len(ids) < min_len:
        return None
    if len(ids) <= W:
        return ids
    h = int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big")
    o = h % (len(ids) - W + 1)
    return ids[o:o + W]


def pick(rows, n, seed):
    rows = sorted(rows, key=lambda r: hashlib.sha256(f"{seed}:{r.get('probe_id') or r.get('doc_id') or r['text'][:64]}"
                                                     .encode()).hexdigest())
    return rows[:n] if n else rows


def is_code(row, prefixes):
    t = (row.get("topics") or [""])[0]
    return any(t.startswith(p) for p in prefixes)


def load_host(a):
    from research.amoeba.carve import ops as O
    from research.amoeba.carve.host import Host
    O.footprint_watchdog(a.watchdog_gb, print)
    O.setup_mlx(a.memory_gb, 2)
    import mlx.core as mx
    if a.cpu:
        mx.set_default_device(mx.cpu)
    log = lambda *m: print(time.strftime("%H:%M:%S"), *m, flush=True)  # noqa: E731
    host = Host(a.model, log=log, moe_units=a.moe_units)
    # Wire the evaluated weights like the core carve does (env AMOEBA_CARVE_WIRED_GB, default off): an unwired
    # resident model near the cap is compressed under pressure (Hy4-flash corecheck: compressor +23 GiB in
    # 12 s right after the carve process exited, 2026-11-21).
    spec = os.environ.get("AMOEBA_CARVE_WIRED_GB", "").strip() or "0"
    if not a.cpu and spec != "0":
        O.wire_weights(O.wired_arg(spec), lambda m: log(m), memory_gb=a.memory_gb)
    return host


# ------------------------------------------------------------------ corecheck

def corecheck(a):
    from research.amoeba.carve import evals as E
    from research.amoeba.carve.__main__ import load_mask
    host = load_host(a)
    core = load_mask(f"{a.core}:core", host)
    rows = [r for r in read_jsonl(a.probes) if r.get("split") == "held" and r.get("origin") != "synthetic"]
    by = {}
    for r in rows:
        t = r["topics"][0]
        if t.startswith("lang.") and t != "lang.eng":
            by.setdefault(t, []).append(r)
    groups = {t: pick(rs, a.per_language, a.seed) for t, rs in sorted(by.items()) if len(rs) >= a.min_rows}
    eng = [r for r in read_jsonl(a.english) if r.get("split") == "held"]
    groups["english:core-corpus"] = pick(eng, a.english_rows, a.seed)
    res = {}
    child_event("steps", steps=len(groups))
    for i, (g, rs) in enumerate(groups.items()):
        seqs = [s for s in (window(host, r["text"], a.window_tokens, r.get("probe_id") or r["text"][:64], a.seed, False)
                            for r in rs) if s is not None]
        if not seqs:
            continue
        full, n = E.text_ce(host, seqs)
        cut, _ = E.text_ce(host, seqs, E.configure(host, core=core))        # configure() returns the setter
        res[g] = {"windows": len(seqs), "tokens": n, "ce_full": round(full, 4), "ce_core": round(cut, 4),
                  "d_ce": round(cut - full, 4)}
        child_event("step", step=i, s_per_step=None)
        print(f"{g}: {res[g]}", flush=True)
    eng_d = (res.get("english:core-corpus") or {}).get("d_ce")
    non = [v["d_ce"] for k, v in res.items() if k.startswith("lang.")]
    rule = {"english_max_d_ce": a.english_max, "non_english_over_english": a.ratio,
            "note": "pass rule set by mlx2 amoeba (the workflow says 'poor' and 'near the full model' without numbers)"}
    ok = eng_d is not None and eng_d <= a.english_max and bool(non) and float(np.median(non)) >= a.ratio * max(eng_d, 1e-3)
    out = {"schema": "mlx2.amoeba-corecheck.v1", "checkpoint": host.checkpoint, "core": a.core, "groups": res,
           "english_d_ce": eng_d, "non_english_median_d_ce": float(np.median(non)) if non else None,
           "languages_scored": len(non), "rule": rule, "english_only": ok}
    Path(a.out).write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------ gates

class _Stop(Exception):
    pass


def layer_input(host, seqs, layer):
    """Residual stream entering ``layer`` for a batch of equal-length sequences (early exit after it)."""
    import mlx.core as mx
    lay = host.layers[layer]
    base = type(lay)
    box = {}

    def call(self_, x, *args, **kw):
        box["x"] = x
        mx.eval(x)
        raise _Stop
    lay.__class__ = type(f"Cap{base.__name__}", (base,), {"__call__": call})
    # The host's trunk forward: the inner module when callable, else the model's own ``hidden`` (Hy4-flash
    # keeps its trunk forward, Engram rows and per-layer caches on the outer model).
    trunk = host.hidden if hasattr(host, "hidden") else host.inner
    try:
        trunk(mx.array(seqs))
    except _Stop:
        pass
    finally:
        lay.__class__ = base
    x = box["x"]
    if x.ndim == 4:
        # hyper-connection streams [B, T, n, D] (Hy4-flash hc_mult = 4): the gate reads their mean
        x = mx.mean(x.astype(mx.float32), axis=2)
        layer_input.hc_streams = True
    return np.array(x.astype(mx.float32))


def features(host, seqs, layer, k, seed, bs=8):
    """[n_windows, k, d] hidden rows at k seeded positions >= 1 of each window."""
    out = [None] * len(seqs)
    by = {}
    for i, s in enumerate(seqs):
        by.setdefault(len(s), []).append(i)
    for _, idx in sorted(by.items()):
        for j in range(0, len(idx), bs):
            g = idx[j:j + bs]
            H = layer_input(host, [seqs[i] for i in g], layer)
            for r, i in enumerate(g):
                T = H.shape[1]
                rng = np.random.default_rng(seed + i)
                pos = np.sort(rng.choice(np.arange(1, T), size=min(k, T - 1), replace=False))
                out[i] = H[r, pos]
    return out


def train_logistic(Xp, Xn, l2=1e-3, iters=400, lr=0.05):
    X = np.concatenate([Xp, Xn]).astype(np.float64)
    y = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn))])
    wt = np.where(y > 0, 0.5 / len(Xp), 0.5 / len(Xn))
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    m = v = np.zeros(Z.shape[1] + 1)
    for t in range(1, iters + 1):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = np.concatenate([Z.T @ (wt * (p - y)) + l2 * w, [np.sum(wt * (p - y))]])
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        step = lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        w, b = w - step[:-1], b - step[-1]
    w_raw = w / sd
    return w_raw.astype(np.float32), float(b - np.sum(w * mu / sd))


def window_score(F, w, b):
    return np.array([float(np.mean(1 / (1 + np.exp(-(f @ w + b))))) for f in F])


def gates(a):
    host = load_host(a)
    layer = a.layer if a.layer >= 0 else max(1, host.L // 4)
    spec = json.loads((Path(a.boxset) / "boxes.json").read_text())
    names = [b for b in a.boxes.split(",") if b]
    if not names:
        raise SystemExit("gates: --boxes is empty (pass the boxes to gate explicitly)")
    prefixes = a.code_prefixes.split(",") if a.code_prefixes else []
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    feats = {}
    child_event("steps", steps=len(names) + 1 + bool(a.negative_probes))
    t0 = time.time()
    extra = ["__english__"] + (["__other__"] if a.negative_probes else [])
    for i, name in enumerate(names + extra):
        rows = {}
        if name == "__other__":           # off-topic probes (targeted extraction: one box, no other boxes)
            ex = a.exclude_topic
            exs = [n for n in ex.split("+") if n] if ex else []        # one id, or several joined by '+'
            pr = [r for r in read_jsonl(a.negative_probes) if r.get("origin") != "synthetic" and not
                  any(r["topics"][0] == n or r["topics"][0].startswith(n + ".") for n in exs)]
            rows = {"fit": pick([r for r in pr if r["split"] == "fit"], a.negatives, a.seed),
                    "held": pick([r for r in pr if r["split"] == "held"], a.negatives, a.seed)}
        elif name == "__english__":
            eng = read_jsonl(a.english)
            rows = {"fit": pick([r for r in eng if r.get("split") != "held"], a.negatives, a.seed),
                    "held": pick([r for r in eng if r.get("split") == "held"], a.negatives, a.seed)}
        else:
            bd = Path(a.boxset) / name
            rows = {"fit": pick(read_jsonl(bd / "docs.jsonl") + read_jsonl(bd / "val.jsonl"), a.windows, a.seed),
                    "held": pick(read_jsonl(bd / "held.jsonl"), a.windows, a.seed)}
        feats[name] = {}
        for sp, rs in rows.items():
            seqs = [s for s in (window(host, r["text"], a.window_tokens, r.get("probe_id") or r.get("doc_id") or r["text"][:64],
                                       a.seed, is_code(r, prefixes)) for r in rs) if s is not None]
            feats[name][sp] = features(host, seqs, layer, a.tokens, a.seed, bs=a.bs) if seqs else []
        child_event("step", step=i, s_per_step=round((time.time() - t0) / (i + 1), 2))
    gw, gb, res = {}, {}, {}
    for name in names:
        pos_fit, pos_held = feats[name]["fit"], feats[name]["held"]
        neg_fit = [f for o in feats if o != name for f in feats[o]["fit"]]
        neg_held = {o: feats[o]["held"] for o in feats if o != name}
        if len(pos_fit) < 4 or len(neg_fit) < 4:
            res[name] = {"status": "too_few_windows", "fit_windows": len(pos_fit)}
            continue
        w, b = train_logistic(np.concatenate(pos_fit), np.concatenate(neg_fit))
        sf_neg = window_score(neg_fit, w, b)
        thr = max(0.5, float(np.quantile(sf_neg, 1 - a.fit_false_open)) + 1e-6)
        sp = window_score(pos_held, w, b) if pos_held else np.array([])
        fo = {o: float(np.mean(window_score(F, w, b) >= thr)) for o, F in neg_held.items() if F}
        allneg = np.concatenate([window_score(F, w, b) for F in neg_held.values() if F]) if fo else np.array([])
        res[name] = {"status": "trained", "layer": layer, "threshold": round(thr, 4),
                     "fit_windows": len(pos_fit), "fit_negatives": len(neg_fit),
                     "fit_miss_rate": float(np.mean(window_score(pos_fit, w, b) < thr)),
                     "held_windows": len(pos_held), "held_negatives": int(allneg.size),
                     "held_miss_rate": float(np.mean(sp < thr)) if sp.size else None,
                     "held_false_open_rate": float(np.mean(allneg >= thr)) if allneg.size else None,
                     "held_false_open_english": fo.get("__english__"),
                     "held_false_open_off_topic": fo.get("__other__"),
                     "held_false_open_worst": max(fo.items(), key=lambda kv: kv[1]) if fo else None}
        gw[name], gb[name] = w, b
        res[name]["threshold_raw"] = thr
    np.savez_compressed(out / "gates.npz", **{f"{n}__w": gw[n] for n in gw}, **{f"{n}__b": np.float32(gb[n]) for n in gb})
    hc_streams = getattr(layer_input, "hc_streams", False)
    rep = {"schema": "mlx2.amoeba-gates.v1", "checkpoint": host.checkpoint, "fingerprint": host.fingerprint,
           "layer": layer, "feature": f"residual stream entering layer {layer} (raw; standardisation folded in"
                      f"{'; mean of the hyper-connection streams' if hc_streams else ''})",
           "law": "token p = sigmoid(x . w + b); a window opens when mean p >= threshold",
           "threshold_rule": f"chosen on fit windows: false-open <= {a.fit_false_open} on fit negatives, >= 0.5",
           "tokens_per_window": a.tokens, "window_tokens": a.window_tokens, "boxes": res}
    (out / "gates.json").write_text(json.dumps(rep, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("corecheck", "gates"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--english", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--window-tokens", type=int, default=128)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--moe-units", default="expert")
        p.add_argument("--memory-gb", type=float, default=56)
        p.add_argument("--watchdog-gb", type=float, default=80)
        p.add_argument("--cpu", action="store_true")
    c = sub.choices["corecheck"]
    c.add_argument("--core", required=True)
    c.add_argument("--probes", required=True)
    c.add_argument("--per-language", type=int, default=16)
    c.add_argument("--min-rows", type=int, default=4)
    c.add_argument("--english-rows", type=int, default=48)
    c.add_argument("--english-max", type=float, default=0.25)
    c.add_argument("--ratio", type=float, default=2.0)
    g = sub.choices["gates"]
    g.add_argument("--boxset", required=True)
    g.add_argument("--boxes", default="")
    g.add_argument("--layer", type=int, default=-1)
    g.add_argument("--bs", type=int, default=8)
    g.add_argument("--tokens", type=int, default=32)
    g.add_argument("--windows", type=int, default=64)
    g.add_argument("--negatives", type=int, default=512)
    g.add_argument("--fit-false-open", type=float, default=0.01)
    g.add_argument("--code-prefixes", default="prog.")
    g.add_argument("--negative-probes", default="", help="probe corpus: off-topic fit/held probes as negatives")
    g.add_argument("--exclude-topic", default="", help="with --negative-probes: drop this taxonomy id and below")
    a = ap.parse_args(argv)
    {"corecheck": corecheck, "gates": gates}[a.cmd](a)


if __name__ == "__main__":
    main()
