#!/usr/bin/env python3
"""PLE-style fact-table tile on Qwen3 (CPU): recogniser neurons, closed-form rows, scoring, controls.

Countries and the fresh/pool split are amoeba27b_lib's (seed 0), so categories line up with the 27B
capitals box: pool (103 countries the box was completed on) and fresh (40 never used to fit it).

Table rows: every pool and fresh country. The recogniser neurons, tau and margin are chosen on the
fit half of the pool only; fresh countries' rows are solved in closed form but never touch any
selection. "Not in table": 10 real countries outside the list, plus 20 pool countries scored on a
second table with their rows deleted (fallback must equal the host).

Usage: PYTHONPATH=$PWD/src:$PWD python scripts/amoeba_det_facts_cpu.py --out output/amoeba-det-tiles/facts-0.6b
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

mx.set_default_device(mx.cpu)

from mlx2.runtime import amoeba_det_tiles as D  # noqa: E402
from research.amoeba.donor import Donor  # noqa: E402

CORPUS = Path.home() / "Library/Application Support/mlx2/research-artifacts/amoeba/corpus-2026-11-13/corpus.json"
CHAT = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
# Rows are built from these wordings (the chat one answers with no leading space).
KEY_TEMPLATES = ["The capital of {} is", "{}'s capital city is",
                 CHAT.format(q="Tell me the capital city of {}."),
                 "What is the capital of {}? It is", "The capital city of {} is called"]
VAL_TEMPLATE = "The capital city of {} is"                                          # tau / margin (fit countries)
NEW_WORDINGS = ["The seat of government of {} is the city of", "Q: Which city is the capital of {}?\nA:",
                "Everyone knows the capital of {} is",
                CHAT.format(q="Name the capital city of {}. Answer with the city only.")]


def sp(template):
    """The answer's leading text: none after a chat assistant header, else a space."""
    return "" if template.endswith("</think>\n\n") else " "


NULL_RELATIONS = ["The official language of {} is", "The currency of {} is", "{} is a country in",
                  "The population of {} is about", "The flag of {} is"]
VAL_NULL = ["The main language spoken in {} is", "The national dish of {} is"]
EXTRA = [("Andorra", "Andorra la Vella"), ("Liechtenstein", "Vaduz"), ("Monaco", "Monaco"),
         ("San Marino", "San Marino"), ("Samoa", "Apia"), ("Tonga", "Nuku'alofa"), ("Belize", "Belmopan"),
         ("Suriname", "Paramaribo"), ("Guyana", "Georgetown"), ("Djibouti", "Djibouti")]
FRENCH = [("la France", "France", "Paris"), ("l'Allemagne", "Germany", "Berlin"), ("l'Italie", "Italy", "Rome"),
          ("l'Espagne", "Spain", "Madrid"), ("le Japon", "Japan", "Tokyo"), ("la Chine", "China", "Pékin"),
          ("la Russie", "Russia", "Moscou"), ("l'Égypte", "Egypt", "Le Caire"), ("le Canada", "Canada", "Ottawa"),
          ("l'Australie", "Australia", "Canberra"), ("le Kenya", "Kenya", "Nairobi"), ("la Grèce", "Greece", "Athènes"),
          ("le Portugal", "Portugal", "Lisbonne"), ("la Pologne", "Poland", "Varsovie"),
          ("l'Autriche", "Austria", "Vienne"), ("la Hongrie", "Hungary", "Budapest"),
          ("la Suède", "Sweden", "Stockholm"), ("la Norvège", "Norway", "Oslo"),
          ("la Finlande", "Finland", "Helsinki"), ("le Danemark", "Denmark", "Copenhague")]
GERMAN = [("Frankreich", "France", "Paris"), ("Deutschland", "Germany", "Berlin"), ("Italien", "Italy", "Rom"),
          ("Spanien", "Spain", "Madrid"), ("Japan", "Japan", "Tokio"), ("China", "China", "Peking"),
          ("Russland", "Russia", "Moskau"), ("Ägypten", "Egypt", "Kairo"), ("Kanada", "Canada", "Ottawa"),
          ("Australien", "Australia", "Canberra"), ("Kenia", "Kenya", "Nairobi"), ("Griechenland", "Greece", "Athen"),
          ("Portugal", "Portugal", "Lissabon"), ("Polen", "Poland", "Warschau"), ("Österreich", "Austria", "Wien"),
          ("Ungarn", "Hungary", "Budapest"), ("Schweden", "Sweden", "Stockholm"), ("Norwegen", "Norway", "Oslo"),
          ("Finnland", "Finland", "Helsinki"), ("Dänemark", "Denmark", "Kopenhagen")]
LANDMARKS = [("the Eiffel Tower", "France"), ("the Colosseum", "Italy"), ("the Brandenburg Gate", "Germany"),
             ("Mount Fuji", "Japan"), ("the Great Wall", "China"), ("the pyramids of Giza", "Egypt"),
             ("the Taj Mahal", "India"), ("Machu Picchu", "Peru"), ("the Acropolis", "Greece"),
             ("Niagara Falls", "Canada"), ("the Sydney Opera House", "Australia"), ("the Alhambra", "Spain"),
             ("the Christ the Redeemer statue", "Brazil"), ("Petra", "Jordan"), ("the Matterhorn", "Switzerland"),
             ("Chichen Itza", "Mexico"), ("the Little Mermaid statue", "Denmark"), ("Angkor Wat", "Cambodia"),
             ("the Blue Mosque", "Turkey"), ("Table Mountain", "South Africa")]


def git_rev():
    rev = subprocess.check_output(["git", "rev-parse", "--short=9", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src", "research", "scripts"], cwd=ROOT).returncode != 0
    return rev + ("-dirty" if dirty else "")


def split_countries(seed=0):
    import amoeba27b_lib as A
    countries = list(A.COUNTRIES)
    random.Random(seed + 7).shuffle(countries)
    fresh, pool = countries[:40], countries[40:]
    idx = list(range(len(pool)))
    random.Random(seed + len("capitals")).shuffle(idx)
    fit = [pool[i] for i in sorted(idx[:len(idx) // 2])]
    held = [pool[i] for i in sorted(idx[len(idx) // 2:])]
    return pool, fresh, fit, held


class Runner:
    def __init__(self, donor, tok):
        self.d, self.tok = donor, tok

    def enc(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    kind = "neurons"

    def neuron_acts(self, ids_list, L, positions="last"):
        """Recogniser features at layer L: MLP hidden activations (kind "neurons") or the MLP input
        itself (kind "residual_dims"); [n, F] at the last position, or a list of [T, F]."""
        out = []
        gw, uw = self.d.p(L, "mlp.gate_proj.weight"), self.d.p(L, "mlp.up_proj.weight")
        for ids in ids_list:
            _, caps, _ = self.d.forward(mx.array([ids]), record={L}, stop_after=L)
            x = caps[L][0][0]
            if self.kind == "residual_dims":
                z = x
            else:
                g = x @ gw.T
                z = (g * mx.sigmoid(g)) * (x @ uw.T)
            z = z.astype(mx.float32)
            out.append(np.array(z[-1] if positions == "last" else z))
        return np.stack(out) if positions == "last" else out

    def mlp_inputs(self, ids_list, L):
        res = []
        for ids in ids_list:
            _, caps, _ = self.d.forward(mx.array([ids]), record={L}, stop_after=L)
            res.append(caps[L][0][0])
        return res


import contextlib  # noqa: E402

_null = contextlib.nullcontext


def unit(a, axis=-1):
    return a / np.maximum(np.linalg.norm(a, axis=axis, keepdims=True), 1e-6)


def solve_value(runner, prompts, target_ids, W, *, target_lp=np.log(0.98), iters=4, max_norm=None):
    """Closed-form min-norm first-order step (repeated ``iters`` times) on a residual delta v at the
    MLP output of layer W, last position, so log p(target) reaches ``target_lp`` on every prompt
    (mean step over prompts). Returns v (np [d]) and the per-iteration log-probs."""
    d = runner.d
    v = mx.zeros((d.d,), dtype=mx.float32)
    hist = []
    for _ in range(iters):
        steps, lps = [], []
        tids = target_ids if isinstance(target_ids, (list, tuple)) else [target_ids] * len(prompts)
        for ids, target_id in zip(prompts, tids):
            arr = mx.array([ids])
            T = len(ids)

            def lp_of(vv):
                def hook(layer, x, y):
                    if layer != W:
                        return None
                    pad = mx.zeros((1, T - 1, d.d), dtype=y.dtype)
                    return y + mx.concatenate([pad, vv[None, None].astype(y.dtype)], axis=1)
                lg, _, _ = d.forward(arr, mlp_hook=hook)
                row = lg[0, -1].astype(mx.float32)
                return row[target_id] - mx.logsumexp(row)
            ctx = getattr(d, "grad_context", None)
            with (ctx() if ctx is not None else _null()):
                lp, g = mx.value_and_grad(lp_of)(v)
            lp, g = float(lp), np.array(g)
            lps.append(lp)
            gap = target_lp - lp
            if gap > 0:
                steps.append(gap * g / max(float(g @ g), 1e-9))
            else:
                steps.append(np.zeros_like(g))
        hist.append(lps)
        if not any(np.any(s) for s in steps):
            break
        v = v + mx.array(np.mean(steps, axis=0).astype(np.float32))
        if max_norm is not None:
            n = float(mx.linalg.norm(v))
            if n > max_norm:
                v = v * (max_norm / n)
    return np.array(v), hist


def values_reuse_mismatch(rm, rsrc, *, src_fp, host_sha, read_layer, write_layer, entities, value_templates, iters):
    """--values-from: what must match for an earlier table's value rows to be this run's. The rows are residual
    deltas solved by gradient steps through one model (source fingerprint + host tiles), at one write layer, on
    --value-templates wordings for --iters steps. Returns (mismatches {field: (recorded, current)}, unverified
    [fields the earlier manifest does not record])."""
    bad, unverified = {}, []

    def cmp(field, old, new):
        if old != new:
            bad[field] = (old, new)

    cmp("source.fingerprint", (rm.get("source") or {}).get("fingerprint"), src_fp)
    cmp("host_tiles_manifest_sha256", rm.get("host_tiles_manifest_sha256"), host_sha)
    cmp("write_layer", int(rm["tiles"][0]["write_layer"]), int(write_layer))
    cmp("read_layer", int(rm["tiles"][0]["read_layer"]), int(read_layer))
    if [r["entity"] for r in rsrc[:len(entities)]] != list(entities):
        bad["row_order"] = ("the earlier table's entities", "this run's")
    solver = rm.get("value_solver")
    if solver:
        cmp("value_templates", solver.get("value_templates"), value_templates)
        cmp("iters", solver.get("iters"), iters)
    else:  # manifests written before value_solver: iters is in each row's method text, the wordings count nowhere
        methods = {r.get("method") for r in rsrc[:len(entities)]}
        want = f"min-norm first-order step on log p(first answer token) x{iters}"
        if methods != {want}:
            bad["iters"] = (sorted(str(m) for m in methods), want)
        unverified.append("value_templates")
    return bad, unverified


def tf_exact(runner, ts, rows, *, hook_override=None, slot=None):
    """Teacher-forced exact match of the whole answer; also whether the tile opened at the last
    prompt position. rows: (prompt, answer_with_leading_space)."""
    d, ok, _opened, wrote = runner.d, 0, 0, []
    for prompt, ans in rows:
        p, a = runner.enc(prompt), runner.enc(ans)
        ids = p + a
        hook = hook_override if hook_override is not None else (ts.mlp_hook if ts is not None and ts.enabled else None)
        if slot is not None:
            slot["pos"] = len(p) - 1
        lg, _, _ = d.forward(mx.array([ids]), mlp_hook=hook)
        pred = mx.argmax(lg[0, len(p) - 1:len(ids) - 1], axis=-1).tolist()
        good = pred == a
        ok += good
        wrote.append({"prompt": prompt, "answer": ans, "pred_first": runner.tok.decode([pred[0]]), "ok": good})
    return {"acc": ok / max(1, len(rows)), "n": len(rows), "rows": wrote}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--read-layers", default="18,21,23,24,25")
    ap.add_argument("--write-layers", default="")
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--values-from", default=None, help="reuse solved value rows from an earlier table manifest")
    ap.add_argument("--value-templates", type=int, default=5, help="key wordings each row's value is solved on")
    ap.add_argument("--backend", choices=("donor", "27b"), default="donor")
    ap.add_argument("--host-tiles", default=None, help="27b: amoeba tile manifest the host serves (core+boxes)")
    ap.add_argument("--host-layout", default="fused")
    ap.add_argument("--model", default=None, help="27b: model path (default the 27B q8 build; a slice for smoke tests)")
    ap.add_argument("--device", choices=("gpu", "cpu"), default="gpu", help="27b backend device")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty {out}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.backend == "27b":
        from research.amoeba.det_backend27 import Backend27
        mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
        policy = ({"amoeba_tiles": args.host_tiles, "amoeba_tiles_layout": args.host_layout}
                  if args.host_tiles else None)
        donor = Backend27(path=args.model, policy=policy)
        src_fp, host_sha = donor.source_fingerprint, donor.host_tiles_sha256
    else:
        donor = Donor(args.repo)
        src_fp, host_sha = donor.artifact, None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(donor.path))
    R = Runner(donor, tok)
    pool, fresh, fit, held = split_countries()
    if args.smoke:
        pool, fresh, fit, held = pool[:12], fresh[:6], fit[:6], held[:6]
        pool = fit + held
    table_c = pool + fresh
    cap = {c: k for c, k in table_c + EXTRA}
    rep = {"schema": "mlx2.amoeba-det-facts-cpu.v1", "checkpoint": donor.artifact, "mlx2_source_revision": git_rev(),
           "device": args.device if args.backend == "27b" else "cpu",
           "dtype": "host-native" if args.backend == "27b" else "float32", "qualification": "unqualified", "args": vars(args),
           "counts": {"pool": len(pool), "fresh": len(fresh), "fit": len(fit), "held": len(held), "extra": len(EXTRA)}}

    # ---- recogniser neurons + read layer (fit countries only) -------------------------------------
    def fisher(L):
        names = [c for c, _ in fit]
        A = {}
        for j, t in enumerate(KEY_TEMPLATES + NULL_RELATIONS[:2]):
            A[j] = R.neuron_acts([R.enc(t.format(c)) for c in names], L)
        nk = len(KEY_TEMPLATES)
        cap_ = np.stack([A[j] for j in range(nk)])                 # [templates, n, F]
        nul = np.stack([A[nk], A[nk + 1]])
        mean_c = cap_.mean(0)
        within = cap_.var(0).mean(0) + 1e-6
        between = mean_c.var(0)
        rel = (cap_.mean((0, 1)) - nul.mean((0, 1))) ** 2 / (np.concatenate([cap_, nul]).var((0, 1)) + 1e-6)
        score = between / within * (1 + rel)
        return np.argsort(-score)[:args.k]

    def centre(L, neurons):
        z = R.neuron_acts([R.enc(t.format(c)) for c, _ in fit for t in KEY_TEMPLATES + NULL_RELATIONS[:2]], L)
        return z[:, neurons].mean(0)

    read_layers = [int(x) for x in args.read_layers.split(",")]
    best = None
    rep["read_layer_sweep"] = {}
    for kind, L in [(k, l) for k in ("neurons", "residual_dims") for l in read_layers]:
        R.kind = kind
        neurons = fisher(L)
        cen = centre(L, neurons)
        keys = {}
        for c, _ in fit:
            z = R.neuron_acts([R.enc(t.format(c)) for t in KEY_TEMPLATES], L)[:, neurons] - cen
            keys[c] = unit(unit(z).mean(0))
        K = np.stack([keys[c] for c, _ in fit])
        zv = unit(R.neuron_acts([R.enc(VAL_TEMPLATE.format(c)) for c, _ in fit], L)[:, neurons] - cen)
        sims = zv @ K.T
        top1 = float((sims.argmax(1) == np.arange(len(fit))).mean())
        zn = unit(R.neuron_acts([R.enc(VAL_NULL[0].format(c)) for c, _ in fit], L)[:, neurons] - cen)
        sep = float(np.median(sims.max(1)) - np.median((zn @ K.T).max(1)))
        rep["read_layer_sweep"][f"{kind}@{L}"] = {"val_top1": top1, "val_capital_vs_null_median_gap": sep}
        print("read", kind, L, top1, round(sep, 3), flush=True)
        if best is None or (top1, sep) > best[0]:
            best = ((top1, sep), L, neurons, cen, kind)
    _, L, neurons, cen, kind = best
    R.kind = kind
    rep["recogniser"] = kind
    rep["read_layer"] = L
    gate_rows = donor.p(L, "mlp.gate_proj.weight")[mx.array(neurons)]
    up_rows = donor.p(L, "mlp.up_proj.weight")[mx.array(neurons)]

    # ---- keys: one row per table country (capital) + null rows (same countries, other relations) ----
    rows_src, keys, null = [], [], []
    for c, k in table_c:
        z = R.neuron_acts([R.enc(t.format(c)) for t in KEY_TEMPLATES], L)[:, neurons] - cen
        keys.append(unit(unit(z).mean(0))); null.append(False)
        rows_src.append({"entity": c, "relation": "capital", "value": k, "key_templates": KEY_TEMPLATES,
                         "in_selection": (c, k) in fit, "source": "amoeba27b_lib.COUNTRIES"})
    for c, _ in table_c:
        z = R.neuron_acts([R.enc(t.format(c)) for t in NULL_RELATIONS], L)[:, neurons] - cen
        for j, t in enumerate(NULL_RELATIONS):
            keys.append(unit(z[j])); null.append(True)
            rows_src.append({"entity": c, "relation": t, "value": None, "null_row": True})
    K = np.stack(keys).astype(np.float32)
    nullv = np.array(null)
    n_cap = len(table_c)

    # ---- write layer + values (closed form per fact) ----------------------------------------------
    write_layers = [int(x) for x in args.write_layers.split(",")] if args.write_layers else sorted({L, min(donor.L - 1, L + 2), donor.L - 1})
    write_layers = sorted({w for w in write_layers if w >= L} | {L})            # the contract: read <= write
    rep["write_layer_sweep"] = {}
    probe_c = fit[:8]
    for W in write_layers:
        lps = []
        for c, k in probe_c:
            tid = R.enc(" " + k)[0]
            vt = KEY_TEMPLATES[:args.value_templates]
            v, hist = solve_value(R, [R.enc(t.format(c)) for t in vt], [R.enc(sp(t) + k)[0] for t in vt], W, iters=args.iters)
            # held-out wording check on the same fact
            def hook(layer, x, y, v=v, W=W):
                if layer != W:
                    return None
                T = y.shape[1]
                pad = mx.zeros((1, T - 1, y.shape[2]), dtype=y.dtype)
                return y + mx.concatenate([pad, mx.array(v)[None, None].astype(y.dtype)], axis=1)
            ids = R.enc(VAL_TEMPLATE.format(c))
            lg, _, _ = donor.forward(mx.array([ids]), mlp_hook=hook)
            row = lg[0, -1].astype(mx.float32)
            lps.append(float(row[tid] - mx.logsumexp(row)))
        rep["write_layer_sweep"][W] = {"val_wording_mean_logp": float(np.mean(lps)),
                                       "val_wording_top1_ok": int(np.sum(np.exp(lps) > 0.5))}
        print("write layer", W, rep["write_layer_sweep"][W], flush=True)
    Wl = max(write_layers, key=lambda w: (rep["write_layer_sweep"][w]["val_wording_mean_logp"], -w))
    rep["write_layer"] = Wl
    V = np.zeros((len(keys), donor.d), dtype=np.float32)
    reuse = None
    if args.values_from:
        rm = D.load_manifest(args.values_from)
        rt = D.load_tensors(rm)
        rsrc = [json.loads(l) for l in (Path(rm["_dir"]) / "rows.jsonl").read_text().splitlines()]
        bad, unverified = values_reuse_mismatch(
            rm, rsrc, src_fp=src_fp, host_sha=host_sha, read_layer=L, write_layer=Wl,
            entities=[c for c, _ in table_c], value_templates=args.value_templates, iters=args.iters)
        if bad:
            raise SystemExit("--values-from: the earlier table's rows are not this run's (recorded, current): "
                             + "; ".join(f"{k} {v[0]!r} != {v[1]!r}" for k, v in bad.items()))
        reuse = (np.array(rt["capitals_table.values"].astype(mx.float32)), rsrc, str(rm["_digest"]))
        rep["values_from"] = {"manifest": args.values_from, "manifest_sha256": reuse[2]}
        if unverified:
            print(f"WARNING --values-from: the earlier manifest does not record {unverified}; reused rows are "
                  "assumed to match this run's settings", flush=True)
            rep["values_from"]["unverified"] = unverified
    for i, (c, k) in enumerate(table_c):
        if reuse is not None:
            V[i] = reuse[0][i]
            rows_src[i].update({key: reuse[1][i][key] for key in ("method", "write_layer", "target_token", "target_logp",
                                                                  "logp_history", "value_norm") if key in reuse[1][i]})
            rows_src[i]["reused_from_manifest_sha256"] = reuse[2]
            continue
        tid = R.enc(" " + k)[0]
        vt = KEY_TEMPLATES[:args.value_templates]
        v, hist = solve_value(R, [R.enc(t.format(c)) for t in vt], [R.enc(sp(t) + k)[0] for t in vt], Wl, iters=args.iters)
        V[i] = v
        rows_src[i].update({"method": f"min-norm first-order step on log p(first answer token) x{args.iters}",
                            "write_layer": Wl, "target_token": tid, "target_logp": float(np.log(0.98)),
                            "logp_history": [float(np.mean(h)) for h in hist], "value_norm": float(np.linalg.norm(v))})
    print("values solved", f"{time.time() - t0:.0f}s", flush=True)

    # ---- tau / margin on val (fit countries only) --------------------------------------------------
    def keys_at(prompts, positions="last"):
        return [unit(z[..., neurons] - cen) for z in R.neuron_acts([R.enc(p) for p in prompts], L, positions)]

    zpos = np.stack(keys_at([VAL_TEMPLATE.format(c) for c, _ in fit]))
    want = np.array([table_c.index((c, k)) for c, k in fit])
    zneg = {"other_relation": np.stack(keys_at([t.format(c) for t in VAL_NULL for c, _ in fit]))}
    interior = keys_at([VAL_TEMPLATE.format(c) + " " + k for c, k in fit], positions="all")
    plen = [len(R.enc(VAL_TEMPLATE.format(c))) for c, _ in fit]
    # Every position but the answer slot (the prompt's last token) is a negative.
    zneg["interior_positions"] = np.concatenate([np.delete(z, p - 1, axis=0) for z, p in zip(interior, plen)])
    blob = json.loads(CORPUS.read_text())["seqs"]
    reenc = getattr(donor, "reencode", None) or (lambda ids: ids)
    eng_val = [reenc(ids) for dname in blob for ids in blob[dname][40:52]]
    zneg["english"] = np.concatenate([unit(z[:, neurons] - cen) for z in R.neuron_acts(eng_val, L, "all")])

    def decide(z, tau, margin):
        s = z @ K.T
        o = np.argsort(-s, axis=1)
        b1 = s[np.arange(len(s)), o[:, 0]]
        b2 = s[np.arange(len(s)), o[:, 1]]
        return (b1 >= tau) & (b1 - b2 >= margin) & ~nullv[o[:, 0]], o[:, 0]

    ent_rows = {}
    for i, r in enumerate(rows_src):
        ent_rows.setdefault(r["entity"], []).append(i)

    def decide_absent(z, own, tau, margin):
        """Open rate when each prompt's own row is absent (its key masked out): an absent entity."""
        s_ = z @ K.T
        for i, o in enumerate(own):
            s_[i, ent_rows[table_c[o][0]]] = -2.0
        o = np.argsort(-s_, axis=1)
        b1 = s_[np.arange(len(s_)), o[:, 0]]
        b2 = s_[np.arange(len(s_)), o[:, 1]]
        return (b1 >= tau) & (b1 - b2 >= margin) & ~nullv[o[:, 0]]

    grid = []
    for tau in [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        for margin in [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
            op, idx = decide(zpos, tau, margin)
            right = float((op & (idx == want)).mean())
            wrong = float((op & (idx != want)).mean())
            ff = {k: float(decide(v, tau, margin)[0].mean()) for k, v in zneg.items()}
            ff["absent_entity"] = float(decide_absent(zpos, want, tau, margin).mean())
            grid.append({"tau": tau, "margin": margin, "recall": right, "wrong_row": wrong, **{"ff_" + k: v for k, v in ff.items()}})
    ok = [g for g in grid if g["wrong_row"] <= 0.02 and all(g["ff_" + k] <= 0.01 for k in zneg)
          and g["ff_absent_entity"] <= 0.02]
    if not ok:
        raise SystemExit("no admissible tau/margin on val")
    pick = max(ok, key=lambda g: (g["recall"], g["tau"], g["margin"]))
    rep["calibration_val"] = {"pick": pick, "rule": "max val recall s.t. wrong-row <= 2%, false-fire <= 1% per negative group, absent-entity (own row masked) open <= 2%",
                              "grid": grid}
    print("pick", pick, flush=True)

    # ---- manifests: real, rows-deleted, shuffled values ------------------------------------------
    rng = np.random.default_rng(0)
    delete = [c for c, _ in rng.permutation(np.array(pool, dtype=object))[:20]] if not args.smoke else [pool[0][0]]
    tile = {"id": "capitals_table", "kind": "fact_table", "write_mode": "delta", "read_layer": int(L), "write_layer": int(Wl),
            "table": {"tau": pick["tau"], "margin": pick["margin"], "rows": len(keys), "value_rows": n_cap,
                      "recogniser": kind,
                      "neurons": int(len(neurons)), "relation": "capital_of"}}

    solver = {"value_templates": args.value_templates, "iters": args.iters}
    if reuse is not None and rep["values_from"].get("unverified"):
        solver = {**solver, "unverified_reused": rep["values_from"]["unverified"]}

    def save(name, Vx, Kx, nullx, rows, tile=tile):
        rec = ({"capitals_table.gate_rows": gate_rows, "capitals_table.up_rows": up_rows} if kind == "neurons"
               else {"capitals_table.dims": mx.array(neurons.astype(np.int32))})
        tens = {**rec,
                "capitals_table.keys": mx.array(Kx), "capitals_table.values": mx.array(Vx),
                "capitals_table.null": mx.array(nullx), "capitals_table.neurons": mx.array(neurons.astype(np.int32)),
                "capitals_table.center": mx.array(cen.astype(np.float32))}
        return D.write_manifest(out / name, source={"artifact": donor.artifact, "fingerprint": src_fp},
                                layers=donor.L, tiles=[tile], tensors=tens, rows_source=rows,
                                extra={"mlx2_source_revision": rep["mlx2_source_revision"], "host_tiles_manifest_sha256": host_sha,
                                       "value_solver": solver})

    man = save("table", V, K, nullv, rows_src)
    keep = [i for i, r in enumerate(rows_src) if r["entity"] not in delete]
    # its own row counts: the shared tile dict states the full table's
    tile_del = {**tile, "table": {**tile["table"], "rows": len(keep), "value_rows": int((~nullv[keep]).sum())}}
    man_del = save("table-rows-deleted", V[keep], K[keep], nullv[keep], [rows_src[i] for i in keep], tile=tile_del)
    perm = rng.permutation(n_cap)
    Vs = V.copy()
    Vs[:n_cap] = V[perm]
    man_shuf = save("table-shuffled-values", Vs, K, nullv,
                    [{**r, "control": "values permuted across capital rows"} for r in rows_src])
    act = lambda x: x * mx.sigmoid(x)  # noqa: E731
    TS = {name: D.build(m, act, source_fingerprint=src_fp, layers_count=donor.L)
          for name, m in (("table", man), ("deleted", man_del), ("shuffled", man_shuf))}

    # ---- scoring -------------------------------------------------------------------------------------
    def de_prompt(fr):
        return "La capitale " + ("du " + fr[3:] if fr.startswith("le ") else "de " + fr) + " est"

    import amoeba27b_lib as A
    allcap = dict(A.COUNTRIES)
    cats = {
        "trained_new_wording": [(t.format(c), sp(t) + k) for c, k in pool for t in NEW_WORDINGS],
        # In-sample for the row: the value was solved on this wording (the recogniser and tau were not).
        "trained_row_wording": [(KEY_TEMPLATES[0].format(c), " " + k) for c, k in pool],
        "unseen_countries_row_wording": [(KEY_TEMPLATES[0].format(c), " " + k) for c, k in fresh],
        "unseen_countries_new_wording": [(NEW_WORDINGS[0].format(c), " " + k) for c, k in fresh],
        "unseen_countries_chat": [(NEW_WORDINGS[3].format(c), k) for c, k in fresh],
        # the 20x20 sanity suite's chat wording (held out: no key or value used it)
        "unseen_countries_sanity_chat": [(CHAT.format(q=f"What is the capital of {c}? One word."), k) for c, k in fresh],
        "french": [(de_prompt(fr), " " + k) for fr, _, k in FRENCH],
        "german": [(f"Die Hauptstadt von {de} ist", " " + k) for de, _, k in GERMAN],
        "multi_step": [(f"The capital of the country with {lm} is", " " + allcap[c]) for lm, c in LANDMARKS if c in allcap],
        "not_in_table_extra": [(KEY_TEMPLATES[0].format(c), " " + k) for c, k in EXTRA],
        "other_relation": [(t.format(c), None) for c, _ in pool[:30] for t in VAL_NULL],
    }
    if args.smoke:
        cats = {k: v[:6] for k, v in cats.items()}
    delete_rows = [(t.format(c), " " + cap[c]) for c in delete for t in KEY_TEMPLATES[:1] + NEW_WORDINGS[:1]]

    def opened_at_last(ts, prompts):
        t = ts.tiles[0]
        res = []
        for p in prompts:
            x = R.mlp_inputs([R.enc(p)], L)[0]
            op, idx, best, second = t.table.lookup(x[-1:])
            res.append((bool(op[0].item()), int(idx[0].item())))
        return res

    rep["scores"] = {}
    for cname, rows in cats.items():
        if cname == "other_relation":
            o = opened_at_last(TS["table"], [p for p, _ in rows])
            rep["scores"][cname] = {"n": len(rows), "tile_open_rate": float(np.mean([a for a, _ in o]))}
            continue
        entry = {}
        TS["table"].enabled = False
        entry["host"] = tf_exact(R, None, rows)
        TS["table"].enabled = True
        entry["tile"] = tf_exact(R, TS["table"], rows)
        entry["shuffled_values"] = tf_exact(R, TS["shuffled"], rows)
        o = opened_at_last(TS["table"], [p for p, _ in rows])
        entry["tile_open_rate"] = float(np.mean([a for a, _ in o]))
        for k in ("host", "tile", "shuffled_values"):
            entry[k] = {**entry[k], "rows": entry[k]["rows"][:5]}
        rep["scores"][cname] = entry
        print(cname, {k: (entry[k]["acc"] if isinstance(entry[k], dict) else entry[k]) for k in entry}, flush=True)
    TS["deleted"].enabled = True
    rep["scores"]["not_in_table_deleted_rows"] = {
        "host": {k: v for k, v in tf_exact(R, None, delete_rows).items() if k != "rows"},
        "tile_rows_deleted": {k: v for k, v in tf_exact(R, TS["deleted"], delete_rows).items() if k != "rows"},
        "tile_full_table": {k: v for k, v in tf_exact(R, TS["table"], delete_rows).items() if k != "rows"},
        "tile_open_rate_rows_deleted": float(np.mean([a for a, _ in opened_at_last(TS["deleted"], [p for p, _ in delete_rows])])),
        "deleted": delete}
    print("deleted", rep["scores"]["not_in_table_deleted_rows"], flush=True)

    # Random trigger: at the answer slot, open a random capital row with the tile's own open rate
    # on that category (so it fires as often as the real tile, where it matters, with no key).
    rr = np.random.default_rng(1)
    Vm = mx.array(V[:n_cap])
    slot = {"pos": None, "rate": 0.0}

    def random_hook(layer, x, y):
        if layer != Wl or slot["pos"] is None or y.shape[1] <= slot["pos"]:
            return None
        if rr.random() >= slot["rate"]:
            return None
        row = Vm[int(rr.integers(0, n_cap))]
        delta = mx.zeros(y.shape, dtype=mx.float32).at[:, slot["pos"]].add(row)
        return (y.astype(mx.float32) + delta).astype(y.dtype)
    rep["scores"]["random_trigger"] = {}
    for cname in ("trained_new_wording", "unseen_countries_new_wording", "unseen_countries_row_wording"):
        slot["rate"] = rep["scores"][cname]["tile_open_rate"]
        res = tf_exact(R, None, cats[cname], hook_override=random_hook, slot=slot)
        rep["scores"]["random_trigger"][cname] = {"rate": slot["rate"], "acc": res["acc"], "n": res["n"]}
    slot["pos"] = None
    print("random", rep["scores"]["random_trigger"], flush=True)

    # English cost: mean next-token CE on test English, tile on vs off; tile open rate per position.
    eng = [reenc(ids) for dname in blob for ids in blob[dname][70:90 if not args.smoke else 72]]

    def ce(hook):
        tot, n = 0.0, 0
        for ids in eng:
            lg, _, _ = donor.forward(mx.array([ids[:-1]]), mlp_hook=hook)
            lp = lg[0].astype(mx.float32)
            lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
            tot += -float(mx.take_along_axis(lp, mx.array(ids[1:])[:, None], axis=-1).sum())
            n += len(ids) - 1
        return tot / n
    ce_host = ce(None)
    before = TS["table"].receipt()["counters"]
    ce_tile = ce(TS["table"].mlp_hook)
    after = TS["table"].receipt()["counters"]
    opened = after["fact_open_rows"] - before["fact_open_rows"]
    rows_ = after["fact_rows"] - before["fact_rows"]
    rep["english_cost"] = {"ce_host": ce_host, "ce_tile": ce_tile, "d_ce": ce_tile - ce_host,
                           "open_positions": opened, "positions": rows_, "open_rate": opened / max(1, rows_),
                           "sequences": len(eng)}
    print("english", rep["english_cost"], flush=True)
    rep["receipt"] = TS["table"].receipt()
    rep["wall_s"] = time.time() - t0
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    print("done", out, f"{rep['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
