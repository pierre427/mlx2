#!/usr/bin/env python3
"""Amoeba Phase A (2026-11-16): get mapped injection and distilled tiles working on a pair with a LARGE measured
teacher gap and a SHARED tokenizer, so donor and host positions pair exactly (same token ids, no warp of text).

Default pair: host Qwen3-0.6B, donor Qwen3-4B; task: two-term --digits-digit addition in plain chat (thinking off),
greedy, exact integer oracle; disjoint train / val / test (a, b) pairs. Both models resident in one process.

Arms
  host, donor            ceilings on the test items (greedy)
  side                   reference: gated logit hitch with the full donor, p = (1 - a g) p_host + a g p_donor,
                         g = [max p_donor >= tau] (--side a:tau); shared vocab, so no id mapping
  H1, distilled tiles (no donor at run time):
    tile_init            tiles at random init with w_out = 0: exactly the host (the random-init control)
    tile_kl              tiles trained with FULL-vocab KL(donor || host + tiles) on the donor's own on-policy
                         continuations (1 greedy + --samples at --temperature per train item)
    tile_sft             the same tiles, cross-entropy on the donor's sampled tokens (hard labels; reference)
  H2, mapped / bridged writes (the donor runs beside the host on the same tokens; its box's unit activations,
     standardised, are the features). --write-layers selects where they are written, one arm set per mode:
       warp   each box layer d writes at the host layer of the same relative depth, reading d's features only
              (the matched-layer map, (a)); several box layers give a multi-band write ((c), part 1)
       stub   the host's own contrastive topic layers (its stub), each reading every box layer ((c), part 2)
    ridge_box            (a) closed-form ridge from the box features to the host's matched-layer INCREMENT
                         h_L - h_{L-1} on the paired training text (centred, linear part only)
    ridge_shuf           the same fit on row-shuffled pairs, applied to the real features, rescaled per token to
                         |ridge_box write| (shuffled map)
    ridge_rand           ridge from random donor units (same layers and counts), applied to those units, rescaled
                         per token to |ridge_box write| (random box at the same write norm)
    kgrad_box            (a') donor-derived target: ridge from the box features (continuation positions) to the unit
                         direction -d KL(donor || host) / d h_L, the map rescaled so its mean write norm equals the
                         host's mean increment there (the cap then sets the size). ridge_box's target is the host's
                         own increment, which carries no donor-minus-host signal; this one does
    kgrad_shuf, kgrad_rand   its row-shuffled fit and its random-unit fit, rescaled per token to |kgrad_box write|
    bridge_box           (b) learned low-rank gated bridge per write layer, initialised from kgrad_box (--bridge-init
                         grad; or ridge_box with inc), truncated SVD (--bridge-rank), gate starting at 0.5, trained with full-vocab KL(donor || host + bridge) on the topic text,
                         host frozen, drift penalty KL(host || host + bridge) on ordinary English
    bridge_shuf          the same bridge trained and evaluated on another item's features (the donor reads a
                         different item's prompt, then this item's text): right statistics, wrong problem
    bridge_rand          the bridge on random donor units (same layers and counts)
    bridge_noise         the bridge on N(0, 1) features: no donor information at all (a bridge "from a random
                         donor"); with the drift penalty it is the capacity control for bridge_box
Blend-in, every write and tile: scaled per token to at most --norm-budget x |h_L - h_{L-1}|, the host layer's own
increment (stop-gradient; layer 0: |h_0|). Rows record the pre-clamp ratio (mean over write layers and tokens) and
the clamped fraction, so "closes the gap only with a large write" is visible (the no-reader check, H3).
Box: per donor layer, units ranked by per-token topic mass minus per-token English mass (|a_u| x |W_down[:, u]|;
topic = the positions that predict the train continuations, English = the retain texts), topic/English ratio >=
--min-ratio, top --box-units per layer, on --box-layers (auto: the --n-box-layers layers with the largest box score,
at least 2 apart). The host's stub is chosen the same way from the host's own units.
Training: Adam, --epochs over the train sequences, each step --batch topic sequences plus one English retain text;
the epoch (0 = init) with the lowest val KL(donor || arm) among those with val English drift <= --drift-cap is kept.
Generation is scored on the test items only.
Report (summary.json): test accuracy, gap closed = (arm - host) / (donor - host), val KL to the donor and the
fraction of the host's KL closed (selection metrics: the same val set picks the epoch; the side arm's val KL is its
teacher-forced mixture and its English drift is not measured), English drift KL(host || arm) per token on held English text, norm ratio, clamped
fraction, trainable parameters. Writes run.json (receipt), items.json, rows.jsonl, curves.jsonl, summary.json.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import amoeba_hub_align as ha  # noqa: E402

CORE = "~/Library/Application Support/mlx2/research-artifacts/amoeba/core-corpus-v2-9b64b3791/corpus.jsonl"
BASES = ("ridge_box", "ridge_shuf", "ridge_rand", "kgrad_box", "kgrad_shuf", "kgrad_rand", "bridge_box", "bridge_shuf", "bridge_rand", "bridge_noise")


def make_items(digits, n_train, n_val, n_test, seed):
    rng = np.random.default_rng(seed)
    lo, hi = 10 ** (digits - 1), 10 ** digits
    seen, out = set(), []
    while len(out) < n_train + n_val + n_test:
        a, b = int(rng.integers(lo, hi)), int(rng.integers(lo, hi))
        if (a, b) in seen or (b, a) in seen:
            continue
        seen.add((a, b))
        out.append({"a": a, "b": b, "answer": a + b})
    for i, it in enumerate(out):
        it["split"] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
        it["id"] = f"{it['split']}{i:04d}"
        it["prompt"] = f"What is {it['a']} + {it['b']}? Answer with only the number."
    return out


def parse(text):
    text = text.split("</think>")[-1]
    m = re.findall(r"-?\d[\d,]*", text)  # the last number: an echoed question ("a + b = c") still scores c
    return int(m[-1].replace(",", "")) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=str(Path.home() / "mlx-models/Qwen3-0.6B-BF16-microscaling-source"))
    ap.add_argument("--donor", default=str(Path.home() / "mlx-models/Qwen3-4B"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--digits", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=120)
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--samples", type=int, default=3, help="donor samples per train item beside the greedy one")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--eng-retain", type=int, default=12)
    ap.add_argument("--eng-val", type=int, default=4)
    ap.add_argument("--eng-test", type=int, default=8)
    ap.add_argument("--eng-tokens", type=int, default=96)
    ap.add_argument("--arms", default="", help="comma list of arm names or bases (default: all)")
    ap.add_argument("--write-layers", default="warp,stub", help="write-layer modes for ridge/bridge arms")
    ap.add_argument("--box-layers", default="auto", help="donor box layers, comma list, or auto")
    ap.add_argument("--n-box-layers", type=int, default=4)
    ap.add_argument("--n-stub-layers", type=int, default=4)
    ap.add_argument("--box-units", type=int, default=64)
    ap.add_argument("--min-ratio", type=float, default=1.5)
    ap.add_argument("--ridge-lam", type=float, default=0.1, help="ridge penalty relative to mean feature variance")
    ap.add_argument("--bridge-rank", type=int, default=32)
    ap.add_argument("--bridge-init", default="grad", choices=["grad", "inc"],
                    help="bridge init: the gradient-direction ridge (kgrad) or the increment ridge; gate starts at 0.5")
    ap.add_argument("--tile-layers", default="all", help="host tile layers: all, stub, or a comma list")
    ap.add_argument("--tile-units", type=int, default=32, help="tile neurons per tile layer")
    ap.add_argument("--norm-budget", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--tile-lr", type=float, default=1e-3)
    ap.add_argument("--bridge-lr", type=float, default=1e-3)
    ap.add_argument("--drift-weight", type=float, default=1.0)
    ap.add_argument("--drift-cap", type=float, default=0.05, help="max val English KL(host || arm) for selection")
    ap.add_argument("--side", default="1:0.5", help="side reference arm alpha:tau ('' = off)")
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--donor-dtype", default="native", choices=["float32", "bfloat16", "native"])
    ap.add_argument("--mem-cap-gb", type=float, default=0)
    ap.add_argument("--budget-s", type=float, default=6600)
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to write into non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map
    if args.cpu:
        mx.set_default_device(mx.cpu)
    if args.mem_cap_gb:
        mx.set_memory_limit(int(args.mem_cap_gb * 1e9))
    mx.set_cache_limit(2 << 30)
    from mlx2.adapters.registry import inspect_model
    from mlx2.experimental.weight_space import residual_ablation as ra
    from mlx2.serving import render_prompt_tokens

    def load(path, dtype):
        ad = inspect_model(path).adapter_type(str(path))
        m = ad.model
        ha.dequantize_model(mx, m, mx.float32 if dtype is None else dtype)
        if dtype is not None:
            m.update(tree_map(lambda a: a.astype(dtype) if a.dtype in (mx.bfloat16, mx.float16) else a, m.parameters()))
            mx.eval(m.parameters())
        m.eval()
        lm = getattr(m, "language_model", m)
        return {"ad": ad, "model": m, "layers": list(lm.model.layers), "tok": ad.tokenizer,
                "ck": f"{Path(path).name}@{ad.identity['fingerprint']}"}

    ddt = {"float32": mx.float32, "bfloat16": mx.bfloat16, "native": None}[args.donor_dtype]
    H = load(args.host, mx.float32)
    Dn = load(args.donor, ddt)
    LH, LD = len(H["layers"]), len(Dn["layers"])
    tok = H["tok"]
    rev = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()) or bool(subprocess.run(
        ["git", "status", "--porcelain", "--", "scripts/amoeba_bridge_lab.py"], cwd=ROOT, capture_output=True,
        text=True).stdout.strip())
    import hashlib
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    receipt = {"schema": "amoeba-bridge-lab/v1", "host_checkpoint": H["ck"], "donor_checkpoint": Dn["ck"],
               "mlx2_source_revision": rev, "tree_dirty": dirty, "script_sha256": script_sha, "args": vars(args),
               "status": "running",
               "device": str(mx.default_device()),
               "training": "tiles and bridges trained on a frozen host (Pierre 2026-11-16: 'please get the mapped "
                           "injections and distilled tiles working'); ridge arms closed form"}

    def save_receipt(**kw):
        receipt.update(kw, wall_s=round(time.time() - t0, 1))
        (out / "run.json").write_text(json.dumps(receipt, indent=1))
    save_receipt()

    def render(M, text):
        return [int(t) for t in render_prompt_tokens(
            M["ad"], {"messages": [{"role": "user", "content": text}], "enable_thinking": False})]

    items = make_items(args.digits, args.n_train, args.n_val, args.n_test, args.seed)
    for it in items:
        it["ids"] = render(H, it["prompt"])
        if render(Dn, it["prompt"]) != it["ids"]:
            sys.exit(f"donor and host render {it['id']} differently: the shared-tokenizer premise fails")
    eng_all = [json.loads(line) for line in open(CORE)]
    fit_e = [r for r in eng_all if r["split"] == "fit"]
    held_e = [r for r in eng_all if r["split"] == "held"]
    g = np.random.default_rng(args.seed + 7)
    pick_f = g.choice(len(fit_e), args.eng_retain + args.eng_val, replace=False)
    pick_h = g.choice(len(held_e), args.eng_test, replace=False)
    eng = ([{"id": fit_e[i]["id"], "role": "retain", "text": fit_e[i]["text"]} for i in pick_f[:args.eng_retain]]
           + [{"id": fit_e[i]["id"], "role": "eng_val", "text": fit_e[i]["text"]} for i in pick_f[args.eng_retain:]]
           + [{"id": held_e[i]["id"], "role": "eng_test", "text": held_e[i]["text"]} for i in pick_h])
    for e in eng:
        e["ids"] = tok.encode(e["text"], add_special_tokens=False)[:args.eng_tokens]
        if Dn["tok"].encode(e["text"], add_special_tokens=False)[:args.eng_tokens] != e["ids"]:
            sys.exit(f"donor tokenizer differs on English text {e['id']}")
    eos = set(tok.eos_token_ids) | set(Dn["tok"].eos_token_ids)
    print(f"host {H['ck']} ({LH} layers), donor {Dn['ck']} ({LD} layers), rev {rev} dirty {dirty}; "
          f"{len(items)} items, {len(eng)} English texts; shared tokenizer verified", flush=True)

    # ------------------------------------------------------------------ hooks: MLP unit capture, layer writes
    MCTL = {}

    def patch_mlp(M):
        base = type(M["layers"][0].mlp)

        class UnitMLP(base):
            def __call__(self, x, **kw):
                c = MCTL.get(id(self))
                if c is None:
                    return base.__call__(self, x, **kw)
                a = nn.silu(self.gate_proj(x)) * self.up_proj(x)
                c["out"] = a[0].astype(mx.float32) if c["idx"] is None else a[0][:, c["idx"]].astype(mx.float32)
                return self.down_proj(a)
        for L in M["layers"]:
            L.mlp.__class__ = UnitMLP
    patch_mlp(H)
    patch_mlp(Dn)

    def out_of(o):
        return o[0] if isinstance(o, tuple) else o

    def with_out(o, x):
        return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x

    def run(M, ids, cache=None, writes=None, tiles=None, cap=(), st=None, units=None):
        """log-probs [T, V] float32. writes {L: [T, D]} and tiles are budgeted; cap: layer outputs kept in st['h'];
        units {layer: idx or None}: MLP unit activations kept in st['u']."""
        layers = M["layers"]
        wl = set(writes or {}) | (set(tiles.layers_set) if tiles is not None else set())
        hooked = wl | {l - 1 for l in wl if l > 0} | set(cap)
        prev = {}

        def hook(l):
            def h(o):
                x = out_of(o)
                if l in wl:
                    x0 = x
                    v = mx.zeros_like(x0)
                    if tiles is not None and l in tiles.layers_set:
                        v = v + tiles(l, x0)
                    if writes and l in writes:
                        v = v + writes[l][None].astype(x0.dtype)
                    ref = prev.get(l - 1)
                    d0 = (x0 - ref) if ref is not None else x0
                    loc = mx.stop_gradient(mx.sqrt((d0.astype(mx.float32) ** 2).sum(-1, keepdims=True)))
                    r_ = mx.sqrt((v.astype(mx.float32) ** 2).sum(-1, keepdims=True) + 1e-12) / (loc + 1e-12)
                    if st is not None:
                        st.setdefault("nr", []).append(mx.stop_gradient(r_).reshape(-1))
                    if args.norm_budget > 0:
                        v = v * mx.minimum(1.0, args.norm_budget / (r_ + 1e-12)).astype(v.dtype)
                    x = x0 + v.astype(x0.dtype)
                if l in cap:
                    st.setdefault("h", {})[l] = x[0].astype(mx.float32)
                prev[l] = x
                return with_out(o, x)
            return h

        if units:
            for l, idx in units.items():
                MCTL[id(layers[l].mlp)] = {"idx": idx}
        try:
            with ra._hooked([(layers[l], hook(l)) for l in sorted(hooked)]):
                cache = M["model"].make_cache() if cache is None else cache
                lg = M["model"](mx.array([ids]), cache=cache)[0].astype(mx.float32)
        finally:
            if units:
                got = {l: MCTL.pop(id(layers[l].mlp)).get("out") for l in units}
                if st is not None:
                    st["u"] = got
        return lg - mx.logsumexp(lg, axis=-1, keepdims=True)

    def budget_left():
        return args.budget_s - (time.time() - t0)

    # ------------------------------------------------------------------ donor stage 1: traces and ceilings
    def donor_generate(it, temp, key):
        c = Dn["model"].make_cache()
        lp = run(Dn, it["ids"], cache=c)[-1]
        toks = []
        for _ in range(args.max_tokens):
            if temp > 0:
                key, sk = mx.random.split(key)
                t = int(mx.random.categorical(lp / temp, key=sk).item())
            else:
                t = int(mx.argmax(lp).item())
            toks.append(t)
            if t in eos:
                break
            lp = run(Dn, [t], cache=c)[-1]
        return toks, key

    rows_f = open(out / "rows.jsonl", "w")
    key = mx.random.key(args.seed)
    seqs = []  # topic training / validation sequences: prompt + donor continuation
    donor_acc = {"train": [], "val": [], "test": []}
    for it in items:
        k_ = 1 + (args.samples if it["split"] == "train" else 0)
        for s in range(k_):
            toks, key = donor_generate(it, 0.0 if s == 0 else args.temperature, key)
            text = tok.decode([t for t in toks if t not in eos])
            ok = parse(text) == it["answer"]
            if s == 0:
                donor_acc[it["split"]].append(ok)
                if it["split"] == "test":
                    rows_f.write(json.dumps({"arm": "donor", "item": it["id"], "text": text, "pred": parse(text),
                                             "correct": ok}) + "\n")
            if it["split"] in ("train", "val") and (it["split"] == "train" or s == 0):
                seqs.append({"item": it["id"], "split": it["split"], "sample": s, "P": len(it["ids"]),
                             "ids": it["ids"] + toks, "donor_correct": ok})
    rows_f.flush()
    print(f"{time.strftime('%H:%M:%S')} donor greedy accuracy train {np.mean(donor_acc['train']):.3f} "
          f"val {np.mean(donor_acc['val']):.3f} test {np.mean(donor_acc['test']):.3f}; {len(seqs)} sequences",
          flush=True)
    tr = [s for s in seqs if s["split"] == "train"]
    va = [s for s in seqs if s["split"] == "val"]
    for s in seqs:
        s["pos"] = list(range(s["P"] - 1, len(s["ids"]) - 1))  # positions predicting the continuation
    by_id = {it["id"]: it for it in items}
    # wrong-item partner for the shuffled-feature arms: same prompt length, different item, fixed derangement
    for group in (tr, va):
        idsets = sorted({s["item"] for s in group})
        perm = {}
        for i, iid in enumerate(idsets):
            P = len(by_id[iid]["ids"])
            cands = [j for j in idsets[i + 1:] + idsets[:i] if len(by_id[j]["ids"]) == P]
            perm[iid] = cands[0] if cands else None
        for s in group:
            s["partner"] = perm[s["item"]]

    # ------------------------------------------------------------------ donor stage 2: teacher, unit statistics
    wn_d = {l: np.sqrt((np.array(Dn["layers"][l].mlp.down_proj.weight.astype(mx.float32)) ** 2).sum(0))
            for l in range(LD)}
    wn_h = {l: np.sqrt((np.array(H["layers"][l].mlp.down_proj.weight.astype(mx.float32)) ** 2).sum(0))
            for l in range(LH)}

    def unit_mass(M, wn, texts, topic):
        L_ = len(M["layers"])
        acc = [np.zeros_like(wn[l]) for l in range(L_)]
        n = 0
        for s in texts:
            st = {}
            run(M, s["ids"], st=st, units={l: None for l in range(L_)})
            p = s["pos"] if topic else list(range(len(s["ids"])))
            for l in range(L_):
                acc[l] += np.abs(np.array(st["u"][l])[p]).sum(0) * wn[l]
            n += len(p)
            del st
        return np.stack(acc) / max(n, 1)

    retain = [e for e in eng if e["role"] == "retain"]
    eng_val = [e for e in eng if e["role"] == "eng_val"]
    eng_test = [e for e in eng if e["role"] == "eng_test"]
    for s in tr + va:  # donor teacher: full-vocab log-probs at the continuation positions (float16)
        s["t_lp"] = np.array(run(Dn, s["ids"])[mx.array(s["pos"])]).astype(np.float16)
    stat_tr = [s for s in tr if s["sample"] == 0]
    md_t, md_e = unit_mass(Dn, wn_d, stat_tr, True), unit_mass(Dn, wn_d, retain, False)
    mh_t, mh_e = unit_mass(H, wn_h, stat_tr, True), unit_mass(H, wn_h, retain, False)
    print(f"{time.strftime('%H:%M:%S')} teacher log-probs and unit masses done", flush=True)

    def select(mt, me, k, nl, fixed):
        score = mt - me
        elig = mt >= args.min_ratio * me
        per = np.where(elig, score, -np.inf)
        sel = {l: np.argsort(-per[l])[:k] for l in range(len(mt))}
        sel = {l: v[np.isfinite(per[l][v])] for l, v in sel.items()}
        lscore = np.array([per[l][sel[l]].sum() if len(sel[l]) else -np.inf for l in range(len(mt))])
        if fixed != "auto":
            layers_ = [int(x) for x in fixed.split(",")]
        else:
            layers_ = []
            for l in np.argsort(-lscore):
                if 0 < l < len(mt) - 1 and np.isfinite(lscore[l]) and all(abs(l - m) >= 2 for m in layers_):
                    layers_.append(int(l))
                if len(layers_) == nl:
                    break
        return sorted(layers_), {l: sel[l] for l in layers_}, lscore

    box_layers, box, d_lscore = select(md_t, md_e, args.box_units, args.n_box_layers, args.box_layers)
    if not box_layers or not all(len(box[l]) for l in box_layers):
        save_receipt(status="failed", reason=f"no donor box: layers {box_layers} (min-ratio {args.min_ratio})")
        sys.exit(f"no donor box passes --min-ratio {args.min_ratio}: layers {box_layers}")
    stub_layers, _, h_lscore = select(mh_t, mh_e, args.box_units, args.n_stub_layers, "auto")
    rand = {}
    for l in box_layers:
        free = np.setdiff1d(np.arange(len(wn_d[l])), box[l])
        rand[l] = rng.choice(free, len(box[l]), replace=False)
    union = {l: mx.array(np.concatenate([box[l], rand[l]]).astype(np.int32)) for l in box_layers}
    kb = {l: len(box[l]) for l in box_layers}
    off = np.cumsum([0] + [kb[l] for l in box_layers])
    colsd = {l: slice(int(off[i]), int(off[i + 1])) for i, l in enumerate(box_layers)}
    FD = int(off[-1])
    warp = {d: int(round(d * (LH - 1) / (LD - 1))) for d in box_layers}
    if len(set(warp.values())) < len(warp):
        save_receipt(status="failed", reason=f"box layers {box_layers} collide under the warp {warp}")
        sys.exit(f"box layers {box_layers} map to the same host layer: {warp}")
    box_info = {"box_layers": box_layers, "units_per_layer": {str(l): int(kb[l]) for l in box_layers},
                "box": {str(l): box[l].tolist() for l in box_layers}, "rand": {str(l): rand[l].tolist() for l in box_layers},
                "topic_over_english": {str(l): float(md_t[l][box[l]].sum() / max(md_e[l][box[l]].sum(), 1e-12))
                                       for l in box_layers},
                "donor_layer_score": d_lscore.tolist(), "host_stub_layers": stub_layers,
                "host_layer_score": h_lscore.tolist(), "warp": {str(d): L for d, L in warp.items()}}
    (out / "box.json").write_text(json.dumps(box_info, indent=1))
    print(f"box layers {box_layers} units {[kb[l] for l in box_layers]} -> warp {[warp[d] for d in box_layers]}; "
          f"host stub {stub_layers}", flush=True)

    # ------------------------------------------------------------------ donor stage 3: features
    def dfeat(ids, cache=None, st=None):
        st = {} if st is None else st
        lp = run(Dn, ids, cache=cache, st=st, units=dict(union))
        fb = np.concatenate([np.array(st["u"][l])[:, :kb[l]] for l in box_layers], 1)
        fr = np.concatenate([np.array(st["u"][l])[:, kb[l]:] for l in box_layers], 1)
        return lp, fb, fr

    for s in tr + va:
        _, s["fb"], s["fr"] = dfeat(s["ids"])
        if s["partner"] is not None:
            _, s["fs"], _ = dfeat(by_id[s["partner"]]["ids"] + s["ids"][s["P"]:])
        else:
            s["fs"] = s["fb"][rng.permutation(len(s["fb"]))]
    for e in eng:
        _, e["fb"], e["fr"] = dfeat(e["ids"])
    for i, e in enumerate(eng):  # wrong-text features for the shuffled arms on English
        o = [x for x in eng if x["role"] == e["role"]]
        j = o[(o.index(e) + 1) % len(o)]
        fs = np.zeros_like(e["fb"])
        n_ = min(len(fs), len(j["fb"]))
        fs[:n_] = j["fb"][:n_]
        e["fs"] = fs
    allb = np.concatenate([s["fb"] for s in tr])
    allr = np.concatenate([s["fr"] for s in tr])
    mu_b, sd_b = allb.mean(0), allb.std(0) + 1e-6
    mu_r, sd_r = allr.mean(0), allr.std(0) + 1e-6
    zb = lambda f: ((f - mu_b) / sd_b).astype(np.float32)  # noqa: E731
    zr = lambda f: ((f - mu_r) / sd_r).astype(np.float32)  # noqa: E731
    for s in tr + va + eng:
        s["fs"] = zb(s["fs"])
        s["fb"], s["fr"] = zb(s["fb"]), zr(s["fr"])
    print(f"{time.strftime('%H:%M:%S')} donor features done ({FD} features)", flush=True)

    # ------------------------------------------------------------------ host references
    for s in tr + va:
        s["h_lp"] = np.array(run(H, s["ids"])[mx.array(s["pos"])]).astype(np.float16)
    for e in eng:
        e["h_lp"] = np.array(run(H, e["ids"])).astype(np.float16)

    def kl_rows(t_lp, lp):
        t = mx.array(t_lp.astype(np.float32))
        return (mx.exp(t) * (t - lp)).sum(-1)

    host_val_kl = float(np.mean([float(kl_rows(s["t_lp"], mx.array(s["h_lp"].astype(np.float32))).mean().item())
                                 for s in va]))
    write_sets = {}
    for mode in [m.strip() for m in args.write_layers.split(",") if m.strip()]:
        if mode == "warp":
            write_sets[mode] = {warp[d]: colsd[d] for d in box_layers}
        elif mode == "stub":
            write_sets[mode] = {L: slice(0, FD) for L in stub_layers}
        else:
            sys.exit(f"unknown write-layer mode {mode}")
    all_w = sorted({L for ws in write_sets.values() for L in ws})
    # host increments at the write layers on the training text (all positions): h_L - h_{L-1}
    capl = sorted(set(all_w) | {L - 1 for L in all_w if L > 0})
    D_h = int(H["layers"][0].mlp.down_proj.weight.shape[0])
    inc = {L: [] for L in all_w}
    for s in tr:
        st = {}
        run(H, s["ids"], cap=capl, st=st)
        for L in all_w:
            inc[L].append(np.array(st["h"][L] - st["h"][L - 1]) if L > 0 else np.array(st["h"][L]))
    inc = {L: np.concatenate(v) for L, v in inc.items()}
    Xb = np.concatenate([s["fb"] for s in tr])
    Xr = np.concatenate([s["fr"] for s in tr])
    # donor-derived target (the host's own increment carries no donor signal): the direction that moves the host
    # toward the donor at each write layer, -d KL(donor || host) / d h_L, at the continuation positions; rows
    # normalised to unit norm (direction only); the fitted map is rescaled so its mean write norm equals the host's
    # mean increment norm there, and the blend-in cap then sets the size
    gdir = {L: [] for L in all_w}
    incp = {L: [] for L in all_w}
    for s in tr:
        pos = mx.array(s["pos"])
        z0 = {L: mx.zeros((len(s["ids"]), D_h)) for L in all_w}
        gr = mx.grad(lambda w, s=s, pos=pos: kl_rows(s["t_lp"], run(H, s["ids"], writes=w)[pos]).mean())(z0)
        st = {}
        run(H, s["ids"], cap=capl, st=st)
        for L in all_w:
            gL = -np.array(gr[L])[s["pos"]]
            gdir[L].append(gL / (np.linalg.norm(gL, axis=1, keepdims=True) + 1e-12))
            d_ = np.array(st["h"][L] - st["h"][L - 1]) if L > 0 else np.array(st["h"][L])
            incp[L].append(np.linalg.norm(d_[s["pos"]], axis=1))
    gdir = {L: np.concatenate(v) for L, v in gdir.items()}
    incp = {L: float(np.concatenate(v).mean()) for L, v in incp.items()}
    Xbp = np.concatenate([s["fb"][s["pos"]] for s in tr])
    Xrp = np.concatenate([s["fr"][s["pos"]] for s in tr])

    def cos_fit(X, M_, Y):
        P_ = (X - X.mean(0)) @ M_
        return float(np.mean(np.sum(P_ * Y, 1) / (np.linalg.norm(P_, axis=1) + 1e-12)))

    def ridge(X, Y):
        Xc, Yc = X - X.mean(0), Y - Y.mean(0)
        G = Xc.T @ Xc
        lam = args.ridge_lam * np.trace(G) / G.shape[0]
        return np.linalg.solve(G + lam * np.eye(G.shape[0]), Xc.T @ Yc).astype(np.float32)

    def r2(X, Y, M_):
        Xc, Yc = X - X.mean(0), Y - Y.mean(0)
        return float(1 - ((Yc - Xc @ M_) ** 2).sum() / (Yc ** 2).sum())

    maps = {}
    for mode, ws in write_sets.items():
        for L, cs in ws.items():
            Y = inc[L]
            sh = rng.permutation(len(Y))
            Mb, Ms, Mr = ridge(Xb[:, cs], Y), ridge(Xb[:, cs], Y[sh]), ridge(Xr[:, cs], Y)
            Yg = gdir[L]
            shg = rng.permutation(len(Yg))

            def scaled(X, M_, L=L):
                n_ = np.linalg.norm((X - X.mean(0)) @ M_, axis=1).mean()
                return (M_ * (incp[L] / max(n_, 1e-12))).astype(np.float32)
            Gb = scaled(Xbp[:, cs], ridge(Xbp[:, cs], Yg))
            Gs = scaled(Xbp[:, cs], ridge(Xbp[:, cs], Yg[shg]))
            Gr = scaled(Xrp[:, cs], ridge(Xrp[:, cs], Yg))
            maps[(mode, L)] = {"M": mx.array(Mb), "Msh": mx.array(Ms), "Mr": mx.array(Mr),
                               "G": mx.array(Gb), "Gsh": mx.array(Gs), "Gr": mx.array(Gr),
                               "r2": r2(Xb[:, cs], Y, Mb), "r2_rand": r2(Xr[:, cs], Y, Mr),
                               "cos_g": cos_fit(Xbp[:, cs], Gb, Yg), "cos_g_rand": cos_fit(Xrp[:, cs], Gr, Yg),
                               "np": Mb, "np_g": Gb, "np_gr": Gr}
    receipt["ridge_r2"] = {f"{m}@{L}": {"inc_box": round(v["r2"], 4), "inc_rand": round(v["r2_rand"], 4),
                                        "graddir_cos_box": round(v["cos_g"], 4),
                                        "graddir_cos_rand": round(v["cos_g_rand"], 4)}
                           for (m, L), v in maps.items()}
    print(f"{time.strftime('%H:%M:%S')} host references done; host val KL to donor {host_val_kl:.4f}; ridge R^2 "
          f"{receipt['ridge_r2']}", flush=True)
    save_receipt(stage="arms", box=box_info, host_val_kl=host_val_kl, donor_accuracy={k: float(np.mean(v)) for k, v in donor_acc.items()})

    # ------------------------------------------------------------------ trainable modules
    D = int(H["layers"][0].mlp.down_proj.weight.shape[0])

    def rmsn(x):
        return x * mx.rsqrt((x * x).mean(-1, keepdims=True) + 1e-6)

    if args.tile_layers == "all":
        tile_layers = list(range(LH))
    elif args.tile_layers == "stub":
        tile_layers = list(stub_layers)
    else:
        tile_layers = [int(x) for x in args.tile_layers.split(",")]

    class Tiles(nn.Module):
        def __init__(self, seed):
            super().__init__()
            g_ = np.random.default_rng(seed)
            self.layers_set = set(tile_layers)
            n = args.tile_units
            self.w_in = {str(l): mx.array((g_.standard_normal((D, n)) / np.sqrt(D)).astype(np.float32)) for l in tile_layers}
            self.w_out = {str(l): mx.zeros((n, D)) for l in tile_layers}  # untrained == host
            self.gate_w = {str(l): mx.zeros((D,)) for l in tile_layers}
            self.gate_b = {str(l): mx.zeros((1,)) for l in tile_layers}

        def __call__(self, l, x):
            k = str(l)
            xn = rmsn(x.astype(mx.float32))
            gt = mx.sigmoid(mx.stop_gradient(xn) @ self.gate_w[k] + self.gate_b[k])[..., None]
            return gt * (nn.silu(xn @ self.w_in[k]) @ self.w_out[k])

    class Bridge(nn.Module):
        def __init__(self, mode, src):
            super().__init__()
            self.A, self.B, self.gw, self.gb = {}, {}, {}, {}
            for L, cs in write_sets[mode].items():
                mp = maps[(mode, L)]
                if args.bridge_init == "grad":
                    M_ = mp["np_g"] if src != "rand" else mp["np_gr"]
                else:
                    M_ = mp["np"] if src != "rand" else np.array(mp["Mr"])
                U, S, Vt = np.linalg.svd(M_, full_matrices=False)
                r = min(args.bridge_rank, len(S))
                self.A[str(L)] = mx.array((U[:, :r] * np.sqrt(S[:r])).astype(np.float32))
                self.B[str(L)] = mx.array((np.sqrt(S[:r])[:, None] * Vt[:r]).astype(np.float32))
                self.gw[str(L)] = mx.zeros((cs.stop - cs.start,))
                self.gb[str(L)] = mx.zeros((1,))

        def write(self, L, z):
            k = str(L)
            gt = mx.sigmoid(z @ self.gw[k] + self.gb[k])[:, None]
            return gt * ((z @ self.A[k]) @ self.B[k])

    def nparams(m):
        return int(sum(v.size for _, v in tree_flatten(m.trainable_parameters())))

    # ------------------------------------------------------------------ arm definitions
    arms = {"host": {"route": "host"}, "tile_init": {"route": "tile", "loss": None},
            "tile_kl": {"route": "tile", "loss": "kl"}, "tile_sft": {"route": "tile", "loss": "sft"}}
    if args.side:
        a_, t_ = (float(x) for x in args.side.split(":"))
        arms[f"side@a{a_:g}t{t_:g}"] = {"route": "side", "alpha": a_, "tau": t_}
    for mode in write_sets:
        for b in BASES:
            kind, src = b.split("_")
            arms[f"{b}@{mode}"] = {"route": "ridge" if kind == "kgrad" else kind, "src": src, "mode": mode,
                                   "target": "grad" if kind == "kgrad" else "inc"}
    if args.arms:
        want = [a.strip() for a in args.arms.split(",") if a.strip()]
        arms = {n: v for n, v in arms.items() if n in want or n.split("@")[0] in want or n == "host"}

    def feats(arm, s):
        return {"box": s["fb"], "shuf": s["fs"], "rand": s["fr"]}.get(arm["src"])

    def writes_of(arm, mod, z, T, g_=None):
        """{L: [T, D]} for a bridge arm from standardised features z [T, FD] (None for noise)."""
        ws = write_sets[arm["mode"]]
        w = {}
        for L, cs in ws.items():
            if arm["src"] == "noise":
                zz = mx.array((g_ or rng).standard_normal((T, cs.stop - cs.start)).astype(np.float32))
            else:
                zz = mx.array(z[:, cs]) if isinstance(z, np.ndarray) else z[:, cs]
            w[L] = mod.write(L, zz)
        return w

    def ridge_writes(arm, zbox, zrand):
        ws = write_sets[arm["mode"]]
        w = {}
        for L, cs in ws.items():
            mp = maps[(arm["mode"], L)]
            k_, ks, kr = ("G", "Gsh", "Gr") if arm["target"] == "grad" else ("M", "Msh", "Mr")
            real = mx.array(zbox[:, cs]) @ mp[k_]
            if arm["src"] == "box":
                w[L] = real
                continue
            v = mx.array(zbox[:, cs]) @ mp[ks] if arm["src"] == "shuf" else mx.array(zrand[:, cs]) @ mp[kr]
            w[L] = v * (mx.sqrt((real * real).sum(-1, keepdims=True)) / (mx.sqrt((v * v).sum(-1, keepdims=True)) + 1e-12))
        return w

    def arm_lp(arm, mod, s, st=None, g_=None):
        T = len(s["ids"])
        if arm["route"] == "tile":
            return run(H, s["ids"], tiles=mod, st=st)
        if arm["route"] == "ridge":
            return run(H, s["ids"], writes=ridge_writes(arm, s["fb"], s["fr"]), st=st)
        if arm["route"] == "bridge":
            return run(H, s["ids"], writes=writes_of(arm, mod, feats(arm, s), T, g_), st=st)
        return run(H, s["ids"], st=st)

    def side_lp(arm, s):  # teacher-forced side mixture at the continuation positions (donor lp = s["t_lp"])
        lh = run(H, s["ids"])[mx.array(s["pos"])]
        pd = mx.exp(mx.array(s["t_lp"].astype(np.float32)))
        gate = (pd.max(-1, keepdims=True) >= arm["tau"]).astype(mx.float32) * arm["alpha"]
        return mx.log((1 - gate) * mx.exp(lh) + gate * pd + 1e-30)

    def val_scores(arm, mod):
        g_ = np.random.default_rng(args.seed + 99)  # the noise arm sees the same draws at every selection
        if arm["route"] == "side":  # English drift needs donor log-probs on English, which are not stored
            return float(np.mean([float(kl_rows(s["t_lp"], side_lp(arm, s)).mean().item()) for s in va])), None
        kv = [float(kl_rows(s["t_lp"], arm_lp(arm, mod, s, g_=g_)[mx.array(s["pos"])]).mean().item()) for s in va]
        dv = []
        for e in eng_val:
            lp = arm_lp(arm, mod, e, g_=g_)
            h = mx.array(e["h_lp"].astype(np.float32))
            dv.append(float((mx.exp(h) * (h - lp)).sum(-1).mean().item()))
        return float(np.mean(kv)), float(np.mean(dv))

    curves_f = open(out / "curves.jsonl", "w")

    def train(name, arm, mod, lr):
        opt = optim.Adam(learning_rate=lr)

        def loss_topic(m_, s):
            lp = arm_lp(arm, m_, s)[mx.array(s["pos"])]
            if arm.get("loss") == "sft":
                tgt = mx.array(s["ids"][s["P"]:])[:, None]
                return -mx.take_along_axis(lp, tgt, axis=-1).mean()
            return kl_rows(s["t_lp"], lp).mean()

        def loss_eng(m_, e):
            lp = arm_lp(arm, m_, e)
            h = mx.array(e["h_lp"].astype(np.float32))
            return args.drift_weight * (mx.exp(h) * (h - lp)).sum(-1).mean()

        gt = nn.value_and_grad(mod, loss_topic)
        ge = nn.value_and_grad(mod, loss_eng)
        kv0, dv0 = val_scores(arm, mod)
        # epoch 0 (init) competes only within the drift cap; if no epoch is within it, keep the lowest-drift one
        best = (kv0, 0, mod.parameters()) if dv0 <= args.drift_cap else (float("inf"), 0, mod.parameters())
        low = (dv0, 0, mod.parameters())
        hist = [{"arm": name, "epoch": 0, "val_kl": kv0, "val_drift": dv0}]
        curves_f.write(json.dumps(hist[-1]) + "\n")
        curves_f.flush()
        print(f"  {name} ep 0: val KL {kv0:.4f} drift {dv0:.4f}", flush=True)
        for ep in range(1, args.epochs + 1):
            order = rng.permutation(len(tr))
            ls, nb = [], 0
            for b0 in range(0, len(order), args.batch):
                acc = None
                for i in order[b0:b0 + args.batch]:
                    l_, g_ = gt(mod, tr[i])
                    ls.append(float(l_.item()))
                    acc = g_ if acc is None else tree_map(lambda a, b: a + b, acc, g_)
                if args.drift_weight > 0 and retain:
                    l_, g_ = ge(mod, retain[nb % len(retain)])
                    acc = tree_map(lambda a, b: a / len(order[b0:b0 + args.batch]) + b, acc, g_)
                else:
                    acc = tree_map(lambda a: a / len(order[b0:b0 + args.batch]), acc)
                opt.update(mod, acc)
                mx.eval(mod.parameters(), opt.state)
                nb += 1
                if budget_left() < 300:
                    break
            kv, dv = val_scores(arm, mod)
            hist.append({"arm": name, "epoch": ep, "train_loss": float(np.mean(ls)), "val_kl": kv, "val_drift": dv})
            curves_f.write(json.dumps(hist[-1]) + "\n")
            curves_f.flush()
            print(f"  {time.strftime('%H:%M:%S')} {name} ep {ep}: train {np.mean(ls):.4f} val KL {kv:.4f} "
                  f"drift {dv:.4f}", flush=True)
            if dv <= args.drift_cap and kv < best[0]:
                best = (kv, ep, mod.parameters())
            if dv < low[0]:
                low = (dv, ep, mod.parameters())
            if budget_left() < 300:
                break
        within = np.isfinite(best[0])
        sel = best if within else (hist[low[1]]["val_kl"], low[1], low[2])
        mod.update(sel[2])
        return {"selected_epoch": sel[1], "val_kl_selected": sel[0], "within_drift_cap": bool(within), "curve": hist}

    # ------------------------------------------------------------------ generation on the test items
    def generate(name, arm, mod, it):
        need_d = arm["route"] == "side" or (arm["route"] in ("ridge", "bridge") and arm["src"] != "noise")
        P = it["ids"]
        dprompt = P
        if arm["route"] == "bridge" and arm["src"] == "shuf":
            same = [x for x in items if x["split"] == "test" and x["id"] != it["id"] and len(x["ids"]) == len(P)]
            dprompt = same[(int(it["id"][-4:]) * 7919) % len(same)]["ids"]
        hc = H["model"].make_cache()
        dc = Dn["model"].make_cache() if need_d else None
        st = {}

        def step(ids_h, ids_d):
            dlp, fb, fr = (None, None, None)
            if need_d:
                if arm["route"] == "side":
                    dlp = run(Dn, ids_d, cache=dc)
                else:
                    dlp, fb, fr = dfeat(ids_d, cache=dc)
                    fb, fr = zb(fb), zr(fr)
            T = len(ids_h)
            if arm["route"] == "tile":
                lp = run(H, ids_h, cache=hc, tiles=mod, st=st)
            elif arm["route"] == "ridge":
                lp = run(H, ids_h, cache=hc, writes=ridge_writes(arm, fb, fr), st=st)
            elif arm["route"] == "bridge":
                z = None if arm["src"] == "noise" else (fr if arm["src"] == "rand" else fb)
                lp = run(H, ids_h, cache=hc, writes=writes_of(arm, mod, z, T), st=st)
            else:
                lp = run(H, ids_h, cache=hc)
            lp = lp[-1]
            if arm["route"] == "side":
                pd = mx.exp(dlp[-1])
                gate = float(pd.max().item()) >= arm["tau"]
                if gate:
                    lp = mx.log((1 - arm["alpha"]) * mx.exp(lp) + arm["alpha"] * pd + 1e-30)
            return lp

        lp = step(P, dprompt)
        toks = []
        for _ in range(args.max_tokens):
            t = int(mx.argmax(lp).item())
            if t in eos:
                break
            toks.append(t)
            lp = step([t], [t])
        text = tok.decode(toks)
        nr = np.concatenate([np.array(x) for x in st.get("nr", [])]) if st.get("nr") else np.zeros(0)
        return {"arm": name, "item": it["id"], "text": text, "pred": parse(text), "correct": parse(text) == it["answer"],
                "norm_ratio": float(nr.mean()) if len(nr) else None,
                "clamped_frac": float((nr > args.norm_budget).mean()) if len(nr) and args.norm_budget > 0 else None}

    def drift_test(arm, mod):
        if arm["route"] == "side":
            return None
        dv = []
        for e in eng_test:
            lp = arm_lp(arm, mod, e)
            h = mx.array(e["h_lp"].astype(np.float32))
            dv.append(float((mx.exp(h) * (h - lp)).sum(-1).mean().item()))
        return float(np.mean(dv))

    test = [it for it in items if it["split"] == "test"]
    summary = {"host_checkpoint": H["ck"], "donor_checkpoint": Dn["ck"], "mlx2_source_revision": rev,
               "tree_dirty": dirty, "task": f"{args.digits}-digit addition, plain chat, thinking off, greedy",
               "n_test": len(test), "donor_test_acc": float(np.mean(donor_acc["test"])),
               "host_val_kl_to_donor": host_val_kl, "box_layers": box_layers, "host_stub_layers": stub_layers,
               "write_sets": {m: sorted(ws) for m, ws in write_sets.items()}, "arms": {}}
    host_acc = None
    for name, arm in arms.items():
        if budget_left() < 600:
            print(f"budget: stopping before {name}", flush=True)
            summary["stopped_before"] = name
            break
        ta = time.time()
        mod, tinfo = None, None
        if arm["route"] == "tile":
            mod = Tiles(args.seed + 1)
            if arm["loss"]:
                tinfo = train(name, arm, mod, args.tile_lr)
        elif arm["route"] == "bridge":
            mod = Bridge(arm["mode"], arm["src"])
            tinfo = train(name, arm, mod, args.bridge_lr)
        rs = [generate(name, arm, mod, it) for it in test]
        for r in rs:
            rows_f.write(json.dumps(r) + "\n")
        rows_f.flush()
        acc = float(np.mean([r["correct"] for r in rs]))
        if name == "host":
            host_acc = acc
        kv, dvv = val_scores(arm, mod)
        nrs = [r["norm_ratio"] for r in rs if r["norm_ratio"] is not None]
        cfs = [r["clamped_frac"] for r in rs if r["clamped_frac"] is not None]
        gap = summary["donor_test_acc"] - (host_acc if host_acc is not None else acc)
        e = {"test_acc": acc,
             "gap_closed": (acc - host_acc) / gap if host_acc is not None and abs(gap) > 1e-9 else None,
             "val_kl": kv, "val_kl_closed": (host_val_kl - kv) / host_val_kl if host_val_kl > 0 else None,
             "eng_drift_test": drift_test(arm, mod) if name != "host" else 0.0, "eng_drift_val": dvv,
             "norm_ratio_mean": float(np.mean(nrs)) if nrs else None,
             "clamped_frac": float(np.mean(cfs)) if cfs else None,
             "trainable_params": nparams(mod) if mod is not None else 0, "wall_s": round(time.time() - ta, 1)}
        if tinfo:
            e |= {k: tinfo[k] for k in ("selected_epoch", "val_kl_selected", "within_drift_cap")}
        summary["arms"][name] = e
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        print(f"{time.strftime('%H:%M:%S')} {name}: acc {acc:.3f} gap closed {e['gap_closed']} val KL {kv:.4f} "
              f"drift {e['eng_drift_test']} norm ratio {e['norm_ratio_mean']} clamped {e['clamped_frac']}",
              flush=True)
        save_receipt(stage=f"done {name}")
        mx.clear_cache()
    save_receipt(status="complete" if "stopped_before" not in summary else "partial (budget)", stage="done",
                 summary=summary)
    print(json.dumps({n: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items()}
                      for n, e in summary["arms"].items()}, indent=1), flush=True)


if __name__ == "__main__":
    main()
