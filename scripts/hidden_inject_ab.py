#!/usr/bin/env python3
"""Hidden-state concept injection A/B (Pierre 2026-11-22): does adding an AdS/CFT concept vector to the residual
stream change the *quality of the reasoning* of a thinking model, not just its final answer?

Unqualified research. The model is loaded through its mlx2 adapter (Flash-Next: the native-MTP route's adapter-owned
defaults, PLE offloaded); generation is the ordinary cached decode (the model's own make_cache, one token per step),
so the no-injection arm IS the ordinary-decode reference. Nothing here changes a route or a default.

Mechanism. One decoder layer L is patched (instance class swap) so that its output h_L gets + delta. FN's residual is
the full hyper-connection stream [B, T, hc*D]; delta lives in that space. Position 0 (the attention sink) is never
touched. Modes:
  all     delta at every prompt position and at every decoded token (persistent steering; no KV slot used)
  prompt  delta only at prompt positions during prefill; decode steps run unmodified, so the concept reaches later
          tokens only through the K/V that the prompt positions wrote (one-shot injection)

Concept vector (build stage). Positive texts: 2026 arXiv hep-th/gr-qc abstracts that are about AdS/CFT / holography
(strict regex, holographic dark energy excluded), CC-BY / BY-SA / CC0 licences only. Negative texts: hep-th abstracts
with no holography / AdS / string / duality terms, same register. v_L = mean_doc(mean_tok h_L | pos) -
mean_doc(mean_tok h_L | neg), tokens 0-3 skipped. The difference isolates the holography content, not "physics
abstract" style. Random control: a Gaussian direction with ||v_L|| (seeded).

Calibration (self-gating, before any generation). Grid layers x rho, rho = ||delta|| / median token ||h_L||.
Teacher-forced NLL (nats/token) on held-out positive abstracts, held-out negative abstracts and generic English
(core corpus v2), for the concept and the random direction. Selected setting: the largest specificity
(random dNLL_pos - concept dNLL_pos) among settings with dNLL_general <= --gen-cap and dNLL_neg <= --neg-cap. If none
passes the caps the smallest rho at the best layer is used and the run is flagged; if specificity is below
--min-specificity the run is flagged "no_specific_effect" (generation still runs: it is the experiment).
Hard gates (stop the run): hook parity (patched layer with no delta must give bit-identical logits), finite vectors.

Generation arms (paired common random numbers: the same seed per (prompt, sample) in every arm):
  base            no injection (ordinary decode reference)
  concept_all     concept vector, mode all
  concept_prompt  concept vector, mode prompt
  random_all      norm-matched random direction, mode all (any-perturbation control)
  primer          no injection; four of the positive abstracts as background reading in a system message (the
                  text-in-context version of the same source material; costs KV)
Order is sample-major (all greedy rows first) so a partial night is still balanced. Resumable: rows already in
gens.jsonl are skipped. --stop-at stops cleanly between generations (exit 75).

Report stage (CPU, no model): per-row automatic metrics and blinded pairwise judging packs (base vs each arm, order
randomized, key kept apart). Judging is done afterwards by fresh-context Claude subagents (see the README).

Stages: build | calibrate | generate | report | all (= build, calibrate, generate, report).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

A = Path.home() / "Library/Application Support/mlx2/research-artifacts/amoeba"
OAI = A / "downloads/mathphys26-oai-20261118"
CORE = A / "core-corpus-v2-9b64b3791/corpus.jsonl"
PROMPTS = ROOT / "docs/research/hidden-inject/adscft-prompts.jsonl"
ARMS = ("base", "concept_all", "concept_prompt", "random_all", "primer")
CC = ("creativecommons.org/licenses/by/", "creativecommons.org/licenses/by-sa/", "creativecommons.org/publicdomain/zero/")
POS = re.compile(r"AdS/CFT|AdS_?\{?\$?\d|anti-de Sitter|Ryu[- ]Takayanagi|holographic entanglement|quantum extremal "
                 r"surface|bulk reconstruction|entanglement wedge|holographic dual|gauge/gravity|holographic CFT|HKLL|"
                 r"island formula", re.I)
NEG_EXCL = re.compile(r"holograph|\bAdS(?=\b|[0-9_])|anti-de Sitter|dS/CFT|celestial|duality|string|brane", re.I)
LEAK = re.compile(r"\bAdS(?=\b|[0-9_])|holograph|Ryu[- ]Takayanagi|bulk|boundary CFT|anti-de Sitter|black hole|entanglement wedge|"
                  r"CFT", re.I)
SKIP = 4                       # leading tokens excluded from means and NLL (attention-sink positions)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def h8(s):
    return hashlib.sha256(s.encode()).hexdigest()


def atomic_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.rename(path)


def git_rev():
    import subprocess
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=9", "HEAD"], capture_output=True, text=True)
    d = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                       capture_output=True, text=True)
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", "--",
                              str(Path(__file__).resolve().relative_to(ROOT))], capture_output=True)
    return r.stdout.strip(), bool(r.returncode or d.returncode or tracked.returncode or d.stdout.strip())


# ------------------------------------------------------------------ corpora


def load_corpora(n_pos, n_neg, n_gen, seed):
    """Deterministic selections. Returns dict of lists of {id, text, license?}."""
    pos, neg, seen = [], [], set()
    for f in ("physics_hep-th.jsonl", "physics_gr-qc.jsonl"):
        for line in open(OAI / f):
            r = json.loads(line)
            if r["id"] in seen:
                continue
            t, a = (r.get("title") or "").strip(), (r.get("abstract") or "").strip()
            s = f"{t} {a}"
            if not a or "dark energy" in s.lower():
                continue
            lic = r.get("license") or ""
            if POS.search(s):
                if any(c in lic for c in CC):
                    seen.add(r["id"])
                    pos.append({"id": r["id"], "title": t, "text": f"{t}\n\n{a}", "license": lic})
            elif f == "physics_hep-th.jsonl" and not NEG_EXCL.search(s):
                seen.add(r["id"])
                neg.append({"id": r["id"], "title": t, "text": f"{t}\n\n{a}", "license": lic})
    gen = []
    for line in open(CORE):
        r = json.loads(line)
        if r.get("register") in ("textbook", "narrative", "howto", "expository", "dialogue") and len(r["text"]) > 800:
            gen.append({"id": h8(r["text"])[:12], "text": r["text"][:2400], "register": r.get("register")})
    rng = random.Random(seed)
    for xs in (pos, neg, gen):
        xs.sort(key=lambda r: h8(f"{seed}:{r['id']}"))
    # held split: hash-sorted, the first 20% held (abstract ids, so it is stable across runs)
    def split(xs, n):
        k = max(8, len(xs) // 5)
        return xs[k:k + n], xs[:k]
    pos_b, pos_h = split(pos, n_pos)
    neg_b, neg_h = split(neg, n_neg)
    rng.shuffle(gen)
    return {"pos_build": pos_b, "pos_held": pos_h, "neg_build": neg_b, "neg_held": neg_h, "gen_held": gen[:n_gen],
            "counts": {"pos_all": len(pos), "neg_all": len(neg), "gen_all": len(gen)}}


def primer_text(pos_build, k=4):
    rows = sorted(pos_build, key=lambda r: (len(r["text"]), r["id"]))[:k]
    body = "\n\n".join(f"[{i + 1}] {r['text']}" for i, r in enumerate(rows))
    return ("Background reading (recent arXiv abstracts on holography and AdS/CFT):\n\n" + body,
            [r["id"] for r in rows])


# ------------------------------------------------------------------ model + hook


class Model:
    def __init__(self, path, cpu=False, watchdog_gb=None):
        import mlx.core as mx
        from mlx2.adapters.registry import inspect_model
        from research.amoeba.carve.host import find_inner
        if cpu:
            mx.set_default_device(mx.cpu)
        if watchdog_gb:
            import amoeba27b_lib as AL
            AL.footprint_watchdog(watchdog_gb, log)
        self.path = Path(path)
        t0 = time.time()
        atype = inspect_model(self.path).adapter_type
        kw = {}
        if atype.__name__ == "FlashNextAdapter":
            from research.amoeba.det_backend_fn import production_adapter_policy
            kw["execution_policy"] = production_adapter_policy()
        self.adapter = atype(str(self.path), **kw)
        self.fn = atype.__name__ == "FlashNextAdapter"
        if self.fn:
            from mlx2.runtime.models import qwen4_exp as Q
            if not getattr(self.adapter, "_tables", None) or not os.environ.get("MLX_QWEN4_PLE_NVME"):
                raise SystemExit("Flash-Next must run with its PLE table offloaded (production profile)")
            if Q._SHAPE_STABLE_SHORT_FORWARD:
                raise SystemExit("shape-stable short forward splits prefill per token: prompt-mode injection "
                                 "would be wrong; refusing")
        self.model, self.tok = self.adapter.model, self.adapter.tokenizer
        self.model.eval()
        _, inner = find_inner(self.model)
        self.layers = inner.layers
        self.L = len(self.layers)
        fp = self.adapter.identity["fingerprint"]
        self.fingerprint = fp
        self.checkpoint = f"{self.path.name}@{fp.split(':')[-1][:8]}"
        self.delta = None          # (layer, mx.array [W]) or None
        self.mode = "off"
        self.record = None         # {layer: [per-forward outputs]} when recording
        self._patched = {}         # layer index -> original class
        log(f"loaded {self.checkpoint} via {atype.__name__}: {self.L} layers in {time.time() - t0:.0f}s, "
            f"MLX active {mx.get_active_memory() / 2**30:.1f} GiB")

    def patch(self, i):
        if i in self._patched:
            return
        layer = self.layers[i]
        base = type(layer)
        owner = self

        def __call__(self_, x, *a, **k):
            out = base.__call__(self_, x, *a, **k)
            return owner._apply(i, out)

        layer.__class__ = type(base.__name__ + "HiddenInject", (base,), {"__call__": __call__})
        self._patched[i] = base

    def unpatch_all(self):
        for i, base in self._patched.items():
            self.layers[i].__class__ = base
        self._patched = {}

    def _apply(self, i, out):
        import mlx.core as mx
        if self.record is not None and i in self.record:
            self.record[i].append(out)
        if self.delta is None or self.delta[0] != i or self.mode == "off":
            return out
        T = out.shape[1]
        if self.mode == "prompt" and T == 1:
            return out
        d = self.delta[1].astype(out.dtype)
        if T == 1:
            return out + d
        return mx.concatenate([out[:, :1], out[:, 1:] + d], axis=1)

    def width(self):
        rec = {0: []}
        self.patch(0)
        self.record = rec
        try:
            self.logits(self.encode("width probe"))
        finally:
            self.record = None
        return int(rec[0][0].shape[-1])

    # text
    def encode(self, s):
        return list(self.tok.encode(s, add_special_tokens=False))

    def decode(self, ids):
        return self.tok.decode(list(ids))

    def prompt_ids(self, messages):
        return list(self.adapter.prompt_tokens({"messages": messages, "enable_thinking": True}))

    def eos(self):
        out = set()
        for name in ("eos_token_ids", "eos_token_id"):
            v = getattr(self.tok, name, None)
            if v is not None:
                out |= {int(x) for x in (v if isinstance(v, (list, tuple, set)) else [v])}
        for s in ("<|im_end|>", "<|endoftext|>"):
            ids = self.encode(s)
            if len(ids) == 1:
                out.add(int(ids[0]))
        return out

    # forward
    def logits(self, ids):
        import mlx.core as mx
        cache = self.model.make_cache()
        lg = self.model(mx.array([ids], dtype=mx.int32), cache=cache)
        mx.eval(lg)
        return lg

    def nll(self, ids):
        """Mean next-token NLL (nats) over targets at positions >= SKIP+1."""
        import mlx.core as mx
        lg = self.logits(ids)[0, :-1].astype(mx.float32)
        tgt = mx.array(ids[1:], dtype=mx.int32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        nl = -mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0]
        nl = nl[SKIP:]
        v = float(nl.mean().item())
        del lg, lp
        mx.clear_cache()
        return v, int(nl.shape[0])

    def capture(self, ids, layers):
        """Per layer: (mean over tokens >= SKIP of h_L as float32 numpy, median token norm)."""
        import mlx.core as mx
        import numpy as np
        for l in layers:
            self.patch(l)
        self.record = {l: [] for l in layers}
        try:
            self.logits(ids)
            out = {}
            for l in layers:
                h = self.record[l][0][0, SKIP:].astype(mx.float32)
                m = h.mean(axis=0)
                n = mx.linalg.norm(h, axis=-1)
                mx.eval(m, n)
                out[l] = (np.array(m), float(np.median(np.array(n))))
        finally:
            self.record = None
            mx.clear_cache()
        return out

    def generate(self, ids, max_new, temperature, top_p, seed, eos, stop_at=None):
        import mlx.core as mx
        import numpy as np
        rng = np.random.default_rng(seed)
        cache = self.model.make_cache()
        t0 = time.time()
        lg = self.model(mx.array([ids], dtype=mx.int32), cache=cache)
        mx.eval(lg)
        t_prefill = time.time() - t0
        out, finish = [], "length"
        t1 = time.time()
        for _ in range(max_new):
            row = lg[0, -1].astype(mx.float32)
            if temperature <= 0:
                t = int(mx.argmax(row).item())
            else:
                # top-p over the top-1024 candidates (exact unless their mass is below top_p: then full sort)
                x = row / temperature
                lse = float(mx.logsumexp(x).item())
                cand = mx.argpartition(-x, kth=1023)[:1024]
                xc, ic = np.array(x[cand]), np.array(cand)
                order = np.argsort(-xc)
                xs, ids_ = xc[order], ic[order]
                p = np.exp(xs - lse)
                if p.sum() < top_p:
                    xf = np.array(x)
                    ids_ = np.argsort(-xf)
                    p = np.exp(xf[ids_] - lse)
                keep = np.cumsum(p) - p < top_p
                p = p[keep] / p[keep].sum()
                t = int(ids_[rng.choice(len(p), p=p)])
            out.append(t)
            if t in eos:
                finish = "eos"
                break
            lg = self.model(mx.array([[t]], dtype=mx.int32), cache=cache)
        t_dec = time.time() - t1
        del cache, lg
        mx.clear_cache()
        return out, finish, t_prefill, t_dec


# ------------------------------------------------------------------ stages


def tok_doc(m, text, max_tokens):
    return m.encode(text)[:max_tokens]


def stage_build(m, out, a):
    import numpy as np
    vec_path = out / "vectors.npz"
    if vec_path.exists():
        log("build: vectors.npz exists, reusing")
        return
    C = load_corpora(a.n_pos, a.n_neg, a.n_gen, a.seed)
    atomic_json(out / "corpora.json", {k: (v if k == "counts" else [{kk: r[kk] for kk in r if kk != "text"} | {
        "sha256": h8(r["text"])} for r in v]) for k, v in C.items()})
    (out / "corpora_text.json").write_text(json.dumps({k: v for k, v in C.items() if k != "counts"}))
    layers = a.layers
    acc = {s: {l: [] for l in layers} for s in ("pos", "neg")}
    norms = {l: [] for l in layers}
    t0 = time.time()
    for s, rows in (("pos", C["pos_build"]), ("neg", C["neg_build"])):
        for j, r in enumerate(rows):
            cap = m.capture(tok_doc(m, r["text"], a.max_doc_tokens), layers)
            for l in layers:
                acc[s][l].append(cap[l][0])
                if s == "neg":
                    norms[l].append(cap[l][1])
            if j % 40 == 0:
                log(f"build {s} {j}/{len(rows)} {time.time() - t0:.0f}s")
    rs = np.random.default_rng(a.seed)
    arrs = {}
    for l in layers:
        v = np.mean(acc["pos"][l], axis=0) - np.mean(acc["neg"][l], axis=0)
        if not np.all(np.isfinite(v)):
            raise SystemExit(f"non-finite concept vector at layer {l}")
        r = rs.standard_normal(v.shape).astype(np.float32)
        r *= np.linalg.norm(v) / np.linalg.norm(r)
        arrs[f"concept_{l}"] = v.astype(np.float32)
        arrs[f"random_{l}"] = r
        arrs[f"hnorm_{l}"] = np.array(float(np.median(norms[l])), dtype=np.float32)
        # how separable: cosine of per-doc means to v, pos vs neg
        vp = [float(np.dot(x, v) / (np.linalg.norm(x) * np.linalg.norm(v) + 1e-9)) for x in acc["pos"][l]]
        vn = [float(np.dot(x, v) / (np.linalg.norm(x) * np.linalg.norm(v) + 1e-9)) for x in acc["neg"][l]]
        log(f"layer {l}: |v| {np.linalg.norm(v):.3f}, median |h| {np.median(norms[l]):.2f}, "
            f"|v|/|h| {np.linalg.norm(v) / np.median(norms[l]):.4f}, cos(doc, v) pos {np.mean(vp):.3f} "
            f"neg {np.mean(vn):.3f}")
    tmp = out / "vectors.tmp.npz"
    np.savez(tmp, **arrs)
    tmp.rename(vec_path)
    log(f"build done in {time.time() - t0:.0f}s")


def set_delta(m, vecs, kind, layer, rho, mode):
    import mlx.core as mx
    import numpy as np
    if kind is None:
        m.delta, m.mode = None, "off"
        return
    v = vecs[f"{kind}_{layer}"]
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm == 0:
        raise SystemExit(f"invalid {kind} direction at layer {layer}")
    v = v / norm * (rho * float(vecs[f"hnorm_{layer}"]))
    m.patch(layer)
    m.delta, m.mode = (layer, mx.array(v)), mode


def stage_calibrate(m, out, a):
    import numpy as np
    cal_path = out / "calib.json"
    if cal_path.exists():
        log("calibrate: calib.json exists, reusing")
        return json.loads(cal_path.read_text())
    vecs = dict(np.load(out / "vectors.npz"))
    C = json.loads((out / "corpora_text.json").read_text())
    sets = {"pos": C["pos_held"][:a.n_eval], "neg": C["neg_held"][:a.n_eval], "gen": C["gen_held"][:a.n_eval]}
    docs = {k: [tok_doc(m, r["text"], a.max_doc_tokens) for r in v] for k, v in sets.items()}

    def evalset():
        res = {}
        for k, ds in docs.items():
            tot, n = 0.0, 0
            for ids in ds:
                v, c = m.nll(ids)
                tot += v * c
                n += c
            res[k] = tot / n
        return res

    # hook parity: the unpatched model vs every candidate layer patched with a zero delta (the concat path the
    # injection uses) must give bit-identical logits
    import mlx.core as mx
    probe = docs["gen"][0][:64]
    W = m.width()
    m.unpatch_all()
    m.delta, m.mode = None, "off"
    ref = m.logits(probe)
    for l in a.layers:
        m.patch(l)
    m.delta, m.mode = (a.layers[len(a.layers) // 2], mx.zeros((W,))), "all"
    got = m.logits(probe)
    parity = float(mx.max(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))).item())
    m.delta, m.mode = None, "off"
    log(f"hook parity (unpatched vs patched + zero delta) max_abs {parity}")
    if parity != 0.0:
        raise SystemExit(f"hook parity failed: max_abs {parity}")
    t0 = time.time()
    base = evalset()
    log(f"base NLL {base}")
    grid = []
    for l in a.layers:
        for rho in a.rhos:
            row = {"layer": l, "rho": rho}
            for kind in ("concept", "random"):
                set_delta(m, vecs, kind, l, rho, "all")
                r = evalset()
                row[kind] = {k: r[k] - base[k] for k in r}
            set_delta(m, vecs, None, 0, 0, "off")
            row["specificity"] = row["random"]["pos"] - row["concept"]["pos"]
            row["pass_caps"] = row["concept"]["gen"] <= a.gen_cap and row["concept"]["neg"] <= a.neg_cap
            grid.append(row)
            log(f"L{l} rho {rho}: concept dpos {row['concept']['pos']:+.4f} dneg {row['concept']['neg']:+.4f} "
                f"dgen {row['concept']['gen']:+.4f} | random dpos {row['random']['pos']:+.4f} dgen "
                f"{row['random']['gen']:+.4f} | spec {row['specificity']:+.4f} caps {row['pass_caps']} "
                f"({time.time() - t0:.0f}s)")
    passing = [g for g in grid if g["pass_caps"]]
    flags = []
    if passing:
        sel = max(passing, key=lambda g: g["specificity"])
    else:
        best_l = max(grid, key=lambda g: g["specificity"])["layer"]
        sel = min((g for g in grid if g["layer"] == best_l), key=lambda g: g["rho"])
        flags.append("no_setting_within_caps")
    if a.force_layer is not None:
        sel = next(g for g in grid if g["layer"] == a.force_layer and g["rho"] == a.force_rho)
        flags.append("forced_setting")
        if not sel["pass_caps"]:
            flags.append("forced_setting_outside_caps")
    if sel["specificity"] < a.min_specificity:
        flags.append("no_specific_effect")
    res = {"base": base, "grid": grid, "selected": {"layer": sel["layer"], "rho": sel["rho"]}, "flags": flags,
           "caps": {"gen": a.gen_cap, "neg": a.neg_cap, "min_specificity": a.min_specificity},
           "hook_parity_max_abs": parity, "n_eval": a.n_eval, "seconds": time.time() - t0}
    atomic_json(cal_path, res)
    log(f"selected L{sel['layer']} rho {sel['rho']} flags {flags}")
    return res


def load_prompts(path):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate prompt id")
    return rows


def stage_generate(m, out, a, ident):
    import numpy as np
    cal = json.loads((out / "calib.json").read_text())
    vecs = dict(np.load(out / "vectors.npz"))
    C = json.loads((out / "corpora_text.json").read_text())
    primer, primer_ids = primer_text(C["pos_build"])
    L, rho = cal["selected"]["layer"], cal["selected"]["rho"]
    atomic_json(out / "primer.json", {"abstract_ids": primer_ids, "text_sha256": h8(primer), "text": primer})
    prompts = load_prompts(a.prompts)
    gpath = out / "gens.jsonl"
    done = set()
    if gpath.exists():
        for line in gpath.read_text().splitlines():
            r = json.loads(line)
            done.add((r["prompt_id"], r["arm"], r["sample"]))
    eos = m.eos()
    arm_cfg = {"base": (None, "off"), "concept_all": ("concept", "all"), "concept_prompt": ("concept", "prompt"),
               "random_all": ("random", "all"), "primer": (None, "off")}
    arms = [x for x in a.arms]
    nmax = max(r.get("samples", 3) for r in prompts)
    todo = [(s, p, arm) for s in range(nmax) for p in prompts if s < p.get("samples", 3) for arm in arms
            if (p["id"], arm, s) not in done]
    log(f"generate: L{L} rho {rho}, {len(done)} rows done, {len(todo)} to do")
    with open(gpath, "a") as fh:
        for k, (s, p, arm) in enumerate(todo):
            if a.stop_at and time.time() >= a.stop_at:
                log("stop-at reached; clean stop")
                return 75
            msgs = list(p["messages"])
            if arm == "primer":
                msgs = [{"role": "system", "content": primer}] + msgs
            ids = m.prompt_ids(msgs)
            kind, mode = arm_cfg[arm]
            set_delta(m, vecs, kind, L, rho, mode)
            temp = 0.0 if s == 0 else a.temperature
            seed = int(h8(f"{p['id']}:{s}")[:8], 16)
            toks, finish, tp, td = m.generate(ids, a.max_new, temp, a.top_p, seed, eos)
            set_delta(m, vecs, None, 0, 0, "off")
            text = m.decode(toks)
            row = {"prompt_id": p["id"], "category": p["category"], "arm": arm, "sample": s, "temperature": temp,
                   "seed": seed, "layer": L if kind else None, "rho": rho if kind else None, "mode": mode,
                   "prompt_tokens": len(ids), "new_tokens": len(toks), "finish": finish, "prefill_s": tp,
                   "decode_s": td, "decode_tok_s": len(toks) / td if td > 0 else None, "text": text,
                   "checkpoint": ident["checkpoint"], "source_revision": ident["source_revision"]}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            log(f"[{k + 1}/{len(todo)}] {p['id']} {arm} s{s}: {len(toks)} tok {finish} "
                f"{row['decode_tok_s'] or 0:.1f} tok/s")
    return 0


# ------------------------------------------------------------------ report (CPU)


def split_think(text):
    if "</think>" in text:
        t, ans = text.split("</think>", 1)
        return t.replace("<think>", "").strip(), ans.strip(), True
    return text.replace("<think>", "").strip(), "", False


def distinct4(text):
    w = text.split()
    g = [tuple(w[i:i + 4]) for i in range(max(0, len(w) - 3))]
    return len(set(g)) / len(g) if g else 1.0


SELF_CORR = re.compile(r"\b(wait|actually|hmm|let me re(check|consider|think)|on second thought|that's not right|"
                       r"I made an error|correction)\b", re.I)


def stage_report(out, a):
    prompts = {p["id"]: p for p in load_prompts(a.prompts)}
    rows = [json.loads(x) for x in (out / "gens.jsonl").read_text().splitlines() if x.strip()]
    per = []
    for r in rows:
        p = prompts[r["prompt_id"]]
        think, ans, closed = split_think(r["text"])
        full = think + "\n" + ans
        rub = p.get("rubric") or []
        hits_any = [lab for lab, rx in rub if re.search(rx, full)]
        hits_ans = [lab for lab, rx in rub if re.search(rx, ans)]
        m = {k: r[k] for k in ("prompt_id", "category", "arm", "sample", "new_tokens", "finish", "decode_tok_s")}
        m.update(think_words=len(think.split()), answer_words=len(ans.split()), think_closed=closed,
                 distinct4_think=round(distinct4(think), 4), self_corrections=len(SELF_CORR.findall(think)),
                 rubric_n=len(rub), rubric_hits_any=len(hits_any), rubric_hits_answer=len(hits_ans),
                 rubric_labels=hits_any, leak_terms=len(LEAK.findall(full)) if p["category"] == "offtopic" else None)
        if p.get("answer_regex"):
            m["correct"] = bool(re.search(p["answer_regex"], ans or think[-600:], re.I))
        per.append(m)
    atomic_json(out / "metrics_rows.json", per)
    # aggregate
    agg = {}
    for m in per:
        key = f"{m['arm']}|{m['category']}"
        g = agg.setdefault(key, {"n": 0})
        g["n"] += 1
        for f in ("new_tokens", "think_words", "answer_words", "distinct4_think", "self_corrections",
                  "decode_tok_s"):
            if m.get(f) is not None:
                g.setdefault(f, []).append(m[f])
        if m["rubric_n"]:
            g.setdefault("rubric_frac_any", []).append(m["rubric_hits_any"] / m["rubric_n"])
            g.setdefault("rubric_frac_answer", []).append(m["rubric_hits_answer"] / m["rubric_n"])
        if "correct" in m:
            g.setdefault("correct", []).append(1.0 if m["correct"] else 0.0)
        if m["leak_terms"] is not None:
            g.setdefault("leak_terms", []).append(m["leak_terms"])
        g.setdefault("finished", []).append(1.0 if m["finish"] == "eos" else 0.0)
    summ = {k: {f: (round(sum(v) / len(v), 4) if isinstance(v, list) else v) for f, v in g.items()}
            for k, g in sorted(agg.items())}
    atomic_json(out / "metrics_summary.json", summ)
    for k, g in summ.items():
        log(k, g)
    # blinded pairwise packs: base vs each other arm, same (prompt, sample)
    byk = {(r["prompt_id"], r["arm"], r["sample"]): r for r in rows}
    rng = random.Random(a.seed)
    pairs, key = [], []
    for (pid, arm, s), r in sorted(byk.items()):
        if arm == "base" or (pid, "base", s) not in byk:
            continue
        b = byk[(pid, "base", s)]
        flip = rng.random() < 0.5
        one, two = (r, b) if flip else (b, r)
        pair_id = h8(f"{a.seed}:{pid}:{arm}:{s}")[:10]
        pairs.append({"pair_id": pair_id, "prompt_id": pid, "question": prompts[pid]["messages"][-1]["content"],
                      "response_1": one["text"], "response_2": two["text"]})
        key.append({"pair_id": pair_id, "prompt_id": pid, "sample": s, "arm": arm,
                    "response_1": one["arm"], "response_2": two["arm"]})
    (out / "judge").mkdir(exist_ok=True)
    rng.shuffle(pairs)
    (out / "judge/pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs))
    atomic_json(out / "judge/key.json", key)
    log(f"report: {len(per)} rows, {len(pairs)} blinded pairs -> {out / 'judge'}")
    return 0


# ------------------------------------------------------------------ main


def resume_inputs(a, out):
    """Inputs bound to each reusable artifact. Arms and the stop time can change on resume."""
    inputs = {
        "vectors.npz": {k: getattr(a, k) for k in
                        ("seed", "layers", "n_pos", "n_neg", "n_gen", "max_doc_tokens", "cpu")},
        "calib.json": {k: getattr(a, k) for k in
                       ("rhos", "n_eval", "gen_cap", "neg_cap", "min_specificity", "force_layer", "force_rho")},
    }
    if a.stage in ("generate", "all") or (out / "gens.jsonl").exists():
        inputs["gens.jsonl"] = {"prompts_sha256": hashlib.sha256(Path(a.prompts).read_bytes()).hexdigest(),
                                "max_new": a.max_new, "temperature": a.temperature, "top_p": a.top_p}
    return inputs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("build", "calibrate", "generate", "report", "all"))
    ap.add_argument("--model", default=str(Path.home() / "mlx-models/Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default=str(PROMPTS))
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--watchdog-gb", type=float, default=None)
    ap.add_argument("--memory-gb", type=float, default=None, help="mx.set_memory_limit")
    ap.add_argument("--layers", type=lambda s: [int(x) for x in s.split(",")], default=[12, 18, 24, 30])
    ap.add_argument("--rhos", type=lambda s: [float(x) for x in s.split(",")], default=[0.05, 0.1, 0.2, 0.4])
    ap.add_argument("--n-pos", type=int, default=160)
    ap.add_argument("--n-neg", type=int, default=160)
    ap.add_argument("--n-gen", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--max-doc-tokens", type=int, default=320)
    ap.add_argument("--gen-cap", type=float, default=0.03)
    ap.add_argument("--neg-cap", type=float, default=0.10)
    ap.add_argument("--min-specificity", type=float, default=0.005)
    ap.add_argument("--force-layer", type=int, default=None)
    ap.add_argument("--force-rho", type=float, default=None)
    ap.add_argument("--arms", type=lambda s: s.split(","), default=list(ARMS))
    ap.add_argument("--max-new", type=int, default=5000)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--stop-at", type=float, default=None, help="epoch seconds; clean stop between generations")
    ap.add_argument("--seed", type=int, default=20261122)
    a = ap.parse_args()
    bad = [x for x in a.arms if x not in ARMS]
    if bad:
        raise SystemExit(f"unknown arms {bad}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run_path = out / "run.json"
    prev = json.loads(run_path.read_text()) if run_path.exists() else None
    if a.stage == "report":
        expected = (prev or {}).get("resume_inputs", {}).get("gens.jsonl", {}).get("prompts_sha256")
        if expected and expected != hashlib.sha256(Path(a.prompts).read_bytes()).hexdigest():
            raise SystemExit("report prompts differ from the generated prompts")
        return stage_report(out, a)
    inputs = resume_inputs(a, out)
    for artifact, values in inputs.items():
        if (out / artifact).exists() and (prev or {}).get("resume_inputs", {}).get(artifact) != values:
            raise SystemExit(f"refusing to reuse {artifact} with unverified or different inputs")
    rev, dirty = git_rev()
    if dirty and not a.cpu:
        raise SystemExit("refusing: dirty tree (every result names its mlx2 commit)")
    import mlx.core as mx
    if a.memory_gb:
        mx.set_memory_limit(int(a.memory_gb * 2**30))
    m = Model(a.model, cpu=a.cpu, watchdog_gb=a.watchdog_gb)
    ident = {"checkpoint": m.checkpoint, "fingerprint": m.fingerprint, "source_revision": rev, "dirty": dirty,
             "model_path": str(m.path), "args": vars(a), "status": "running", "started": time.time(),
             "qualification": "unqualified research; not a route or default change"}
    ident["resume_inputs"] = inputs
    ident["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if prev and (prev.get("fingerprint") != m.fingerprint or prev["checkpoint"] != m.checkpoint
                 or prev["source_revision"] != rev or prev.get("runner_sha256") != ident["runner_sha256"]):
        raise SystemExit(f"refusing to resume: {out} belongs to {prev['checkpoint']} @ {prev['source_revision']}")
    atomic_json(run_path, ident)
    rc = 0
    if a.stage in ("build", "all"):
        stage_build(m, out, a)
    if a.stage in ("calibrate", "all"):
        stage_calibrate(m, out, a)
    if a.stage in ("generate", "all"):
        rc = stage_generate(m, out, a, ident)
    ident["status"] = "complete" if rc == 0 else "stopped"
    ident["finished"] = time.time()
    atomic_json(run_path, ident)
    if a.stage == "all" and rc == 0:
        stage_report(out, a)
    return rc


if __name__ == "__main__":
    sys.exit(main())
