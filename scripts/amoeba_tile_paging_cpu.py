#!/usr/bin/env python3
"""Tile-paging feasibility on Qwen3-0.6B, CPU (research, unqualified).

Three questions for a tile-paged image (docs/research/2026-11-15-amoeba-tile-paging.md):

a. Locality. Per token and domain (code, English, French, arithmetic, capitals):
   - the hard-gate open fraction of each learned per-token box (capitals, arithmetic, french from
     the z075 tiled-ids set), per (token, layer) and per token in any layer;
   - for router tiles (no per-token gate), per-token *need*: the next-token loss gain of adding the
     tile to the core, and the fraction of tokens whose gain from the tiles left out of a
     preselected set exceeds a threshold (a would-be miss).
b. Quality. Arms on code (teacher-forced loss on mlx2 source at a pinned commit, and 20 functional
   tasks, benchmarks/amoeba_code/tasks.jsonl): full, core, core+prog, core+random (same per-layer
   counts, 3 seeds), core+rest (non-core minus prog), core+boxes, core+prog+boxes. English loss as
   the control.

Masking is zero ablation (the 27B partition's rule) through a hook on every layer's SwiGLU MLP:
used = m * act, m the arm's 0/1 neuron mask (+ hard box gates per token). The core is the z075
English core with layers 0-2 whole (masks_core_full3.pt, the sink rule). The prog tile is built
here from the topic arrays: the non-core neurons with the highest fit-split d for node ``prog``,
as many as the 27B partition's prog tile holds of its non-core (15.4%).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ART = Path.home() / "Library/Application Support/mlx2/research-artifacts/amoeba"
CORE = ROOT / "docs/research/amoeba-cone-width-2026-11-12/qwen3-0.6b-sink-export-5b32bb9ef/masks_core_full3.pt"
BOXES = ART / "Qwen3-0.6B-amoeba-z075-tiled-ids-9c52b2be9"
TOPICS = ART / "topic-arrays-qwen3-0.6b-280655b53"
CORPUS = ART / "corpus-2026-11-13/corpus.json"
PROG_FRAC_OF_NONCORE = 79543 / (1114112 - 598052)      # the 27B partition's prog tile share of its non-core
NEED_NATS = 0.5

FRENCH = [
    "Le petit chat dort sur le canapé pendant que la pluie tombe doucement sur la ville.",
    "Nous avons décidé de partir tôt demain matin pour éviter les embouteillages sur l'autoroute.",
    "La boulangerie au coin de la rue vend le meilleur pain que j'aie jamais goûté.",
    "Elle a ouvert la fenêtre et a regardé les enfants qui jouaient dans le jardin.",
    "Il faut absolument que tu lises ce livre avant la réunion de vendredi prochain.",
    "Mon grand-père racontait toujours des histoires de son enfance à la campagne.",
    "Les étudiants ont travaillé toute la nuit pour terminer leur projet à temps.",
    "Pourriez-vous m'indiquer le chemin le plus court pour aller à la gare, s'il vous plaît ?",
    "Après le dîner, nous nous sommes promenés le long de la rivière jusqu'au vieux pont.",
    "Le musée sera fermé lundi en raison des travaux de rénovation de la grande salle.",
    "Je ne comprends pas pourquoi le train est encore en retard ce matin.",
    "Ils ont planté des tomates, des courgettes et des haricots dans leur potager.",
]
CAPITALS = [("France", "Paris"), ("Japan", "Tokyo"), ("Kenya", "Nairobi"), ("Peru", "Lima"), ("Canada", "Ottawa"),
            ("Egypt", "Cairo"), ("Norway", "Oslo"), ("Chile", "Santiago"), ("Vietnam", "Hanoi"), ("Poland", "Warsaw"),
            ("Ghana", "Accra"), ("Portugal", "Lisbon"), ("Thailand", "Bangkok"), ("Austria", "Vienna"),
            ("Argentina", "Buenos Aires"), ("Greece", "Athens")]


def git(*a, cwd=ROOT):
    return subprocess.run(["git", *a], capture_output=True, text=True, cwd=cwd, check=True).stdout


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def code_files(rev, per_ext=4):
    """Four files per language from mlx2's tree at rev, chosen by SHA-256 of the path (fixed)."""
    paths = git("ls-tree", "-r", "--name-only", rev).split("\n")
    out = []
    for ext in (".py", ".sh", ".metal", ".swift"):
        cand = [p for p in paths if p.endswith(ext) and p.split("/")[0] in ("src", "scripts", "apps")
                and "amoeba" not in p and "tile_paging" not in p]
        cand.sort(key=lambda p: hashlib.sha256(p.encode()).hexdigest())
        out += cand[:per_ext]
    return out


def strip_header(text):
    lines = text.split("\n")
    i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith(("#", "//", "/*", "*", "*/"))):
        i += 1
    return "\n".join(lines[i:])


def code_seqs(tok, rev, n_tok):
    seqs, src = [], []
    for p in code_files(rev):
        text = strip_header(git("show", f"{rev}:{p}"))
        ids = tok(text, add_special_tokens=False)["input_ids"][:n_tok]
        if len(ids) >= 64:
            seqs.append(ids)
            src.append({"path": p, "tokens": len(ids)})
    return seqs, src


def domain_seqs(tok, rev, n_tok, corpus):
    code, src = code_seqs(tok, rev, n_tok)
    c = json.loads(Path(corpus).read_text())
    eng = [s[:n_tok] for k in ("repo_markdown", "tinystories_train") for s in c["seqs"][k][-8:]]
    fr = [tok(s, add_special_tokens=False)["input_ids"] for s in FRENCH]
    rng = np.random.default_rng(7)
    ar = []
    for _ in range(12):
        lines = [f"{a} + {b} = {a + b}" for a, b in rng.integers(10, 999, size=(6, 2))]
        ar.append(tok("\n".join(lines), add_special_tokens=False)["input_ids"])
    caps = [tok(" ".join(f"The capital of {k} is {v}." for k, v in CAPITALS[i:i + 4]), add_special_tokens=False)["input_ids"]
            for i in range(0, len(CAPITALS), 4)]
    return {"code": code, "english": eng, "french": fr, "arithmetic": ar, "capitals": caps}, src


class Hook:
    """Every layer's MLP: used = m * act, m = arm mask [L, N] + sum of hard-gated boxes."""

    def __init__(self, model):
        self.model = model
        self.mask = None
        self.boxes = []          # (name, mask [L,N], W [L,d], b [L])
        self.gate_log = None     # dict name -> list of [T, L] bools
        self.region_log = None   # dict region -> list of per-layer [T] norms
        self.regions = None
        mlps = [layer.mlp for layer in model.model.layers]
        for l, m in enumerate(mlps):
            m.forward = self._make(m, l)

    def _make(self, m, l):
        def fwd(x):
            act = torch.nn.functional.silu(m.gate_proj(x)) * m.up_proj(x)
            if self.region_log is not None:
                W = m.down_proj.weight
                for name, R in self.regions.items():
                    self.region_log[name].append(((act * R[l]) @ W.T).norm(dim=-1)[0])
            if self.mask is None and not self.boxes:
                if self.gate_log is not None:
                    self._gates(x, l)
                return m.down_proj(act)
            mk = self.mask[l].to(act.dtype).expand_as(act).clone()
            for name, bm, W, b in self.boxes:
                g = (torch.sigmoid((x * W[l]).sum(-1, keepdim=True) + b[l]) > 0.5).to(act.dtype)
                mk = torch.maximum(mk, bm[l] * g)
            if self.gate_log is not None:
                self._gates(x, l)
            return m.down_proj(act * mk)
        return fwd

    def _gates(self, x, l):
        for name, _, W, b in self.all_boxes:
            g = torch.sigmoid((x * W[l]).sum(-1) + b[l]) > 0.5
            self.gate_log[name].append(g[0])


def load_boxes(L, N, d):
    from safetensors.torch import load_file
    t = load_file(str(BOXES / "tiles.safetensors"))
    out = []
    for name in ("capitals", "arithmetic", "french"):
        bm = torch.zeros(L, N)
        W = torch.zeros(L, d)
        b = torch.zeros(L)
        for l in range(L):
            ids = t[f"box.{name}.{l}.donor_neuron_ids"].long()
            bm[l, ids] = 1
            W[l] = t[f"box.{name}.{l}.gate_w"].float()
            b[l] = t[f"box.{name}.{l}.gate_b"].float()[0]
        out.append((name, bm, W, b))
    return out


def prog_tile(core, seed_counts=3):
    nodes = json.loads((TOPICS / "neuron_d_nodes.json").read_text())
    k = nodes.index("prog")
    d = np.asarray(np.load(TOPICS / "neuron_d_fit.npy", mmap_mode="r")[k], np.float32)
    non = (core.numpy() < 0.5)
    K = int(round(PROG_FRAC_OF_NONCORE * non.sum()))
    score = np.where(non, d, -np.inf).ravel()
    sel = np.zeros(score.size, bool)
    sel[np.argsort(-score)[:K]] = True
    prog = torch.from_numpy(sel.reshape(core.shape).astype(np.float32))
    rand = []
    for s in range(seed_counts):
        rng = np.random.default_rng(s)
        r = np.zeros(core.shape, np.float32)
        for l in range(core.shape[0]):
            pool = np.flatnonzero(non[l])
            r[l, rng.choice(pool, int(prog[l].sum()), replace=False)] = 1
        rand.append(torch.from_numpy(r))
    return prog, rand, {"K": K, "noncore": int(non.sum()), "d_min_selected": float(np.sort(score)[-K])}


@torch.no_grad()
def nll_rows(model, ids):
    x = torch.tensor([ids])
    lg = model(x).logits[0, :-1].float()
    lp = torch.log_softmax(lg, -1)
    tgt = torch.tensor(ids[1:])
    return -lp[torch.arange(len(tgt)), tgt], lp


@torch.no_grad()
def generate(model, tok, prompt, max_new=448):
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def body_until_stop(text):
    lines, out = text.split("\n"), []
    for ln in lines:
        if ln and not ln[0].isspace():
            break
        out.append(ln)
    return "\n".join(out)


def run_tests(prompt, body, tests, timeout=10):
    src = prompt + body + "\n\n" + tests + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
    try:
        r = subprocess.run([sys.executable, "-I", f.name], capture_output=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(f.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--n-tok", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--functional-arms", default="full,core,core+prog,core+rand0,core+rest,core+prog+boxes")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to overwrite non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    rev = git("rev-parse", "--short=9", args.rev).strip()
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    snap = snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True)
    tok = AutoTokenizer.from_pretrained(snap)
    model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.float32).eval()
    L, N, d = model.config.num_hidden_layers, model.config.intermediate_size, model.config.hidden_size
    ckpt = f"Qwen/Qwen3-0.6B@{Path(snap).name[:8]}"
    hook = Hook(model)
    core = torch.load(CORE, map_location="cpu", weights_only=False)["core"].float()
    boxes = load_boxes(L, N, d)
    hook.all_boxes = boxes
    prog, rand, prog_info = prog_tile(core)
    rest = ((1 - core) * (1 - prog)).clamp(0, 1)
    allbox = torch.stack([b[1] for b in boxes]).amax(0)
    arms = {"full": (None, []), "core": (core, []), "core+prog": (core + prog, []),
            **{f"core+rand{i}": (core + r, []) for i, r in enumerate(rand)},
            "core+rest": (core + rest, []), "core+boxes": (core, boxes), "core+prog+boxes": (core + prog, boxes)}
    doms, src = domain_seqs(tok, rev, args.n_tok, CORPUS)
    log = open(out / "run.log", "w")

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    say(f"checkpoint {ckpt}  mlx2 {rev}  core kept {float(core.mean()):.4f}  prog {prog_info}  "
        f"box neurons {[int(b[1].sum()) for b in boxes]} (in core: {[int((b[1] * core).sum()) for b in boxes]})")
    t0 = time.time()
    # ---- per-token losses for every arm and domain, and full-model gate / region logs
    per_tok = {}
    full_lp = {}
    for arm, (mask, bx) in arms.items():
        hook.mask, hook.boxes = mask, bx
        per_tok[arm] = {}
        for dom, seqs in doms.items():
            rows = []
            for i, ids in enumerate(seqs):
                nll, lp = nll_rows(model, ids)
                rows.append(nll.numpy())
                if arm == "full":
                    full_lp[(dom, i)] = lp
                elif dom == "code":
                    pass
            per_tok[arm][dom] = rows
        say(f"arm {arm}: " + "  ".join(f"{dm} {np.mean(np.concatenate(r)):.3f}" for dm, r in per_tok[arm].items())
            + f"  ({time.time() - t0:.0f}s)")
    # agreement and KL on code against the full model
    agree = {}
    for arm, (mask, bx) in arms.items():
        hook.mask, hook.boxes = mask, bx
        a, kls = [], []
        for i, ids in enumerate(doms["code"]):
            _, lp = nll_rows(model, ids)
            fl = full_lp[("code", i)]
            a.append((lp.argmax(-1) == fl.argmax(-1)).float().numpy())
            kls.append((fl.exp() * (fl - lp)).sum(-1).numpy())
        agree[arm] = {"top1_agree_vs_full": float(np.mean(np.concatenate(a))), "kl_full_arm": float(np.mean(np.concatenate(kls)))}
    # locality: gates and regions, full model
    hook.mask, hook.boxes = None, []
    regions = {"core": core, "prog": prog, "rest": rest}
    loc = {}
    for dom, seqs in doms.items():
        opens = {b[0]: [] for b in boxes}
        shares = {k: [] for k in regions}
        for ids in seqs:
            hook.gate_log = {b[0]: [] for b in boxes}
            hook.region_log = {k: [] for k in regions}
            hook.regions = regions
            nll_rows(model, ids)
            for name, g in hook.gate_log.items():
                opens[name].append(torch.stack(g, 1).numpy())          # [T, L]
            norms = {k: torch.stack(v, 1).numpy() for k, v in hook.region_log.items()}   # [T, L]
            tot = sum(norms.values())
            for k in regions:
                shares[k].append((norms[k] / np.maximum(tot, 1e-9)).mean(1))
        hook.gate_log = hook.region_log = None
        rec = {}
        for name, g in opens.items():
            G = np.concatenate([x[1:] for x in g], 0)       # drop position 0 of every sequence
            rec[f"gate_{name}"] = {"open_token_layer": float(G.mean()), "open_any_layer_token": float(G.any(1).mean()),
                                   "requests_never_open": float(np.mean([not x[1:].any() for x in g]))}
        for k, v in shares.items():
            rec[f"output_norm_share_{k}"] = float(np.mean(np.concatenate(v)))
        # router-tile need per token (next-token loss gain)
        c = np.concatenate(per_tok["core"][dom])
        cp = np.concatenate(per_tok["core+prog"][dom])
        cr = np.concatenate(per_tok["core+rest"][dom])
        f = np.concatenate(per_tok["full"][dom])
        cpb = np.concatenate(per_tok["core+prog+boxes"][dom])
        rec["need_prog_frac"] = float(np.mean(c - cp > NEED_NATS))
        rec["need_rest_frac"] = float(np.mean(c - cr > NEED_NATS))
        rec["miss_if_core+prog_frac"] = float(np.mean(cp - f > NEED_NATS))
        rec["miss_if_core+prog+boxes_frac"] = float(np.mean(cpb - f > NEED_NATS))
        rec["miss_if_core_frac"] = float(np.mean(c - f > NEED_NATS))
        rec["tokens"] = int(c.size)
        loc[dom] = rec
        say(f"locality {dom}: " + json.dumps(rec))
    # quality table
    qual = {}
    for arm in arms:
        q = {dm: float(np.mean(np.concatenate(per_tok[arm][dm]))) for dm in doms}
        q.update({f"dce_{dm}": q[dm] - float(np.mean(np.concatenate(per_tok["full"][dm]))) for dm in doms})
        q.update(agree[arm])
        qual[arm] = q
    # functional
    tasks = [json.loads(x) for x in (ROOT / "benchmarks/amoeba_code/tasks.jsonl").read_text().splitlines() if x.strip()]
    func = {}
    for arm in args.functional_arms.split(","):
        hook.mask, hook.boxes = arms[arm]
        res = []
        for t in tasks:
            body = body_until_stop(generate(model, tok, t["prompt"]))
            res.append({"id": t["id"], "pass": run_tests(t["prompt"], body, t["tests"]), "body": body})
        func[arm] = {"pass": sum(r["pass"] for r in res), "of": len(res), "items": res}
        say(f"functional {arm}: {func[arm]['pass']}/{len(res)}  ({time.time() - t0:.0f}s)")
    result = {"schema": "mlx2.amoeba-tile-paging-cpu.v1", "checkpoint": ckpt, "mlx2_commit": rev,
              "status": "research, unqualified", "device": "cpu fp32",
              "inputs": {"core_masks": f"{CORE.relative_to(ROOT)}@{sha256(CORE)[:12]}",
                         "boxes": f"{BOXES.name}@{sha256(BOXES / 'tiles.safetensors')[:12]}",
                         "topic_arrays": TOPICS.name, "corpus": f"{CORPUS.name}@{sha256(CORPUS)[:12]}",
                         "code_files_rev": rev, "code_files": src, "tasks": "benchmarks/amoeba_code/tasks.jsonl@"
                         + sha256(ROOT / "benchmarks/amoeba_code/tasks.jsonl")[:12]},
              "ablation": "zero", "need_threshold_nats": NEED_NATS, "prog_tile": prog_info,
              "kept_fraction": {a: (None if m is None else float(m.clamp(0, 1).mean())) for a, (m, _) in arms.items()},
              "quality": qual, "locality": loc, "functional": func, "seconds": time.time() - t0}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    say("wrote", out / "result.json")


if __name__ == "__main__":
    main()
