"""Amoeba milestone M0: one box (geo.capitals), in the donor only.

Subcommands (each writes atomic JSON receipts into --out and refuses a
nonempty directory, as the repo's campaign rules require):

  window1  baseline evals, donor-generated world-knowledge corpus,
           per-layer TopK skip transcoders for layers 6-18
  window2  attribution-based box extraction, the absorb loop with resample
           ablation, and the necessity / specificity / sufficiency tests

Run them inside an owned GPU window (see owned.py). CPU smoke: --smoke.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from research.amoeba import evals, transcoders as T  # noqa: E402
from research.amoeba.donor import Donor  # noqa: E402

DATA = ROOT / "output/amoeba-m0/data"
LAYERS = list(range(6, 19))


def atomic(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


def source_rev():
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def tokenizer(donor):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(donor.path))


# ---------------------------------------------------------------- window 1
SUBJECTS_EXTRA = """the Roman Empire;photosynthesis;the French Revolution;black holes;the honey bee;the violin;volcanoes;the printing press;
the Amazon rainforest;the Great Wall of China;penicillin;the Olympic Games;chess;the Moon;the Nile;the Industrial Revolution;
the human heart;electricity;the Silk Road;dinosaurs;the Renaissance;the Pacific Ocean;coffee;the bicycle;earthquakes;the internet;
the Ottoman Empire;the Sahara;jazz;the steam engine;glaciers;the Aztec Empire;vaccines;the telescope;the Mediterranean Sea;rice;
the Cold War;the Himalayas;democracy;the elephant;the Byzantine Empire;the Danube;tea;the Mongol Empire;the Alps;the Andes;
the European Union;the United Nations;the Atlantic Ocean;cities of Europe;cities of Asia;cities of Africa;cities of South America;
world geography;the history of trade;famous rivers;famous mountains;island nations;the Caribbean;Scandinavia;the Balkans;Central Asia;
the Middle East;West Africa;East Africa;Southeast Asia;Oceania;the Baltic states;Latin America;North America""".replace("\n", "")


def gen_corpus(donor, tok, spec, out: Path, n_per=20, new=280, bs=64):
    countries = sorted({it["country"] for v in spec["items"].values() for it in v} |
                       {it["country"] for it in spec["control_topic"]["items"]})
    subjects = countries + [s.strip() for s in SUBJECTS_EXTRA.split(";") if s.strip()]
    frames = ["Encyclopedia article: {s}\n\n", "{s}\n\nOverview\n", "A short history of {s}.\n\n",
              "Travel guide: {s}\n\n", "Notes on {s}:\n"]
    rng = random.Random(3)
    prompts = []
    for s in subjects:
        for f in rng.sample(frames, k=min(len(frames), max(1, n_per // 4))):
            prompts.append(f.format(s=s))
    rng.shuffle(prompts)
    texts, t0 = [], time.time()
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        ids = [tok(p, add_special_tokens=False)["input_ids"] for p in chunk]
        gen = donor.generate(ids, new, temperature=0.8, top_p=0.95, seed=i, eos={151643, 151645})
        for p, g in zip(chunk, gen):
            texts.append(p + tok.decode(g, skip_special_tokens=True))
    with open(out, "w") as f:
        for t in texts:
            f.write(json.dumps({"text": t}) + "\n")
    n_tok = sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts)
    return {"samples": len(texts), "tokens": n_tok, "subjects": len(subjects), "seconds": time.time() - t0}


class Mix:
    """Training mix of fixed-length token windows; the last 2,000 rows of each file are held out."""
    WEIGHTS = {"donor_world.jsonl": 0.15, "tinystories_train.jsonl": 0.30, "billsum.jsonl": 0.20,
               "scifact.jsonl": 0.20, "gsm8k.jsonl": 0.15}

    def __init__(self, tok, T=512, seed=0, world="donor_world.jsonl", max_rows=40000):
        self.T, self.rng = T, random.Random(seed)
        self.streams = {}
        for name in self.WEIGHTS:
            src = DATA / (world if name == "donor_world.jsonl" else name)
            rows = [json.loads(l)["text"] for l in src.read_text().splitlines()]
            rows = rows[:-2000] if len(rows) > 6000 else rows
            ids = []
            for r in rows[:max_rows]:
                ids += tok(r, add_special_tokens=False)["input_ids"] + [151643]
            self.streams[name] = np.array(ids, dtype=np.int32)
        self.names = list(self.WEIGHTS)
        self.p = np.array([self.WEIGHTS[n] for n in self.names])

    def batch(self, B):
        out = []
        for n in self.rng.choices(self.names, weights=self.p, k=B):
            s = self.streams[n]
            o = self.rng.randrange(0, len(s) - self.T - 1)
            out.append(s[o:o + self.T])
        return mx.array(np.stack(out))

    def sizes(self):
        return {n: int(len(s)) for n, s in self.streams.items()}


def train_transcoders(donor, mix, out: Path, *, F=8192, k=32, steps=2500, B=8, lr=3e-4, log=None):
    tcs = {i: T.Transcoder(donor.d, F, k) for i in LAYERS}
    first = mix.batch(B)
    _, caps, _ = donor.forward(first, record=set(LAYERS), stop_after=max(LAYERS))
    for i, (x, y) in caps.items():
        tcs[i].sx = mx.sqrt(mx.mean(mx.sum(x * x, -1))) / donor.d ** 0.5
        tcs[i].sy = mx.sqrt(mx.mean(mx.sum(y * y, -1))) / donor.d ** 0.5
    trainers = {i: T.make_trainer(tcs[i], lr)[0] for i in LAYERS}
    fired = {i: mx.zeros((F,)) for i in LAYERS}
    hist, t0 = [], time.time()
    for s in range(steps):
        ids = mix.batch(B)
        _, caps, _ = donor.forward(ids, record=set(LAYERS), stop_after=max(LAYERS))
        losses = {}
        for i in LAYERS:
            x, y = caps[i]
            x, y = x.reshape(-1, donor.d), y.reshape(-1, donor.d)
            loss, a = trainers[i](x, y)
            if s >= steps - 500:
                fired[i] = fired[i] + mx.sum(a > 0, axis=0)
            losses[i] = loss
        mx.eval(losses, [t.parameters() for t in tcs.values()], fired)
        if s % 250 == 0 or s == steps - 1:
            row = {"step": s, "sec": round(time.time() - t0, 1), "mse": {i: float(v) for i, v in losses.items()}}
            hist.append(row)
            if log:
                log(row)
    # held-out FVU on mix windows not seen in order (fresh rng) and on the off-topic slice
    ev = Mix.__new__(Mix); ev.__dict__.update(mix.__dict__); ev.rng = random.Random(999)
    ids = ev.batch(8)
    _, caps, _ = donor.forward(ids, record=set(LAYERS), stop_after=max(LAYERS))
    held = {}
    for i in LAYERS:
        x, y = (c.reshape(-1, donor.d) for c in caps[i])
        y_hat, a = tcs[i](x)
        held[i] = {"fvu": float(T.fvu(y, y_hat)), "l0": float(mx.mean(mx.sum(a > 0, -1))),
                   "dead_frac_last500": float(mx.mean(fired[i] == 0))}
    meta = {"layers": LAYERS, "d": donor.d, "F": F, "k": k, "steps": steps, "batch_tokens": B * mix.T,
            "train_tokens": steps * B * mix.T, "lr": lr, "donor": donor.artifact, "mix": Mix.WEIGHTS,
            "mix_stream_tokens": mix.sizes(), "heldout": held}
    T.save(tcs, out / "transcoders", meta)
    return meta, hist


def window1(args):
    out = Path(args.out)
    donor = Donor()
    tok = tokenizer(donor)
    spec = evals.load_topic()
    rec = {"schema": "amoeba.m0.window1.v1", "donor": donor.artifact, "source_revision": source_rev(),
           "started": time.time(), "device": str(mx.default_device())}
    # 1. baseline
    base = {split: evals.topic_accuracy(donor, evals.topic_prompts(spec, split, tok)) for split in ("extract", "heldout")}
    base["control_language"] = evals.topic_accuracy(donor, evals.topic_prompts(spec, None, tok, control=True))
    base["chance"] = spec["chance"]; base["control_chance"] = spec["control_topic"]["chance"]
    off = evals.offtopic_texts(DATA, tok)
    base["offtopic_nll"] = evals.offtopic_loss(donor, off)
    rec["baseline"] = base
    atomic(out / "window1.json", rec)
    print("baseline", json.dumps(base), flush=True)
    # 2. donor-generated corpus
    world = DATA / ("donor_world_smoke.jsonl" if args.smoke else "donor_world.jsonl")
    if not world.exists() or args.smoke:
        rec["corpus"] = gen_corpus(donor, tok, spec, world, new=24 if args.smoke else 280,
                                   bs=8 if args.smoke else 64)
    else:
        rec["corpus"] = {"reused": str(world)}
    atomic(out / "window1.json", rec)
    print("corpus", rec["corpus"], flush=True)
    # 3. transcoders
    mix = Mix(tok, world=world.name, max_rows=500 if args.smoke else 40000)
    logf = open(out / "train_log.jsonl", "a")
    meta, hist = train_transcoders(donor, mix, out, steps=20 if args.smoke else args.steps,
                                   B=2 if args.smoke else 8,
                                   log=lambda r: (logf.write(json.dumps(r) + "\n"), logf.flush(),
                                                  print("train", r["step"], r["sec"], flush=True)))
    rec["transcoders"] = {k: v for k, v in meta.items() if k != "mix_stream_tokens"} | {"path": str(out / "transcoders")}
    rec["finished"] = time.time()
    atomic(out / "window1.json", rec)
    print("done", json.dumps(meta["heldout"])[:2000], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["window1", "window2"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tc", help="transcoder directory (window2)")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    owned = os.environ.get("AMOEBA_OWNED_OUT")
    owned_files = {"desktop-lease.json", "paired-locks.json"}
    if out.exists() and any(p.name not in owned_files and not p.name.startswith(".") for p in out.iterdir()) \
            and not (owned and Path(owned).resolve() == out and not (out / "window1.json").exists()
                     and not (out / "window2.json").exists()):
        raise SystemExit("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    if args.cpu:
        mx.set_default_device(mx.cpu)
    if args.cmd == "window1":
        window1(args)
    else:
        from research.amoeba import extract
        extract.window2(args)


if __name__ == "__main__":
    main()
