#!/usr/bin/env python3
"""Calculator tile on Qwen3 (CPU): probe layer sweep, false-fire, controls, generation arms, manifest.

Usage: PYTHONPATH=$PWD/src:$PWD python scripts/amoeba_det_calc_cpu.py --out output/amoeba-det-tiles/calc-0.6b
Every result names the donor checkpoint (<repo>@<snapshot>) and the mlx2 commit.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

mx.set_default_device(mx.cpu)

from mlx2.runtime import amoeba_det_tiles as D  # noqa: E402
from research.amoeba import det_calc as C  # noqa: E402
from research.amoeba.donor import Donor  # noqa: E402

CORPUS = Path.home() / "Library/Application Support/mlx2/research-artifacts/amoeba/corpus-2026-11-13/corpus.json"
EOS = {151643, 151645}


def git_rev():
    rev = subprocess.check_output(["git", "rev-parse", "--short=9", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src", "research", "scripts"], cwd=ROOT).returncode != 0
    return rev + ("-dirty" if dirty else "")


REENCODE = None


ENGLISH_N = None


def english(split):
    blob = json.loads(CORPUS.read_text())["seqs"]
    rng = {"train": (0, 40), "val": (40, 70), "test": (70, 130)}[split]
    if ENGLISH_N is not None:
        rng = (rng[0], rng[0] + ENGLISH_N)
    f = REENCODE or (lambda ids: ids)
    return {d: [f(x) for x in blob[d][rng[0]:rng[1]]] for d in blob}


def mult_items():
    """The 27B multiplication task (amoeba27b_lib, seed 0): 160 pairs, held half."""
    rng2 = random.Random(4242)
    pairs = set()
    while len(pairs) < 160:
        pairs.add((rng2.randint(20, 999), rng2.randint(11, 99)))
    pairs = sorted(pairs)
    idx = list(range(len(pairs)))
    random.Random(0 + len("multiplication")).shuffle(idx)
    return [pairs[i] for i in sorted(idx[80:])]


def chained_items(n=40, seed=515):
    """Held-out chained expressions (precedence, brackets, unary minus): the calc-parse-v2 task.

    Integer results only; templates differ from the probe's training templates."""
    rng = random.Random(seed)
    forms = [
        lambda a, b, c: (f"{a} + {b} * {c}", a + b * c),
        lambda a, b, c: (f"({a} + {b}) * {c}", (a + b) * c),
        lambda a, b, c: (f"{a} * {b} - {c}", a * b - c),
        lambda a, b, c: (f"({a} - {b}) * {c} + {a}", (a - b) * c + a),
        lambda a, b, c: (f"{a * c} / {c} + {b}", a + b),
        lambda a, b, c: (f"-{a} + {b} * {c}", -a + b * c),
    ]
    temps = [
        lambda e: (f"We need {e} =", " "),
        lambda e: (f"So the total is <<{e.replace(' ', '')}=", ""),
        lambda e: (f"Question: what is {e}?\nAnswer:", " "),
        lambda e: (C.CHAT.format(q=f"What is {e}? Reply with the number only."), ""),
    ]
    out = []
    while len(out) < n:
        a, b, c = rng.randint(2, 99), rng.randint(2, 99), rng.randint(2, 30)
        e, v = rng.choice(forms)(a, b, c)
        tmpl = rng.choice(temps)
        if "<<" in tmpl(e)[0] and "-" in e:
            continue                           # GSM8K writes negatives rarely; keep that form plain
        prompt, _ = tmpl(e)
        out.append((prompt, str(v)))
    return out


def whole_number_present(text, exp):
    """The expected number appears as a whole number (not inside a longer one), commas stripped:
    "36" passes on "3 times 12 is **36**", fails on "360" and on "36.5". A sign is part of the number: "-36"
    (ASCII hyphen or U+2212) is not 36, while the subtraction echo "50-36" still shows 36."""
    t = text.replace(",", "")
    e = re.escape(exp.replace(",", ""))
    # a minus right before the number negates it unless a digit precedes the minus (an operator, "50-36")
    return re.search(rf"(?<![\d.])(?<!(?<!\d)[-\u2212]){e}(?!\d|\.\d)", t) is not None


def first_number(text):
    m = re.search(r"-?\$?\d[\d,]*(?:\.\d+)?", text)
    return None if m is None else m.group(0).replace("$", "").replace(",", "")


def generate(donor, ts, tok, tile_id, prompt, arm, *, max_new=10, push_scale=0.0, gate_override=None):
    ids = tok.encode(prompt, add_special_tokens=False)
    start = len(ids)
    proc = None
    if arm in ("force", "constrain", "random_force", "shuffled_force"):
        proc = D.CalculatorProcessor(ts, tile_id, tok, mode="constrain" if arm == "constrain" else "force",
                                     gate_override=gate_override, prompt_length=start)
    plan, pos = None, 0
    ts.enabled = arm != "host"
    for _ in range(max_new):
        ts.push_rows.clear()
        pushing_first = False
        if arm == "push":
            if plan is not None:
                ts.push_rows[tile_id] = push_scale * PUSH_DIRS[plan[pos]][None]
                ts.push_gated[tile_id] = False
            elif plan is None:
                ans, _ = D.calculator_answer(tok.decode(ids[-D.CONTEXT_TOKENS:]), truncated=len(ids) > D.CONTEXT_TOKENS,
                                             parser=next(x for x in ts.tiles if x.id == tile_id).parser)
                if ans is not None:
                    target = tok.encode(ans, add_special_tokens=False)
                    ts.push_rows[tile_id] = push_scale * PUSH_DIRS[target[0]][None]
                    ts.push_gated[tile_id] = True
                    pushing_first = True
        logits, _, _ = donor.forward(mx.array([ids]), mlp_hook=ts.mlp_hook if ts.enabled else None)
        lg = logits[:, -1].astype(mx.float32)
        if proc is not None:
            proc.bind_row(0)
            lg = proc(mx.array(ids), lg)
        t = int(mx.argmax(lg[0]).item())
        if arm == "push":
            if pushing_first and ts.calc_gate_row(tile_id, 0):
                plan, pos = target, 0
                ts.counters["calc"]["plans"] += 1
            if plan is not None:
                if t != plan[pos]:
                    ts.counters["calc"]["plan_aborted"] += 1
                    plan = None
                else:
                    pos += 1
                    if pos >= len(plan):
                        plan = None
        ids.append(t)
        if t in EOS or "\n" in tok.decode([t]) and len(ids) - start > 1:
            break
    ts.push_rows.clear()
    return tok.decode(ids[start:]), (proc.events if proc else [])


PUSH_DIRS = None


def main():
    global PUSH_DIRS
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="4,8,12,16,20,24")
    ap.add_argument("--n-pos", type=int, default=600)
    ap.add_argument("--n-neg", type=int, default=60)
    ap.add_argument("--gen-items", type=int, default=80)
    ap.add_argument("--english-n", type=int, default=None, help="smoke: English sequences per domain and split")
    ap.add_argument("--parser", default="calc-parse-v1", choices=("calc-parse-v1", "calc-parse-v2"))
    ap.add_argument("--arms", default="host,force,constrain,push@0.5,push@1,push@2,random_force,shuffled_force")
    ap.add_argument("--backend", choices=("donor", "27b", "fn", "hy4"), default="donor",
                    help="hy4: Hy4-flash through its adapter (ordinary route, stream-mean layer outputs)")
    ap.add_argument("--execution-policy", type=json.loads, default=None,
                    help='hy4: adapter policy (default {"engram_table_format": "iq4_nl"}, the served base)')
    ap.add_argument("--footprint-limit-gb", type=float, default=104.0,
                    help="hy4: abort when this process's footprint exceeds this (Hy4 loaded and wired is ~104-106 GB)")
    ap.add_argument("--min-available-gb", type=float, default=15.0,
                    help="hy4: watchdog floor on machine available memory (default 15)")
    ap.add_argument("--available-counter", choices=("vm_stat", "serving"), default="vm_stat",
                    help="hy4: available-memory counter for the floor: serving = the serving admission's "
                         "estimate (file-backed and purgeable pages reclaimable), vm_stat = free+inactive+"
                         "speculative+purgeable (the historical watchdog)")
    ap.add_argument("--read-point", default="mlp_input", choices=D.READ_POINTS,
                    help="manifest read point; the fn backend captures layer outputs (layer_output_capture)")
    ap.add_argument("--skip-generation", action="store_true",
                    help="probe fit, controls and manifest only (served A/B measures generation)")
    ap.add_argument("--host-tiles", default=None, help="27b: amoeba tile manifest the host serves (core+boxes)")
    ap.add_argument("--host-layout", default="fused")
    ap.add_argument("--model", default=None, help="27b: model path (default the 27B q8 build; a slice for smoke tests)")
    ap.add_argument("--device", choices=("gpu", "cpu"), default="gpu", help="27b backend device")
    ap.add_argument("--probe-only", action="store_true", help="Save calibrated probe and held-out diagnostics without generation arms")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty {out}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.backend == "fn":
        if args.read_point != "layer_output_capture" or not args.skip_generation:
            raise SystemExit("--backend fn needs --read-point layer_output_capture --skip-generation")
        from research.amoeba.det_backend_fn import BackendFN
        mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
        donor = BackendFN(path=args.model)
        src_fp, host_sha = donor.source_fingerprint, None
    elif args.backend == "hy4":
        if args.read_point != "layer_output_capture" or not args.skip_generation:
            raise SystemExit("--backend hy4 needs --read-point layer_output_capture --skip-generation")
        from research.amoeba.det_backend_hy4 import BackendHy4
        mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
        donor = BackendHy4(path=args.model, policy=args.execution_policy, wire=args.device == "gpu",
                           min_available_gb=args.min_available_gb, available_counter=args.available_counter,
                           footprint_limit_gb=args.footprint_limit_gb)
        src_fp, host_sha = donor.source_fingerprint, None
        # The positive templates' chat string: the host's own template (no_think), not Qwen's.
        if donor.chat_format is None:
            raise SystemExit(f"{donor.path} renders no chat template")
        C.CHAT = donor.chat_format
    elif args.backend == "27b":
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
    global REENCODE, EOS, ENGLISH_N
    ENGLISH_N = args.english_n
    REENCODE = getattr(donor, "reencode", None)
    EOS = {i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>"))
           if isinstance(i, int) and i >= 0}
    layers = [int(x) for x in args.layers.split(",")]
    rep = {"schema": "mlx2.amoeba-det-calc-cpu.v1", "checkpoint": donor.artifact, "mlx2_source_revision": git_rev(),
           "device": args.device if args.backend in ("27b", "fn", "hy4") else "cpu",
           "dtype": "host-native" if args.backend in ("27b", "fn", "hy4") else "float32", "qualification": "unqualified", "args": vars(args)}
    if getattr(donor, "receipt", None):
        rep["host"] = donor.receipt

    # ---- data -----------------------------------------------------------------------------
    def build(split, seed):
        seqs, labs, kinds = [], [], []
        for p, a in C.positive_rows(split, args.n_pos if split == "train" else args.n_pos // 2, seed):
            ids, lab = C.encode_positive(tok, p, a)
            seqs.append(ids); labs.append(lab); kinds.append("arith")
        for kind, rows in C.negative_rows(split, args.n_neg, seed + 1).items():
            for r in rows:
                ids = tok.encode(r, add_special_tokens=False)
                seqs.append(ids); labs.append([0] * len(ids)); kinds.append(kind)
        for dom, rows in english(split).items():
            for ids in rows:
                seqs.append(list(ids)); labs.append([0] * len(ids)); kinds.append("english:" + dom)
        return seqs, labs, kinds

    data = {s: build(s, seed) for s, seed in (("train", 1), ("val", 2), ("test", 3))}
    # Features stream to <out>/features/<split>/layerNNN.npy as they are captured (resident feature
    # memory is one sequence's), and every later step reads the read-only memmaps.
    feats = {s: C.capture_to_disk(donor, data[s][0], layers, out / "features" / s,
                                  log=lambda m: print(m, flush=True)) for s in data}
    rep["features"] = {"storage": "npy memmap per split and layer, float32", "dir": str(out / "features")}
    if args.backend == "hy4":
        rep["memory_watch"] = {"min_available_gb": args.min_available_gb, "footprint_limit_gb": args.footprint_limit_gb,
                               "available_counter": args.available_counter,
                               # live watchdog record (peak footprint, lowest available seen)
                               "watch": getattr(donor, "watchdog", None)}
    if args.skip_generation and callable(getattr(donor, "close", None)):
        # The fit, controls and manifest read only the features: free the model before them.
        donor.close()
        rep["model_unloaded_before_fit"] = True
        print(f"model unloaded, MLX active {mx.get_active_memory() / 1e9:.1f} GB", flush=True)
    lab ={s: np.array([x for l in data[s][1] for x in l]) for s in data}
    kind_pos = {s: np.array([k for k, l in zip(data[s][2], data[s][1]) for _ in l]) for s in data}
    rep["data"] = {s: {"sequences": len(data[s][0]), "positions": int(lab[s].size), "positives": int((lab[s] == 1).sum())}
                   for s in data}
    print("captured", rep["data"], f"{time.time() - t0:.0f}s", flush=True)

    def neg_groups(split, p):
        groups = {}
        for k in sorted(set(kind_pos[split])):
            m = (kind_pos[split] == k) & (lab[split] == 0)
            if m.any():
                groups["arith_nonanswer" if k == "arith" else k] = p[m]
        return groups

    # ---- probe sweep (val chooses) ---------------------------------------------------------
    sweep = {}
    for L in layers:
        w, b = C.train_probe(feats["train"][L], lab["train"])
        pv = C.probs(feats["val"][L], w, b)
        pick = C.pick_threshold(pv[lab["val"] == 1], neg_groups("val", pv))
        sweep[L] = {"w": w, "b": b, "pick": pick,
                    "positive_quantiles": np.quantile(pv[lab["val"] == 1], [0, .5, .9, 1]).tolist(),
                    "negative_maxima": {k: float(v.max()) for k, v in neg_groups("val", pv).items()}}
        print("layer", L, None if pick is None else (pick[0], round(pick[1], 4), round(max(pick[2].values()), 4)), flush=True)
    rep["sweep_val"] = {str(L): (None if v["pick"] is None else {"threshold": v["pick"][0], "recall": v["pick"][1],
                                                                  "worst_ff": max(v["pick"][2].values())})
                        for L, v in sweep.items()}
    admissible = {L: v for L, v in sweep.items() if v["pick"] is not None}
    if not admissible:
        rep["calibration_failure"] = {str(L): {k: v[k] for k in ("positive_quantiles", "negative_maxima")}
                                      for L, v in sweep.items()}
        rep["wall_s"] = time.time() - t0
        (out / "report.json").write_text(json.dumps(rep, indent=1, default=float))
        raise ValueError("No probe meets the validation false-fire bound; refusing to create a tile")
    L = max(admissible, key=lambda l: (admissible[l]["pick"][1], -l))
    w, b, th = sweep[L]["w"], sweep[L]["b"], sweep[L]["pick"][0]
    rep["chosen"] = {"layer": L, "threshold": th, "rule": "max val recall s.t. val false-fire <= 1% on every negative group"}

    # ---- test: probe-only, parse-only, combined ---------------------------------------------
    def tf_table(split, w, b, th, gate_fn=None):
        seqs, labs, kinds = data[split]
        p = C.probs(feats[split][L], w, b)
        off = 0
        stats = {}
        for ids, lb, k in zip(seqs, labs, kinds):
            for i in range(len(ids)):
                g = bool(p[off + i] > th) if gate_fn is None else gate_fn()
                if lb[i] < 0:
                    continue
                text = tok.decode(ids[max(0, i + 1 - D.CONTEXT_TOKENS):i + 1])
                parsed = D.calculator_answer(text, truncated=i + 1 > D.CONTEXT_TOKENS, parser=args.parser)[0] is not None
                key = ("arith_answer" if lb[i] == 1 else "arith_nonanswer") if k == "arith" else k
                s = stats.setdefault(key, {"n": 0, "probe": 0, "parse": 0, "fire": 0})
                s["n"] += 1; s["probe"] += g; s["parse"] += parsed; s["fire"] += g and parsed
            off += len(ids)
        return {k: {"n": v["n"], "probe_rate": v["probe"] / v["n"], "parse_rate": v["parse"] / v["n"],
                    "fire_rate": v["fire"] / v["n"]} for k, v in sorted(stats.items())}

    rep["test_teacher_forced"] = tf_table("test", w, b, th)
    # Controls, matched where they act: the calculator consults its gate only where the parser
    # already found an expression, so both controls fire at the real probe's rate on parse-positive
    # val positions (not its unconditional rate).
    def parse_mask(split):
        seqs, labs, _ = data[split]
        out = []
        for ids, lb in zip(seqs, labs):
            for i in range(len(ids)):
                text = tok.decode(ids[max(0, i + 1 - D.CONTEXT_TOKENS):i + 1])
                out.append(lb[i] >= 0 and D.calculator_answer(text, truncated=i + 1 > D.CONTEXT_TOKENS, parser=args.parser)[0] is not None)
        return np.array(out)
    pm_val = parse_mask("val")
    pv = C.probs(feats["val"][L], w, b)
    rate = float((pv[pm_val] > th).mean()) if pm_val.any() else 0.0
    rng = np.random.default_rng(0)
    ys = lab["train"].copy()
    keep = ys >= 0
    ys[keep] = rng.permutation(ys[keep])
    ws, bs = C.train_probe(feats["train"][L], ys)
    pvs = C.probs(feats["val"][L], ws, bs)
    ths = float(np.clip(np.quantile(pvs[pm_val], 1 - rate), 1e-6, 1 - 1e-6)) if pm_val.any() else 0.5
    rep["control_shuffled_label"] = {"parse_positive_val_fire_rate_matched": rate, "threshold": ths,
                                     "test": tf_table("test", ws, bs, ths)}
    rr = random.Random(0)
    rep["control_random_gate"] = {"rate": rate, "rate_basis": "real probe's fire rate on parse-positive val positions",
                                  "test": tf_table("test", w, b, th, gate_fn=lambda: rr.random() < rate)}
    print(json.dumps(rep["test_teacher_forced"], indent=0)[:3000], flush=True)

    # ---- manifest (force; push directions for the research arm) ------------------------------
    want_push = not args.skip_generation and any(a.startswith("push") for a in args.arms.split(","))
    if want_push:
        normw = donor.w["model.norm.weight"].astype(mx.float32)
        head = donor.head.astype(mx.float32)
        dirs = head / normw[None]
        dirs = dirs / mx.linalg.norm(dirs, axis=-1, keepdims=True)
        # Residual scale at the last layer on English (push_scale is in units of it).
        _, caps, _ = donor.forward(mx.array([english("val")["tinystories_train"][0]]), record={donor.L - 1})
        res_norm = float(mx.linalg.norm(caps[donor.L - 1][1].astype(mx.float32), axis=-1).mean())
    else:
        res_norm = None
    tiles = [{"id": "calculator", "kind": "calculator", "write_mode": "force", "read_layer": L,
              "write_layer": L, "gate": {"bias": float(b), "threshold": th, "probe": "logistic, balanced, val-chosen"},
              "parser": args.parser,
              **({"read_point": args.read_point} if args.read_point != "mlp_input" else {})}]
    man = D.write_manifest(out / "tiles-force", source={"artifact": donor.artifact, "fingerprint": src_fp},
                           layers=donor.L, tiles=tiles, tensors={"calculator.probe_w": mx.array(w)},
                           extra={"mlx2_source_revision": rep["mlx2_source_revision"], "host_tiles_manifest_sha256": host_sha,
                                  # Backends that declare their capture and fit numerics (hy4): the adapter
                                  # refuses the tile under another capture or other trunk numerics.
                                  **({"capture": donor.capture} if getattr(donor, "capture", None) else {}),
                                  **({"fit_execution_policy": donor.fit_execution_policy}
                                     if getattr(donor, "fit_execution_policy", None) else {})})
    rep["manifest"] = str(man)
    if args.probe_only:
        rep["wall_s"] = time.time() - t0
        rep["generation"] = "not run: probe-only calibration"
        (out / "report.json").write_text(json.dumps(rep, indent=1, default=float))
        print("probe complete", out, f"{rep['wall_s']:.0f}s", flush=True)
        return
    man_s = D.write_manifest(out / "tiles-shuffled-probe", source={"artifact": donor.artifact, "fingerprint": src_fp},
                             layers=donor.L,
                             tiles=[{**tiles[0], "gate": {"bias": float(bs), "threshold": ths,
                                                          "probe": "control: shuffled labels, parse-positive rate matched"}}],
                             tensors={"calculator.probe_w": mx.array(ws)},
                             extra={"host_tiles_manifest_sha256": host_sha, "control": "shuffled-label probe"})
    if args.skip_generation:
        rep["receipt_force"] = None
        rep["wall_s"] = time.time() - t0
        (out / "report.json").write_text(json.dumps(rep, indent=1, default=float))
        print("done (no generation)", out, f"{rep['wall_s']:.0f}s")
        return
    ts_shuf = D.build(man_s, None, source_fingerprint=src_fp, layers_count=donor.L)
    ts = D.build(man, None, source_fingerprint=src_fp, layers_count=donor.L)
    tsp = None
    if want_push:
        # Research-only push arm: same probe, write at the last layer toward the planned token.
        push_tiles = [{**tiles[0], "write_mode": "push", "write_layer": donor.L - 1}]
        man_p = D.write_manifest(out / "tiles-push", source={"artifact": donor.artifact, "fingerprint": src_fp},
                                 layers=donor.L, tiles=push_tiles, tensors={"calculator.probe_w": mx.array(w)})
        tsp = D.build(man_p, None, source_fingerprint=src_fp, layers_count=donor.L, allow_research_modes=True)
        PUSH_DIRS = dirs * res_norm
        mx.eval(PUSH_DIRS)

    # ---- generation arms --------------------------------------------------------------------
    mult = [(f"{a} * {b2} =", str(a * b2)) for a, b2 in mult_items()[:args.gen_items]]
    sanity = [(C.CHAT.format(q=f"What is {r + 3} times 12? Reply with the number only."), str((r + 3) * 12)) for r in range(20)]
    gsm = [(p, a.strip()) for p, a in C.positive_rows("test", 40, 99)]
    tasks = {"multiplication_27b_held": mult, "sanity_20x20_mul": sanity, "test_templates_mixed": gsm,
             "chained_expressions": chained_items()}
    # Teacher-forced host exact match on the 27B multiplication items (that task's own metric).
    tf_ok = 0
    for prompt, exp in mult:
        p_ids = tok.encode(prompt, add_special_tokens=False)
        a_ids = tok.encode(" " + exp, add_special_tokens=False)
        lg, _, _ = donor.forward(mx.array([p_ids + a_ids]))
        pred = mx.argmax(lg[0, len(p_ids) - 1:len(p_ids) + len(a_ids) - 1], axis=-1).tolist()
        tf_ok += pred == a_ids
    rep["multiplication_teacher_forced_host"] = {"correct": tf_ok, "n": len(mult), "acc": tf_ok / len(mult)}
    print("teacher-forced host mult", tf_ok, len(mult), flush=True)
    arms = args.arms.split(",")
    rep["generation"] = {}
    for tname, items in tasks.items():
        rep["generation"][tname] = {}
        for arm in arms:
            correct, sub_correct, outs = 0, 0, []
            for prompt, exp in items:
                kw = {}
                use = ts
                if arm.startswith("push"):
                    use, kw = tsp, {"push_scale": float(arm.split("@")[1])}
                    a_name = "push"
                else:
                    a_name = arm
                if arm == "random_force":
                    import zlib
                    rr2 = random.Random(zlib.crc32(prompt.encode()))
                    kw["gate_override"] = lambda text, rr2=rr2: rr2.random() < rate
                if arm == "shuffled_force":
                    use, a_name = ts_shuf, "force"
                text, ev = generate(donor, use, tok, "calculator", prompt, a_name, **kw)
                # Whole-number match (commas stripped; "36" fails on "360"); the stricter first-number
                # rule is recorded beside it.
                ok = whole_number_present(text, exp)
                sub_ok = first_number(text) == exp.replace(",", "")
                correct += ok
                sub_correct += sub_ok
                outs.append({"prompt": prompt[-60:], "expect": exp, "output": text, "ok": ok, "first_number_ok": sub_ok,
                             "events": ev})
            rep["generation"][tname][arm] = {"correct": correct, "n": len(items), "acc": correct / len(items),
                                             "first_number_correct": sub_correct,
                                             "samples": outs[:6]}
            print(tname, arm, correct, "/", len(items), flush=True)
    rep["push_residual_norm"] = res_norm
    rep["receipt_force"] = ts.receipt()
    rep["receipt_push"] = tsp.receipt() if tsp is not None else None
    rep["wall_s"] = time.time() - t0
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    print("done", out, f"{rep['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
