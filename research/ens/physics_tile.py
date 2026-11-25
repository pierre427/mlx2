"""ENS H43 on a real LM, step 2: path-grown knowledge tiles for 2026 physics papers through frozen Qwen3-4B + record port.

Pierre (2026-11-23): tiles whose paths grow per training row (gradient-guided plus random), carry usage counters and
are trimmed every x rows to the most used, on a frozen core that gives directions but is never written; relevance
anchored on fact endpoints, so rare but important paths survive.

A tile reads the original model's state at a paper mention (after block `site`, unit RMS) and returns the paper's
record: per slot (field, month, concept) a distribution over that slot's values, written as the probability-weighted
mix of the values' frozen embeddings. It learns **only through the frozen model + port's answer loss** on retrieval
and reasoning prompts about the knowledge world (train templates), as A0 contributors learned through the core.

Arms, one per run, all at the same number of active weights (the dense arm's parameter count):
  dense        an MLP d -> h -> 63 (h chosen for the budget)
  set          sparse d -> 512 -> 63; every round prune the smallest |w|, regrow at random (SET)
  rigl         same; prune smallest |w|, regrow where the dense gradient is largest (RigL)
  path         Pierre's rule: prune the lowest usage (EMA of |w * dL/dw_dense|, a Taylor contribution), regrow half
               by dense gradient and half at random
  path_anchor  path + relevance anchoring: edges with the highest usage on rare-paper rows are protected from
               pruning (budget 10% of edges)
Regrown weights start at 0 with zero optimizer moments. The prune fraction anneals (cosine) from 0.2 to 0.

Rare papers: 5% of the world, sampled so they make up about 1% of training rows. Measured: retrieval (field, month,
concept) and reasoning (same field, earlier, which concept) on train and held templates; rare-paper retrieval;
base model and oracle record (the true facts) as floor and ceiling; rare retrieval at every evaluation checkpoint.

Kill measures (across seeds, runner/summary): path_anchor or path no more reliable (seed spread) and no more accurate
than dense and set; anchoring keeps no more rare facts than path; tile reasoning <= base reasoning + 0.10.

  python -m research.ens.physics_tile --bus <physics_bus out> --arm path --seed 0 --out <dir>
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import signal
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from research.ens import common as C
from research.ens import lm_bus as LB
from research.ens import physics_bus as PB
from research.ens import physics_qa as QA
from research.ens import qwen3 as Q

ARMS = ("dense", "static", "set", "rigl", "path", "path_anchor")
# static (review): sparse at the same width and budget with a fixed random mask and no rounds, so a sparse win over
# dense can be attributed to topology updates rather than to width
SPARSE_H = 512
DENSE_H = 64


def widths(width: int = DENSE_H):
    """(dense hidden, sparse hidden) for a --width; the default is the original 64 / 512."""
    return width, width * SPARSE_H // DENSE_H
PROTECT_FRAC = 0.10


class Tile(nn.Module):
    """d -> H -> n_out. Sparse arms use masks (m1, m2) and zero 'probe' arrays (z1, z2) whose gradients are the dense
    gradients of the effective weights (for RigL-style regrowth and usage)."""

    def __init__(self, d, H, n_out, sparse, density1=1.0, seed=0):
        super().__init__()
        k = 1 / math.sqrt(d)
        self.w1 = mx.random.uniform(-k, k, (d, H))
        self.b1 = mx.zeros((H,))
        self.w2 = mx.random.uniform(-1 / math.sqrt(H), 1 / math.sqrt(H), (H, n_out))
        self.b2 = mx.zeros((n_out,))
        self.sparse = sparse
        if sparse:
            n_on = int(round(density1 * d * H))           # exact count (review: a Bernoulli mask overshot +0.9%)
            order = mx.argsort(mx.random.uniform(shape=(d * H,), key=mx.random.key(seed)))
            flat = mx.zeros((d * H,))
            flat[order[:n_on]] = 1.0
            self.m1 = flat.reshape(d, H)
            self.m2 = mx.ones((H, n_out))
            self.z1 = mx.zeros((d, H))
            self.z2 = mx.zeros((H, n_out))
            # same starting pre-activation scale as dense: rescale by 1/sqrt(density) (review)
            self.w1 = self.w1 * self.m1 / math.sqrt(density1)

    def __call__(self, x):
        w1 = self.w1 * self.m1 + self.z1 if self.sparse else self.w1
        w2 = self.w2 * self.m2 + self.z2 if self.sparse else self.w2
        return nn.silu(x @ w1 + self.b1) @ w2 + self.b2

    def active(self):
        if not self.sparse:
            return self.w1.size + self.w2.size
        return int(mx.sum(self.m1).item()) + int(mx.sum(self.m2).item())


def payload(tile_out, voc, model):
    """(B, K, 63) logits -> {slot: (B, K, d)} unit-RMS soft embeddings of each slot's values."""
    out, i = {}, 0
    for s in PB.SLOTS:
        n = len(voc.ids[s])
        p = mx.softmax(tile_out[..., i:i + n].astype(mx.float32), -1)
        E = PB.unit_rms(mx.stop_gradient(model.model.embed_tokens(mx.array(voc.ids[s]))))
        out[s] = PB.unit_rms(p @ E)
        i += n
    return out


def world_rows(tok, voc, papers, weights, rng, n, templates, tasks=None):
    rows = []
    while len(rows) < n:
        task = rng.choice(tasks or (PB.RETRIEVAL + PB.REASONING))
        k = 1 if task in PB.RETRIEVAL else 2
        ps = rng.choices(papers, weights=weights, k=k)
        if k == 2 and ps[0]["id"] == ps[1]["id"]:
            continue
        ps = PB.balance_same_field(task, ps, rng, pool=papers, weights=weights)
        r = PB.make_row(tok, voc, task, ps, rng, templates)
        if r is not None:
            rows.append((task, [p["id"] for p in ps]) + r)
    return rows


_FREE = {"max_seconds": None, "stop_after": None, "no_memory_gate": None, "chunk_tag": None}
# kill measures (summary over seeds): reasoning floor = max(0.5, unseen-pair tile reasoning, swapped-record reasoning)
# (review: the base model ignores the answer format and scores 0-0.17, too low to be a floor)


def in_b(p) -> bool:
    return int(hashlib.sha256(f"h43.split:{p['id']}".encode()).hexdigest()[:8], 16) % 2 == 1


def load_init(tile, init_dir: Path):
    """Phase 2 starts from a phase-1 tile (weights and masks) and its final usage counters."""
    rec = json.loads((init_dir / "receipt.json").read_text())
    if rec.get("width", DENSE_H) * (1 if not tile.sparse else SPARSE_H // DENSE_H) != tile.w1.shape[1]:
        raise SystemExit("--init must be a phase-1 run of the same width")   # before load_weights (review)
    C.load_checkpoint_weights(tile, init_dir / "tile.safetensors", rec["tile_checkpoint"])
    rd = PB.resume_dir(init_dir)
    u = dict(mx.load(str(rd / "model.safetensors"))).get("usage") if rd else None
    return rec, u


def same_args(saved: dict, now: dict) -> bool:
    """Resume check: options added after a run started count as their defaults (a run started before --train-split
    existed resumes with --train-split all); per-launch options in _FREE may change."""
    defaults = vars(build_parser().parse_args(["--bus", "x", "--arm", "dense", "--out", "x"]))
    full = {k: saved.get(k, defaults.get(k)) for k in now}
    return full | _FREE == now | _FREE


def run(args) -> int:
    out = Path(args.out)
    if (out / "receipt.json").exists():
        print("already finished", flush=True)
        return 0
    resuming = PB.resume_dir(out) is not None
    if not resuming:
        if out.is_dir() and [q.name for q in out.iterdir()] == ["args.json"]:
            (out / "args.json").unlink()
        out = C.fresh_out(args.out)
        C.write_json(out / "args.json", vars(args))
    elif not same_args(json.loads((out / "args.json").read_text()), vars(args)):
        raise SystemExit("resume: arguments differ")
    mx.random.seed(args.seed)
    brec = json.loads((Path(args.bus) / "train_receipt.json").read_text())   # written before the bus evaluation
    bargs = argparse.Namespace(**brec["args"])
    if not args.no_memory_gate:
        PB.memory_gate(gib=PB.gate_gib(bargs))
    facts, model, cfg, ckpt, tok, voc = PB.load_world(bargs)
    guard = PB.SwapGuard()
    if ckpt != brec["base_checkpoint"] or facts["sha256"] != brec["facts_sha256"]:
        raise SystemExit("base checkpoint or facts differ from the bus receipt")
    s = brec["port"]
    port = PB.RecordPort(model, s["site"], s["rank"], s["tau"])
    C.load_checkpoint_weights(port, Path(args.bus) / "port.safetensors", brec["port_checkpoint"])
    port.freeze()
    allp = [p for p in facts["papers"] if "'" not in p["title"]]
    prng = random.Random(123 + args.world_seed)
    unseen_ids = set(p["id"] for p in prng.sample(allp, max(1, len(allp) // 20)))
    # unseen: 5% of papers the tile never trains on, the no-memorisation floor (what can be guessed from the title;
    # review). Fixed across arms by world_seed.
    unseen = [p for p in allp if p["id"] in unseen_ids]
    papers = [p for p in allp if p["id"] not in unseen_ids]
    # post hoc learning (H43b, Pierre 2026-11-23): a random 50/50 split of the trained papers by id hash (not by month:
    # a month split would let a tile infer "new papers are later"). Phase 1 trains on A; phase 2 adds B.
    split = {"A": [q for q in papers if not in_b(q)], "B": [q for q in papers if in_b(q)]}
    if args.train_split != "all":
        papers = split[args.train_split]
    rare_ids = set(p["id"] for p in prng.sample(papers, max(1, len(papers) // 20)))
    n_rare = len(rare_ids)
    w_rare = 0.01 / n_rare
    w_common = 0.99 / (len(papers) - n_rare)
    weights = [w_rare if p["id"] in rare_ids else w_common for p in papers]
    rare = [p for p in papers if p["id"] in rare_ids]
    d = cfg.hidden_size
    n_out = sum(len(voc.ids[sl]) for sl in PB.SLOTS)
    # --width (Pierre 2026-11-23, "and also possibly path width"): the dense tile's hidden width; sparse arms keep the
    # equal active-weight budget of that dense tile with SPARSE_H/DENSE_H times as many hidden units
    if args.width < 1:
        raise SystemExit("--width must be positive")
    dense_h, sparse_h = widths(args.width)
    budget = d * dense_h + dense_h * n_out
    if args.arm == "dense":
        tile = Tile(d, dense_h, n_out, sparse=False)
        dens = 1.0
    else:
        dens = (budget - sparse_h * n_out) / (d * sparse_h)
        tile = Tile(d, sparse_h, n_out, sparse=True, density1=dens, seed=args.seed)
    trainable = ["w1", "b1", "w2", "b2"]
    opt = optim.Adam(learning_rate=args.lr)
    init_rec, init_usage = (None, None)
    if args.init:
        init_rec, init_usage = load_init(tile, Path(args.init))
        # review 2026-11-23: phase 2 continues a phase-1 run trained on A only, on the same bus/port and facts, and
        # trains on B only (A data never re-seen)
        if init_rec["arm"] != args.arm or init_rec["world_seed"] != args.world_seed or init_rec["seed"] != args.seed \
                or init_rec.get("width", DENSE_H) != args.width:
            # same seed too: evaluation rows are seeded by --seed, so old_A retention compares the same questions
            raise SystemExit("--init must be a phase-1 run of the same arm, width, seed and world seed")
        if init_rec["world"]["train_split"] != "A" or args.train_split != "B" or init_rec.get("init") is not None:
            raise SystemExit("--init needs a fresh phase-1 run on split A, and this run must use --train-split B")
        if init_rec["port_checkpoint"] != brec["port_checkpoint"] or init_rec["facts_sha256"] != brec["facts_sha256"]:
            raise SystemExit("--init was trained on another bus/port or facts file")
        if tile.sparse and init_usage is None:
            # without phase-1 usage path/path_anchor would run with zero usage; anchoring then protects every edge
            # and regrowth exceeds the weight budget (review)
            raise SystemExit("--init: the phase-1 run's resume/ state (usage counters) is missing")

    def lf(t, T, W, Y, M, X, H):
        pl = payload(t(X), voc, model)
        return PB.answer_loss(PB.forward(model, T, port, M, pl, h_site=H), W, Y)
    g = nn.value_and_grad(tile, lf)
    rng = random.Random(40_000 + args.seed)          # data rows only: identical across arms (review)
    topo_rng = random.Random(41_000 + args.seed)     # regrowth draws
    rare_rng = random.Random(42_000 + args.seed)     # path_anchor's rare batches
    usage = mx.zeros_like(tile.w1) if tile.sparse else None
    usage_rare = mx.zeros_like(tile.w1) if tile.sparse else None
    if args.init and tile.sparse and init_usage is not None:
        usage = init_usage
        if args.arm == "path_anchor":
            # phase 2 has no phase-1 data: anchoring protects the edges phase 1 used most (its final usage), fixed
            usage_rare = init_usage
    log, rare_track, rounds, t0, t_used, it0 = [], [], [], time.time(), 0.0, 0
    if resuming:
        d_ = PB.resume_dir(out)
        import pickle
        st = pickle.loads((d_ / "stream.pkl").read_bytes())
        w = dict(mx.load(str(d_ / "model.safetensors")))
        tile.load_weights([(k.removeprefix("tile."), v) for k, v in w.items() if k.startswith("tile.")])
        if tile.sparse:
            usage, usage_rare = w["usage"], w["usage_rare"]
        from mlx.utils import tree_unflatten
        opt.init({k: getattr(tile, k) for k in trainable})
        opt.state = tree_unflatten(list(mx.load(str(d_ / "opt.safetensors")).items()))
        for r_, s_ in zip((rng, topo_rng, rare_rng), st["rngs"]):
            r_.setstate(s_)
        it0, t_used = st["it"], st["train_seconds"]
        log, rare_track, rounds = st["log"]["log"], st["log"]["rare_track"], st["log"]["rounds"]
        print(f"resumed at {it0}", flush=True)

    def save(it):
        tree = {"tile": tile.parameters()}
        if tile.sparse:
            tree |= {"usage": usage, "usage_rare": usage_rare}
        PB.save_state(out, tree, opt, it, [rng, topo_rng, rare_rng], {"log": log, "rare_track": rare_track, "rounds": rounds},
                      t_used + time.time() - t0)

    def batch_inputs(rows):
        T, W, Y, M, R = PB.stack(tok, [r[2:] for r in rows])
        H = mx.stop_gradient(PB.prefix(model, T, port.site))       # shared by the tile input and the forward
        X = PB.unit_rms(PB.site_states(model, T, port.site, M, h=H))
        return T, W, Y, M, mx.stop_gradient(X), H
    PB._TERM["requested"] = False
    old_handler = signal.signal(signal.SIGTERM, PB._on_term)
    it = it0
    while it < args.steps:
        n_launch = it - it0
        swap = None
        if not args.no_memory_gate and (n_launch <= PB.SWAP_WARMUP_STEPS or n_launch % 10 == 0):
            had_base = guard.base is not None
            swap = guard.check(n_launch)
            if guard.base is not None and not had_base:
                log.append({"step": it, "swap_warmup_gib": guard.warmup_gib})
                print(log[-1], flush=True)
        if PB._TERM["requested"] or swap or (args.max_seconds and time.time() - t0 > args.max_seconds) or \
                (args.stop_after is not None and it - it0 >= args.stop_after):
            if swap:
                log.append({"step": it, "swap_stop": swap, "swap_warmup_gib": guard.warmup_gib})
                print(swap, flush=True)
            save(it)
            signal.signal(signal.SIGTERM, old_handler)
            print(f"stopped at {it} (resume state saved)", flush=True)
            return 3
        it += 1
        rows = world_rows(tok, voc, papers, weights, rng, args.batch, PB.T_TRAIN)
        T, W, Y, M, X, H = batch_inputs(rows)
        loss, gr = g(tile, T, W, Y, M, X, H)
        if tile.sparse:
            # usage counter: EMA (0.98, ~50 steps of a 100-step round) of the Taylor contribution |w * dL/dw| on the
            # batch-mean gradient; signs cancel across rows and rare rows are ~0.16 per batch, which is why anchoring
            # exists (review 10)
            usage = 0.98 * usage + 0.02 * mx.abs(tile.w1 * tile.m1 * gr["z1"])
            if args.arm == "path_anchor" and it % 10 == 0 and not args.init:
                rr = world_rows(tok, voc, rare, None, rare_rng, args.batch, PB.T_TRAIN, PB.RETRIEVAL)
                _, grr = g(tile, *batch_inputs(rr))
                usage_rare = 0.8 * usage_rare + 0.2 * mx.abs(tile.w1 * tile.m1 * grr["z1"])
            dense_grad = gr["z1"]
        upd = {k: gr[k] for k in trainable}
        opt.update(tile, {**upd, **({"z1": mx.zeros_like(tile.z1), "z2": mx.zeros_like(tile.z2),
                                     "m1": mx.zeros_like(tile.m1), "m2": mx.zeros_like(tile.m2)}
                                    if tile.sparse else {})})
        if tile.sparse:                                    # probes and masks never move; pruned weights stay 0
            tile.z1 = mx.zeros_like(tile.z1)
            tile.z2 = mx.zeros_like(tile.z2)
            tile.w1 = tile.w1 * tile.m1
        mx.eval(tile.parameters(), opt.state)
        if tile.sparse and args.arm not in ("dense", "static") and it % args.round_every == 0 and it < args.steps * 0.75:
            frac = args.prune_frac * 0.5 * (1 + math.cos(math.pi * it / (args.steps * 0.75)))
            rounds.append(update_topology(tile, opt, args.arm, frac, usage, usage_rare, dense_grad, topo_rng, it))
        if it % args.log_every == 0 or it == args.steps:
            log.append({"step": it, "loss": round(float(loss.item()), 5), "active": tile.active(),
                        "elapsed_s": round(t_used + time.time() - t0, 1), **PB.mem_stats()})
            print(log[-1], flush=True)
        if (it % args.track_every == 0 or it == args.steps) and not (
                args.max_seconds and time.time() - t0 > args.max_seconds - 120):
            # tracking generates answers: save first at the last step, and skip it close to the deadline (review)
            if it == args.steps:
                save(it)
            rare_track.append({"step": it, **retrieval_acc(model, port, tile, tok, voc, rare, args, 24, batch_inputs)})
        if args.save_every and it % args.save_every == 0 and it < args.steps:
            save(it)
    signal.signal(signal.SIGTERM, old_handler)
    if not (out / "tile.safetensors").exists():
        save(it)                                       # training done: a kill during evaluation resumes straight to it
    tile.freeze()
    wpath = out / "tile.safetensors"
    if not wpath.exists():
        mx.save_safetensors(str(wpath), dict(tree_flatten(tile.parameters())))
    with wpath.open("rb") as fh:
        tfp = hashlib.file_digest(fh, "sha256").hexdigest()[:12]
    deadline = (t0 + args.max_seconds) if args.max_seconds else None
    if args.train_split == "all":
        ev = evaluate(model, port, tile, tok, voc, papers, rare, unseen, args, batch_inputs, PB.EvalCells(out, deadline))
    else:
        ev = evaluate_posthoc(lambda X: tile(X), model, port, tok, voc, split, unseen, args, batch_inputs,
                              PB.EvalCells(out, deadline))
    receipt = {"schema": "ens.h43.physics_tile_receipt.v1", "arm": args.arm, "seed": args.seed,
               "world_seed": args.world_seed, "width": args.width, "base_checkpoint": ckpt,
               "port_checkpoint": brec["port_checkpoint"],
               "tile_checkpoint": f"ens-h43-tile-{args.arm}{'' if args.width == DENSE_H else f'-w{args.width}'}-s{args.seed}@{tfp}", "facts_sha256": facts["sha256"],
               "budget": {"active_weights": tile.active(), "target": budget, "steps": args.steps, "batch": args.batch,
                          "lr": args.lr, "round_every": args.round_every, "prune_frac": args.prune_frac},
               "world": {"papers_trained": len(papers), "unseen": len(unseen), "rare": n_rare, "rare_row_share": 0.01,
                         "train_split": args.train_split, "split_sizes": {k: len(v) for k, v in split.items()}},
               "init": ({"dir": args.init, "tile_checkpoint": init_rec["tile_checkpoint"],
                         # review note: phase 2 starts a fresh Adam (MLX default bias_correction=False, so the first
                         # steps are larger than steady state) and path arms restart the prune schedule; this hits
                         # every continued arm alike and adds to early forgetting
                         "optimizer": "fresh Adam, bias_correction False; prune schedule restarted"}
                        if init_rec else None),
               "asymmetry": "path_anchor knows which papers are rare and gets a rare-only gradient batch every 10 "
                            "steps (privileged information the other arms lack)",
               "bias_note": f"sparse arms have {widths(args.width)[1]} hidden biases vs {args.width} for dense "
                            "(biases not counted in the budget)",
               "log": log, "rare_track": rare_track, "rounds": rounds, "eval": ev,
               **C.source_revision(), **C.host_record()}
    C.write_json(out / "receipt.json", receipt)
    print("eval", json.dumps(ev), flush=True)
    return 0


def update_topology(tile, opt, arm, frac, usage, usage_rare, dense_grad, rng, it):
    m = tile.m1
    n_active = int(mx.sum(m).item())
    k = int(frac * n_active)
    if k < 1:
        return {"step": it, "pruned": 0, "grown": 0}
    score = usage if arm in ("path", "path_anchor") else mx.abs(tile.w1)
    score = mx.where(m > 0, score, mx.inf)
    saved = 0
    if arm == "path_anchor":                               # protect the top rare-usage edges from pruning
        n_prot = int(PROTECT_FRAC * n_active)
        thr = mx.sort(mx.where(m > 0, usage_rare, -mx.inf).reshape(-1))[-n_prot] if n_prot else mx.inf
        prot = (m > 0) & (usage_rare >= thr)
        # how many protected edges would have been pruned without protection (is anchoring doing anything?)
        unprot_bottom = mx.argsort(score.reshape(-1))[:k]
        saved = int(mx.sum(prot.reshape(-1)[unprot_bottom]).item())
        score = mx.where(prot, mx.inf, score)
    flat = score.reshape(-1)
    prune_idx = mx.argsort(flat)[:k]
    newm = m.reshape(-1)
    newm[prune_idx] = 0.0
    off = (newm == 0)
    n_grad = k if arm == "rigl" else (k // 2 if arm in ("path", "path_anchor") else 0)
    grown = mx.zeros_like(newm)
    if n_grad:
        gs = mx.where(off, mx.abs(dense_grad.reshape(-1)), -mx.inf)
        gi = mx.argsort(gs)[-n_grad:]
        grown[gi] = 1.0
    n_rand = k - n_grad
    if n_rand:
        offn = (off & (grown == 0)).astype(mx.float32)
        r = mx.random.uniform(shape=offn.shape, key=mx.random.key(rng.randrange(1 << 30))) * offn
        ri = mx.argsort(r)[-n_rand:]
        grown[ri] = 1.0
    newm = mx.minimum(newm + grown, 1.0).reshape(m.shape)
    changed = (newm != m)
    tile.m1 = newm
    tile.w1 = mx.where(changed, 0.0, tile.w1) * newm       # pruned -> 0, regrown start at 0
    st = opt.state
    for key in ("m", "v"):
        if "w1" in st and key in st["w1"]:
            st["w1"][key] = mx.where(changed, 0.0, st["w1"][key])
    mx.eval(tile.m1, tile.w1)
    return {"step": it, "pruned": k, "grown_grad": n_grad, "grown_random": n_rand, "frac": round(frac, 4),
            "turnover": int(mx.sum(changed).item()), "anchor_saved": saved}


def retrieval_acc(model, port, tile, tok, voc, papers, args, n, batch_inputs):
    """Tracking metric during training: retrieval answers generated and parsed (review: a teacher-forced exact match
    was capped by the random choice between two answer templates)."""
    rng = random.Random(90_000 + args.seed)
    res = {}
    for task in PB.RETRIEVAL:
        rows = world_rows(tok, voc, papers, None, rng, n, PB.T_TRAIN, [task])
        hit = 0
        for a0 in range(0, n, 16):
            sub = rows[a0:a0 + 16]
            T, W, Y, M, X, H = batch_inputs(sub)
            texts = PB.generate(model, tok, port, [r[2] for r in sub], M, payload(tile(X), voc, model),
                                QA.MAX_NEW[task])
            hit += sum(int(s["correct"]) for s in PB.score_batch([task] * len(sub), texts, [r[6] for r in sub],
                                                                 voc.values["concept"]))
        res[task] = round(hit / n, 4)
    return res


def evaluate(model, port, tile, tok, voc, papers, rare, unseen, args, batch_inputs, cells):
    """Free generation, parsed (physics_qa.score), resumable by cell. Tile arms always: the tile's record, and the
    tile's records rotated to another row (swapped: if answers stay correct they did not come from the record).
    With --with-floors (run once per seed, not per arm; review): the base model and the oracle record. Cells: train
    and held questions over trained papers; rare papers (retrieval); unseen papers (retrieval and reasoning pairs:
    the no-memorisation floor)."""
    concepts = voc.values["concept"]

    def run_cell(key, temps, pool, task, n):
        if cells.done(key):
            return
        rng = random.Random(70_000 + args.seed + PB.hash_task(key))
        rows = world_rows(tok, voc, pool, None, rng, n, temps, [task])
        tally, ex = {}, None
        for sub in PB.eval_batches(rows):
            cells.check()
            T, W, Y, M, X, H = batch_inputs(sub)
            _, _, _, _, R = PB.stack(tok, [r[2:] for r in sub])
            prompts, metas, mn = [r[2] for r in sub], [r[6] for r in sub], QA.MAX_NEW[task]
            pl = payload(tile(X), voc, model)
            gens = {"tile": PB.generate(model, tok, port, prompts, M, pl, mn)}
            if len(sub) > 1:
                pls = {s: mx.concatenate([v[1:], v[:1]]) for s, v in pl.items()}
                gens["tile_swapped"] = PB.generate(model, tok, port, prompts, M, pls, mn)
            if args.with_floors:
                gens["base"] = PB.generate(model, tok, None, prompts, M, None, mn)
                gens["oracle_record"] = PB.generate(model, tok, port, prompts, M, PB.payload_from_ids(model, R), mn)
            for arm, texts in gens.items():
                for sc in PB.score_batch([task] * len(sub), texts, metas, concepts):
                    for kk, v in sc.items():
                        tally[f"{arm}.{kk}"] = tally.get(f"{arm}.{kk}", 0) + int(v)
            if ex is None:
                ex = {"q": tok.t.decode(prompts[0]), **{a: g_[0] for a, g_ in gens.items()}}
        cells.put(key, {k: round(v / len(rows), 4) for k, v in sorted(tally.items())} | {"n": len(rows), "example": ex})
    for name, temps in (("train_questions", PB.T_TRAIN), ("held_questions", PB.T_HELD)):
        for task in PB.RETRIEVAL + PB.REASONING:
            run_cell(f"{name}.{task}", temps, papers, task, args.eval_n)
    for task in PB.RETRIEVAL:
        run_cell(f"rare_papers.{task}", PB.T_TRAIN, rare, task, args.eval_n)
    for task in PB.RETRIEVAL + PB.REASONING:
        run_cell(f"unseen_papers.{task}", PB.T_TRAIN, unseen, task, args.eval_n)
    res = dict(cells.cells)
    res["majority_rate"] = {s: round(max(collections.Counter(p[s] for p in papers).values()) / len(papers), 4)
                            for s in PB.SLOTS}
    mean = {}
    arms = ("tile", "tile_swapped") + (("base", "oracle_record") if args.with_floors else ())
    for name in ("train_questions", "held_questions", "unseen_papers"):
        for grp, tasks in (("retrieval", PB.RETRIEVAL), ("reasoning", PB.REASONING)):
            for arm in arms:
                vals = [res[f"{name}.{tk}"].get(f"{arm}.correct") for tk in tasks]
                if all(v is not None for v in vals):
                    mean[f"{name}.{grp}.{arm}"] = round(sum(vals) / len(vals), 4)
    res["mean"] = mean
    return res


def pair_rows(tok, voc, first, second, rng, n, templates, tasks):
    """Reasoning rows with one paper from `first` and one from `second` (cross old/new pairs). The order is random per
    row (review: a fixed old-first order confounds position with retention); meta["first_pos"] records where the
    `first` paper is."""
    rows = []
    while len(rows) < n:
        task = rng.choice(tasks)
        A = rng.choice(first)
        B = rng.choice(second)
        if A["id"] == B["id"]:
            continue
        A, B = PB.balance_same_field(task, [A, B], rng, pool=second)
        flip = rng.random() < 0.5
        pair = [B, A] if flip else [A, B]
        r = PB.make_row(tok, voc, task, pair, rng, templates)
        if r is not None:
            r[4]["first_pos"] = 1 if flip else 0
            rows.append((task, [q["id"] for q in pair]) + r)
    return rows


def evaluate_posthoc(tile_fn, model, port, tok, voc, split, unseen, args, batch_inputs, cells, extra=None):
    """H43b cells, resumable: old papers (A) and new papers (B), retrieval and reasoning; cross pairs (one old, one
    new); held questions on B retrieval; unseen papers (floor). tile_fn maps site states (B, K, d) -> logits; extra
    (optional) adds per-batch diagnostics (routing). Also the tile with records rotated (swapped)."""
    concepts = voc.values["concept"]

    def run_cell(key, rows, task):
        if cells.done(key):
            return
        tally, ex = {}, None
        for sub in PB.eval_batches(rows):
            cells.check()
            T, W, Y, M, X, H = batch_inputs(sub)
            prompts, metas, mn = [r[2] for r in sub], [r[6] for r in sub], QA.MAX_NEW[task]
            pl = payload(tile_fn(X), voc, model)
            gens = {"tile": PB.generate(model, tok, port, prompts, M, pl, mn)}
            if len(sub) > 1:
                gens["tile_swapped"] = PB.generate(model, tok, port, prompts, M,
                                                   {s: mx.concatenate([v[1:], v[:1]]) for s, v in pl.items()}, mn)
            for arm, texts in gens.items():
                for m_, sc in zip(metas, PB.score_batch([task] * len(sub), texts, metas, concepts)):
                    for kk, v in sc.items():
                        tally[f"{arm}.{kk}"] = tally.get(f"{arm}.{kk}", 0) + int(v)
                    if "first_pos" in m_:
                        # cross cells: split by where the old paper sits (review); the rate for position p is
                        # {arm}.correct_oldpos{p} / oldpos{p}_share (both are divided by the cell's row count)
                        p_ = m_["first_pos"]
                        k_ = f"{arm}.correct_oldpos{p_}"
                        tally[k_] = tally.get(k_, 0) + int(sc["correct"])
                        if arm == "tile":
                            tally[f"oldpos{p_}_share"] = tally.get(f"oldpos{p_}_share", 0) + 1
            if extra:
                for kk, v in extra(sub, X).items():
                    tally[kk] = tally.get(kk, 0) + v
            if ex is None:
                ex = {"q": tok.t.decode(prompts[0]), **{a: g_[0] for a, g_ in gens.items()}}
        cells.put(key, {k: round(v / len(rows), 4) for k, v in sorted(tally.items())} | {"n": len(rows), "example": ex})
    n = args.eval_n
    for grp, pool in (("old_A", split["A"]), ("new_B", split["B"])):
        for task in PB.RETRIEVAL + PB.REASONING:
            rng = random.Random(70_000 + args.seed + PB.hash_task(grp, task))
            run_cell(f"{grp}.{task}", world_rows(tok, voc, pool, None, rng, n, PB.T_TRAIN, [task]), task)
    for task in PB.REASONING:
        rng = random.Random(71_000 + args.seed + PB.hash_task("cross", task))
        run_cell(f"cross.{task}", pair_rows(tok, voc, split["A"], split["B"], rng, n, PB.T_TRAIN, [task]), task)
    for task in PB.RETRIEVAL:
        rng = random.Random(72_000 + args.seed + PB.hash_task("heldB", task))
        run_cell(f"held_questions_B.{task}", world_rows(tok, voc, split["B"], None, rng, n, PB.T_HELD, [task]), task)
        rng = random.Random(73_000 + args.seed + PB.hash_task("unseen", task))
        run_cell(f"unseen_papers.{task}", world_rows(tok, voc, unseen, None, rng, n, PB.T_TRAIN, [task]), task)
    res = dict(cells.cells)
    mean = {}
    for grp in ("old_A", "new_B", "cross", "held_questions_B", "unseen_papers"):
        for kind, tasks in (("retrieval", PB.RETRIEVAL), ("reasoning", PB.REASONING)):
            vals = [res.get(f"{grp}.{tk}", {}).get("tile.correct") for tk in tasks]
            if all(v is not None for v in vals):
                mean[f"{grp}.{kind}"] = round(sum(vals) / len(vals), 4)
    # cross headline without which_concept: it is solvable by elimination from the old paper alone (review)
    vals = [res.get(f"cross.{tk}", {}).get("tile.correct") for tk in ("same_field", "earlier")]
    if all(v is not None for v in vals):
        mean["cross.reasoning_two_sided"] = round(sum(vals) / len(vals), 4)
    res["mean"] = mean
    return res


def main(argv=None):
    sys.exit(run(build_parser().parse_args(argv)))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bus", required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--world-seed", type=int, default=0, help="which 5%% of papers are rare (fixed across arms)")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--width", type=int, default=DENSE_H, help="dense tile hidden width (sparse arms: equal budget)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--round-every", type=int, default=100)
    p.add_argument("--prune-frac", type=float, default=0.2)
    p.add_argument("--eval-n", type=int, default=24, help="questions per evaluation cell (answers are generated)")
    p.add_argument("--train-split", choices=("all", "A", "B"), default="all",
                   help="H43b post hoc: train on half A (phase 1) or B (phase 2); 'all' is the main H43 run")
    p.add_argument("--init", default=None, help="H43b phase 2: continue from this phase-1 run directory (same arm)")
    p.add_argument("--with-floors", action="store_true",
                   help="also evaluate the base model and the oracle record (once per seed; they do not depend on the arm)")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--track-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--stop-after", type=int, default=None, help="stop after this many updates in this launch (tests)")
    p.add_argument("--no-memory-gate", action="store_true", help="skip the M5 memory gate and swap guard (CPU tests)")
    p.add_argument("--chunk-tag", default=None, help="ignored; lets the M5 slot watchdog find this process")
    p.add_argument("--max-seconds", type=float, default=None, help="stop cleanly after this long; rerun resumes")
    p.add_argument("--out", required=True)
    return p


if __name__ == "__main__":
    main()
