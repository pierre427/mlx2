"""H45: a core trained with a loop port, against a fixed core on the same data (Pierre, 2026-11-23).

The loop port is a block range [start, end) the core may run k >= 1 times per row (model.Core.looped). During
core training a stand-in drives it, as the oracle mock provider drove the bus: **one step per pass**. A row's pass
count is its number of sequential steps (graph hops, FST input bits, program operations; 1 for PERM). The bus
writes z once, before the loop, exactly as for C6 (seed-1 recipe: early site, tau 0.5, w 64, ordinary share 0.25).

  loop    passes = steps; the loop body is blocks 2-3 (just after the early bus site)
  fixed   the seed-1 (C6) recipe: one pass, same data, seed, init and updates
  uniform3  compute control (review 2026-11-23): the loop body runs 3 times for every row in training, whatever
          its steps, so it has the loop arm's training compute (10 block applications per row, against 6 for
          fixed) without pass counts tied to steps. Separates "passes = steps" from "more depth".
  loopres  H45c (LoopRes, Pierre 2026-11-25): the loop arm with the LoopRes read (model.Core._looped_res): no
          residual stream in the loop body; each looped sub-layer's input is a zero-query softmax mix over the
          prelude output, previous-pass and this-pass block outputs and the block's partial sum, scored by content
          and relative role keys; a separate read forms the loop output the post-loop blocks consume. Role keys and
          queries get AdamW weight decay 0.01 like every other parameter (the paper does not say it decays them)
  loopres_content  the LoopRes ablation: content scores only, no role keys

The training mix is the 17 A0 training families plus depth-3 families (graph 4-3 and 6-3, FST 2-3, a 3-op program),
so both arms see depths 1-3. Depths 4-6 are never trained: the depth evaluation (graph 4-d and 6-d, FST 2-d,
d = 1..6; programs of 1-3 ops) asks whether the loop core runs unseen depth by running more passes.

Resumable: --max-seconds stops cleanly between updates and saves the core, optimizer and data-stream state to
<out>/resume/; rerunning the same command continues (exit code 3 = stopped early, 0 = finished). This lets a 48k
core run in <= 20-minute M5 slots or in one M3 window.

  python -m research.ens.loop_train --arm loop --seed 0 --steps 48000 --out <dir> [--max-seconds 1100]
"""
from __future__ import annotations

import argparse
import fcntl
import os
import hashlib
import json
import math
import pickle
import random
import shutil
import signal
import sys
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from research.ens import common as C
from research.ens import families as F
from research.ens import vocab as V
from research.ens.model import BusCall, Core, CoreConfig, value_logits

LOOP_BODY = (2, 4)
TRAIN_MAX_PASSES = 3
EXTRA = (F.Family("h45.graph.4-3", "GRAPH", (4, 3), "h45_train"), F.Family("h45.graph.6-3", "GRAPH", (6, 3), "h45_train"),
         F.Family("h45.fst.2-3", "FST", (2, 3), "h45_train"),
         F.Family("h45.program.ADD.MUL.XOR", "PROGRAM", (("ADD", "MUL", "XOR"),), "h45_train"))


def steps_of(fam: F.Family) -> int:
    if fam.cls in ("GRAPH", "FST"):
        return fam.knobs[1]
    if fam.cls == "PROGRAM":
        return len(fam.knobs[0])
    return 1


def train_mix():
    return list(F.split("train")) + list(EXTRA)


def depth_families():
    out = []
    for d in range(1, 7):
        out += [F.Family(f"h45.graph.4-{d}", "GRAPH", (4, d), "h45_eval"),
                F.Family(f"h45.graph.6-{d}", "GRAPH", (6, d), "h45_eval"),
                F.Family(f"h45.fst.2-{d}", "FST", (2, d), "h45_eval")]
    out += [F.Family("h45.program.ADD", "PROGRAM", (("ADD",),), "h45_eval"),
            F.Family("h45.program.ADD.MUL", "PROGRAM", (("ADD", "MUL"),), "h45_eval"),
            F.Family("h45.program.ADD.MUL.XOR", "PROGRAM", (("ADD", "MUL", "XOR"),), "h45_eval")]
    return out


def trained_structure(fam) -> bool:
    return (fam.cls, fam.knobs) in {(f.cls, f.knobs) for f in train_mix()}


def answer_at(fam, row, k):
    """The row's answer after k sequential steps from its z (graph hops, FST bits, program ops)."""
    vals = [t - V.V0 for t in row.op]
    z = row.z
    if fam.cls == "GRAPH":
        succ = vals[1:1 + fam.knobs[0]]
        for _ in range(k):
            z = succ[z]
        return z
    if fam.cls == "FST":
        s = fam.knobs[0]
        table = [(vals[1 + 2 * i], vals[2 + 2 * i]) for i in range(s)]
        bits = [1 if tk == V.tok("B1") else 0 for tk in row.op[2 + 2 * s:len(row.op) - 1]]
        if k > len(bits):
            raise ValueError("no input bit left")
        for b in bits[:k]:
            z = table[z][b]
        return z
    if fam.cls == "PROGRAM":
        prog = [(V.TOKENS[row.op[i]], vals[i + 1]) for i in range(0, len(row.op), 2)]
        if k > len(prog):
            raise ValueError("no operation left")
        return F._apply_prog(z, prog[:k])
    raise ValueError(fam.cls)


def op_guess_rate(fam, row):
    """Best accuracy without z: the share of z values whose answer is the most common one (as in H44)."""
    counts = {}
    for z in range(fam.z_domain):
        v = F.answer_for(fam, row, z)
        counts[v] = counts.get(v, 0) + 1
    return max(counts.values()) / fam.z_domain


# H45b (Pierre 2026-11-24: decay plus an attention-sharpening log-gain): every row runs GAIN_PASSES passes in
# training and evaluation, never told its depth. fixed6 = plain 6 passes; decay6 = updates scaled by gamma^(k-1);
# gain6 = also by (c_k / c_(k-1))^kappa (model.Core._looped_gain). Effective depth = 1 + sum of the round gains.
GAIN_PASSES = 6
GAIN_ARMS = {"fixed6": "", "decay6": "decay", "gain6": "decaygain"}
# H45c: the loop arm with a LoopRes read (model.CoreConfig.loop_read); passes = steps, as loop
LOOP_READ_ARMS = {"loopres": "loopres", "loopres_content": "loopres_content"}
BLOCKS_PER_ROW = {"fixed": 6, "loop": 6 + 2 * (TRAIN_MAX_PASSES - 1), "uniform3": 6 + 2 * (TRAIN_MAX_PASSES - 1),
                  **{a: 6 + 2 * (GAIN_PASSES - 1) for a in GAIN_ARMS},
                  **{a: 6 + 2 * (TRAIN_MAX_PASSES - 1) for a in LOOP_READ_ARMS}}


def passes_for(arm, passes):
    """The passes a batch runs in training and in the logged accuracy."""
    if arm in GAIN_ARMS:
        return mx.full(passes.shape, GAIN_PASSES)
    return mx.full(passes.shape, TRAIN_MAX_PASSES) if arm == "uniform3" else passes


def build_batch(rng, jrng, fams, n, p_ordinary, M):
    """passes are each row's steps; the uniform3 arm overrides them in train()."""
    """tokens, q, f, a, answers, message (zero on ordinary rows), passes (steps per row)."""
    picks = [fams[rng.randrange(len(fams))] for _ in range(n)]
    rows = [F.sample_row(fam, rng) for fam in picks]
    n_ord = int(round(p_ordinary * n))
    modes = ["ordinary"] * n_ord + ["bus"] * (n - n_ord)
    rng.shuffle(modes)
    toks, q, f, a, ans = [], [], [], [], []
    for r, m in zip(rows, modes):
        ids, qi, fi, ai = F.layout(r, m)
        toks.append(ids); q.append(qi); f.append(fi); a.append(ai); ans.append(r.answer)
    z = mx.array([r.z for r in rows])
    has = mx.array([m == "bus" for m in modes])[:, None]
    msg = C.jitter(M[z], jrng) * has
    passes = mx.array([steps_of(fam) for fam in picks])
    return (*(mx.array(x) for x in (toks, q, f, a, ans)), msg, passes)


# ------------------------------------------------------------------ resume state
def save_resume(out: Path, core, opt, it, rngs, log, t_used):
    """Write a complete state to resume.tmp/, then swap whole directories (review: per-file swaps could pair new
    weights with an old stream). A crash leaves either the old resume/ or, mid-swap, resume.old/ intact."""
    tmp, cur, old = out / "resume.tmp", out / "resume", out / "resume.old"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    mx.save_safetensors(str(tmp / "core.safetensors"), dict(tree_flatten(core.parameters())))
    mx.save_safetensors(str(tmp / "opt.safetensors"), dict(tree_flatten(opt.state)))
    (tmp / "stream.pkl").write_bytes(pickle.dumps({"it": it, "rngs": [r.getstate() for r in rngs], "log": log,
                                                   "train_seconds": t_used}))
    for name in ("core.safetensors", "opt.safetensors", "stream.pkl"):
        _fsync(tmp / name)                             # data on disk before the renames make it current
    _fsync(tmp)
    if cur.exists():
        if old.exists():                               # a kill during an earlier cleanup (review); resume/ is valid
            shutil.rmtree(old)
        cur.rename(old)
    tmp.rename(cur)
    _fsync(out)
    if old.exists():
        shutil.rmtree(old)


def _fsync(path: Path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def resume_dir(out: Path):
    for name in ("resume", "resume.old"):              # resume.old only if a crash hit between the two renames
        if (out / name / "stream.pkl").exists():
            return out / name
    return None


def load_resume(out: Path, core, opt, rngs):
    d = resume_dir(out)
    s = pickle.loads((d / "stream.pkl").read_bytes())
    core.load_weights(str(d / "core.safetensors"))
    opt.state = tree_unflatten(list(mx.load(str(d / "opt.safetensors")).items()))
    for r, st in zip(rngs, s["rngs"]):
        r.setstate(st)
    return s["it"], s["log"], s["train_seconds"]


# ------------------------------------------------------------------ evaluation
def acc(core, tokens, q, f, a, ans, msg, passes):
    """passes=None runs the plain graph (one pass); otherwise the loop port. A LoopRes core (H45c) has no plain
    one-pass graph (its read is part of every pass), so passes=None runs its loop port at one pass."""
    if passes is None and core.cfg.loop_read:
        passes = mx.ones((tokens.shape[0],), dtype=mx.int32)
    hits, n, c = 0, tokens.shape[0], 128                   # 128-row chunks bound memory; the count is exact
    for i in range(0, n, c):
        s = slice(i, i + c)
        b = None if msg is None else BusCall("early", q[s], f[s], msg[s])
        logits = core(tokens[s], b) if passes is None else core(tokens[s], b, passes=passes[s])
        hits += int(mx.sum(mx.argmax(value_logits(logits, a[s]), axis=-1) == ans[s]).item())
    return round(hits / n, 4)


def depth_eval(core, arm, M, n=256):
    """Per depth family, ordinary and bus (clean M[z]) accuracy against the true answer at the family's steps s.
    Loop-port arms also run: s - 1 passes (and s + 1 for graphs), scored against the true answer at that many
    steps *and* against the answer at s, with the oracle agreement between the two as the floor (review: walks
    land on cycles, so a shorter run matches the s answer 30-50% of the time by chance); one pass on the plain
    graph (passes=None); and, for uniform3, its trained 3 passes."""
    out = {}
    for fam in depth_families():
        rng = random.Random(int(hashlib.sha256(f"h45.depth_eval:{fam.fid}".encode()).hexdigest()[:8], 16))
        rows = [F.sample_row(fam, rng) for _ in range(n)]
        s = steps_of(fam)
        row = {"steps": s, "class": fam.cls, "trained_depth": s <= TRAIN_MAX_PASSES,
               "trained_structure": trained_structure(fam),
               "op_guess": round(sum(op_guess_rate(fam, r) for r in rows) / n, 4)}
        for mode in ("ordinary", "bus"):
            tokens, q, f, a, ans, zs = C.batch(rows, mode)
            msg = M[zs] if mode == "bus" else None
            if arm == "fixed":
                row[mode] = acc(core, tokens, q, f, a, ans, msg, None)
                continue
            if arm in GAIN_ARMS:
                six = mx.full((n,), GAIN_PASSES)
                row[mode] = acc(core, tokens, q, f, a, ans, msg, six)
                row[f"{mode}_one_pass"] = acc(core, tokens, q, f, a, ans, msg, None)
                if core.cfg.loop_gain:
                    bus = None if msg is None else BusCall("early", q, f, msg)
                    _, gains, concs = core.looped(tokens, six, bus, return_gains=True)
                    eff = 1 + mx.sum(gains, axis=0)                          # (n,) per row
                    row[f"{mode}_eff_depth"] = round(float(mx.mean(eff).item()), 4)
                    # per-row values for the kill measure (review: family means let rank-only tests pass on noise)
                    row[f"{mode}_eff_depth_rows"] = [round(v, 5) for v in eff.tolist()]
                    row[f"{mode}_gain_by_pass"] = [round(v, 5) for v in mx.mean(gains, axis=1).tolist()]
                    if concs is not None:
                        row[f"{mode}_conc_by_pass"] = [round(v, 5) for v in mx.mean(concs, axis=1).tolist()]
                continue
            row[mode] = acc(core, tokens, q, f, a, ans, msg, mx.full((n,), s))
            row[f"{mode}_one_pass"] = acc(core, tokens, q, f, a, ans, msg, None)
            if arm == "uniform3":
                row[f"{mode}_three_passes"] = acc(core, tokens, q, f, a, ans, msg, mx.full((n,), 3))
            shifts = [-1] + ([1] if fam.cls == "GRAPH" else [])      # FST and programs have no step past the end
            for dk in shifts:
                k = s + dk
                if k < 1:
                    continue
                truth = mx.array([answer_at(fam, r, k) for r in rows])
                lab = "minus1" if dk < 0 else "plus1"
                row[f"{mode}_{lab}_vs_true"] = acc(core, tokens, q, f, a, truth, msg, mx.full((n,), k))
                row[f"{mode}_{lab}_vs_s"] = acc(core, tokens, q, f, a, ans, msg, mx.full((n,), k))
                row[f"{lab}_agreement"] = round(float(mx.mean(truth == ans).item()), 4)
        out[fam.fid] = row
    return out


def summarize(depth):
    """Means per class (GRAPH / FST / PROGRAM) and over all, for trained (1-3) and unseen (4-6) depths."""
    keys = sorted({k for r in depth.values() for k in r if k not in ("steps", "class", "trained_depth",
                                                                       "trained_structure")
                   and isinstance(r[k], (int, float)) and not isinstance(r[k], bool)})   # H45b lists are per-row data
    out = {}
    for cls in ("ALL", "GRAPH", "FST", "PROGRAM"):
        for td, lab in ((True, "trained"), (False, "unseen")):
            sel = [r for r in depth.values() if r["trained_depth"] == td and cls in ("ALL", r["class"])]
            if not sel:
                continue
            out[f"{cls}.{lab}"] = {k: round(sum(r[k] for r in sel if k in r) / len([r for r in sel if k in r]), 4)
                                   for k in keys if any(k in r for r in sel)}
    return out


# ------------------------------------------------------------------ main
_FREE = {"max_seconds": None, "chunk_tag": None, "stop_after": None, "save_every": None}       # per-launch arguments a resume may change
_TERM = {"requested": False}


def _saved_args(out: Path) -> dict:
    """A run's args.json; runs written before --accum existed ran without accumulation (accum 1)."""
    return {"accum": 1} | json.loads((out / "args.json").read_text())


def _on_term(signum, frame):
    """The M5 GPU-lock wrapper sends TERM and waits 30 s before KILL (it fails closed when the CPG radio stops
    answering, 2026-11-23). Finish the current update, save the resume state (about 1 s) and exit 3."""
    _TERM["requested"] = True


def train(args) -> int:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = open(out.parent / f".{out.name}.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"{out}: another launch holds this run directory")
    try:
        return _train(args, out)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def _train(args, out) -> int:
    resuming = resume_dir(out) is not None
    if (out / "receipt.json").exists():
        if _saved_args(out) | _FREE != vars(args) | _FREE:
            raise SystemExit("a finished run with different arguments is already in this directory")
        print("already finished", flush=True)
        return 0
    if not resuming and out.is_dir() and [p.name for p in out.iterdir()] == ["args.json"] and \
            _saved_args(out) | _FREE == vars(args) | _FREE:
        (out / "args.json").unlink()                   # a launch killed before its first save left only its args
    if not resuming:
        out = C.fresh_out(args.out)
        C.write_json(out / "args.json", vars(args))
    elif _saved_args(out) | _FREE != vars(args) | _FREE:
        raise SystemExit("resume: arguments differ from the run being resumed (only --max-seconds and --chunk-tag "
                         "may change)")
    mx.random.seed(args.seed)
    cfg = CoreConfig(bus_width=64, tau=0.5, layers=6, d=256, ffn=1024, heads=4,
                     loop_body=LOOP_BODY if args.arm in ("loop", "uniform3", *GAIN_ARMS, *LOOP_READ_ARMS) else (),
                     loop_gain=GAIN_ARMS.get(args.arm, ""), loop_read=LOOP_READ_ARMS.get(args.arm, ""),
                     loop_key_seed=args.seed if args.arm in LOOP_READ_ARMS else 0)
    core = Core(cfg)                                    # identical init for both arms at a seed (loop_body has no weights)
    core.freeze()
    core.backbone.unfreeze()
    core.ports["early"].unfreeze()
    core.ports["early"].tap.freeze()
    gain_keys = [k for k in ("loop_gamma_off", "loop_kappa_raw", "loopres_qc", "loopres_qr", "loopres_keys")
                 if k in core]
    if gain_keys:
        core.unfreeze(recurse=False, keys=gain_keys)
    M = C.codebook(64)
    fams = train_mix()
    sched = optim.join_schedules(
        [optim.linear_schedule(1e-6, args.lr, args.warmup),
         optim.cosine_decay(args.lr, max(1, args.steps - args.warmup), args.lr * 0.05)], [args.warmup])
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.01)

    def loss_fn(model, tokens, q, f, a, ans, msg, passes):
        bus = BusCall("early", q, f, msg)
        if args.arm == "fixed":
            logits = model(tokens, bus)
        else:
            mp = GAIN_PASSES if args.arm in GAIN_ARMS else TRAIN_MAX_PASSES
            logits = model(tokens, bus, passes=passes_for(args.arm, passes), max_passes=mp)
        return nn.losses.cross_entropy(value_logits(logits, a), ans, reduction="mean")

    grad_fn = nn.value_and_grad(core, loss_fn)
    data_rng, jitter_rng = random.Random(10_000 + args.seed), random.Random(30_000 + args.seed)
    it0, log, t_used = 0, [], 0.0
    opt.init(core.trainable_parameters())
    if resuming:
        it0, log, t_used = load_resume(out, core, opt, [data_rng, jitter_rng])
        print(f"resumed at step {it0}", flush=True)
    state = [core.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(tokens, q, f, a, ans, msg, passes):
        loss, g = grad_fn(core, tokens, q, f, a, ans, msg, passes)
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(core, g)
        return loss

    if args.accum != 1:
        # gradient accumulation (2026-11-24: batch 256 peaks ~4.5 GiB, more than a small shared-GPU memory budget
        # allows). The same 256-row batch is split into `accum` equal micro-batches; the mean of their mean losses
        # and gradients equals the full-batch mean, then one clip and one update, as in the default step.
        if args.accum < 1 or args.batch % args.accum:
            raise SystemExit("--accum must be >= 1 and divide --batch")
        micro_grad = lambda *b: grad_fn(core, *b)                   # noqa: E731 (compiled, it fails to capture)

        @partial(mx.compile, inputs=state, outputs=state)
        def apply(g):
            g, _ = optim.clip_grad_norm(g, 1.0)
            opt.update(core, g)

        def step(*batch):                                           # noqa: F811
            k, m = args.accum, args.batch // args.accum
            total, gsum = None, None
            for i in range(k):
                loss, g = micro_grad(*(x[i * m:(i + 1) * m] for x in batch))
                total = loss if total is None else total + loss
                gsum = g if gsum is None else tree_map(mx.add, gsum, g)
                mx.eval(total, gsum)
            apply(tree_map(lambda x: x / k, gsum))
            return total / k

    t0 = time.time()
    it = it0
    _TERM["requested"] = False
    old_handler = signal.signal(signal.SIGTERM, _on_term)
    try:
        return _loop(args, out, core, opt, step, state, fams, M, data_rng, jitter_rng, it, it0, log, t_used, t0, cfg)
    finally:
        signal.signal(signal.SIGTERM, old_handler)


def _loop(args, out, core, opt, step, state, fams, M, data_rng, jitter_rng, it, it0, log, t_used, t0, cfg):
    while it < args.steps:
        if _TERM["requested"] or (args.max_seconds and time.time() - t0 > args.max_seconds) or \
                (args.stop_after is not None and it - it0 >= args.stop_after):
            save_resume(out, core, opt, it, [data_rng, jitter_rng], log, t_used + time.time() - t0)
            print(f"stopped at step {it} (resume state saved)", flush=True)
            return 3
        it += 1
        batch = build_batch(data_rng, jitter_rng, fams, args.batch, args.p_ordinary, M)
        loss = step(*batch)
        mx.eval(state)
        if it % args.log_every == 0 or it == args.steps:
            row = {"step": it, "loss": float(loss.item()), "elapsed_s": round(t_used + time.time() - t0, 1)}
            if cfg.loop_gain:                                   # the gain trajectory (review)
                row["gamma"] = round(float(core.gamma()), 5)
                if "loop_kappa_raw" in core:
                    row["kappa"] = round(float(core.kappa()), 4)
            if it % (args.log_every * 5) == 0 or it == args.steps:
                rng = random.Random(99)
                tokens, q, f, a, ans, msg, passes = build_batch(rng, random.Random(98), fams, 512, 0.0, M)
                row["acc_bus"] = acc(core, tokens, q, f, a, ans, msg, None if args.arm == "fixed" else
                                     passes_for(args.arm, passes))
            log.append(row)
            print(row, flush=True)
            if not math.isfinite(row["loss"]):
                raise SystemExit("non-finite loss")
        if args.save_every and it % args.save_every == 0 and it < args.steps:
            # periodic autosave (Pierre, 2026-11-23): a hard kill or crash costs at most save_every steps. After the
            # log row and the finite-loss check, so a resume keeps every row and never saves a diverged state
            save_resume(out, core, opt, it, [data_rng, jitter_rng], log, t_used + time.time() - t0)
    if it == args.steps and not (out / "core.safetensors").exists():
        # 6: training is done; save first so a kill during the final save / evaluation resumes straight to it
        save_resume(out, core, opt, it, [data_rng, jitter_rng], log, t_used + time.time() - t0)
    t_used += time.time() - t0
    name = f"ens-h45-core-{args.arm}-early-w64-s{args.seed}"
    ckpt = C.save_core(core, out, name)
    depth = depth_eval(core, args.arm, M)
    receipt = {"schema": "ens.h45.core_receipt.v1", "checkpoint": ckpt, "arm": args.arm, "seed": args.seed,
               "site": "early", "config": core.cfg.to_json(), "args": vars(args), "params": C.n_params(core),
               "loop_body": list(cfg.loop_body), "loop_read": cfg.loop_read,
               "train_max_passes": GAIN_PASSES if args.arm in GAIN_ARMS else TRAIN_MAX_PASSES,
               "loop_gain": {"kind": cfg.loop_gain,
                             "gamma": round(float(core.gamma()), 5) if cfg.loop_gain else None,
                             "kappa": round(float(core.kappa()), 5) if "loop_kappa_raw" in core else None},
               "train_families": [f.fid for f in fams], "extra_families": [f.record() for f in EXTRA],
               "codebook_sha256": C.codebook_digest(64), "families": F.registry_record(),
               "train_seconds": round(t_used, 1), "blocks_per_row_train": BLOCKS_PER_ROW[args.arm], "log": log, "depth_eval": depth, "summary": summarize(depth),
               **C.source_revision(), **C.host_record()}
    C.write_json(out / "receipt.json", receipt)
    print("summary", json.dumps(receipt["summary"]), flush=True)
    return 0


def main(argv=None):
    sys.exit(train(build_parser().parse_args(argv)))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=("loop", "fixed", "uniform3", *GAIN_ARMS, *LOOP_READ_ARMS), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=48000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--accum", type=int, default=1, help="micro-batches per step (gradient accumulation; same "
                   "batch and update, less memory)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--p-ordinary", type=float, default=0.25)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--max-seconds", type=float, default=None, help="stop cleanly after this long; rerun resumes")
    p.add_argument("--save-every", type=int, default=2000,
                   help="autosave the resume state every N updates (0 = off); training is unchanged by it")
    p.add_argument("--stop-after", type=int, default=None,
                   help="stop cleanly after this many updates in this launch (tests); rerun resumes")
    p.add_argument("--chunk-tag", default=None,
                   help="ignored by training; lets an M5 slot watchdog find this process by its chunk directory")
    p.add_argument("--out", required=True)
    return p


if __name__ == "__main__":
    main()
