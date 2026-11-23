"""H7 hierarchical-bus probe: compose k one-step calls of a frozen bus-trained core through the codebook.

A k-step GRAPH walk (succ table, k hops) or FST run (transition table, k input bits) is computed as k calls
of the same frozen core on the one-step structure (graph.<m>-1 / fst.<s>-1). Each call reads z through the
bus and answers op(z); its answer is re-encoded for the next call:

  hard   M[argmax answer]                       (a discrete hand-off)
  soft   sum_v p(v) M[v] over the 64 values      (expected codebook vector)

Controls on the same rows: the single-call oracle (one call on the k-step structure with M[z], the
"bigger core" route), a shuffled hand-off (each call after the first gets another row's previous answer),
and the product of single-step accuracies. Rows are generated ad hoc from a fixed seed; no hidden law is
involved (z is random), so no sealed family is touched.

  python -m research.ens.chain --core <c6 dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import random

import mlx.core as mx

from research.ens import common as C
from research.ens import families as F
from research.ens import vocab as V
from research.ens.contrib import check_core_families, core_site
from research.ens.model import BusCall, value_logits


def _rows(cls: str, m: int, k: int, n: int, seed: int):
    """n chains: (z, per-step one-step Rows sharing one table, the k-step Row, the final answer)."""
    if cls not in ("GRAPH", "FST") or not (2 <= m <= V.N_VALUES) or k < 1 or n < 1:
        raise ValueError("a chain needs GRAPH or FST, at least two representable states, and positive steps and rows")
    rng = random.Random(seed)
    one = F.Family(f"{cls.lower()}.{m}-1", cls, (m, 1), "adhoc")
    many = F.Family(f"{cls.lower()}.{m}-{k}", cls, (m, k), "adhoc")
    out = []
    while len(out) < n:
        z = rng.randrange(m)
        if cls == "GRAPH":
            succ = [rng.randrange(m) for _ in range(m)]
            steps = [tuple([V.tok("GR")] + [V.val(s) for s in succ] + [V.tok("HOP"), V.val(1)])] * k
            full = tuple([V.tok("GR")] + [V.val(s) for s in succ] + [V.tok("HOP"), V.val(k)])
        else:
            table = [(rng.randrange(m), rng.randrange(m)) for _ in range(m)]
            bits = [rng.randrange(2) for _ in range(k)]
            head = [V.tok("FST")] + [v for a, b in table for v in (V.val(a), V.val(b))] + [V.tok("IN")]
            steps = [tuple(head + [V.tok("B1" if b else "B0"), V.tok("END")]) for b in bits]
            full = tuple(head + [V.tok("B1" if b else "B0") for b in bits] + [V.tok("END")])
        ans = F.answer_for(many, F.Row(0, z, full, 0), z)
        if len({F.answer_for(many, F.Row(0, z2, full, 0), z2) for z2 in range(m)}) < 2:
            continue                                   # same non-constant rule as the families
        out.append((z, [F.Row(0, z, s, 0) for s in steps], F.Row(0, z, full, ans), ans))
    return one, many, out


def _call(core, site, rows, msg):
    tokens, q, f, a, _, _ = C.batch(rows, "bus")
    return value_logits(core(tokens, BusCall(site, q, f, msg), check_finite=True), a)


def chain(core, site, cls, m, k, n=512, seed=0) -> dict:
    M = C.codebook(core.cfg.bus_width)
    one, many, data = _rows(cls, m, k, n, seed)
    if k < 2:
        raise ValueError("a chain needs k >= 2")
    zs = mx.array([d[0] for d in data])
    target = mx.array([d[3] for d in data])
    train_struct = {(f.cls, f.knobs) for f in F.split("train")}
    res = {"cls": cls, "m": m, "k": k, "n": n, "single_call_in_training": (cls, (m, k)) in train_struct}
    # true intermediates, and per-step accuracy from a clean (oracle) input: the independence baseline
    inter = [zs.tolist()]
    for i in range(k):
        inter.append([F.answer_for(one, d[1][i], s) for d, s in zip(data, inter[-1])])
    clean = []
    for i in range(k):
        pred = mx.argmax(_call(core, site, [d[1][i] for d in data], M[mx.array(inter[i])]), -1)
        clean.append(float(mx.mean(pred == mx.array(inter[i + 1])).item()))
    res["step_acc_clean_input"] = clean
    prod = 1.0
    for c in clean:
        prod *= c
    res["product_of_clean_steps"] = prod
    # single-call oracle on the k-step structure
    res["single_call_oracle"] = float(mx.mean(mx.argmax(_call(core, site, [d[2] for d in data], M[zs]), -1)
                                              == target).item())
    perm = mx.array(random.Random(seed + 1).sample(range(n), n))
    for mode in ("hard", "soft", "hard_shuffled"):
        msg = M[zs]
        step_acc = []
        state = zs
        all_ok = mx.ones((n,), dtype=mx.bool_)
        soft_rms = []
        for i in range(k):
            step_rows = [d[1][i] for d in data]
            logits = _call(core, site, step_rows, msg)
            pred = mx.argmax(logits, -1)
            # per-step accuracy against the true intermediate, given this step's actual input
            # (a previous step may have answered outside the domain; that step was already wrong, and
            # this step's "truth" is then undefined and counted as a miss)
            truth = mx.array([F.answer_for(one, r, s) if 0 <= s < m else -1
                              for r, s in zip(step_rows, state.tolist())])
            step_acc.append(float(mx.mean(pred == truth).item()))
            if mode == "hard":
                all_ok = all_ok & (pred == mx.array(inter[i + 1]))
            state = pred
            if mode == "soft":
                msg = mx.softmax(logits, axis=-1) @ M
                soft_rms.append(float(mx.sqrt(mx.mean(msg * msg)).item()))
            else:
                msg = M[pred]
            if mode == "hard_shuffled" and i == 0:
                msg = msg[perm]
                state = state[perm]
        res[f"chain_{mode}"] = float(mx.mean(state == target).item()) if mode != "soft" else \
            float(mx.mean(mx.argmax(logits, -1) == target).item())
        res[f"step_acc_{mode}"] = step_acc
        if mode == "hard":
            res["all_steps_correct"] = float(mx.mean(all_ok).item())
        if mode == "soft":
            res["soft_msg_rms"] = soft_rms        # the core was trained on RMS 0.7-1.3
    # a wrong intermediate can still land on the right final value (shared successors):
    res["collision_recovery"] = res["chain_hard"] - res["all_steps_correct"]
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=512)
    args = ap.parse_args(argv)
    out = C.fresh_out(args.out)
    core, ckpt = C.load_core(args.core)
    core.freeze()
    check_core_families(args.core)
    rec = json.loads(open(f"{args.core}/receipt.json").read())
    if rec["arm"] not in ("c6", "c6r", "c6a", "c6l"):
        raise SystemExit("the chain probe needs a bus-reading arm (c6, c6r, c6a or c6l)")
    site = core_site(args.core)
    cells = [("GRAPH", 6, 2), ("GRAPH", 8, 2), ("GRAPH", 8, 3), ("GRAPH", 8, 4),
             ("FST", 3, 2), ("FST", 3, 3), ("FST", 3, 4), ("FST", 2, 4)]
    results = [chain(core, site, c, m, k, args.n) for c, m, k in cells]
    receipt = {"schema": "ens.a0.chain_receipt.v0", "hypothesis": "H7", "core_checkpoint": ckpt, "arm": rec["arm"],
               "site": site, "results": results, **C.source_revision(), **C.host_record()}
    C.write_json(out / "receipt.json", receipt)
    for r in results:
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}))


if __name__ == "__main__":
    main()
