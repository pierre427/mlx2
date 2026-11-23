"""H17 self-claiming tiles: each admitted tile carries its own claim head; the highest calibrated claim writes.

Each tile's claim head is trained in isolation, with no knowledge of the other tiles:
  positives  its own family's training rows
  negatives  a shared, published negative pool: rows of every dev/route structure with uniformly random keys
             and random z (unlabeled with respect to any tile, a positive-unlabeled setup)
The head is a logistic unit on the core's site state at the SLOT position (the key read is a control only).
Each head is Platt-calibrated on a calibration half of its own held rows against fresh negatives, and its
probability threshold is set there to a common recall target; evaluation uses the other half.

At evaluation on the mixed held pool of all admitted tiles' families:
  self_claim     the tile with the highest calibrated score writes; no score above 0 -> bypass (fail closed)
  oracle         the row's own tile
  add_tile       a third tile (another family) admitted later with its own head; the first two are not retrained
Reports accuracy, claim accuracy, abstain rate, and the abstain rate on a fresh negative pool.

  python -m research.ens.claim --core <c6 dir> --modules <dir> <dir> [<dir> ...] --out <dir>
"""
from __future__ import annotations

import argparse
import json
import random
import hashlib
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from research.ens import common as C
from research.ens import families as F
from research.ens.contrib import check_core_families, core_site, load_module
from research.ens.evaluate import module_messages
from research.ens.model import BusCall, value_logits
from research.ens.route import key_split, site_hidden

RECALL = 0.95
NEG_STRUCTS = [f for f in F.FAMILIES if f.split in ("dev", "route_dev")]


def negative_pool(n, seed, structs=None, exclude_held=True):
    """The shared published negative pool. Rows with operations from a structure's held bucket are left out,
    so the heads never see held operations as negatives while positives never contain them (review F2)."""
    structs = structs or NEG_STRUCTS
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        fam = structs[rng.randrange(len(structs))]
        r = F.sample_row(fam, rng)
        if exclude_held and F._is_held(fam, r.signature):
            continue
        out.append(r)
    return out


def _pick(h, q, f, where):
    idx = mx.arange(h.shape[0])
    if where == "both":                     # structure (SLOT, after the op) and key identity (key position)
        return mx.concatenate([h[idx, f - 1], h[idx, q]], axis=-1)
    return h[idx, q if where == "key" else f - 1]


def _read(core, site, rows, where):
    tokens, q, f, *_ = C.batch(rows, "bus")
    return _pick(site_hidden(core, tokens, site), q, f, where)


def owned_rows(fams, i, split, part=None):
    """Tile i's rows. Two tiles on one structure each own half the key space (as in H8). Held rows are split
    into a calibration half and an evaluation half by operation hash (part = 'cal' / 'eval'; review F2)."""
    rows = F.family_rows(fams[i], split)
    for j, other in enumerate(fams):
        if j != i and (other.cls, other.knobs) == (fams[i].cls, fams[i].knobs):
            a, b = (i, j) if i < j else (j, i)
            keys = key_split(fams[a], fams[b])[0 if i == a else 1]
            rows = [r for r in rows if r.x in keys]
    if part is not None:
        # an independent hash: held ops already share crc32 % held_div == 0, and crc32 is affine over GF(2),
        # so no crc-derived bit is independent of that; use sha256
        rows = [r for r in rows if (hashlib.sha256(b"ens.claim.cal\0" + repr(r.op).encode()).digest()[0] & 1)
                == (0 if part == "cal" else 1)]
    return rows


def train_head(core, site, owners, i, where, *, steps=1500, batch=128, seed=0):
    """Claim head for tile i, knowing only its own rows (owners: the tiles admitted when it was built)."""
    pos = owned_rows(owners, i, "train")
    neg = negative_pool(4096, 1000 + seed)
    mx.random.seed(seed + i)
    head = nn.Linear(core.cfg.d * (2 if where == "both" else 1), 1)
    opt = optim.Adam(learning_rate=1e-3)
    rng = random.Random(seed + i)

    def lf(hd, h, y):
        return nn.losses.binary_cross_entropy(hd(h)[:, 0], y, with_logits=True, reduction="mean")
    g = nn.value_and_grad(head, lf)
    for _ in range(steps):
        rows = [pos[rng.randrange(len(pos))] for _ in range(batch // 2)] + \
               [neg[rng.randrange(len(neg))] for _ in range(batch // 2)]
        y = mx.array([1.0] * (batch // 2) + [0.0] * (batch // 2))
        _, gr = g(head, _read(core, site, rows, where), y)
        opt.update(head, gr)
        mx.eval(head.parameters(), opt.state)
    # Platt calibration on the calibration half (own cal rows vs fresh negatives), then a probability
    # threshold at RECALL on the same calibration half; evaluation never sees these rows (review F2, F6)
    cal_pos = head(_read(core, site, owned_rows(owners, i, "held", "cal"), where))[:, 0]
    cal_neg = head(_read(core, site, negative_pool(len(cal_pos.tolist()), 5000 + seed + i), where))[:, 0]
    x = mx.concatenate([cal_pos, cal_neg]); y = mx.concatenate([mx.ones_like(cal_pos), mx.zeros_like(cal_neg)])
    ab = mx.array([1.0, 0.0])
    o2 = optim.Adam(learning_rate=0.05)
    pg = mx.value_and_grad(lambda p: nn.losses.binary_cross_entropy(p[0] * x + p[1], y, with_logits=True,
                                                                     reduction="mean"))
    for _ in range(300):
        _, gp = pg(ab)
        ab = ab - 0.05 * gp
    prob = mx.sigmoid(ab[0] * cal_pos + ab[1])
    t = float(mx.sort(prob)[int((1 - RECALL) * prob.shape[0])].item())
    return {"head": head, "platt": ab, "t": t}


def _score(h, hd):
    return mx.sigmoid(hd["platt"][0] * hd["head"](h)[:, 0] + hd["platt"][1]) - hd["t"]


def _evaluate(core, site, mods, fams, heads, where, label):
    k = len(heads)
    rows, lab = [], []
    for i in range(k):
        r = owned_rows(fams[:k], i, "held", "eval")
        rows += r; lab += [i] * len(r)
    tokens, q, f, a, ans, zs = C.batch(rows, "bus")
    msgs = mx.stack([module_messages(core, site, m, tokens, q, f, zs) for m in mods[:k]], 1)
    hr = _pick(site_hidden(core, tokens, site), q, f, where)
    score = mx.stack([_score(hr, heads[i]) for i in range(k)], 1)
    choice = mx.argmax(score, -1)
    claimed = mx.max(score, -1) >= 0
    lab_a = mx.array(lab); idx = mx.arange(len(rows))
    byp = mx.argmax(value_logits(core(tokens), a), -1)

    def pred(msg):
        return mx.argmax(value_logits(core(tokens, BusCall(site, q, f, msg)), a), -1)
    oracle_hit = pred(msgs[idx, lab_a]) == ans
    sc_hit = mx.where(claimed, pred(msgs[idx, choice]), byp) == ans
    res = {"n": len(rows), "families": [x.fid for x in fams[:k]],
           "oracle": float(mx.mean(oracle_hit).item()), "self_claim": float(mx.mean(sc_hit).item()),
           "bypass": float(mx.mean(byp == ans).item()),
           "claim_acc": float(mx.mean((choice == lab_a) & claimed).item()),
           "abstain": float(mx.mean(~claimed).item()),
           "per_family": {fams[i].fid: float((mx.sum(sc_hit & (lab_a == i)) / mx.sum(lab_a == i)).item())
                          for i in range(k)}}
    # fresh negatives only from structures no admitted tile uses (review F4)
    used = {(x.cls, x.knobs) for x in fams[:k]}
    foreign = [x for x in F.FAMILIES if x.split in ("dev", "confirm", "route_dev") and (x.cls, x.knobs) not in used
               and x.split != "confirm"]
    if foreign:
        hn = _read(core, site, negative_pool(1024, 9000, foreign, exclude_held=False), where)
        sn = mx.stack([_score(hn, heads[i]) for i in range(k)], 1)
        res["abstain_on_foreign_structures"] = float(mx.mean(mx.max(sn, -1) < 0).item())
    return res


def run(core, site, mods, fams, where, *, seed=0) -> dict:
    if len(fams) == 3 and (fams[2].cls, fams[2].knobs) in {(x.cls, x.knobs) for x in fams[:2]}:
        raise SystemExit("the added tile must not share a structure with the first two (ownership would change)")
    pair = [train_head(core, site, fams[:2], i, where, seed=seed) for i in range(2)]   # built before tile 3 exists
    out = {"read": where, "recall_target": RECALL,
           "pair": _evaluate(core, site, mods, fams, pair, where, "pair")}
    if len(fams) == 3:
        third = train_head(core, site, fams, 2, where, seed=seed)                        # tiles 0, 1 untouched
        out["add_tile"] = _evaluate(core, site, mods, fams, pair + [third], where, "add_tile")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core", required=True)
    ap.add_argument("--modules", nargs="+", required=True, help="2 tiles, optionally a 3rd added later")
    ap.add_argument("--reads", nargs="+", default=["slot", "key", "both"],
                    help="slot is the claim read; key is a control; both concatenates them (H17 follow-up)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if len(args.modules) not in (2, 3):
        ap.error("--modules requires two tiles and optionally one additional tile")
    out = C.fresh_out(args.out)
    core, ckpt = C.load_core(args.core)
    core.freeze()
    check_core_families(args.core)
    site = core_site(args.core)
    mods, fams = [], []
    for d in args.modules:
        rec = json.loads((Path(d) / "receipt.json").read_text())
        fam = F.BY_ID[rec["family"]["fid"]]
        if rec["core_checkpoint"] != ckpt or rec["family_digest"] != fam.digest or fam.split in F.SEALED:
            raise SystemExit(f"{d}: wrong core, wrong law, or sealed family")
        if rec.get("kind", "mlp") == "oracle-table":
            raise SystemExit(f"{d}: oracle tables read the true intermediate; the claim probe needs contributors")
        if fam in fams:
            raise SystemExit(f"{d}: duplicate claim family {fam.fid}")
        if rec.get("site") != site:
            raise SystemExit(f"{d}: site mismatch")
        mods.append(load_module(d, core.cfg.bus_width)); fams.append(fam)
    # SLOT is the claim read; the key read is a labelled control (it sees only [BOS Q x]; review F1)
    res = {w: run(core, site, mods, fams, w) for w in args.reads}
    receipt = {"schema": "ens.a0.claim_receipt.v0", "hypothesis": "H17", "core_checkpoint": ckpt, "site": site,
               "modules": args.modules, "results": res, **C.source_revision(), **C.host_record()}
    C.write_json(out / "receipt.json", receipt)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
