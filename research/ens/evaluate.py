"""Held evaluation, controls and the BUS-001 paired intervention (spec v0.4 section 6.1, step 5).

For one (core, module, family) on the family's held pool, on identical prompts:

  on              the module writes the core's bus site                     (primary)
  bypass          no bus call (the ordinary graph)
  random          norm-matched Gaussian message per row
  wrong           the partner row's message, scored against the original target (should fall)
  swap            the same message, scored against the counterfactual target op(z') (should rise)
  both            per pair: original right with its own message AND swapped right with the partner's
  untrained_port  the module's message written through a port the core never trained (another site)

Partners are chosen so the *targets* differ, op(z') != op(z) for the row's own op, not merely
z' != z: on many-to-one ops (GRAPH, FST) equal targets would let a prompt-only predictor score
'both' (review finding, 2026-11-21). Rows with no such partner are left out of the pair metrics
and counted in n_pairs.

Oracle ceilings are labelled by channel and are not comparable across arms:
  oracle_codebook / oracle_codebook_swap / oracle_codebook_both   bus arms, published codebook M[z]
  oracle_typed                                                     c7, z as an ordinary typed token
The bus-capacity ceiling for both arms is a separate oracle-table module (contrib --kind oracle-table).

  python -m research.ens.evaluate --core <core dir> --module <module dir> --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import mlx.core as mx
import numpy as np

from research.ens import common as C
from research.ens import families as F
from research.ens.contrib import check_core_families, core_site, load_module, message
from research.ens.model import BusCall, value_logits


CEILING_MIN = 0.9      # A0 re-scope: an extension family counts only where the oracle reaches this


def _acc(pred, target, mask=None):
    hit = (pred == target)
    if mask is None:
        return float(mx.mean(hit).item())
    return float((mx.sum(hit & mask) / mx.maximum(mx.sum(mask), 1)).item())


def pair_rows(fam: F.Family, rows, seed=0):
    """Partner index per row with a different target under the row's own op; -1 when none exists."""
    rng = random.Random(seed)
    by_z = {}
    for i, r in enumerate(rows):
        by_z.setdefault(r.z, []).append(i)
    partners = []
    for r in rows:
        ok = [z for z in by_z if z != r.z and F.answer_for(fam, r, z) != r.answer]
        if not ok:
            partners.append(-1)
            continue
        z2 = ok[rng.randrange(len(ok))]
        partners.append(by_z[z2][rng.randrange(len(by_z[z2]))])
    return partners


def module_messages(core, site, module, tokens, q, f, zs):
    """The module's messages for a batch, captured from one core pass."""
    src = message(module, tokens, zs, q)
    if not callable(src):
        return src
    box = {}

    def provider(b):
        box["m"] = src(b)
        return box["m"]
    core(tokens, BusCall(site, q, f, provider))
    return box["m"]


def other_port(core, site):
    names = [n for n, after in core.cfg.sites if n != site and -1 <= after < core.cfg.layers]
    if not names:
        raise ValueError("untrained-port control requires another reachable site")
    return "late" if "late" in names else names[-1]


def quantize_table(module, mode: str, width: int):
    """Store a table tile cheaply (H11) and return (module, bytes per entry). In place on module.table."""
    W = module.table.weight
    if mode == "none":
        return module, 4 * width
    if mode in ("int8", "int4"):
        levels = 127 if mode == "int8" else 7
        # per-row scale; unit_rms after the lookup cancels any positive row scale, so it is not stored
        scale = mx.maximum(mx.max(mx.abs(W), axis=-1, keepdims=True), 1e-12) / levels
        module.table.weight = mx.round(W / scale) * scale
        return module, width if mode == "int8" else (width + 1) // 2
    if mode == "codebook":                                                 # snap each row to its nearest M row
        M = C.codebook(width)
        Wn = W / mx.sqrt(mx.mean(W * W, axis=-1, keepdims=True))
        idx = mx.argmax(Wn @ M.T, axis=-1)
        module.table.weight = M[idx]
        return module, 1
    raise ValueError(mode)


def _half_mask(w, half):
    if half == "full":
        return None
    keep = mx.arange(w) < w // 2 if half == "low" else mx.arange(w) >= w // 2
    return keep.astype(mx.float32)[None, :]


def evaluate(core, arm: str, site: str, module, fam: F.Family, rows, *, seed=0, half="full") -> dict:
    tokens, q, f, a, ans, zs = C.batch(rows, "bus")
    partners = pair_rows(fam, rows, seed)
    has = mx.array([j >= 0 for j in partners])
    p = mx.array([j if j >= 0 else i for i, j in enumerate(partners)])
    cf_target = mx.array([F.answer_for(fam, r, rows[j].z) if j >= 0 else r.answer
                          for r, j in zip(rows, partners)])

    def run(bus, toks=tokens):
        return mx.argmax(value_logits(core(toks, bus, check_finite=True), a), axis=-1)

    counts = {}
    for r in rows:
        counts[r.answer] = counts.get(r.answer, 0) + 1
    res = {"n": len(rows), "n_pairs": int(mx.sum(has).item()), "site": site,
           "n_ops": len({r.op for r in rows}),
           "majority": max(counts.values()) / len(rows),      # best constant answer
           "zblind": zblind_bayes(fam, rows)}                 # best answer from the op without z (review F1)
    hm = _half_mask(core.cfg.bus_width, half)
    if module is not None:
        msg = module_messages(core, site, module, tokens, q, f, zs)
        if hm is not None:                 # H26: only part of the bus is written
            msg = msg * hm
        on = run(BusCall(site, q, f, msg))
        res["on"] = _acc(on, ans)
        res["bypass"] = _acc(run(None), ans)
        norms = mx.linalg.norm(msg, axis=-1, keepdims=True)
        g = mx.random.normal(msg.shape, key=mx.random.key(seed))
        res["random"] = _acc(run(BusCall(site, q, f, g / mx.linalg.norm(g, axis=-1, keepdims=True) * norms)), ans)
        wrong = run(BusCall(site, q, f, msg[p]))
        res["wrong"] = _acc(wrong, ans, has)
        res["swap"] = _acc(wrong, cf_target, has)
        res["both"] = _acc((on == ans) & (wrong == cf_target), mx.array(True), has)
        res["untrained_port"] = _acc(run(BusCall(other_port(core, site), q, f, msg)), ans)
        res["msg_rms"] = float(mx.sqrt(mx.mean(msg * msg)).item())
    if arm == "c7":
        t_tokens, *_ = C.batch(rows, "typed")
        res["oracle_typed"] = _acc(run(None, t_tokens), ans)
    else:
        M = C.codebook(core.cfg.bus_width)
        if hm is not None:
            M = M * hm
        o = run(BusCall(site, q, f, M[zs]))
        o2 = run(BusCall(site, q, f, M[zs[p]]))
        res["oracle_codebook"] = _acc(o, ans)
        res["oracle_codebook_swap"] = _acc(o2, cf_target, has)
        res["oracle_codebook_both"] = _acc((o == ans) & (o2 == cf_target), mx.array(True), has)
    # The module ceiling is the bus channel the module writes. c7's port never trained, so the published
    # codebook means nothing there; its bus capacity is the oracle-table probe, and oracle_typed is c7's
    # own typed-field reference, not a module ceiling (review F2).
    res["ceiling_ok"] = None if arm == "c7" else res["oracle_codebook"] >= CEILING_MIN
    return res


def zblind_bayes(fam: F.Family, rows) -> float:
    """Accuracy of the Bayes predictor that reads the op but not z, z ~ the hidden law's marginal."""
    pz = {}
    for z in fam.hidden:
        pz[z] = pz.get(z, 0) + 1
    hit = 0
    for r in rows:
        score = {}
        for z, c in pz.items():
            a = F.answer_for(fam, r, z)
            score[a] = score.get(a, 0) + c
        best = max(score.values())
        winners = [a for a, v in score.items() if v == best]
        hit += (r.answer in winners) / len(winners)          # ties split evenly
    return hit / len(rows)


def ordinary_suite(core, n=2048, seed=7) -> float:
    """Ordinary-task accuracy (z given in the prompt, no provider) on fresh train-family rows."""
    rng = random.Random(seed)
    fams = F.split("train")
    rows = [F.sample_row(fams[rng.randrange(len(fams))], rng) for _ in range(n)]
    tokens, q, f, a, ans, _ = C.batch(rows, "ordinary")
    return _acc(mx.argmax(value_logits(core(tokens), a), axis=-1), ans)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core", required=True)
    ap.add_argument("--module", help="module dir; omit for oracle-only")
    ap.add_argument("--family", help="required without --module")
    ap.add_argument("--split", default="held", choices=("held", "train"))
    ap.add_argument("--unseal-confirm", action="store_true")
    ap.add_argument("--half", default="full", choices=("full", "low", "high"),
                    help="H26: score with only the low or high half of the bus written (module and oracle messages)")
    ap.add_argument("--quant", default="none", choices=("none", "int8", "int4", "codebook"),
                    help="table modules only: evaluate the tile stored at this precision (H11)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.family and F.BY_ID[args.family].split in F.SEALED and not args.unseal_confirm:
        raise SystemExit(f"{args.family} is sealed; pass --unseal-confirm after the decision record")
    out = C.fresh_out(args.out)
    core, ckpt = C.load_core(args.core)
    core.freeze()
    check_core_families(args.core)
    crec = json.loads((Path(args.core) / "receipt.json").read_text())
    arm, site = crec["arm"], core_site(args.core)
    module, mrec = None, None
    if args.module:
        mrec = json.loads((Path(args.module) / "receipt.json").read_text())
        if mrec["core_checkpoint"] != ckpt:
            raise SystemExit(f"module built for {mrec['core_checkpoint']}, not {ckpt}")
        fam = F.BY_ID[mrec["family"]["fid"]]
        if mrec["family_digest"] != fam.digest:
            raise SystemExit(f"{fam.fid}: module was trained on a different hidden law "
                             f"({mrec['family_digest'][:12]} != {fam.digest[:12]})")
        module = load_module(args.module, core.cfg.bus_width)
        if args.quant != "none" and not hasattr(module, "table"):
            raise SystemExit("--quant applies to table modules only")
        entry_bytes = None
        if hasattr(module, "table"):
            module, entry_bytes = quantize_table(module, args.quant, core.cfg.bus_width)
        elif args.quant != "none":
            raise SystemExit("--quant applies to table modules only")
    else:
        fam = F.BY_ID[args.family]
    if fam.split in F.SEALED and not args.unseal_confirm:     # either split: oracle ceilings are results too
        raise SystemExit(f"{fam.fid} is sealed; pass --unseal-confirm after the decision record")
    rows = F.family_rows(fam, args.split)
    res = evaluate(core, arm, site, module, fam, rows, half=args.half)
    res["half"] = args.half
    if module is not None and getattr(module, "kind", "") == "key-choice":
        res["quant"] = "choice"
        res["table_entries"] = int(module.logits.shape[0])
        res["table_bytes"] = res["table_entries"]           # one codebook index per key
        idx = mx.argmax(module.logits, axis=-1).tolist()
        res["codebook_index_match"] = sum(i == w for i, w in zip(idx, fam.hidden)) / len(idx)
    if module is not None and hasattr(module, "table"):
        snapped = getattr(module, "snap", False)
        res["quant"] = args.quant
        res["snap_codebook"] = snapped
        res["table_entries"] = int(module.table.weight.shape[0])
        # a snapped tile serves codebook rows whatever --quant says, so it is stored as 1-byte indices
        res["table_bytes"] = res["table_entries"] * (1 if snapped else entry_bytes)
        res["table_weight_sha256"] = hashlib.sha256(
            np.array(module.table.weight).astype("<f4").tobytes()).hexdigest()
        if (args.quant == "codebook" or snapped) and module.kind in ("key-table", "oracle-table"):
            M = C.codebook(core.cfg.bus_width)
            idx = mx.argmax(module.table.weight @ M.T, axis=-1).tolist()
            want = list(fam.hidden) if module.kind == "key-table" else list(range(len(idx)))
            res["codebook_index_match"] = sum(i == w for i, w in zip(idx, want)) / len(idx)
    res["ordinary_suite"] = ordinary_suite(core)
    receipt = {"schema": "ens.a0.eval_receipt.v1", "core_checkpoint": ckpt, "arm": arm, "family": fam.fid,
               "quant": res.get("quant", "none"), "snap_codebook": res.get("snap_codebook", False),
               "family_digest": fam.digest, "split": args.split,
               "module_kind": mrec.get("kind", "mlp") if mrec else None,
               "module_package_digest": mrec["package_digest"] if mrec else None,
               "results": res, **C.source_revision(), **C.host_record()}
    C.write_json(out / "receipt.json", receipt)
    print(json.dumps(receipt["results"], indent=1))


if __name__ == "__main__":
    main()
