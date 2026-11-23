"""H41 v2a/v2b: backward endpoint->evidence edges on a recall-trained core (k = 1, CPU).

v2b adds "backward-attributed": edge values read off the core by occlusion (attributed_values), no label.

v1 (research/ens/memtile.py) showed the backward arm cannot work on a C6 core: the endpoint (field) state never
depends on the cue. A recall-trained core (research/ens/recall_core.py) must carry the cue to the endpoint to
answer in-context recall rows. v2a asks only the first question: does the endpoint state now identify the cue
across contexts and ops, well enough to key a memory?

Per world: a table T (16 cue keys -> distinct ends in the z domain). Earlier sessions hold the facts
"GR K(x) V(T[x])" with distractor facts, and some cues are queried in-session (a recall row of the family, facts
in context, op o1): those are the "used" cues.

  backward-oracle   for used cues: key = the endpoint (field) state of the in-session query, value = T[x] (the true
                    evidence, supplied: an upper bound on use attribution, v2b replaces it with attention-attributed
                    evidence). Later query: a bus row with the same cue, another op o2, no facts in context.
  knn               every fact, key = the state at the fact's cue token in its session; query = the later row's cue
                    state (the v1 arm, here on the recall core).
  symbolic / none   ceiling and floor.

Key spaces as in v1 (raw, endpoint, orth, anchored) plus "+learned": a linear encoder trained contrastively on
TRAIN-pool rows and training tables only: (in-session endpoint state, later endpoint state) for the same cue, and
(fact cue state, query cue state) for knn.

  python -m research.ens.memtile_v2 --core <recall core dir> --out <dir>
"""
from __future__ import annotations

import argparse
import random

import mlx.core as mx

from research.ens import common as C
from research.ens import families as F
from research.ens import memtile as MT
from research.ens import vocab as V
from research.ens.contrib import core_site
from research.ens.recall_core import strip


def world(rng, z_domain):
    return rng.sample(range(z_domain), V.N_KEYS)              # distinct ends: no cross-cue credit


def session_rows(rng, table, cues, n_facts, distract_pool, T):
    """Fact-only sessions covering every cue once (plus distractor keys' facts from another table)."""
    order = list(cues)
    rng.shuffle(order)
    out = []
    per = n_facts
    for i in range(0, len(order), per):
        ks = order[i:i + per]
        facts = [(V.key(k), V.val(table[k])) for k in ks]
        ids = [V.BOS] + [t for a, b in facts for t in (V.tok("GR"), a, b)]
        out.append(ids + [V.PAD] * (T - len(ids)))
    return out


def insession_query(rng, fam, row, table, x, n_facts, T):
    """A recall row: facts in context (x among them, values from the world table), then the query."""
    keys = [x] + rng.sample([k for k in range(V.N_KEYS) if k != x], n_facts - 1)
    rng.shuffle(keys)
    facts = [t for k in keys for t in (V.tok("GR"), V.key(k), V.val(table[k]))]
    tail = strip(F.layout(row, "bus")[0])[1:]
    ids = [V.BOS] + facts + tail
    return ids + [V.PAD] * (T - len(ids)), len(ids) - 2      # field index


def attributed_values(core, ins, n_facts, z_domain):
    """v2b: the evidence each in-session query used, read off the core itself by occlusion (no generator label).
    For every fact j of the row, replace its value token by values ABSENT from the row (averaged over 3, so a
    non-injective op cannot hide the true fact; review 2026-11-23) and measure the mean drop in the
    log-probability of the core's own predicted answer at ANS; the fact with the largest drop is the evidence, and
    its value token (as it appears in the row) is the edge value. Returns (values, top-margin per row)."""
    rows = [i[0] for i in ins]
    a_idx = mx.array([i[1] + 1 for i in ins])
    base = MT.value_logits(core(mx.array(rows)), a_idx)
    pred = mx.argmax(base, axis=-1)
    lp0 = base - mx.logsumexp(base, axis=-1, keepdims=True)
    lp0 = mx.take_along_axis(lp0, pred[:, None], axis=-1)[:, 0]
    drops = []
    for j in range(n_facts):
        vpos = 1 + 3 * j + 2
        acc = 0
        for rep in range(3):
            occl = []
            for r in rows:
                rr = list(r)
                present = {rr[1 + 3 * i + 2] - V.V0 for i in range(n_facts)}
                absent = [v for v in range(z_domain) if v not in present]
                rr[vpos] = V.val(absent[(j * 3 + rep) % len(absent)])
                occl.append(rr)
            lg = MT.value_logits(core(mx.array(occl)), a_idx)
            lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
            acc = acc + (lp0 - mx.take_along_axis(lp, pred[:, None], axis=-1)[:, 0])
        drops.append(acc / 3)
    D = mx.stack(drops, axis=1)                                   # (rows, facts)
    best = mx.argmax(D, axis=1).tolist()
    srt = mx.sort(D, axis=1)
    margin = (srt[:, -1] - srt[:, -2]).tolist()
    vals = [rows[i][1 + 3 * b + 2] for i, b in enumerate(best)]
    return vals, margin


def states_at(core, rows, idx, site, T):
    S = MT.site_states(core, MT.pad(rows, T), site)
    return S[mx.arange(len(rows)), mx.array(idx)]


def run(args) -> dict:
    out = C.fresh_out(args.out)
    core, ckpt = C.load_core(args.core)
    core.freeze()
    site, inject = args.key_site, core_site(args.core)
    if site == "ingress":
        raise SystemExit("--key-site ingress: every later field state is the same vector there (v1 review)")
    fam = F.BY_ID[args.family]
    M, T = C.codebook(core.cfg.bus_width), core.cfg.seq_len
    rng = random.Random(f"ens.memtile_v2:{args.seed}")
    held, train = F.family_rows(fam, "held"), F.family_rows(fam, "train")
    by_x = lambda pool: {x: [r for r in pool if r.x == x] for x in range(V.N_KEYS)}
    hx, tx = by_x(held), by_x(train)
    ord_rows = [F.layout(r, "ordinary")[0] for r in train[:256]]
    ot = MT.pad(ord_rows, T)
    Hall = MT.site_states(core, ot, site).reshape(-1, core.cfg.d)
    H = Hall[mx.array([i for i, t in enumerate(ot.reshape(-1).tolist()) if t != V.PAD])]
    spaces = {s: MT.KeySpace(core, inject, H, s, r=args.rank) for s in MT.SPACES}
    fit_log = {}
    if args.learn:
        A, B = training_pairs(core, fam, tx, site, T, random.Random(f"ens.memtile_v2.train:{args.seed}"), args)
        for s_ in MT.SPACES:
            lk = MT.LearnedKeys(spaces[s_], core.cfg.d, steps=args.learn_steps, seed=args.seed)
            fit_log[s_ + "+learned"] = lk.fit(A, B, random.Random(args.seed))
            spaces[s_ + "+learned"] = lk
    results, recall_check, attribution_check = [], [], []
    for w in range(args.worlds):
        table = world(rng, fam.z_domain)
        cues = [x for x in range(V.N_KEYS) if len(hx[x]) >= 2]
        used = {x for x in cues if rng.random() < args.p_used}
        sessions = session_rows(rng, table, cues, args.n_facts, None, T)
        # later queries (bus rows, no facts) and in-session queries (recall rows, other op)
        qrows, srows = [], []
        for x in cues:
            r, r_s = rng.sample(hx[x], 2)
            qrows.append(F.Row(x=x, z=table[x], op=r.op, answer=F.answer_for(fam, r, table[x])))
            srows.append(F.Row(x=x, z=table[x], op=r_s.op, answer=F.answer_for(fam, r_s, table[x])))
        Q = [MT.query_row(fam, r) for r in qrows]
        Hq_cue = states_at(core, [q[0] for q in Q], [q[1] for q in Q], site, T)
        Hq_end = states_at(core, [q[0] for q in Q], [q[2] for q in Q], site, T)
        ins = [insession_query(rng, fam, r, table, r.x, args.n_facts, T) for r in srows]
        Hs_end = states_at(core, [i[0] for i in ins], [i[1] for i in ins], site, T)
        # does the core actually recall in context on these held-op rows? (no bus)
        toks = MT.pad([i[0] for i in ins], T)
        a_idx = mx.array([i[1] + 1 for i in ins])
        pred = mx.argmax(MT.value_logits(core(toks), a_idx), axis=-1).tolist()
        recall_check.append(sum(p == r.answer for p, r in zip(pred, srows)) / len(srows))
        S = MT.site_states(core, MT.pad(sessions, T), site)
        att_vals, att_margin = attributed_values(core, ins, args.n_facts, fam.z_domain)
        ub = [(v, r) for v, r in zip(att_vals, srows) if r.x in used]           # the rows that feed the store
        attribution_check.append(sum(v == V.val(table[r.x]) for v, r in ub) / max(1, len(ub)))
        for space, ks in spaces.items():
            knn = MT.Store()
            for si, ses in enumerate(sessions):
                for p in range(1, len(strip(ses)) - 2, 3):
                    knn.add(ks(S[si, p + 1]), ses[p + 2])
            back = MT.Store()
            for j, r in enumerate(srows):
                if r.x in used:
                    back.add(ks(Hs_end[j]), V.val(table[r.x]))
            batt = MT.Store()                                    # v2b: values attributed by occlusion
            for j, r in enumerate(srows):
                if r.x in used:
                    batt.add(ks(Hs_end[j]), att_vals[j])
            for arm, got in (("knn", knn.lookup(ks(Hq_cue))),
                             ("backward-oracle", back.lookup(ks(Hq_end)) if len(back) else [None] * len(qrows)),
                             ("backward-attributed", batt.lookup(ks(Hq_end)) if len(batt) else [None] * len(qrows))):
                results.append(MT.score(core, fam, qrows, Q, got, M, inject, T, arm, space, 1, used,
                                        len(knn) if arm == "knn" else len(back), w))
        sym = {a: b for ses in sessions for a, b in MT._facts(strip(ses))}
        results.append(MT.score(core, fam, qrows, Q, [sym.get(V.key(r.x)) for r in qrows], M, inject, T,
                                "symbolic", "-", 1, used, len(sym), w))
        results.append(MT.score(core, fam, qrows, Q, [None] * len(qrows), M, inject, T, "none", "-", 1, used, 0, w))
    summary = MT.summarize(results)
    receipt = {"schema": "ens.h41.memtile_v2_receipt.v0", "core_checkpoint": ckpt, "key_site": site,
               "inject_site": inject, "family": fam.record(), "args": vars(args),
               "use_attribution": ("backward-oracle: generator-supplied true evidence (upper bound); "
                                   "backward-attributed: occlusion on the core's own answer (no label); on a core "
                                   "with perfect in-context recall the two coincide by construction"),
               "in_context_recall_acc_heldops": round(sum(recall_check) / len(recall_check), 4),
               "attribution_acc": round(sum(attribution_check) / len(attribution_check), 4),
               "key_fit_log": fit_log, "summary": summary, "rows": results,
               **C.source_revision(), **C.host_record()}
    C.write_json(out / "receipt.json", receipt)
    print("in-context recall on held ops:", receipt["in_context_recall_acc_heldops"],
          "| occlusion attribution finds the true fact:", receipt["attribution_acc"])
    for row in summary:
        print(row)
    return receipt


def training_pairs(core, fam, tx, site, T, rng, args, n_worlds=24):
    """Positives from TRAIN-pool ops and training tables only."""
    A, B = [], []
    for _ in range(n_worlds):
        table = world(rng, fam.z_domain)
        cues = [x for x in range(V.N_KEYS) if len(tx[x]) >= 2]
        r1, r2 = zip(*[rng.sample(tx[x], 2) for x in cues])
        ins = [insession_query(rng, fam, r, table, r.x, args.n_facts, T) for r in r1]
        Q2 = [MT.query_row(fam, r) for r in r2]
        A.append(states_at(core, [i[0] for i in ins], [i[1] for i in ins], site, T))
        B.append(states_at(core, [q[0] for q in Q2], [q[2] for q in Q2], site, T))
        sessions = session_rows(rng, table, cues, args.n_facts, None, T)
        S = MT.site_states(core, MT.pad(sessions, T), site)
        src, qq = [], []
        for si, ses in enumerate(sessions):
            for p in range(1, len(strip(ses)) - 2, 3):
                src.append(S[si, p + 1])
                qq.append(MT.query_row(fam, rng.choice(tx[ses[p + 1] - V.K0])))
        A.append(mx.stack(src))
        B.append(states_at(core, [q[0] for q in qq], [q[1] for q in qq], site, T))
    # known limit (review 2026-11-23): same-cue rows from other worlds act as in-batch negatives in the InfoNCE fit,
    # which biases +learned backward scores down; and +learned kNN reduces to cue-token identity (= symbolic)
    return mx.concatenate(A), mx.concatenate(B)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--core", required=True)
    p.add_argument("--family", default="program.ADD.h1")
    p.add_argument("--key-site", default="middle", help="middle exists for 4- and 6-layer cores; not ingress")
    p.add_argument("--worlds", type=int, default=16)
    p.add_argument("--n-facts", type=int, default=5)
    p.add_argument("--p-used", type=float, default=0.5)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--learn", action="store_true")
    p.add_argument("--learn-steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
