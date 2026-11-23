"""H41 v1: a context memory tile over earlier sessions, with a backward-built graph and anchored keys (CPU toy).

World. A random chain world over the closed vocabulary: each of the 16 cue keys x starts a chain
x -> v1 -> ... -> vk of value facts. Earlier sessions are write-only: each is a sequence of facts
"GR src dst" (src a key or value token, dst a value token) for some chain links plus distractor links,
at most the core's 32 positions. A session also "uses" some chains: the tile is told which chain an in-session
query resolved. In v0 that use attribution is supplied by the generator (oracle use attribution), not read from
attention weights.

Later query. An ordinary bus row of a contributor family, [BOS Q K(x) op.. SLOT NEUTRAL ANS], with no facts in
context. z must come from memory: z = the end of x's chain after k hops (kept inside the family's z domain). The
tile retrieves z and injects the codebook message M[z] through the bus; the frozen core computes f(z).

Arms
  none          no memory: message 0 (chance baseline for the core).
  knn           a kNN memory over every stored fact (Memorizing-Transformers style): key = the core's state at the
                fact's source token, value = its destination value. k hops = k chained lookups; the next hop's key
                is the state of the retrieved value in a query-shaped probe row.
  backward-oracle  an UPPER BOUND on use attribution, not a competitor to knn: for used chains only, one edge
                from the endpoint (field-position) state of an in-session query with another op to the chain's TRUE
                end, which the generator supplies. It never reads the sessions, so it does not depend on k; it
                measures only whether endpoint states identify the cue across ops (review 2026-11-22). Refused at
                the ingress key site, where every field state is the same vector (NEUTRAL + its position).
  symbolic      an oracle over token identities (the ceiling for retrieval: every stored link, exact match).

Key spaces (the "orthogonal but anchored on the endpoints" question), for knn and backward:
  raw           the full d-wide state at the site.
  endpoint      the projection onto the bus reader's column space: the directions the bus writes and the blocks
                after the site read.
  orth          the projection onto the complement of the top-r principal directions of ordinary-row states.
  anchored      orthogonal to the ordinary directions except along the endpoint directions: removes only the
                part of the top-r ordinary subspace that lies outside the bus reader span.

The tile only ever writes through the bus, so the ordinary graph is untouched by construction (CORE-001).

  python -m research.ens.memtile --core <core dir> --family program.ADD.h1 --out <dir>
"""
from __future__ import annotations

import argparse
import random
import time

import mlx.core as mx

from research.ens import common as C
from research.ens import families as F
from research.ens import vocab as V
from research.ens.contrib import core_site
from research.ens.model import BusCall, value_logits

ARMS = ("none", "knn", "backward-oracle", "symbolic")
SPACES = ("raw", "endpoint", "orth", "anchored")


# ------------------------------------------------------------------ world and sessions
def make_world(rng: random.Random, k: int, z_domain: int):
    """16 chains, one per cue key: [K(x), V(v1), ..., V(vk)]. Intermediates are distinct across all chains and lie
    outside the z domain; the ends vk are distinct values inside it, so every link source is unique, the symbolic
    ceiling is exact, and a retrieval is correct only for its own chain."""
    mids = rng.sample(range(z_domain, V.N_VALUES), V.N_KEYS * (k - 1))
    if z_domain < V.N_KEYS:
        raise ValueError("ends must be distinct across chains: need z_domain >= 16")
    ends = rng.sample(range(z_domain), V.N_KEYS)    # distinct ends: a wrong chain never scores as a right one
    chains = []
    for x in range(V.N_KEYS):
        vals = mids[x * (k - 1):(x + 1) * (k - 1)]
        chains.append([V.key(x)] + [V.val(v) for v in vals + [ends[x]]])
    return chains


def links(chain):
    return list(zip(chain[:-1], chain[1:]))


def make_sessions(rng, chains, n_sessions, p_used, seq_len, n_distract):
    """Each chain's links are spread over the sessions; distractor links (random value pairs) fill the rest."""
    all_links = [(ci, l) for ci, c in enumerate(chains) for l in links(c)]
    sources = {a for _, (a, _) in all_links}
    free = [v for v in range(V.N_VALUES) if V.val(v) not in sources]   # distractors never shadow a chain link
    rng.shuffle(all_links)
    per = max(1, (seq_len - 1) // 3 - n_distract)
    sessions, used = [], set()
    for s in range(n_sessions):
        mine = all_links[s * per:(s + 1) * per]
        if not mine:
            break
        facts = [l for _, l in mine] + [(V.val(rng.choice(free)), V.val(rng.randrange(V.N_VALUES)))
                                        for _ in range(n_distract)]
        rng.shuffle(facts)
        sessions.append([V.BOS] + [t for a, b in facts for t in (V.tok("GR"), a, b)])
    if len(all_links) > n_sessions * per:
        raise ValueError("not enough sessions for every chain link")
    for ci in range(len(chains)):
        if rng.random() < p_used:
            used.add(ci)
    return sessions, used


# ------------------------------------------------------------------ states and key spaces
def site_states(core, tokens, site):
    """(B, T, d) state at the bus site before any write; positions beyond a row are padded."""
    return core.prefix(tokens, site)


def pad(rows, T):
    return mx.array([r + [V.PAD] * (T - len(r)) for r in rows])


class KeySpace:
    def __init__(self, core, site, ordinary_states, kind, r=8):
        self.kind = kind
        R = core.ports[site].reader.weight                  # (d, w): bus write / downstream read directions
        Q, _ = mx.linalg.qr(R, stream=mx.cpu)
        Q = Q[:, :R.shape[1]]                                # orthonormal basis of the reader span (d, w)
        self.P_end = Q @ Q.T
        X = ordinary_states - mx.mean(ordinary_states, axis=0, keepdims=True)
        _, _, Vt = mx.linalg.svd(X, stream=mx.cpu)
        U = Vt[:r].T                                         # top-r principal directions of ordinary states
        self.P_orth = mx.eye(R.shape[0]) - U @ U.T
        # anchored: orthogonal to the ordinary directions EXCEPT along the endpoint (reader) span. Remove only the
        # part of the ordinary subspace that lies outside the reader span: U' = orthonormal basis of (I - P_end) U.
        # (A union of the orth range and the reader span is the whole space when r << d, so it equals raw.)
        Up, sv, _ = mx.linalg.svd((mx.eye(R.shape[0]) - self.P_end) @ U, stream=mx.cpu)
        Up = Up[:, :int(mx.sum(sv > 1e-4 * sv[0]).item())]
        self.P_anch = mx.eye(R.shape[0]) - Up @ Up.T

    def __call__(self, h):
        if self.kind == "raw":
            k = h
        elif self.kind == "endpoint":
            k = h @ self.P_end
        elif self.kind == "orth":
            k = h @ self.P_orth
        else:
            k = h @ self.P_anch
        return k / (mx.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)


class LearnedKeys:
    """The tile's trainable part (v1): a linear key encoder on top of a fixed key space, trained contrastively
    (InfoNCE) on training worlds and the family's train pool only. Positives: (a) the endpoint (field) states of
    two queries with the same cue and different ops, for backward edges; (b) a fact's source-token state in a
    session and the cue-position state of the same token in a query-shaped probe row, for kNN hops."""

    def __init__(self, base: KeySpace, d, w=64, steps=400, lr=3e-3, temp=0.1, seed=0):
        import mlx.nn as nn
        import mlx.optimizers as optim
        self.base, self.temp = base, temp
        mx.random.seed(seed)
        self.lin = nn.Linear(d, w, bias=False)
        self._nn, self._optim, self.steps, self.lr = nn, optim, steps, lr

    def fit(self, A, B, rng):
        nn, optim = self._nn, self._optim
        A, B = self.base(A), self.base(B)
        opt = optim.Adam(learning_rate=self.lr)

        def loss(lin, a, b):
            za, zb = lin(a), lin(b)
            za = za / (mx.linalg.norm(za, axis=-1, keepdims=True) + 1e-6)
            zb = zb / (mx.linalg.norm(zb, axis=-1, keepdims=True) + 1e-6)
            logits = za @ zb.T / self.temp
            lab = mx.arange(a.shape[0])
            return (nn.losses.cross_entropy(logits, lab, reduction="mean")
                    + nn.losses.cross_entropy(logits.T, lab, reduction="mean")) / 2
        g = nn.value_and_grad(self.lin, loss)
        n, log = A.shape[0], []
        for it in range(self.steps):
            idx = mx.array(rng.sample(range(n), min(128, n)))
            l, gr = g(self.lin, A[idx], B[idx])
            opt.update(self.lin, gr)
            mx.eval(self.lin.parameters(), opt.state)
            if it % 100 == 0 or it == self.steps - 1:
                log.append(round(float(l.item()), 4))
        return log

    def __call__(self, h):
        z = self.lin(self.base(h))
        return z / (mx.linalg.norm(z, axis=-1, keepdims=True) + 1e-6)


def training_pairs(core, fam, site, T, rng, n_worlds=24):
    """Positive pairs from training worlds and the family's train pool (disjoint from the held evaluation)."""
    pool = F.family_rows(fam, "train")
    by_x = {}
    for r in pool:
        by_x.setdefault(r.x, []).append(r)
    A, B = [], []
    for _ in range(n_worlds):
        chains = make_world(rng, 3, fam.z_domain)
        sessions, _ = make_sessions(rng, chains, 24, 0.5, T, 4)
        S = site_states(core, pad(sessions, T), site)
        # (b) fact source state vs the same token at the cue of a probe row
        src, toks = [], []
        for si, ses in enumerate(sessions):
            for p in range(1, len(ses) - 2, 3):
                src.append(S[si, p + 1]); toks.append(ses[p + 1])
        rows = [rng.choice(pool) for _ in toks]
        pr = probe_rows(fam, rows, toks)
        Hp = site_states(core, pad([q[0] for q in pr], T), site)[mx.arange(len(pr)), mx.array([q[1] for q in pr])]
        A.append(mx.stack(src)); B.append(Hp)
        # (a) endpoint states of two queries with the same cue and different ops
        xs = [x for x in by_x if len(by_x[x]) >= 2]
        r1, r2 = zip(*[rng.sample(by_x[x], 2) for x in xs])
        Q1, Q2 = [query_row(fam, r) for r in r1], [query_row(fam, r) for r in r2]
        H1 = site_states(core, pad([q[0] for q in Q1], T), site)[mx.arange(len(Q1)), mx.array([q[2] for q in Q1])]
        H2 = site_states(core, pad([q[0] for q in Q2], T), site)[mx.arange(len(Q2)), mx.array([q[2] for q in Q2])]
        A.append(H1); B.append(H2)
    return mx.concatenate(A), mx.concatenate(B)


# ------------------------------------------------------------------ memory arms
class Store:
    def __init__(self):
        self.keys, self.vals = [], []

    def add(self, k, v):
        self.keys.append(k)
        self.vals.append(v)

    def lookup(self, q):
        K = mx.stack(self.keys)
        return [self.vals[i] for i in mx.argmax(q @ K.T, axis=-1).tolist()]

    def __len__(self):
        return len(self.keys)


def query_row(fam, row):
    ids, qi, fi, ai = F.layout(row, "bus")
    return ids, qi, fi, ai


def probe_rows(fam, rows, tokens):
    """Query-shaped rows with the cue token replaced (the next hop's key comes from the same geometry)."""
    out = []
    for r, t in zip(rows, tokens):
        ids, qi, _, _ = query_row(fam, r)
        ids = list(ids)
        ids[qi] = t
        out.append((ids, qi))
    return out


def run(args) -> dict:
    out = C.fresh_out(args.out)
    core, ckpt = C.load_core(args.core)
    core.freeze()
    site = args.key_site                              # where keys are read
    inject = args.inject_site
    if inject != core_site(args.core):
        raise SystemExit(f"--inject-site {inject} is not the core's trained bus site {core_site(args.core)}")
    fam = F.BY_ID[args.family]
    M = C.codebook(core.cfg.bus_width)
    T = core.cfg.seq_len
    rng = random.Random(f"ens.memtile.v0:{args.seed}")
    pool = F.family_rows(fam, "held")
    # ordinary-row states define the "ordinary computation" subspace for the orth key spaces
    # fitted on the train pool, non-PAD positions only (75% of ordinary-row positions are padding; review)
    ord_rows = [F.layout(r, "ordinary")[0] for r in F.family_rows(fam, "train")[:256]]
    ot = pad(ord_rows, T)
    Hall = site_states(core, ot, site).reshape(-1, core.cfg.d)
    keep = mx.array([i for i, t in enumerate(ot.reshape(-1).tolist()) if t != V.PAD])
    H = Hall[keep]
    spaces = {s: KeySpace(core, inject, H, s, r=args.rank) for s in SPACES}   # endpoint = the injection reader
    fit_log = {}
    if args.learn:
        A, B = training_pairs(core, fam, site, T, random.Random(f"ens.memtile.train:{args.seed}"))
        for s_ in SPACES:
            lk = LearnedKeys(spaces[s_], core.cfg.d, steps=args.learn_steps, seed=args.seed)
            fit_log[s_ + "+learned"] = lk.fit(A, B, random.Random(args.seed))
            spaces[s_ + "+learned"] = lk
    results = []
    for k in args.hops:
        for world_i in range(args.worlds):
            chains = make_world(rng, k, fam.z_domain)
            sessions, used = make_sessions(rng, chains, args.sessions, args.p_used, T, args.distract)
            S = site_states(core, pad(sessions, T), site)                 # (n_sess, T, d)
            # queries: every chain, a fresh op per query (held pool rows with x = the chain's cue)
            by_x = {}
            for r in pool:
                by_x.setdefault(r.x, []).append(r)
            qrows, srows = [], []                            # later queries; the in-session queries (other op)
            for x in range(V.N_KEYS):
                if len(by_x.get(x, [])) < 2:
                    continue
                r, r_s = rng.sample(by_x[x], 2)
                z = V.value_of(chains[x][-1])
                qrows.append(F.Row(x=x, z=z, op=r.op, answer=F.answer_for(fam, r, z)))
                srows.append(F.Row(x=x, z=z, op=r_s.op, answer=F.answer_for(fam, r_s, z)))
            Q = [query_row(fam, r) for r in qrows]
            qt = pad([q[0] for q in Q], T)
            qi = mx.array([q[1] for q in Q])
            Hq = site_states(core, qt, site)[mx.arange(len(Q)), qi]      # state at the cue (knn's first hop)
            # the endpoint: the field position, where the bus writes; its state carries the cue and the op
            fi = mx.array([q[2] for q in Q])
            Hf = site_states(core, qt, site)[mx.arange(len(Q)), fi]
            Qs = [query_row(fam, r) for r in srows]
            Hs = site_states(core, pad([q[0] for q in Qs], T), site)[mx.arange(len(Qs)), mx.array([q[2] for q in Qs])]
            for space, ks in spaces.items():
                # knn: every fact, keyed by the source token's state in its session
                knn = Store()
                for si, ses in enumerate(sessions):
                    for p in range(1, len(ses) - 2, 3):
                        knn.add(ks(S[si, p + 1]), ses[p + 2])
                # backward: used chains only, endpoint state of the in-session query -> evidence end
                back = Store()
                for ci in sorted(used):
                    j = [n for n, r in enumerate(srows) if r.x == ci]
                    if j:                                    # keyed by the in-session query's endpoint state
                        back.add(ks(Hs[j[0]]), chains[ci][-1])
                arms = {"knn": None} if site == "ingress" else {"knn": None, "backward-oracle": None}
                # knn: k chained lookups
                cur = knn.lookup(ks(Hq))
                for _ in range(k - 1):
                    pr = probe_rows(fam, qrows, cur)
                    Hp = site_states(core, pad([p[0] for p in pr], T), site)[mx.arange(len(pr)),
                                                                             mx.array([p[1] for p in pr])]
                    cur = knn.lookup(ks(Hp))
                arms["knn"] = cur
                if "backward-oracle" in arms:
                    arms["backward-oracle"] = back.lookup(ks(Hf)) if len(back) else [None] * len(qrows)
                for arm, got in arms.items():
                    results.append(score(core, fam, qrows, Q, got, M, inject, T, arm, space, k, used,
                                         len(knn) if arm == "knn" else len(back), world_i))
            # spaceless arms
            sym = {a: b for ses in sessions for a, b in _facts(ses)}
            got = []
            for r in qrows:
                cur = V.key(r.x)
                for _ in range(k):
                    cur = sym.get(cur)
                got.append(cur)
            results.append(score(core, fam, qrows, Q, got, M, inject, T, "symbolic", "-", k, used,
                                 len(sym), world_i))
            results.append(score(core, fam, qrows, Q, [None] * len(qrows), M, inject, T, "none", "-", k, used, 0,
                                 world_i))
    summary = summarize(results)
    receipt = {"schema": "ens.h41.memtile_receipt.v0", "core_checkpoint": ckpt, "key_site": site,
               "inject_site": inject,
               "family": fam.record(), "args": vars(args), "use_attribution": "oracle (generator-supplied)", "key_fit_log": fit_log,
               "summary": summary, "rows": results, **C.source_revision(), **C.host_record()}
    C.write_json(out / "receipt.json", receipt)
    for row in summary:
        print(row)
    return receipt


def _facts(ses):
    return [(ses[p + 1], ses[p + 2]) for p in range(1, len(ses) - 2, 3)]


def score(core, fam, qrows, Q, got, M, inject, T, arm, space, k, used, store, world_i):
    zs = [None if g is None or not V.V0 <= g < V.V0 + fam.z_domain else g - V.V0 for g in got]
    msg = mx.stack([M[z] if z is not None else mx.zeros(M.shape[1]) for z in zs])
    tokens = pad([q[0] for q in Q], T)
    bus = BusCall(inject, mx.array([q[1] for q in Q]), mx.array([q[2] for q in Q]), msg)
    pred = mx.argmax(value_logits(core(tokens, bus), mx.array([q[3] for q in Q])), axis=-1).tolist()
    rec = {"arm": arm, "space": space, "k": k, "world": world_i, "store": store}
    for part, sel in (("used", [r.x in used for r in qrows]), ("unused", [r.x not in used for r in qrows])):
        idx = [i for i, s in enumerate(sel) if s]
        rec[f"n_{part}"] = len(idx)
        rec[f"retrieval_{part}"] = sum(zs[i] == qrows[i].z for i in idx)
        rec[f"answer_{part}"] = sum(pred[i] == qrows[i].answer for i in idx)
    return rec


def summarize(rows):
    keys = sorted({(r["arm"], r["space"], r["k"]) for r in rows})
    out = []
    for a, s, k in keys:
        rs = [r for r in rows if (r["arm"], r["space"], r["k"]) == (a, s, k)]
        row = {"arm": a, "space": s, "k": k, "store_mean": round(sum(r["store"] for r in rs) / len(rs), 1)}
        for part in ("used", "unused"):
            n = sum(r[f"n_{part}"] for r in rs)
            row[f"retrieval_{part}"] = round(sum(r[f"retrieval_{part}"] for r in rs) / max(n, 1), 3)
            row[f"answer_{part}"] = round(sum(r[f"answer_{part}"] for r in rs) / max(n, 1), 3)
            row[f"n_{part}"] = n
        out.append(row)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--core", required=True)
    p.add_argument("--family", default="program.ADD.h1")
    p.add_argument("--key-site", default="ingress", help="site whose states form the keys")
    p.add_argument("--inject-site", default="early", help="bus site for the write (the core's trained site)")
    p.add_argument("--hops", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--worlds", type=int, default=8)
    p.add_argument("--sessions", type=int, default=24)
    p.add_argument("--distract", type=int, default=4)
    p.add_argument("--p-used", type=float, default=0.5)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--learn", action="store_true", help="also train a key encoder per key space (v1)")
    p.add_argument("--learn-steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
