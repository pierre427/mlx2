"""Count hypergrid -> collapse on a real aMoEba category corpus (word level, CPU, numpy).

Arms (all fit on the box's `fit` split; tuned on `val`; scored on `held`):
  A  hypergrid   interpolated Kneser-Ney-style trigram over words (the uncollapsed memory)
  B  collapse    left/right context counts -> PPMI -> SVD -> k-means state merge ->
                 same trigram estimator over classes, times P(word | class). No gradients.
  C  A+B         linear interpolation, weight tuned on val
  D  MLP         gradient-trained trigram neural LM (emb + tanh hidden + softmax), numpy Adam

Run: python hypergrid_real.py corpus.jsonl --out results.json
"""

import argparse
import collections
import json
import math
import re
import time

import numpy as np

TOK = re.compile(r"[a-z]+|[0-9]+|[^\sa-z0-9]")
UNK, BOS = 0, 1


def tokenize(text):
    return TOK.findall(text.lower())


def load(path):
    docs = collections.defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        docs[r["split"]].append(tokenize(r["text"]))
    return docs


def build_vocab(docs, size):
    c = collections.Counter(w for d in docs for w in d)
    words = [w for w, _ in c.most_common(size - 2)]
    return ["<unk>", "<s>"] + words


def events(docs, idx):
    rows = []
    for d in docs:
        ids = [BOS, BOS] + [idx.get(w, UNK) for w in d]
        a = np.array(ids, dtype=np.int64)
        rows.append(np.stack([a[:-2], a[1:-1], a[2:]], 1))
    return np.concatenate(rows)


class Counts:
    """Sorted-key count table with vectorised lookup."""

    def __init__(self, keys):
        self.k, self.c = np.unique(keys, return_counts=True)

    def get(self, q):
        i = np.searchsorted(self.k, q)
        i = np.minimum(i, len(self.k) - 1)
        hit = self.k[i] == q
        return np.where(hit, self.c[i], 0)


class KNTrigram:
    """Interpolated absolute discounting, KN continuation counts at the unigram level."""

    def __init__(self, ev, V, D=0.75):
        self.V, self.D = V, D
        a, b, w = ev[:, 0], ev[:, 1], ev[:, 2]
        self.tri = Counts((a * V + b) * V + w)
        self.ctx3 = Counts(a * V + b)
        tri_ab = self.tri.k // V
        self.n1_ab = Counts(tri_ab)                     # distinct w after (a,b)
        self.bi = Counts(b * V + w)
        self.ctx2 = Counts(b)
        self.n1_b = Counts(self.bi.k // V)              # distinct w after b
        cont = np.bincount(self.bi.k % V, minlength=V)  # distinct b before w
        self.p1 = (cont + 0.5) / (cont.sum() + 0.5 * V)
        self.n_stored = len(self.tri.k) + len(self.bi.k) + V

    def prob(self, ev):
        V, D = self.V, self.D
        a, b, w = ev[:, 0], ev[:, 1], ev[:, 2]
        c2 = self.ctx2.get(b).astype(float)
        cbw = self.bi.get(b * V + w)
        n1b = self.n1_b.get(b)
        with np.errstate(divide="ignore", invalid="ignore"):
            p2 = np.where(c2 > 0, (np.maximum(cbw - D, 0) + D * n1b * self.p1[w]) / c2, self.p1[w])
            c3 = self.ctx3.get(a * V + b).astype(float)
            cabw = self.tri.get((a * V + b) * V + w)
            n1ab = self.n1_ab.get(a * V + b)
            p3 = np.where(c3 > 0, (np.maximum(cabw - D, 0) + D * n1ab * p2) / c3, p2)
        return p3


class ClassLM:
    def __init__(self, ev, cls, K, V):
        self.cls = cls
        self.ng = KNTrigram(cls[ev], K)
        wc = np.bincount(ev[:, 2], minlength=V) + 0.1
        cc = np.bincount(cls, weights=wc, minlength=K)
        self.pw = wc / cc[cls]
        self.n_stored = self.ng.n_stored + V

    def prob(self, ev):
        return self.ng.prob(self.cls[ev]) * self.pw[ev[:, 2]]


def collapse_embeddings(ev, V, n_ctx, rank):
    a, b, w = ev[:, 0], ev[:, 1], ev[:, 2]
    keep = lambda x: np.where(x < n_ctx, x, UNK)
    L = np.zeros((V, n_ctx), np.float32)
    R = np.zeros((V, n_ctx), np.float32)
    np.add.at(L, (w, keep(b)), 1)
    np.add.at(R, (b, keep(w)), 1)
    M = np.concatenate([L, R], 1)
    tot = M.sum()
    r = M.sum(1, keepdims=True)
    c = M.sum(0, keepdims=True) ** 0.75          # context-distribution smoothing
    c = c / c.sum() * tot
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.log(M * tot / (r @ c))
    P[~np.isfinite(P)] = 0
    P = np.maximum(P, 0)
    U, S, _ = np.linalg.svd(P, full_matrices=False)
    E = U[:, :rank] * np.sqrt(S[:rank])
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    return E, S


def kmeans(E, k, rng, iters=40):
    weights = None
    C = E[rng.choice(len(E), k, replace=False)]
    for _ in range(iters):
        assign = np.argmax(E @ C.T, 1)
        sums = np.zeros_like(C)
        np.add.at(sums, assign, E)
        n = np.bincount(assign, minlength=k)
        empty = n == 0
        sums[empty] = E[rng.choice(len(E), empty.sum(), replace=False)]
        C = sums / (np.linalg.norm(sums, axis=1, keepdims=True) + 1e-9)
    return np.argmax(E @ C.T, 1)


class MLP:
    def __init__(self, V, d, h, rng):
        s = lambda *shape: rng.normal(0, 0.05, shape).astype(np.float32)
        self.P = {"E": s(V, d), "W1": s(2 * d, h), "b1": np.zeros(h, np.float32),
                  "W2": s(h, V), "b2": np.zeros(V, np.float32)}
        self.m = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.t = 0

    def n_params(self):
        return sum(v.size for v in self.P.values())

    def logits(self, X):
        P = self.P
        e = P["E"][X].reshape(len(X), -1)
        hh = np.tanh(e @ P["W1"] + P["b1"])
        z = hh @ P["W2"] + P["b2"]
        return e, hh, z

    def prob(self, ev, bs=4096):
        out = np.empty(len(ev))
        for i in range(0, len(ev), bs):
            _, _, z = self.logits(ev[i:i + bs, :2])
            z = z - z.max(1, keepdims=True)
            lse = np.log(np.exp(z).sum(1))
            out[i:i + bs] = np.exp(z[np.arange(len(z)), ev[i:i + bs, 2]] - lse)
        return out

    def step(self, X, y, lr):
        P = self.P
        e, hh, z = self.logits(X)
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        n = len(X)
        p[np.arange(n), y] -= 1
        p /= n
        g = {"W2": hh.T @ p, "b2": p.sum(0)}
        gh = (p @ P["W2"].T) * (1 - hh ** 2)
        g["W1"] = e.T @ gh
        g["b1"] = gh.sum(0)
        ge = (gh @ P["W1"].T).reshape(n, 2, -1)
        gE = np.zeros_like(P["E"])
        np.add.at(gE, X[:, 0], ge[:, 0])
        np.add.at(gE, X[:, 1], ge[:, 1])
        g["E"] = gE
        self.t += 1
        for k in P:
            self.m[k] = 0.9 * self.m[k] + 0.1 * g[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * g[k] ** 2
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            P[k] -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)


def nll(p):
    return float(-np.mean(np.log(np.maximum(p, 1e-12))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", type=int, default=10000)
    ap.add_argument("--ks", default="64,256,1024")
    ap.add_argument("--mlp-epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    log = lambda *m: print(f"[{time.time() - t0:7.1f}s]", *m, flush=True)

    docs = load(args.corpus)
    vocab = build_vocab(docs["fit"], args.vocab)
    V = len(vocab)
    idx = {w: i for i, w in enumerate(vocab)}
    tr, val, held = (events(docs[s], idx) for s in ("fit", "val", "held"))
    log(f"V={V} fit={len(tr)} val={len(val)} held={len(held)} "
        f"unk_rate_held={np.mean(held[:, 2] == UNK):.3f}")

    # held-out slices: positions whose bigram / trigram never occurs in fit
    bi_fit = Counts(tr[:, 1] * V + tr[:, 2])
    tri_fit = Counts((tr[:, 0] * V + tr[:, 1]) * V + tr[:, 2])
    known = (held[:, 2] != UNK) & (held[:, 1] != UNK)
    bi_unseen = known & (bi_fit.get(held[:, 1] * V + held[:, 2]) == 0)
    tri_unseen_bi_seen = known & ~bi_unseen & (
        tri_fit.get((held[:, 0] * V + held[:, 1]) * V + held[:, 2]) == 0)
    slices = {"all": np.ones(len(held), bool), "unseen_bigram": bi_unseen,
              "unseen_trigram_seen_bigram": tri_unseen_bi_seen}
    log({k: int(v.sum()) for k, v in slices.items()})

    def score(p):
        return {k: nll(p[m]) for k, m in slices.items()}

    results = {"vocab": V, "fit_tokens": len(tr), "held_tokens": len(held),
               "slice_sizes": {k: int(v.sum()) for k, v in slices.items()}, "arms": {}}

    A = KNTrigram(tr, V)
    pA_val, pA = A.prob(val), A.prob(held)
    results["arms"]["A hypergrid"] = {"stored": A.n_stored, "val": nll(pA_val), **score(pA)}
    log("A", results["arms"]["A hypergrid"])

    E, S = collapse_embeddings(tr, V, n_ctx=2000, rank=100)
    log("svd done; top singular values", np.round(S[:8], 1).tolist())
    examples = ["velocity", "turbulent", "pressure", "equation", "increases", "we", "flow", "reynolds"]
    best_C = None
    for k in [int(x) for x in args.ks.split(",")]:
        cls = kmeans(E, k, rng)
        cls[UNK], cls[BOS] = k, k + 1           # keep specials in their own classes
        B = ClassLM(tr, cls, k + 2, V)
        pB_val, pB = B.prob(val), B.prob(held)
        lam = min((x / 20 for x in range(1, 20)),
                  key=lambda l: nll(l * pA_val + (1 - l) * pB_val))
        pC = lam * pA + (1 - lam) * pB
        nb = {w: [vocab[j] for j in np.where(cls == cls[idx[w]])[0][:10]]
              for w in examples if w in idx}
        results["arms"][f"B collapse k={k}"] = {"stored": B.n_stored, "val": nll(pB_val), **score(pB),
                                               "class_samples": nb}
        results["arms"][f"C A+B k={k}"] = {"lambda_memory": lam,
                                          "val": nll(lam * pA_val + (1 - lam) * pB_val), **score(pC)}
        log(f"B k={k}", {kk: round(v, 3) for kk, v in score(pB).items()},
            f"| C lam={lam}", {kk: round(v, 3) for kk, v in score(pC).items()})

    mlp = MLP(V, d=64, h=256, rng=rng)
    X, y = tr[:, :2], tr[:, 2]
    vsub = val[rng.choice(len(val), 50000, replace=False)]
    best = (1e9, None, -1)
    for ep in range(args.mlp_epochs):
        perm = rng.permutation(len(X))
        for i in range(0, len(X), 512):
            j = perm[i:i + 512]
            mlp.step(X[j], y[j], lr=2e-3)
        d = nll(mlp.prob(vsub))
        log(f"MLP epoch {ep} val_sub {d:.3f}")
        if d < best[0]:
            best = (d, {k: v.copy() for k, v in mlp.P.items()}, ep)
    mlp.P = best[1]
    pD = mlp.prob(held)
    results["arms"]["D MLP"] = {"stored": mlp.n_params(), "best_epoch": best[2],
                               "val": nll(mlp.prob(val)), **score(pD)}
    lamD = min((x / 20 for x in range(1, 20)), key=lambda l: nll(l * pA_val + (1 - l) * mlp.prob(val)))
    results["arms"]["A+D"] = {"lambda_memory": lamD, **score(lamD * pA + (1 - lamD) * pD)}
    log("D", results["arms"]["D MLP"], "A+D", results["arms"]["A+D"])
    json.dump(results, open(args.out, "w"), indent=1)
    log("wrote", args.out)


if __name__ == "__main__":
    main()
