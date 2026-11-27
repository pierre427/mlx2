"""Analysis of reason_probe.py output (CPU).

Groups: H = think right & no-think wrong (reasoning helps), B = both right, W = both wrong.
Only think runs that closed </think> count as think-correct.
"""

import json
import sys

import numpy as np

P = sys.argv[1]
J = json.load(open(P + ".json"))
Z = np.load(P + ".key.npz")
rows = {r["i"]: r for r in J["rows"]}


def ok(r, m):
    return bool(r[m]["correct"]) and (m == "nothink" or r[m]["closed"])


grp = {}
for i, r in rows.items():
    t, n = ok(r, "think"), ok(r, "nothink")
    grp[i] = "H" if t and not n else "B" if t and n else "W" if not t and not n else "R"
cnt = {g: sum(1 for v in grp.values() if v == g) for g in "HBWR"}
N = len(rows)
acc_n = sum(ok(r, "nothink") for r in rows.values()) / N
acc_t = sum(ok(r, "think") for r in rows.values()) / N
closed = sum(r["think"]["closed"] for r in rows.values()) / N
print(f"problems {N}: no-think acc {acc_n:.2%}, think acc {acc_t:.2%} (think closed {closed:.0%})")
print(f"groups: H (reasoning helps) {cnt['H']}, B (both right) {cnt['B']}, W (both wrong) {cnt['W']}, R (think hurts) {cnt['R']}")
tl = [len(r["think"]["gen"]) for r in rows.values() if r["think"]["closed"]]
print(f"think length (closed): median {np.median(tl):.0f} tokens")

# ---------- 1. answer-moment confidence
print("\n1) Confidence at the answer-choice token (position predicting the first answer digit)")
pos = J["pos"]
def at_ans(p):
    k = p["ans_k"]
    if k is None:
        return None
    idx = k + 1 - 1 + 0  # entropy list starts at last prompt position -> index k predicts gen[k]
    e = p["entropy"][k]
    top = p["lp_top2"][k]
    return e, np.exp(top[0]), top[0] - top[1]
tab = {}
for p in pos:
    v = at_ans(p)
    if v is None:
        continue
    key = (p["mode"], grp[p["problem"]], p["correct"])
    tab.setdefault(key, []).append(v)
for key in sorted(tab):
    a = np.array(tab[key])
    print(f"  {key[0]:8s} group {key[1]} {'right' if key[2] else 'wrong'} n={len(a):3d}: entropy {a[:,0].mean():.3f}  top1 p {a[:,1].mean():.3f}  margin {a[:,2].mean():.2f} nats")
nt = [(at_ans(p), p["correct"]) for p in pos if p["mode"] == "nothink" and at_ans(p)]
if nt:
    e = np.array([x[0][0] for x in nt]); y = np.array([x[1] for x in nt])
    def auc(score, y):
        pos_, neg = score[y], score[~y]
        if len(pos_) == 0 or len(neg) == 0:
            return float("nan")
        return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos_ for b in neg]))
    print(f"  no-think: AUC of (low entropy -> correct) = {auc(-e, y):.3f}")

# ---------- 2. logit lens on the gold first token
print("\n2) Logit lens at the no-think answer position: rank of the CORRECT first token per layer")
km = J["key_meta"]
for g in ("H", "B", "W"):
    ranks = [k["lens"]["gold_first"]["rank"] for k in km if k["mode"] == "nothink" and grp[k["problem"]] == g and "gold_first" in k["lens"]]
    if not ranks:
        continue
    R = np.array(ranks)
    L = R.shape[1]
    sel = [0, L // 4, L // 2, 3 * L // 4, L - 3, L - 2, L - 1]
    med = np.median(R, 0)
    best = np.min(R, 1)
    print(f"  group {g} n={len(R)}: median rank at layers {[s + 1 for s in sel]}: {[int(med[s]) for s in sel]}"
          f" | problems where gold is top-10 at SOME layer: {np.mean(best < 10):.0%}, top-1 at some layer: {np.mean(best == 0):.0%}")

# ---------- 3. layer profile of firing per phase
print("\n3) Mean |MLP activation| per layer, by phase (averaged over problems)")
pm = Z["phase_means"].astype(np.float32)          # [n, L, I]
meta = J["phase_meta"]
prof = {}
for v, m in zip(pm, meta):
    key = (m["mode"], m["phase"])
    prof.setdefault(key, []).append(v.mean(-1))
L = pm.shape[1]
show = [0, 3, 7, 11, 15, 19, 23, 25, 27]
print("  layer:            " + " ".join(f"{s + 1:6d}" for s in show))
for key in [("nothink", "prompt"), ("nothink", "answer"), ("think", "think"), ("think", "answer")]:
    if key in prof:
        a = np.mean(prof[key], 0)
        print(f"  {key[0]:8s} {key[1]:7s}  " + " ".join(f"{a[s]:6.3f}" for s in show))

# sparsity / concentration: share of total |act| carried by the top 1% neurons, per phase
print("\n   share of |activation| mass in the top 1% of neurons (mean over layers):")
for key in [("nothink", "prompt"), ("nothink", "answer"), ("think", "think"), ("think", "answer")]:
    vs = [v for v, m in zip(pm, meta) if (m["mode"], m["phase"]) == key]
    if vs:
        V = np.stack(vs)                             # [n, L, I]
        k = max(1, V.shape[-1] // 100)
        top = np.sort(V, -1)[..., -k:].sum(-1) / V.sum(-1)
        print(f"   {key[0]:8s} {key[1]:7s} {top.mean():.3f}")

# ---------- 4. reasoning-phase neurons and whether they predict no-think success
print("\n4) Neurons that fire more while thinking than while answering directly")
think_v = {m["problem"]: v for v, m in zip(pm, meta) if m["mode"] == "think" and m["phase"] == "think"}
nans_v = {m["problem"]: v for v, m in zip(pm, meta) if m["mode"] == "nothink" and m["phase"] == "answer"}
common = sorted(set(think_v) & set(nans_v))
T = np.stack([think_v[i] for i in common]); A = np.stack([nans_v[i] for i in common])
ratio = np.log((T.mean(0) + 1e-3) / (A.mean(0) + 1e-3))     # [L, I]
for l in (6, 13, 20, 27):
    top = np.argsort(-ratio[l])[:5]
    print(f"  layer {l + 1}: top think/answer log-ratio neurons {top.tolist()} ratios {np.round(ratio[l, top], 2).tolist()}")
cos = [float((T.mean(0)[l] @ A.mean(0)[l]) / (np.linalg.norm(T.mean(0)[l]) * np.linalg.norm(A.mean(0)[l]))) for l in range(L)]
print(f"  cosine(think-phase profile, direct-answer profile) by layer: min {min(cos):.3f} at layer {int(np.argmin(cos)) + 1}, max {max(cos):.3f}")

# held-out: select neurons on even problems, test on odd ones, using the KEY (answer position) acts
acts = Z["act"].astype(np.float32)                       # [nk, L, I]
kmn = [(i, k) for i, k in enumerate(km) if k["mode"] == "nothink"]
idx_even = [i for i in common if i % 2 == 0]
if idx_even:
    Te = np.stack([think_v[i] for i in idx_even]).mean(0); Ae = np.stack([nans_v[i] for i in idx_even]).mean(0)
    r_e = np.log((Te + 1e-3) / (Ae + 1e-3))
    for l in (13, 20, 27):
        sel = np.argsort(-r_e[l])[:50]
        sc, yy = [], []
        for j, k in kmn:
            if k["problem"] % 2 == 1:
                sc.append(np.abs(acts[j, l, sel]).mean()); yy.append(k["correct"])
        sc, yy = np.array(sc), np.array(yy)
        pos_, neg = sc[yy], sc[~yy]
        a = np.mean([(x > y_) + 0.5 * (x == y_) for x in pos_ for y_ in neg]) if len(pos_) and len(neg) else float("nan")
        print(f"  held-out (odd problems) layer {l + 1}: AUC of 'reasoning-neuron' activation at the no-think answer token for correct vs wrong = {a:.3f} (n right {len(pos_)}, wrong {len(neg)})")

# ---------- 5. same problem, two modes: where do the answer-moment states diverge?
print("\n5) Same problem, answer-choice token, think vs no-think: per-layer cosine of MLP activation and residual")
res = Z["res"].astype(np.float32)
by = {}
for j, k in enumerate(km):
    by.setdefault(k["problem"], {})[k["mode"]] = j
for g in ("H", "B"):
    ca, cr = [], []
    for i, d in by.items():
        if grp[i] != g or len(d) < 2:
            continue
        a1, a2 = acts[d["think"]], acts[d["nothink"]]
        r1, r2 = res[d["think"]], res[d["nothink"]]
        ca.append([(a1[l] @ a2[l]) / (np.linalg.norm(a1[l]) * np.linalg.norm(a2[l]) + 1e-9) for l in range(L)])
        cr.append([(r1[l] @ r2[l]) / (np.linalg.norm(r1[l]) * np.linalg.norm(r2[l]) + 1e-9) for l in range(L)])
    if ca:
        ca, cr = np.mean(ca, 0), np.mean(cr, 0)
        print(f"  group {g} n={len(by) and sum(1 for i in by if grp[i] == g)}: MLP-act cosine layers {[s + 1 for s in show]}: {np.round(ca[show], 2).tolist()}")
        print(f"  {'':10s} residual cosine: {np.round(cr[show], 2).tolist()}")

# ---------- 6. entropy during thinking
print("\n6) Next-token entropy by phase (mean over tokens)")
for mode, ph in (("think", "think"), ("think", "answer"), ("nothink", "answer")):
    es = []
    for p in pos:
        if p["mode"] != mode:
            continue
        es += [e for e, f in zip(p["entropy"], p["phase"]) if f == ph]
    if es:
        es = np.array(es)
        print(f"  {mode:8s} {ph:7s} tokens {len(es):6d}: mean entropy {es.mean():.3f}, share of tokens with entropy > 1 nat {np.mean(es > 1):.1%}")
