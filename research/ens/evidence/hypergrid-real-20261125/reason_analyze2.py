"""Analysis of reason_probe2.py output (CPU)."""

import glob
import itertools
import json
import re
import sys

import numpy as np

P = sys.argv[1]
R = json.load(open(P + ".json"))
prev = json.load(open("reason_q06f.json")) if len(sys.argv) < 3 else json.load(open(sys.argv[2]))
rows = {r["i"]: r for r in R["rows"]}
RO = {int(k): v for k, v in R["readouts"].items()}
ids = sorted(RO)
n = len(ids)


def auc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, bool)
    pos, neg = score[y], score[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


def acc(arm, sel=None):
    s = [i for i in ids if sel is None or sel(i)]
    return np.mean([RO[i][arm]["correct"] for i in s]) if s else float("nan"), len(s)


print(f"=== {n} problems with readouts, {len(rows)} traces ===")
tl = np.array([rows[i]["think_len"] for i in ids])
trips = {}
for i in ids:
    trips[rows[i]["trip_reason"]] = trips.get(rows[i]["trip_reason"], 0) + 1
print(f"think length median {np.median(tl):.0f} (p10 {np.percentile(tl,10):.0f}, p90 {np.percentile(tl,90):.0f}); guard trips: {trips}; "
      f"natural closes {sum(rows[i]['natural_close'] for i in ids)}")
prev_rows = {r["i"]: r for r in prev["rows"]}
pt = [prev_rows[i]["think"]["correct"] for i in ids if i in prev_rows]
pn = [prev_rows[i]["nothink"]["correct"] for i in ids if i in prev_rows]
print(f"previous run (no guard, cap 1800), same problems: think {np.mean(pt):.1%}, no-think {np.mean(pn):.1%}")
by_trip = {}
for i in ids:
    by_trip.setdefault(rows[i]["trip_reason"], []).append(RO[i]["full"]["correct"])
print("full-trace accuracy by guard outcome:", {k: f"{np.mean(v):.0%} (n={len(v)})" for k, v in by_trip.items()})

print("\n--- 1) Is it just more compute / search space? (forced answer after each kind of 'thinking')")
arms = ["nothink", "empty", "dots", "other_trace", "shuffled", "cut25", "cut50", "cut75", "full"]
for a in arms:
    v, m = acc(a)
    e = np.mean([RO[i][a]["entropy"] for i in ids])
    print(f"  {a:12s} accuracy {v:6.1%}   mean answer entropy {e:.3f}")


# does the trace prefix already contain the gold number?
def gold_written(i, frac):
    return None
# use the trace text via the stored readout text is not enough; recompute from traces is not stored,
# so approximate with: cut arm correct vs full correct transitions
print("  cut-arm transitions (problems right at full): ", end="")
full_ok = [i for i in ids if RO[i]["full"]["correct"]]
for a in ("cut25", "cut50", "cut75"):
    print(f"{a} already right {np.mean([RO[i][a]['correct'] for i in full_ok]):.0%}", end="; ")
print()

print("\n--- 2) Confidence as a predictor")
for a in ("full", "cut50", "cut75", "shuffled", "dots", "nothink"):
    y = [RO[i][a]["correct"] for i in ids]
    e = [-RO[i][a]["entropy"] for i in ids]
    t = [RO[i][a]["top1"] for i in ids]
    print(f"  {a:9s} AUC(low entropy -> right) {auc(e, y):.3f}   AUC(top1 -> right) {auc(t, y):.3f}   acc {np.mean(y):.0%}")
y = np.array([RO[i]["full"]["correct"] for i in ids])
e = np.array([RO[i]["full"]["entropy"] for i in ids])
for lo, hi in ((0, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 0.7), (0.7, 99)):
    m = (e >= lo) & (e < hi)
    if m.any():
        print(f"    full-trace entropy in [{lo}, {hi}): {m.sum():3d} problems, accuracy {y[m].mean():.0%}")
# early exit: if the answer at 50 % / 75 % is confident, does it equal the final answer?
for a in ("cut50", "cut75"):
    conf = [i for i in ids if RO[i][a]["entropy"] < 0.05]
    same = np.mean([RO[i][a]["pred"] == RO[i]["full"]["pred"] for i in conf]) if conf else float("nan")
    ok = np.mean([RO[i][a]["correct"] for i in conf]) if conf else float("nan")
    print(f"  early exit at {a}: confident (entropy<0.05) on {len(conf)}/{n}; those answers right {ok:.0%}, equal to full-trace answer {same:.0%}")
# confidence under perturbation: confident-but-wrong rate
for a in ("full", "shuffled", "other_trace", "dots"):
    cw = np.mean([RO[i][a]["entropy"] < 0.05 and not RO[i][a]["correct"] for i in ids])
    print(f"  confidently wrong (entropy<0.05 and wrong) with {a:11s}: {cw:.0%} of problems")

print("\n--- 3) Ablations of MLP neurons during the readout (layers 12-28, zeroed; trace fixed)")
base = np.array([RO[i]["full"]["correct"] for i in ids])
blp = np.array([RO[i]["full"]["gold_first_lp"] for i in ids])
for a in ("sel64", "act64", "rnd64_s0", "rnd64_s1", "rnd64_s2", "sel256", "rnd256_s0"):
    v = np.array([RO[i]["abl_" + a]["correct"] for i in ids])
    lp = np.array([RO[i]["abl_" + a]["gold_first_lp"] for i in ids])
    print(f"  {a:10s} accuracy {v.mean():6.1%} (full {base.mean():.1%})  right->wrong {int((base & ~v).sum()):2d}  wrong->right {int((~base & v).sum()):2d}"
          f"  mean d(gold first-token logprob) {np.mean(lp - blp):+.3f}  entropy {np.mean([RO[i]['abl_' + a]['entropy'] for i in ids]):.3f}")

if R["ablated_gen"]:
    print("\n--- 4) Thinking GENERATED with neurons ablated")
    G = R["ablated_gen"]
    for a in ("sel64", "rnd64_s0"):
        print(f"  {a:9s} n={len(G)} accuracy {np.mean([g[a]['correct'] for g in G]):.0%}  median think length {np.median([g[a]['think_len'] for g in G]):.0f}"
              f"  trips {dict((t, sum(1 for g in G if g[a]['trip'] == t)) for t in set(g[a]['trip'] for g in G))}")
    print(f"  unablated, same problems: accuracy {np.mean([RO[g['i']]['full']['correct'] for g in G]):.0%}, median length {np.median([rows[g['i']]['think_len'] for g in G]):.0f}")

print("\n--- 5) Attention of the answer-choice token (mean over heads)")
A = R["attention"]
L = len(A[0]["trace"])
show = [0, 3, 7, 11, 13, 15, 17, 19, 21, 23, 25, 27]
tr = np.mean([a["trace"] for a in A], 0)
pr = np.mean([a["prompt"] for a in A], 0)
fo = np.mean([a["force"] for a in A], 0)
G_ = [a for a in A if a["gold_in_trace"]]
gp = np.mean([a["gold_pos"] for a in G_], 0) if G_ else np.zeros(L)
gm = np.mean([a["max_head_gold"] for a in G_], 0) if G_ else np.zeros(L)
print(f"  gold number found in trace for {len(G_)}/{len(A)} problems")
print("  layer          " + " ".join(f"{s + 1:5d}" for s in show))
for name, v in (("on prompt", pr), ("on trace", tr), ("on 'answer is'", fo), ("on gold number", gp), ("best head gold", gm)):
    print(f"  {name:14s} " + " ".join(f"{v[s]:5.2f}" for s in show))
# copy attention vs correctness
if G_:
    best_l = int(np.argmax(gm))
    yy = [RO[a["i"]]["full"]["correct"] for a in G_ if a["i"] in RO]
    ss = [a["max_head_gold"][best_l] for a in G_ if a["i"] in RO]
    print(f"  strongest copy layer {best_l + 1}: AUC(best-head attention on gold number -> right) {auc(ss, yy):.3f}")

print("\n--- 6) How selective is thinking?")
SEL = R["selectivity"]
prt = np.mean([s["pr_think"] for s in SEL], 0)
prp = np.mean([s["pr_prompt"] for s in SEL], 0)
print("  participation ratio (1 = all neurons equally used)")
print("  layer          " + " ".join(f"{s + 1:5d}" for s in show))
print("  thinking       " + " ".join(f"{prt[s]:5.2f}" for s in show))
print("  prompt reading " + " ".join(f"{prp[s]:5.2f}" for s in show))


def jac(sets):
    v = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in itertools.combinations(sets, 2)]
    return np.mean(v)
jt = [jac([s["top1pct_think"][l] for s in SEL]) for l in range(L)]
jp = [jac([s["top1pct_prompt"][l] for s in SEL]) for l in range(L)]
print("  cross-problem overlap (Jaccard) of each problem's top-1% neurons")
print("  thinking       " + " ".join(f"{jt[s]:5.2f}" for s in show))
print("  prompt reading " + " ".join(f"{jp[s]:5.2f}" for s in show))
dw = [np.load(f) for f in sorted(glob.glob(P + ".dw*.npz"))]
if dw:
    D = np.mean([d["digit"].astype(np.float32) for d in dw], 0)
    W = np.mean([d["word"].astype(np.float32) for d in dw], 0)
    lr = np.log((D + 1e-3) / (W + 1e-3))
    frac_d = (lr > 1).mean(1)
    frac_w = (lr < -1).mean(1)
    print("  share of neurons >e x more active on digit/operator tokens | on word tokens (within thinking)")
    print("  digit-selective" + " ".join(f"{frac_d[s]:5.1%}" for s in show))
    print("  word-selective " + " ".join(f"{frac_w[s]:5.1%}" for s in show))
    S = json.load(open("neuron_sets.json"))["sets"]
    for name in ("sel64", "act64", "rnd64_s0"):
        v = [lr[int(l), idx].mean() for l, idx in S[name].items()]
        print(f"  {name}: mean digit-vs-word log ratio {np.mean(v):+.2f}")
