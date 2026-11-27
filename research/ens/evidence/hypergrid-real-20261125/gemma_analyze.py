"""Analysis of gemma_probe.py output, laid out to compare with reason_analyze2.py (Qwen3-0.6B)."""

import itertools
import json
import sys

import numpy as np

R = json.load(open(sys.argv[1] + ".json"))
rows = {r["i"]: r for r in R["rows"]}
RO = {int(k): v for k, v in R["readouts"].items()}
ids = sorted(RO)
odd = [i for i in ids if i % 2]


def auc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, bool)
    p, n = score[y], score[~y]
    if not len(p) or not len(n):
        return float("nan")
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in p for b in n]))


tl = np.array([rows[i]["think_len"] for i in ids])
print(f"=== {len(ids)} problems; think length median {np.median(tl):.0f} (p10 {np.percentile(tl,10):.0f}, p90 {np.percentile(tl,90):.0f}); "
      f"budget-forced {sum(rows[i]['forced'] for i in ids)}; natural closes {sum(rows[i]['closed_naturally'] for i in ids)}")

print("\n--- 1) what preceded 'The answer is'")
for a in ["nothink", "empty", "dots", "other_trace", "shuffled", "cut25", "cut50", "cut75", "full"]:
    print(f"  {a:12s} accuracy {np.mean([RO[i][a]['correct'] for i in ids]):6.1%}   answer entropy {np.mean([RO[i][a]['entropy'] for i in ids]):.3f}")
ok = [i for i in ids if RO[i]["full"]["correct"]]
print("  of problems right at full, already right at: " + ", ".join(f"{a} {np.mean([RO[i][a]['correct'] for i in ok]):.0%}" for a in ("cut25", "cut50", "cut75")))

print("\n--- 2) confidence")
for a in ("full", "cut50", "cut75", "shuffled", "nothink"):
    y = [RO[i][a]["correct"] for i in ids]
    print(f"  {a:9s} AUC(low entropy -> right) {auc([-RO[i][a]['entropy'] for i in ids], y):.3f}  acc {np.mean(y):.0%}")
y = np.array([RO[i]["full"]["correct"] for i in ids]); e = np.array([RO[i]["full"]["entropy"] for i in ids])
for lo, hi in ((0, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 0.7), (0.7, 99)):
    m = (e >= lo) & (e < hi)
    if m.any():
        print(f"    full entropy [{lo}, {hi}): {m.sum():3d} problems, accuracy {y[m].mean():.0%}")
for a in ("cut50", "cut75"):
    c = [i for i in ids if RO[i][a]["entropy"] < 0.05]
    if c:
        print(f"  early exit {a}: confident {len(c)}/{len(ids)}, right {np.mean([RO[i][a]['correct'] for i in c]):.0%}, same as full {np.mean([RO[i][a]['pred'] == RO[i]['full']['pred'] for i in c]):.0%}")
for a in ("full", "shuffled", "other_trace"):
    print(f"  confidently wrong with {a:11s}: {np.mean([RO[i][a]['entropy'] < 0.05 and not RO[i][a]['correct'] for i in ids]):.0%}")

print(f"\n--- 3) ablations on held-out odd problems (n={len(odd)}); sets chosen on even problems, layers {R['sets_info']['layers_0idx'][0] + 1}-{R['sets_info']['layers_0idx'][-1] + 1}")
print(f"  sel64 vs act64 overlap {R['sets_info']['sel64_act64_overlap']:.3f}")
base = np.array([RO[i]["full"]["correct"] for i in odd])
for a in ("sel64", "act64", "rnd64_s0", "rnd64_s1", "esel8", "eact8", "ernd8"):
    k = "abl_" + a
    if k not in RO[odd[0]]:
        continue
    v = np.array([RO[i][k]["correct"] for i in odd])
    print(f"  {a:9s} accuracy {v.mean():6.1%} (full {base.mean():.1%})  right->wrong {int((base & ~v).sum()):2d}  wrong->right {int((~base & v).sum()):2d}"
          f"  entropy {np.mean([RO[i][k]['entropy'] for i in odd]):.3f}")

if "ntabl_act64" in RO[odd[0]]:
    print(f"\n--- 3b) same ablations with NO thinking (control: is act64 damage specific to reading a trace?)")
    nb = np.array([RO[i]["nothink"]["correct"] for i in odd])
    for a in ("sel64", "act64", "rnd64_s0", "rnd64_s1", "esel8", "eact8", "ernd8"):
        k = "ntabl_" + a
        v = np.array([RO[i][k]["correct"] for i in odd])
        print(f"  {a:9s} no-think accuracy {v.mean():6.1%} (no-think unablated {nb.mean():.1%})  right->wrong {int((nb & ~v).sum()):2d}  wrong->right {int((~nb & v).sum()):2d}"
              f"  entropy {np.mean([RO[i][k]['entropy'] for i in odd]):.3f}")

print("\n--- 6) selectivity")
S = R["selectivity"]
L = len(S[0]["pr"]["think"])
show = sorted(set([0, L // 6, L // 3, L // 2, 2 * L // 3, 5 * L // 6, L - 1]))
for ph in ("think", "prompt"):
    pr = np.mean([s["pr"][ph] for s in S], 0)
    print(f"  dense participation ratio {ph:6s} layers {[s + 1 for s in show]}: {np.round(pr[show], 2).tolist()}")


def jac(sets):
    return np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in itertools.combinations(sets, 2)])
for ph in ("think", "prompt"):
    j = [jac([s["top1pct"][ph][l] for s in S]) for l in range(L)]
    print(f"  dense top-1% cross-problem Jaccard {ph:6s}: {np.round(np.array(j)[show], 2).tolist()}")
if S[0]["expert_entropy"]["think"]:
    Le = len(S[0]["expert_entropy"]["think"])
    for ph in ("think", "prompt"):
        ee = np.mean([s["expert_entropy"][ph] for s in S], 0)
        je = [jac([s["expert_top8"][ph][l] for s in S]) for l in range(Le)]
        print(f"  expert usage entropy {ph:6s} (max {np.log(128):.2f}): mean {ee.mean():.2f}, layers {[s + 1 for s in show if s < Le]}: {np.round(ee[[s for s in show if s < Le]], 2).tolist()}")
        print(f"  expert top-8 cross-problem Jaccard {ph:6s}: mean {np.mean(je):.2f}")
