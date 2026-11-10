#!/usr/bin/env python3
"""Offline probe of an output-margin re-run (Pierre 2026-11-09, idea from an unverified HN post):
serve experts reduced, re-run a token at full precision only when the reduced arm's top-1 minus
top-2 logprob margin is below tau.

From ablation-ladder logs (greedy, top-2 logprobs per token) for each arm prompt vs the exact arm:
  d = first position where the arm's token differs from exact.
  caught(tau) = the arm's own margin at d < tau (a re-run there would emit exact's token).
  cost(tau)   = share of tokens before d (shared prefix) whose arm margin < tau (re-runs paid).
Only the FIRST divergence is simulable: after a corrected token the reduced continuation is unknown,
so caught is an upper bound on rescue, not proof of exact output.
"""
import json, sys
from pathlib import Path

TAUS = (0.1, 0.25, 0.5, 1.0, 2.0)


def margin(t):
    top = t.get("top") or []
    return (top[0][1] - top[1][1]) if len(top) >= 2 else float("inf")


def load(d):
    p = d / "replies.json"
    return json.loads(p.read_text()) if p.exists() else None


def main(roots):
    rows = []
    for root in map(Path, roots):
        base = load(root / "exact")
        if base is None:
            continue
        classes = {a["arm"]: a.get("per_prompt", {}) for a in json.loads((root / "summary.json").read_text())}
        for arm in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "exact"):
            reps = load(arm)
            if not reps:
                continue
            for pid, r in reps.items():
                b = base.get(pid)
                if not b or not r.get("tokens") or not b.get("tokens"):
                    continue
                at, bt = r["tokens"], b["tokens"]
                d = next((i for i, (x, y) in enumerate(zip(at, bt)) if x["token"] != y["token"]), None)
                if d is None and len(at) == len(bt):
                    continue                     # identical: nothing to rescue
                d = min(len(at), len(bt)) if d is None else d
                if d >= len(at):
                    continue
                pp = classes.get(arm.name, {}).get(pid, {})
                rows.append(dict(ladder=root.name, arm=arm.name, prompt=pid, d=d, n=len(at),
                                 m=margin(at[d]), cls=pp.get("class"), runaway=bool(pp.get("runaway")),
                                 prefix=[margin(t) for t in at[:d]], all=[margin(t) for t in at]))
    bad = [r for r in rows if r["cls"] in ("diverged_fail", "failed") or r["runaway"]]
    print(f"{len(rows)} divergent arm-prompts ({len(bad)} failing or runaway) across {len(roots)} ladders")
    print("tau | caught all | caught bad | re-run share of prefix tokens | re-run share of all arm tokens")
    for tau in TAUS:
        c = sum(r["m"] < tau for r in rows); cb = sum(r["m"] < tau for r in bad)
        pre = sum(sum(m < tau for m in r["prefix"]) for r in rows) / max(1, sum(len(r["prefix"]) for r in rows))
        alls = sum(sum(m < tau for m in r["all"]) for r in rows) / max(1, sum(len(r["all"]) for r in rows))
        print(f"{tau:4} | {c}/{len(rows)} | {cb}/{len(bad)} | {pre:.3f} | {alls:.3f}")
    ms = sorted(r["m"] for r in rows)
    print("divergence margins: median %.3f, p75 %.3f, p90 %.3f, max %.3f" % (
        ms[len(ms) // 2], ms[3 * len(ms) // 4], ms[9 * len(ms) // 10], ms[-1]))
    print("worst bad cases (largest margin at first divergence):")
    for r in sorted(bad, key=lambda r: -r["m"])[:8]:
        print(f"  {r['ladder']}/{r['arm']}/{r['prompt']}: d={r['d']} margin={r['m']:.3f} cls={r['cls']} runaway={r['runaway']}")


if __name__ == "__main__":
    main(sys.argv[1:])
