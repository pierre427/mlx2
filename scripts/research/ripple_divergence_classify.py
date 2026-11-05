#!/usr/bin/env python3
"""Classify Ripple first divergences under the AGENTS.md near-tie contract.

Inputs: the plan (shared prefixes, original continuations) and the bounded
log-prob window receipt from both arms. A row is ``near_tie_equivalent`` only
when both arms reproduced their tokens through the divergent position, the
prefix is shared, both continuations pass the functional oracle, and one of
the contract clauses holds:

* two-way: the same unordered top-two in both arms and both selected-versus-
  alternate margins at most 0.5 nat;
* exact tie: one arm's margin is exactly zero, the other arm selected one of
  the tied tokens with a margin at most 0.5 nat;
* multi-way: each arm's selected token is within 0.5 nat of the other arm's
  top log-probability, read from that arm's window. A token missing from a
  window leaves the row unclassified unless the window proves it is more than
  0.5 nat below the top, which fails the row.

Anything else is ``unclassified`` (missing evidence) or ``failed``.
"""
import argparse
import json
from pathlib import Path

CEILING = 0.5
TOPICS = {"decode0": (("print",),), "decode1": (("refrigerant", "heat"),), "decode2": (("star",),),
          "copy0": (("def",), ("reviewed",)), "copy1": (("def",), ("reviewed",))}


def degenerate(ids):
    run = best = 1
    for a, b in zip(ids, ids[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    grams = {}
    for i in range(len(ids) - 5):
        key = tuple(ids[i:i + 6])
        grams[key] = grams.get(key, 0) + 1
    return best >= 16 or max(grams.values(), default=0) >= 4


def oracle(cell, ids, text):
    lowered = text.lower()
    on_topic = all(any(word in lowered for word in group) for group in TOPICS[cell])
    return {"on_topic": on_topic, "degenerate": degenerate(ids), "passed": on_topic and not degenerate(ids)}


def margin(window):
    top = window["top"]
    return top[0][1] - top[1][1] if len(top) > 1 else None


def lookup(window, token):
    for tid, lp in window["top"]:
        if tid == token:
            return lp
    return None


def contract(control, arm):
    c_margin, a_margin = margin(control), margin(arm)
    if c_margin is None or a_margin is None:
        return "unclassified", "window shorter than two entries"
    c_top2 = {t for t, _ in control["top"][:2]}
    a_top2 = {t for t, _ in arm["top"][:2]}
    if c_top2 == a_top2 and c_margin <= CEILING and a_margin <= CEILING:
        return "near_tie_equivalent", "two-way"
    for tied, other, other_margin in ((control, arm, a_margin), (arm, control, c_margin)):
        if margin(tied) == 0.0:
            tied_tokens = {t for t, lp in tied["top"] if lp == tied["top"][0][1]}
            if other["selected"] in tied_tokens and other_margin <= CEILING:
                return "near_tie_equivalent", "exact-tie"
    verdicts = []
    for this, other in ((control, arm), (arm, control)):
        value = lookup(other, this["selected"])
        top = other["top"][0][1]
        if value is None:
            floor = other["top"][-1][1]
            verdicts.append("fail" if top - floor > CEILING else "missing")
        else:
            verdicts.append("ok" if top - value <= CEILING else "fail")
    if verdicts == ["ok", "ok"]:
        return "near_tie_equivalent", "multi-way"
    if "fail" in verdicts:
        return "failed", f"selected token more than {CEILING} nat below the other arm's top"
    return "unclassified", "a selected token is missing from the other arm's window"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer, local_files_only=True)
    plan = json.loads(args.plan.read_text())
    windows = json.loads(args.windows.read_text())["arms"]
    rows = []
    for row in plan["rows"]:
        key = (row["arm"], row["cell"], row["position"])

        def find(arm):
            return next((r for r in windows.get(arm, {}).get("rows", [])
                         if (r["row_arm"], r["cell"], r["position"]) == key), None)

        control, arm = find("control"), find(row["arm"])
        oracles = {who: oracle(row["cell"], row[f"{who}_tokens"], tokenizer.decode(row[f"{who}_tokens"]))
                   for who in ("control", "arm")}
        result = {"arm": row["arm"], "cell": row["cell"], "position": row["position"],
                  "control_token": row["control_token"], "arm_token": row["arm_token"],
                  "control_window": control and control["window"], "arm_window": arm and arm["window"],
                  "oracles": oracles}
        if control is None or arm is None or not control["reproduced"] or not arm["reproduced"]:
            result.update(verdict="unclassified", reason="an arm did not reproduce its tokens through the divergence")
        elif control["window"]["selected"] != row["control_token"] or arm["window"]["selected"] != row["arm_token"]:
            result.update(verdict="unclassified", reason="window selected token differs from the original run")
        else:
            verdict, reason = contract(control["window"], arm["window"])
            if not oracles["control"]["passed"]:
                # An oracle that rejects the ordinary reference cannot
                # attribute anything to the arm (copy-cell reasoning repeats
                # code 6-grams). Not a pass, not a failure.
                verdict, reason = "unclassified", f"oracle rejects the ordinary reference (contract: {verdict})"
            elif verdict == "near_tie_equivalent" and not oracles["arm"]["passed"]:
                verdict, reason = "failed", "the arm continuation failed the functional oracle"
            result.update(verdict=verdict, reason=reason, margins={
                "control": margin(control["window"]), "arm": margin(arm["window"])})
        rows.append(result)
    summary = {}
    for r in rows:
        summary[r["verdict"]] = summary.get(r["verdict"], 0) + 1
    args.output.write_text(json.dumps({"schema": "mlx2.ripple-near-tie-classification.v1", "ceiling_nat": CEILING,
                                       "summary": summary, "rows": rows}, indent=1) + "\n")
    print(json.dumps(summary))
    for r in rows:
        print(r["arm"], r["cell"], r["position"], r["verdict"], r.get("reason"), r.get("margins"))


if __name__ == "__main__":
    main()
