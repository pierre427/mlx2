#!/usr/bin/env python3
"""Classify first divergences between two GSM8K arms (e.g. B1 vs B4).

    python scripts/classify_b1b4_divergence.py ARM_A/rows.json ARM_B/rows.json > out.json

Both arms must have run with ``request: {"logprobs": true, "top_logprobs": 5}``
(scripts/spec_acceptance_gsm8k.py keeps ``logprob_windows``).  Per test index:
identical token sequences -> ``identical``; otherwise the first divergent
position is localized on the token ids and only that position's windows are
read (AGENTS.md: never a post-divergence row), then classified with
``mlx2.near_tie.near_tie_equivalent`` (two-way, exact-tie and multi-way
clauses, 0.5-nat ceiling). The functional oracle is GSM8K correctness of both
arms.  A token missing from the other arm's window is ``high_margin`` when the
window proves it is further than 0.5 nat below the top, else
``unclassified``.  Original mlx2 script (2026-11-27).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

CEILING = 0.5


def _rows(path):
    data = json.loads(Path(path).read_text())
    data = data if isinstance(data, list) else data.get("rows", data)
    return {r["test_index"]: r for r in data}


def classify_pair(a, b):
    from mlx2.near_tie import near_tie_equivalent

    wa, wb = a.get("logprob_windows"), b.get("logprob_windows")
    if wa is None or wb is None:
        return {"verdict": "unclassified", "reason": "no logprob windows"}
    ta, tb = [t for t, _ in wa], [t for t, _ in wb]
    if ta == tb:
        return {"verdict": "identical", "tokens": len(ta)}
    i = next((n for n, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
    out = {"first_divergent_token": i, "tokens": [len(ta), len(tb)]}
    if i >= min(len(ta), len(tb)):
        out["verdict"] = "unclassified"
        out["reason"] = "one sequence is a prefix of the other (length or stop difference)"
        return out
    left, right = {int(k): v for k, v in wa[i][1].items()}, {int(k): v for k, v in wb[i][1].items()}
    ls, rs = ta[i], tb[i]
    out.update(a_token=ls, b_token=rs, a_window=left, b_window=right)
    for name, w, sel in (("a", left, ls), ("b", right, rs)):
        top = sorted(w.values(), reverse=True)
        out[f"{name}_margin"] = round(top[0] - top[1], 4) if len(top) > 1 else None
    if rs not in left or ls not in right:
        far = any(sel not in w and max(w.values()) - min(w.values()) > CEILING
                  for w, sel in ((left, rs), (right, ls)) if w)
        out["verdict"] = "high_margin" if far else "unclassified"
        return out
    if not near_tie_equivalent(ls, left, rs, right, ceiling=CEILING):
        out["verdict"] = "high_margin"
    elif a.get("correct") and b.get("correct"):
        out["verdict"] = "near_tie_equivalent"
    else:
        out["verdict"] = "near_tie_oracle_failed"
    out["correct"] = [a.get("correct"), b.get("correct")]
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    a, b = _rows(argv[0]), _rows(argv[1])
    rows = {i: classify_pair(a[i], b[i]) for i in sorted(set(a) & set(b))}
    report = {"arms": argv[:2], "counts": dict(Counter(r["verdict"] for r in rows.values())),
              "ceiling": CEILING, "rows": rows}
    print(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
