"""Acceptance gate for a sweep parity run: exit 0 only when every criterion holds.

  python -m research.amoeba.sweep.parity_check --exact DIR/parity.json --packed DIR/parity.json

Criteria (each text of each arm; ``rel`` = max|sweep - ordinary| / max|ordinary| over the logits):
  exact  arm (--ffn-texts 1, same operator shapes as the ordinary route):
         bitwise equal logits, or (rel <= 1e-3 and top-1 agreement == 1.0; reported as not bitwise);
         routed-signal check ran (check_signal_calls > 0) with max relative deviation <= 1e-3
         between the ordered-combine capture and the independent per-slot probe;
         the all-retained selection replay ran (select_replays > 0);
         every routed call was captured through the ordered combine (combine_captures > 0)
  packed arm (texts packed through the FFN as in the capture):
         rel <= 2e-2 and top-1 agreement >= 0.98
  both:  same checkpoint and fingerprint, same number of layers, at least 3 texts
"""
from __future__ import annotations

import argparse
import json
import sys


def check(exact, packed):
    bad = []
    for name, run in (("exact", exact), ("packed", packed)):
        if len(run.get("texts", [])) < 3:
            bad.append(f"{name}: fewer than 3 texts")
    if (exact.get("checkpoint"), exact.get("fingerprint"), exact.get("layers")) != (
            packed.get("checkpoint"), packed.get("fingerprint"), packed.get("layers")):
        bad.append("arms ran different checkpoints or depths")
    for t in exact.get("texts", []):
        rel = t["max_abs"] / max(t["ref_max_abs"], 1e-30)
        if not (t["bitwise_equal"] or (rel <= 1e-3 and t["top1_agree"] == 1.0)):
            bad.append(f"exact text {t['text']}: rel {rel:.3g} top1 {t['top1_agree']:.4f}")
    c = exact.get("counters") or {}
    if not c.get("check_signal_calls"):
        bad.append("exact: routed-signal probe check did not run")
    elif c.get("check_signal_max_rel", 1.0) > 1e-3:
        bad.append(f"exact: routed signal deviates from the probe by {c['check_signal_max_rel']:.3g}")
    if not c.get("select_replays"):
        bad.append("exact: selection replay never ran")
    if not c.get("combine_captures"):
        bad.append("exact: no routed call went through the ordered combine")
    for t in packed.get("texts", []):
        rel = t["max_abs"] / max(t["ref_max_abs"], 1e-30)
        if rel > 2e-2 or t["top1_agree"] < 0.98:
            bad.append(f"packed text {t['text']}: rel {rel:.3g} top1 {t['top1_agree']:.4f}")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exact", required=True)
    ap.add_argument("--packed", required=True)
    a = ap.parse_args(argv)
    exact, packed = (json.load(open(p)) for p in (a.exact, a.packed))
    bad = check(exact, packed)
    summary = {"passed": not bad, "failures": bad,
               "exact_bitwise": all(t["bitwise_equal"] for t in exact["texts"]),
               "exact_max_rel": max(t["max_abs"] / max(t["ref_max_abs"], 1e-30) for t in exact["texts"]),
               "packed_max_rel": max(t["max_abs"] / max(t["ref_max_abs"], 1e-30) for t in packed["texts"]),
               "packed_min_top1": min(t["top1_agree"] for t in packed["texts"]),
               "signal_max_rel": (exact.get("counters") or {}).get("check_signal_max_rel")}
    print(json.dumps(summary, indent=1))
    return 0 if not bad else 3


if __name__ == "__main__":
    sys.exit(main())
