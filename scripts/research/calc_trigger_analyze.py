#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Pierre-Lamy-Calculator-NC-ND-1.0
# Copyright (c) 2026 Pierre Lamy
# Scope and terms: provenance/LICENSE-calculator-scope.json; provenance/LICENSE-calculator.txt
"""Summarize a calculator-trigger GSM8K receipt (scripts/spec_acceptance_gsm8k.py) per arm (unqualified).

Per arm: accuracy, calculator fires (plans) and the questions they touched, markers written,
parse refusals, and, paired by test index against ``--reference`` (default arm ``off``), the
questions helped and hurt. Each wrong answer gets one class:

* ``truncated``: finish reason ``length`` (no final answer was reached);
* ``fired``: the calculator wrote at least one result in that answer;
* ``arithmetic_slip``: a written ``<<E=R>>`` or ``a OP b = R`` step whose exact value is not R
  (found offline with the marker / v2 parsers; the calculator did not fire);
* ``other``: no arithmetic slip found, so the setup or reading was wrong.

  python scripts/research/calc_trigger_analyze.py RECEIPT [--reference off]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mlx2.runtime import amoeba_det_tiles as D  # noqa: E402

_MARKED = re.compile(r"<<([^<>\n]*)=([^<>\n]*)>>")
_STEP = re.compile(r"=\s*\$?(-?\d[\d,]*(?:\.\d+)?)")


def _number(s):
    try:
        return Fraction(s.replace(",", "").replace("$", "").strip())
    except (ValueError, ZeroDivisionError):
        return None


def slips(text):
    """Written arithmetic steps whose exact value differs from what the text states."""
    found = []
    for m in _MARKED.finditer(text):
        got, _ = D.calculator_answer("<<" + m.group(1) + "=", parser=D.PARSER_MARKER)
        stated = _number(m.group(2))
        if got is not None and stated is not None and _number(got) != stated:
            found.append({"step": m.group(0), "exact": got})
    plain = _MARKED.sub("", text)
    for m in _STEP.finditer(plain):
        got, _ = D.calculator_answer(plain[:m.start() + 1], parser=D.PARSER_V2)
        stated = _number(m.group(1))
        if got is not None and stated is not None and _number(got) != stated:
            found.append({"step": plain[max(0, m.start() - 40):m.end()].strip(), "exact": got.strip()})
    return found


def arm_summary(arm, reference=None):
    rows = arm.get("rows") or []
    fires = sum((r.get("det_tiles") or {}).get("plans", 0) for r in rows)
    refused = {}
    for r in rows:
        for k, v in ((r.get("det_tiles") or {}).get("parse_refused") or {}).items():
            refused[k] = refused.get(k, 0) + v
    classes, wrong = {}, []
    for r in rows:
        if r.get("correct"):
            continue
        s = slips(r.get("text") or "")
        cls = ("truncated" if r.get("finish_reason") == "length" else
               "fired" if (r.get("det_tiles") or {}).get("plans") else
               "arithmetic_slip" if s else "other")
        classes[cls] = classes.get(cls, 0) + 1
        wrong.append({"test_index": r.get("test_index"), "class": cls, "expected": r.get("expected"),
                      "predicted": r.get("predicted"), "slips": s[:3]})
    out = {"n": len(rows), "correct": sum(bool(r.get("correct")) for r in rows),
           "accuracy": (sum(bool(r.get("correct")) for r in rows) / len(rows)) if rows else None,
           "fires": fires, "questions_fired": sum(bool((r.get("det_tiles") or {}).get("plans")) for r in rows),
           "marker_open": sum((r.get("det_tiles") or {}).get("marker_open", 0) for r in rows),
           "questions_with_markers": sum(bool(r.get("markers") or r.get("reasoning_markers")) for r in rows),
           "markers_written": sum((r.get("markers") or 0) + (r.get("reasoning_markers") or 0) for r in rows),
           "parse_refused": refused, "wrong_classes": classes, "wrong": wrong,
           "det_tiles_recorded": sum(r.get("det_tiles") is not None for r in rows)}
    if reference is not None and reference is not arm:
        ref = {r.get("test_index"): bool(r.get("correct")) for r in reference.get("rows") or []}
        mine = {r.get("test_index"): bool(r.get("correct")) for r in rows}
        shared = sorted(set(ref) & set(mine))
        out["vs_reference"] = {"reference": reference.get("name"), "paired": len(shared),
                               "helped": [i for i in shared if mine[i] and not ref[i]],
                               "hurt": [i for i in shared if ref[i] and not mine[i]]}
    return out


def analyze(receipt, reference_name="off"):
    arms = receipt.get("arms") or []
    reference = next((a for a in arms if a.get("name") == reference_name), None)
    return {"schema": "mlx2.calc-trigger-gsm8k-analysis.v1", "qualification": "unqualified",
            "checkpoint": receipt.get("checkpoint"), "driver_revision": receipt.get("driver_revision"),
            "prompt": receipt.get("prompt"), "reference": reference_name if reference else None,
            "arms": {a.get("name"): {**arm_summary(a, reference), "passed": a.get("passed")} for a in arms}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("receipt", type=Path)
    ap.add_argument("--reference", default="off")
    a = ap.parse_args(argv)
    print(json.dumps(analyze(json.loads(a.receipt.read_text()), a.reference), indent=1))


if __name__ == "__main__":
    main()
