"""Per-round phase table of an ``mtp_profile`` window phase (diagnostic only).

Reads ``spec_acceptance_gsm8k.py``'s ``receipt.json`` (arms served with
``MLX2_EXTERNAL_ROUND_TIMING=1``, some also ``_SYNC=1``) and writes, per arm,
the tree round's phases in ms per round (``external_phase_tree_*_ns`` over
``external_phase_rounds``), the blocked host time (``external_sync_eval_ns``,
``_synchronize_ns``, ``_async_eval_ns`` per round: command encoding plus the
device wait), the rest of the round (host work that does not block), the
tile-budget fit and cost, decode tok/s and tokens per round.  Arms named
``<arm>-a`` / ``<arm>-b`` (interleaved order) are also averaged into
``<arm>``, which cancels linear drift across the phase.

Host-wall phases charge device work to whichever phase waits for it: on the
host round the draft's GPU time sits in ``tree_draft_build`` (its read), on a
device round in ``tree_target_wait``.  Compare round totals and blocked time;
the ``-fenced`` arms (``_SYNC``) fence every boundary and charge work to the
phase that submitted it, at the cost of serializing the round.

Usage: summarize_device_tree_profile.py RECEIPT OUT
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

KEY = re.compile(r"^external_(phase|sync|gc)_")


def _find(doc, out):
    if isinstance(doc, dict):
        for key, value in doc.items():
            if isinstance(key, str) and (KEY.match(key) or key in (
                    "external_tree_rounds", "external_device_tree_rounds",
                    "external_tree_tile_budget")) and key not in out:
                out[key] = value
            else:
                _find(value, out)
    elif isinstance(doc, list):
        for item in doc:
            _find(item, out)
    return out


def arm_table(arm):
    counters = _find(arm.get("final_status") or {}, {})
    summary = arm.get("summary") or {}
    acceptance = summary.get("acceptance") or {}
    rounds = int(counters.get("external_phase_rounds") or 0)
    table = {"rounds": rounds, "decode_tps_median": summary.get("decode_tps_median"),
             "verify_rounds": acceptance.get("verify_rounds"),
             "env": arm.get("env"), "passed": arm.get("passed")}
    if rounds:
        phases = {key[len("external_phase_"):-3]: value / rounds / 1e6
                  for key, value in counters.items()
                  if key.startswith("external_phase_tree_") and key.endswith("_ns")
                  and key != "external_phase_tree_gap_ns"}
        table["phases_ms_per_round"] = dict(sorted(phases.items()))
        table["round_ms"] = sum(phases.values())
        gap = counters.get("external_phase_tree_gap_ns")
        table["gap_ms_per_round"] = None if gap is None else gap / rounds / 1e6
        blocked = {}
        for kind in ("eval", "synchronize", "async_eval"):
            ns, calls = counters.get(f"external_sync_{kind}_ns"), counters.get(
                f"external_sync_{kind}_calls")
            if ns is not None:
                blocked[kind] = {"ms_per_round": ns / rounds / 1e6,
                                 "calls_per_round": (calls or 0) / rounds}
        table["blocked"] = blocked
        table["blocked_ms_per_round"] = sum(item["ms_per_round"] for item in blocked.values())
        table["unblocked_ms_per_round"] = table["round_ms"] - table["blocked_ms_per_round"]
    budget = (counters.get("external_tree_tile_budget") or {}).get("1") or {}
    table["tile_budget"] = {key: budget.get(key) for key in (
        "fit", "cost_ms", "tokens_per_round", "verified_nodes", "chosen")}
    return table


def main():
    receipt_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    receipt = json.loads(receipt_path.read_text())
    arms = {arm["name"]: arm_table(arm) for arm in receipt["arms"]}
    pooled = {}
    for name, a in arms.items():
        if name.endswith("-a") and name[:-2] + "-b" in arms:
            b = arms[name[:-2] + "-b"]
            pooled[name[:-2]] = {
                key: (a[key] + b[key]) / 2 for key in (
                    "round_ms", "blocked_ms_per_round", "unblocked_ms_per_round",
                    "decode_tps_median") if a.get(key) is not None and b.get(key) is not None}
    doc = {"schema": "mlx2.device-tree-round-profile.v1", "receipt": str(receipt_path),
           "checkpoint": receipt.get("checkpoint"), "driver_revision": receipt.get("driver_revision"),
           "kind": "diagnostic, no thermal hold", "arms": arms, "pooled_a_b": pooled}
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    os.replace(tmp, out_path)
    for name, table in arms.items():
        print(json.dumps({"arm": name, "tps": table.get("decode_tps_median"),
                          "round_ms": table.get("round_ms"),
                          "blocked_ms": table.get("blocked_ms_per_round"),
                          "unblocked_ms": table.get("unblocked_ms_per_round")}))
    for name, table in pooled.items():
        print(json.dumps({"pooled": name, **table}))


if __name__ == "__main__":
    main()
