"""Reconcile whole-route invariance evidence, without running qualification.

Consumes source/checkpoint-bound cells produced by ordinary, verify and replay
probes. Missing cells or missing state/logit evidence never imply invariance.
This diagnostic does not broaden verify_bitexact's quantized-matmul receipt.
"""
from __future__ import annotations

import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

from mlx2.generation_trace import read_token_trace

AXES = ("phase", "width", "order", "chunk", "replay", "membership")
BINDING = ("checkpoint", "source_revision", "numerical_profile",
           "runtime_identity", "cache_bytes", "persistence", "logical_request_id",
           "prompt_sha256", "generation_config_sha256")


def plan():
    return [dict(zip(AXES, values)) for values in itertools.product(
        ("ordinary", "verify", "refeed"), (1, 2), ("forward", "reverse"),
        (512, 2048), ("cold", "warm", "disk"), ("static", "ragged_join", "ragged_leave"))
        if values[1] > 1 or values[-1] == "static"]


def _key(cell):
    return tuple(cell.get(axis) for axis in AXES)


def evaluate(records, *, required=None):
    required = plan() if required is None else required
    if (not isinstance(required, list) or not required
            or any(not isinstance(c, dict) for c in required)
            or len({_key(c) for c in required}) != len(required)):
        raise ValueError("nonempty unique required matrix cells needed")
    rows = {}
    binding = None
    for cell in records:
        if not isinstance(cell, dict) or any(key not in cell for key in BINDING):
            raise ValueError("complete invariance binding required")
        checkpoint = cell["checkpoint"]
        if (not isinstance(checkpoint, str) or "@" not in checkpoint
                or checkpoint.startswith("*@")
                or not all(checkpoint.split("@", 1))
                or not isinstance(cell["source_revision"], str)
                or re.fullmatch("[0-9a-f]{40}", cell["source_revision"]) is None
                or type(cell["cache_bytes"]) is not int or cell["cache_bytes"] < 0
                or type(cell["persistence"]) is not bool
                or any(not isinstance(cell[k], str) or not cell[k]
                       for k in ("numerical_profile", "runtime_identity", "logical_request_id"))
                or any(not isinstance(cell[k], str)
                       or re.fullmatch("[0-9a-f]{64}", cell[k]) is None
                       for k in ("prompt_sha256", "generation_config_sha256"))):
            raise ValueError("exact checkpoint/source/profile/cache binding required")
        current = {key: cell[key] for key in BINDING}
        if binding is not None and binding != current:
            raise ValueError("invariance cells have different source/artifact/serving profiles")
        binding = current
        key = _key(cell)
        if key not in {_key(c) for c in required} or key in rows:
            raise ValueError("unexpected or duplicate invariance cell")
        count = cell.get("completion_tokens")
        if type(count) is not int or count < 0:
            raise ValueError("explicit completion token count required")
        trace = read_token_trace(cell, expected_tokens=count)
        if (cell.get("functional_passed") is not True or cell.get("route_observed") is not True
                or any(not isinstance(cell.get(k), str)
                       or re.fullmatch("[0-9a-f]{64}", cell[k]) is None
                       for k in ("state_sha256", "logits_sha256"))):
            raise ValueError("functional, state, logit and observed-route evidence required")
        rows[key] = (cell, trace)
    missing = [cell for cell in required if _key(cell) not in rows]
    comparisons = []
    if rows:
        base, base_trace = next(iter(rows.values()))
        for key, (cell, trace) in rows.items():
            left, right = base_trace["token_ids"], trace["token_ids"]
            first = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), None)
            if first is None and len(left) != len(right):
                first = min(len(left), len(right))
            exact = (first is None and base["state_sha256"] == cell["state_sha256"]
                     and base["logits_sha256"] == cell["logits_sha256"])
            comparisons.append({"cell": dict(zip(AXES, key)), "exact": exact,
                                "first_divergence_output_token": first,
                                "numerical_equivalence": "not_assessed" if not exact else "exact"})
    return {"schema": "mlx2.whole-route-invariance-diagnostic.v1", "binding": binding,
            "missing": missing, "comparisons": comparisons,
            "exact_matrix_passed": not missing and bool(rows) and all(x["exact"] for x in comparisons),
            "qualified": False,
            "correctness_verdict": "not_assessed; near-tie classification requires position-local evidence",
            "scope": "exact numerical-profile diagnostic; not a general correctness verdict"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refuse to overwrite invariance evidence")
    report = (evaluate(json.loads(args.records.read_text())) if args.records else
              {"required_cells": plan(), "status": "planned_not_run", "qualified": False})
    report["harness_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
