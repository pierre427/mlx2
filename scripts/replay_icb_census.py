#!/usr/bin/env python3
"""P3 of docs/research/2026-11-26-metal-replay-feasibility-probes.md: ICB-compatibility census.

CPU only. Reads the round recordings that ``round_probe`` writes under
``MLX2_ROUND_PROBE_RECORD_DIR`` (``round-<pid>-<n>.json``: step_replay summary,
dispatch table and pipeline reflection of one decode round) and classifies every
dispatch for an indirect command buffer (ICB):

* ``as_is``: a compute dispatch with buffer and threadgroup-memory bindings only;
  an ``MTLIndirectComputeCommand`` can encode it unchanged.
* ``after_substitution``: also binds inline ``setBytes`` data, which ICB commands
  cannot carry; eligible once P4 moves those bytes into a constants buffer.
* ``ineligible``: an indirect dispatch (ICB compute commands have no indirect form;
  P6), or any dispatch of a recording step_replay marked unsupported (texture,
  sampler, ``useResource``, a nested ICB, two encoders at once).

Per round it reports ``segments``: maximal runs of dispatches eligible after
substitution. That count, not the eligible percentage, is the kill criterion
(more than 8 segments per round stops Phase 2), because scattered ineligible
dispatches split a mostly eligible round. Command-buffer and encoder boundaries
inside a round are MLX's own batching and do not split a replayed segment (one
encoder can hold the whole ICB), so they are reported, not counted as segments.
Host syncs inside a round would split it, but a step_replay recording covers one
round scope, and rounds with more than one blocking sync are listed separately by
the round probe.

    scripts/replay_icb_census.py RECORDING_DIR [RECORDING_DIR ...] --output census.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

MAX_SEGMENTS = 8  # kill criterion (probe doc, P3)

def own_slots(entry, reflected):
    """The slots this dispatch's kernel uses. A recording carries the encoder's whole binding
    table, including bindings an earlier kernel left behind (MLX never clears them); keep only
    the slot indices in the kernel's reflection (buffer or threadgroup), or every slot when the
    kernel is unreflected."""
    if reflected is None:
        return list(entry.get("slots", []))
    return [s for s in entry.get("slots", []) if s[0] in reflected]



def kernel_origin(name):
    if name.startswith("custom_kernel"):
        return "mx.fast.metal_kernel"
    if name == "<unreflected>":
        return "unreflected"
    return "mlx_builtin_or_library"


def classify(dispatch, unsupported, slots=None):
    kinds = [slot[1] for slot in (dispatch.get("slots", []) if slots is None else slots)]
    if unsupported:
        return "ineligible", f"recording_unsupported:{unsupported}"
    if dispatch.get("form") == "indirect":
        return "ineligible", "indirect_dispatch"
    if "bytes" in kinds:
        return "after_substitution", "set_bytes"
    return "as_is", None


def census_one(doc):
    summary = doc["summary"]
    kernels = summary.get("kernels") or []
    unsupported = summary.get("unsupported") or ""
    rows = doc.get("dispatches") or []
    reflected = {p["function"]: {sl[0] for sl in p.get("slots", [])} for p in doc.get("pipelines", [])}
    leftover = 0
    classes = collections.Counter()
    reasons = collections.Counter()
    segments, in_segment = 0, False
    bindings_max, bytes_slots, bytes_bytes, tg_dispatches = 0, 0, 0, 0
    origins = collections.Counter()
    per_kernel = collections.Counter()
    for d in rows:
        name0 = kernels[d["pipeline"]] if d["pipeline"] < len(kernels) else None
        mine = own_slots(d, reflected.get(name0))
        leftover += len(d.get("slots", [])) - len(mine)
        cls, why = classify(d, unsupported, mine)
        classes[cls] += 1
        if why:
            reasons[why] += 1
        eligible = cls != "ineligible"
        if eligible and not in_segment:
            segments += 1
        in_segment = eligible
        slots = mine
        bindings = sum(1 for s in slots if s[1] in ("buffer", "bytes"))
        bindings_max = max(bindings_max, bindings)
        for s in slots:
            if s[1] == "bytes":
                bytes_slots += 1
                bytes_bytes += int(s[3])
        tg_dispatches += any(s[1] == "threadgroup" for s in slots)
        name = kernels[d["pipeline"]] if d["pipeline"] < len(kernels) else "<missing>"
        origins[kernel_origin(name)] += 1
        per_kernel[name] += 1
    n = len(rows) or 1
    return {
        "file": doc.get("_file"), "route": doc.get("route"), "index": doc.get("index"),
        "compare_with_previous": doc.get("compare_with_previous"),
        "dispatches": len(rows),
        "leftover_slots_ignored": leftover,
        "barriers": summary.get("barriers"),
        "command_buffers_committed": summary.get("command_buffers_committed"),
        "encoders_opened": summary.get("encoders_opened"),
        "distinct_buffers": summary.get("buffers"),
        "distinct_pipelines": summary.get("pipelines"),
        "unreflected_kernels": summary.get("unreflected_kernels"),
        "unsupported": unsupported or None,
        "drained": summary.get("drained"),
        "methods": summary.get("methods"),
        "as_is_pct": round(100 * classes["as_is"] / n, 2),
        "after_substitution_pct": round(100 * (classes["as_is"] + classes["after_substitution"]) / n, 2),
        "ineligible_pct": round(100 * classes["ineligible"] / n, 2),
        "ineligible_reasons": dict(reasons.most_common()),
        "segments": segments,
        "max_bindings_per_dispatch": bindings_max,
        "over_31_bindings": bindings_max > 31,
        "set_bytes_slots": bytes_slots,
        "set_bytes_total_bytes": bytes_bytes,
        "dispatches_with_threadgroup_memory": tg_dispatches,
        "kernel_origin": dict(origins.most_common()),
        "top_kernels": dict(per_kernel.most_common(12)),
        "_kernel_sequence": [kernels[d["pipeline"]] if d["pipeline"] < len(kernels) else "<missing>"
                             for d in rows],
        "_grids": [tuple(d.get("grid", ())) + tuple(d.get("group", ())) for d in rows],
        "_tg": [tuple(s[3] for s in own_slots(d, reflected.get(kernels[d["pipeline"]] if d["pipeline"] < len(kernels) else None))
                      if s[1] == "threadgroup") for d in rows],
    }


def shape_sensitivity(a, b):
    """How two recordings of one route differ (input to P6)."""
    same_seq = a["_kernel_sequence"] == b["_kernel_sequence"]
    out = {"a": a["file"], "b": b["file"], "same_kernel_sequence": same_seq,
           "dispatches": [a["dispatches"], b["dispatches"]]}
    if same_seq:
        out["grid_or_group_changes"] = sum(x != y for x, y in zip(a["_grids"], b["_grids"]))
        out["threadgroup_memory_changes"] = sum(x != y for x, y in zip(a["_tg"], b["_tg"]))
        out["kernels_with_grid_change"] = sorted({k for k, x, y in zip(a["_kernel_sequence"], a["_grids"], b["_grids"]) if x != y})[:12]
    else:
        sa, sb = collections.Counter(a["_kernel_sequence"]), collections.Counter(b["_kernel_sequence"])
        out["kernels_only_in_a"] = sorted(set(sa) - set(sb))[:12]
        out["kernels_only_in_b"] = sorted(set(sb) - set(sa))[:12]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    docs = []
    for directory in args.dirs:
        for path in sorted(directory.glob("round-*.json")):
            doc = json.loads(path.read_text())
            doc["_file"] = f"{directory.name}/{path.name}"
            docs.append(doc)
    if not docs:
        raise SystemExit("no recordings found")
    rounds = [census_one(doc) for doc in docs]
    by_route = collections.defaultdict(list)
    for r in rounds:
        by_route[r["route"]].append(r)
    routes = {}
    for route, rs in by_route.items():
        pairs = [shape_sensitivity(rs[i], rs[j]) for i in range(len(rs)) for j in range(i + 1, len(rs))]
        worst = max(r["segments"] for r in rs)
        routes[route] = {
            "recordings": len(rs),
            "dispatches": [r["dispatches"] for r in rs],
            "segments": [r["segments"] for r in rs],
            "as_is_pct": [r["as_is_pct"] for r in rs],
            "after_substitution_pct": [r["after_substitution_pct"] for r in rs],
            "unsupported": sorted({r["unsupported"] for r in rs if r["unsupported"]}),
            "verdict": ("kill" if worst > MAX_SEGMENTS else "pass") if all(r["drained"] for r in rs) else "invalid_undrained",
            "kill_criterion": f"segments per round > {MAX_SEGMENTS}",
            "pairs": pairs,
        }
    public = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rounds]
    out = {"schema": "mlx2.replay-icb-census.v1", "probe": "P3", "rounds": public, "routes": routes,
           "note": "segment breaks counted: ineligible dispatches only; MLX command-buffer and encoder "
                   "boundaries are reported but do not split a replayed segment"}
    args.output.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({route: {k: v for k, v in row.items() if k != "pairs"} for route, row in routes.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
