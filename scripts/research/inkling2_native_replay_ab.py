#!/usr/bin/env python3
"""``inkling2_native_replay`` on the real checkpoint: option off vs on (GPU).

One process, the adapter's default kernel set, B1 greedy direct decode (no
server). For each context: prefill the same corpus tokens (512-row chunks),
then decode ``--tokens`` greedy tokens with the option off and on. Records
the token streams; at the first divergent position (shared prefix by
construction) both arms' log-probability windows (top ``--window``, plus
each arm's log-probability of the other arm's selection), classified by the
AGENTS.md near-tie rule (``mlx2.runtime.replay.near_tie``: top-two,
exact-tie and multi-way clauses, 0.5-nat ceiling; only that position, later
ones are downstream consequence); the drift over the matching prefix (the
largest |log-probability change| of the selected token); the site and
engine receipts (fixed_step count, declines, module_missing), and sanity
rates (not performance evidence). Refuses a nonempty output directory.

    $PY scripts/research/inkling2_native_replay_ab.py $MODEL $RUNS/native-replay-ab \
        [--contexts 1500 5000] [--tokens 96]
"""

from __future__ import annotations
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src


import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def corpus(tokenizer, n):
    parts, ids = [], []
    for path in sorted((ROOT / "docs").rglob("*.md")):
        parts.append(path.read_text(errors="replace"))
        if sum(len(p) for p in parts) > 6 * n:
            ids = tokenizer.encode("\n\n".join(parts), add_special_tokens=False)
            if len(ids) >= n:
                return ids[:n]
    ids = tokenizer.encode("\n\n".join(parts), add_special_tokens=False)
    if len(ids) < n:
        raise SystemExit(f"corpus too short for {n} tokens")
    return ids[:n]


def arm(model, policy, ids, tokens, on):
    import mlx.core as mx

    from mlx2.runtime.models.inkling2 import capacity_state

    model.configure_kernels(**{**policy, "native_replay": on})
    cache = model.make_cache()
    prompt = mx.array([ids], mx.int32)
    for start in range(0, len(ids) - 1, 512):
        model(prompt[:, start:min(start + 512, len(ids) - 1)], cache=cache)
    token = prompt[:, -1:]
    out, rows, logps = [], [], []
    began = time.perf_counter()
    for _ in range(tokens):
        logits = model(token, cache=cache)[0, -1].astype(mx.float32)
        logp = logits - mx.logsumexp(logits)
        top = mx.argsort(-logp)[:2]
        mx.eval(logp, top)
        a, b = top.tolist()
        rows.append({"top": [a, b], "logp": [logp[a].item(), logp[b].item()]})
        logps.append(logp)  # kept on device for the divergence window
        out.append(a)
        token = mx.array([[a]], mx.int32)
    elapsed = time.perf_counter() - began
    receipt = capacity_state.receipt(model)
    return {"tokens": out, "rows": rows, "decode_tok_s_sanity": tokens / elapsed,
            "receipt": receipt, "_logps": logps}


def divergence(off, on, first, window):
    """Near-tie evidence at the first divergence and the prefix drift."""
    import numpy as np

    from mlx2.runtime.replay import near_tie

    drift = 0.0
    for i in range(first if first is not None else len(off["tokens"])):
        token = off["tokens"][i]
        drift = max(drift, abs(off["_logps"][i][token].item() - on["_logps"][i][token].item()))
    out = {"prefix_selected_logp_drift": drift}
    if first is not None:
        a = np.array(off["_logps"][first], dtype=np.float64)
        b = np.array(on["_logps"][first], dtype=np.float64)
        out["classification"] = near_tie.classify(a, b, k=window)
        out["classification"]["note"] = (
            "arm a = option off, b = on; shared prefix by construction; the "
            "functional oracle on the continuations is not run by this script")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--contexts", type=int, nargs="+", default=[1500, 5000])
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing nonempty {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    from mlx2.adapters.inkling2 import Inkling2Adapter

    adapter = Inkling2Adapter(str(args.model), execution_policy={})
    try:
        model = adapter.model
        policy = dict(model.kernel_policy)
        report = {"checkpoint": adapter.checkpoint,
                  "source_revision": subprocess.run(
                      ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
                      capture_output=True, check=False).stdout.strip(),
                  "policy": policy, "cells": []}
        for n in args.contexts:
            ids = corpus(adapter.tokenizer, n)
            off = arm(model, policy, ids, args.tokens, False)
            on = arm(model, policy, ids, args.tokens, True)
            first = next((i for i, (x, y) in enumerate(zip(off["tokens"], on["tokens"]))
                          if x != y), None)
            cell = {"context": n, "decode_tokens": args.tokens, "first_divergence": first,
                    "matching_prefix": args.tokens if first is None else first,
                    "off": off, "on": on}
            if first is not None:
                cell["divergence_rows"] = {"off": off["rows"][first], "on": on["rows"][first]}
            cell.update(divergence(off, on, first, args.window))
            off.pop("_logps")
            on.pop("_logps")
            report["cells"].append(cell)
            print(json.dumps({"context": n, "first_divergence": first,
                              "verdict": (cell.get("classification") or {}).get("verdict"),
                              "prefix_drift": cell["prefix_selected_logp_drift"],
                              "on_fixed_steps": on["receipt"].get("counts"),
                              "on_declines": on["receipt"].get("declines"),
                              "engine": on["receipt"].get("engine", {}).get("declines")}))
        tmp = args.out / "report.json.tmp"
        tmp.write_text(json.dumps(report, indent=1) + "\n")
        tmp.replace(args.out / "report.json")
    finally:
        adapter.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
