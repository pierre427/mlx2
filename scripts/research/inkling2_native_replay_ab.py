#!/usr/bin/env python3
"""``inkling2_native_replay`` on the real checkpoint: option off vs on (GPU).

One process, the adapter's default kernel set, B1 greedy direct decode (no
server). For each context: prefill the same corpus tokens (512-row chunks),
then decode ``--tokens`` greedy tokens with the option off and on. Records
the token streams, the first divergent position with both arms' top-2
log-probabilities there (near-tie classification input), the site and
engine receipts (fixed_step count, declines, module_missing), and sanity
rates (not performance evidence). Refuses a nonempty output directory.

    $PY scripts/research/inkling2_native_replay_ab.py $MODEL $RUNS/native-replay-ab \
        [--contexts 1500 5000] [--tokens 96]
"""

from __future__ import annotations

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
    out, rows = [], []
    began = time.perf_counter()
    for _ in range(tokens):
        logits = model(token, cache=cache)[0, -1].astype(mx.float32)
        logp = logits - mx.logsumexp(logits)
        top = mx.argsort(-logp)[:2]
        mx.eval(logp, top)
        a, b = top.tolist()
        rows.append({"top": [a, b], "logp": [logp[a].item(), logp[b].item()]})
        out.append(a)
        token = mx.array([[a]], mx.int32)
    elapsed = time.perf_counter() - began
    receipt = capacity_state.receipt(model)
    return {"tokens": out, "rows": rows, "decode_tok_s_sanity": tokens / elapsed,
            "receipt": receipt}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--contexts", type=int, nargs="+", default=[1500, 5000])
    parser.add_argument("--tokens", type=int, default=96)
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
            report["cells"].append(cell)
            print(json.dumps({"context": n, "first_divergence": first,
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
