"""GLM-5.3 IndexShare probe: would one DSA layer's top-k selection serve another?

GLM-5.3 trains a separate indexer in each of its 11 DSA layers (official
modeling_glm5_next.py builds one per layer); IndexShare (one layer's selection
reused by the next layers, as DeepSeek V4.1 and GLM-5.2 are trained to do)
would be a post-hoc approximation here. This probe measures the falsifier
before any implementation: at greedy decode after a realistic long prompt,
each DSA layer's dense attention probabilities (head mean) are captured, and
the attention mass a layer keeps under

* ``own``: its own indexer selection (what the model actually uses),
* ``prev``: the previous DSA layer's selection,
* ``first``: the first DSA layer's selection for every layer,
* ``every4``: sources at DSA layers 0/4/8, the rest reuse the latest source
  (GLM-5.2's index_topk_freq 4 shape),
* ``recent``: the most recent ``len(own)`` tokens (a free heuristic),

is reported with the Jaccard overlap of the selections. The 2026-09-23
cross-layer reuse experiment set the bar: post-hoc reuse kept 0.45-0.60 of
the mass where 0.90 was needed. Research only; the model's output is unchanged
(the hooks observe, they do not alter the selection or the attention).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm53_ordinary_profile import CHECKPOINT, MODEL, prompt_ids  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[8192, 16384])
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--decode", type=int, default=16)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--source-revision", default="unrecorded")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to reuse a nonempty evidence directory")
    args.out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx

    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter
    from mlx2.runtime.models import glm53_gguf

    info = mx.device_info()
    if info.get("max_recommended_working_set_size"):
        mx.set_wired_limit(info["max_recommended_working_set_size"])
    policy = json.loads(args.policy.read_text()) if args.policy else {}
    adapter = Glm53GGUFAdapter(MODEL, execution_policy=policy)
    model = adapter.model

    step = {"active": False, "selections": [], "probs": []}
    original_select = glm53_gguf.pooled_selection
    original_sdpa = mx.fast.scaled_dot_product_attention

    def select(*a, **k):
        ids = original_select(*a, **k)
        if step["active"]:
            if ids is None:
                raise RuntimeError("context fits index_topk; nothing to share")
            row = np.array(ids[0, -1])
            step["selections"].append(np.unique(row[row >= 0]))
        return ids

    def sdpa(q, k, v, *, scale, mask=None, **kw):
        out = original_sdpa(q, k, v, scale=scale, mask=mask, **kw)
        if step["active"] and q.shape[2] == 1:
            scores = (q.astype(mx.float32) * scale) @ k.astype(mx.float32).swapaxes(-1, -2)
            probs = mx.softmax(scores, axis=-1)[0, :, 0]  # (heads, T)
            step["probs"].append(np.array(probs))
        return out

    glm53_gguf.pooled_selection = select
    mx.fast.scaled_dot_product_attention = sdpa
    record = {"schema": "mlx2.glm53-index-share-probe.v1", "checkpoint": CHECKPOINT,
              "source_revision": args.source_revision, "host": platform.node(),
              "mlx": mx.__version__, "execution_policy": policy, "cells": []}
    try:
        for tokens in args.tokens:
            ids = prompt_ids(adapter, tokens)
            cache = model.make_cache()
            t0 = time.perf_counter()
            for begin in range(0, len(ids), args.chunk):
                piece = mx.array([ids[begin:begin + args.chunk]])
                with model._prompt_scope(piece.shape[1]):
                    logits = model(piece, cache, logits_to_keep=1)
                mx.eval(logits, [c.state for c in cache])
            prefill = time.perf_counter() - t0
            token = int(mx.argmax(logits[0, -1]).item())
            rows = []
            for _ in range(args.decode):
                step.update(active=True, selections=[], probs=[])
                logits = model(mx.array([[token]]), cache)
                mx.eval(logits)
                step["active"] = False
                token = int(mx.argmax(logits[0, -1]).item())
                sel, probs = step["selections"], step["probs"]
                if len(sel) != len(probs) or not sel:
                    raise RuntimeError(f"captured {len(sel)} selections vs {len(probs)} attentions")
                rows.append(score_step(sel, probs))
            cell = summarize(rows)
            cell.update(tokens=len(ids), prefill_seconds=prefill, decode_steps=args.decode,
                        dsa_layers=len(rows[0]["own"]))
            record["cells"].append(cell)
            print(json.dumps({k: cell[k] for k in ("tokens", "mass_mean", "mass_min_layer", "jaccard_prev_mean")}),
                  flush=True)
            del cache
            mx.clear_cache()
    finally:
        glm53_gguf.pooled_selection = original_select
        mx.fast.scaled_dot_product_attention = original_sdpa
    (args.out / "index-share.json").write_text(json.dumps(record, indent=1) + "\n")


def score_step(selections, probs):
    """Per-layer mass kept under each schedule at one decode position."""
    n = len(selections)
    total = probs[0].shape[-1]
    sources = {"own": list(range(n)), "prev": [max(0, j - 1) for j in range(n)],
               "first": [0] * n, "every4": [j - j % 4 for j in range(n)]}
    out = {name: [] for name in (*sources, "recent")}
    out["own_head_p10"] = []
    for j in range(n):
        head_mean = probs[j].mean(axis=0)
        for name, src in sources.items():
            out[name].append(float(head_mean[selections[src[j]]].sum()))
        out["recent"].append(float(head_mean[total - len(selections[j]):].sum()))
        out["own_head_p10"].append(float(np.percentile(probs[j][:, selections[j]].sum(axis=1), 10)))
    out["jaccard_prev"] = [
        len(np.intersect1d(selections[j - 1], selections[j])) / len(np.union1d(selections[j - 1], selections[j]))
        for j in range(1, n)
    ]
    return out


def summarize(rows):
    names = ("own", "prev", "first", "every4", "recent")
    stack = {name: np.array([r[name] for r in rows]) for name in names}  # (steps, layers)
    return {
        "mass_mean": {name: float(v.mean()) for name, v in stack.items()},
        "mass_min_layer": {name: float(v.mean(axis=0).min()) for name, v in stack.items()},
        "mass_per_layer": {name: v.mean(axis=0).round(4).tolist() for name, v in stack.items()},
        "own_head_p10_mean": float(np.mean([r["own_head_p10"] for r in rows])),
        "jaccard_prev_mean": float(np.mean([r["jaccard_prev"] for r in rows])),
        "jaccard_prev_per_pair": np.mean([r["jaccard_prev"] for r in rows], axis=0).round(4).tolist(),
    }


if __name__ == "__main__":
    main()
