"""Inkling 2 ``inkling2_verify_deferred_commit``: exactness + sanity A/B on the checkpoint.

Original mlx2 script. One GPU unit; run it under the paired-lock / CPG lease.

Per context depth: two caches prefilled identically (A option off, B on;
prefill never speculates, so the last-row logits must be bitwise equal).
Then ``--rounds`` DSpark-shaped verify rounds of ``--rows`` rows (anchor +
proposals) through ``HybridVerifyRows`` as the external-draft engine builds
it (B's owner arms deferred commits). Proposals are the corpus continuation;
the kept count follows the greedy law (1 + proposals matching the argmax of
the previous row) of the round's first arm, and every ``--reject-every``-th
round keeps only the anchor and every third round keeps a seeded random
count at least the greedy one, so rejects, partial and full accepts all occur.
Both arms take those kept rows and that next anchor (forced), so they see the
same history.

Per round: logits of A vs B (bitwise equal expected; if not, the first
differing row, max |diff|, argmax agreement and both top-two margins), and
seconds of forward + commit + evaluation of the committed ring/conv state,
ABBA order. At the end of each cell, every plane's committed state A vs B
(rings: logical window; global KV: rows below offset; conv: four histories).
Kernel-receipt deltas show the option engaged (ring calls, commit counters).
A sanity A/B: no thermal hold, not publishable statistics.

Usage:
  python scripts/research/inkling2_deferred_commit_ab.py <model dir> <run dir>
      [--contexts 1024,4096,16384] [--rounds 24] [--rows 8] [--reject-every 4]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHUNK = 512
GIB = 2**30


def corpus_tokens(tokenizer, n):
    parts = []
    for pattern in ("docs/**/*.md", "src/mlx2/**/*.py", "scripts/**/*.py", "tests/**/*.py"):
        for path in sorted(ROOT.glob(pattern)):
            parts.append(path.read_text(errors="replace"))
            if len(parts) % 25 == 0:
                ids = tokenizer.encode("\n\n".join(parts), add_special_tokens=False)
                if len(ids) >= n:
                    return ids[:n]
    raise SystemExit("corpus too short")


def set_arm(model, on):
    """Toggle the option without re-running configure_kernels."""
    object.__setattr__(model, "supports_deferred_verify_commit", bool(on))
    for layer in model.model.layers:
        object.__setattr__(layer, "_deferred_commit", bool(on))
        object.__setattr__(layer.self_attn, "_deferred_commit", bool(on))


def counters(model):
    r = model.kernel_receipt()
    return {"deferred": r["verify_deferred_commit"], "splitk_ring": r["verify_splitk_ring"],
            "verify_splitk_calls": r["verify_splitk_calls"]}


def committed_arrays(cache):
    from mlx2.runtime.models.cache import KVCache, RotatingKVCache

    out = []
    for plane in cache:
        if type(plane) in (KVCache, RotatingKVCache) and plane.keys is not None:
            out += [plane.keys, plane.values]
        elif type(plane) not in (KVCache, RotatingKVCache):
            out += [s for s in plane.cache if s is not None]
    return out


def compare_planes(a, b):
    import mlx.core as mx

    from mlx2.runtime.models.cache import KVCache, RotatingKVCache

    bad = []
    for i, (x, y) in enumerate(zip(a, b)):
        if type(x) is RotatingKVCache:
            if x.offset != y.offset:
                bad.append((i, "offset"))
                continue
            n = min(x.offset, x.max_size)
            for name in ("keys", "values"):
                p = x._temporal_order(getattr(x, name))
                q = y._temporal_order(getattr(y, name))
                if not mx.array_equal(p[..., p.shape[2] - n:, :], q[..., q.shape[2] - n:, :]):
                    bad.append((i, name))
        elif type(x) is KVCache:
            if x.offset != y.offset or not (
                    mx.array_equal(x.keys[..., :x.offset, :], y.keys[..., :y.offset, :])
                    and mx.array_equal(x.values[..., :x.offset, :], y.values[..., :y.offset, :])):
                bad.append((i, "kv"))
        else:
            for j, (s, t) in enumerate(zip(x.cache, y.cache)):
                if (s is None) != (t is None) or (s is not None and not mx.array_equal(s, t)):
                    bad.append((i, f"conv{j}"))
    return bad


def top2(row):
    import mlx.core as mx

    logp = row.astype(mx.float32) - mx.logsumexp(row.astype(mx.float32))
    idx = mx.argsort(-logp)[:2].tolist()
    return idx, [logp[i].item() for i in idx]


def logits_diff(a, b):
    import mlx.core as mx

    if mx.array_equal(a, b):
        return None
    for row in range(a.shape[0]):
        if not mx.array_equal(a[row], b[row]):
            (ta, va), (tb, vb) = top2(a[row]), top2(b[row])
            return {"first_row": row, "max_abs_diff": mx.abs(a - b).max().item(),
                    "argmax_equal": ta[0] == tb[0], "off_top2": ta,
                    "off_margin_nat": va[0] - va[1], "on_top2": tb, "on_margin_nat": vb[0] - vb[1]}


def prefill(model, tokens, on):
    import mlx.core as mx

    set_arm(model, on)
    cache = model.make_cache()
    for a in range(0, len(tokens), CHUNK):
        logits = model(mx.array([tokens[a:a + CHUNK]]), cache=cache)
        mx.eval(logits)
        mx.clear_cache()
    return cache, logits[0, -1]


def verify(model, owner, cache, tokens, kept_fn):
    import mlx.core as mx

    mx.synchronize()
    t0 = time.perf_counter()
    tx = owner.begin([len(tokens)])
    logits = model(mx.array([tokens]), cache=tx.caches)[0]
    mx.eval(logits)
    kept = kept_fn(logits)
    tx.commit([kept])
    mx.eval(committed_arrays(cache))
    mx.synchronize()
    return logits, kept, time.perf_counter() - t0


def model_cell(model, corpus, n, rounds, rows, reject_every):
    import mlx.core as mx

    from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows

    import random

    rng = random.Random(n)
    caches, lasts = {}, {}
    for arm, on in (("off", False), ("on", True)):
        caches[arm], lasts[arm] = prefill(model, corpus[:n], on)
    cell = {"context": n, "prefill_last_logits_bitwise": bool(mx.array_equal(lasts["off"], lasts["on"]))}
    owners = {"off": HybridVerifyRows([caches["off"]]),
              "on": HybridVerifyRows([caches["on"]], deferred_commit=True)}
    anchor, pos = int(mx.argmax(lasts["off"]).item()), n
    times = {"off": [], "on": []}
    diffs, kept_hist = [], []
    before = counters(model)
    for r in range(rounds):
        proposals = corpus[pos:pos + rows - 1]
        tokens = [anchor] + proposals
        decided = {}

        def kept_fn(logits, r=r, tokens=tokens):
            if "kept" not in decided:
                greedy = mx.argmax(logits, axis=-1).tolist()
                kept = 1
                while kept < len(tokens) and greedy[kept - 1] == tokens[kept]:
                    kept += 1
                if reject_every and r % reject_every == reject_every - 1:
                    kept = 1
                elif r % 3 == 1:
                    # Forced partial/full accepts (corpus proposals rarely
                    # match): the commit path does not care whether the kept
                    # rows were the argmax.
                    kept = max(kept, rng.randint(1, len(tokens)))
                decided["kept"], decided["next"] = kept, greedy[kept - 1]
            return decided["kept"]

        order = ("off", "on") if r % 2 == 0 else ("on", "off")
        out = {}
        for arm in order:
            set_arm(model, arm == "on")
            out[arm] = verify(model, owners[arm], caches[arm], tokens, kept_fn)
            times[arm].append(out[arm][2])
        diff = logits_diff(out["off"][0], out["on"][0])
        if diff is not None:
            diffs.append({"round": r, **diff})
        kept_hist.append(decided["kept"])
        anchor = decided["next"]
        pos += rows - 1
    after = counters(model)
    set_arm(model, False)
    cell.update({
        "rounds": rounds, "rows": rows, "kept": kept_hist,
        "logits_bitwise_rounds": rounds - len(diffs), "logits_diffs": diffs,
        "planes_mismatched": compare_planes(caches["off"], caches["on"]),
        "median_round_s": {a: statistics.median(t) for a, t in times.items()},
        "median_round_s_after_first": {a: statistics.median(t[1:]) for a, t in times.items()},
        "counters_before": before, "counters_after": after,
    })
    return cell


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--contexts", default="1024,4096,16384")
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--reject-every", type=int, default=4)
    args = ap.parse_args()
    contexts = [int(c) for c in args.contexts.split(",")]
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} is not empty")
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx

    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src"],
                           capture_output=True, text=True).stdout.strip()
    receipt = {"script": "scripts/research/inkling2_deferred_commit_ab.py", "mlx2": head,
               "src_dirty": bool(dirty), "mlx_version": mx.__version__,
               "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "kind": "exactness + sanity A/B (no thermal hold; not publishable statistics)",
               "contexts": contexts, "cells": []}
    out = run_dir / "deferred_commit_ab.json"

    def save():
        tmp = out.with_suffix(".partial")
        tmp.write_text(json.dumps(receipt, indent=1, default=str))
        os.replace(tmp, out)

    from mlx2.adapters.inkling2 import Inkling2Adapter

    adapter = Inkling2Adapter(str(args.model_dir))  # current defaults
    try:
        model = adapter.model
        receipt["checkpoint"] = adapter.checkpoint
        receipt["adapter_kernel_policy"] = dict(model.kernel_policy)
        corpus = corpus_tokens(adapter.tokenizer, max(contexts) + args.rounds * args.rows + 8)
        with mx.stream(mx.gpu):
            for n in contexts:
                cell = model_cell(model, corpus, n, args.rounds, args.rows, args.reject_every)
                receipt["cells"].append(cell)
                print({k: cell[k] for k in ("context", "logits_bitwise_rounds",
                                            "planes_mismatched", "median_round_s")}, flush=True)
                save()
                mx.clear_cache()
        receipt["peak_memory_gb"] = round(mx.get_peak_memory() / GIB, 2)
    finally:
        adapter.close()
    receipt["passed"] = all(c["prefill_last_logits_bitwise"] and not c["logits_diffs"]
                            and not c["planes_mismatched"] for c in receipt["cells"])
    receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
