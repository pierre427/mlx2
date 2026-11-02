"""GLM-5.3 glm53_compile_regions bit gate on a recorded decode trace.

Loads the replay snapshot twice and feeds the recorded tokens through two
caches step for step: arm A eager (glm53_compile_regions=off), arm B compiled
(v1). Every step's logits must match bit for bit, the route counters must
advance identically, and at the end every cache state array must match bit
for bit. Also writes the graph node count of one decode step per arm
(mx.export_to_dot). Research only; no serving path is changed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm53_decode_replay import load
from glm53_ordinary_profile import CHECKPOINT


def ledgers(model):
    from mlx2.runtime.models import deepseek_v4_hc, glm53_hc_mega

    w = model.weights
    return {"calls": dict(model.calls), "dense": dict(w.scheduled_dense_calls),
            "grouped": dict(w.scheduled_grouped_calls), "expert": dict(w.scheduled_expert_calls),
            "hc_mega": dict(glm53_hc_mega._SCHEDULED), "hc": dict(deepseek_v4_hc._SCHEDULED)}


def delta(after, before):
    return {k: {n: v - before[k].get(n, 0) for n, v in after[k].items() if v != before[k].get(n, 0)}
            for k in after}


def node_count(array):
    import mlx.core as mx

    with tempfile.NamedTemporaryFile("w+", suffix=".dot") as f:
        mx.export_to_dot(f.name, array)
        text = Path(f.name).read_text()
    # Primitive nodes are drawn as rectangles; arrays as ovals.
    prims = re.findall(r'\[label\s*=\s*"([^"]+)"[^\]]*shape\s*=\s*"?rectangle', text)
    return {"primitives": len(prims), "lines": text.count("\n"),
            "top": sorted({p.split()[0] for p in prims})[:40]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-revision", default="unrecorded")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=128)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to reuse a nonempty evidence directory")
    args.out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx
    import numpy as np

    from mlx2.runtime.models import glm53_compile_regions
    from mlx2.runtime.models.cache import load_prompt_cache

    adapter, policy = load(args.policy)
    model = adapter.model
    trace = json.loads((args.trace / "trace.json").read_text())
    tokens = trace["tokens"]
    steps = min(args.steps, len(tokens) - 1)
    if model.hc_mega:
        # First-use validation launches would land in arm A's first step only.
        from mlx2.runtime.models import glm53_hc_mega

        glm53_hc_mega.self_check()
    caches = []
    for _ in range(2):
        cache = load_prompt_cache(str(args.trace / "cache.safetensors"))
        for c in cache:
            if hasattr(c, "identity"):
                c.identity = model.identity
        mx.eval([c.state for c in cache])
        caches.append(cache)

    def step(arm, i, *, dot=False):
        model.compile_regions = "v1" if arm else "off"
        before = ledgers(model)
        t0 = time.perf_counter()
        logits = model(mx.array([[tokens[i]]]), caches[arm])
        build = time.perf_counter() - t0
        nodes = node_count(logits) if dot else None
        mx.eval(logits)
        return logits, delta(ledgers(model), before), build, nodes

    mismatches, counter_mismatch, builds = [], [], {0: [], 1: []}
    nodes = {}
    for i in range(steps):
        dot = i == steps // 2
        a, da, ba, na = step(0, i, dot=dot)
        b, db, bb, nb = step(1, i, dot=dot)
        builds[0].append(ba)
        builds[1].append(bb)
        if dot:
            nodes = {"eager": na, "compiled": nb}
        av, bv = np.array(a), np.array(b)
        if not np.array_equal(av.view(np.uint32), bv.view(np.uint32)):
            mismatches.append({"i": i, "words": int(np.count_nonzero(av.view(np.uint32) != bv.view(np.uint32))),
                               "argmax": [int(av.argmax()), int(bv.argmax())]})
        if da != db:
            counter_mismatch.append({"i": i, "eager": da, "compiled": db})
    model.compile_regions = "off"
    state_diff = []
    for index, (ca, cb) in enumerate(zip(*caches)):
        sa, sb = ca.state, cb.state
        flat_a = sa if isinstance(sa, (list, tuple)) else [sa]
        flat_b = sb if isinstance(sb, (list, tuple)) else [sb]

        def leaves(x):
            if isinstance(x, (list, tuple)):
                for y in x:
                    yield from leaves(y)
            elif x is not None:
                yield x
        la, lb = list(leaves(flat_a)), list(leaves(flat_b))
        mx.eval(la, lb)
        for j, (x, y) in enumerate(zip(la, lb)):
            xv, yv = np.array(x), np.array(y)
            if xv.shape != yv.shape or xv.tobytes() != yv.tobytes():
                state_diff.append({"layer": index, "leaf": j})
        if len(la) != len(lb):
            state_diff.append({"layer": index, "leaves": [len(la), len(lb)]})

    def p50(values):
        values = sorted(values)
        return values[len(values) // 2] * 1e3
    result = {"schema": "mlx2.glm53-compile-regions-probe.v1", "checkpoint": CHECKPOINT,
              "source_revision": args.source_revision, "policy": policy, "trace": str(args.trace),
              "steps": steps, "logits_bitwise": not mismatches, "mismatches": mismatches[:20],
              "counters_equal": not counter_mismatch, "counter_mismatches": counter_mismatch[:5],
              "state_bitwise": not state_diff, "state_diff": state_diff[:20],
              "build_p50_ms": {"eager": p50(builds[0]), "compiled": p50(builds[1][4:])},
              "graph_nodes_one_step": nodes, "regions": glm53_compile_regions.route_status()}
    (args.out / "probe.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("logits_bitwise", "counters_equal", "state_bitwise",
                                             "build_p50_ms")}),
          json.dumps({k: (v or {}).get("primitives") for k, v in nodes.items()}), flush=True)
    if mismatches:
        print("first mismatches", mismatches[:5], flush=True)
    if counter_mismatch:
        print("counter mismatch", json.dumps(counter_mismatch[0])[:2000], flush=True)


if __name__ == "__main__":
    main()
