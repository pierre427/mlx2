"""GLM-5.3 routed-expert union per verify width (one owned GPU window).

Prefills a realistic prompt, greedy-decodes ``--steps`` tokens eagerly
(glm53_compile_regions off, so the router ids are visible) and records every
MoE layer's top-8 expert ids per token. For each width W, every window of W
consecutive tokens (a chain verify block) gives each layer's distinct-expert
count; the report is the mean over windows and layers, the ratio to the
one-token read (8), and the routed bytes a union MoE reads per block.

Research only; reads routing, changes nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from glm53_ordinary_profile import CHECKPOINT, MODEL, prompt_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--widths", type=lambda s: [int(v) for v in s.split(",")],
                        default=[1, 2, 3, 4, 6, 8, 16])
    parser.add_argument("--source-revision", default="unrecorded")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to reuse a nonempty evidence directory")
    args.out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx
    import numpy as np

    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter

    info = mx.device_info()
    if info.get("max_recommended_working_set_size"):
        mx.set_wired_limit(info["max_recommended_working_set_size"])
    policy = {**json.loads(args.policy.read_text()), "glm53_compile_regions": "off"}
    adapter = Glm53GGUFAdapter(MODEL, execution_policy=policy)
    model, weights = adapter.model, adapter.weights
    ids = prompt_ids(adapter, args.tokens)
    cache = model.make_cache()
    for start in range(0, len(ids), args.chunk):
        prompt = mx.array([ids[start:start + args.chunk]])
        with model._prompt_scope(prompt.shape[1]):
            logits = model(prompt, cache, logits_to_keep=1)
        mx.eval(logits, [c.state for c in cache])
    token = int(mx.argmax(logits[0, -1]).item())

    recorded = []
    inner, inner_shared = weights.moe, getattr(weights, "moe_with_shared", None)

    def moe(x, chosen, scores, prefix, *rest):
        recorded.append((prefix, chosen))
        return inner(x, chosen, scores, prefix, *rest)

    def moe_with_shared(x, chosen, scores, prefix, *rest):
        recorded.append((prefix, chosen))
        return inner_shared(x, chosen, scores, prefix, *rest)

    weights.moe = moe
    if inner_shared is not None:
        weights.moe_with_shared = moe_with_shared
    routing = []  # per step: {layer prefix: sorted top-8 ids}
    for _ in range(args.steps):
        recorded.clear()
        out = model(mx.array([[token]]), cache, logits_to_keep=1)
        mx.eval(out, [c for _, c in recorded])
        routing.append({p: np.array(c).reshape(-1).tolist() for p, c in recorded})
        token = int(mx.argmax(out[0, -1]).item())
    weights.moe = inner
    if inner_shared is not None:
        weights.moe_with_shared = inner_shared
    if not routing or not routing[0]:
        raise SystemExit("no routed MoE call was recorded")
    layers = sorted(routing[0], key=lambda p: int(p.split(".")[1]))
    sizes = {}
    for prefix in layers:
        tensors = [weights.tensors[f"{prefix}.ffn_{p}_exps.weight"] for p in ("gate", "up", "down")]
        per_expert = sum(getattr(t, "nbytes", 0) or 0 for t in tensors)
        sizes[prefix] = per_expert / 144 if per_expert else None
    rows = []
    for width in args.widths:
        unions = []
        for start in range(len(routing) - width + 1):
            window = routing[start:start + width]
            unions.append([len(set().union(*(set(step[p]) for step in window))) for p in layers])
        unions = np.array(unions, dtype=np.float64)
        mean = float(unions.mean())
        routed_bytes = None
        if all(sizes.values()):
            routed_bytes = float(np.mean([sum(u * sizes[p] for u, p in zip(w, layers)) for w in unions]))
        rows.append({"width": width, "mean_union_per_layer": mean, "ratio_to_one_token": mean / 8,
                     "per_row_new_experts": (mean - 8) / max(width - 1, 1) if width > 1 else 8.0,
                     "min": float(unions.min()), "max": float(unions.max()),
                     "routed_bytes_per_block": routed_bytes, "windows": len(unions)})
        print(f"W{width:<3d} union/layer {mean:6.2f} of 144 = {mean/8:5.2f}x one token"
              + (f"  routed {routed_bytes/1e9:6.2f} GB/block" if routed_bytes else ""), flush=True)
    record = {"schema": "mlx2.glm53-expert-union.v1", "checkpoint": CHECKPOINT,
              "source_revision": args.source_revision, "execution_policy": policy,
              "context_tokens": len(ids), "steps": args.steps, "moe_layers": len(layers), "rows": rows}
    (args.out / "expert_union.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
