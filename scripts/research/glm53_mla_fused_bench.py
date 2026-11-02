"""GLM-5.3 MLA layer microbench: one sparse-MLA layer's decode body per policy arm.

Loads the model and a recorded replay trace (cache snapshot), then for each arm
reloads the snapshot and runs ``--steps`` B1 decode calls of ``model._mla`` on
one MLA layer with the same random inputs, timing each call to completion
(graph build + eval of the layer output; deferred cache work lands where the
policy puts it). Reports p50 per call overall and per phase (position % 4;
phase 3 completes a pool), and checks each gathered arm's outputs bit for bit
against the reference arm. Research only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm53_decode_replay import load  # noqa: E402

ARMS = {
    "reference": {"mla_fused": "reference", "sparse_attention": "gathered_v1"},
    "decode_v1": {"mla_fused": "decode_v1", "sparse_attention": "gathered_v1"},
    "reference_fused_attn": {"mla_fused": "reference", "sparse_attention": "fused_decode_v1"},
    "decode_v1_fused_attn": {"mla_fused": "decode_v1", "sparse_attention": "fused_decode_v1"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--skip", type=int, default=8, help="warm-up calls not counted")
    parser.add_argument("--layer", type=int, default=None, help="MLA layer index (default: the first)")
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--source-revision", default="unrecorded")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to reuse a nonempty evidence directory")
    args.out.mkdir(parents=True, exist_ok=True)
    import mlx.core as mx
    import numpy as np

    from mlx2.runtime.models.cache import load_prompt_cache

    adapter, policy = load(args.policy)
    model = adapter.model
    layer = args.layer if args.layer is not None else next(
        i for i, l in enumerate(model.layers) if not l.is_linear)
    prefix = f"blk.{layer}"
    rng = np.random.default_rng(5310)
    xs = [mx.array(rng.normal(size=(1, 1, 4096)).astype(np.float32)) for _ in range(args.steps)]
    trace = json.loads((args.trace / "trace.json").read_text())
    result = {"schema": "mlx2.glm53-mla-fused-bench.v1", "source_revision": args.source_revision,
              "trace": str(args.trace), "prompt_tokens": trace["prompt_tokens"], "policy": policy,
              "layer": layer, "steps": args.steps, "skip": args.skip, "arms": {}}
    outputs = {}
    for arm in args.arms:
        for attr, value in ARMS[arm].items():
            setattr(model, attr, value)
        cache = load_prompt_cache(str(args.trace / "cache.safetensors"))
        state = cache[layer]
        mx.eval(state.state)
        start = state.offset
        before = dict(model.calls)
        times, outs = [], []
        for i, x in enumerate(xs):
            t0 = time.perf_counter()
            out = model._mla(x, state, prefix, frontier=start + i)
            mx.eval(out)
            times.append(time.perf_counter() - t0)
            outs.append(np.array(out).view(np.uint32))
        outputs[arm] = outs
        phases = {}
        for i, t in enumerate(times[args.skip:], start=args.skip):
            phases.setdefault(str((start + i) % 4), []).append(t)
        p50 = lambda v: sorted(v)[len(v) // 2] * 1e3
        result["arms"][arm] = {
            "attrs": ARMS[arm], "start_tokens": start, "p50_ms": p50(times[args.skip:]),
            "phase_p50_ms": {k: p50(v) for k, v in sorted(phases.items())},
            "calls": {k: v - before.get(k, 0) for k, v in model.calls.items() if v != before.get(k, 0)},
        }
        print(json.dumps({"arm": arm, **{k: result["arms"][arm][k] for k in ("p50_ms", "phase_p50_ms")}}),
              flush=True)
        del cache, state
        mx.clear_cache()
    pairs = [("decode_v1", "reference"), ("decode_v1_fused_attn", "reference_fused_attn")]
    result["bitwise"] = {
        f"{a}=={b}": all(np.array_equal(x, y) for x, y in zip(outputs[a], outputs[b], strict=True))
        for a, b in pairs if a in outputs and b in outputs
    }
    print(json.dumps(result["bitwise"]), flush=True)
    (args.out / "bench.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
