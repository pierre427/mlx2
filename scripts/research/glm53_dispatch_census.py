"""GLM-5.3 decode dispatch census: kernels one B1 decode step launches, by primitive.

Loads a recorded replay trace (cache snapshot + tokens), runs a few decode steps
under the given policy (compiled regions warm up), then walks one more step's
graph (mx.export_to_dot) and counts primitives, separating view-only ops (no
kernel) from dispatching ones. Research only.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm53_decode_replay import load  # noqa: E402

VIEWS = {"Reshape", "Broadcast", "ExpandDims", "Squeeze", "Flatten", "Unflatten", "Slice",
         "AsStrided", "Transpose", "StopGradient", "Depends", "Split", "Full"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--warm", type=int, default=6)
    parser.add_argument("--source-revision", default="unrecorded")
    parser.add_argument("--attrs", default="{}",
                        help='model attributes set after load, e.g. {"mla_fused": "decode_v1"}')
    parser.add_argument("--count-steps", type=int, default=1,
                        help="census N consecutive steps (phases by position %% 4); 1 = one step")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import mlx.core as mx

    from mlx2.runtime.models.cache import load_prompt_cache

    adapter, policy = load(args.policy)
    model = adapter.model
    attrs = json.loads(args.attrs)
    for attr, value in attrs.items():
        setattr(model, attr, value)
    tokens = json.loads((args.trace / "trace.json").read_text())["tokens"]
    cache = load_prompt_cache(str(args.trace / "cache.safetensors"))
    for c in cache:  # replay under the current source identity
        if hasattr(c, "identity"):
            c.identity = model.identity
    mx.eval([c.state for c in cache])
    for i in range(args.warm):
        mx.eval(model(mx.array([[tokens[i]]]), cache))
    prompt = json.loads((args.trace / "trace.json").read_text())["prompt_tokens"]
    steps = []
    for k in range(args.count_steps):
        i = args.warm + k
        logits = model(mx.array([[tokens[i]]]), cache)
        dot = args.out / ("step.dot" if k == 0 else f"step{k}.dot")
        mx.export_to_dot(str(dot), logits)
        counts = collections.Counter(re.findall(r'label ="([^"]+)"', dot.read_text()))
        dispatch = {key: v for key, v in counts.items() if key not in VIEWS}
        steps.append({"position": prompt + i, "phase": (prompt + i) % 4,
                      "total_nodes": sum(counts.values()), "dispatching": sum(dispatch.values()),
                      "views": sum(v for key, v in counts.items() if key in VIEWS),
                      "by_primitive": dict(sorted(dispatch.items(), key=lambda kv: -kv[1]))})
        mx.eval(logits)
    record = {"policy": policy, "attrs": attrs, "source_revision": args.source_revision,
              **{key: steps[0][key] for key in ("total_nodes", "dispatching", "views", "by_primitive")}}
    if args.count_steps > 1:
        record["steps"] = steps
        record["mean_dispatching"] = sum(s["dispatching"] for s in steps) / len(steps)
    (args.out / "census.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("total_nodes", "dispatching", "views")}))
    for step in record.get("steps", []):
        print(json.dumps({k: step[k] for k in ("position", "phase", "dispatching")}))
    print(json.dumps(dict(list(record["by_primitive"].items())[:25])))


if __name__ == "__main__":
    main()
