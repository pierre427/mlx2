"""GLM-5.3 B1 decode kernel bandwidth by weight class and format (in-graph).

For every rank-2 projection role (tensor name without ``blk.N.``) grouped by
GGUF type and shape, one graph runs the m=1 ``weights.linear`` of that role in
every layer that has it, with distinct inputs, and is evaluated once per rep:
no per-call sync, cold weights per call (each layer's tensor is read once per
rep), the same kernel the decode step dispatches under the execution policy.
The routed MoE (and the shared-folded MoE) is timed the same way across its
layers with random distinct expert ids. Reports achieved GB/s against the
packed bytes read, the share of a decode step's weight bytes, and the time
above the streaming roofline. Research only; no model output is produced.
"""

from __future__ import annotations

import argparse
import collections
import json
import platform
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm53_ordinary_profile import CHECKPOINT, MODEL

STREAM_GB_S = 557.0  # measured M5 Max streaming read (tuning library, roofline campaign)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--source-revision", default="unrecorded")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to reuse a nonempty evidence directory")
    args.out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx

    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter

    info = mx.device_info()
    if info.get("max_recommended_working_set_size"):
        mx.set_wired_limit(info["max_recommended_working_set_size"])
    policy = json.loads(args.policy.read_text())
    adapter = Glm53GGUFAdapter(MODEL, execution_policy=policy)
    model, weights = adapter.model, adapter.weights

    def timed(build):
        outs = build()
        mx.eval(outs)  # warm: compile kernels, upload weights
        times = []
        for _ in range(args.reps):
            outs = build()
            t0 = time.perf_counter()
            mx.eval(outs)
            times.append(time.perf_counter() - t0)
        times.sort()
        return times[len(times) // 2]

    roles = collections.defaultdict(list)
    for name, tensor in weights.tensors.items():
        m = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
        if len(tensor.shape) != 2 or "_exps" in name:
            continue
        role = m.group(2) if m else name
        roles[(role, tensor.format, tuple(tensor.shape))].append(name)

    rows = []
    rng = mx.random.key(5310)
    for (role, fmt, shape), names in sorted(roles.items()):
        if role in ("token_embd.weight",):
            continue
        k_in = shape[0]
        xs = [mx.random.normal((1, k_in), key=mx.random.split(rng, len(names))[i]).astype(mx.float32)
              for i in range(len(names))]
        mx.eval(xs)
        try:
            seconds = timed(lambda: [weights.linear(n)(x) for n, x in zip(names, xs)])
        except Exception as error:  # noqa: BLE001 -- a refused role is a result too
            rows.append({"role": role, "format": fmt, "shape": list(shape), "error": repr(error)})
            continue
        nbytes = sum(weights.tensors[n].nbytes for n in names)
        rows.append({"role": role, "format": fmt, "shape": list(shape), "layers": len(names),
                     "bytes": nbytes, "ms": seconds * 1e3, "gb_s": nbytes / seconds / 1e9,
                     "roofline_ms": nbytes / STREAM_GB_S / 1e6})
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}),
              flush=True)

    # Routed MoE across its layers: 8 distinct random experts per layer.
    moe_layers = sorted({int(re.match(r"blk\.(\d+)\.", n).group(1)) for n in weights.tensors
                         if n.endswith("ffn_gate_exps.weight")})
    hidden = weights.tensors[f"blk.{moe_layers[0]}.ffn_gate_exps.weight"].shape[0]
    experts = weights.tensors[f"blk.{moe_layers[0]}.ffn_gate_exps.weight"].shape[2]
    x = mx.random.normal((1, hidden)).astype(mx.float32)
    ids = [mx.random.permutation(experts, key=mx.random.key(i))[:8].astype(mx.uint32)[None]
           for i in range(len(moe_layers))]
    scores = mx.full((1, 8), 0.125, dtype=mx.float32)
    mx.eval(x, ids)
    per_expert = {}
    for layer in moe_layers:
        per_expert[layer] = sum(weights.tensors[f"blk.{layer}.ffn_{p}_exps.weight"].nbytes
                                for p in ("gate", "up", "down")) / experts
    moe_bytes = sum(8 * per_expert[layer] for layer in moe_layers)
    for label, fn in (
        ("moe_routed_top8", lambda: [weights.moe(x, i, scores, f"blk.{layer}")
                                     for layer, i in zip(moe_layers, ids)]),
        ("moe_routed_top8+shared_fold", lambda: [weights.moe_with_shared(
            x, i, scores, f"blk.{layer}", f"blk.{layer}.ffn_{{}}_shexp.weight")
            for layer, i in zip(moe_layers, ids)]),
    ):
        try:
            seconds = timed(fn)
        except Exception as error:  # noqa: BLE001
            rows.append({"role": label, "error": repr(error)})
            continue
        nbytes = moe_bytes + (sum(weights.tensors[f"blk.{layer}.ffn_{p}_shexp.weight"].nbytes
                                  for layer in moe_layers for p in ("gate", "up", "down"))
                              if "shared" in label else 0)
        rows.append({"role": label, "layers": len(moe_layers), "bytes": nbytes, "ms": seconds * 1e3,
                     "gb_s": nbytes / seconds / 1e9, "roofline_ms": nbytes / STREAM_GB_S / 1e6})
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}),
              flush=True)

    measured = [r for r in rows if "ms" in r]
    excess = sorted(measured, key=lambda r: -(r["ms"] - r["roofline_ms"]))
    record = {"schema": "mlx2.glm53-decode-kernel-bench.v1", "checkpoint": CHECKPOINT,
              "source_revision": args.source_revision, "host": platform.node(), "mlx": mx.__version__,
              "execution_policy": policy, "stream_gb_s": STREAM_GB_S, "reps": args.reps, "rows": rows,
              "excess_ranked": [{"role": r["role"], "format": r.get("format"), "excess_ms": r["ms"] - r["roofline_ms"],
                                 "gb_s": r["gb_s"]} for r in excess]}
    (args.out / "kernel-bench.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
