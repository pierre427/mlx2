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

Trunk only by default: the NextN block (``blk.45`` on GLM-5.3, any block at
or above the model's trunk layer count) never runs in ordinary decode and is excluded
unless ``--with-nextn``.

``--row-policies a,b``: every role and MoE row is timed under each provider
``row_policy`` in turn (rep order alternates A B / B A so drift lands on both
arms), and each arm's outputs are compared bit for bit with the first arm's
on the same inputs (``bitwise_vs_first``), i.e. on the checkpoint's real
tensors of every format it holds.

Two delegating modes keep GPU work inside the bench.sh lease recipe:
``--gpu-tests <pytest args>`` runs pytest with MLX2_RUN_GPU_TESTS=1 at this
checkout's ``src``; ``--replay <glm53_decode_replay.py args>`` runs that
harness with ``weights.<attr>`` variant attributes set on the weight
provider (e.g. ``{"weights.row_policy": "kblock_rows_ahead2"}``).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm53_ordinary_profile import CHECKPOINT, MODEL

STREAM_GB_S = 557.0  # measured M5 Max streaming read (tuning library, roofline campaign)


def _gpu_tests(out, revision, rest):
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, MLX2_RUN_GPU_TESTS="1", PYTHONPATH=str(root / "src"))
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", *rest], cwd=root, env=env, check=False,
                          capture_output=True, text=True)
    (out / "pytest.txt").write_text(proc.stdout + proc.stderr)
    (out / "result.json").write_text(json.dumps({"checkpoint": CHECKPOINT, "source_revision": revision,
                                                 "args": rest, "returncode": proc.returncode}) + "\n")
    print(proc.stdout[-4000:], proc.stderr[-2000:], flush=True)
    return proc.returncode


def _replay(out, revision, rest):
    import glm53_decode_replay as replay

    def assign(obj, attr, value):
        # "weights.<attr>" sets an attribute of the weight provider; anything
        # else is a model attribute (the harness's own behaviour).
        if attr.startswith("weights."):
            return setattr(obj.weights, attr.removeprefix("weights."), value)
        return setattr(obj, attr, value)

    replay.setattr = assign  # module global shadows the builtin inside replay()
    sys.argv = [replay.__file__, "--out", str(out), "--source-revision", revision, *rest]
    replay.main()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--source-revision", default="unrecorded")
    parser.add_argument("--with-nextn", action="store_true", help="include the NextN block (blk.45)")
    parser.add_argument("--row-policies", default=None,
                        help="comma-separated provider row_policy arms, each optionally @ROWS "
                             "(e.g. kblock_rows,kblock_rows_ahead2,kblock_rows_ahead2@2); default: the policy's own")
    parser.add_argument("--roles", default=None, help="regex: only roles matching it (MoE rows always)")
    parser.add_argument("--gpu-tests", action="store_true", help="run pytest (remaining args) on the GPU")
    parser.add_argument("--replay", action="store_true", help="run glm53_decode_replay.py (remaining args)")
    # Everything after --gpu-tests / --replay belongs to the delegate verbatim
    # (the replay harness has its own --policy).
    argv = sys.argv[1:]
    cut = next((i for i, a in enumerate(argv) if a in ("--gpu-tests", "--replay")), len(argv))
    args = parser.parse_args(argv[:cut + 1])
    rest = argv[cut + 1:]
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to reuse a nonempty evidence directory")
    if args.gpu_tests or args.replay:
        if args.gpu_tests:
            args.out.mkdir(parents=True, exist_ok=True)
            sys.exit(_gpu_tests(args.out, args.source_revision, rest))
        sys.exit(_replay(args.out, args.source_revision, rest))
    if args.policy is None:
        parser.error("--policy is required")
    args.out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx
    import numpy as np

    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter

    info = mx.device_info()
    if info.get("max_recommended_working_set_size"):
        mx.set_wired_limit(info["max_recommended_working_set_size"])
    policy = json.loads(args.policy.read_text())
    adapter = Glm53GGUFAdapter(MODEL, execution_policy=policy)
    model, weights = adapter.model, adapter.weights
    arms = args.row_policies.split(",") if args.row_policies else [weights.row_policy]
    trunk = len(model.layers)  # trunk blocks; NextN blocks are numbered from here

    def in_trunk(name):
        m = re.match(r"blk\.(\d+)\.", name)
        return args.with_nextn or m is None or int(m.group(1)) < trunk

    def use(arm):
        # "policy" or "policy@rows" (rows per simdgroup of the one-token row launches).
        policy_name, _, block = arm.partition("@")
        weights.row_policy = policy_name
        weights.kblock_row_block = int(block or 4)

    def timed(build):
        """Median seconds per arm, counterbalanced; outputs of each arm's warm call."""
        warm = {}
        for arm in arms:
            use(arm)
            outs = build()
            mx.eval(outs)  # warm: compile kernels, upload weights
            warm[arm] = outs
        times = {arm: [] for arm in arms}
        for rep in range(args.reps):
            for arm in (arms if rep % 2 == 0 else arms[::-1]):
                use(arm)
                outs = build()
                t0 = time.perf_counter()
                mx.eval(outs)
                times[arm].append(time.perf_counter() - t0)
        first = [np.array(o).view(np.uint32) for o in warm[arms[0]]]
        same = {arm: all(np.array_equal(np.array(o).view(np.uint32), b) for o, b in zip(warm[arm], first))
                for arm in arms}
        return {arm: sorted(t)[len(t) // 2] for arm, t in times.items()}, same

    def row(label, extra, nbytes, seconds, same):
        r = dict(extra, role=label, bytes=nbytes, arms={})
        for arm in arms:
            r["arms"][arm] = {"ms": seconds[arm] * 1e3, "gb_s": nbytes / seconds[arm] / 1e9,
                              "bitwise_vs_first": same[arm]}
        base = r["arms"][arms[0]]
        r.update(ms=base["ms"], gb_s=base["gb_s"], roofline_ms=nbytes / STREAM_GB_S / 1e6)
        print(json.dumps({"role": label, **{k: v for k, v in extra.items() if k != "names"},
                          **{a: [round(v["ms"], 3), round(v["gb_s"], 1), v["bitwise_vs_first"]]
                             for a, v in r["arms"].items()}}), flush=True)
        return r

    roles = collections.defaultdict(list)
    for name, tensor in weights.tensors.items():
        m = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
        if len(tensor.shape) != 2 or "_exps" in name or not in_trunk(name):
            continue
        role = m.group(2) if m else name
        roles[(role, tensor.format, tuple(tensor.shape))].append(name)

    rows = []
    rng = mx.random.key(5310)
    for (role, fmt, shape), names in sorted(roles.items()):
        if role in ("token_embd.weight",) or (args.roles and not re.search(args.roles, role)):
            continue
        k_in = shape[0]
        xs = [mx.random.normal((1, k_in), key=mx.random.split(rng, len(names))[i]).astype(mx.float32)
              for i in range(len(names))]
        mx.eval(xs)
        try:
            seconds, same = timed(lambda names=names, xs=xs: [weights.linear(n)(x) for n, x in zip(names, xs)])
        except Exception as error:  # noqa: BLE001 -- a refused role is a result too
            rows.append({"role": role, "format": fmt, "shape": list(shape), "error": repr(error)})
            continue
        nbytes = sum(weights.tensors[n].nbytes for n in names)
        rows.append(row(role, {"format": fmt, "shape": list(shape), "layers": len(names)}, nbytes, seconds, same))

    # Routed MoE across its trunk layers: 8 distinct random experts per layer.
    moe_layers = sorted({int(re.match(r"blk\.(\d+)\.", n).group(1)) for n in weights.tensors
                         if n.endswith("ffn_gate_exps.weight") and in_trunk(n)})
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
            seconds, same = timed(fn)
        except Exception as error:  # noqa: BLE001
            rows.append({"role": label, "error": repr(error)})
            continue
        nbytes = moe_bytes + (sum(weights.tensors[f"blk.{layer}.ffn_{p}_shexp.weight"].nbytes
                                  for layer in moe_layers for p in ("gate", "up", "down"))
                              if "shared" in label else 0)
        rows.append(row(label, {"layers": len(moe_layers)}, nbytes, seconds, same))

    measured = [r for r in rows if "ms" in r]
    excess = sorted(measured, key=lambda r: -(r["ms"] - r["roofline_ms"]))
    record = {"schema": "mlx2.glm53-decode-kernel-bench.v2", "checkpoint": CHECKPOINT,
              "source_revision": args.source_revision, "host": platform.node(), "mlx": mx.__version__,
              "execution_policy": policy, "row_policy_arms": arms, "trunk_blocks": trunk,
              "with_nextn": args.with_nextn, "stream_gb_s": STREAM_GB_S, "reps": args.reps, "rows": rows,
              # Green only when every requested role ran: an errored role or an
              # empty run is not evidence of bitwise equality.
              "errors": [r["role"] for r in rows if "error" in r],
              "all_bitwise": bool(measured) and len(measured) == len(rows)
              and all(a["bitwise_vs_first"] for r in measured for a in r["arms"].values()),
              "excess_ranked": [{"role": r["role"], "format": r.get("format"), "excess_ms": r["ms"] - r["roofline_ms"],
                                 "gb_s": r["gb_s"]} for r in excess]}
    (args.out / "kernel-bench.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
