"""Metal micro-profile of the device-tree round's added operations (GPU, no model).

Diagnostic only, never a serving or performance verdict; book it on the GPU
through the lease (one process, about a minute).  At the Qwen3.8-27B law
width (vocab 248,320, bf16 logits) and every node budget the tile chooser
uses (3/5/7/9/11/15), it times, with an ``mx.eval`` fence after each group
(median of ``--reps`` warm repetitions):

* ``host_laws``       the host round's greedy law fence (``_launch_tree_laws``)
* ``decide_packet``   the ``tree_decide`` packet (``DeviceTree._plan``)
* ``fused_decide``    ``decide_rows`` (one Metal kernel)
* ``lazy_input``      drafter token assembly + ``[anchor] + clip`` + uint32 cast
* ``fused_input``     ``verify_row`` (one Metal kernel) + uint32 cast
* ``sync``            one fence on an already idle queue (``mx.eval`` of a
                      one-element op): the floor of any host read
* ``*_build``         graph construction only (Python + MLX op creation, no
                      eval): the host share of each group

It also checks the fused kernels against their reference form on these
shapes (``parity``).  Each group's fenced time is its full serialized cost
(encode + GPU + wake); in a served round the GPU part overlaps host work.

Usage: PYTHONPATH=src python scripts/profile_mtp_device_tree_ops.py --out receipt.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

VOCAB = 248320
BUDGETS = (3, 5, 7, 9, 11, 15)


def _owner():
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    owner = object.__new__(ExternalDraftBatchGenerator)
    owner.mx = mx
    owner.tree_gates = {"batched_laws": True, "single_fence": True, "logprobs_on_request": True}
    owner.stops = set()
    return owner


def _time(build, reps, *, evaluate=True):
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        outputs = build()
        if evaluate:
            mx.eval(*outputs)
        samples.append((time.perf_counter() - start) * 1e3)
    return {"median_ms": statistics.median(samples), "p10_ms": float(np.percentile(samples, 10)),
            "p90_ms": float(np.percentile(samples, 90))}


def profile(budget, reps, rng):
    from mlx2.runtime import speculative_device_tree_fused as fused
    from mlx2.runtime.ast_tree_proposals import pattern_for_budget
    from mlx2.runtime.external_speculative import TreeDraftRow
    from mlx2.runtime.speculative_device_round import DeviceRoundPolicy
    from mlx2.runtime.speculative_device_tree import DeviceTree, chain_branch_parents

    pattern = pattern_for_budget(None, budget)
    parents = list(chain_branch_parents(pattern))
    nodes, rows, width = len(parents), len(parents) + 1, max(pattern)
    logits = mx.array(rng.normal(size=(1, rows, VOCAB)).astype(np.float32)).astype(mx.bfloat16)
    grid = mx.array(rng.integers(0, VOCAB, size=(len(pattern), width)), dtype=mx.uint32)
    mx.eval(logits, grid)
    host_tokens = np.asarray(grid).reshape(-1)
    index = [b * width + k for b, d in enumerate(pattern) for k in range(d)]
    tokens_host = [int(host_tokens[i]) for i in index]
    lane = SimpleNamespace(uid=1, anchor=7, history=[1, 2, 3], processors=[],
                           sampling={"sampling_temp": 0, "emit_logprobs": False})
    owner = _owner()
    decide = DeviceTree(SimpleNamespace(policy=DeviceRoundPolicy.from_value(
        {"tree_lazy": True, "tree_decide": True})))
    host_block = TreeDraftRow(tokens_host, parents, 1)

    def host_laws():
        return owner._launch_tree_laws(lane, host_block, logits)[0]

    def lazy_input():
        tokens = mx.concatenate([grid[b, :d] for b, d in enumerate(pattern)]).astype(mx.int32)
        inputs = mx.concatenate([mx.array([7], dtype=mx.int32), mx.clip(tokens, 0, VOCAB - 1)])
        return [inputs.reshape(1, -1).astype(mx.uint32), tokens]

    # The drafts as an evaluated array: their assembly is ``lazy_input``'s.
    tokens_leaf = mx.array(tokens_host, dtype=mx.int32)
    mx.eval(tokens_leaf)

    def decide_packet():
        return [decide._plan(owner, lane, TreeDraftRow(tokens_leaf, parents, 1), logits).packet]

    def fused_input():
        row, raw = fused.verify_row(mx, grid, index, 7, VOCAB - 1)
        return [row.reshape(1, -1).astype(mx.uint32), raw]

    def fused_decide():
        return [fused.decide_rows(mx, logits[0, :rows])]

    groups = {"host_laws": host_laws, "decide_packet": decide_packet,
              "fused_decide": fused_decide, "lazy_input": lazy_input, "fused_input": fused_input}
    for build in groups.values():  # warm: kernel compiles, allocator
        for _ in range(3):
            mx.eval(*build())
    out = {"budget": budget, "pattern": list(pattern), "rows": rows, "nodes": nodes}
    for name, build in groups.items():
        out[name] = _time(build, reps)
        out[f"{name}_build"] = _time(build, reps, evaluate=False)
    one = mx.array([1.0])
    mx.eval(one)
    out["sync"] = _time(lambda: [one + 1], reps)
    # Parity on these shapes (also non-finite rows).
    special = np.asarray(rng.normal(size=(rows, VOCAB)), dtype=np.float32)
    special[0, :] = -np.inf
    if rows > 1:
        special[1, [11, 11 + 1024]] = 7.0
    if rows > 2:
        special[2, 99] = np.nan
    checks = []
    for values in (logits[0, :rows], mx.array(special).astype(mx.bfloat16), mx.array(special)):
        got = np.asarray(fused.decide_rows(mx, values)).tolist()
        want = np.asarray(fused.reference_decide_rows(mx, values)).tolist()
        argmax = np.asarray(mx.argmax(values.astype(mx.float32), axis=-1)).tolist()
        checks.append(got[rows:] == want[rows:] and got[:rows] == argmax)
    row, raw = fused.verify_row(mx, grid, index, 7, VOCAB - 1)
    reference = [7, *[min(max(token, 0), VOCAB - 1) for token in tokens_host]]
    checks.append(np.asarray(row).tolist() == reference
                  and np.asarray(raw).tolist() == tokens_host)
    out["parity"] = all(checks)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--reps", type=int, default=200)
    args = parser.parse_args()
    if not mx.metal.is_available():
        raise SystemExit("Metal is not available")
    mx.set_default_device(mx.gpu)
    rng = np.random.default_rng(0)
    results = [profile(budget, args.reps, rng) for budget in BUDGETS]
    receipt = {
        "schema": "mlx2.device-tree-op-profile.v1",
        "kind": "diagnostic micro-profile, no model, no thermal hold",
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain",
                                               "--untracked-files=no"], text=True).strip()),
        "mlx_version": mx.__version__, "device": str(mx.default_device()),
        "vocab": VOCAB, "reps": args.reps, "budgets": results,
        "parity": all(item["parity"] for item in results),
    }
    tmp = args.out + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(receipt, handle, indent=1)
    os.replace(tmp, args.out)
    for item in results:
        print(json.dumps({"budget": item["budget"], "parity": item["parity"], **{
            name: round(item[name]["median_ms"], 4) for name in (
                "host_laws", "decide_packet", "fused_decide", "lazy_input", "fused_input",
                "sync", "decide_packet_build", "host_laws_build")}}))
    raise SystemExit(0 if receipt["parity"] else 1)


if __name__ == "__main__":
    main()
