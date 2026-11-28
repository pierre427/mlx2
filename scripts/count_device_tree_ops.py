"""Count the device operations a lazy MTP tree round adds per round (CPU only).

Builds, on the CPU and without evaluating anything, the device-side graph of
the parts of one B1 greedy ``tree15`` round that differ between the host
round and the device arms (``device_round.tree_lazy``, ``+tree_decide``,
``+tree_fused``), at the real Qwen3.8-27B law width (vocab 248,320, bf16
logits), and counts its primitives with ``mx.export_to_dot``:

* draft token assembly (the native MTP branch grid to the proposal's int32
  tokens; the host round reads the per-branch slices instead),
* the verify input (``[anchor] + clip(tokens)`` and TensorFold's uint32 cast),
* the decision (host ``_launch_tree_laws`` greedy fence, the ``tree_decide``
  packet, or the fused ``decide_rows``).

The draft head steps and the target forward are identical in every arm and
are not counted.  ``view`` primitives (Reshape, Slice, Broadcast, ExpandDims,
Squeeze) usually launch no kernel; every other node is counted as a launch.
On the GPU the fused arm's ``verify_row`` and ``decide_rows`` are one
``CustomKernel`` launch each; on the CPU they run their reference form, which
is counted separately (``fused_reference``).  Host-to-device constant uploads
(``mx.array`` of host data) are counted as ``uploads``.

Usage: PYTHONPATH=src MLX2_PREFLIGHT_CPU_ONLY=1 python scripts/count_device_tree_ops.py [--json out]
"""

from __future__ import annotations

import argparse
import io
import json
import re
from collections import Counter
from types import SimpleNamespace

import mlx.core as mx

VIEWS = {"Reshape", "Slice", "Broadcast", "ExpandDims", "Squeeze", "Copy?"}
VOCAB = 248320


def _count(*outputs):
    buffer = io.StringIO()
    mx.export_to_dot(buffer, *outputs)
    labels = re.findall(r'label ="([A-Za-z]+)', buffer.getvalue())
    counter = Counter(labels)
    launches = sum(n for label, n in counter.items() if label not in VIEWS)
    return {"launches": launches, "views": sum(counter.values()) - launches,
            "by_primitive": dict(sorted(counter.items()))}


def _owner(batched=True):
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    owner = object.__new__(ExternalDraftBatchGenerator)
    owner.mx = mx
    owner.tree_gates = {"batched_laws": batched, "single_fence": batched,
                        "logprobs_on_request": True}
    owner.stops = set()
    return owner


def _lane(emit):
    return SimpleNamespace(uid=1, anchor=7, history=[1, 2, 3], processors=[],
                           sampling={"sampling_temp": 0, "emit_logprobs": emit})


def count(budget, *, emit=False):
    from mlx2.runtime import speculative_device_tree_fused as fused
    from mlx2.runtime.ast_tree_proposals import pattern_for_budget
    from mlx2.runtime.external_speculative import TreeDraftRow
    from mlx2.runtime.speculative_device_round import DeviceRoundPolicy
    from mlx2.runtime.speculative_device_tree import (
        DeviceTree,
        _device_tokens,
        chain_branch_parents,
    )

    pattern = pattern_for_budget(None, budget)
    parents = list(chain_branch_parents(pattern))
    nodes, rows, width = len(parents), len(parents) + 1, max(pattern)
    grid = mx.zeros((len(pattern), width), dtype=mx.uint32)
    logits = mx.zeros((1, rows, VOCAB), dtype=mx.bfloat16)
    mx.eval(grid, logits)  # leaves: their producers are not counted
    owner, lane = _owner(), _lane(emit)
    out = {"budget": budget, "pattern": list(pattern), "rows": rows}

    # Draft token assembly (device arms only; the host reads the slices).
    tokens = mx.concatenate([grid[b, :d] for b, d in enumerate(pattern)]).astype(mx.int32)
    assembly = _count(tokens)
    tokens_leaf = mx.array(list(range(nodes)), dtype=mx.int32)
    mx.eval(tokens_leaf)
    # The unfused verify input + TensorFold's uint32 row cast.
    inputs = mx.concatenate([mx.array([7], dtype=mx.int32),
                             mx.clip(_device_tokens(mx, tokens_leaf), 0, VOCAB - 1)])
    unfused_input = _count(inputs.reshape(1, -1).astype(mx.uint32))
    # Host round decision (``_launch_tree_laws`` greedy fence).
    block = TreeDraftRow(list(range(nodes)), parents, 1)
    fence, _state = owner._launch_tree_laws(lane, block, logits)
    host_laws = _count(*fence)
    # tree_decide packet.
    decide = DeviceTree(SimpleNamespace(policy=DeviceRoundPolicy.from_value(
        {"tree_lazy": True, "tree_decide": True})))
    plan = decide._plan(owner, lane, TreeDraftRow(tokens_leaf, parents, 1), logits)
    decide_packet = _count(plan.packet)
    # tree_fused reference form (CPU); on the GPU one CustomKernel each.
    row, raw = fused.verify_row(mx, grid, [b * width + k for b, d in enumerate(pattern)
                                           for k in range(d)], 7, VOCAB - 1)
    fused_ref_input = _count(row.reshape(1, -1).astype(mx.uint32), raw)
    fused_ref_decision = _count(fused.decide_rows(mx, logits[0, :rows]))
    # ``emit_logprobs``: the response normalizers join the read (unfused too).
    norms = (_count(mx.logsumexp(logits[0, :rows].astype(mx.float32), axis=-1))["launches"]
             if emit else 0)

    out["host"] = {"draft_assembly": 0, "verify_input": 0, "decision": host_laws["launches"],
                   "uploads": 1, "syncs": 2,
                   "detail": {"decision": host_laws}}
    out["tree_lazy"] = {"draft_assembly": assembly["launches"],
                        "verify_input": unfused_input["launches"],
                        "decision": host_laws["launches"], "uploads": 1, "syncs": 2,
                        "detail": {"draft_assembly": assembly, "verify_input": unfused_input}}
    out["tree_decide"] = {"draft_assembly": assembly["launches"],
                          "verify_input": unfused_input["launches"],
                          "decision": decide_packet["launches"],
                          "uploads": 1 + 4,  # anchor + parent rows, sibling, ancestor, depth masks
                          "syncs": 1, "detail": {"decision": decide_packet}}
    out["tree_fused"] = {"draft_assembly": 0, "verify_input": 2,  # verify_row + uint32 cast
                         "decision": 1 + norms, "uploads": 1, "syncs": 1,
                         "note": "verify_input = verify_row CustomKernel + TensorFold cast; "
                                 "decision = decide_rows CustomKernel; index table cached",
                         "reference_form_on_cpu": {"verify_input": fused_ref_input,
                                                   "decision": fused_ref_decision}}
    for arm in ("host", "tree_lazy", "tree_decide", "tree_fused"):
        arm_out = out[arm]
        arm_out["total_launches"] = (arm_out["draft_assembly"] + arm_out["verify_input"]
                                     + arm_out["decision"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json")
    parser.add_argument("--emit-logprobs", action="store_true")
    args = parser.parse_args()
    mx.set_default_device(mx.cpu)
    results = [count(budget, emit=args.emit_logprobs) for budget in (3, 5, 7, 9, 11, 15)]
    for item in results:
        print(json.dumps({"budget": item["budget"], **{
            arm: {k: item[arm][k] for k in ("total_launches", "uploads", "syncs")}
            for arm in ("host", "tree_lazy", "tree_decide", "tree_fused")}}))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
