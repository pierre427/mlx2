# SPDX-License-Identifier: MIT
"""Opt-in lockstep MoE site: one expert acquisition per distinct expert per verify.

``deepseek_verify_moe_rows: m1_rows_v1`` (requires
``deepseek_verify_rows: lockstep_multirow_v1``). The lockstep chain verifier
(deepseek_v41_dspark_chain.py) shares dense and head coefficient reads across
verify rows, but routed MoE stays width-one per row: a W-row verify acquires
and multiplies each row's six experts separately even when rows route to the
same expert. Routed experts are most of the bytes per token on the streamed
Source checkpoint, so this is the verify cost that matters there.

With this policy every row parks at its layer's ``weights.moe`` site (after its
router, outside the attention/cache critical region). When all rows are parked
at the same layer the site runs once: each distinct expert is acquired once,
and every occurrence row is computed with the Source AST M1 law (the
``source_ast_m1_rows_v1`` middle/DOWN kernels: grid.y owns occurrence rows,
per-occurrence arithmetic is the ordinary M1 law), then each row's slots are
summed in ascending expert order by the ordinary ordered combine. That law was
raw-bit exact against ordinary M1 decode at 16 rows on the checkpoint
(model-feature-decisions: source_ast_target_m1_row_invariant_16); this site
reuses it at 2..6 rows. Any structural mismatch declines to the ordinary
per-row ``moe`` (exact), counted in ``lockstep_moe_declines``.

Counters (``verified_commit_stats``): ``lockstep_moe_calls``,
``lockstep_moe_rows``, ``lockstep_moe_occurrences``,
``lockstep_moe_expert_jobs`` (distinct experts acquired) and
``lockstep_moe_acquisitions_saved`` (occurrences minus distinct experts).
The distinct/occurrence ratio is the measured expert overlap of verify rows.

Implemented, unqualified, not selected by default, untested on GPU.
Provenance: provenance/deepseek-v41-dspark-verify-ledger.json.
"""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

KEY = "deepseek_verify_moe_rows"
LAW = "m1_rows_v1"
POLICIES = ("reference", LAW)
TOPK = 6
MAX_ROWS = 16  # kernel occurrence-row limit; lockstep verifies at most six rows


class Decline(Exception):
    """Structural refusal: each row runs its ordinary ``moe``."""


def plan(values):
    """Pure route plan for stacked rows: expert-sorted occurrence order."""
    if (not isinstance(values, np.ndarray) or values.dtype != np.uint32 or values.ndim != 2
            or values.shape[1] != TOPK or not 1 <= values.shape[0] <= MAX_ROWS):
        raise Decline
    ordered = np.sort(values, axis=1)
    if np.any(values >= 384) or np.any(ordered[:, 1:] == ordered[:, :-1]):
        raise Decline
    flat = values.reshape(-1)
    order = np.argsort(flat, kind="stable")
    sorted_ids = flat[order]
    experts = np.unique(sorted_ids)
    return SimpleNamespace(values=values, order=order, sorted_ids=sorted_ids, experts=experts,
                           occurrences=int(flat.size), distinct=int(experts.size))


def refusal(weights):
    if getattr(weights, "expert_math_policy", None) != "routed_fused_fp32":
        return "m1_rows_v1 requires routed_fused_fp32 expert math"
    session = getattr(weights, "source_resident_bank_session", None)
    if session is not None and getattr(session, "active", False):
        return "an active resident bank session owns MoE"
    stream = getattr(weights, "expert_stream", None)
    if stream is None or not callable(getattr(stream, "acquire", None)):
        return "no target expert provider"
    return None


def _kernels():
    from ..runtime.gguf.source_ast_down_rows import down
    from ..runtime.gguf.source_ast_m1_expert import middle
    from ..runtime.gguf.source_middle_fusion import exp_spelling
    from ..runtime.models.deepseek_v41_quant import fused_quantize
    from .deepseek_v41_source_math import ordered_combine
    import mlx.core as mx

    return SimpleNamespace(
        prepare=lambda x: fused_quantize(x.astype(mx.float32), "fp8").astype(x.dtype).astype(mx.float32),
        middle=lambda prepared, rows, scores, data: middle(prepared, rows, scores, data, exp_spelling=exp_spelling()),
        down=down, combine=ordered_combine)


def rows_moe(weights, parks, stats, *, kernels=None):
    """Serve W parked ``moe(x[1,H], ids[1,6], scores[1,6], prefix)`` calls at one site."""
    import mlx.core as mx

    reason = refusal(weights)
    if reason is not None:
        raise Decline
    if any(p.kwargs or len(p.args) != 4 for p in parks):
        raise Decline
    prefix = parks[0].args[3]
    if any(p.args[3] != prefix for p in parks) or type(prefix) is not str:
        raise Decline
    xs, ids, scores = zip(*(p.args[:3] for p in parks))
    hidden = xs[0].shape[-1] if isinstance(xs[0], mx.array) else None
    if (hidden is None or any(not isinstance(a, mx.array) for a in (*xs, *ids, *scores))
            or any(x.shape != (1, hidden) or x.dtype != mx.float32 for x in xs)
            or any(i.shape != (1, TOPK) for i in ids) or any(s.shape != (1, TOPK) for s in scores)):
        raise Decline
    x = mx.concatenate(xs, 0)
    values = np.asarray(mx.concatenate(ids, 0)).astype(np.uint32, copy=False)
    score = mx.concatenate(scores, 0).astype(mx.float32)
    route = plan(values)
    k = kernels or _kernels()
    weights._ready()
    weights.source_route_host_materializations = getattr(weights, "source_route_host_materializations", 0) + 1
    stream = weights.expert_stream
    memory_plan = weights._adaptive_expert_plan(values, prefix) if callable(
        getattr(weights, "_adaptive_expert_plan", None)) else None
    sequential = memory_plan is not None and memory_plan.get("expert_strategy") == "sequential_expert"
    if callable(getattr(stream, "observe_routes", None)):
        stream.observe_routes(prefix, values)
    prepared = k.prepare(x)
    owned = not getattr(stream, "_active", True)
    parts = []
    with stream.window() if owned else nullcontext():
        for expert in route.experts:
            locations = route.order[route.sorted_ids == expert]
            rows = mx.array(locations // TOPK, mx.uint32)
            slots = mx.array(locations, mx.uint32)
            data = stream.acquire(prefix, int(expert))
            parts.append(k.down(k.middle(prepared, rows, score.reshape(-1)[slots], data), data))
            if sequential:
                stream.release_completed_experts(((int(prefix.split(".")[1]), int(expert)),), roots=(parts[-1],))
        output = k.combine(parts, values, route.order)
        if owned:
            mx.eval(output)
    for key, amount in (("lockstep_moe_calls", 1), ("lockstep_moe_rows", len(parks)),
                        ("lockstep_moe_occurrences", route.occurrences),
                        ("lockstep_moe_expert_jobs", route.distinct),
                        ("lockstep_moe_acquisitions_saved", route.occurrences - route.distinct)):
        stats[key] = stats.get(key, 0) + amount
    counts = getattr(weights, "scheduled_expert_calls", None)
    if isinstance(counts, dict):
        counts["lockstep_m1_rows_moe"] = counts.get("lockstep_m1_rows_moe", 0) + 1
    return [output[r:r + 1] for r in range(len(parks))]
