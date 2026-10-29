# SPDX-License-Identifier: MIT
"""Explicit AST M16 routed MoE with M1 arithmetic for every occurrence.

One logical target-layer call; GPU grid.y owns occurrence rows. Python loops
over unique experts, never over token-row forward calls. Ordinary Source
prefill/weights.moe dispatch is unchanged. This route remains unqualified.
"""
from contextlib import nullcontext
import numpy as np

LAW = "source_ast_m1_rows_v1"


def moe(weights, x, ids, scores, prefix):
    import mlx.core as mx
    from ..runtime.gguf.source_ast_m1_expert import middle
    from ..runtime.gguf.source_ast_down_rows import down
    from ..runtime.gguf.source_middle_fusion import exp_spelling
    from ..runtime.models.deepseek_v41_quant import fused_quantize
    from .deepseek_v41_source_math import ordered_combine
    weights._ready()
    if (x.shape != (16, 5120) or x.dtype != mx.float32 or ids.shape != (16, 6)
            or ids.dtype != mx.uint32 or scores.shape != ids.shape or scores.dtype != mx.float32):
        raise ValueError("Source AST M1 law requires F32[16,5120],U32[16,6],F32scores")
    if getattr(weights, "expert_math_policy", None) != "routed_fused_fp32":
        raise ValueError("Source AST M1 law requires the served routed_fused_fp32 boundaries")
    session = getattr(weights, "source_resident_bank_session", None)
    if session is not None and session.active:
        raise ValueError("Source AST M1 law cannot enter an active B1 bank transaction")
    stream = weights.expert_stream
    values = np.asarray(ids)
    weights.source_route_host_materializations = getattr(weights, "source_route_host_materializations", 0) + 1
    if (np.any(values >= 384) or np.any(np.sort(values, axis=1)[:, 1:] == np.sort(values, axis=1)[:, :-1])):
        raise ValueError("Source AST routes require six distinct in-range experts per node")
    manager=getattr(weights,'_source_ast_bank_manager',None)
    if manager is not None:manager.record_routes(prefix,values)
    flat = values.reshape(-1)
    order = np.argsort(flat, kind="stable")
    sorted_ids = flat[order]
    memory_plan = weights._adaptive_expert_plan(values, prefix)
    sequential = memory_plan is not None and memory_plan["expert_strategy"] == "sequential_expert"
    if hasattr(stream, "observe_routes"):
        stream.observe_routes(prefix, values)
    prepared = fused_quantize(x.astype(mx.float32), "fp8").astype(x.dtype).astype(mx.float32)
    owned = not stream._active
    parts = []
    formats = {}
    with stream.window() if owned else nullcontext():
        for expert_id in np.unique(sorted_ids):
            locations = order[sorted_ids == expert_id]
            rows = mx.array(locations // 6, mx.uint32)
            slots = mx.array(locations, mx.uint32)
            data = stream.acquire(prefix, int(expert_id))
            kind = getattr(getattr(data, "layout", None), "kind", "unclassified")
            formats[kind] = formats.get(kind, 0) + len(locations)
            activation = middle(prepared, rows, scores.reshape(-1)[slots], data,
                                exp_spelling=exp_spelling())
            parts.append(down(activation, data))
            if sequential:
                stream.release_completed_experts(((int(prefix.split(".")[1]), int(expert_id)),), roots=(parts[-1],))
        output = ordered_combine(parts, values, order)
        if owned:
            mx.eval(output)
    counts = weights.scheduled_expert_calls
    for key, amount in (("source_ast_m1_logical_m16_moe", 1),
                        ("source_ast_m1_gpu_occurrences", 96),
                        ("source_ast_m1_expert_jobs", len(parts)),
                        ("source_ast_m1_python_token_forward_calls", 0)):
        counts[key] = counts.get(key, 0) + amount
    for kind, occurrences in formats.items():
        key = "source_ast_m1_" + kind + "_occurrences"
        counts[key] = counts.get(key, 0) + occurrences
    return output
