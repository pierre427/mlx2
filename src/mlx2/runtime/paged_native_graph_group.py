"""Default-off ready B2 step over one shared native Qwen3 graph."""

from __future__ import annotations

import os

from contextlib import ExitStack

from ..adapters.qwen3_paged_candidate import PackedLane
from .paged_native_continuation import NativeQwen3Continuation
from .paged_request_transaction import CandidateRequest
from .paged_native_contract import native_layer_count


def can_prime_serving_graph_b2(lanes: tuple[NativeQwen3Continuation, ...]) -> bool:
    """Both admitted lanes must exist before either first token is emitted."""
    if type(lanes) is not tuple or len(lanes) != 2 or lanes[0] is lanes[1]:
        return False
    first, second = lanes
    if (type(first) is not NativeQwen3Continuation or
            type(second) is not NativeQwen3Continuation):
        return False
    candidate = first.candidate
    return bool(
        getattr(candidate, "_serving_b2", False) is True and
        getattr(candidate, "_research_staged_graph", False) is True and
        second.candidate is candidate and first.owner is not second.owner and
        first.owner.supported_planes == second.owner.supported_planes == getattr(candidate, "state_planes", ("kv",)) and
        not first.research_only and not second.research_only and
        not first.closed and not second.closed and
        first._first_logits is not None and second._first_logits is not None and
        first._pending_token is None and second._pending_token is None
    )


def can_run_research_graph_b2(lanes: tuple[NativeQwen3Continuation, ...]) -> bool:
    """Require two live ready decode lanes sharing an explicit graph candidate."""
    if type(lanes) is not tuple or len(lanes) != 2 or lanes[0] is lanes[1]:
        return False
    first, second = lanes
    if (type(first) is not NativeQwen3Continuation or
            type(second) is not NativeQwen3Continuation):
        return False
    candidate = first.candidate
    serving = getattr(candidate, "_serving_b2", False) is True
    return bool(
        first.research_only == second.research_only and
        first.research_only == (not serving) and
        getattr(candidate, "_research_staged_graph", False) is True and
        second.candidate is candidate and first.owner is not second.owner and
        not first.closed and not second.closed and
        first._first_logits is None and second._first_logits is None and
        type(first._pending_token) is int and type(second._pending_token) is int and
        first.owner.supported_planes == second.owner.supported_planes == getattr(candidate, "state_planes", ("kv",))
    )


def run_research_graph_b2(lanes: tuple[NativeQwen3Continuation, ...]):
    """Return two sampled responses after one terminal-proved packed step.

    The owning BatchGenerator holds its lifecycle lock and retires both lanes
    on any exception. Serving needs an exact source-bound cohort admission;
    this physical step never establishes a performance price or qualification.
    """
    if not can_run_research_graph_b2(lanes):
        raise ValueError("two ready native graph lanes are required")
    candidate = lanes[0].candidate
    if getattr(candidate, 'bootstrap_generation', 1) == 0:
        from .packed_prefill_receipt import bootstrap_prefill_attribution
        bootstrap_prefill_attribution(candidate)
    serving = getattr(candidate, "_serving_b2", False) is True
    backend = candidate.backend
    branches = []
    prepared = []
    with ExitStack() as snapshots:
        public = tuple(snapshots.enter_context(lane.owner.snapshot()) for lane in lanes)
        if any(view.revision != lane.revision or view.offset != len(lane.tokens) or
               view.generation < getattr(candidate, "bootstrap_generation", 1) or len(view.layer_owners) != native_layer_count(candidate)
               for lane, view in zip(lanes, public)):
            raise RuntimeError("native B2 public reader state drifted")
        public_metadata = tuple((view.generation, view.offset) for view in public)
        if all(getattr(lane.owner, "_reuse_private_tail", False) for lane in lanes):
            # The BatchGenerator lifecycle lock excludes competing request
            # operations. Snapshot itself submits no native read; begin below
            # separately rejects any still-pending writer or reader epoch.
            # Release these metadata-only leases before requesting the private
            # tail permit. Subsequent response construction uses scalar copies.
            snapshots.close()
        physical_before = backend.profile_counters_snapshot() if serving else None
        rope_before = getattr(candidate, "_vector_q1_rope_calls", 0)
        direct_fence_before = getattr(backend, "direct_grouped_fence_reads", 0)
        before_reads = backend.read_submissions
        before_terminals = backend.terminal_successes
        before_spans = len(getattr(backend, "staged_read_spans", ()))
        try:
            for lane in lanes:
                branches.append(lane.owner.begin(CandidateRequest(
                    lane.uid, lane.revision, 1, getattr(candidate, "state_planes", ("kv",)))))
            packed = tuple(
                candidate.packed_lane((lane._pending_token,), branch)
                if callable(getattr(candidate, "packed_lane", None)) else
                PackedLane((lane._pending_token,), branch.layers)
                for lane, branch in zip(lanes, branches))
            logits, graph_receipt = candidate.forward_staged(
                packed, tuple(branches), permit_candidate=True,
                **({"reserve_scratch": candidate.reserve_serving_scratch}
                   if callable(getattr(candidate, "reserve_serving_scratch", None)) else {}))
            depth = native_layer_count(candidate)
            if (len(logits) != 2 or graph_receipt.get("packed_lanes") != 2 or
                    backend.read_submissions - before_reads != depth or
                    backend.terminal_successes - before_terminals != depth or
                    tuple(getattr(backend, "staged_read_spans", ())[before_spans:]) !=
                    (2,) * depth):
                raise RuntimeError("native B2 lacks one two-span read per layer")
            physical = None
            combined_counts = None
            if serving and not getattr(candidate, "owns_physical_dispatch_proof", False):
                after = backend.profile_counters_snapshot()
                stock_sdpa = bool(getattr(candidate, "_b2_stock_sdpa", False))
                physical = {
                    key: after[key] - physical_before[key]
                    for key in ("grouped_q1_writes", "native_write_dispatches",
                                "q1_tile_dispatches")
                }
                if physical != {"grouped_q1_writes": depth,
                                "native_write_dispatches": 0,
                                "q1_tile_dispatches": 0 if stock_sdpa else depth}:
                    raise RuntimeError("serving B2 physical grouped/tile dispatch proof differs")
                if (getattr(candidate, "_vector_q1_rope_calls", 0) - rope_before !=
                        (depth if getattr(candidate, "_vector_q1_rope", False) else 0)):
                    raise RuntimeError("serving B2 vector Q1 RoPE call proof differs")
                if (getattr(backend, "direct_grouped_fence_reads", 0) -
                        direct_fence_before !=
                        (depth if getattr(backend, "direct_grouped_fence", False) else 0)):
                    raise RuntimeError("serving B2 direct grouped fence proof differs")
                deferred = bool(getattr(candidate, "_defer_staged_q1_eval", False))
                write_only = bool(getattr(candidate, "_defer_staged_q1_writes", False))
                if deferred and write_only:
                    raise RuntimeError("mutually exclusive Q1 deferral modes")
                retain_roots = deferred or write_only
                inline = bool(getattr(candidate, "_b2_inline_metadata", False))
                expected = {
                    "grouped_write_async_evals": 0 if retain_roots else depth,
                    "staged_read_async_evals": 0 if deferred else depth,
                    "deferred_q1_write_roots": depth if retain_roots else 0,
                    "deferred_q1_read_roots": depth if retain_roots else 0,
                    "deferred_q1_final_evals": 1 if retain_roots else 0,
                    "deferred_q1_failure_flushes": 0,
                    "q1_metadata_dispatches": depth if inline and not stock_sdpa else 0,
                    "q1_gather_dispatches": depth if stock_sdpa else 0,
                    "stock_sdpa_graph_calls": depth if stock_sdpa else 0,
                    **{f"q1_stripe_dispatches_{stripes}": (
                        depth if not stock_sdpa and
                        getattr(candidate, "_b2_q1_stripes", 4) == stripes else 0)
                        for stripes in (8, 16, 32)},
                }
                if any(after[key] - physical_before[key] != value
                       for key, value in expected.items()):
                    raise RuntimeError("serving B2 combined submission proof differs")
                combined_counts = {key: after[key] - physical_before[key]
                                   for key in expected}
            for branch in branches:
                prepared.append(branch.prepare(1))
            for state in prepared:
                state.publish()
            published_layer_offsets = tuple(
                [layer.offset for layer in branch.layers] for branch in branches)
            for lane in lanes:
                lane.tokens.append(lane._pending_token)
                lane._pending_token = None
                lane.native_read_calls = backend.read_submissions
                lane.terminal_successes = backend.terminal_successes
            # Explicit candidate arm: preserve processor/sampler/RNG invocation
            # order, then materialize both lane graphs with one host eval.
            grouped_sampling = (bool(getattr(candidate, "_b2_grouped_sampler", False))
                                if serving else
                                os.environ.get("MLX2_PAGED_GROUPED_SAMPLER") == "1")
            samples = None
            if grouped_sampling:
                from .paged_native_continuation import mx

                samples = tuple(lane._stage_sample(logits[index])
                                for index, lane in enumerate(lanes))
                mx.eval(*(value for sample in samples for value in sample))
                candidate._grouped_sampler_evals = getattr(
                    candidate, "_grouped_sampler_evals", 0) + 1
            responses = []
            for index, (lane, metadata) in enumerate(zip(lanes, public_metadata)):
                if samples is None:
                    response = lane._next_with_reader(
                        metadata[0], metadata[1], logits_override=logits[index])
                else:
                    response = lane._response_from_sample(
                        metadata[0], metadata[1], *samples[index])
                response.execution_width = 2
                response.mtp_receipt.update({
                    "route": getattr(candidate, "serving_route",
                        "native_qwen3_paged_b2" if serving else "native_qwen3_paged_graph_research"),
                    "selected": serving,
                    "observed_used": serving,
                    "research_executed": not serving,
                    "qualified": False,
                    "packed_lanes": 2,
                    "grouped_sampler_eval": grouped_sampling,
                    "grouped_sampler_evals": getattr(
                        candidate, "_grouped_sampler_evals", 0),
                    "native_read_delta": depth,
                    "terminal_success_delta": depth,
                    "native_span_counts": [2] * depth,
                    "published_layer_offsets": published_layer_offsets[index],
                    "ordinary_forward_calls": 0,
                })
                if getattr(candidate, "owns_physical_dispatch_proof", False):
                    response.mtp_receipt["hybrid_graph_proof"] = dict(graph_receipt)
                if "native_host_profile" in graph_receipt:
                    response.mtp_receipt["native_host_profile"] = graph_receipt[
                        "native_host_profile"]
                if serving:
                    response.mtp_receipt.update({
                        "admission_profile": candidate._b2_profile_id,
                        "prefill_mode": ("packed_staged" if getattr(
                            candidate, "_b2_packed_prefill", False) else "serial_native"),
                        "price_usable": False,
                        "physical_dispatches": physical,
                        "combined_optimizations": {
                            "deferred_eval": bool(getattr(candidate,
                                                        "_defer_staged_q1_eval", False)),
                            "deferred_write_eval": bool(getattr(
                                candidate, "_defer_staged_q1_writes", False)),
                            "inline_metadata": bool(getattr(candidate,
                                                          "_b2_inline_metadata", False)),
                            "grouped_sampler": grouped_sampling,
                            "stock_sdpa": bool(getattr(candidate, "_b2_stock_sdpa", False)),
                        },
                        "combined_submission_delta": combined_counts,
                        "q1_simd_stripes": getattr(candidate, "_b2_q1_stripes", 4),
                        "vector_q1_rope_calls": getattr(
                            candidate, "_vector_q1_rope_calls", 0),
                        "direct_grouped_fence_reads": getattr(
                            backend, "direct_grouped_fence_reads", 0),
                    })
                    if getattr(candidate, "bootstrap_generation", 1) == 0:
                        from .packed_prefill_receipt import bootstrap_prefill_attribution
                        response.mtp_receipt.update(bootstrap_prefill_attribution(candidate))
                    if getattr(candidate, "owns_physical_dispatch_proof", False):
                        # The adapter proof describes the kernel actually used
                        # by this forward, including stock32 and B1 fallback.
                        response.mtp_receipt["q1_simd_stripes"] = graph_receipt["q1_simd_stripes"]
                    if getattr(candidate, "_b2_prefill_proof", None) is not None:
                        response.mtp_receipt["native_prefill_proof"] = dict(
                            candidate._b2_prefill_proof)
                responses.append(response)
        except BaseException:
            for state in prepared:
                state.rollback()
            for branch in branches:
                branch.rollback()
            raise
    for lane in lanes:
        try:
            lane.owner.reap_retired()
        except Exception:
            pass
    return tuple(responses)
