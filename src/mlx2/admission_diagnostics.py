"""Required-versus-available breakdown of one lane admission, for refusals.

Lane admission compares one number (``lane_admission_required_gib``) with one
measured reading (``execution_headroom`` less unmaterialized grants), so an
HTTP 429 that says only "host memory admission did not recover before
deadline" hides which term refused the request.  This module re-derives the
same requirement term by term from the same controller, names the headroom
reading it was compared against, and identifies the blocking term.  It never
decides admission: the scheduler calls it only after a request was deferred.
"""

from __future__ import annotations

import math

GIB = float(1 << 30)


def lane_admission_breakdown(
    controller,
    *,
    context_tokens,
    draft_depth,
    cache_gib,
    prefill_gib=0.0,
    prompt_lookup_num_draft=None,
    tested_required_gib=None,
    available_bytes,
    granted_bytes=0,
    headroom_terms=None,
):
    """Every term of one lane's admission requirement and of the headroom.

    The terms sum to ``lane_admission_required_gib`` for the same arguments.
    ``tested_required_gib`` is the requirement the scheduler actually tested
    last (a depth-floor rung, a capped prompt-lookup span or an elastic
    reserve shift can make it differ); any difference is reported as the
    ``other`` term, so the terms always sum to the tested requirement.

    ``blocking_term`` is the largest term whose removal alone would have
    admitted the request; ``sufficient_terms`` lists every such term.  When
    no single term covers the shortfall, ``blocking_term`` is the largest
    term and ``single_term_covers_shortfall`` is false.
    """
    from .runtime.memory_policy import SelfMTPLaneAdmissionController as C

    scale = C.TRANSIENT_SCALE[draft_depth]
    lane = controller.lane_gib(int(context_tokens), draft_depth, cache_gib=cache_gib)
    transient = controller.transient_gib_per_lane * scale
    context_transient = (controller.context_transient_gib(int(context_tokens))
                         if hasattr(controller, "context_transient_gib") else 0.0)
    estimator = getattr(controller, "cache_estimator", None)
    terms = {
        "service_reserve": float(controller.service_reserve_gib),
        "driver_allowance": float(controller.driver_allowance_gib),
        "stream_reserve": float(controller.stream_reserve_gib),
        "workspace_reserve": float(controller.workspace_reserve_gib),
        "lane_context_cache": lane - transient - context_transient,
        "verify_transient": transient,
        "context_transient": context_transient,
        "prefill_transient": float(prefill_gib),
    }
    if prompt_lookup_num_draft is not None:
        from .serving import prompt_lookup_verification_gib

        terms["prompt_lookup_verify"] = prompt_lookup_verification_gib(
            controller, prompt_lookup_num_draft
        )
    required = sum(terms.values())
    if tested_required_gib is not None:
        other = float(tested_required_gib) - required
        if abs(other) > 1e-6:
            terms["other"] = other
            required = float(tested_required_gib)
    available = max(0, int(available_bytes)) / GIB
    shortfall = required - available
    positive = sorted(
        ((name, value) for name, value in terms.items() if value > 0),
        key=lambda item: (-item[1], item[0]),
    )
    sufficient = [name for name, value in positive if value >= shortfall]
    blocking = sufficient[0] if sufficient else (positive[0][0] if positive else None)
    out = {
        "schema": "mlx2.lane-admission-breakdown.v1",
        "context_tokens": int(context_tokens),
        "draft_depth": int(draft_depth),
        "cache_estimate": (
            "adapter_projection" if estimator is not None
            else f"envelope_{C.CACHE_GIB_PER_1K_TOKENS:g}_gib_per_1k_tokens"
        ),
        "required_gib": round(required, 4),
        "available_gib": round(available, 4),
        "shortfall_gib": round(shortfall, 4),
        "terms_gib": {name: round(value, 4) for name, value in terms.items()},
        "blocking_term": blocking,
        "sufficient_terms": sufficient,
        "single_term_covers_shortfall": bool(sufficient) or shortfall <= 0,
        "granted_to_active_lanes_gib": round(max(0, int(granted_bytes)) / GIB, 4),
    }
    if headroom_terms:
        out["headroom"] = _headroom_side(headroom_terms)
    return out


def _headroom_side(terms):
    """Which reading bounds ``min(host term, advisory term)``, in GiB."""
    host = terms.get("host_term")
    advisory = terms.get("advisory_term")
    side = {
        key: round(int(value) / GIB, 4)
        for key, value in terms.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    if host is not None and advisory is not None:
        side["binding"] = "advisory_term" if advisory <= host else "host_term"
    elif advisory is not None:
        side["binding"] = "advisory_term"
    elif host is not None:
        side["binding"] = "host_term"
    return side


def summarize(breakdown):
    """One line for the 429 body and the server log."""
    if not breakdown:
        return ""
    terms = breakdown["terms_gib"]
    shown = ", ".join(
        f"{name} {value:.2f}" for name, value in sorted(
            terms.items(), key=lambda item: (-item[1], item[0])
        ) if value
    )
    line = (
        f"required {breakdown['required_gib']:.2f} GiB > available "
        f"{breakdown['available_gib']:.2f} GiB (short {breakdown['shortfall_gib']:.2f}) "
        f"at {breakdown['context_tokens']} context tokens; blocking term "
        f"{breakdown['blocking_term']} {terms.get(breakdown['blocking_term'], 0.0):.2f} GiB"
    )
    if not breakdown["single_term_covers_shortfall"]:
        line += " (no single term covers the shortfall)"
    line += f"; cache estimate {breakdown['cache_estimate']}; terms GiB: {shown}"
    headroom = breakdown.get("headroom") or {}
    if headroom.get("binding"):
        line += (
            f"; headroom bound by {headroom['binding']} "
            f"{headroom.get(headroom['binding'], float('nan')):.2f} GiB"
        )
    granted = breakdown.get("granted_to_active_lanes_gib") or 0.0
    if granted and math.isfinite(granted):
        line += f", less {granted:.2f} GiB granted to active lanes"
    return line
