# SPDX-License-Identifier: MIT
"""Self-MTP request admission on a 36 GiB host (CPU only; no model, no Metal).

Measured on an M3 Pro (36 GiB physical, Metal advisory 28.08 GiB) serving
``Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-oQ4e-mtp`` (20 GiB
resident) after the host-scaled reserve of 3fa14e7.  Every request on the
self-MTP route retried admission (466 retries per arm) and 429'd on the
admission deadline, while the ordinary route on the same model and box served
normally.  The numbers below are the server's own ``/status`` report from
``qualification/junction-snapshots-20260919``.

The request gate charged the *top rung* of the degradation ladder -- the
k=``num_draft`` verify transient -- as a precondition for admitting a request
at all.  On this host that rung plus the reserve exceeds the whole Metal
headroom before a single context token is costed, so no request of any size
could ever pass.  The lane controller, which re-decides depth every decode
cycle, would have run the same lane at k=1.
"""

import pytest

from mlx2.adapters.qwen38_memory import Qwen38CacheBudget
from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController as C
from mlx2.serving import lane_admission_required_gib

GIB = float(1 << 30)

M3_HOST_GIB = 36.0
# ``execution_headroom()`` as the running server reported it in
# /status.headroom_bytes, i.e. min(host available, 28.08 GiB advisory minus
# this process' 19.74 GiB physical footprint).
M3_HEADROOM_MTP_GIB = 8950808880 / GIB  # 8.336 -- self-MTP arm
M3_HEADROOM_ORD_GIB = 8076296424 / GIB  # 7.522 -- ordinary arm

# /status.settings.cache_budget from the same two arms.
_GEOMETRY = dict(
    attention_layers=10,
    recurrent_layers=30,
    kv_heads=2,
    head_dim=256,
    recurrent_heads=32,
    recurrent_key_heads=16,
    recurrent_key_dim=128,
    recurrent_value_dim=128,
    conv_kernel=4,
)
MTP_BUDGET = Qwen38CacheBudget(mtp_layers=1, **_GEOMETRY)
ORD_BUDGET = Qwen38CacheBudget(mtp_layers=0, **_GEOMETRY)

NUM_DRAFT = 2
MAX_LANES = 4
# The smoke request that 429'd: 18 prompt tokens, max_tokens 16.
SMOKE_CONTEXT = 18 + 16


def controller_for(budget, host_gib=M3_HOST_GIB, draft=NUM_DRAFT):
    return C(
        host_memory_gib=host_gib,
        saturation_lane_cap=MAX_LANES,
        verification_row_cap=MAX_LANES * (draft + 1),
        cache_estimator=budget.project,
        transient_gib_per_lane=budget.transient_gib_per_lane,
    )


def test_36_gib_self_mtp_gate_refuses_every_request_at_full_depth():
    """The defect, term by term. This is the arithmetic that 429'd."""
    controller = controller_for(MTP_BUDGET)
    reserve = controller.hard_reserve_gib
    cache = MTP_BUDGET.project(SMOKE_CONTEXT) / GIB
    transient = MTP_BUDGET.transient_gib_per_lane * C.TRANSIENT_SCALE[NUM_DRAFT]

    assert reserve == pytest.approx(5.625)  # host-scaled, 3fa14e7
    assert transient == pytest.approx(3.1)  # qwen38 forward workspace
    assert cache == pytest.approx(0.1443, abs=1e-4)  # 34 tokens: negligible

    required = lane_admission_required_gib(
        controller,
        context_tokens=SMOKE_CONTEXT,
        draft_depth=NUM_DRAFT,
        cache_gib=0.0,
    )
    assert required == pytest.approx(reserve + cache + transient)
    assert required == pytest.approx(8.8693, abs=1e-3)
    assert required > M3_HEADROOM_MTP_GIB
    assert M3_HEADROOM_MTP_GIB - required == pytest.approx(-0.533, abs=1e-2)

    # Not a context problem: the two context-dependent terms are 0.14 GiB.
    # reserve + transient alone already overruns the whole headroom, so a
    # zero-token request is refused just as surely as a 64K one.
    assert reserve + transient > M3_HEADROOM_MTP_GIB
    for context in (0, 1, SMOKE_CONTEXT, 4096, 65536):
        assert (
            lane_admission_required_gib(
                controller,
                context_tokens=context,
                draft_depth=NUM_DRAFT,
                cache_gib=0.0,
            )
            > M3_HEADROOM_MTP_GIB
        )


def test_the_ordinary_route_escapes_because_it_pays_a_third_of_the_transient():
    """Same model, same box, same reserve -- and it serves."""
    controller = controller_for(ORD_BUDGET, draft=0)
    required = lane_admission_required_gib(
        controller, context_tokens=SMOKE_CONTEXT, draft_depth=0, cache_gib=0.0
    )
    # TRANSIENT_SCALE[0] is 1/3: 1.033 GiB rather than 3.1 GiB.
    assert required == pytest.approx(6.8007, abs=1e-3)
    assert required <= M3_HEADROOM_ORD_GIB
    assert M3_HEADROOM_ORD_GIB - required == pytest.approx(0.721, abs=1e-2)
    # The entire difference between the two routes is the verify transient.
    mtp = controller_for(MTP_BUDGET)
    gap = (
        lane_admission_required_gib(
            mtp, context_tokens=SMOKE_CONTEXT, draft_depth=NUM_DRAFT, cache_gib=0.0
        )
        - required
    )
    assert gap == pytest.approx(3.1 - 3.1 / 3.0, abs=2e-3)


def test_the_depth_floor_admits_and_the_controller_then_runs_the_lane():
    """The fix: admit on the rung the ladder can always fall back to."""
    controller = controller_for(MTP_BUDGET)
    floor = lane_admission_required_gib(
        controller, context_tokens=SMOKE_CONTEXT, draft_depth=0, cache_gib=0.0
    )
    assert floor == pytest.approx(6.8027, abs=1e-3)
    assert floor <= M3_HEADROOM_MTP_GIB

    # ...and admitting it is not a bluff. At the first cycle boundary the
    # controller sees the same live headroom and seats the lane at k=1.
    decision = controller.decide(
        [SMOKE_CONTEXT], M3_HEADROOM_MTP_GIB, max_draft=NUM_DRAFT
    )
    assert decision.modes == ("self_mtp",)
    assert decision.draft_depths == (1,)
    assert decision.stage == "lower_k"
    assert decision.estimated_gib <= decision.usable_gib
    # The plain rung fits with room to spare, so the lane can never wedge.
    assert controller.lane_gib(SMOKE_CONTEXT, 0) < decision.usable_gib


def test_full_depth_still_fits_on_the_128_gib_calibration_host():
    """No behaviour change where the top rung already fit.

    The fallback is reached only on the branch that previously returned
    "refuse", so any host that admitted at full depth is untouched.
    """
    controller = controller_for(MTP_BUDGET, host_gib=128.0)
    assert controller.hard_reserve_gib == 20.0  # bit for bit, as 3fa14e7
    # Flash-Next operating point: ~96 GiB advisory, 72.5 GiB resident.
    headroom = 96.0 - 72.5
    required = lane_admission_required_gib(
        controller, context_tokens=1024, draft_depth=NUM_DRAFT, cache_gib=0.0
    )
    assert required <= headroom
    floor = lane_admission_required_gib(
        controller, context_tokens=1024, draft_depth=0, cache_gib=0.0
    )
    assert floor < required  # the fallback is strictly weaker, never stronger
    # The gate's first test passes, so the fallback branch never executes.
    assert required == pytest.approx(
        controller.hard_reserve_gib + controller.lane_gib(1024, NUM_DRAFT), abs=0
    )


def test_required_helper_is_exactly_the_old_expression():
    """``lane_admission_required_gib`` is a pure extraction."""
    from mlx2.serving import prompt_lookup_verification_gib

    controller = controller_for(MTP_BUDGET)
    for context in (0, 34, 4096, 65536):
        for depth in (0, 1, 2):
            for cache in (0.0, 0.75):
                expected = controller.hard_reserve_gib + controller.lane_gib(
                    context, depth, cache_gib=cache
                )
                assert lane_admission_required_gib(
                    controller,
                    context_tokens=context,
                    draft_depth=depth,
                    cache_gib=cache,
                ) == pytest.approx(expected, abs=0)
                assert lane_admission_required_gib(
                    controller,
                    context_tokens=context,
                    draft_depth=depth,
                    cache_gib=cache,
                    prompt_lookup_num_draft=3,
                ) == pytest.approx(
                    expected + prompt_lookup_verification_gib(controller, 3), abs=0
                )


def _gate(controller, headroom_gib, depth, **kwargs):
    """Drive the real serving gate with a fixed, measured headroom."""
    from mlx2.serving import admit_lane_headroom

    calls = []

    def headroom():
        calls.append(1)
        return int(headroom_gib * GIB)

    return admit_lane_headroom(
        controller,
        context_tokens=SMOKE_CONTEXT,
        draft_depth=depth,
        cache_gib=0.0,
        headroom=headroom,
        reclaim=lambda: None,
        evict=lambda: False,
        evictable=lambda: 0,
        **kwargs,
    )


def test_gate_admits_the_self_mtp_request_the_36_gib_host_used_to_429():
    controller = controller_for(MTP_BUDGET)
    # Pre-fix the gate was a single ``ensure_admission_headroom`` at full
    # depth; that test is still the first thing tried and it still fails.
    from mlx2.serving import ensure_admission_headroom

    full = lane_admission_required_gib(
        controller, context_tokens=SMOKE_CONTEXT, draft_depth=NUM_DRAFT, cache_gib=0.0
    )
    assert not ensure_admission_headroom(
        full * GIB,
        headroom=lambda: int(M3_HEADROOM_MTP_GIB * GIB),
        reclaim=lambda: None,
        evict=lambda: False,
        evictable=lambda: 0,
    )
    # With the depth floor the request is admitted instead of deferred.
    (admitted, depth_floor, required) = _gate(
        controller, M3_HEADROOM_MTP_GIB, NUM_DRAFT
    )
    assert (admitted, depth_floor) == (True, True)
    # ``required`` reports the rung that was actually paid for, which the
    # interior-checkpoint budget downstream subtracts from live headroom.
    assert required == pytest.approx(
        lane_admission_required_gib(
            controller, context_tokens=SMOKE_CONTEXT, draft_depth=0, cache_gib=0.0
        ),
        abs=0,
    )


def test_gate_is_unchanged_when_the_top_rung_fits():
    controller = controller_for(MTP_BUDGET, host_gib=128.0)
    assert _gate(controller, 23.5, NUM_DRAFT)[:2] == (True, False)
    # An ordinary (depth 0) request has no floor to fall back to.
    ordinary = controller_for(ORD_BUDGET, host_gib=128.0, draft=0)
    assert _gate(ordinary, 23.5, 0)[:2] == (True, False)


def test_gate_still_fails_closed_when_even_the_plain_rung_does_not_fit():
    controller = controller_for(MTP_BUDGET)
    plain = lane_admission_required_gib(
        controller, context_tokens=SMOKE_CONTEXT, draft_depth=0, cache_gib=0.0
    )
    assert _gate(controller, plain - 0.01, NUM_DRAFT)[:2] == (False, False)
    assert _gate(controller, 0.0, NUM_DRAFT)[:2] == (False, False)


def test_prompt_lookup_verification_is_charged_on_both_rungs():
    controller = controller_for(MTP_BUDGET)
    from mlx2.serving import prompt_lookup_verification_gib

    floor = lane_admission_required_gib(
        controller, context_tokens=SMOKE_CONTEXT, draft_depth=0, cache_gib=0.0
    ) + prompt_lookup_verification_gib(controller, 3)
    assert _gate(
        controller, floor - 0.01, NUM_DRAFT, prompt_lookup_num_draft=3
    )[:2] == (False, False)
    assert _gate(
        controller, floor + 0.01, NUM_DRAFT, prompt_lookup_num_draft=3
    )[:2] == (True, True)
