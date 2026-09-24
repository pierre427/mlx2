import threading
from collections import Counter
from types import SimpleNamespace as NS

import mlx.core as mx
import pytest

from mlx2 import serving
from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController
from mlx2.serving import (
    apc_interior_checkpoint_policy,
    budget_interior_checkpoint_positions,
    ensure_admission_headroom,
    prompt_lookup_verification_gib,
    warm_cache_copy_gib,
)


def test_apc_interior_checkpoint_policy_is_default_off_and_strict():
    assert apc_interior_checkpoint_policy(None) == {"count": 0, "min_stride": 1}
    assert apc_interior_checkpoint_policy({"count": 3, "min_stride": 4}) == {
        "count": 3,
        "min_stride": 4,
    }
    for invalid in (
        False,
        [],
        {"count": True},
        {"count": 33},
        {"min_stride": 0},
        {"unknown": 1},
    ):
        try:
            apc_interior_checkpoint_policy(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid APC interior policy: {invalid!r}")


def test_checkpoint_budget_degrades_to_deepest_positions_without_blocking_base():
    candidates = (8, 16, 32)

    def project(position):
        return position * 10

    assert budget_interior_checkpoint_positions(
        candidates, available_bytes=560, cache_projection=project
    ) == (candidates, 560)
    assert budget_interior_checkpoint_positions(
        candidates, available_bytes=400, cache_projection=project
    ) == ((32,), 320)
    assert budget_interior_checkpoint_positions(
        candidates, available_bytes=319, cache_projection=project
    ) == ((), 0)
    assert budget_interior_checkpoint_positions(
        candidates, available_bytes=10_000, cache_projection=None
    ) == ((), 0)


def test_reclamation_is_observed_before_destroying_warm_cache():
    state = {"free": 24, "evictions": 0}
    def reclaim(): state["free"] = 32
    def evict(): state["evictions"] += 1; return False
    assert ensure_admission_headroom(27, headroom=lambda: state["free"],
                                      reclaim=reclaim, evict=evict)
    assert state["evictions"] == 0


def test_rate_limited_reclaim_is_bypassed_before_first_admission_eviction(
    monkeypatch,
):
    engine = serving.ServingEngine.__new__(serving.ServingEngine)
    engine.counts = Counter()
    engine._memory_reclaim_lock = threading.Lock()
    engine._memory_reclaim_last = 0.0
    state = {"free": 10, "cached": 2, "evictions": 0}
    monkeypatch.setattr(mx, "synchronize", lambda: None)

    def clear_cache():
        state["free"] += state["cached"]
        state["cached"] = 0

    monkeypatch.setattr(mx, "clear_cache", clear_cache)

    def reclaim():
        return engine._clear_allocator_cache_before_reject(synchronize=True)

    def evict():
        state["evictions"] += 1
        state["cached"] += 1
        engine._permit_allocator_reclaim_after_eviction()
        return True

    assert ensure_admission_headroom(
        11, headroom=lambda: state["free"], reclaim=reclaim, evict=evict
    )
    state["cached"] = 8
    assert ensure_admission_headroom(
        18, headroom=lambda: state["free"], reclaim=reclaim, evict=evict
    )
    assert state["evictions"] == 0
    assert engine.counts["memory_cache_reclaims_before_reject"] == 2


def test_each_pressure_eviction_is_reclaimed_before_next_measurement():
    state = {"free": 20, "pending": 0, "entries": 3}
    def reclaim():
        state["free"] += state["pending"]
        state["pending"] = 0
    def evict():
        if not state["entries"]: return False
        state["entries"] -= 1
        state["pending"] += 4
        return True
    assert ensure_admission_headroom(27, headroom=lambda: state["free"],
                                      reclaim=reclaim, evict=evict)
    assert state["entries"] == 1
    assert not ensure_admission_headroom(40, headroom=lambda: state["free"],
                                          reclaim=reclaim, evict=evict)


def test_warm_copy_counts_target_draft_and_future_extent_without_zero_credit():
    hit = NS(cache=[NS(nbytes=3 * 2**30)], cached_tokens=131000,
             remaining_tokens=[7], sidecar=NS(nbytes=2**30))
    actual = warm_cache_copy_gib(hit, context_tokens=131064, prefill_step=2048, mtp=True)
    assert 4.0 < actual < 4.01
    hit.sidecar = None
    assert warm_cache_copy_gib(hit, context_tokens=131064, prefill_step=2048, mtp=True) == 0
    hit.remaining_tokens = list(range(2049))
    assert warm_cache_copy_gib(hit, context_tokens=134000, prefill_step=2048, mtp=False) == 0


def test_unsatisfiable_request_does_not_purge_warm_checkpoints():
    """If eviction cannot possibly reach the requirement the request must wait
    for active lanes instead of destroying every unleased checkpoint on each
    retry."""
    evictions = []

    def evict():
        evictions.append(1)
        return True

    admitted = ensure_admission_headroom(
        100, headroom=lambda: 10, reclaim=lambda: None, evict=evict,
        evictable=lambda: 20,  # 10 + 20 * 1.25 < 100: eviction cannot help
    )
    assert admitted is False
    assert evictions == []
    # When eviction can plausibly satisfy the requirement it still proceeds.
    state = {"free": 10}

    def evict_ok():
        state["free"] += 50
        return True

    assert ensure_admission_headroom(
        100, headroom=lambda: state["free"], reclaim=lambda: None,
        evict=evict_ok, evictable=lambda: 100,
    )


def test_prompt_lookup_verification_width_requires_more_headroom_than_plain_decode():
    controller = SelfMTPLaneAdmissionController()
    ordinary = controller.hard_reserve_gib + controller.lane_gib(4096, 0)
    prompt_lookup = ordinary + prompt_lookup_verification_gib(controller, 8)
    assert prompt_lookup > ordinary
    headroom = (ordinary + prompt_lookup) / 2
    assert ensure_admission_headroom(
        ordinary, headroom=lambda: headroom, reclaim=lambda: None,
        evict=lambda: False,
    )
    assert not ensure_admission_headroom(
        prompt_lookup, headroom=lambda: headroom, reclaim=lambda: None,
        evict=lambda: False,
    )


def test_warm_copy_charges_fixed_recurrent_state_once():
    """Qwen3.8-27B on a 36 GiB M3 Pro, series 2026-09-24.

    Every warm follow-up (the same prompt again, n=2's second sample, a
    thinking round trip) waited out its 60 s deadline and answered 429 while
    the identical cold request was served.  The checkpoint's 48 GDN layers
    hold 0.29 GiB of fixed-size state (APC layer segments: 1,847,328,768
    bytes over 6 entries), and its 16 attention layers hold exactly the
    cached tokens.  Scaling the recurrent state by context/capacity charged
    it 2.5x for a 24-token answer; the controller then extrapolated the
    excess per token.
    """
    from mlx2.adapters.qwen38_memory import Qwen38CacheBudget
    from mlx2.serving import lane_admission_required_gib

    recurrent = 1_847_328_768 // 6 // 48
    kv_per_token = 2 * 4 * 256 * 2  # K and V, 4 heads x 256, bf16

    def hit(cached):
        rows = [NS(nbytes=recurrent) for _ in range(48)] + [
            NS(nbytes=kv_per_token * cached, keys=NS(shape=(1, 4, cached, 256)))
            for _ in range(16)
        ]
        return NS(cache=rows, cached_tokens=cached, remaining_tokens=[1], sidecar=None)

    fixed = 48 * recurrent / 2**30
    copy = warm_cache_copy_gib(hit(17), context_tokens=42, prefill_step=2048, mtp=False)
    assert copy == pytest.approx(fixed + 16 * kv_per_token * 42 / 2**30)

    budget = Qwen38CacheBudget(
        attention_layers=16, recurrent_layers=48, mtp_layers=0, kv_heads=4,
        head_dim=256, recurrent_heads=48, recurrent_key_heads=16,
        recurrent_key_dim=128, recurrent_value_dim=128, conv_kernel=4,
    )
    controller = SelfMTPLaneAdmissionController(
        host_memory_gib=36.0, advisory_gib=28.08, cache_estimator=budget.project,
        transient_gib_per_lane=budget.transient_gib_per_lane,
        saturation_lane_cap=4, verification_row_cap=4,
    )
    # 8.54 GiB: the smallest headroom the M3 reported during that run.
    headroom = 9_166_750_408 / 2**30
    for cached, context in ((17, 42), (19, 36), (15, 79), (58, 2190)):
        warm = lane_admission_required_gib(
            controller, context_tokens=context, draft_depth=0,
            cache_gib=warm_cache_copy_gib(
                hit(cached), context_tokens=context, prefill_step=2048, mtp=False
            ),
        )
        cold = lane_admission_required_gib(
            controller, context_tokens=context, draft_depth=0, cache_gib=0.0
        )
        assert warm < headroom, (cached, context, warm)
        assert warm < cold + 0.1, (cached, context, warm, cold)
