"""Warm latency must follow uncached work; optional speculation retains APC."""
from collections import deque
from types import SimpleNamespace as NS

from mlx2.runtime.generate import BatchGenerator
from mlx2.runtime.memory_policy import (
    SelfMTPLaneAdmissionController,
    _make_self_mtp_admission_callback,
)
from mlx2.serving import admit_lane_headroom, lane_admission_required_gib


def test_warm_coarrival_does_not_wait_behind_a_cold_same_context():
    batch = BatchGenerator.__new__(BatchGenerator)
    batch.self_mtp = {"segment_aware_live_tip": True, "segment_aware_cohort_size": 4}
    batch.completion_batch_size = 4
    batch._generation_batch = NS(mtp_cycle_state=lambda: (), has_deferred_lanes=False,
                                 uids=(), _paused={}, _plain_ready=[])
    class Plain:
        uids = ()
        def __len__(self):
            return 0
    batch._plain_fallback_batch = Plain()
    batch._mtp_configs = {}
    batch._mtp_initial_prefill_tokens = {1: 8, 2: 32768}
    batch._unprocessed_sequences = deque([
        (1, [[9]], 128, [], range(32767)),
        (2, [range(32768)], 128, [], []),
    ])
    assert not batch._mtp_coarrival_hold(0)
    # Equal *cold* coarrivals still form a single cohort after partial prefill.
    batch._mtp_initial_prefill_tokens[1] = 32768
    assert batch._mtp_coarrival_hold(0)
    batch._mtp_initial_prefill_tokens[1] = 8
    batch._note_coarrival_window(2)
    batch._unprocessed_sequences.popleft()
    batch._generation_batch.uids = (1,)
    batch._has_active_decode = lambda: True
    assert batch._coarrival_exempt_rows(1) == 0


def test_cycle_admission_preserves_checkpoints_when_lower_depth_fits():
    controller = SelfMTPLaneAdmissionController(cache_estimator=lambda _: 0)
    evictions = []
    policy = _make_self_mtp_admission_callback(
        controller, free_memory=lambda: 21.5,
        reclaim_memory=lambda: None,
        evict_unused_cache=lambda: evictions.append(True) or True,
    )
    result = policy([(1, 32768, 2, True, 0.0)])
    assert result == {1: 1}
    assert evictions == []


def test_arrival_uses_plain_floor_without_eviction_for_optional_depth():
    controller = SelfMTPLaneAdmissionController(cache_estimator=lambda _: 0)
    options = dict(context_tokens=32768, cache_gib=0.0)
    full = lane_admission_required_gib(controller, draft_depth=2, **options)
    floor = lane_admission_required_gib(controller, draft_depth=0, **options)
    evictions = []
    admitted, used_floor, required = admit_lane_headroom(
        controller, draft_depth=2, **options,
        headroom=lambda: int((full + floor) / 2 * (1 << 30)),
        reclaim=lambda: None,
        evict=lambda: evictions.append(True) or True,
        evictable=lambda: 1 << 30,
    )
    assert admitted and used_floor and required == floor
    assert evictions == []


def test_planned_ordinary_cohort_reserves_serial_preparation_peak():
    controller = SelfMTPLaneAdmissionController(cache_estimator=lambda _: 0)
    evictions = []
    policy = _make_self_mtp_admission_callback(
        controller, free_memory=lambda: 24.0,
        evict_unused_cache=lambda: evictions.append(True) or True,
    )
    rows = [(i, 32768, 2, True, 0.0) for i in range(4)]
    assert policy.ordinary_preparation(rows) == {i: "plain" for i in range(4)}
    assert evictions == []
    # At two rows, ordinary transients are smaller than one k2 preparation.
    # The extra reservation must still refuse a plan that cannot prepare.
    low = _make_self_mtp_admission_callback(controller, free_memory=lambda: 21.3)
    assert "queue" in low.ordinary_preparation(rows[:2]).values()


def test_idle_handoff_does_not_spend_unpriced_ordinary_merge_copies():
    from mlx2.runtime.adaptive_policy import MTPOrdinaryHandoffPolicy
    from mlx2.runtime.generate import MTPGenerationBatch

    batch = MTPGenerationBatch.__new__(MTPGenerationBatch)
    batch.ordinary_handoff_policy = MTPOrdinaryHandoffPolicy(True, 3)
    batch.adaptive_depth_policy = None
    batch.state = NS(lanes=[])
    batch._paused = {
        i: NS(detached=NS(lane=NS(_batch_cohort=None),
                         caches=NS(target=[NS(nbytes=1 << 30)])))
        for i in range(4)
    }
    batch.mtp_cycle_state = lambda: [(i, 32768, 2, True, 1.0, 0.0) for i in range(4)]
    batch.mtp_admission = _make_self_mtp_admission_callback(
        SelfMTPLaneAdmissionController(cache_estimator=lambda _: 0),
        free_memory=lambda: 24.0,
    )
    # The plain transients fit, but the extra 4 GiB merge copy does not.
    assert not batch._handoff_prepared_idle_cohort()
