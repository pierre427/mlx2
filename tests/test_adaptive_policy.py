from collections import defaultdict

import pytest

from mlx2.runtime.adaptive_policy import (
    AdaptiveMTPDepthPolicy,
    CohortAdaptiveMTPDepth,
    DecodeTimeFairness,
)


def test_decode_time_fairness_caps_by_wall_time_and_repays_debt():
    policy = DecodeTimeFairness(
        enabled=True, fair_share=0.5, stall_target_ms=100, fallback_cap=256,
        floor=64, grid=64,
    )
    assert policy.cap(2048, contended=True) == 256
    policy.observe_prefill(256, 0.2, contended=True)
    assert policy.debt_seconds == pytest.approx(0.1)
    assert not policy.may_prefill(contended=True)
    policy.observe_decode(0.04)
    assert not policy.may_prefill(contended=True)
    policy.observe_decode(0.06)
    assert policy.may_prefill(contended=True)
    assert policy.cap(2048, contended=True) == 128
    assert policy.counters == {
        "prefill_chunks": 1,
        "debt_deferrals": 2,
        "debt_repayments": 2,
        "cap_clamps": 2,
    }


def test_decode_time_fairness_resets_debt_when_contention_ends():
    policy = DecodeTimeFairness(enabled=True)
    policy.observe_prefill(512, 0.1, contended=True)
    assert policy.debt_seconds > 0
    assert policy.may_prefill(contended=False)
    assert policy.debt_seconds == 0


def test_cohort_depth_parks_and_reenters_as_one_decision():
    policy = CohortAdaptiveMTPDepth(
        max_depth=2, ewma_alpha=1.0, loss_rounds=1, gain_rounds=2,
        park_rounds=2,
    )
    assert policy.select() == 2
    policy.observe(4, 0)
    assert policy.select() == 1
    policy.observe(2, 0)
    assert policy.select() == 0
    assert policy.select() == 0
    assert policy.select() == 1
    assert policy.counters["parks"] == 1
    assert policy.counters["reentries"] == 1
    policy.observe(2, 2)
    policy.observe(2, 2)
    assert policy.select() == 2


def test_cohort_depth_respects_admission_cap_without_splitting_lanes():
    policy = CohortAdaptiveMTPDepth(max_depth=3)
    assert policy.select(admitted_cap=1) == 1
    assert policy.select(admitted_cap=0) == 0
    with pytest.raises(ValueError):
        policy.observe(1, 2)


def test_serving_adaptive_depth_policy_is_strict_and_default_off():
    disabled = AdaptiveMTPDepthPolicy.from_value(None)
    assert disabled.as_dict() == {
        "enabled": False,
        "ewma_alpha": 0.25,
        "shrink_gate": 0.35,
        "grow_gate": 0.8,
        "loss_rounds": 3,
        "gain_rounds": 3,
        "park_rounds": 4,
    }
    selected = AdaptiveMTPDepthPolicy.from_value(
        {"enabled": True, "ewma_alpha": 0.5, "loss_rounds": 2}
    )
    assert selected.enabled
    assert selected.controller_kwargs()["ewma_alpha"] == 0.5
    assert selected.controller_kwargs()["loss_rounds"] == 2
    with pytest.raises(ValueError, match="unknown adaptive MTP settings"):
        AdaptiveMTPDepthPolicy.from_value({"enabled": True, "mystery": 1})
    with pytest.raises(ValueError, match="enabled must be boolean"):
        AdaptiveMTPDepthPolicy.from_value({"enabled": 1})
    with pytest.raises(ValueError, match="loss_rounds must be a positive integer"):
        AdaptiveMTPDepthPolicy.from_value({"loss_rounds": True})
    with pytest.raises(ValueError, match="gates"):
        AdaptiveMTPDepthPolicy.from_value(
            {"shrink_gate": 0.9, "grow_gate": 0.8}
        )


def test_batch_scheduler_turns_defer_prefill_until_decode_repays_debt(monkeypatch):
    from mlx2.runtime import generate

    class GenerationBatch:
        def __len__(self):
            return 1

        def next(self):
            return []

    class PromptBatch:
        uids = [7]

        def __init__(self):
            self.calls = []

        def __len__(self):
            return 0

        def prompt(self, prompts):
            self.calls.append(prompts)

    class Scheduler(generate.BatchGenerator):
        def __del__(self):
            pass

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.self_mtp = None
    scheduler._generation_batch = GenerationBatch()
    scheduler._prompt_batch = PromptBatch()
    scheduler._unprocessed_sequences = []
    scheduler._currently_processing = [
        [[list(range(100))], 0, 100, None, None, 0.0]
    ]
    scheduler.completion_batch_size = 4
    scheduler.prefill_batch_size = 1
    scheduler.prefill_step_size = 64
    scheduler.decode_priority_cadence = 1
    scheduler.adaptive_prefill = False
    scheduler.adaptive_prefill_target_itl_ms = 1500.0
    scheduler.adaptive_prefill_max_defer_ms = 2000.0
    scheduler.adaptive_prefill_slices = (64,)
    scheduler._last_decode_completed_s = None
    scheduler._last_decode_interval_ms = None
    scheduler._last_decode_duration_ms = None
    scheduler._prefill_ms_per_token_ewma = None
    scheduler.decode_time_fairness = DecodeTimeFairness(
        enabled=True, fair_share=0.5
    )
    scheduler.decode_time_fairness.debt_seconds = 0.5
    scheduler.scheduler_stats = defaultdict(int)
    scheduler._gen_tokens_counter = 0
    scheduler._steps_counter = 0
    scheduler._prompt_tokens_counter = 0
    scheduler._prompt_time_counter = 0.0
    scheduler._promote_ready_prompts = lambda: []
    scheduler.state_budget = None

    clock = iter((0.0, 0.1, 0.2, 1.0, 1.5, 1.6, 1.7, 1.9))
    monkeypatch.setattr(generate.time, "perf_counter", lambda: next(clock))

    scheduler._next()
    assert scheduler._prompt_batch.calls == []
    assert scheduler.decode_time_fairness.debt_seconds == pytest.approx(0.4)
    scheduler._next()
    assert scheduler._prompt_batch.calls == [[list(range(64))]]
    assert scheduler.decode_time_fairness.debt_seconds == pytest.approx(0.1)
    assert scheduler.scheduler_stats["decode_fairness_debt_deferrals"] == 1
    assert scheduler.scheduler_stats["decode_fairness_debt_repayments"] == 2
