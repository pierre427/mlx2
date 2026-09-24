from collections import defaultdict

import pytest

from mlx2.runtime.adaptive_policy import (
    AdaptiveMTPDepthPolicy,
    CohortAdaptiveMTPDepth,
    DecodeTimeFairness,
    MTPOrdinaryHandoffPolicy,
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
    assert policy.select(width=2) == 2
    policy.observe(4, 0, width=2)
    assert policy.select(width=2) == 1
    policy.observe(2, 0, width=2)
    assert policy.select(width=2) == 0
    assert policy.select(width=2) == 0
    assert policy.select(width=2) == 1
    assert policy.counters["parks"] == 1
    assert policy.counters["reentries"] == 1
    policy.observe(2, 2, width=2)
    policy.observe(2, 2, width=2)
    assert policy.select(width=2) == 2


def test_cohort_depth_respects_admission_cap_without_splitting_lanes():
    policy = CohortAdaptiveMTPDepth(max_depth=3)
    assert policy.select(admitted_cap=1) == 1
    assert policy.select(admitted_cap=0) == 0
    with pytest.raises(ValueError):
        policy.observe(1, 2)


def test_cohort_depth_rejects_cross_bucket_boundary_accounting():
    policy = CohortAdaptiveMTPDepth(max_depth=2)
    policy.select(width=1)
    with pytest.raises(ValueError, match="cohort width must match selection width"):
        policy.observe(2, 1, width=8)


def test_width_one_is_qualified_depth_anchor_after_concurrent_decrease():
    policy = CohortAdaptiveMTPDepth(
        max_depth=2,
        ewma_alpha=1.0,
        loss_rounds=1,
    )
    assert policy.select(width=8) == 2
    policy.observe(16, 0, width=8)
    assert policy.current_depth == 1
    assert policy.counters["depth_decreases_concurrent"] == 1

    assert policy.select(width=1) == 2
    assert policy.counters["depth_recoveries_alone"] == 1
    for _ in range(8):
        policy.observe(2, 0, width=1)
        assert policy.select(width=1) == 2
    assert policy.diagnostics()["buckets"]["1"]["chosen_depth"] == 2


def test_parked_probe_waits_for_a_nonzero_admission_cap():
    policy = CohortAdaptiveMTPDepth(
        max_depth=1,
        ewma_alpha=1.0,
        loss_rounds=1,
        park_rounds=1,
    )
    assert policy.select(width=2) == 1
    policy.observe(1, 0, width=2)
    assert policy.select(admitted_cap=0, width=2) == 0
    assert policy.select(admitted_cap=0, width=2) == 0
    assert policy.counters["reentries"] == 0
    assert policy.counters["probes"] == 0
    assert policy.select(admitted_cap=1, width=2) == 1
    policy.observe(1, 1, width=2)
    assert policy.counters["reentries"] == 1
    assert policy.counters["probes"] == 1
    assert policy.counters["depth_recoveries_alone"] == 0


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
    with pytest.raises(
        ValueError, match="min_samples_per_depth must be a positive integer"
    ):
        AdaptiveMTPDepthPolicy.from_value({"min_samples_per_depth": 0})
    with pytest.raises(ValueError, match="gates"):
        AdaptiveMTPDepthPolicy.from_value(
            {"shrink_gate": 0.9, "grow_gate": 0.8}
        )


def test_ordinary_handoff_policy_requires_explicit_enable_and_static_threshold():
    assert not MTPOrdinaryHandoffPolicy.from_value(None).enabled
    with pytest.raises(ValueError, match="enabled.*true"):
        MTPOrdinaryHandoffPolicy.from_value({"max_mtp_width": 8})
    with pytest.raises(ValueError, match="requires an object"):
        MTPOrdinaryHandoffPolicy.from_value(True)
    with pytest.raises(ValueError, match="unknown MTP ordinary handoff"):
        MTPOrdinaryHandoffPolicy.from_value(
            {"enabled": True, "max_mtp_width": 8, "mystery": 1}
        )

    policy = MTPOrdinaryHandoffPolicy.from_value(
        {"enabled": True, "max_mtp_width": 8}
    )
    adaptive = CohortAdaptiveMTPDepth(max_depth=2)
    assert policy.decision(width=8, width_locked=False, adaptive=adaptive) is None
    assert policy.decision(width=8, width_locked=True, adaptive=adaptive) is None
    assert (
        policy.decision(width=9, width_locked=False, adaptive=adaptive)["reason"]
        == "static_width_threshold"
    )
    assert (
        policy.decision(width=9, width_locked=True, adaptive=adaptive)["reason"]
        == "segmented_width_lock"
    )


def test_cost_aware_mtp_depth_converges_by_width_and_bounds_probes():
    policy = CohortAdaptiveMTPDepth(
        max_depth=2,
        goodput_alpha=1.0,
        goodput_hysteresis=0.05,
        stale_rounds=1_000,
    )
    synthetic_goodput = {
        "1": {0: 30.31, 1: 39.0, 2: 45.45},
        "5-8": {0: 107.35, 1: 101.0, 2: 98.72},
        "9-16": {0: 150.85, 1: 120.0, 2: 102.06},
    }

    def run(width, rounds):
        selected = []
        for _ in range(rounds):
            depth = policy.select(width=width)
            selected.append(depth)
            committed = width * (depth + 1)
            bucket = policy.width_bucket(width)
            policy.observe(
                proposed=width * depth,
                accepted=int(width * depth * 0.75),
                width=width,
                committed=committed,
                elapsed_seconds=committed / synthetic_goodput[bucket][depth],
            )
        return selected

    assert set(run(1, 8)) == {2}
    concurrent_8 = run(8, 56)
    concurrent_16 = run(16, 56)
    assert concurrent_8.count(0) >= policy.min_samples_per_depth
    assert concurrent_16.count(0) >= policy.min_samples_per_depth
    assert concurrent_8[-1] == 0
    assert concurrent_16[-1] == 0
    assert policy.select(width=1) == 2

    diagnostics = policy.diagnostics()["buckets"]
    assert diagnostics["1"]["chosen_depth"] == 2
    assert diagnostics["1"]["probes"] == 0
    assert diagnostics["5-8"]["chosen_depth"] == 0
    assert diagnostics["9-16"]["chosen_depth"] == 0
    assert set(diagnostics["5-8"]["goodput_tokens_per_second"]) == {"0", "2"}
    assert diagnostics["5-8"]["goodput_tokens_per_second"]["0"] == pytest.approx(
        107.35
    )
    assert diagnostics["9-16"]["goodput_tokens_per_second"]["0"] == pytest.approx(
        150.85
    )
    assert diagnostics["5-8"]["probe_fraction"] <= 1 / policy.probe_interval
    assert diagnostics["9-16"]["probe_fraction"] <= 1 / policy.probe_interval
    assert policy.counters["depth_decreases_concurrent"] >= 2
    assert policy.counters["depth_recoveries_alone"] >= 1


def test_cost_aware_mtp_depth_requires_hysteresis_before_switching():
    policy = CohortAdaptiveMTPDepth(
        max_depth=1,
        goodput_alpha=1.0,
        goodput_hysteresis=0.05,
        min_samples_per_depth=1,
        goodput_window=1,
        probe_interval=2,
        stale_rounds=2,
    )

    assert policy.select(width=8) == 1
    policy.observe(8, 6, width=8, committed=8, elapsed_seconds=0.08)
    assert policy.select(width=8) == 0
    policy.observe(0, 0, width=8, committed=8, elapsed_seconds=8 / 104)
    assert policy.select(width=8) == 1
    policy.observe(8, 6, width=8, committed=8, elapsed_seconds=0.08)
    assert policy.select(width=8) == 0
    policy.observe(0, 0, width=8, committed=8, elapsed_seconds=8 / 106)
    assert policy.select(width=8) == 0


def test_cold_depth_needs_minimum_probe_samples_before_reselection():
    policy = CohortAdaptiveMTPDepth(
        max_depth=1,
        goodput_alpha=1.0,
        min_samples_per_depth=3,
        probe_interval=2,
        stale_rounds=1_000,
    )
    for boundary in range(1, 7):
        depth = policy.select(width=8)
        rate = 120.0 if depth == 0 else 80.0
        policy.observe(
            proposed=8 * depth,
            accepted=8 * depth,
            width=8,
            committed=8,
            elapsed_seconds=8 / rate,
        )
        if boundary < 6:
            assert policy.diagnostics()["buckets"]["5-8"]["chosen_depth"] == 1
    bucket = policy.diagnostics()["buckets"]["5-8"]
    assert bucket["samples"]["0"] == 3
    assert bucket["estimate_updates"]["0"] == 1
    assert bucket["chosen_depth"] == 0


def test_goodput_publishes_complete_windows_not_terminal_partial_round():
    policy = CohortAdaptiveMTPDepth(
        max_depth=2,
        goodput_alpha=1.0,
        min_samples_per_depth=3,
        goodput_window=8,
    )
    for _ in range(3):
        assert policy.select(width=1) == 2
        policy.observe(2, 2, width=1, committed=4, elapsed_seconds=0.1)
    bucket = policy.diagnostics()["buckets"]["1"]
    assert bucket["goodput_tokens_per_second"]["2"] == pytest.approx(40.0)

    # A request-tail round is real work, but an incomplete one-round window
    # must not replace the stable estimate operators use for depth selection.
    assert policy.select(width=1) == 2
    policy.observe(2, 2, width=1, committed=1, elapsed_seconds=0.1)
    bucket = policy.diagnostics()["buckets"]["1"]
    assert bucket["goodput_tokens_per_second"]["2"] == pytest.approx(40.0)
    assert bucket["pending_samples"]["2"] == 1


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


def test_default_adaptive_slices_reach_the_fused_attention_width():
    """Contended prefill may use >=1024-row slices when the ITL budget allows
    (below 1024 rows head_dim-256 SDPA leaves the fused kernel)."""
    import inspect

    from mlx2.runtime.generate import BatchGenerator

    default = inspect.signature(BatchGenerator.__init__).parameters["adaptive_prefill_slices"].default
    assert max(default) >= 1024 and 2048 in default
