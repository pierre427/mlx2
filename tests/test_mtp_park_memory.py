"""Measured MTP park with memory (oMLX #4112 port): CPU tests."""

import gc

import mlx.core as mx
import pytest

from mlx2.runtime.adaptive_policy import MTPOrdinaryHandoffPolicy
from mlx2.runtime.mtp_park_memory import (
    KILL_SWITCH_ENV,
    AdaptiveParkSettings,
    ParkMemory,
    park_memory_for,
    reset_park_memories,
)
from mlx2.runtime.sample_utils import LaneRNG


@pytest.fixture(autouse=True)
def _fresh_memories(monkeypatch):
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    reset_park_memories()
    yield
    reset_park_memories()


def _memory(stats=None, **overrides):
    settings = AdaptiveParkSettings.from_value({"enabled": True, **overrides})
    return ParkMemory(settings, static_max_width=3, stats=stats)


def _feed(memory, kind, width, tokens_per_ms, n):
    for _ in range(n):
        assert memory.observe(kind, width, int(tokens_per_ms * 1000), 1.0)


# -- configuration -----------------------------------------------------------

def test_adaptive_park_is_off_by_default_and_absent_from_receipts():
    policy = MTPOrdinaryHandoffPolicy.from_value(
        {"enabled": True, "max_mtp_width": 3}
    )
    assert policy.adaptive_park is None
    assert policy.as_dict() == {"enabled": True, "max_mtp_width": 3}
    assert AdaptiveParkSettings.from_value(None) is None
    assert AdaptiveParkSettings.from_value(False) is None
    assert AdaptiveParkSettings.from_value({"enabled": False}) is None


def test_adaptive_park_selected_appears_in_policy_receipt():
    policy = MTPOrdinaryHandoffPolicy.from_value(
        {"enabled": True, "max_mtp_width": 3, "adaptive_park": {"enabled": True}}
    )
    assert policy.adaptive_park == AdaptiveParkSettings(enabled=True)
    receipt = policy.as_dict()["adaptive_park"]
    assert receipt["clear_loss_ratio"] == 0.9
    assert receipt["clear_loss_decisions"] == 4
    assert receipt["hold_decisions"] == 32
    assert MTPOrdinaryHandoffPolicy.from_value(
        {"enabled": True, "max_mtp_width": 3, "adaptive_park": True}
    ).adaptive_park.enabled


@pytest.mark.parametrize(
    "value, message",
    [
        ({"enabled": True, "bogus": 1}, "unknown adaptive_park"),
        ({"enabled": True, "clear_loss_ratio": 1.1}, "hysteresis"),
        ({"enabled": True, "hold_ratio": 1.0}, "hysteresis"),
        ({"enabled": True, "ema_alpha": 0.0}, "ema_alpha"),
        ({"enabled": True, "hold_decisions": 0}, "hold_decisions"),
        ({"enabled": True, "cooldown_cohorts": 8, "max_cooldown_cohorts": 4}, "max_cooldown"),
        ({"enabled": False, "hold_decisions": 3}, "require enabled"),
        ("yes", "object or boolean"),
    ],
)
def test_adaptive_park_refuses_invalid_settings(value, message):
    with pytest.raises(ValueError, match=message):
        MTPOrdinaryHandoffPolicy.from_value(
            {"enabled": True, "max_mtp_width": 3, "adaptive_park": value}
        )


def test_adaptive_park_requires_an_enabled_handoff():
    with pytest.raises(ValueError, match="requires an enabled"):
        MTPOrdinaryHandoffPolicy(adaptive_park={"enabled": True})


# -- rate model --------------------------------------------------------------

def test_cold_start_uses_the_static_threshold_until_ordinary_is_measured():
    memory = _memory()
    assert memory.decide(2) is None
    cold = memory.decide(4)
    assert cold["reason"] == "static_width_threshold"
    assert cold["adaptive"] == "cold_start_no_ordinary_rate"
    _feed(memory, "ordinary", 4, 10.0, 4)
    # Ordinary known, MTP not: MTP runs and is measured (probe).
    assert memory.decide(4) is None


def test_clear_loss_parks_after_four_fresh_decisions_only():
    stats = {}
    memory = _memory(stats)
    _feed(memory, "ordinary", 4, 10.0, 4)
    _feed(memory, "mtp", 4, 8.5, 4)  # 0.85x ordinary
    assert memory.decide(4) is None  # decision 1
    # Boundaries without a fresh MTP sample are not decisions.
    assert memory.decide(4) is None
    assert memory.decide(4) is None
    for _ in range(2):
        _feed(memory, "mtp", 4, 8.5, 1)
        assert memory.decide(4) is None  # decisions 2, 3
    _feed(memory, "mtp", 4, 8.5, 1)
    verdict = memory.decide(4)  # decision 4
    assert verdict["reason"] == "measured_loss"
    assert verdict["verdict_width"] == 4
    assert verdict["ratio"] == pytest.approx(0.85)
    assert stats["mtp_adaptive_park_decisions"] == 4
    assert stats["mtp_adaptive_park_verdicts_set"] == 1
    assert stats["mtp_adaptive_park_measured_loss"] == 1


def test_a_sustained_small_loss_parks_after_sixteen_decisions():
    memory = _memory()
    _feed(memory, "ordinary", 4, 10.0, 4)
    _feed(memory, "mtp", 4, 9.5, 4)  # 0.95x: not a clear loss
    for _ in range(15):
        _feed(memory, "mtp", 4, 9.5, 1)
        assert memory.decide(4) is None
    _feed(memory, "mtp", 4, 9.5, 1)
    verdict = memory.decide(4)
    assert verdict["reason"] == "measured_loss"
    assert verdict["rule"] == "sustained_loss"


def test_mtp_clearly_ahead_never_parks():
    memory = _memory()
    _feed(memory, "ordinary", 4, 10.0, 4)
    _feed(memory, "mtp", 4, 10.5, 4)  # 1.05x: ahead by more than 3%
    for _ in range(64):
        _feed(memory, "mtp", 4, 10.5, 1)
        assert memory.decide(4) is None


def test_unmeasured_width_uses_a_conservative_bound_from_a_wider_width():
    memory = _memory()
    _feed(memory, "ordinary", 8, 16.0, 4)  # 8 rows: 2 tokens/ms per row
    # Width 4 has no ordinary sample: bound = 16 * 4 / 8 = 8 tokens/ms, so the
    # static cold start no longer decides and MTP is measured instead.
    assert memory.decide(4) is None
    _feed(memory, "mtp", 4, 6.0, 4)  # 0.75x the bound
    for _ in range(4):
        _feed(memory, "mtp", 4, 6.0, 1)
        verdict = memory.decide(4)
    assert verdict["reason"] == "measured_loss"
    assert verdict["ordinary_rate_source"] == "bound_from_width_8"
    assert verdict["ordinary_tokens_per_ms"] == pytest.approx(8.0)
    # A narrower measured width never bounds a wider one.
    other = _memory()
    _feed(other, "ordinary", 2, 4.0, 4)
    assert other.decide(4)["reason"] == "static_width_threshold"


def _park_at(memory, width):
    _feed(memory, "ordinary", width, 10.0, 4)
    _feed(memory, "mtp", width, 5.0, 4)
    for _ in range(4):
        _feed(memory, "mtp", width, 5.0, 1)
        decision = memory.decide(width)
    assert decision["reason"] == "measured_loss"


def test_verdict_outlives_its_cohort_and_parks_wider_cohorts_only():
    """oMLX test_batch_park_verdict_outlives_its_cohort, mapped to widths."""
    memory = _memory()
    _park_at(memory, 4)
    wider = memory.decide(8)
    assert wider["reason"] == "park_memory" and wider["verdict_width"] == 4
    assert memory.decide(4)["reason"] == "park_memory"
    assert memory.decide(2) is None
    assert memory.would_park(5) and not memory.would_park(3)


def test_short_hold_keeps_the_verdict_and_a_long_hold_clears_it():
    memory = _memory()
    _park_at(memory, 4)
    _feed(memory, "ordinary", 2, 5.0, 4)
    _feed(memory, "mtp", 2, 6.0, 4)  # MTP wins at width 2
    for _ in range(31):
        _feed(memory, "mtp", 2, 6.0, 1)
        memory.decide(2)
    # A width-2 cohort cannot clear a verdict at width 4 (only k <= rows).
    assert memory.decide(4)["reason"] == "park_memory"
    # Expire the width-4 verdict (one more parked cohort), then let a width-4
    # cohort hold up.
    assert memory.decide(4)["reason"] == "park_memory"
    assert memory.decide(4) is None  # re-measuring MTP from scratch
    _feed(memory, "mtp", 4, 11.0, 4)
    for _ in range(31):
        _feed(memory, "mtp", 4, 11.0, 1)
        assert memory.decide(4) is None
    assert "4" in memory.snapshot()["verdicts"]
    _feed(memory, "mtp", 4, 11.0, 1)
    assert memory.decide(4) is None
    assert memory.snapshot()["verdicts"] == {}


def test_verdict_expires_after_parked_cohorts_and_repeat_loss_doubles():
    stats = {}
    memory = _memory(stats, cooldown_cohorts=2, max_cooldown_cohorts=6)
    _park_at(memory, 4)
    assert memory.snapshot()["verdicts"]["4"] == {"cooldown": 2, "remaining": 2}
    # Bare peeks (latch release) never consume the cooldown; a width-locked
    # join consumes it through commit() once its handoff commits.
    for _ in range(5):
        assert memory.would_park(4)
    assert memory.decide(4)["remaining_cohorts"] == 2
    assert memory.decide(8)["reason"] == "park_memory"  # second parked cohort
    assert stats["mtp_adaptive_park_verdicts_expired"] == 1
    assert not memory.would_park(4)
    assert "mtp:4" not in memory.snapshot()["rates"]  # MTP re-measured
    _park_at(memory, 4)
    assert memory.snapshot()["verdicts"]["4"]["cooldown"] == 4
    for _ in range(4):
        memory.decide(4)
    _park_at(memory, 4)
    assert memory.snapshot()["verdicts"]["4"]["cooldown"] == 6  # capped


def test_invalid_samples_are_skipped_and_counted():
    stats = {}
    memory = _memory(stats)
    assert not memory.observe("mtp", 2, 0, 1.0)
    assert not memory.observe("mtp", 2, 4, 0.0)
    assert stats["mtp_adaptive_park_samples_skipped"] == 2
    with pytest.raises(ValueError):
        memory.observe("draft", 2, 4, 1.0)


def test_memory_registry_is_per_model_and_route_and_resettable():
    class Model:
        pass

    a, b = Model(), Model()
    settings = AdaptiveParkSettings(enabled=True)
    first = park_memory_for(a, settings, static_max_width=3, route_key="r")
    assert park_memory_for(a, settings, static_max_width=3, route_key="r") is first
    assert park_memory_for(a, settings, static_max_width=3, route_key="s") is not first
    assert park_memory_for(a, settings, static_max_width=4, route_key="r") is not first
    assert park_memory_for(b, settings, static_max_width=3, route_key="r") is not first
    reset_park_memories(a)
    assert park_memory_for(a, settings, static_max_width=3, route_key="r") is not first
    del b
    gc.collect()


# -- BatchGenerator integration (tiny Qwen4, CPU) ----------------------------

def _tiny_model():
    from tests.test_batched_mtp import _tiny_qwen4_model

    return _tiny_qwen4_model()


def _generator(model, handoff, width=2):
    from mlx2.runtime.generate import BatchGenerator

    return BatchGenerator(
        model,
        completion_batch_size=width,
        prefill_batch_size=width,
        prefill_step_size=32,
        self_mtp={
            "num_draft": 2,
            "persistent": True,
            "segment_aware_live_tip": True,
            "segment_aware_cohort_size": width,
        },
        mtp_ordinary_handoff=MTPOrdinaryHandoffPolicy.from_value(handoff),
    )


def _finish(generator, uids):
    output = {uid: [] for uid in uids}
    terminal = {}
    for _ in range(200):
        _, responses = generator.next()
        for response in responses:
            output[response.uid].append(response.token)
            if response.finish_reason:
                terminal[response.uid] = response
        if len(terminal) == len(uids):
            return output, terminal
    raise AssertionError("generator did not terminate")


PROMPTS = [[1, 7, 3, 9, 2], [4, 5, 6, 2, 8]]


@pytest.fixture
def cpu_model():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(924)
    try:
        yield _tiny_model()
    finally:
        mx.set_default_device(previous)


def _insert(generator, n=2, tokens=8):
    return generator.insert(
        PROMPTS[:n],
        max_tokens=[tokens] * n,
        lane_rngs=[LaneRNG(31 + i) for i in range(n)],
        self_mtp_configs=[{"sampling_temp": 0.0}] * n,
    )


def test_static_handoff_route_builds_no_park_memory(cpu_model):
    gen = _generator(cpu_model, {"enabled": True, "max_mtp_width": 4})
    try:
        assert gen.mtp_park_memory is None
        assert gen.mtp_park_memory_snapshot() is None
        _finish(gen, _insert(gen))
        assert not any(k.startswith("mtp_adaptive_park") for k in gen.scheduler_stats)
    finally:
        gen.close()


def test_adaptive_park_measures_mtp_and_keeps_running_below_static(cpu_model):
    gen = _generator(
        cpu_model,
        {"enabled": True, "max_mtp_width": 4, "adaptive_park": {"enabled": True}},
    )
    try:
        assert gen.mtp_park_memory is not None
        assert gen.scheduler_stats["mtp_adaptive_park_engaged"] == 1
        _, terminal = _finish(gen, _insert(gen, tokens=16))
        assert gen.scheduler_stats.get("mtp_ordinary_handoff_events", 0) == 0
        assert gen.scheduler_stats["mtp_adaptive_park_mtp_samples"] >= 1
        assert "mtp:2" in gen.mtp_park_memory_snapshot()["rates"]
        assert all(
            r.mtp_receipt["route"] == "segmented_self_mtp" for r in terminal.values()
        )
    finally:
        gen.close()


def test_remembered_verdict_hands_off_a_new_cohort_exactly(cpu_model):
    from mlx2.runtime.generate import BatchGenerator

    handoff = {"enabled": True, "max_mtp_width": 4, "adaptive_park": {"enabled": True}}
    first = _generator(cpu_model, handoff)
    ordinary = BatchGenerator(
        cpu_model, completion_batch_size=2, prefill_batch_size=2, prefill_step_size=32
    )
    second = None
    try:
        # A verdict recorded by an earlier generator on this model and route.
        _park_at(first.mtp_park_memory, 2)
        first.close()
        first = None
        second = _generator(cpu_model, handoff)
        assert second.mtp_park_memory.would_park(2)
        out, terminal = _finish(second, _insert(second))
        ref_uids = ordinary.insert(PROMPTS, max_tokens=[8, 8])
        ref, _ = _finish(ordinary, ref_uids)
        uids = sorted(out)
        assert [out[u] for u in uids] == [ref[u] for u in ref_uids]
        stats = second.scheduler_stats
        assert stats["mtp_ordinary_handoff_events"] == 1
        assert stats["mtp_ordinary_handoff_park_memory"] == 1
        receipt = next(iter(terminal.values())).mtp_receipt
        assert receipt["route"] == "ordinary_after_mtp_handoff"
        decision = receipt["mtp_ordinary_handoff"]["decision"]
        assert decision["reason"] == "park_memory" and decision["verdict_width"] == 2
        assert "adaptive_park" in receipt["mtp_ordinary_handoff"]["policy"]
        # Ordinary decode after the handoff was measured; the parked cohort
        # counted against the verdict's cooldown.
        assert stats["mtp_adaptive_park_ordinary_samples"] >= 1
        assert second.mtp_park_memory_snapshot()["verdicts"]["2"]["remaining"] == 1
    finally:
        for gen in (first, second, ordinary):
            if gen is not None:
                gen.close()


def test_kill_switch_forces_the_static_threshold(cpu_model, monkeypatch):
    monkeypatch.setenv(KILL_SWITCH_ENV, "0")
    gen = _generator(
        cpu_model,
        {"enabled": True, "max_mtp_width": 1, "adaptive_park": {"enabled": True}},
    )
    try:
        assert gen.mtp_park_memory is None
        assert gen.scheduler_stats["mtp_adaptive_park_kill_switch"] == 1
        _finish(gen, _insert(gen))
        assert gen.scheduler_stats["mtp_ordinary_handoff_static_width_threshold"] == 1
    finally:
        gen.close()


# -- joining-cohort handoff consumes the cooldown (Codex review item 6) -------


def test_commit_consumes_a_remembered_verdict_once():
    stats = {}
    memory = _memory(stats, cooldown_cohorts=2)
    _park_at(memory, 4)
    decision = memory.peek(4)
    assert decision["reason"] == "park_memory"
    memory.commit(decision)
    assert memory.snapshot()["verdicts"]["4"]["remaining"] == 1
    memory.commit(memory.peek(6))
    assert memory.snapshot()["verdicts"] == {} or (
        memory.snapshot()["verdicts"]["4"]["remaining"] == 0
    )
    assert not memory.would_park(4)
    assert stats["mtp_adaptive_park_verdicts_expired"] == 1
    # Static and measured decisions carry no cooldown to consume.
    memory.commit({"reason": "static_width_threshold", "max_mtp_width": 3})
    memory.commit(None)


def _joining_batch(memory, *, lanes=2):
    """The width-locked joining path of MTPGenerationBatch._attach_packages,
    on a stand-in batch whose handoff records the call."""
    from types import SimpleNamespace

    from mlx2.runtime import generate as G

    handoffs = []
    batch = SimpleNamespace(
        _ordinary_handoff_latched=False,
        state=SimpleNamespace(lanes=[SimpleNamespace(num_draft=2)] * lanes),
        segmented_live_tip=True,
        _segmented_compute_width_locked=True,
        ordinary_handoff_policy=MTPOrdinaryHandoffPolicy(enabled=True, max_mtp_width=3),
        park_memory=memory,
        _paused={},
        _width_lock_deferrals={},
        adaptive_depth_policy=None,
        _plain_ready=[],
    )

    def handoff(packages, *, decision, projected_width):
        handoffs.append((decision, projected_width))
        batch._ordinary_handoff_latched = True

    batch._handoff_all_to_plain = handoff

    def join(n=2):
        batch._ordinary_handoff_latched = False
        batch._paused.clear()
        packages = [
            SimpleNamespace(handoff_receipt=None,
                            detached=SimpleNamespace(lane=SimpleNamespace(num_draft=2, uid=i)))
            for i in range(n)
        ]
        G.MTPGenerationBatch._attach_packages(batch, packages)

    return join, handoffs


def test_joining_cohort_handoffs_consume_the_cooldown():
    """Width-two cohorts joined by two lanes (projected width four) under a
    width-four verdict: each committed handoff counts one parked cohort, so
    the verdict expires after its cooldown instead of parking forever."""
    stats = {}
    memory = _memory(stats, cooldown_cohorts=2)
    _park_at(memory, 4)
    join, handoffs = _joining_batch(memory)
    join()
    assert handoffs[-1][0]["reason"] == "park_memory" and handoffs[-1][1] == 4
    assert memory.snapshot()["verdicts"]["4"]["remaining"] == 1
    join()
    assert len(handoffs) == 2
    assert not memory.would_park(4)
    assert stats["mtp_adaptive_park_verdicts_expired"] == 1
    for _ in range(3):
        join()
    # Expired: MTP is re-measured at width 4, no further remembered handoffs.
    assert all(d["reason"] != "park_memory" for d, _w in handoffs[2:])
