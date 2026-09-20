"""Item 10: SRPT prefill order with a bypass cap and one-slice contention.

Unit tests pin ``PrefillOrder`` semantics; scheduler tests drive the real
``BatchGenerator`` on CPU with the tiny hybrid GDN + MTP model from
``test_short_request_prefill_starvation`` and measure in scheduler rounds.
"""

from collections import deque
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.runtime.adaptive_policy import (
    DecodeTimeFairness,
    PrefillCandidate,
    PrefillOrder,
)
from mlx2.runtime.generate import BatchGenerator
from mlx2.runtime.sample_utils import LaneRNG

from test_short_request_prefill_starvation import tiny_model

SRPT = {"order": "srpt", "max_bypass": 3, "one_slice_contention": True}


@pytest.fixture(autouse=True)
def _plain_self_mtp_route(monkeypatch):
    """Pin the plain self-MTP prefill path these scheduler tests measure.

    ``adapters.flash_next.configure_environment`` writes its serving profile
    into ``os.environ`` process-wide, so any earlier test that calls it leaves
    ``MLX_LM_SEGMENTED_SELF_MTP=1`` behind.  The segmented route drives prefill
    from its own segmented loop and never hands a long prefill back to
    ``_select_prefill_indices`` mid-prompt, so SRPT has nothing to reorder and
    the bypass counters stay at zero.  That is a real coverage gap for the
    segmented route (tracked in the item 10 report), not an outcome these
    tests intend to measure.
    """
    monkeypatch.delenv("MLX_LM_SEGMENTED_SELF_MTP", raising=False)


def _prompt(length, salt):
    return [(salt * 7 + 5 * i) % 120 + 2 for i in range(length)]


# -- PrefillOrder -----------------------------------------------------------


def test_absent_policy_is_disabled_and_counter_free():
    order = PrefillOrder.from_value(None)
    assert not order.enabled and not order.one_slice_contention
    assert order.counters == {}


def test_parse_defaults_and_rejections():
    order = PrefillOrder.from_value({})
    assert order.enabled
    assert order.as_dict() == SRPT
    assert PrefillOrder.from_value({"max_bypass": 1}).max_bypass == 1
    for bad in (
        {"order": "fifo"},
        {"max_bypass": 0},
        {"max_bypass": True},
        {"one_slice_contention": 1},
        {"unknown": 1},
        [],
    ):
        with pytest.raises(ValueError):
            PrefillOrder.from_value(bad)
    with pytest.raises(ValueError):
        PrefillOrder(one_slice_contention=True)


def test_disabled_select_is_main_mtp_shortest_residual_key():
    """Unification: disabled == main's (residual, -cached, position) key."""
    order = PrefillOrder()
    candidates = [
        PrefillCandidate(uid=0, remaining=900, cached=0, bypassed=99),
        PrefillCandidate(uid=1, remaining=300, cached=0),
        PrefillCandidate(uid=2, remaining=300, cached=64),
        PrefillCandidate(uid=3, remaining=300, cached=64),
    ]
    assert order.select(candidates) == 2  # bypass counts ignored when off
    order.commit([2], candidates)
    assert order._bypassed == {} and order.counters == {}


def test_bypass_cap_forces_the_overtaken_request_first():
    order = PrefillOrder.from_value({"max_bypass": 3})
    long_uid = 0
    served = []
    next_uid = 1
    # A fresh short prompt arrives before every decision.
    for _ in range(12):
        candidates = [
            order.candidate(long_uid, 10_000),
            order.candidate(next_uid, 8),
        ]
        pick = candidates[order.select(candidates)].uid
        order.commit([pick], candidates, pending=[long_uid, next_uid])
        served.append(pick)
        if pick != long_uid:
            next_uid += 1
    gaps = []
    run = 0
    for uid in served:
        if uid == long_uid:
            gaps.append(run)
            run = 0
        else:
            run += 1
    assert gaps and max(gaps) == 3, served
    assert order.counters["bypass_forced"] == len(gaps)


def test_younger_requests_are_not_counted_as_overtaken():
    order = PrefillOrder.from_value({})
    candidates = [order.candidate(5, 10), order.candidate(9, 999)]
    order.commit([5], candidates)
    assert order._bypassed == {}
    candidates = [order.candidate(5, 999), order.candidate(9, 10)]
    order.commit([9], candidates, pending=[5, 9])
    assert order._bypassed == {5: 1}
    order.commit([9], [order.candidate(9, 1)], pending=[9])
    assert order._bypassed == {}  # pruned once no longer pending


def test_stall_bound_ignores_enabled_and_counters():
    fairness = DecodeTimeFairness(enabled=False, stall_target_ms=1e-6)
    assert fairness.stall_bound(2048) == fairness.fallback_cap
    fairness.observe_prefill(1000, 1.0, contended=False)
    assert fairness.stall_bound(2048) == 64
    assert fairness.cap(2048, contended=True) == 2048
    assert fairness.counters["cap_clamps"] == 0


# -- one-slice contention (boundary-clamped fit) ------------------------------


def _one_slice_host(queued_short, boundary=None):
    long_row = [[list(range(900)), [1]], 100, 1001, False, 0, 0.0, None]
    return SimpleNamespace(
        _currently_processing=[long_row],
        _prompt_batch=SimpleNamespace(uids=[0]),
        _unprocessed_sequences=deque(
            [(1, [list(range(queued_short)), [1]], 4, [], [], None, [], None, 0.0, None)]
        ),
        _next_interior_checkpoint=lambda uid, covered: (
            boundary if uid == 1 else None
        ),
    )


def test_one_slice_contention_uses_boundary_clamped_step():
    assert BatchGenerator._one_slice_contended(_one_slice_host(40), 64)
    assert not BatchGenerator._one_slice_contended(_one_slice_host(200), 64)
    # An interior checkpoint inside the short prompt clamps its chunk: it no
    # longer finishes in one slice, so it does not contend.
    assert not BatchGenerator._one_slice_contended(
        _one_slice_host(40, boundary=20), 64
    )


# -- real scheduler -----------------------------------------------------------


def _make(model, mtp, *, step, lanes=4, prefill_batch_size=None,
          _segmented=False, _cohort_size=None, **kw):
    if mtp:
        kw["self_mtp"] = {
            "num_draft": 2,
            "persistent": True,
            "rate_gate": False,
            "prefill_step_size": step,
            **({"segment_aware_live_tip": True} if _segmented else {}),
            **({} if _cohort_size is None
               else {"segment_aware_cohort_size": _cohort_size}),
        }
    return BatchGenerator(
        model,
        completion_batch_size=lanes,
        prefill_batch_size=prefill_batch_size or min(2, lanes),
        prefill_step_size=step,
        prefill_batch_window=1,
        adaptive_prefill=True,
        **kw,
    )


def _insert(gen, prompt, mtp, seed, config=None):
    extra = (
        {
            "lane_rngs": [LaneRNG(seed)],
            "self_mtp_configs": [{"sampling_temp": 0.0, **(config or {})}],
        }
        if mtp
        else {}
    )
    return gen.insert([prompt], max_tokens=[4], **extra)[0]


@pytest.fixture(scope="module")
def model():
    return tiny_model()


def test_default_off_adds_no_scheduler_stats(model):
    gen = _make(model, False, step=32)
    try:
        _insert(gen, _prompt(40, 1), False, 1)
        for _ in range(6):
            gen.next()
        assert not any(k.startswith("prefill_scheduling_") for k in gen.scheduler_stats)
    finally:
        gen.close()


@pytest.mark.parametrize("mtp", [False, True], ids=["ordinary", "self_mtp"])
def test_short_ttft_is_bounded_behind_a_64k_prefill(model, mtp):
    step = 2048
    long_prompt = _prompt(65536, 3)
    gen = _make(model, mtp, step=step, prefill_scheduling=SRPT)
    first = {}
    try:
        long_uid = _insert(gen, long_prompt, mtp, 10)
        gen.next()  # the 64k prefill is now in progress
        short_uid = _insert(gen, _prompt(12, 4), mtp, 11)
        long_progress = 0
        for rnd in range(1, 12):
            prompts, generated = gen.next()
            for r in prompts:
                if r.uid == long_uid:
                    long_progress = max(long_progress, int(r.progress[0]))
            for r in generated:
                first.setdefault(r.uid, rnd)
            if short_uid in first:
                break
        assert short_uid in first and first[short_uid] <= 4, first
        assert long_uid not in first
        assert long_progress < len(long_prompt) // 4
    finally:
        gen.close()


def test_bypass_cap_refuses_a_window_it_could_never_fire_in(model):
    """The cap needs an ordering window wider than itself, or it is a lie.

    ``_next_mtp`` admits a FIFO prefix of the prefill queue and applies
    ``PrefillOrder`` inside it, so a request only accrues bypasses while it
    stays a candidate.  Measured on CPU with ``max_bypass=3``: a window of 2
    or 3 yields ``bypass_forced == 0``; the cap first fires at 4.

    In serving the window is ``--max-lanes`` -- every adapter's
    ``execution_config`` sets ``segment_aware_cohort_size`` to ``max_lanes``,
    so the module default of 2 is unreachable there.  The reachable way in is
    a low ``--max-lanes`` (37 qualification runs used 1), which is what the
    first case below models.
    """
    # What an adapter produces: cohort size == lanes.  Below max_bypass + 1
    # the cap cannot fire, so startup refuses.
    for lanes in (1, 2, 3):
        with pytest.raises(ValueError, match="admission window of at least 4"):
            _make(model, True, step=32, lanes=lanes, prefill_scheduling=SRPT,
                  _segmented=True, _cohort_size=lanes)
    # The --max-lanes default of 4 is the boundary and is accepted, as is any
    # wider deployment; lowering the cap instead also works.
    for lanes, policy in ((4, SRPT), (20, SRPT), (2, {**SRPT, "max_bypass": 1})):
        gen = _make(model, True, step=32, lanes=lanes, prefill_scheduling=policy,
                    _segmented=True, _cohort_size=lanes)
        gen.close()


def test_mtp_bypass_cap_bounds_long_prefill_wait(model):
    """Shorts arriving every round overtake a long prefill at most 3 in a row."""
    step = 32
    gen = _make(model, True, step=step, lanes=16, prefill_scheduling=SRPT)
    try:
        long_uid = _insert(gen, _prompt(40 * step, 5), True, 10)
        gen.next()
        served = []
        seed = 100
        for _ in range(16):
            _insert(gen, _prompt(8, seed), True, seed)
            seed += 1
            prompts, _ = gen.next()
            served.append(any(r.uid == long_uid for r in prompts))
        run = longest = 0
        for hit in served:
            run = 0 if hit else run + 1
            longest = max(longest, run)
        assert longest <= 3, served
        assert any(served)
        assert gen.scheduler_stats["prefill_scheduling_bypass_forced"] >= 1
        assert gen.scheduler_stats["prefill_scheduling_bypasses"] >= 3
    finally:
        gen.close()


@pytest.mark.parametrize("enabled", [False, True], ids=["fifo", "srpt"])
def test_ordinary_admission_is_srpt_with_bypass_cap(model, enabled):
    step = 32
    gen = _make(
        model,
        False,
        step=step,
        lanes=8,
        prefill_batch_size=1,
        prefill_scheduling=SRPT if enabled else None,
    )
    admitted = []
    try:
        long_uid = _insert(gen, _prompt(20 * step, 6), False, 1)
        shorts = [_insert(gen, _prompt(8, 7 + i), False, 2 + i) for i in range(5)]
        for _ in range(12):
            prompts, _ = gen.next()
            for r in prompts:
                if r.uid not in admitted:
                    admitted.append(r.uid)
            if long_uid in admitted and len(admitted) >= 4:
                break
        if enabled:
            assert admitted[:4] == shorts[:3] + [long_uid], admitted
        else:
            assert admitted[0] == long_uid
    finally:
        gen.close()


@pytest.mark.parametrize("contention", [False, True], ids=["off", "on"])
def test_one_slice_contention_bounds_the_long_slice(model, contention):
    step = 256
    policy = {"one_slice_contention": contention}
    gen = _make(
        model,
        False,
        step=step,
        # Deterministic bound: any measured rate floors to the 64-token grid.
        decode_time_fairness={"enabled": False, "stall_target_ms": 1e-6},
        prefill_scheduling=policy,
    )
    try:
        long_uid = _insert(gen, _prompt(20 * step, 8), False, 1)
        deltas = []
        last = 0

        def advance():
            nonlocal last
            prompts, _ = gen.next()
            for r in prompts:
                if r.uid == long_uid and r.progress[0] > last:
                    deltas.append(r.progress[0] - last)
                    last = r.progress[0]

        advance()
        _insert(gen, _prompt(40, 9), False, 2)
        advance()
        advance()
        assert deltas[0] == step
        if contention:
            assert deltas[1] == 64
            assert gen.scheduler_stats["prefill_scheduling_one_slice_clamps"] >= 1
        else:
            assert deltas[1] == step
    finally:
        gen.close()


@pytest.mark.parametrize("enabled", [False, True], ids=["main", "srpt"])
def test_srpt_keeps_declared_cohort_atomic(model, enabled):
    step = 32
    gen = _make(
        model, True, step=step, lanes=4, prefill_scheduling=SRPT if enabled else None
    )
    cohort = {"tenant_id": "t", "id": "srpt-c2", "size": 2}
    first = {}
    try:
        members = [
            _insert(gen, _prompt(6 * step, 20 + i), True, 30 + i, {"batch_cohort": dict(cohort), "num_draft": 2})
            for i in range(2)
        ]
        gen.next()
        _insert(gen, _prompt(8, 40), True, 41)
        for rnd in range(1, 80):
            _, generated = gen.next()
            for r in generated:
                first.setdefault(r.uid, rnd)
            if all(uid in first for uid in members):
                break
        # Both members attach in one ``_make_mtp_batch`` (a split would raise
        # "declared batch cohort must be prepared as one whole batch").
        assert first[members[0]] == first[members[1]], first
    finally:
        gen.close()


# -- server-owned policy --------------------------------------------------------


@pytest.mark.parametrize(
    ("prompt_lookup", "backend"), [(True, None), (False, "external_draft")]
)
def test_serving_rejects_prefill_scheduling_off_batch_generator_routes(
    monkeypatch, prompt_lookup, backend
):
    from mlx2 import serving
    from test_apc_interior_route_selection import _UnsupportedInteriorAdapter

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "fake"})

    class Adapter(_UnsupportedInteriorAdapter):
        pass

    Adapter.backend = backend
    engine = serving.ServingEngine(
        "fixture",
        adapter_factory=Adapter,
        qualification_mode=True,
        mtp=False,
        prompt_lookup=prompt_lookup,
        execution_policy={"prefill_scheduling": {"max_bypass": 2}},
    )
    try:
        engine.thread.join(5)
        assert not engine.ready.is_set()
        assert "prefill_scheduling requires" in (engine.error or "")
    finally:
        engine.close()


def test_serving_parses_policy_at_construction():
    from mlx2 import serving

    with pytest.raises(ValueError, match="max_bypass"):
        serving.ServingEngine(
            "fixture",
            adapter_factory=lambda *_a, **_k: None,
            execution_policy={"prefill_scheduling": {"max_bypass": 0}},
        )


def test_qualification_feature_only_when_selected():
    from mlx2.qualification import required_feature_checks

    assert "feature_prefill_scheduling" not in required_feature_checks({})
    assert "feature_prefill_scheduling" in required_feature_checks(
        {"prefill_scheduling": dict(SRPT)}
    )


def test_scheduler_counters_export_under_bounded_mechanism():
    from mlx2.prometheus import PrometheusBuilder, _add_scheduler

    builder = PrometheusBuilder()
    _add_scheduler(
        builder,
        {
            "prefill_scheduling_bypasses": 5,
            "prefill_scheduling_bypass_forced": 2,
            "prefill_scheduling_one_slice_clamps": 1,
        },
    )
    text = builder.render()
    for event, value in (
        ("prefill_scheduling_bypasses", 5),
        ("prefill_scheduling_bypass_forced", 2),
        ("prefill_scheduling_one_slice_clamps", 1),
    ):
        assert (
            f'mlx2_scheduler_events_total{{event="{event}",'
            f'mechanism="prefill_scheduling"}} {value}'
        ) in text, text
