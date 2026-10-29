"""APCv2 refuses prompt state prefilled off the route's planned slice grid.

Prefill state follows its slice boundaries bitwise (measured 2026-10-28:
Qwen3.8-27B and Sage change every layer's state under any schedule other
than the route's normal chunk; same schedule twice is bit-exact).  The
generators record each request's executed slice ends against the idle plan;
serving refuses to publish a checkpoint past the first deviation and counts
``apcv2_store_skipped_slice_schedule``.  CPU only: synthetic schedules over
the tiny hybrid models the depth-bound tests use.
"""

from collections import Counter, deque
from types import SimpleNamespace as NS

import mlx.core as mx
import pytest

from mlx2.runtime.generate import BatchGenerator, interior_checkpoint_positions
from mlx2.runtime.prefill_slice_schedule import (
    PrefillSliceSchedule,
    changed_at,
    mark_deviation,
    record_slice,
)
from mlx2.runtime.sample_utils import LaneRNG
from mlx2.serving import (
    ServingEngine,
    adapter_owns_prefill_partition,
    checkpoint_slice_schedule_changed,
)
from route_harness import collect, make_engine, patch_host, tiny_qwen38_mtp
from test_batched_mtp import _tiny_qwen4_model

STEP = 32


# -- the record ---------------------------------------------------------------


def test_planned_slices_keep_the_schedule_exact():
    schedule = PrefillSliceSchedule(0)
    for start in (0, 32, 64):
        schedule.record(start, 32, 32)
    schedule.record(96, 7, 7)  # the arithmetic remainder is planned too
    receipt = schedule.receipt()
    assert receipt["planned"] and receipt["first_deviation"] is None
    assert receipt["boundaries"] == [32, 64, 96, 103]
    assert not changed_at(receipt, 103)


def test_first_narrow_slice_taints_every_later_position_even_after_realigning():
    schedule = PrefillSliceSchedule(0)
    schedule.record(0, 32, 32)
    schedule.record(32, 16, 32)  # a load-chosen slice
    schedule.record(48, 16, 16)  # back on a 32-row boundary at 64
    schedule.record(64, 32, 32)
    receipt = schedule.receipt()
    assert receipt["first_deviation"] == 32 and receipt["deviation_reason"] == "width"
    assert (receipt["deviation_planned_rows"], receipt["deviation_executed_rows"]) == (32, 16)
    assert not changed_at(receipt, 32)  # boundaries below 32 are all planned
    assert changed_at(receipt, 33) and changed_at(receipt, 64) and changed_at(receipt, 96)


def test_reasoned_and_untracked_slices_deviate_and_warm_starts_open_at_offset():
    registry = {}
    record_slice(registry, 1, 100, 16, 16)  # warm request resumes at 100
    record_slice(registry, 1, 116, 16, 16, reason="mixed_forward")
    assert registry[1].deviation == 116 and registry[1].reason == "mixed_forward"
    record_slice(registry, 2, 0, 16, 16)
    record_slice(registry, 2, 40, 16, 16)  # a slice ran outside the record
    assert registry[2].deviation == 16 and registry[2].reason == "untracked_slice"
    mark_deviation(registry, 3, 50, "recovery_replay")
    assert registry[3].receipt()["first_deviation"] == 50


# -- serving gate -------------------------------------------------------------


def _engine(adapter=None):
    engine = ServingEngine.__new__(ServingEngine)
    engine.apc_reuse_disabled_reason = None
    engine.counts = Counter()
    if adapter is not None:
        engine.adapter = adapter
    return engine


def test_exact_schedule_publishes_and_off_plan_schedule_refuses_with_counter():
    stored = []
    apc = NS(store=lambda key, tokens, cache, **kw: stored.append(len(tokens)))
    engine = _engine()
    owner = NS(apcv2_slice_schedule_refusals=0)
    assert engine._publish_checkpoint(apc, "k", [1] * 8, [], prefill_slice_schedule_changed=False)
    assert not engine._publish_checkpoint(
        apc, "k", [1] * 8, [], prefill_slice_schedule_changed=True,
        slice_schedule_owner=owner,
    )
    assert stored == [8]
    assert engine.counts["apcv2_store_skipped_slice_schedule"] == 1
    assert owner.apcv2_slice_schedule_refusals == 1


def test_partition_law_adapter_keeps_its_own_hook_behaviour():
    """DS4.1 ``position_grid_v1`` decides exactness itself; unchanged here."""
    refusals = []

    class GridAdapter:
        def apc_partition_checkpoint_position(self, tokens, *, cached_tokens=0, cuts=()):
            return None

        def apc_publication_refusal(self, prompt_cache, tokens):
            refusals.append(len(tokens))
            return None if len(tokens) % 4 == 0 else "off-grid"

    adapter = GridAdapter()
    assert adapter_owns_prefill_partition(adapter)
    assert not adapter_owns_prefill_partition(object())
    stored = []
    apc = NS(store=lambda key, tokens, cache, **kw: stored.append(len(tokens)))
    engine = _engine(adapter)
    assert engine._publish_checkpoint(apc, "k", [1] * 8, [], prefill_slice_schedule_changed=True)
    assert not engine._publish_checkpoint(apc, "k", [1] * 7, [], prefill_slice_schedule_changed=True)
    assert stored == [8] and refusals == [8, 7]
    assert engine.counts["apcv2_store_skipped_slice_schedule"] == 0
    assert engine.counts["apcv2_store_skipped_partition"] == 1


def test_evidence_sources_owner_snapshot_live_record_and_capture_override():
    deviated = PrefillSliceSchedule(0)
    deviated.record(0, 16, 32)
    receipt = deviated.receipt()  # first deviation at 0
    owner = NS(uid=4, prefill_slice_schedule=receipt)
    assert checkpoint_slice_schedule_changed(None, owner, [0] * 16)
    assert not checkpoint_slice_schedule_changed(None, owner, [])
    live = NS(prefill_slice_schedule=lambda uid: receipt if uid == 5 else None)
    fresh = NS(uid=5, prefill_slice_schedule=None)
    assert checkpoint_slice_schedule_changed(live, fresh, [0] * 3)
    assert not checkpoint_slice_schedule_changed(live, NS(uid=6), [0] * 3)
    assert checkpoint_slice_schedule_changed(live, None, [0] * 3, uid=5)
    assert not checkpoint_slice_schedule_changed(
        live, fresh, [0] * 3, checkpoint={"prefill_slice_schedule_changed": False}
    )
    # Prompt-only state is covered (unlike the numerical-geometry gate).
    assert checkpoint_slice_schedule_changed(None, NS(prompt_tokens=99, prefill_slice_schedule=receipt), [0] * 4)


# -- BatchGenerator: ordinary and self-MTP prefill ----------------------------


@pytest.fixture(scope="module")
def model():
    mx.random.seed(41)
    return _tiny_qwen4_model()


def _run(model, prompt, *, mtp, checkpoints=None, live_cap=None, monkeypatch=None,
         depth_budget=None):
    if live_cap is not None:
        original = BatchGenerator._prompt_prefill_step
        calls = {"n": 0}

        def capped(self, total_tokens):
            # A load-dependent cap (a live memory recommendation, say) that
            # engages only after the first ``live_cap[0]`` step queries.
            calls["n"] += 1
            step = original(self, total_tokens)
            return step if calls["n"] <= live_cap[0] else min(step, live_cap[1])

        monkeypatch.setattr(BatchGenerator, "_prompt_prefill_step", capped)
    kwargs = dict(prefill_step_size=STEP, completion_batch_size=1, prefill_batch_size=1,
                  adaptive_prefill=True, prefill_depth_budget=depth_budget)
    insert = dict(max_tokens=[3], lane_rngs=[LaneRNG(5)])
    if mtp:
        kwargs["self_mtp"] = {"num_draft": 2, "persistent": True}
        insert.update(mtp_states=[None],
                      self_mtp_configs=[{"sampling_temp": 0.0, "num_draft": 2}])
    if checkpoints is not None:
        kwargs["apc_interior_checkpoints"] = checkpoints
    gen = BatchGenerator(model, **kwargs)
    try:
        uid = gen.insert([prompt], **insert)[0]
        schedule = None
        for _ in range(1000):
            prompts, responses = gen.next()
            if schedule is None and any(
                getattr(p, "end_of_prompt", False) and p.uid == uid for p in prompts
            ):
                schedule = gen.pop_prefill_slice_schedule(uid)
            if any(r.finish_reason for r in responses):
                break
        else:
            raise AssertionError("request did not finish")
        if schedule is None:
            schedule = gen.pop_prefill_slice_schedule(uid)
        interiors = gen.pop_interior_checkpoints(uid) if checkpoints else []
        boundary = gen.pop_prompt_boundary(uid)
    finally:
        gen.close()
    return schedule, interiors, boundary


@pytest.mark.parametrize("mtp", [False, True])
def test_idle_schedule_is_the_planned_grid(model, mtp):
    prompt = [(5 * i + 3) % 60 + 2 for i in range(150)]
    schedule, _, boundary = _run(model, prompt, mtp=mtp)
    assert schedule is not None and schedule["planned"], schedule
    assert schedule["boundaries"][0] == STEP
    assert schedule["covered"] == len(prompt) - 1
    assert boundary is not None
    assert not changed_at(schedule, len(boundary["tokens"]))


@pytest.mark.parametrize("mtp", [False, True])
def test_static_depth_budget_is_part_of_the_plan(model, monkeypatch, mtp):
    monkeypatch.setattr(BatchGenerator, "PREFILL_DEPTH_FLOOR", 4)
    prompt = [(5 * i + 3) % 60 + 2 for i in range(200)]
    schedule, _, _ = _run(model, prompt, mtp=mtp, depth_budget=STEP * 64)
    assert schedule["planned"], schedule
    assert min(b - a for a, b in zip([0] + schedule["boundaries"], schedule["boundaries"])) < STEP


@pytest.mark.parametrize("mtp", [False, True])
def test_load_cap_refuses_past_the_deviation_and_keeps_earlier_interiors(model, monkeypatch, mtp):
    prompt = [(3 * i + 5) % 60 + 2 for i in range(300)]
    policy = {"count": 3, "min_stride": 4}
    positions = interior_checkpoint_positions(len(prompt), **policy)
    clean, clean_interiors, _ = _run(model, prompt, mtp=mtp, checkpoints=policy)
    assert clean["planned"]
    assert [item["covered_tokens"] for item in clean_interiors] == list(positions)
    capped, interiors, boundary = _run(
        model, prompt, mtp=mtp, checkpoints=policy, live_cap=(3, 8),
        monkeypatch=monkeypatch,
    )
    deviation = capped["first_deviation"]
    assert not capped["planned"] and capped["deviation_reason"] == "width"
    assert capped["deviation_executed_rows"] < capped["deviation_planned_rows"]
    covered = [item["covered_tokens"] for item in interiors]
    assert covered == list(positions)  # cut placement is unchanged
    before = [p for p in covered if p <= deviation]
    after = [p for p in covered if p > deviation]
    assert before and after, (deviation, covered)
    assert all(not changed_at(capped, p) for p in before)
    assert all(changed_at(capped, p) for p in after)
    assert changed_at(capped, len(boundary["tokens"]))


def test_mixed_forward_slice_is_off_plan_at_any_width():
    gen = BatchGenerator.__new__(BatchGenerator)
    gen._prefill_slice_schedules = {}
    gen._record_prefill_slice(7, 0, 16, 16)
    gen._record_prefill_slice(7, 16, 16, 16, reason="mixed_forward")
    receipt = gen.prefill_slice_schedule(7)
    assert receipt["first_deviation"] == 16 and receipt["deviation_reason"] == "mixed_forward"
    assert gen.pop_prefill_slice_schedule(7)["first_deviation"] == 16
    assert gen.prefill_slice_schedule(7) is None


# -- external-draft (ast_dflash2) target prefill ------------------------------


def _external(step):
    from collections import defaultdict

    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    gen = ExternalDraftBatchGenerator.__new__(ExternalDraftBatchGenerator)
    gen.prefill_step = step
    gen.prefill_step_autoscale = False
    gen.scheduler_stats = defaultdict(int)
    gen.mx = mx
    gen.layers = ()
    gen.model = NS(prefill_body=lambda x, cache, layers: mx.zeros((1, x.shape[1], 2)))
    gen.decode_time_fairness = NS(enabled=False)
    gen.prefill_allocator_reclaim = False
    gen._prefill_response = lambda lane, **kw: NS(uid=lane.uid)
    gen._append_context = lambda lane, following: None
    return gen


def _lane(uid, length):
    from mlx2.runtime.external_speculative import Lane

    return Lane(uid=uid, history=[], remaining=deque(range(length)), cache=[],
                draft_cache=[], tail=mx.zeros((1, 0, 2)), rng=None, maximum=4,
                processors=[], sampling={})


def test_external_prefill_records_fair_and_admission_narrowing():
    gen = _external(8)
    idle = _lane(1, 30)
    while len(idle.remaining) > 1:
        gen._prefill(idle)
    assert gen.prefill_slice_schedule(1)["planned"]
    assert gen.prefill_slice_schedule(1)["boundaries"] == [8, 16, 24, 29]
    contended = _lane(2, 30)
    gen._prefill(contended)
    gen._prefill(contended, step=4)  # a fairness cap / halved admission
    gen._prefill(contended)
    receipt = gen.pop_prefill_slice_schedule(2)
    assert receipt["first_deviation"] == 8 and not changed_at(receipt, 8)
    assert changed_at(receipt, 12) and changed_at(receipt, 20)
    assert gen.prefill_slice_schedule(2) is None


# -- served route, end to end -------------------------------------------------


def test_served_route_publishes_idle_state_and_refuses_load_capped_state(monkeypatch):
    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    prompt = [(7 * i + 2) % (vocab - 2) + 1 for i in range(120)]
    body = {"tokens": prompt, "max_tokens": 3, "temperature": 0}

    idle = make_engine(model, vocab, mtp=False)
    try:
        first = collect(idle.submit(dict(body)))
        second = collect(idle.submit(dict(body)))
        idle_counts = dict(idle.counts)
    finally:
        idle.close()
    assert "error" not in first and "error" not in second
    schedule = first["receipt"]["prefill_slice_schedule"]
    assert schedule["planned"] and schedule["apcv2_publications_refused"] == 0
    assert idle_counts.get("apcv2_store_skipped_slice_schedule", 0) == 0
    assert idle_counts.get("cached_prompt_tokens", 0) > 0  # the warm request hit

    original = BatchGenerator._prompt_prefill_step
    monkeypatch.setattr(
        BatchGenerator, "_prompt_prefill_step",
        lambda self, total: min(original(self, total), 8),
    )
    loaded = make_engine(model, vocab, mtp=False)
    try:
        first = collect(loaded.submit(dict(body)))
        second = collect(loaded.submit(dict(body)))
        loaded_counts = dict(loaded.counts)
    finally:
        loaded.close()
    assert "error" not in first and "error" not in second
    schedule = first["receipt"]["prefill_slice_schedule"]
    assert not schedule["planned"] and schedule["first_deviation"] == 0
    assert schedule["apcv2_publications_refused"] >= 1
    assert loaded_counts["apcv2_store_skipped_slice_schedule"] >= 1
    assert loaded_counts.get("prefill_slice_schedule_off_plan_requests", 0) >= 1
    assert loaded_counts.get("cached_prompt_tokens", 0) == 0  # nothing to hit
