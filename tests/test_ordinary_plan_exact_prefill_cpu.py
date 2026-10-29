"""Ordinary-route prefill keeps the idle plan's slices beside live decode.

GPU 2026-10-28 (Flash-Next ``--ordinary``, 4 lanes, cold bursts): lanes
prefilled beside decoding lanes ran load-shortened slices (deviation_reason
"width", planned 4095/8192 vs executed 64/192/256), so their prompt state
left the planned grid and APCv2 refused it (apcv2_store_skipped_slice_schedule
51); warm hits were 2/4 in every run.  An adapter that declares
``prefill_plan_exact`` in its execution config gets the self-MTP law
(e30132478) on the ordinary BatchGenerator round too: load may defer a slice,
never reshape it, and a live memory cap below the plan still wins.  CPU only,
on the tiny hybrid model the slice-schedule tests use.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.generate import BatchGenerator
from mlx2.runtime.sample_utils import LaneRNG
from test_batched_mtp import _tiny_qwen4_model

STEP = 32
# Under contention the adaptive round takes its smallest measured slice:
# the shape the GPU receipts showed (64-256 rows against an 8192 plan).
CONTENDED_SLICE = 8
PROMPT = [(5 * i + 3) % 60 + 2 for i in range(150)]
SHORT = [7, 9, 11, 13, 15]


@pytest.fixture(scope="module")
def model():
    mx.random.seed(41)
    return _tiny_qwen4_model()


def _generator(model, *, plan_exact, lanes=2, prefill_batch=1):
    return BatchGenerator(
        model, prefill_step_size=STEP, completion_batch_size=lanes,
        prefill_batch_size=prefill_batch, adaptive_prefill=True,
        adaptive_prefill_slices=(CONTENDED_SLICE,), prefill_plan_exact=plan_exact,
    )


def _drive(gen, uid, *, decoding=None, budget=2000):
    """Run until ``uid`` finishes; return its schedule and prompt boundary."""
    schedule = None
    seen_decode = False
    for _ in range(budget):
        prompts, responses = gen.next()
        if decoding is not None and any(r.uid == decoding for r in responses):
            seen_decode = True
        if schedule is None and any(
            getattr(p, "end_of_prompt", False) and p.uid == uid for p in prompts
        ):
            schedule = gen.pop_prefill_slice_schedule(uid)
        if any(r.uid == uid and r.finish_reason for r in responses):
            break
    else:
        raise AssertionError("request did not finish")
    if schedule is None:
        schedule = gen.pop_prefill_slice_schedule(uid)
    return schedule, gen.pop_prompt_boundary(uid), seen_decode


def _idle(model, *, plan_exact):
    gen = _generator(model, plan_exact=plan_exact, lanes=1)
    try:
        uid = gen.insert([PROMPT], max_tokens=[3], lane_rngs=[LaneRNG(5)])[0]
        schedule, boundary, _ = _drive(gen, uid)
        return schedule, boundary, dict(gen.scheduler_stats)
    finally:
        gen.close()


def _contended(model, *, plan_exact):
    """PROMPT prefilled while another lane decodes."""
    gen = _generator(model, plan_exact=plan_exact)
    try:
        neighbour = gen.insert([SHORT], max_tokens=[400], lane_rngs=[LaneRNG(3)])[0]
        for _ in range(50):
            _, responses = gen.next()
            if any(r.uid == neighbour for r in responses):
                break
        else:
            raise AssertionError("neighbour never decoded")
        uid = gen.insert([PROMPT], max_tokens=[3], lane_rngs=[LaneRNG(5)])[0]
        schedule, boundary, seen = _drive(gen, uid, decoding=neighbour)
        assert seen, "the prompt was not prefilled beside live decode"
        return schedule, boundary, dict(gen.scheduler_stats)
    finally:
        gen.close()


def _state(boundary):
    return [np.array(a) for a in _flatten(boundary["target_cache"])]


def _flatten(cache):
    out = []
    for entry in cache:
        state = entry.state
        if isinstance(state, (list, tuple)):
            out.extend(a for a in state if isinstance(a, mx.array))
        elif isinstance(state, mx.array):
            out.append(state)
    return out


def _identical(a, b):
    return len(a) == len(b) and all(
        x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b)
    )


def test_contended_ordinary_slices_go_off_plan_without_the_law(model):
    schedule, boundary, stats = _contended(model, plan_exact=False)
    assert not schedule["planned"] and schedule["deviation_reason"] == "width"
    assert schedule["deviation_executed_rows"] < schedule["deviation_planned_rows"]
    assert stats["ordinary_prefill_plan_exact_reshaped"] == 0
    # Falsifier for the state check below: the load-sliced state differs.
    _, idle_boundary, _ = _idle(model, plan_exact=False)
    assert _state(idle_boundary)
    assert not _identical(_state(boundary), _state(idle_boundary))


def test_plan_exact_law_keeps_contended_ordinary_slices_planned(model):
    schedule, boundary, stats = _contended(model, plan_exact=True)
    assert schedule["planned"], schedule
    assert schedule["boundaries"][0] == STEP
    assert schedule["covered"] == len(PROMPT) - 1
    assert stats["ordinary_prefill_plan_exact_reshaped"] >= 1
    # The point of the law: the contended prompt state is the idle state.
    _, idle_boundary, _ = _idle(model, plan_exact=False)
    assert _identical(_state(boundary), _state(idle_boundary))


def test_plan_exact_law_is_inert_when_idle(model):
    with_law, boundary, stats = _idle(model, plan_exact=True)
    without, idle_boundary, _ = _idle(model, plan_exact=False)
    assert with_law["planned"] and with_law["boundaries"] == without["boundaries"]
    assert stats["ordinary_prefill_plan_exact_reshaped"] == 0
    assert _identical(_state(boundary), _state(idle_boundary))


def test_live_memory_cap_still_wins_over_the_law(model, monkeypatch):
    original = BatchGenerator._prompt_prefill_step
    monkeypatch.setattr(
        BatchGenerator, "_prompt_prefill_step",
        lambda self, total: min(original(self, total), 8),
    )
    schedule, _, stats = _contended(model, plan_exact=True)
    widths = [b - a for a, b in zip([0] + schedule["boundaries"], schedule["boundaries"])]
    assert max(widths) <= 8
    assert not schedule["planned"] and schedule["deviation_reason"] == "width"
    assert stats["ordinary_prefill_plan_exact_reshaped"] == 0


def test_plan_exact_law_refuses_the_mixed_forward(model, monkeypatch):
    monkeypatch.setenv("MLX2_MIXED_PREFILL_DECODE", "1")
    gen = _generator(model, plan_exact=True)
    try:
        assert gen._mixed_round_ready() is False
    finally:
        gen.close()


def test_serving_passes_the_adapter_law_to_the_ordinary_generator(monkeypatch):
    from route_harness import make_engine, patch_host, tiny_qwen38_mtp

    patch_host(monkeypatch)
    seen = []
    original = BatchGenerator.__init__

    def recording(self, *args, **kwargs):
        seen.append(kwargs.get("prefill_plan_exact"))
        original(self, *args, **kwargs)

    monkeypatch.setattr(BatchGenerator, "__init__", recording)
    tiny, vocab = tiny_qwen38_mtp()
    for extra, expected in (({"prefill_plan_exact": True}, True), ({}, False)):
        engine = make_engine(tiny, vocab, mtp=False, extra=extra)
        engine.close()
        assert seen[-1] is expected


# -- B>1 ordinary prompt batching ---------------------------------------------


OTHER = [(3 * i + 1) % 60 + 2 for i in range(len(PROMPT))]


def _together(model, prompts, *, plan_exact):
    """Insert ``prompts`` at once into a two-slot prefill batch."""
    gen = _generator(model, plan_exact=plan_exact, lanes=2, prefill_batch=2)
    try:
        uids = gen.insert(
            prompts, max_tokens=[3] * len(prompts),
            lane_rngs=[LaneRNG(5 + i) for i in range(len(prompts))],
        )
        schedules, boundaries = {}, {}
        for _ in range(500):
            batch_prompts, _ = gen.next()
            for p in batch_prompts:
                if getattr(p, "end_of_prompt", False) and p.uid not in schedules:
                    schedules[p.uid] = gen.pop_prefill_slice_schedule(p.uid)
            for uid in uids:
                if uid not in boundaries:
                    boundary = gen.pop_prompt_boundary(uid)
                    if boundary is not None:
                        boundaries[uid] = boundary
            if len(boundaries) == len(uids):
                break
        else:
            raise AssertionError("prompts did not reach their boundaries")
        return (
            [schedules[u] for u in uids], [boundaries[u] for u in uids],
            dict(gen.scheduler_stats),
        )
    finally:
        gen.close()


def test_batched_prompt_forward_changes_bits(model):
    _, boundaries, stats = _together(model, [PROMPT, OTHER], plan_exact=False)
    assert stats["prefill_rounds"] == 5  # both rows in every forward
    # Full-width rows at planned widths, yet not the B=1 state (CPU).
    _, idle_boundary, _ = _idle(model, plan_exact=False)
    assert not _identical(_state(boundaries[0]), _state(idle_boundary))


def test_plan_exact_law_prefills_one_prompt_per_forward(model):
    schedules, boundaries, stats = _together(model, [PROMPT, OTHER], plan_exact=True)
    assert stats["prefill_rounds"] == 10  # deferred, not batched
    assert all(s["planned"] for s in schedules), schedules
    _, idle_boundary, _ = _idle(model, plan_exact=False)
    assert _identical(_state(boundaries[0]), _state(idle_boundary))


def test_plan_exact_law_refuses_one_chunk_overflow_admission(model):
    gen = _generator(model, plan_exact=True, lanes=4, prefill_batch=2)
    try:
        assert gen.prefill_batch_size == 1
        assert gen._admit_one_chunk_overflow(STEP) is False
    finally:
        gen.close()
