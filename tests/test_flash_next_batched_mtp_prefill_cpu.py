"""Opt-in batched cold self-MTP prefill (FlashNextPolicy.mtp_batched_prefill_lanes).

GPU 2026-10-28 (Flash-Next, 4-lane server): cold width-4 bursts prepared lanes
one per round (16K x4 per-lane TTFT 17-52 s).  The batched path runs one
planned slice for several queued lanes at the same covered offset in one
trunk + draft-head forward.  It is not bit-identical to B=1 prefill (row
variant kernels; this tiny model differs at ~1e-7 on CPU), so each batched
lane's slice schedule records the ``batched_lanes`` deviation and APCv2
refuses that state.  CPU only, on the tiny hybrid model the slice-schedule
tests use.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2 import prometheus
from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.runtime.generate import BatchGenerator
from mlx2.runtime.hybrid_speculative import (
    BatchedMTPPrefillRefused,
    advance_self_mtp_prefill,
    advance_self_mtp_prefill_batched,
)
from mlx2.runtime.models.qwen4_exp import QSAKVCache
from mlx2.runtime.prefill_slice_schedule import changed_at
from mlx2.runtime.sample_utils import LaneRNG
from mlx2.serving import checkpoint_slice_schedule_changed
from test_batched_mtp import _tiny_qwen4_model

STEP = 32


@pytest.fixture(scope="module")
def model():
    mx.random.seed(41)
    return _tiny_qwen4_model()


def _prompt(length, salt):
    return [(5 * i + 3 + 7 * salt) % 60 + 2 for i in range(length)]


def _arrays(cache):
    values = []
    stack = [[c.state for c in cache]]
    while stack:
        value = stack.pop()
        if isinstance(value, mx.array):
            values.append(value)
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
    return values


def _serial(model, prompt, slices):
    tokens = mx.array(prompt, dtype=mx.uint32)
    (cache, state) = (None, None)
    for _ in range(slices):
        (tokens, cache, state, _n) = advance_self_mtp_prefill(
            tokens, model, prompt_cache=cache, mtp_state=state, max_tokens=STEP
        )
    return (tokens, cache, state)


def test_batched_slices_track_per_lane_state_within_float_noise(model):
    prompts = [_prompt(70, 0), _prompt(90, 1), _prompt(70, 2)]
    serial = [_serial(model, p, 2) for p in prompts]
    tokens = [mx.array(p, dtype=mx.uint32) for p in prompts]
    caches = [None] * 3
    states = [None] * 3
    for _ in range(2):
        results = advance_self_mtp_prefill_batched(
            tokens, model, prompt_caches=caches, mtp_states=states, max_tokens=STEP
        )
        tokens = [r[0] for r in results]
        caches = [r[1] for r in results]
        states = [r[2] for r in results]
        assert all(r[3] == STEP for r in results)
    for (s_tokens, s_cache, s_state), b_tokens, b_cache, b_state in zip(
        serial, tokens, caches, states
    ):
        assert s_tokens.tolist() == b_tokens.tolist()
        assert [type(c) for c in s_cache] == [type(c) for c in b_cache]
        assert [getattr(c, "offset", None) for c in b_cache] == [
            getattr(c, "offset", None) for c in s_cache
        ]
        for left, right in (
            (_arrays(s_cache), _arrays(b_cache)),
            (_arrays(s_state[0]), _arrays(b_state[0])),
            ([s_state[1]], [b_state[1]]),
        ):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                assert a.ndim == b.ndim
                # Step-grown KV buffers may be preallocated past the offset;
                # compare the live region (offsets were checked above).
                live = tuple(slice(0, min(x, y)) for x, y in zip(a.shape, b.shape))
                (a, b) = (a[live], b[live])
                # Close, not equal: the batched law is a distinct numerical
                # law, which is why the scheduler marks it off-plan.
                assert mx.allclose(
                    a.astype(mx.float32), b.astype(mx.float32), atol=1e-4
                ).item()


def test_refuses_lanes_at_different_offsets(model):
    (_, cache, state) = _serial(model, _prompt(70, 0), 1)
    with pytest.raises(BatchedMTPPrefillRefused, match="offsets"):
        advance_self_mtp_prefill_batched(
            [mx.array(_prompt(70, 0)[STEP:], dtype=mx.uint32),
             mx.array(_prompt(70, 1), dtype=mx.uint32)],
            model, prompt_caches=[cache, None], mtp_states=[state, None],
            max_tokens=STEP,
        )
    # Refusal precedes any forward: the warm lane's cache is untouched.
    assert cache[-1].offset == STEP


def test_refuses_a_slice_reaching_a_lane_final_token(model):
    with pytest.raises(BatchedMTPPrefillRefused, match="full slice"):
        advance_self_mtp_prefill_batched(
            [mx.array(_prompt(70, 0), dtype=mx.uint32),
             mx.array(_prompt(STEP, 1), dtype=mx.uint32)],
            model, prompt_caches=[None, None], mtp_states=[None, None],
            max_tokens=STEP,
        )


def test_refuses_a_single_lane(model):
    with pytest.raises(BatchedMTPPrefillRefused):
        advance_self_mtp_prefill_batched(
            [mx.array(_prompt(70, 0), dtype=mx.uint32)],
            model, prompt_caches=[None], mtp_states=[None], max_tokens=STEP,
        )


def test_fails_closed_for_a_cache_type_that_cannot_merge(model, monkeypatch):
    monkeypatch.setattr(QSAKVCache, "merge", None)
    with pytest.raises(BatchedMTPPrefillRefused, match="cannot merge"):
        advance_self_mtp_prefill_batched(
            [mx.array(_prompt(70, 0), dtype=mx.uint32),
             mx.array(_prompt(70, 1), dtype=mx.uint32)],
            model, prompt_caches=[None, None], mtp_states=[None, None],
            max_tokens=STEP,
        )


def _serve(model, prompts, *, lanes, max_tokens=3, monkeypatch=None, refuse=False,
           segmented=False):
    self_mtp = {"num_draft": 2, "persistent": True, "prefill_plan_exact": True}
    if segmented:
        # The Flash-Next route's live-tip shape (FlashNextPolicy.batch_config).
        self_mtp["segment_aware_live_tip"] = True
        self_mtp["segment_aware_cohort_size"] = len(prompts)
    if lanes:
        self_mtp["batched_prefill_lanes"] = lanes
        self_mtp["batched_prefill_max_rows"] = 4096
    if refuse:
        import mlx2.runtime.hybrid_speculative as hs

        def refuse_all(*args, **kwargs):
            raise BatchedMTPPrefillRefused("test refusal")

        monkeypatch.setattr(hs, "advance_self_mtp_prefill_batched", refuse_all)
    gen = BatchGenerator(
        model, prefill_step_size=STEP, completion_batch_size=len(prompts),
        prefill_batch_size=len(prompts), adaptive_prefill=True,
        self_mtp=self_mtp,
    )
    try:
        uids = gen.insert(
            prompts, max_tokens=[max_tokens] * len(prompts),
            lane_rngs=[LaneRNG(5 + i) for i in range(len(prompts))],
            mtp_states=[None] * len(prompts),
            self_mtp_configs=[{"sampling_temp": 0.0, "num_draft": 2}] * len(prompts),
        )
        schedules = {}
        receipts = {}
        tokens = {uid: [] for uid in uids}
        finished = set()
        for _ in range(2000):
            prompt_responses, responses = gen.next()
            for p in prompt_responses:
                if getattr(p, "end_of_prompt", False) and p.uid not in schedules:
                    schedules[p.uid] = gen.pop_prefill_slice_schedule(p.uid)
            for r in responses:
                tokens[r.uid].append(int(r.token))
                if r.finish_reason:
                    finished.add(r.uid)
                    receipts[r.uid] = getattr(r, "mtp_receipt", None)
            if finished == set(uids):
                break
        else:
            raise AssertionError("requests did not finish")
        for uid in uids:
            if schedules.get(uid) is None:
                schedules[uid] = gen.pop_prefill_slice_schedule(uid)
        return uids, schedules, receipts, tokens, dict(gen.scheduler_stats)
    finally:
        gen.close()


def test_default_off_prefills_one_lane_per_round_on_plan(model):
    prompts = [_prompt(150, i) for i in range(3)]
    uids, schedules, _, tokens, stats = _serve(model, prompts, lanes=0)
    assert stats.get("mtp_batched_prefill_rounds", 0) == 0
    assert all(schedules[uid]["planned"] for uid in uids)
    assert all(len(tokens[uid]) == 3 for uid in uids)


@pytest.mark.parametrize("segmented", [False, True])
def test_batched_cold_burst_runs_planned_slices_together_and_marks_them(
    model, segmented
):
    prompts = [_prompt(150, i) for i in range(3)]
    uids, schedules, receipts, tokens, stats = _serve(
        model, prompts, lanes=4, segmented=segmented
    )
    assert stats["mtp_batched_prefill_rounds"] >= 1
    assert stats["mtp_batched_prefill_max_width"] == 3
    assert stats["mtp_batched_prefill_lanes"] >= 3
    for uid in uids:
        schedule = schedules[uid]
        # Every executed slice is still the idle plan's slice...
        assert schedule["boundaries"][:4] == [32, 64, 96, 128], schedule
        assert schedule["covered"] == 149
        # ...but the batched law is recorded, so APCv2 refuses that state.
        assert not schedule["planned"]
        assert schedule["deviation_reason"] == "batched_lanes"
        assert schedule["first_deviation"] == 0
        assert changed_at(schedule, 64) and not changed_at(schedule, 0)
        # The serving publication gate refuses the prompt-end state.
        owner = SimpleNamespace(prefill_slice_schedule=schedule, uid=None)
        assert checkpoint_slice_schedule_changed(None, owner, prompts[0][:149])
        assert len(tokens[uid]) == 3
        evidence = receipts[uid]["batched_prefill"]
        assert evidence["observed_used"] and evidence["qualified"] is False
        assert evidence["exact_vs_b1"] is False
        assert evidence["max_width"] == 3


def test_unequal_slices_stay_serial_and_on_plan(model):
    # Two lanes share the first planned slices; the third starts warm-free
    # but its single short slice differs, so it is prefilled alone.
    prompts = [_prompt(150, 0), _prompt(150, 1), _prompt(20, 2)]
    uids, schedules, _, _, stats = _serve(model, prompts, lanes=4)
    assert stats["mtp_batched_prefill_max_width"] == 2
    assert schedules[uids[2]]["planned"], schedules[uids[2]]


def test_equal_one_chunk_residuals_are_teacher_forced_together(model):
    prompts = [_prompt(20, i) for i in range(3)]
    uids, schedules, receipts, tokens, stats = _serve(model, prompts, lanes=4)
    assert stats["mtp_batched_prefill_one_chunk_lanes"] == 3
    assert stats["mtp_batched_prefill_max_width"] == 3
    for uid in uids:
        assert schedules[uid]["boundaries"] == [19]
        assert schedules[uid]["deviation_reason"] == "batched_lanes"
        assert receipts[uid]["batched_prefill"]["slices"] == 1
        assert len(tokens[uid]) == 3


def test_model_refusal_falls_back_to_serial_prefill(model, monkeypatch):
    prompts = [_prompt(150, i) for i in range(2)]
    uids, schedules, receipts, tokens, stats = _serve(
        model, prompts, lanes=4, monkeypatch=monkeypatch, refuse=True
    )
    assert stats["mtp_batched_prefill_refused"] >= 1
    assert stats.get("mtp_batched_prefill_rounds", 0) == 0
    for uid in uids:
        assert schedules[uid]["planned"], schedules[uid]
        assert receipts[uid]["batched_prefill"] is None
        assert len(tokens[uid]) == 3


def test_flash_next_policy_declares_the_opt_in():
    policy = FlashNextPolicy()
    assert policy.mtp_batched_prefill_lanes == 0
    assert "mtp_batched_prefill_lanes" not in policy.as_dict()
    assert "mtp_batched_prefill_max_rows" not in policy.as_dict()
    config = policy.batch_config(max_lanes=4, prefill_step=8192)
    assert "batched_prefill_lanes" not in config
    on = FlashNextPolicy(mtp_batched_prefill_lanes=4)
    assert on.as_dict()["mtp_batched_prefill_lanes"] == 4
    config = on.batch_config(max_lanes=4, prefill_step=8192)
    assert config["batched_prefill_lanes"] == 4
    assert config["batched_prefill_max_rows"] == 16384
    for bad in (1, -1, True, 2.0):
        with pytest.raises(ValueError):
            FlashNextPolicy(mtp_batched_prefill_lanes=bad)
    with pytest.raises(ValueError):
        FlashNextPolicy(mtp_batched_prefill_lanes=2, mtp_batched_prefill_max_rows=1)


def test_counters_are_exported():
    for key in (
        "mtp_batched_prefill_rounds",
        "mtp_batched_prefill_lanes",
        "mtp_batched_prefill_rows",
        "mtp_batched_prefill_one_chunk_lanes",
        "mtp_batched_prefill_refused",
        "mtp_batched_prefill_declined_contended",
    ):
        assert key in prometheus._SCHEDULER_EVENTS
    assert "mtp_batched_prefill_max_width" in prometheus._SCHEDULER_GAUGES


def test_declines_beside_live_decode(model):
    self_mtp = {
        "num_draft": 2, "persistent": True, "prefill_plan_exact": True,
        "batched_prefill_lanes": 4, "batched_prefill_max_rows": 4096,
    }
    gen = BatchGenerator(
        model, prefill_step_size=STEP, completion_batch_size=3,
        prefill_batch_size=3, adaptive_prefill=True, self_mtp=self_mtp,
    )
    config = {"sampling_temp": 0.0, "num_draft": 2}
    try:
        (first,) = gen.insert(
            [_prompt(10, 9)], max_tokens=[400], lane_rngs=[LaneRNG(1)],
            mtp_states=[None], self_mtp_configs=[config],
        )
        for _ in range(50):
            _, responses = gen.next()
            if any(r.uid == first for r in responses):
                break
        later = gen.insert(
            [_prompt(150, 1), _prompt(150, 2)], max_tokens=[2, 2],
            lane_rngs=[LaneRNG(2), LaneRNG(3)], mtp_states=[None, None],
            self_mtp_configs=[config, config],
        )
        done = set()
        for _ in range(2000):
            _, responses = gen.next()
            done.update(r.uid for r in responses if r.finish_reason)
            if set(later) <= done:
                break
        else:
            raise AssertionError("later requests did not finish")
        stats = dict(gen.scheduler_stats)
        assert stats["mtp_batched_prefill_declined_contended"] >= 1
        assert stats.get("mtp_batched_prefill_rounds", 0) == 0
    finally:
        gen.close()
