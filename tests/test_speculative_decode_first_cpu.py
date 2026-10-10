"""Exercise the shared phase contract through both speculative executors."""

import mlx.core as mx
import numpy as np
import pytest
from test_decode_first_publish import tiny_model
from test_external_dflash2_cpu import tiny

from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.pld import PromptLookupBatchGenerator
from mlx2.runtime.sample_utils import LaneRNG


def make(kind, policy):
    if kind == "external":
        model, draft = tiny()
        return ExternalDraftBatchGenerator(
            model,
            draft_model=draft,
            binding="phase-test",
            num_draft=2,
            prefill_step_size=3,
            decode_first=policy,
            ready_drain="all",
        )
    return PromptLookupBatchGenerator(
        tiny_model(),
        prefill_step_size=3,
        decode_first=policy,
        prompt_lookup={"num_draft": 2, "ngram_min": 2, "ngram_max": 2},
    )


def insert(batch, prompt, maximum=12, seed=17):
    return batch.insert([prompt], max_tokens=[maximum], lane_rngs=[LaneRNG(seed)])[0]


def warm(batch):
    uid = insert(batch, [1, 2, 3], maximum=60)
    # Stop before a publication is pending; isolate the new contention round.
    while batch.lanes[uid].anchor is None:
        batch.next()
    pending = batch._decode_first_pending
    if pending is not None:
        from mlx2.runtime.round_phases import drive_round

        batch._decode_first_pending = None
        drive_round(pending[0])
    return uid


@pytest.mark.parametrize("kind", ["pld", "external"])
def test_publishes_before_prefill_and_rechecks_cancelled_candidates(kind, monkeypatch):
    batch = make(kind, {"shared_prefill_budget": False})
    try:
        active = warm(batch)
        waiting = insert(batch, [4, 5, 6] * 8)
        events = []
        original = batch._prefill

        def prefill(lane, **kwargs):
            events.append(lane.uid)
            return original(lane, **kwargs)

        monkeypatch.setattr(batch, "_prefill", prefill)
        _, responses = batch.next()
        assert any(r.uid == active for r in responses)
        assert not events
        assert batch._decode_first_pending is not None
        batch.remove([waiting])
        batch.next()
        assert waiting not in events
        assert waiting not in batch.lanes
    finally:
        batch.close()
    assert batch._decode_first_pending is None


@pytest.mark.parametrize("kind", ["pld", "external"])
def test_new_arrival_does_not_join_a_suspended_prefill_snapshot(kind, monkeypatch):
    batch = make(kind, True)
    try:
        warm(batch)
        batch.next()
        late = insert(batch, [7, 8, 9] * 6)
        calls = []
        original = batch._prefill

        def prefill(lane, **kwargs):
            calls.append(lane.uid)
            return original(lane, **kwargs)

        monkeypatch.setattr(batch, "_prefill", prefill)
        batch.next()
        assert late not in calls
        batch.next()
        assert late in calls
    finally:
        batch.close()


@pytest.mark.parametrize("kind", ["pld", "external"])
@pytest.mark.parametrize("kill", [False, True])
def test_pending_phase_kill_switch_and_budget(kind, kill, monkeypatch):
    batch = make(kind, {"prefill_token_budget": 2})
    try:
        warm(batch)
        waiting = insert(batch, [4, 5, 6] * 10)
        batch.next()
        before = len(batch.lanes[waiting].remaining)
        if kill:
            monkeypatch.setenv("MLX2_DECODE_FIRST", "off")
        batch.next()
        consumed = before - len(batch.lanes[waiting].remaining)
        if kill:
            assert batch._decode_first_pending is None
            assert batch.decode_first.counters["kill_switch_rounds"] >= 1
        else:
            assert consumed == 2
            assert batch.decode_first.counters["budget_split_rounds"] >= 1
    finally:
        batch.close()


def snapshot(value):
    if isinstance(value, mx.array):
        return np.asarray(value).copy()
    if isinstance(value, (tuple, list)):
        return [snapshot(x) for x in value]
    return value


def equal(a, b):
    if isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("kind", ["pld", "external"])
@pytest.mark.parametrize("maximum", [1, 17])
def test_single_lane_tokens_rng_and_terminal_state_match_reference(kind, maximum):
    def run(policy):
        batch = make(kind, policy)
        try:
            uid = insert(batch, [1, 2, 3, 1, 2, 3], maximum)
            lane = batch.lanes[uid]
            tokens, boundaries, final = [], [], None
            for _ in range(100):
                prompts, responses = batch.next()
                boundaries.extend(p.uid for p in prompts if p.end_of_prompt)
                for response in responses:
                    tokens.append(response.token)
                    if response.finish_reason:
                        final = response
                if final is not None:
                    break
            assert final is not None
            assert boundaries == [uid]
            state = [snapshot(c.state) for c in final.prompt_cache]
            rng = lane.rng.snapshot() if kind == "external" else None
            return tokens, state, rng
        finally:
            batch.close()

    expected = run(False)
    actual = run({"shared_prefill_budget": False})
    equal(expected, actual)


@pytest.mark.parametrize("kind", ["pld", "external"])
def test_uneven_lane_completion_and_cancel_leave_peer_outputs_and_state_exact(kind):
    def run(policy):
        batch = make(kind, policy)
        try:
            uids = batch.insert(
                [[1, 2, 3], [4, 5, 6] * 4, [7, 8, 9] * 2],
                max_tokens=[3, 12, 11],
                lane_rngs=[LaneRNG(i) for i in range(3)],
            )
            lanes = dict(batch.lanes)
            outputs = {uid: [] for uid in uids}
            boundaries = []
            states = {}
            cancelled = False
            for _ in range(100):
                prompts, responses = batch.next()
                boundaries.extend(p.uid for p in prompts if p.end_of_prompt)
                for response in responses:
                    outputs[response.uid].append(response.token)
                    if response.finish_reason:
                        assert response.uid not in states
                        states[response.uid] = [
                            snapshot(c.state) for c in response.prompt_cache
                        ]
                if outputs[uids[0]] and not cancelled:
                    batch.remove([uids[1]])
                    cancelled = True
                if len(states) == 2:
                    break
            assert set(states) == {uids[0], uids[2]}
            assert outputs[uids[1]] == []
            assert sorted(boundaries) == [uids[0], uids[2]]
            assert not batch.lanes
            return (
                outputs,
                [states[uid] for uid in (uids[0], uids[2])],
                [
                    lanes[uid].rng.snapshot() if kind == "external" else None
                    for uid in (uids[0], uids[2])
                ],
            )
        finally:
            batch.close()

    expected, actual = run(False), run({"shared_prefill_budget": False})
    assert expected[0] == actual[0]
    equal(expected[1:], actual[1:])


def test_external_packed_prefill_respects_shared_padded_token_budget():
    from contextlib import nullcontext

    from test_external_dflash2_cpu import drain, generator

    model, draft = tiny()
    shapes = []

    def context(lengths, *, width):
        shapes.append((len(lengths), width))
        return nullcontext()

    model.prefill_row_context = context
    batch = generator(
        model,
        draft,
        external_varlen_prefill=True,
        decode_first={"prefill_token_budget": 4},
    )
    try:
        ids = batch.insert([[1, 2, 3] * 4, [4, 5, 6] * 3], max_tokens=[4, 4])
        output, _ = drain(batch)
        assert all(len(output[uid]) == 4 for uid in ids)
        assert shapes and all(rows * width <= 4 for rows, width in shapes)
        assert batch.scheduler_stats["decode_first_budget_split_rounds"] > 0
    finally:
        batch.close()


def test_external_memory_stall_survives_the_publication_boundary(monkeypatch):
    batch = make("external", True)
    try:
        warm(batch)
        waiting = insert(batch, [4, 5, 6] * 8)
        batch.next()
        original = batch._admit
        monkeypatch.setattr(
            batch,
            "_admit",
            lambda *args, **kwargs: (
                False if kwargs.get("prefill") else original(*args, **kwargs)
            ),
        )
        monkeypatch.setattr(batch, "_reclaim_for_admission", lambda: False)
        _, responses = batch.next()
        assert responses  # decode continues while prefill is memory-blocked
        assert waiting in batch._memory_waiting_uids
        assert waiting not in batch.scheduler_waiting_uids()
        monkeypatch.setattr(batch, "_admit", original)
        batch.next()
        assert waiting not in batch._memory_waiting_uids
        assert waiting in batch.scheduler_waiting_uids()
    finally:
        batch.close()


def test_external_one_token_drain_stays_one_per_lane_per_poll(monkeypatch):
    from types import SimpleNamespace

    batch = make("external", True)
    try:
        active = warm(batch)
        batch.ready_drain = "one"
        monkeypatch.setattr(batch, "_decode_phase", lambda ordered: None)
        batch.lanes[active].ready.extend(
            SimpleNamespace(uid=active, token=token, finish_reason=None)
            for token in (21, 22, 23, 24)
        )
        for token in (21, 22, 23):
            _, responses = batch.next()
            assert [r.token for r in responses] == [token]
    finally:
        batch.close()


def test_external_terminal_boundary_from_resumed_prefill_is_not_lost():
    batch = make("external", {"shared_prefill_budget": False})
    try:
        warm(batch)
        waiting = insert(batch, [4, 5, 6], maximum=1)
        batch.next()  # publish the active lane, leaving the new prefill pending
        prompts, responses = batch.next()
        assert [p.uid for p in prompts if p.end_of_prompt] == [waiting]
        terminal = [r for r in responses if r.uid == waiting]
        assert len(terminal) == 1 and terminal[0].finish_reason
        assert batch.pop_prompt_boundary(waiting) is not None
        assert batch.pop_prompt_boundary(waiting) is None
    finally:
        batch.close()
