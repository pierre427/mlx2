"""CPU falsifiers for output-budget work and terminal cache ownership."""

from unittest.mock import patch

import mlx.core as mx
import pytest
from test_segmented_mtp import _detached

from mlx2.runtime.generate import (
    GenerationBatch,
    MTPGenerationBatch,
    StopSequenceMatcher,
)
from mlx2.runtime.hybrid_speculative import MTPToken, SelfMTPCycleResult


@pytest.fixture(autouse=True)
def _cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


class _LaneCache:
    def __init__(self, offset):
        self.offset = int(offset)


class _BatchCache:
    def __init__(self, offsets):
        self.offsets = list(offsets)

    @property
    def state(self):
        return mx.array(self.offsets, dtype=mx.int32)

    def extract(self, index):
        return _LaneCache(self.offsets[index])

    def filter(self, keep):
        self.offsets = [self.offsets[index] for index in keep]


class _CountingModel:
    def __init__(self):
        self.forward_widths = []
        self.mixed_widths = []

    @staticmethod
    def _logits(width):
        rows = mx.zeros((width, 1, 8), dtype=mx.float32)
        rows[:, :, 3] = 1
        return rows

    def __call__(self, inputs, cache):
        self.forward_widths.append(int(inputs.shape[0]))
        for plane in cache:
            plane.offsets = [offset + 1 for offset in plane.offsets]
        return self._logits(inputs.shape[0])

    def mixed_forward(self, rows):
        (prompt_tokens, prompt_cache), (decode_tokens, decode_cache) = rows
        self.mixed_widths.append((prompt_tokens.shape[0], decode_tokens.shape[0]))
        for plane in prompt_cache:
            plane.offsets = [offset + prompt_tokens.shape[1] for offset in plane.offsets]
        for plane in decode_cache:
            plane.offsets = [offset + 1 for offset in plane.offsets]
        return (self._logits(prompt_tokens.shape[0]), self._logits(decode_tokens.shape[0]))

    def logits(self, hidden):
        return hidden


def _batch(model, maximums):
    width = len(maximums)
    return GenerationBatch(
        model,
        uids=list(range(width)),
        inputs=mx.array([7] * width, dtype=mx.uint32),
        prompt_cache=[_BatchCache([2] * width)],
        tokens=[[1, 2] for _ in maximums],
        samplers=[],
        fallback_sampler=lambda rows: mx.argmax(rows, axis=-1),
        logits_processors=[],
        stop_matchers=[StopSequenceMatcher() for _ in maximums],
        max_tokens=list(maximums),
    )


def test_ordinary_final_pending_output_does_not_launch_discard_forward():
    model = _CountingModel()
    batch = _batch(model, [2])

    first = batch.next()
    final = batch.next()

    assert model.forward_widths == [1, 1]
    assert [response.finish_reason for response in first + final] == [None, "length"]
    assert final[0].token == 3
    assert final[0].prompt_cache[0].offset == len(final[0].all_tokens) == 4
    assert final[0].all_tokens + [final[0].token] == [1, 2, 7, 3, 3]


def test_ragged_budget_terminal_row_is_removed_before_survivor_forward():
    model = _CountingModel()
    batch = _batch(model, [1, 3])

    first = batch.next()
    second = batch.next()
    third = batch.next()

    assert model.forward_widths == [2, 1, 1]
    assert [(r.uid, r.finish_reason) for r in first] == [(0, "length"), (1, None)]
    assert [(r.uid, r.finish_reason) for r in second] == [(1, None)]
    assert [(r.uid, r.finish_reason) for r in third] == [(1, "length")]
    for response in (first[0], third[0]):
        assert response.prompt_cache[0].offset == len(response.all_tokens)


def test_budget_terminal_lane_does_not_consume_pending_mixed_prompt_slice():
    model = _CountingModel()
    batch = _batch(model, [1])
    prompt_cache = [_BatchCache([2])]
    batch._mixed_segment = (mx.array([[4, 5]], dtype=mx.uint32), prompt_cache)

    (final,) = batch.next()

    assert final.finish_reason == "length"
    assert model.forward_widths == [1]
    assert model.mixed_widths == []
    assert batch._mixed_segment is not None
    assert prompt_cache[0].offsets == [2]


def test_mixed_ragged_round_detaches_terminal_lane_before_shared_forward():
    model = _CountingModel()
    batch = _batch(model, [1, 2])
    prompt_cache = [_BatchCache([2])]
    batch._mixed_segment = (mx.array([[4, 5]], dtype=mx.uint32), prompt_cache)

    first = batch.next()
    (final,) = batch.next()

    assert model.forward_widths == [2]
    assert model.mixed_widths == [(1, 1)]
    assert [(r.uid, r.finish_reason) for r in first] == [
        (0, "length"),
        (1, None),
    ]
    assert (final.uid, final.finish_reason) == (1, "length")
    assert first[0].prompt_cache[0].offset == len(first[0].all_tokens) == 3
    assert final.prompt_cache[0].offset == len(final.all_tokens) == 4
    assert prompt_cache[0].offsets == [4]


def test_native_mtp_terminal_commit_never_schedules_a_followup_proposal():
    detached = _detached(9)
    detached.lane.max_tokens = 2
    calls = []
    batch = MTPGenerationBatch(
        object(),
        [detached],
        [MTPToken(3, mx.zeros((8,)), False)],
        [StopSequenceMatcher()],
        segmented_live_tip=True,
    )

    def propose(_model, state):
        calls.append(tuple(lane.uid for lane in state.lanes))
        proposal = SelfMTPCycleResult(
            membership_epoch=state.membership_epoch,
            lane_uids=(9,),
            draft_depths=(0,),
            accepted_lengths=(0,),
            target_drops=(0,),
            head_drops=(0,),
            outputs=((MTPToken(4, mx.zeros((8,)), False),),),
        )
        state.proposal_open = True
        state._open_proposal = proposal
        return proposal

    def commit(state, _proposal, *, emitted_counts, terminal):
        assert emitted_counts == [1]
        assert terminal == [True]
        state.proposal_open = False
        state._open_proposal = None
        state.lanes[0].ntoks += 1

    with (
        patch(
            "mlx2.runtime.hybrid_speculative.advance_batched_self_mtp_zero",
            side_effect=propose,
        ),
        patch(
            "mlx2.runtime.hybrid_speculative.commit_batched_self_mtp",
            side_effect=commit,
        ),
    ):
        assert batch.next()[0].finish_reason is None
        (final,) = batch.next()
        assert final.finish_reason == "length"
        assert batch.next() == []

    assert calls == [(9,)]
    assert final.prompt_cache is not None
    assert final.all_tokens is not None
    batch.close()
