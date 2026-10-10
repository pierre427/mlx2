"""A corrupt sampled token must not enter decode history or client output."""

from collections import deque
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator, LaneFailure
from mlx2.runtime.generate import (
    GenerationBatch,
    MTPGenerationBatch,
    StopSequenceMatcher,
)
from mlx2.runtime.hybrid_speculative import (
    SegmentedSelfMTPState,
    SelfMTPCachePair,
    _sample_from_logprobs,
    advance_batched_self_mtp_zero,
)
from mlx2.runtime.pld import PromptLookupBatchGenerator, PromptLookupLaneFailure


@pytest.fixture(autouse=True)
def _cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


class _TwoLaneModel:
    def __init__(self, invalid_logits=False):
        self.invalid_logits = invalid_logits
        self.inputs = []

    def __call__(self, inputs, cache):
        del cache
        self.inputs.append(inputs.tolist())
        rows = []
        for token in inputs[:, 0].tolist():
            rows.append(
                [-float("inf")] * 3
                if self.invalid_logits and token == 8
                else [0.0, 1.0, 2.0]
            )
        return mx.array(rows, dtype=mx.float32)[:, None, :]


def _batch(model, sampler):
    return GenerationBatch(
        model,
        uids=[1, 2],
        inputs=mx.array([7, 8], dtype=mx.uint32),
        prompt_cache=[],
        tokens=[[], []],
        samplers=[],
        fallback_sampler=sampler,
        logits_processors=[],
        stop_matchers=[StopSequenceMatcher(), StopSequenceMatcher()],
        max_tokens=[2, 2],
    )


@pytest.fixture(autouse=True)
def _current_step_validity(monkeypatch):
    """These tests pin the ``current`` contract (a bad row never reaches the
    next forward).  The shipped default is ``deferred``; its contract lives in
    test_step_validity_deferred.py."""
    monkeypatch.setenv("MLX2_STEP_VALIDITY", "current")


def test_nonfinite_law_drops_only_bad_lane_before_next_forward_or_response():
    model = _TwoLaneModel(invalid_logits=True)
    batch = _batch(model, lambda rows: mx.argmax(rows, axis=-1))

    assert batch.uids == [1, 2]
    assert [response.uid for response in batch.next()] == [1]
    assert batch.uids == [1]
    assert batch.tokens == [[7, 2]]
    assert batch.take_lane_failures() == [
        {"uid": 2, "reason": "sampled token 0 has non-finite log probability"}
    ]
    assert model.inputs == [[[7], [8]], [[2]]]


def test_out_of_vocab_sampler_drops_only_bad_lane_before_history_or_response():
    model = _TwoLaneModel()
    batch = _batch(
        model,
        lambda rows: mx.array(
            [2, 0xFFFFFFFF][: rows.shape[0]], dtype=mx.uint32
        ),
    )

    assert batch.uids == [1, 2]
    assert [response.token for response in batch.next()] == [2]
    assert batch.take_lane_failures() == [
        {"uid": 2, "reason": "sampled token 4294967295 is outside vocabulary of size 3"}
    ]
    assert model.inputs == [[[7], [8]], [[2]]]


def test_self_mtp_initial_output_rejected_before_response_bookkeeping():
    batch = object.__new__(MTPGenerationBatch)
    batch._initial_outputs = [
        SimpleNamespace(token=0xFFFFFFFF, logprobs=mx.array([0.0, -1.0]))
    ]
    with pytest.raises(RuntimeError, match="outside vocabulary"):
        batch._emit_initial()
    assert batch._initial_outputs[0].token == 0xFFFFFFFF


def test_self_mtp_sampling_rejects_nonfinite_law_before_lane_mutation():
    with pytest.raises(RuntimeError, match="non-finite log probability"):
        _sample_from_logprobs(mx.array([float("nan"), float("nan")]))


def test_zero_fast_late_invalid_row_does_not_mutate_earlier_lane(monkeypatch):
    from mlx2.runtime import hybrid_speculative, segmented_self_mtp

    monkeypatch.setattr(
        hybrid_speculative, "_model_allows_segmented_true_batch", lambda model: True
    )
    monkeypatch.setattr(
        segmented_self_mtp, "true_batched_segmented_self_mtp_enabled", lambda: True
    )

    class Model:
        forwards = 0

        def mtp_backbone(self, tokens, cache):
            del cache
            self.forwards += 1
            hidden = mx.zeros((tokens.shape[0], 1, 2))
            return hidden, hidden

        def logits(self, hidden):
            del hidden
            return mx.array(
                [[[0.0, 1.0]], [[-float("inf"), -float("inf")]]]
            )

    class Transaction:
        def __init__(self):
            self.position = 0
            self.publishes = 0
            self.branch = None

        def validate(self, pair, lane, position):
            del pair, lane, position

        def fork(self, name):
            del name
            self.branch = SimpleNamespace(closed=False)
            self.branch.close = lambda: setattr(self.branch, "closed", True)
            return self.branch

        def publish(self, *args, **kwargs):
            del args, kwargs
            self.publishes += 1
            return self

    def lane(uid, cur):
        return SimpleNamespace(
            uid=uid, cur=cur, num_draft=0, max_tokens=3, ntoks=0,
            seed_h=mx.ones((1, 1, 2)), pending_hs=None, pending_ts=[],
            token_prefix=mx.array([7], dtype=mx.uint32),
            logits_processors=[], logprob_transform=None, sampling_temp=0.0,
            rng=None, copy_draft=None,
            stats=SimpleNamespace(cycles=0, draft_cycles=0, bonus_tokens=0),
        )

    model = Model()
    lanes = [lane(1, 8), lane(2, 9)]
    transactions = [Transaction(), Transaction()]
    pair = SelfMTPCachePair(target=[], draft=[])
    state = SegmentedSelfMTPState(
        lanes=lanes,
        row_caches=[pair, pair],
        transactions=transactions,
        membership_epoch=1,
    )
    state._segmented_caches = pair

    with pytest.raises(RuntimeError, match="non-finite log probability"):
        advance_batched_self_mtp_zero(model, state)

    assert model.forwards == 1  # cache forward precedes the sample guard
    assert state.poisoned and not state.proposal_open
    assert [(item.cur, item.ntoks, item.pending_ts) for item in lanes] == [
        (8, 0, []), (9, 0, [])
    ]
    assert [item.token_prefix.tolist() for item in lanes] == [[7], [7]]
    assert [item.stats.cycles for item in lanes] == [0, 0]
    assert [item.publishes for item in transactions] == [0, 0]
    assert all(item.branch.closed for item in transactions)


@pytest.mark.parametrize("emit_logprobs", [False, True])
def test_external_greedy_rejects_nonfinite_target_law(emit_logprobs):
    executor = object.__new__(ExternalDraftBatchGenerator)
    executor.mx = mx
    lane = SimpleNamespace(
        uid=17,
        processors=[],
        sampling={"sampling_temp": 0, "emit_logprobs": emit_logprobs},
    )
    response_rows = [] if emit_logprobs else None
    with pytest.raises(LaneFailure, match="not a probability distribution"):
        executor._target_law(
            lane,
            mx.array([-float("inf"), -float("inf")]),
            [1],
            response_rows=response_rows,
        )
    assert response_rows == ([] if emit_logprobs else None)


@pytest.mark.parametrize("invalid", ["nonfinite", "out_of_range"])
def test_prompt_lookup_rejects_sample_before_emitted_or_cache_commit(invalid):
    generator = object.__new__(PromptLookupBatchGenerator)
    generator.num_draft = 0
    lane = SimpleNamespace(
        uid=29,
        maximum=2,
        generated=0,
        config={},
        cost_latch=None,
        ordinary=True,
        anchor=7,
        processors=[],
        lookup_history=[],
        sampler=(
            (lambda row: mx.argmax(row, axis=-1))
            if invalid == "nonfinite"
            else (lambda row: mx.array([0xFFFFFFFF], dtype=mx.uint32))
        ),
    )
    steps = generator._round_steps(lane)
    assert next(steps) == ([7], [])
    logits = (
        mx.array([[-float("inf"), -float("inf")]])
        if invalid == "nonfinite"
        else mx.array([[0.0, 1.0]])
    )
    with pytest.raises(PromptLookupLaneFailure):
        steps.send((logits, None))
    assert lane.anchor == 7
    assert lane.generated == 0


@pytest.mark.parametrize("batched", [False, True])
def test_prompt_lookup_lane_failure_keeps_peer_and_executor_live(batched):
    generator = object.__new__(PromptLookupBatchGenerator)
    bad = SimpleNamespace(
        uid=29, anchor=7, ready=deque(), batched=batched, cost_latch=None,
        speculation_started=False, cache=[],
    )
    peer = SimpleNamespace(
        uid=30, anchor=8, ready=deque(), batched=batched, cost_latch=None,
        speculation_started=False, cache=[],
    )
    generator.lanes = {29: bad, 30: peer}
    generator.boundaries = {}
    generator._lane_failures = []
    # Bypassing __init__: the stall-bound fairness it builds is inert here.
    from mlx2.runtime.adaptive_policy import DecodeTimeFairness

    generator.decode_time_fairness = DecodeTimeFairness()
    from mlx2.runtime.round_phases import initialize
    initialize(generator, None)
    generator.scheduler_stats = {}
    if batched:
        generator._round_batched = lambda lanes: (_ for _ in ()).throw(
            PromptLookupLaneFailure(29, "invalid sampled token")
        )
    else:
        def run(lane):
            if lane.uid == 29:
                raise PromptLookupLaneFailure(29, "invalid sampled token")
        generator._round = run

    assert generator.next() == ([], [])
    assert list(generator.lanes) == [30]
    assert generator.take_lane_failures() == [
        {"uid": 29, "reason": "invalid sampled token"}
    ]
