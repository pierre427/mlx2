# Adapted from unified tests/test_segmented_self_mtp.py at 1e2bc604, Apache-2.0.
from itertools import product
from unittest.mock import patch

import mlx.core as mx
import pytest

from mlx2.runtime.hybrid_speculative import (
    DetachedSelfMTPLane,
    HybridStats,
    MTPToken,
    SelfMTPCachePair,
    SelfMTPCycleResult,
    SelfMTPLane,
    abort_batched_self_mtp,
    attach_segmented_self_mtp_lanes,
    close_segmented_self_mtp_state,
    commit_batched_self_mtp,
    detach_self_mtp_lanes,
    propose_batched_self_mtp,
)
from mlx2.runtime.segmented_self_mtp import (
    SegmentedLaneTransaction,
    require_segmented_self_mtp_engagement,
    require_true_batched_segmented_self_mtp_engagement,
    segmented_self_mtp_enabled,
    segmented_self_mtp_stats,
    shared_qsa_suffix_admission,
)
from mlx2.runtime.models.qwen4_exp import QSAKVCache
from mlx2.runtime.qsa_shared_suffix import SharedSuffixQSAKVCache
from mlx2.structured_output import StructuredOutputProcessor, compile_constraint


class _ABTokenizer:
    eos_token_ids = [0]
    vocab_size = 64

    def decode(self, tokens, *, skip_special_tokens=False, **_kwargs):
        pieces = []
        for token in tokens:
            token = int(token)
            if token == 0 and skip_special_tokens:
                continue
            pieces.append({0: "", 10: "a", 11: "b"}.get(token, "x"))
        return "".join(pieces)


def _tree_arrays(value):
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tree_arrays(item)


def _force_invalid_mtp_logits(model):
    original = model.mtp_step

    def forced(hidden, tokens, cache):
        (_logits, post) = original(hidden, tokens, cache)
        invalid = mx.arange(64).astype(post.dtype)
        return (mx.broadcast_to(invalid, (*post.shape[:2], 64)), post)

    model.mtp_step = forced
    return original


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def test_true_batched_segmented_mtp_constrains_each_draft_prefix(monkeypatch):
    from test_batched_mtp import _prepare_lane, _tiny_qwen4_model

    monkeypatch.setenv("MLX_LM_TRUE_BATCHED_SEGMENTED_MTP", "1")
    model = _tiny_qwen4_model()
    prompts = ([1, 2, 3, 4], [5, 6, 7, 8, 9])
    rows = [_prepare_lane(model, uid, prompt) for uid, prompt in enumerate(prompts)]
    processors = []
    for detached, prompt in zip(rows, prompts):
        processor = StructuredOutputProcessor(
            _ABTokenizer(), len(prompt), compile_constraint(grammar="ab")
        )
        processors.append(processor)
        detached.lane.cur = 10
        detached.lane.num_draft = 2
        detached.lane.sampling_temp = 0.0
        detached.lane.logprob_transform = None
        detached.lane.logits_processors = [processor]

    original_mtp_step = _force_invalid_mtp_logits(model)
    state = attach_segmented_self_mtp_lanes(model, None, rows)
    try:
        proposal = propose_batched_self_mtp(model, state)
        assert state._batched_state is not None
        assert proposal._drafts == ((11, 0), (11, 0))
        # Every row's processor saw both constrained draft prefixes.
        # (The scanner memoizes per prefix; the automaton engine tracks the
        # decoded text it advanced through, one mask per constrained step.)
        seen = lambda processor, prefix: (
            prefix in processor._allowed_cache
            or prefix in processor._partial_cache
            or (
                processor.engine == "automaton"
                and processor._track_text.startswith(prefix)
                and processor.constrained_steps >= len(prefix) + 1
            )
        )
        assert all(seen(processor, "a") for processor in processors)
        assert all(seen(processor, "ab") for processor in processors)
        abort_batched_self_mtp(state, proposal)
    finally:
        model.mtp_step = original_mtp_step
        close_segmented_self_mtp_state(state)


def test_segmented_mtp_invalid_structured_prefix_fails_closed(monkeypatch):
    from test_batched_mtp import _prepare_lane, _tiny_qwen4_model

    monkeypatch.setenv("MLX_LM_TRUE_BATCHED_SEGMENTED_MTP", "1")
    model = _tiny_qwen4_model()
    detached = _prepare_lane(model, 0, [1, 2, 3, 4])
    detached.lane.cur = 63
    detached.lane.sampling_temp = 0.0
    detached.lane.logprob_transform = None
    processor = StructuredOutputProcessor(
        _ABTokenizer(), 4, compile_constraint(grammar="ab")
    )
    detached.lane.logits_processors = [processor]
    state = attach_segmented_self_mtp_lanes(model, None, [detached])
    try:
        # A grammar dead end is one lane's failure.  The processor records it
        # and stops masking so the shared segmented proposal stays consistent
        # for the other rows; the serving layer fails that lane closed (502).
        # Raising here used to poison the state and kill every inflight lane.
        propose_batched_self_mtp(model, state)
        assert processor.failure is not None
        assert "no valid token continuation" in processor.failure
        assert state.poisoned is False
    finally:
        close_segmented_self_mtp_state(state)


class _FakeCache:
    def __init__(self, offset, nbytes=32):
        self.offset = int(offset)
        self.nbytes = int(nbytes)
        self.speculating = False

    @property
    def state(self):
        return ()

    def is_trimmable(self):
        return self.speculating

    def supports_ragged_trim(self):
        return True

    def start_speculation(self, rollback_window=None):
        self.speculating = True

    def stop_speculation(self):
        self.speculating = False

    def trim(self, count):
        count = int(count)
        self.offset -= count
        return count

    def trim_ragged(self, counts, validate=True):
        counts = list(counts)
        if len(counts) != 1:
            raise ValueError("fake B1 cache requires exactly one row")
        self.offset -= int(counts[0])
        return counts

    def preflight_ragged_trim(self, counts, validate=True):
        counts = list(counts)
        if len(counts) != 1:
            raise ValueError("fake B1 cache requires exactly one row")
        if validate and int(counts[0]) > self.offset:
            raise ValueError("fake B1 cache trim exceeds its offset")

    def prepare(self, lengths, right_padding):
        return None

    def finalize(self):
        return None


class _FakeKVCache(_FakeCache):
    keys = object()


class _FakeQSAKVCache(_FakeKVCache):
    index_keys = object()


class _FakeArraysCache(_FakeCache):
    def __init__(self, offset):
        super().__init__(offset)
        self.cache = ["fake-recurrent-state"]


class _FakeMTPKVCache(_FakeKVCache):
    pass


class _FakeModel:
    def mtp_step(self, hidden, tokens, caches):
        width = int(tokens.shape[1])
        for cache in caches:
            cache.offset += width
        return hidden


class _FakeMatcher:
    def make_state(self):
        return None


class _CloseableCacheList(list):
    def __init__(self, values):
        super().__init__(values)
        self.closed = False

    def close(self):
        self.closed = True


def _detached(uid, position=6):
    lane = SelfMTPLane(
        uid=uid,
        cur=20 + uid,
        seed_h=mx.zeros((1, 1, 2)),
        pending_hs=None,
        pending_ts=[],
        token_prefix=mx.array([1, 2, 3], mx.uint32),
        rng=None,
        ntoks=1,
        max_tokens=8,
        num_draft=2,
        sampling_temp=0.0,
        accept_rule="residual",
        logprob_transform=None,
        logits_processors=[],
        stats=HybridStats(),
        share_qsa_indices=False,
    )
    return DetachedSelfMTPLane(
        lane,
        SelfMTPCachePair(
            target=[
                _FakeKVCache(position),
                _FakeQSAKVCache(position),
                _FakeArraysCache(position),
            ],
            draft=[_FakeMTPKVCache(position - 1)],
        ),
    )


def _real_qsa_detached(uid, position=8):
    item = _detached(uid, position=position)
    identity = {
        "format_version": 1,
        "model_config_hash": "tiny",
        "block_size": 4,
        "compress_ratio": 4,
        "producer_version": "test",
        "layer_id": "1",
        "complete_blocks": position // 4,
    }
    cache = QSAKVCache(identity)
    cache.keys = mx.arange(position * 6, dtype=mx.float32).reshape(1, 2, position, 3)
    cache.values = cache.keys + 100
    cache.index_keys = mx.arange(position * 5, dtype=mx.float32).reshape(1, position, 5)
    cache.offset = position
    cache._qsa_pooled_keys = mx.arange((position // 4) * 5, dtype=mx.float32).reshape(
        1, position // 4, 5
    )
    cache._qsa_pooled_ratio = 4
    cache._qsa_summary_identity = identity
    item.caches.target[1] = cache
    return item


def test_shared_qsa_prefix_attestation_is_host_only_and_initial_cohort_scoped():
    first = _detached(0, position=8)
    second = _detached(1, position=8)
    first.shared_qsa_prefix_id = "same-host-token-digest"
    second.shared_qsa_prefix_id = "same-host-token-digest"

    state = attach_segmented_self_mtp_lanes(_FakeModel(), None, [first, second])
    assert state.shared_qsa_prefix_id == "same-host-token-digest"

    state, detached = detach_self_mtp_lanes(_FakeModel(), state, [1])
    assert state.shared_qsa_prefix_id is None
    assert detached[0].shared_qsa_prefix_id is None
    close_segmented_self_mtp_state(state)
    detached[0].segment_transaction.close()


def test_shared_qsa_auto_policy_tracks_context_budget_crossover(monkeypatch):
    monkeypatch.delenv("MLX_LM_SHARED_QSA_SUFFIX", raising=False)
    assert shared_qsa_suffix_admission(
        base_tokens=16 * 1024 - 4, remaining_tokens=16
    ) == (True, "auto_admitted", 16)
    assert shared_qsa_suffix_admission(
        base_tokens=16 * 1024 - 4, remaining_tokens=17
    ) == (False, "output_budget_above_cutoff", 16)
    assert shared_qsa_suffix_admission(
        base_tokens=32 * 1024 - 4, remaining_tokens=32
    ) == (True, "auto_admitted", 32)
    assert shared_qsa_suffix_admission(
        base_tokens=64 * 1024 - 4, remaining_tokens=64
    ) == (True, "auto_admitted", 64)

    monkeypatch.setenv("MLX_LM_SHARED_QSA_SUFFIX", "0")
    assert (
        shared_qsa_suffix_admission(base_tokens=64 * 1024, remaining_tokens=1)[0]
        is False
    )
    monkeypatch.setenv("MLX_LM_SHARED_QSA_SUFFIX", "1")
    assert shared_qsa_suffix_admission(base_tokens=1, remaining_tokens=1000)[0] is True


def test_shared_qsa_auto_policy_is_storage_reachable_at_16k(monkeypatch):
    monkeypatch.delenv("MLX_LM_SHARED_QSA_SUFFIX", raising=False)
    monkeypatch.delenv("MLX_LM_SHARED_QSA_SUFFIX_MIN_CONTEXT", raising=False)
    monkeypatch.delenv("MLX_LM_SHARED_QSA_SUFFIX_MAX_REMAINING", raising=False)
    monkeypatch.delenv("MLX_LM_QSA_PRIVATE_DELTA", raising=False)
    monkeypatch.delenv("MLX_LM_QSA_PRIVATE_DELTA_MIN_CONTEXT", raising=False)
    monkeypatch.delenv("MLX_LM_QSA_PRIVATE_DELTA_MIN_CONTEXT_MN", raising=False)
    rows = [_real_qsa_detached(uid, position=16 * 1024 - 4) for uid in range(2)]
    for item in rows:
        item.shared_qsa_prefix_id = "auto-admitted-16k-live-tip"

    state = attach_segmented_self_mtp_lanes(_FakeModel(), None, rows)

    assert state.shared_qsa_prefix_id == "auto-admitted-16k-live-tip"
    assert all(
        isinstance(pair.target[1], SharedSuffixQSAKVCache) for pair in state.row_caches
    )
    close_segmented_self_mtp_state(state)


def test_shared_qsa_suffix_materializes_on_detach_and_later_join(monkeypatch):
    monkeypatch.setenv("MLX_LM_SHARED_QSA_SUFFIX", "1")
    monkeypatch.setenv("MLX_LM_QSA_PRIVATE_DELTA", "1")
    monkeypatch.setenv("MLX_LM_QSA_PRIVATE_DELTA_MIN_CONTEXT", "0")
    first = _real_qsa_detached(0)
    second = _real_qsa_detached(1)
    for item in (first, second):
        item.shared_qsa_prefix_id = "same-live-tip"

    state = attach_segmented_self_mtp_lanes(_FakeModel(), None, [first, second])
    left = state.row_caches[0].target[1]
    right = state.row_caches[1].target[1]
    assert isinstance(left, SharedSuffixQSAKVCache)
    assert isinstance(right, SharedSuffixQSAKVCache)
    assert left.base is right.base

    state, detached = detach_self_mtp_lanes(_FakeModel(), state, [1])
    assert type(detached[0].caches.target[1]) is QSAKVCache
    assert isinstance(state.row_caches[0].target[1], SharedSuffixQSAKVCache)

    third = _real_qsa_detached(2)
    state = attach_segmented_self_mtp_lanes(_FakeModel(), state, [third])
    assert all(type(pair.target[1]) is QSAKVCache for pair in state.row_caches)
    assert state.shared_qsa_prefix_id is None
    close_segmented_self_mtp_state(state)
    detached[0].segment_transaction.close()


def test_shared_qsa_b2_to_b1_survivor_next_forward_matches_physical(monkeypatch):
    from test_batched_mtp import _prepare_lane, _tiny_qwen4_model

    model = _tiny_qwen4_model()
    # The production shared-suffix ABI is four-token aligned; the generic
    # tiny-model fixture otherwise uses two-token QSA blocks.
    indexer = model.language_model.model.layers[1].self_attn.indexer
    indexer.compress_ratio = 4
    indexer.summary_identity["block_size"] = 4
    indexer.summary_identity["compress_ratio"] = 4
    prompts = ([1, 2, 3, 4], [1, 2, 3, 4])
    monkeypatch.setenv("MLX_LM_TRUE_BATCHED_SEGMENTED_MTP", "1")
    monkeypatch.setenv("MLX_LM_QSA_PRIVATE_DELTA", "1")
    monkeypatch.setenv("MLX_LM_QSA_PRIVATE_DELTA_MIN_CONTEXT", "0")

    def run(shared):
        monkeypatch.setenv("MLX_LM_SHARED_QSA_SUFFIX", "1" if shared else "0")
        rows = [_prepare_lane(model, uid, prompt) for uid, prompt in enumerate(prompts)]
        for item in rows:
            cache = item.caches.target[1]
            blocks = cache.offset // indexer.compress_ratio
            starts = mx.arange(blocks) * indexer.compress_ratio
            cache._qsa_pooled_keys = indexer._pool_blocks(cache.index_keys, starts)
            cache._qsa_pooled_ratio = indexer.compress_ratio
            cache._qsa_summary_identity = dict(indexer.summary_identity)
            cache._qsa_summary_identity["complete_blocks"] = blocks
            mx.eval(cache._qsa_pooled_keys)
        if shared:
            for item in rows:
                item.shared_qsa_prefix_id = "same-four-token-live-tip"
        state = attach_segmented_self_mtp_lanes(model, None, rows)
        state, leaving = detach_self_mtp_lanes(model, state, [1])
        survivor = state.row_caches[0].target[1]
        assert isinstance(survivor, SharedSuffixQSAKVCache) is shared

        def force(logprobs, *_args, **_kwargs):
            return 0, int(mx.argmax(logprobs[0]).item())

        with (
            patch(
                "mlx2.runtime.segmented_batch_cache."
                "qwen4_qsa_indexed_private_delta_preflight",
                return_value=(True, "engaged"),
            ),
            patch(
                "mlx2.runtime.segmented_batch_cache."
                "qwen4_qsa_indexed_private_delta_attention",
                side_effect=RuntimeError("synthetic CPU decline"),
            ),
            patch(
                "mlx2.runtime.models.qwen4_exp.QSAIndexer.select_shared_suffix_batch",
                side_effect=AssertionError("B1 survivor used the batch selector"),
            ),
            patch(
                "mlx2.runtime.hybrid_speculative._batched_residual_verify",
                side_effect=force,
            ),
        ):
            proposal = propose_batched_self_mtp(model, state)
        outputs = tuple(token.token for token in proposal.outputs[0])
        logprobs = tuple(mx.array(token.logprobs) for token in proposal.outputs[0])
        abort_batched_self_mtp(state, proposal)
        close_segmented_self_mtp_state(state)
        leaving[0].segment_transaction.close()
        return outputs, logprobs

    expected_outputs, expected_logprobs = run(False)
    actual_outputs, actual_logprobs = run(True)

    assert actual_outputs == expected_outputs
    assert len(actual_logprobs) == len(expected_logprobs)
    assert all(
        mx.array_equal(actual, expected).item()
        for actual, expected in zip(actual_logprobs, expected_logprobs)
    )


def test_shared_qsa_prefix_attestation_rejects_mixed_host_identities():
    first = _detached(0, position=8)
    second = _detached(1, position=8)
    first.shared_qsa_prefix_id = "prefix-a"
    second.shared_qsa_prefix_id = "prefix-b"

    state = attach_segmented_self_mtp_lanes(_FakeModel(), None, [first, second])
    assert state.shared_qsa_prefix_id is None
    close_segmented_self_mtp_state(state)


def test_serial_cycle_permanently_invalidates_shared_qsa_prefix(monkeypatch):
    first = _detached(0, position=8)
    second = _detached(1, position=8)
    first.shared_qsa_prefix_id = "same-host-token-digest"
    second.shared_qsa_prefix_id = "same-host-token-digest"
    state = attach_segmented_self_mtp_lanes(_FakeModel(), None, [first, second])
    assert state.shared_qsa_prefix_id is not None

    monkeypatch.setenv("MLX_LM_TRUE_BATCHED_SEGMENTED_MTP", "0")
    with patch(
        "mlx2.runtime.hybrid_speculative._propose_batched_self_mtp_impl",
        side_effect=lambda _model, row: _fake_row_proposal({row.lanes[0].uid: 0})(
            _model, row
        ),
    ):
        proposal = propose_batched_self_mtp(_FakeModel(), state)
    assert state.shared_qsa_prefix_id is None
    commit_batched_self_mtp(
        state,
        proposal,
        emitted_counts=[1, 1],
        terminal=[False, False],
    )

    monkeypatch.setenv("MLX_LM_TRUE_BATCHED_SEGMENTED_MTP", "1")
    # Re-enabling batching must not recreate a shared-prefix proof from equal
    # host offsets after the serial rows were allowed to diverge.
    assert state.shared_qsa_prefix_id is None
    close_segmented_self_mtp_state(state)


def _fake_row_proposal(accepted_by_uid):
    def propose(_model, row_state):
        lane = row_state.lanes[0]
        accepted = int(accepted_by_uid[lane.uid])
        for cache in row_state.caches.target:
            cache.offset += accepted + 1
        drafts = (31 + lane.uid, 41 + lane.uid)
        bonus = 51 + lane.uid
        logprobs = mx.zeros((3, 64))
        hidden = mx.zeros((1, 3, 2))
        outputs = tuple(
            [
                MTPToken(drafts[index], logprobs[index], True)
                for index in range(accepted)
            ]
            + [MTPToken(bonus, logprobs[accepted], False)]
        )
        proposal = SelfMTPCycleResult(
            membership_epoch=row_state.membership_epoch,
            lane_uids=(lane.uid,),
            draft_depths=(2,),
            accepted_lengths=(accepted,),
            target_drops=(2 - accepted,),
            head_drops=(2,),
            outputs=(outputs,),
            _old_curs=(lane.cur,),
            _old_seed_hs=(lane.seed_h,),
            _drafts=(drafts,),
            _vhidden=(hidden,),
            _logprobs=(logprobs,),
            _bonuses=(bonus,),
        )
        row_state.proposal_open = True
        row_state._open_proposal = proposal
        return proposal

    return propose


def test_segmented_self_mtp_gate_defaults_off(monkeypatch):
    monkeypatch.delenv("MLX_LM_SEGMENTED_SELF_MTP", raising=False)
    assert segmented_self_mtp_enabled() is False


def test_serving_knob_honors_env_and_explicit_override(monkeypatch):
    from mlx2.runtime.generate import (
        _segment_aware_cohort_size,
        _segment_aware_live_tip_enabled,
    )

    monkeypatch.setenv("MLX_LM_SEGMENTED_SELF_MTP", "1")
    assert _segment_aware_live_tip_enabled({}) is True
    assert _segment_aware_live_tip_enabled({"segment_aware_live_tip": False}) is False
    assert _segment_aware_cohort_size({"segment_aware_live_tip": True}) == 2
    assert (
        _segment_aware_cohort_size(
            {"segment_aware_live_tip": True, "segment_aware_cohort_size": 4}
        )
        == 4
    )
    with pytest.raises(ValueError, match="must be positive"):
        _segment_aware_cohort_size({"segment_aware_cohort_size": 0})


def test_async_qsa_promotion_is_nested_and_default_off(monkeypatch):
    from mlx2.runtime.generate import (
        _segmented_async_qsa_min_remaining_tokens,
        _segmented_async_qsa_promotion_enabled,
        _segmented_async_qsa_promotion_for_budget,
    )

    monkeypatch.delenv("MLX_LM_SEGMENTED_ASYNC_QSA_PROMOTION", raising=False)
    assert not _segmented_async_qsa_promotion_enabled({"segment_aware_live_tip": True})
    monkeypatch.setenv("MLX_LM_SEGMENTED_ASYNC_QSA_PROMOTION", "1")
    assert _segmented_async_qsa_promotion_enabled({"segment_aware_live_tip": True})
    assert not _segmented_async_qsa_promotion_enabled({"segment_aware_live_tip": False})
    assert not _segmented_async_qsa_promotion_enabled(
        {
            "segment_aware_live_tip": True,
            "segment_aware_async_qsa_promotion": False,
        }
    )
    monkeypatch.delenv("MLX_LM_SEGMENTED_ASYNC_QSA_MIN_REMAINING_TOKENS", raising=False)
    assert _segmented_async_qsa_min_remaining_tokens({}) == 16
    policy = {
        "segment_aware_live_tip": True,
        "segment_aware_async_qsa_promotion": True,
        "segment_aware_async_qsa_min_remaining_tokens": 8,
    }
    assert _segmented_async_qsa_min_remaining_tokens(policy) == 8
    assert not _segmented_async_qsa_promotion_for_budget(policy, 8)
    assert _segmented_async_qsa_promotion_for_budget(policy, 9)
    segmented_self_mtp_stats(reset=True)
    assert not _segmented_async_qsa_promotion_for_budget(policy, 7, record=True)
    assert _segmented_async_qsa_promotion_for_budget(policy, 10, record=True)
    budget = segmented_self_mtp_stats()
    assert budget["async_qsa_budget_checks"] == 2
    assert budget["async_qsa_budget_retained_segmented"] == 1
    assert budget["async_qsa_budget_promotions"] == 1
    assert budget["async_qsa_budget_remaining_tokens_cumulative"] == 17
    assert budget["async_qsa_budget_cutoff_tokens_cumulative"] == 16
    with pytest.raises(ValueError, match="must be non-negative"):
        _segmented_async_qsa_min_remaining_tokens(
            {"segment_aware_async_qsa_min_remaining_tokens": -1}
        )


def test_independent_b1_cycle_engages_without_physical_b2_and_promotes():
    segmented_self_mtp_stats(reset=True)
    detached = [_detached(0), _detached(1)]
    original_cache_ids = [
        {id(cache) for cache in item.caches.target + item.caches.draft}
        for item in detached
    ]
    assert original_cache_ids[0].isdisjoint(original_cache_ids[1])
    state = attach_segmented_self_mtp_lanes(object(), None, detached)

    with patch(
        "mlx2.runtime.hybrid_speculative._propose_batched_self_mtp_impl",
        side_effect=_fake_row_proposal({0: 0, 1: 1}),
    ):
        proposal = propose_batched_self_mtp(object(), state)
    commit_batched_self_mtp(
        state,
        proposal,
        emitted_counts=[1, 2],
        terminal=[False, False],
    )

    assert [transaction.position for transaction in state.transactions] == [7, 8]
    assert [lane.cur for lane in state.lanes] == [51, 52]
    counters = segmented_self_mtp_stats()
    require_segmented_self_mtp_engagement(counters)
    assert counters["b1_target_forwards"] == 2
    assert counters["b1_draft_forwards"] == 4
    assert counters["physical_b2_formations"] == 0
    assert counters["transaction_promotions"] == 2
    assert counters["accepted_zero"] == 1
    assert counters["accepted_partial"] == 1
    assert counters["proposal_ns"] == counters["commit_ns"] == 0
    close_segmented_self_mtp_state(state)


def test_zero_delivery_rejects_transaction_and_restores_target_position():
    segmented_self_mtp_stats(reset=True)
    state = attach_segmented_self_mtp_lanes(object(), None, [_detached(7)])
    with patch(
        "mlx2.runtime.hybrid_speculative._propose_batched_self_mtp_impl",
        side_effect=_fake_row_proposal({7: 0}),
    ):
        proposal = propose_batched_self_mtp(object(), state)
    commit_batched_self_mtp(
        state,
        proposal,
        emitted_counts=[0],
        terminal=[True],
    )
    assert state.transactions[0].position == 6
    assert all(cache.offset == 6 for cache in state.row_caches[0].target)
    counters = segmented_self_mtp_stats()
    assert counters["transaction_rejections"] == 1
    assert counters["transaction_promotions"] == 0
    close_segmented_self_mtp_state(state)


def test_segmented_attach_rejects_shared_mutable_cache_objects():
    left = _detached(0)
    right = _detached(1)
    right.caches = left.caches
    with pytest.raises(ValueError, match="alias"):
        attach_segmented_self_mtp_lanes(object(), None, [left, right])


def test_gate_refuses_silent_fallback_or_nonengagement():
    with pytest.raises(RuntimeError, match="never engaged"):
        require_segmented_self_mtp_engagement(
            {"engaged": 0, "b1_target_forwards": 0, "physical_b2_formations": 0}
        )
    with pytest.raises(RuntimeError, match="physical B2"):
        require_segmented_self_mtp_engagement(
            {"engaged": 1, "b1_target_forwards": 2, "physical_b2_formations": 1}
        )
    with pytest.raises(RuntimeError, match="completed no transaction"):
        require_segmented_self_mtp_engagement(
            {
                "engaged": 1,
                "b1_target_forwards": 2,
                "physical_b2_formations": 0,
                "transaction_branches": 2,
                "committed_cycles": 0,
            }
        )


def test_generation_batch_seam_keeps_two_concrete_b1_rows():
    from mlx2.runtime.generate import MTPGenerationBatch, StopSequenceMatcher

    detached = [_detached(0), _detached(1)]
    with patch(
        "mlx2.runtime.hybrid_speculative.attach_self_mtp_lanes",
        side_effect=AssertionError("physical B2 attach must not run"),
    ):
        batch = MTPGenerationBatch(
            object(),
            detached,
            [None, None],
            [StopSequenceMatcher(), StopSequenceMatcher()],
            segmented_live_tip=True,
        )
    assert batch.state.row_caches == [item.caches for item in detached]
    assert not hasattr(batch.state, "caches")
    assert batch.cache_nbytes == 256
    batch.close()


def test_generation_batch_adapts_one_depth_for_whole_cohort_and_parks():
    from mlx2.runtime.adaptive_policy import CohortAdaptiveMTPDepth
    from mlx2.runtime.generate import MTPGenerationBatch, StopSequenceMatcher

    policy = CohortAdaptiveMTPDepth(
        max_depth=2,
        ewma_alpha=1.0,
        loss_rounds=1,
        gain_rounds=2,
        park_rounds=1,
    )
    batch = MTPGenerationBatch(
        object(),
        [_detached(0), _detached(1)],
        [None, None],
        [StopSequenceMatcher(), StopSequenceMatcher()],
        segmented_live_tip=True,
        adaptive_depth_policy=policy,
    )
    observed = []

    def propose(_model, state):
        depths = tuple(lane.num_draft for lane in state.lanes)
        observed.append(depths)
        outputs = tuple(
            (MTPToken(50 + lane.uid, mx.zeros((64,)), False),)
            for lane in state.lanes
        )
        proposal = SelfMTPCycleResult(
            membership_epoch=state.membership_epoch,
            lane_uids=tuple(lane.uid for lane in state.lanes),
            draft_depths=depths,
            accepted_lengths=tuple(0 for _ in state.lanes),
            target_drops=depths,
            head_drops=depths,
            outputs=outputs,
        )
        state.proposal_open = True
        state._open_proposal = proposal
        return proposal

    def commit(state, proposal, **_kwargs):
        assert proposal.lane_uids == tuple(lane.uid for lane in state.lanes)
        state.proposal_open = False
        state._open_proposal = None

    with (
        patch("mlx2.runtime.hybrid_speculative.propose_batched_self_mtp", side_effect=propose),
        patch("mlx2.runtime.hybrid_speculative.commit_batched_self_mtp", side_effect=commit),
    ):
        for _ in range(4):
            assert len(batch.next()) == 2

    assert observed == [(2, 2), (1, 1), (0, 0), (1, 1)]
    assert policy.counters["parks"] == 2
    assert policy.counters["reentries"] == 1
    batch.close()


def test_generation_batch_receipt_identifies_adaptive_depth_policy():
    from mlx2.runtime.adaptive_policy import CohortAdaptiveMTPDepth
    from mlx2.runtime.generate import MTPGenerationBatch, StopSequenceMatcher

    detached = _detached(0)
    detached.lane.max_tokens = 1
    policy = CohortAdaptiveMTPDepth(
        max_depth=2,
        ewma_alpha=0.5,
        loss_rounds=2,
        gain_rounds=4,
        park_rounds=3,
    )
    batch = MTPGenerationBatch(
        object(),
        [detached],
        [None],
        [StopSequenceMatcher()],
        segmented_live_tip=True,
        adaptive_depth_policy=policy,
    )

    def propose(_model, state):
        output = (MTPToken(50, mx.zeros((64,)), False),)
        proposal = SelfMTPCycleResult(
            membership_epoch=state.membership_epoch,
            lane_uids=(state.lanes[0].uid,),
            draft_depths=(state.lanes[0].num_draft,),
            accepted_lengths=(0,),
            target_drops=(state.lanes[0].num_draft,),
            head_drops=(state.lanes[0].num_draft,),
            outputs=(output,),
        )
        state.proposal_open = True
        state._open_proposal = proposal
        return proposal

    def commit(state, _proposal, **_kwargs):
        state.proposal_open = False
        state._open_proposal = None

    with (
        patch(
            "mlx2.runtime.hybrid_speculative.propose_batched_self_mtp",
            side_effect=propose,
        ),
        patch(
            "mlx2.runtime.hybrid_speculative.commit_batched_self_mtp",
            side_effect=commit,
        ),
    ):
        responses = batch.next()

    adaptive = responses[0].mtp_receipt["adaptive_depth"]
    assert adaptive["selected"] is True
    assert adaptive["max_depth"] == adaptive["current"] == 2
    assert adaptive["policy"] == {
        "ewma_alpha": 0.5,
        "shrink_gate": 0.35,
        "grow_gate": 0.8,
        "loss_rounds": 2,
        "gain_rounds": 4,
        "park_rounds": 3,
        "goodput_alpha": 0.25,
        "goodput_hysteresis": 0.05,
        "min_samples_per_depth": 3,
        "goodput_window": 8,
        "probe_interval": 16,
        "stale_rounds": 128,
    }
    assert adaptive["counters"]["boundaries"] == 1
    batch.close()


def test_generation_boundary_accounts_cost_probe_to_live_cohort_width():
    """Selection and observation must charge the same physical cohort bucket."""
    from mlx2.runtime.adaptive_policy import CohortAdaptiveMTPDepth
    from mlx2.runtime.generate import MTPGenerationBatch, StopSequenceMatcher

    detached = [_detached(uid) for uid in range(8)]
    for item in detached:
        item.lane.max_tokens = 64
    policy = CohortAdaptiveMTPDepth(max_depth=2)
    batch = MTPGenerationBatch(
        object(),
        detached,
        [None] * len(detached),
        [StopSequenceMatcher() for _ in detached],
        segmented_live_tip=True,
        adaptive_depth_policy=policy,
    )

    def cycle(state, *, zero_fast_path=False):
        depths = tuple(lane.num_draft for lane in state.lanes)
        proposal = SelfMTPCycleResult(
            membership_epoch=state.membership_epoch,
            lane_uids=tuple(lane.uid for lane in state.lanes),
            draft_depths=depths,
            accepted_lengths=depths,
            target_drops=depths,
            head_drops=depths,
            outputs=tuple(
                (MTPToken(50 + lane.uid, mx.zeros((64,)), False),)
                for lane in state.lanes
            ),
            zero_fast_path=zero_fast_path,
            true_batched=True,
        )
        if not zero_fast_path:
            state.proposal_open = True
            state._open_proposal = proposal
        return proposal

    def propose(_model, state):
        return cycle(state)

    def advance_zero(_model, state):
        return cycle(state, zero_fast_path=True)

    def commit(state, proposal, **_kwargs):
        assert proposal.lane_uids == tuple(lane.uid for lane in state.lanes)
        state.proposal_open = False
        state._open_proposal = None

    with (
        patch(
            "mlx2.runtime.hybrid_speculative.propose_batched_self_mtp",
            side_effect=propose,
        ),
        patch(
            "mlx2.runtime.hybrid_speculative.advance_batched_self_mtp_zero",
            side_effect=advance_zero,
        ),
        patch(
            "mlx2.runtime.hybrid_speculative.commit_batched_self_mtp",
            side_effect=commit,
        ),
    ):
        for _ in range(policy.probe_interval + 1):
            assert len(batch.next()) == 8

    bucket = policy.diagnostics()["buckets"]["5-8"]
    assert bucket["rounds"] == policy.probe_interval + 1
    assert bucket["probes"] > 0
    assert bucket["samples"]["0"] > 0
    assert all(item["width_bucket"] == "5-8" for item in policy.trace)
    assert all(item["cohort_width"] == 8 for item in policy.trace)
    assert all(item["observed_compute_width"] == 8 for item in policy.trace)
    batch.close()


def test_generation_round_goodput_excludes_idle_and_policy_time(monkeypatch):
    from mlx2.runtime import generate as generate_module
    from mlx2.runtime.adaptive_policy import CohortAdaptiveMTPDepth
    from mlx2.runtime.generate import MTPGenerationBatch, StopSequenceMatcher

    class Clock:
        now = 0.0

        def __call__(self):
            return self.now

        def advance(self, seconds):
            self.now += seconds

    clock = Clock()
    monkeypatch.setattr(generate_module.time, "perf_counter", clock)
    detached = _detached(0)
    detached.lane.max_tokens = 16
    policy = CohortAdaptiveMTPDepth(max_depth=2)
    original_select = policy.select

    def delayed_select(**kwargs):
        # Admission/policy work is outside the measured decode round.
        clock.advance(10.0)
        return original_select(**kwargs)

    monkeypatch.setattr(policy, "select", delayed_select)
    batch = MTPGenerationBatch(
        object(),
        [detached],
        [None],
        [StopSequenceMatcher()],
        segmented_live_tip=True,
        adaptive_depth_policy=policy,
    )

    def propose(_model, state):
        clock.advance(0.25)
        depths = tuple(lane.num_draft for lane in state.lanes)
        proposal = SelfMTPCycleResult(
            membership_epoch=state.membership_epoch,
            lane_uids=tuple(lane.uid for lane in state.lanes),
            draft_depths=depths,
            accepted_lengths=depths,
            target_drops=depths,
            head_drops=depths,
            outputs=((MTPToken(50, mx.zeros((64,)), False),),),
        )
        state.proposal_open = True
        state._open_proposal = proposal
        return proposal

    def commit(state, _proposal, **_kwargs):
        state.proposal_open = False
        state._open_proposal = None

    with (
        patch(
            "mlx2.runtime.hybrid_speculative.propose_batched_self_mtp",
            side_effect=propose,
        ),
        patch(
            "mlx2.runtime.hybrid_speculative.commit_batched_self_mtp",
            side_effect=commit,
        ),
    ):
        for _ in range(policy.min_samples_per_depth):
            clock.advance(100.0)  # idle between requests/rounds
            assert len(batch.next()) == 1

    bucket = policy.diagnostics()["buckets"]["1"]
    assert bucket["goodput_tokens_per_second"]["2"] == pytest.approx(4.0)
    assert bucket["estimate_updates"]["2"] == 1
    assert all(item["elapsed_seconds"] == pytest.approx(0.25) for item in policy.trace)
    batch.close()


def test_empty_cohort_retains_compatible_adaptive_controller_state():
    from mlx2.runtime.adaptive_policy import CohortAdaptiveMTPDepth
    from mlx2.runtime.generate import MTPGenerationBatch

    retained = CohortAdaptiveMTPDepth(max_depth=2, current_depth=1)
    active = MTPGenerationBatch.empty(
        object(), segmented_live_tip=True, adaptive_depth_policy=retained
    )
    incoming = MTPGenerationBatch.empty(
        object(),
        segmented_live_tip=True,
        adaptive_depth_policy=CohortAdaptiveMTPDepth(max_depth=2),
    )
    active.extend(incoming)
    assert active.adaptive_depth_policy is retained
    assert active.adaptive_depth_policy.current_depth == 1
    active.close()
    incoming.close()


def test_live_true_batched_width_lock_defers_join_until_empty_cohort():
    """A live B2 consumer must not silently become B3 mid-generation."""
    from mlx2.runtime.generate import MTPGenerationBatch, StopSequenceMatcher

    segmented_self_mtp_stats(reset=True)
    active = MTPGenerationBatch(
        object(),
        [_detached(0), _detached(1)],
        [None, None],
        [StopSequenceMatcher(), StopSequenceMatcher()],
        segmented_live_tip=True,
    )
    arriving = MTPGenerationBatch(
        object(),
        [_detached(2)],
        [None],
        [StopSequenceMatcher()],
        segmented_live_tip=True,
    )
    active._segmented_compute_width_locked = True

    active.extend(arriving)

    assert active.uids == [0, 1]
    assert list(active._paused) == [2]
    assert segmented_self_mtp_stats()["live_width_change_deferrals"] == 1

    # The preserved detached row enters a new, ownership-free cohort.
    active.state.lanes.clear()
    active.state.row_caches.clear()
    active.state.transactions.clear()
    active._attach_packages(list(active._paused.values()))
    active._paused.clear()
    assert active.uids == [2]
    active.close()


def test_generation_batch_promotes_after_one_uniform_segmented_cycle():
    from types import SimpleNamespace

    from mlx2.runtime.generate import MTPGenerationBatch, StopSequenceMatcher
    from mlx2.runtime.segmented_physical_promotion import (
        SegmentedPhysicalPromotionReceipt,
    )

    segmented_self_mtp_stats(reset=True)
    holder = {}

    class Ticket:
        def finish(self):
            state = holder["state"]
            assert not state.proposal_open
            physical = SimpleNamespace(
                lanes=list(state.lanes),
                caches=SimpleNamespace(target=[], draft=[]),
                proposal_open=False,
            )
            return physical, SegmentedPhysicalPromotionReceipt(
                queued_ns=10,
                finish_ns=20,
                stream_wait_ns=5,
                reserved_bytes=30,
                patched_bytes=40,
                recurrent_arrays_reused=4,
                cleanup_error_count=0,
                rows=2,
                layers=2,
                advance=1,
            )

    def begin(state, **_kwargs):
        holder["state"] = state
        return Ticket()

    detached = [_detached(0), _detached(1)]
    with patch(
        "mlx2.runtime.segmented_physical_promotion.begin_segmented_physical_promotion",
        side_effect=begin,
    ):
        batch = MTPGenerationBatch(
            object(),
            detached,
            [None, None],
            [StopSequenceMatcher(), StopSequenceMatcher()],
            segmented_live_tip=True,
            async_qsa_promotion=True,
        )

    with patch(
        "mlx2.runtime.hybrid_speculative._propose_batched_self_mtp_impl",
        side_effect=_fake_row_proposal({0: 1, 1: 1}),
    ):
        responses = batch.next()

    assert len(responses) == 4
    assert batch.segmented_live_tip is False
    assert batch._async_qsa_receipt["advance"] == 1
    counters = segmented_self_mtp_stats()
    assert counters["async_qsa_promotion_requests"] == 1
    assert counters["async_qsa_promotion_queued"] == 1
    assert counters["async_qsa_promotion_engaged"] == 1
    assert counters["async_qsa_promotion_reserved_bytes"] == 30
    assert counters["async_qsa_promotion_patched_bytes"] == 40
    assert counters["async_qsa_promotion_wait_ns"] == 5
    assert batch._async_qsa_receipts_by_uid[0]["advance"] == 1
    assert batch._async_qsa_receipts_by_uid[1]["advance"] == 1


def test_empty_physical_batch_preserves_policy_and_resets_segmented_admission():
    from mlx2.runtime.generate import MTPGenerationBatch

    batch = MTPGenerationBatch.empty(
        object(), segmented_live_tip=False, async_qsa_promotion=True
    )
    assert batch.async_qsa_promotion is True
    assert batch._async_qsa_pending is False

    batch._normalize_empty_segmented_admission()

    assert batch.segmented_live_tip is True
    assert batch._async_qsa_pending is True


def test_short_output_join_suppresses_unpublished_async_destination():
    from mlx2.runtime.generate import MTPGenerationBatch, StopSequenceMatcher

    destination = MTPGenerationBatch.empty(
        object(), segmented_live_tip=True, async_qsa_promotion=True
    )
    short = MTPGenerationBatch(
        object(),
        [_detached(0)],
        [None],
        [StopSequenceMatcher()],
        segmented_live_tip=True,
        async_qsa_promotion=False,
    )

    destination.extend(short)

    assert destination.async_qsa_promotion is False
    assert destination._async_qsa_pending is False
    assert destination.uids == [0]
    destination.close()


def test_generation_batch_binds_prequeued_candidate_without_rearming():
    from mlx2.runtime.generate import MTPGenerationBatch, StopSequenceMatcher

    ticket = object()

    class Prequeue:
        def __init__(self):
            self.bound = 0

        def bind(self, state, *, note):
            self.bound += 1
            assert len(state.lanes) == 2
            return ticket

        def cancel_and_drain(self):
            raise AssertionError("valid prequeue must not be cancelled")

    prequeue = Prequeue()
    batch = MTPGenerationBatch(
        object(),
        [_detached(0), _detached(1)],
        [None, None],
        [StopSequenceMatcher(), StopSequenceMatcher()],
        segmented_live_tip=True,
        async_qsa_promotion=True,
        async_qsa_prequeue=prequeue,
    )
    assert prequeue.bound == 1
    assert batch._async_qsa_ticket is ticket
    assert segmented_self_mtp_stats()["async_qsa_prequeue_bound"] >= 1
    batch.close()


def test_async_candidate_and_joined_recurrent_state_count_toward_peak_memory():
    from types import SimpleNamespace

    from mlx2.runtime.generate import MTPGenerationBatch, StopSequenceMatcher

    batch = MTPGenerationBatch(
        object(),
        [_detached(0), _detached(1)],
        [None, None],
        [StopSequenceMatcher(), StopSequenceMatcher()],
        segmented_live_tip=True,
    )
    baseline = batch.cache_nbytes
    batch.state._segmented_caches = SimpleNamespace(
        target=[SimpleNamespace(nbytes=70)],
        draft=[SimpleNamespace(nbytes=30)],
    )
    batch._async_qsa_ticket = SimpleNamespace(reserved_bytes=200)

    assert batch.cache_nbytes == baseline + 300
    rows = batch.mtp_cycle_state()
    assert len(rows) == 2
    # Segmented arrays are allocated; the async ticket is still pending.
    expected_gib = (baseline / 2 + 50) / float(1 << 30)
    assert rows[0][4] == pytest.approx(expected_gib)
    assert rows[1][4] == pytest.approx(expected_gib)
    assert rows[0][5] == pytest.approx(100 / float(1 << 30))
    assert rows[1][5] == pytest.approx(100 / float(1 << 30))


def test_discarded_generation_row_releases_transaction_and_cache_owner():
    from mlx2.runtime.generate import MTPGenerationBatch

    item = _detached(0)
    target = _CloseableCacheList(item.caches.target)
    item.caches.target = target
    batch = MTPGenerationBatch(
        _FakeModel(),
        [item],
        [None],
        [_FakeMatcher()],
        segmented_live_tip=True,
    )
    transaction = batch.state.transactions[0]
    batch.filter([])
    assert transaction.closed is True
    assert target.closed is True
    batch.close()


def test_detach_flushes_pending_draft_to_caught_up_live_tip():
    state = attach_segmented_self_mtp_lanes(object(), None, [_detached(4)])
    with patch(
        "mlx2.runtime.hybrid_speculative._propose_batched_self_mtp_impl",
        side_effect=_fake_row_proposal({4: 1}),
    ):
        proposal = propose_batched_self_mtp(object(), state)
    commit_batched_self_mtp(
        state,
        proposal,
        emitted_counts=[2],
        terminal=[False],
    )
    assert state.lanes[0].pending_hs is not None
    assert state.lanes[0].pending_ts

    state, detached = detach_self_mtp_lanes(_FakeModel(), state, [0])
    assert not state.lanes
    assert detached[0].lane.pending_hs is None
    assert detached[0].lane.pending_ts == []
    target_position = max(cache.offset for cache in detached[0].caches.target)
    draft_position = max(cache.offset for cache in detached[0].caches.draft)
    assert draft_position == target_position - 1
    assert detached[0].segment_transaction.predecessor_lineage_id is not None
    state = attach_segmented_self_mtp_lanes(object(), state, detached)
    assert len(state.lanes) == 1
    close_segmented_self_mtp_state(state)


def test_attach_partial_failure_releases_only_new_transaction():
    first = _detached(0)
    second = _detached(1, position=7)
    reused = SegmentedLaneTransaction(second.caches, second.lane, 7)
    second.segment_transaction = reused
    for cache in second.caches.target:
        cache.offset += 1
    second.caches.draft[0].offset += 1
    with pytest.raises(ValueError, match="state changed|position disagrees"):
        attach_segmented_self_mtp_lanes(object(), None, [first, second])
    assert first.segment_transaction is None
    assert reused.closed is False
    assert all(not cache.speculating for cache in first.caches.target)
    assert all(not cache.speculating for cache in second.caches.target)
    reused.close()


def test_generation_stale_parallel_branch_is_rejected():
    item = _detached(0)
    transaction = SegmentedLaneTransaction(item.caches, item.lane, 6)
    winner = transaction.fork("winner")
    stale = transaction.fork("stale")
    for cache in item.caches.target:
        cache.offset += 1
    item.caches.draft[0].offset += 1
    transaction.publish(winner, item.caches, item.lane, 1, proposed=2, accepted=1)
    with pytest.raises(Exception, match="generation|stale"):
        transaction.publish(stale, item.caches, item.lane, 1, proposed=2, accepted=1)
    stale.close()
    transaction.close()


@pytest.mark.parametrize("mutation", ["cur", "qsa", "gdn"])
def test_reattach_rejects_same_position_out_of_lineage_state_mutation(mutation):
    item = _detached(0)
    transaction = SegmentedLaneTransaction(item.caches, item.lane, 6)
    item.segment_transaction = transaction
    if mutation == "cur":
        item.lane.cur += 1
    elif mutation == "qsa":
        item.caches.target[1].index_keys = object()
    else:
        item.caches.target[2].cache = [object()]
    with pytest.raises(ValueError, match="state changed outside its lineage"):
        attach_segmented_self_mtp_lanes(object(), None, [item])
    transaction.close()


def test_zero_delivery_refuses_changed_state_before_cheap_rejection():
    item = _detached(0)
    transaction = SegmentedLaneTransaction(item.caches, item.lane, 6)
    branch = transaction.fork("zero")
    item.caches.target[2].cache = [object()]
    with pytest.raises(ValueError, match="state changed outside its lineage"):
        transaction.publish(branch, item.caches, item.lane, 0, proposed=2, accepted=0)
    branch.close()
    transaction.close()


def test_each_position_bearing_target_cache_must_be_aligned():
    item = _detached(0)
    transaction = SegmentedLaneTransaction(item.caches, item.lane, 6)
    item.caches.target[1].offset -= 1
    with pytest.raises(ValueError, match="positions .* disagree"):
        transaction.validate(item.caches, item.lane, 6)
    transaction.close()


def test_reused_transaction_rejects_same_position_cache_swap():
    original = _detached(0)
    transaction = SegmentedLaneTransaction(original.caches, original.lane, 6)
    replacement = _detached(0)
    replacement.segment_transaction = transaction
    with pytest.raises(ValueError, match="fingerprint changed"):
        attach_segmented_self_mtp_lanes(object(), None, [replacement])
    assert transaction.closed is False
    transaction.close()


def test_qsa_transaction_planes_name_nonoverlapping_components():
    item = _detached(0)
    transaction = SegmentedLaneTransaction(item.caches, item.lane, 6)
    bases = {base.kind.value: base for base in transaction.lineage.current_view().bases}
    attention_fields = {
        field
        for entry in bases["attention_kv"].payload.fingerprint
        if "qsa" in entry[0].lower()
        for field in entry[2]
    }
    summary_fields = {
        field
        for entry in bases["qsa_summary"].payload.fingerprint
        for field in entry[2]
    }
    assert attention_fields == {"keys", "values", "key_scale", "value_scale"}
    assert summary_fields == {
        "index_keys",
        "_qsa_pooled_keys",
        "_qsa_pooled_ratio",
        "_qsa_summary_identity",
        "_qsa_summary_restored",
        "_qsa_pending_pooled",
        "_mtp_share_topk",
        "_mtp_shared_topk",
    }
    assert attention_fields.isdisjoint(summary_fields)
    transaction.close()


def test_mtp_plane_requires_physical_cache_plus_pending_sidecar_alignment():
    item = _detached(0)
    transaction = SegmentedLaneTransaction(item.caches, item.lane, 6)
    item.lane.pending_ts = [9]
    item.lane.pending_hs = mx.zeros((1, 1, 2))
    with pytest.raises(ValueError, match="logical coordinate|not caught up"):
        transaction.validate(item.caches, item.lane, 6)
    item.lane.pending_hs = None
    with pytest.raises(ValueError, match="geometry disagrees"):
        transaction.validate(item.caches, item.lane, 6)
    transaction.close()


def test_commit_failure_restores_committed_rows_and_clears_open_ownership():
    state = attach_segmented_self_mtp_lanes(object(), None, [_detached(0)])
    with patch(
        "mlx2.runtime.hybrid_speculative._propose_batched_self_mtp_impl",
        side_effect=_fake_row_proposal({0: 0}),
    ):
        proposal = propose_batched_self_mtp(object(), state)
    with (
        patch.object(state.transactions[0], "publish", side_effect=RuntimeError("CAS")),
        pytest.raises(RuntimeError, match="CAS"),
    ):
        commit_batched_self_mtp(
            state,
            proposal,
            emitted_counts=[1],
            terminal=[False],
        )
    assert state.poisoned is False
    assert state._row_states == []
    assert state._row_proposals == []
    assert state._transaction_branches == []
    assert len(state.transactions) == len(state.row_caches) == 1
    state.transactions[0].validate(
        state.row_caches[0], state.lanes[0], state.transactions[0].position
    )
    close_segmented_self_mtp_state(state)


def test_tiny_qwen4_gdn_qsa_mtp_runs_real_independent_b1_cycle():
    from test_batched_mtp import _prepare_lane, _tiny_qwen4_model

    segmented_self_mtp_stats(reset=True)
    model = _tiny_qwen4_model()
    detached = [
        _prepare_lane(model, 0, [1, 2, 3, 4]),
        _prepare_lane(model, 1, [5, 6, 7, 8, 9]),
    ]
    assert [type(cache).__name__ for cache in detached[0].caches.target] == [
        "Qwen4ArraysCache",
        "QSAKVCache",
    ]
    assert [type(cache).__name__ for cache in detached[0].caches.draft] == [
        "QSAKVCache"
    ]
    state = attach_segmented_self_mtp_lanes(model, None, detached)
    accepted = iter([0, 1])
    mtp_calls = 0
    target_calls = 0
    original_mtp_step = model.mtp_step
    original_mtp_backbone = model.mtp_backbone

    def counted_mtp_step(*args, **kwargs):
        nonlocal mtp_calls
        mtp_calls += 1
        return original_mtp_step(*args, **kwargs)

    def counted_mtp_backbone(*args, **kwargs):
        nonlocal target_calls
        target_calls += 1
        return original_mtp_backbone(*args, **kwargs)

    model.mtp_step = counted_mtp_step
    model.mtp_backbone = counted_mtp_backbone

    def force(logprobs, *_args, **_kwargs):
        count = next(accepted)
        return count, int(mx.argmax(logprobs[count]).item())

    with patch(
        "mlx2.runtime.hybrid_speculative._batched_residual_verify",
        side_effect=force,
    ):
        proposal = propose_batched_self_mtp(model, state)
    assert proposal.accepted_lengths == (0, 1)
    commit_batched_self_mtp(
        state,
        proposal,
        emitted_counts=[len(row) for row in proposal.outputs],
        terminal=[False, False],
    )
    counters = segmented_self_mtp_stats()
    require_true_batched_segmented_self_mtp_engagement(counters)
    assert counters["batched_target_forwards"] == 1
    assert counters["batched_draft_forwards"] == 2
    assert target_calls == 1
    assert mtp_calls == 2
    assert counters["b1_target_forwards"] == 0
    assert counters["physical_b2_formations"] == 0
    assert counters["segmented_attention_calls"] > 0
    assert counters["full_prefix_materializations"] == 0
    assert counters["full_prefix_materialized_bytes"] == 0
    assert counters["row_state_splits"] > 0
    assert [transaction.position for transaction in state.transactions] == [5, 7]
    compute_caches = state._segmented_caches
    assert compute_caches is not None
    with pytest.raises(RuntimeError, match="forbids dense index-ledger"):
        compute_caches.target[1].update_index_keys(mx.zeros((2, 1, 1)))
    with pytest.raises(RuntimeError, match="forbids dense K/V"):
        compute_caches.target[1].update_and_fetch(
            mx.zeros((2, 1, 1, 1)), mx.zeros((2, 1, 1, 1))
        )
    planes = {
        base.kind.value for base in state.transactions[0].lineage.current_view().bases
    }
    assert planes == {"attention_kv", "qsa_summary", "gdn_recurrent", "mtp_draft"}

    accepted = iter([2, 0])
    with patch(
        "mlx2.runtime.hybrid_speculative._batched_residual_verify",
        side_effect=force,
    ):
        proposal = propose_batched_self_mtp(model, state)
    assert state._segmented_caches is compute_caches
    commit_batched_self_mtp(
        state,
        proposal,
        emitted_counts=[len(row) for row in proposal.outputs],
        terminal=[False, False],
    )
    assert [transaction.position for transaction in state.transactions] == [8, 8]

    state, lanes = detach_self_mtp_lanes(model, state, [0, 1])
    assert not state.lanes
    assert state._segmented_caches is None
    for item in lanes:
        target_position = max(
            int(getattr(cache, "offset", 0)) for cache in item.caches.target
        )
        draft_position = max(
            int(getattr(cache, "offset", 0)) for cache in item.caches.draft
        )
        assert draft_position == target_position - 1
    state = attach_segmented_self_mtp_lanes(model, state, lanes)
    assert len(state.lanes) == 2
    close_segmented_self_mtp_state(state)


def test_private_delta_late_decline_consumes_existing_selection_once(monkeypatch):
    from test_batched_mtp import _tiny_qwen4_model
    from mlx2.runtime.models.qwen4_exp import QSAKVCache
    from mlx2.runtime.segmented_batch_cache import SegmentedBatchQSAKVCache
    from mlx2.runtime.segmented_self_mtp import note_segmented_self_mtp

    segmented_self_mtp_stats(reset=True)
    monkeypatch.setenv("MLX_LM_QSA_PRIVATE_DELTA_MIN_CONTEXT", "0")
    monkeypatch.setenv("MLX_LM_QSA_PRIVATE_DELTA_EXACT_SET_FOLD", "1")
    model = _tiny_qwen4_model()
    attention = model.language_model.model.layers[1].self_attn
    prefix = mx.random.normal((1, 12, 32), key=mx.random.key(101))
    prefix_mask = (mx.arange(12)[:, None] >= mx.arange(12)[None, :])[None, None]
    private_rows = []
    for _ in range(2):
        cache = QSAKVCache(attention.indexer.summary_identity)
        output = attention(prefix, prefix_mask, cache)
        mx.eval(output)
        private_rows.append(cache)

    serial_rows = []
    for source in private_rows:
        cache = QSAKVCache(attention.indexer.summary_identity)
        cache.state = source.state
        cache.meta_state = source.meta_state
        serial_rows.append(cache)

    counts = {"index": [0, 0], "kv": [0, 0]}
    for row_index, cache in enumerate(private_rows):
        original_index = cache.update_index_keys
        original_kv = cache.update_and_fetch

        def counted_index(keys, *, _index=row_index, _original=original_index):
            counts["index"][_index] += 1
            return _original(keys)

        def counted_kv(keys, values, *, _index=row_index, _original=original_kv):
            counts["kv"][_index] += 1
            return _original(keys, values)

        cache.update_index_keys = counted_index
        cache.update_and_fetch = counted_kv

    hidden = mx.random.normal((2, 3, 32), key=mx.random.key(102))
    serial = SegmentedBatchQSAKVCache(serial_rows, shared_qsa_prefix=False)
    serial.prepare(lengths=[3, 3], right_padding=[0, 0])
    expected = serial.segmented_attention(attention, hidden, None)

    private = SegmentedBatchQSAKVCache(
        private_rows,
        note=note_segmented_self_mtp,
        shared_qsa_prefix=True,
    )
    private.prepare(lengths=[3, 3], right_padding=[0, 0])
    with (
        patch(
            "mlx2.runtime.segmented_batch_cache.qwen4_qsa_indexed_private_delta_preflight",
            return_value=(True, "engaged"),
        ),
        patch(
            "mlx2.runtime.segmented_batch_cache."
            "qwen4_qsa_indexed_private_delta_exact_set_preflight",
            return_value=(True, "engaged"),
        ),
        patch(
            "mlx2.runtime.segmented_batch_cache."
            "qwen4_qsa_indexed_private_delta_exact_set_attention",
            side_effect=RuntimeError("synthetic exact-set dispatch decline"),
        ),
        patch(
            "mlx2.runtime.segmented_batch_cache.qwen4_qsa_indexed_private_delta_attention",
            side_effect=RuntimeError("synthetic dispatch decline"),
        ),
    ):
        actual = private.segmented_attention(attention, hidden, None)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()
    assert counts == {"index": [1, 1], "kv": [1, 1]}
    assert [row.offset for row in private_rows] == [15, 15]
    assert [row.offset for row in serial_rows] == [15, 15]
    counters = segmented_self_mtp_stats()
    assert counters["private_delta_requests"] == 1
    assert counters["private_delta_declines"] == 1
    assert counters["private_delta_attention_calls"] == 0
    assert counters["private_delta_late_gather_fallbacks"] == 1
    assert counters["exact_set_fold_requests"] == 1
    assert counters["exact_set_fold_device_proofs"] == 1
    assert counters["exact_set_fold_attention_calls"] == 0
    assert counters["exact_set_fold_declines"] == 1
    assert counters["exact_set_fold_private_fallbacks"] == 1
    assert counters["exact_set_fold_decline_reasons"] == {"dispatch_raised": 1}


def test_segmented_qsa_zero_length_row_uses_pre_o_width():
    from test_batched_mtp import _tiny_qwen4_model
    from mlx2.runtime.models.qwen4_exp import QSAKVCache
    from mlx2.runtime.segmented_batch_cache import SegmentedBatchQSAKVCache

    model = _tiny_qwen4_model()
    source_attention = model.language_model.model.layers[1].self_attn
    rows = [
        QSAKVCache(source_attention.indexer.summary_identity),
        QSAKVCache(source_attention.indexer.summary_identity),
    ]

    class WiderPreOAttention:
        num_heads = 3
        head_dim = 16

        def _project_segmented_qsa(self, hidden):
            return (hidden, hidden, hidden, hidden)

        def __call__(self, hidden, _mask, _cache, *, _projected, _return_pre_o):
            assert _return_pre_o
            shape = (*hidden.shape[:-1], self.num_heads * self.head_dim)
            return mx.ones(shape, hidden.dtype), mx.zeros(shape, hidden.dtype)

        @staticmethod
        def o_proj(value):
            return value

    segmented = SegmentedBatchQSAKVCache(rows, shared_qsa_prefix=False)
    segmented.prepare(lengths=[3, 0], right_padding=[0, 3])
    hidden = mx.random.normal((2, 3, 32), key=mx.random.key(1001))
    actual = segmented.segmented_attention(WiderPreOAttention(), hidden, None)
    mx.eval(actual)

    assert actual.shape == (2, 3, 48)
    assert mx.allclose(actual[0], mx.full((3, 48), 0.5)).item()
    assert mx.array_equal(actual[1], mx.zeros((3, 48))).item()


def test_segmented_qsa_preserves_ragged_row_local_shared_topk():
    from test_batched_mtp import _tiny_qwen4_model
    from mlx2.runtime.models.qwen4_exp import QSAKVCache
    from mlx2.runtime.segmented_batch_cache import SegmentedBatchQSAKVCache

    model = _tiny_qwen4_model()
    attention = model.language_model.model.layers[1].self_attn
    rows = [
        QSAKVCache(attention.indexer.summary_identity),
        QSAKVCache(attention.indexer.summary_identity),
    ]
    segmented = SegmentedBatchQSAKVCache(rows, shared_qsa_prefix=False)
    segmented._mtp_share_topk = True
    rows[0]._mtp_shared_topk = mx.array([[1, 2]], dtype=mx.uint32)
    rows[1]._mtp_shared_topk = mx.array([[3, 4, 5]], dtype=mx.uint32)

    segmented._capture_row_qsa_share()
    assert segmented._mtp_shared_topk is None
    segmented._arm_row_qsa_share()
    assert rows[0]._mtp_shared_topk.shape == (1, 2)
    assert rows[1]._mtp_shared_topk.shape == (1, 3)

    segmented.release_qsa_cycle("test")
    assert all(row._mtp_shared_topk is None for row in rows)
    assert all(not row._mtp_share_topk for row in rows)


def test_segmented_qsa_keeps_same_shape_different_grid_topk_row_local():
    from test_batched_mtp import _tiny_qwen4_model
    from mlx2.runtime.models.qwen4_exp import QSAKVCache
    from mlx2.runtime.segmented_batch_cache import SegmentedBatchQSAKVCache

    model = _tiny_qwen4_model()
    attention = model.language_model.model.layers[1].self_attn
    rows = [
        QSAKVCache(attention.indexer.summary_identity),
        QSAKVCache(attention.indexer.summary_identity),
    ]
    segmented = SegmentedBatchQSAKVCache(rows, shared_qsa_prefix=False)
    segmented._mtp_share_topk = True
    rows[0]._mtp_shared_topk = mx.array([[1, 2]], dtype=mx.uint32)
    rows[0]._mtp_shared_topk_n_blocks = 4
    rows[1]._mtp_shared_topk = mx.array([[2, 3]], dtype=mx.uint32)
    rows[1]._mtp_shared_topk_n_blocks = 5

    segmented._capture_row_qsa_share()
    assert segmented._mtp_shared_topk is None
    assert segmented._mtp_shared_topk_n_blocks is None

    segmented._arm_row_qsa_share()
    assert rows[0]._mtp_shared_topk_n_blocks == 4
    assert rows[1]._mtp_shared_topk_n_blocks == 5

    segmented.release_qsa_cycle("test")


def test_segmented_qsa_aggregates_same_shape_common_grid_topk():
    from test_batched_mtp import _tiny_qwen4_model
    from mlx2.runtime.models.qwen4_exp import QSAKVCache
    from mlx2.runtime.segmented_batch_cache import SegmentedBatchQSAKVCache

    model = _tiny_qwen4_model()
    attention = model.language_model.model.layers[1].self_attn
    rows = [
        QSAKVCache(attention.indexer.summary_identity),
        QSAKVCache(attention.indexer.summary_identity),
    ]
    segmented = SegmentedBatchQSAKVCache(rows, shared_qsa_prefix=False)
    segmented._mtp_share_topk = True
    rows[0]._mtp_shared_topk = mx.array([[1, 2]], dtype=mx.uint32)
    rows[1]._mtp_shared_topk = mx.array([[2, 3]], dtype=mx.uint32)
    rows[0]._mtp_shared_topk_n_blocks = rows[1]._mtp_shared_topk_n_blocks = 6

    segmented._capture_row_qsa_share()

    assert segmented._mtp_shared_topk.shape == (2, 2)
    assert segmented._mtp_shared_topk_n_blocks == 6
    segmented.release_qsa_cycle("test")


def test_exact_set_preflight_decline_uses_proven_private_path(monkeypatch):
    from test_batched_mtp import _tiny_qwen4_model
    from mlx2.runtime.models.qwen4_exp import QSAKVCache
    from mlx2.runtime.segmented_batch_cache import SegmentedBatchQSAKVCache
    from mlx2.runtime.segmented_self_mtp import note_segmented_self_mtp

    segmented_self_mtp_stats(reset=True)
    monkeypatch.setenv("MLX_LM_QSA_PRIVATE_DELTA_MIN_CONTEXT", "0")
    monkeypatch.setenv("MLX_LM_QSA_PRIVATE_DELTA_EXACT_SET_FOLD", "1")
    model = _tiny_qwen4_model()
    attention = model.language_model.model.layers[1].self_attn
    prefix = mx.random.normal((1, 12, 32), key=mx.random.key(111))
    prefix_mask = (mx.arange(12)[:, None] >= mx.arange(12)[None, :])[None, None]
    rows = []
    for _ in range(2):
        cache = QSAKVCache(attention.indexer.summary_identity)
        mx.eval(attention(prefix, prefix_mask, cache))
        rows.append(cache)
    segmented = SegmentedBatchQSAKVCache(
        rows,
        note=note_segmented_self_mtp,
        shared_qsa_prefix=True,
    )
    segmented.prepare(lengths=[3, 3], right_padding=[0, 0])
    hidden = mx.random.normal((2, 3, 32), key=mx.random.key(112))
    sentinel = mx.zeros_like(hidden)
    with (
        patch(
            "mlx2.runtime.segmented_batch_cache.qwen4_qsa_indexed_private_delta_preflight",
            return_value=(True, "engaged"),
        ),
        patch(
            "mlx2.runtime.segmented_batch_cache."
            "qwen4_qsa_indexed_private_delta_exact_set_preflight",
            return_value=(False, "synthetic_exact_preflight_decline"),
        ),
        patch.object(
            segmented,
            "_private_delta_attention",
            return_value=sentinel,
        ) as consume,
    ):
        actual = segmented.segmented_attention(attention, hidden, None)

    assert actual is sentinel
    consume.assert_called_once_with(attention, hidden, exact_set_fold=False)
    counters = segmented_self_mtp_stats()
    assert counters["exact_set_fold_requests"] == 1
    assert counters["exact_set_fold_declines"] == 1
    assert counters["exact_set_fold_preflight_declines"] == 1
    assert counters["exact_set_fold_device_proofs"] == 0
    assert counters["exact_set_fold_decline_reasons"] == {
        "synthetic_exact_preflight_decline": 1
    }


def test_exact_set_success_records_engagement_and_updates_each_row_once(monkeypatch):
    from test_batched_mtp import _tiny_qwen4_model
    from qsa_oracle import private_delta_reference
    from mlx2.runtime.models.qwen4_exp import QSAKVCache
    from mlx2.runtime.segmented_batch_cache import SegmentedBatchQSAKVCache
    from mlx2.runtime.segmented_self_mtp import note_segmented_self_mtp

    segmented_self_mtp_stats(reset=True)
    monkeypatch.setenv("MLX_LM_QSA_PRIVATE_DELTA_MIN_CONTEXT", "0")
    monkeypatch.setenv("MLX_LM_QSA_PRIVATE_DELTA_EXACT_SET_FOLD", "1")
    model = _tiny_qwen4_model()
    attention = model.language_model.model.layers[1].self_attn
    prefix = mx.random.normal((1, 12, 32), key=mx.random.key(121))
    prefix_mask = (mx.arange(12)[:, None] >= mx.arange(12)[None, :])[None, None]
    rows = []
    for _ in range(2):
        cache = QSAKVCache(attention.indexer.summary_identity)
        mx.eval(attention(prefix, prefix_mask, cache))
        rows.append(cache)

    updates = {"index": 0, "kv": 0}
    for cache in rows:
        original_index = cache.update_index_keys
        original_kv = cache.update_and_fetch

        def counted_index(keys, *, _original=original_index):
            updates["index"] += 1
            return _original(keys)

        def counted_kv(keys, values, *, _original=original_kv):
            updates["kv"] += 1
            return _original(keys, values)

        cache.update_index_keys = counted_index
        cache.update_and_fetch = counted_kv

    segmented = SegmentedBatchQSAKVCache(
        rows,
        note=note_segmented_self_mtp,
        shared_qsa_prefix=True,
    )
    segmented.prepare(lengths=[3, 3], right_padding=[0, 0])
    hidden = mx.random.normal((2, 3, 32), key=mx.random.key(122))

    def exact_reference(
        q,
        base_k,
        base_v,
        delta_k,
        delta_v,
        _lengths,
        compact,
        *,
        scale,
    ):
        return private_delta_reference(
            q,
            base_k,
            base_v,
            delta_k,
            delta_v,
            compact,
            scale=scale,
            splits=8,
        )

    with (
        patch(
            "mlx2.runtime.segmented_batch_cache.qwen4_qsa_indexed_private_delta_preflight",
            return_value=(True, "engaged"),
        ),
        patch(
            "mlx2.runtime.segmented_batch_cache."
            "qwen4_qsa_indexed_private_delta_exact_set_preflight",
            return_value=(True, "engaged"),
        ),
        patch(
            "mlx2.runtime.segmented_batch_cache."
            "qwen4_qsa_indexed_private_delta_exact_set_attention",
            side_effect=exact_reference,
        ) as folded,
        patch(
            "mlx2.runtime.segmented_batch_cache.qwen4_qsa_indexed_private_delta_attention"
        ) as proven,
    ):
        actual = segmented.segmented_attention(attention, hidden, None)
    mx.eval(actual)

    assert bool(mx.all(mx.isfinite(actual)).item())
    assert updates == {"index": 2, "kv": 2}
    assert folded.call_count == 1
    proven.assert_not_called()
    counters = segmented_self_mtp_stats()
    assert counters["private_delta_requests"] == 1
    assert counters["private_delta_attention_calls"] == 1
    assert counters["exact_set_fold_requests"] == 1
    assert counters["exact_set_fold_device_proofs"] == 1
    assert counters["exact_set_fold_attention_calls"] == 1
    assert counters["exact_set_fold_rows"] == 2
    assert counters["exact_set_fold_declines"] == 0


@pytest.mark.parametrize("accepts", list(product(range(3), repeat=2)))
def test_true_batched_segmented_matches_serial_b1_oracle(monkeypatch, accepts):
    from test_batched_mtp import _prepare_lane, _tiny_qwen4_model

    model = _tiny_qwen4_model()
    prompts = ([1, 2, 3, 4], [5, 6, 7, 8, 9])

    def run(true_batched):
        monkeypatch.setenv(
            "MLX_LM_TRUE_BATCHED_SEGMENTED_MTP", "1" if true_batched else "0"
        )
        detached = [
            _prepare_lane(model, uid, prompt) for uid, prompt in enumerate(prompts)
        ]
        state = attach_segmented_self_mtp_lanes(model, None, detached)
        accepted = iter(accepts)

        def force(logprobs, *_args, **_kwargs):
            count = next(accepted)
            return count, int(mx.argmax(logprobs[count]).item())

        with patch(
            "mlx2.runtime.hybrid_speculative._batched_residual_verify",
            side_effect=force,
        ):
            proposal = propose_batched_self_mtp(model, state)
        commit_batched_self_mtp(
            state,
            proposal,
            emitted_counts=[len(row) for row in proposal.outputs],
            terminal=[False, False],
        )
        state, rows = detach_self_mtp_lanes(model, state, [0, 1])
        return proposal, rows

    serial, serial_rows = run(False)
    segmented, segmented_rows = run(True)
    assert segmented.lane_uids == serial.lane_uids
    assert segmented.draft_depths == serial.draft_depths
    assert segmented.accepted_lengths == serial.accepted_lengths
    assert tuple(tuple(token.token for token in row) for row in segmented.outputs) == (
        tuple(tuple(token.token for token in row) for row in serial.outputs)
    )
    for expected, actual in zip(serial_rows, segmented_rows):
        assert actual.lane.cur == expected.lane.cur
        assert actual.lane.pending_ts == expected.lane.pending_ts
        assert mx.allclose(
            actual.lane.seed_h,
            expected.lane.seed_h,
            rtol=1e-5,
            atol=1e-6,
        ).item()
        assert mx.array_equal(actual.lane.rng.key, expected.lane.rng.key).item()
        assert actual.lane.stats.draft_proposed == expected.lane.stats.draft_proposed
        assert actual.lane.stats.draft_accepted == expected.lane.stats.draft_accepted
        for expected_cache, actual_cache in zip(
            expected.caches.target + expected.caches.draft,
            actual.caches.target + actual.caches.draft,
        ):
            expected_arrays = list(_tree_arrays(expected_cache.state))
            actual_arrays = list(_tree_arrays(actual_cache.state))
            assert len(actual_arrays) == len(expected_arrays)
            for expected_array, actual_array in zip(expected_arrays, actual_arrays):
                if expected_array.dtype in (mx.uint32, mx.int32, mx.int64):
                    assert mx.array_equal(actual_array, expected_array).item()
                else:
                    assert mx.allclose(
                        actual_array, expected_array, rtol=1e-5, atol=1e-6
                    ).item()


def _qsa_prefix_snapshot(rows):
    from mlx2.runtime.models.qwen4_exp import QSAKVCache

    snapshots = {}
    for row, item in enumerate(rows):
        for group_name, caches in (
            ("target", item.caches.target),
            ("draft", item.caches.draft),
        ):
            for layer, cache in enumerate(caches):
                if not isinstance(cache, QSAKVCache):
                    continue
                width = int(cache.offset)
                values = {
                    "keys": None
                    if cache.keys is None
                    else mx.array(cache.keys[..., :width, :]),
                    "values": (
                        None
                        if cache.values is None
                        else mx.array(cache.values[..., :width, :])
                    ),
                    "index_keys": (
                        None
                        if cache.index_keys is None
                        else mx.array(cache.index_keys[:, :width])
                    ),
                }
                mx.eval(*(value for value in values.values() if value is not None))
                snapshots[(row, group_name, layer)] = (width, values)
    return snapshots


def _assert_qsa_prefix_unchanged(snapshots, rows):
    from oracle_helpers import _array_atom

    for (row, group_name, layer), (width, values) in snapshots.items():
        cache = getattr(rows[row].caches, group_name)[layer]
        assert int(cache.offset) >= width
        current = {
            "keys": None if cache.keys is None else cache.keys[..., :width, :],
            "values": None if cache.values is None else cache.values[..., :width, :],
            "index_keys": (
                None if cache.index_keys is None else cache.index_keys[:, :width]
            ),
        }
        for name, expected in values.items():
            actual = current[name]
            assert (actual is None) == (expected is None)
            if expected is not None:
                assert _array_atom(actual) == _array_atom(expected), (
                    row,
                    group_name,
                    layer,
                    name,
                )


def _instrument_qsa_updates(rows):
    from mlx2.runtime.models.qwen4_exp import QSAKVCache

    counts = {
        "target_index": 0,
        "target_kv": 0,
        "draft_index": 0,
        "draft_kv": 0,
    }
    for item in rows:
        for group_name, caches in (
            ("target", item.caches.target),
            ("draft", item.caches.draft),
        ):
            for cache in caches:
                if not isinstance(cache, QSAKVCache):
                    continue
                original_index = cache.update_index_keys
                original_kv = cache.update_and_fetch

                def counted_index(keys, *, _group=group_name, _original=original_index):
                    counts[f"{_group}_index"] += 1
                    return _original(keys)

                def counted_kv(
                    keys, values, *, _group=group_name, _original=original_kv
                ):
                    counts[f"{_group}_kv"] += 1
                    return _original(keys, values)

                cache.update_index_keys = counted_index
                cache.update_and_fetch = counted_kv
    return counts


def _assert_detached_rows_oracle(expected_rows, actual_rows):
    from oracle_helpers import (
        _array_atom,
        capture_cache_list,
    )

    assert len(actual_rows) == len(expected_rows)
    for expected, actual in zip(expected_rows, actual_rows):
        for name in ("uid", "cur", "ntoks", "pending_ts"):
            assert getattr(actual.lane, name) == getattr(expected.lane, name)
        for name in ("seed_h", "pending_hs", "token_prefix"):
            expected_value = getattr(expected.lane, name)
            actual_value = getattr(actual.lane, name)
            assert (actual_value is None) == (expected_value is None)
            if expected_value is not None:
                assert mx.allclose(
                    actual_value, expected_value, rtol=1e-5, atol=1e-6
                ).item()
        assert _array_atom(actual.lane.rng.key) == _array_atom(expected.lane.rng.key)
        assert vars(actual.lane.stats) == vars(expected.lane.stats)

        for group_name in ("target", "draft"):
            expected_caches = getattr(expected.caches, group_name)
            actual_caches = getattr(actual.caches, group_name)
            expected_capture = capture_cache_list(expected_caches, group_name)
            actual_capture = capture_cache_list(actual_caches, group_name)
            assert set(actual_capture) == set(expected_capture)
            for path, expected_atom in expected_capture.items():
                actual_atom = actual_capture[path]
                assert actual_atom.kind == expected_atom.kind, path
                assert actual_atom.dtype == expected_atom.dtype, path
                assert actual_atom.shape == expected_atom.shape, path
                if not (
                    expected_atom.kind == "array" and "float" in expected_atom.dtype
                ):
                    assert actual_atom == expected_atom, path

            for expected_cache, actual_cache in zip(expected_caches, actual_caches):
                expected_arrays = list(_tree_arrays(expected_cache.state))
                actual_arrays = list(_tree_arrays(actual_cache.state))
                assert len(actual_arrays) == len(expected_arrays)
                for expected_array, actual_array in zip(expected_arrays, actual_arrays):
                    if "float" in str(expected_array.dtype):
                        assert mx.allclose(
                            actual_array,
                            expected_array,
                            rtol=1e-5,
                            atol=1e-6,
                        ).item()
                    else:
                        assert _array_atom(actual_array) == _array_atom(expected_array)


def test_private_qsa_repeated_cycles_have_bounded_state_oracle(monkeypatch):
    from test_batched_mtp import _prepare_lane, _tiny_qwen4_model

    # The oracle asserts that two deterministic lane trajectories actually
    # diverge after their shared prefix.  Pin the randomly initialized tiny
    # model so suite order cannot occasionally produce a degenerate model
    # whose greedy suffixes remain identical.
    mx.random.seed(0)
    model = _tiny_qwen4_model()
    prompts = ([1, 2, 3, 4], [1, 2, 3, 4])
    accept_cycles = ((0, 0), (1, 1), (2, 2), (1, 2), (2, 0))

    def run(private_delta):
        monkeypatch.setenv("MLX_LM_TRUE_BATCHED_SEGMENTED_MTP", "1")
        monkeypatch.setenv("MLX_LM_QSA_PRIVATE_DELTA", "1" if private_delta else "0")
        monkeypatch.setenv("MLX_LM_QSA_PRIVATE_DELTA_MIN_CONTEXT", "0")
        rows = [_prepare_lane(model, uid, prompt) for uid, prompt in enumerate(prompts)]
        for item in rows:
            item.lane.max_tokens = 64
            item.lane.sampling_temp = 0.0
            item.shared_qsa_prefix_id = "identical-four-token-prefix"
        base = _qsa_prefix_snapshot(rows)
        counts = _instrument_qsa_updates(rows) if private_delta else None
        state = attach_segmented_self_mtp_lanes(model, None, rows)
        private_calls = 0
        outputs = []

        def decline_private(*_args, **_kwargs):
            nonlocal private_calls
            private_calls += 1
            assert counts["target_index"] == 2 * private_calls
            assert counts["target_kv"] == 2 * private_calls
            raise RuntimeError("synthetic exact private-delta dispatch decline")

        private_patches = (
            patch(
                "mlx2.runtime.segmented_batch_cache."
                "qwen4_qsa_indexed_private_delta_preflight",
                return_value=(True, "engaged"),
            ),
            patch(
                "mlx2.runtime.segmented_batch_cache."
                "qwen4_qsa_indexed_private_delta_attention",
                side_effect=decline_private,
            ),
        )
        with private_patches[0], private_patches[1]:
            for cycle, accepts in enumerate(accept_cycles):
                if cycle == 3:
                    state, detached = detach_self_mtp_lanes(model, state, [0, 1])
                    assert state.shared_qsa_prefix_id is None
                    state = attach_segmented_self_mtp_lanes(model, state, detached)
                    monkeypatch.setenv("MLX_LM_TRUE_BATCHED_SEGMENTED_MTP", "0")
                elif cycle == 4:
                    monkeypatch.setenv("MLX_LM_TRUE_BATCHED_SEGMENTED_MTP", "1")
                    calls_before_reenable = private_calls

                accepted = iter(accepts)

                def force(logprobs, *_args, **_kwargs):
                    count = next(accepted)
                    return count, int(mx.argmax(logprobs[count]).item())

                with patch(
                    "mlx2.runtime.hybrid_speculative._batched_residual_verify",
                    side_effect=force,
                ):
                    proposal = propose_batched_self_mtp(model, state)
                outputs.append(
                    tuple(
                        tuple(token.token for token in row) for row in proposal.outputs
                    )
                )
                commit_batched_self_mtp(
                    state,
                    proposal,
                    emitted_counts=[len(row) for row in proposal.outputs],
                    terminal=[False, False],
                )

                if cycle == 4:
                    assert private_calls == calls_before_reenable

        state, rows = detach_self_mtp_lanes(model, state, [0, 1])
        _assert_qsa_prefix_unchanged(base, rows)
        close_segmented_self_mtp_state(state)
        return outputs, rows, counts, private_calls

    expected_outputs, expected_rows, _, _ = run(False)
    actual_outputs, actual_rows, counts, private_calls = run(True)
    assert actual_outputs == expected_outputs
    assert private_calls > 0
    assert counts["target_index"] == counts["target_kv"]
    assert counts["draft_index"] == counts["draft_kv"]
    _assert_detached_rows_oracle(expected_rows, actual_rows)

    # Equal prompt prefixes stay bit-identical, while the private suffixes
    # diverge after the two RNG streams produce different live tokens.
    from oracle_helpers import _array_atom

    for group_name in ("target", "draft"):
        left = next(
            cache
            for cache in getattr(actual_rows[0].caches, group_name)
            if hasattr(cache, "index_keys")
        )
        right = next(
            cache
            for cache in getattr(actual_rows[1].caches, group_name)
            if hasattr(cache, "index_keys")
        )
        base_width = 4 if group_name == "target" else 3
        assert _array_atom(left.keys[..., :base_width, :]) == _array_atom(
            right.keys[..., :base_width, :]
        )
        assert _array_atom(left.keys[..., base_width:, :]) != _array_atom(
            right.keys[..., base_width:, :]
        )


def test_private_qsa_abort_preserves_base_and_restores_committed_boundary(monkeypatch):
    from test_batched_mtp import _prepare_lane, _tiny_qwen4_model

    monkeypatch.setenv("MLX_LM_TRUE_BATCHED_SEGMENTED_MTP", "1")
    monkeypatch.setenv("MLX_LM_QSA_PRIVATE_DELTA", "1")
    monkeypatch.setenv("MLX_LM_QSA_PRIVATE_DELTA_MIN_CONTEXT", "0")
    model = _tiny_qwen4_model()
    rows = [_prepare_lane(model, uid, [1, 2, 3, 4]) for uid in range(2)]
    for item in rows:
        item.shared_qsa_prefix_id = "abort-base"
    base = _qsa_prefix_snapshot(rows)
    counts = _instrument_qsa_updates(rows)
    state = attach_segmented_self_mtp_lanes(model, None, rows)
    private_calls = 0

    def decline_private(*_args, **_kwargs):
        nonlocal private_calls
        private_calls += 1
        assert counts["target_index"] == 2 * private_calls
        assert counts["target_kv"] == 2 * private_calls
        raise RuntimeError("synthetic exact private-delta dispatch decline")

    accepted = iter((0, 2))

    def force(logprobs, *_args, **_kwargs):
        count = next(accepted)
        return count, int(mx.argmax(logprobs[count]).item())

    with (
        patch(
            "mlx2.runtime.segmented_batch_cache."
            "qwen4_qsa_indexed_private_delta_preflight",
            return_value=(True, "engaged"),
        ),
        patch(
            "mlx2.runtime.segmented_batch_cache."
            "qwen4_qsa_indexed_private_delta_attention",
            side_effect=decline_private,
        ),
        patch(
            "mlx2.runtime.hybrid_speculative._batched_residual_verify",
            side_effect=force,
        ),
    ):
        proposal = propose_batched_self_mtp(model, state)

    _assert_qsa_prefix_unchanged(base, rows)
    assert counts["target_index"] == counts["target_kv"] == 2 * private_calls
    assert counts["draft_index"] == counts["draft_kv"]
    abort_batched_self_mtp(state, proposal, cause=RuntimeError("client gone"))
    assert state.poisoned is False
    assert state.poison_reason is None
    assert state.proposal_open is False
    assert state._open_proposal is None
    assert state._batched_state is None
    assert state._segmented_caches is None
    assert state._transaction_branches == []
    state.transactions[0].validate(
        state.row_caches[0], state.lanes[0], state.transactions[0].position
    )
    close_segmented_self_mtp_state(state)


def test_true_batched_segmented_shared_qsa_cycle_is_disarmed():
    from test_batched_mtp import _prepare_lane, _tiny_qwen4_model

    model = _tiny_qwen4_model()
    detached = [
        _prepare_lane(
            model,
            uid,
            prompt,
            share_qsa_indices=True,
        )
        for uid, prompt in enumerate(([1, 2, 3, 4], [5, 6, 7, 8, 9]))
    ]
    state = attach_segmented_self_mtp_lanes(model, None, detached)
    original_mtp_step = model.mtp_step
    mtp_calls = 0

    def checked_mtp_step(hidden, tokens, cache):
        nonlocal mtp_calls
        qsa = cache[0]
        if mtp_calls == 1:
            assert qsa._mtp_shared_topk is not None
            assert qsa._mtp_shared_topk.shape[0] == 2
            assert all(row._mtp_shared_topk is not None for row in qsa.rows)
        output = original_mtp_step(hidden, tokens, cache)
        mtp_calls += 1
        return output

    model.mtp_step = checked_mtp_step
    proposal = propose_batched_self_mtp(model, state)
    model.mtp_step = original_mtp_step
    assert mtp_calls == 2
    assert state._batched_state is not None
    qsa = state._batched_state.caches.draft[0]
    assert qsa._mtp_share_topk is False
    assert qsa._mtp_shared_topk is None
    commit_batched_self_mtp(
        state,
        proposal,
        emitted_counts=[len(row) for row in proposal.outputs],
        terminal=[False, False],
    )
    for pair in state.row_caches:
        for cache in pair.draft:
            assert cache._mtp_share_topk is False
            assert cache._mtp_shared_topk is None
    close_segmented_self_mtp_state(state)

    zero = _prepare_lane(model, 3, [11, 12, 13, 14])
    state = attach_segmented_self_mtp_lanes(model, None, [zero])

    def reject_all(logprobs, *_args, **_kwargs):
        return 0, int(mx.argmax(logprobs[0]).item())

    with patch(
        "mlx2.runtime.hybrid_speculative._batched_residual_verify",
        side_effect=reject_all,
    ):
        proposal = propose_batched_self_mtp(model, state)
    commit_batched_self_mtp(
        state,
        proposal,
        emitted_counts=[0],
        terminal=[True],
    )
    assert state.transactions[0].position == 4
    assert state.transactions[0].predecessor_lineage_id is not None
    close_segmented_self_mtp_state(state)


def test_async_promotion_reenabled_after_short_cohort_drains(monkeypatch):
    from mlx2.runtime.generate import MTPGenerationBatch, StopSequenceMatcher
    monkeypatch.setattr(MTPGenerationBatch, "_arm_async_qsa_promotion", lambda self: None)
    destination = MTPGenerationBatch.empty(object(), segmented_live_tip=True, async_qsa_promotion=True)
    short = MTPGenerationBatch(object(), [_detached(0)], [None], [StopSequenceMatcher()],
        segmented_live_tip=True, async_qsa_promotion=False)
    destination.extend(short)
    assert not destination.async_qsa_promotion
    destination.remove_uids([0])
    long = MTPGenerationBatch(object(), [_detached(1)], [None], [StopSequenceMatcher()],
        segmented_live_tip=True, async_qsa_promotion=True)
    destination.extend(long)
    assert destination.async_qsa_promotion and destination._async_qsa_pending
    destination.close()


def _promoted_physical_cohort():
    """A segmented B2 cohort after its first uniform cycle promoted it."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from mlx2.runtime.generate import MTPGenerationBatch, StopSequenceMatcher
    from mlx2.runtime.segmented_physical_promotion import (
        SegmentedPhysicalPromotionReceipt,
    )

    segmented_self_mtp_stats(reset=True)
    holder = {}

    class Ticket:
        def finish(self):
            state = holder["state"]
            physical = SimpleNamespace(
                lanes=list(state.lanes),
                caches=SimpleNamespace(target=[], draft=[]),
                proposal_open=False,
            )
            return physical, SegmentedPhysicalPromotionReceipt(
                queued_ns=1, finish_ns=2, stream_wait_ns=1, reserved_bytes=1,
                patched_bytes=1, recurrent_arrays_reused=1, cleanup_error_count=0,
                rows=2, layers=2, advance=1,
            )

    def begin(state, **_):
        holder["state"] = state
        return Ticket()

    with patch(
        "mlx2.runtime.segmented_physical_promotion.begin_segmented_physical_promotion",
        side_effect=begin,
    ):
        batch = MTPGenerationBatch(
            object(), [_detached(0), _detached(1)], [None, None],
            [StopSequenceMatcher(), StopSequenceMatcher()],
            segmented_live_tip=True, async_qsa_promotion=True,
        )
    with patch(
        "mlx2.runtime.hybrid_speculative._propose_batched_self_mtp_impl",
        side_effect=_fake_row_proposal({0: 1, 1: 1}),
    ):
        batch.next()
    assert batch.segmented_live_tip is False and batch.async_qsa_promotion is True
    return batch


def test_short_output_arrival_joins_a_promoted_cohort_without_raising():
    from mlx2.runtime.generate import (
        MTPGenerationBatch,
        _segmented_async_qsa_promotion_for_budget,
    )

    """A ≤16-token arrival's budget gate returns False; the promoted cohort's
    flag must stay authoritative (it re-arms segmented admission once empty)
    and the join must not take the worker down."""
    dest = _promoted_physical_cohort()
    policy = {"segment_aware_live_tip": True, "segment_aware_async_qsa_promotion": True}
    assert _segmented_async_qsa_promotion_for_budget(policy, 7) is False
    arriving = MTPGenerationBatch.empty(
        object(), segmented_live_tip=False, async_qsa_promotion=False
    )
    dest.extend(arriving)
    assert dest.async_qsa_promotion is True
    assert segmented_self_mtp_stats()["physical_join_flag_reconciled"] == 1


def test_arrival_into_an_empty_physical_batch_owns_the_promotion_flag():
    from mlx2.runtime.generate import MTPGenerationBatch

    dest = MTPGenerationBatch.empty(
        object(), segmented_live_tip=False, async_qsa_promotion=True
    )
    arriving = MTPGenerationBatch.empty(
        object(), segmented_live_tip=False, async_qsa_promotion=False
    )
    dest.extend(arriving)
    assert dest.async_qsa_promotion is False


def test_width_lock_deferral_is_reported_as_scheduler_waiting_not_memory():
    """A lane deferred by the live segmented width lock emits nothing until the
    cohort drains.  It must be visible as scheduler-waiting so the serving
    watchdog does not fail it with a memory-admission error."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_batched_mtp import _tiny_qwen4_model

    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.memory_policy import (
        SelfMTPLaneAdmissionController,
        _make_self_mtp_admission_callback,
    )

    mx.random.seed(7)
    segmented_self_mtp_stats(reset=True)
    model = _tiny_qwen4_model()
    batch = BatchGenerator(
        model, completion_batch_size=4, prefill_step_size=2048,
        self_mtp={
            "num_draft": 2, "persistent": True, "segment_aware_live_tip": True,
            "segment_aware_cohort_size": 4, "segment_aware_async_qsa_promotion": False,
        },
        mtp_admission=_make_self_mtp_admission_callback(
            SelfMTPLaneAdmissionController(saturation_lane_cap=4),
            free_memory=lambda: 100.0, max_draft=2,
        ),
    )
    try:
        a, b = batch.insert([[1, 2, 3, 4], [5, 6, 7, 8]], max_tokens=[40, 40])
        for _ in range(3):
            batch.next()
        (late,) = batch.insert([[9, 10, 11]], max_tokens=[3])
        seen_late = False
        waiting_observed = False
        for _ in range(200):
            _prompts, responses = batch.next()
            if late in batch.scheduler_waiting_uids():
                waiting_observed = True
                assert late not in batch._generation_batch._memory_queued
            if any(r.uid == late for r in responses):
                seen_late = True
                break
        assert seen_late, "late arrival never decoded"
        assert waiting_observed, "deferred lane was never reported as scheduler-waiting"
        assert segmented_self_mtp_stats()["live_width_change_deferrals"] > 0
        assert late not in batch.scheduler_waiting_uids()
    finally:
        batch.close()


def test_width_lock_deferral_is_bounded_and_falls_back_to_plain_decode():
    """A lane a live width lock keeps deferring continues as ordinary decode
    beside the cohort after WIDTH_LOCK_DEFERRALS_BEFORE_PLAIN attempts."""
    from mlx2.runtime.generate import (
        WIDTH_LOCK_DEFERRALS_BEFORE_PLAIN,
        MTPGenerationBatch,
        StopSequenceMatcher,
    )

    segmented_self_mtp_stats(reset=True)
    active = MTPGenerationBatch(
        object(), [_detached(0), _detached(1)], [None, None],
        [StopSequenceMatcher(), StopSequenceMatcher()], segmented_live_tip=True,
    )
    arriving = MTPGenerationBatch(
        object(), [_detached(2)], [None], [StopSequenceMatcher()], segmented_live_tip=True,
    )
    active._segmented_compute_width_locked = True
    active.extend(arriving)
    assert list(active._paused) == [2]
    for attempt in range(1, WIDTH_LOCK_DEFERRALS_BEFORE_PLAIN):
        package = active._paused.pop(2)
        active._attach_packages([package])
        assert list(active._paused) == [2], attempt
    stats = segmented_self_mtp_stats()
    assert stats["live_width_change_deferrals"] == WIDTH_LOCK_DEFERRALS_BEFORE_PLAIN
    assert stats["width_lock_plain_fallbacks"] == 0
    # One more deferral crosses the bound: the lane leaves the paused set for
    # the plain-decode hand-off while the cohort keeps its lock.
    package = active._paused.pop(2)
    active._attach_packages([package])
    assert list(active._paused) == []
    assert [p.detached.lane.uid for p in active._plain_ready] == [2]
    assert active._segmented_compute_width_locked is True
    assert segmented_self_mtp_stats()["width_lock_plain_fallbacks"] == 1
    assert active.scheduler_waiting_uids() == []
    active.close()


def test_width_lock_handoff_moves_active_and_late_lanes_atomically():
    from mlx2.runtime.adaptive_policy import MTPOrdinaryHandoffPolicy
    from mlx2.runtime.generate import MTPGenerationBatch, StopSequenceMatcher

    scheduler = {}
    policy = MTPOrdinaryHandoffPolicy.from_value({"enabled": True, "max_mtp_width": 2})
    active = MTPGenerationBatch(
        object(),
        [_detached(0), _detached(1)],
        [None, None],
        [StopSequenceMatcher(), StopSequenceMatcher()],
        segmented_live_tip=True,
        ordinary_handoff_policy=policy,
        scheduler_stats=scheduler,
    )
    arriving = MTPGenerationBatch(
        object(),
        [_detached(2), _detached(3)],
        [None, None],
        [StopSequenceMatcher(), StopSequenceMatcher()],
        segmented_live_tip=True,
        ordinary_handoff_policy=policy,
        scheduler_stats=scheduler,
    )
    active._segmented_compute_width_locked = True
    active.extend(arriving)

    assert active._ordinary_handoff_latched
    assert not active.state.lanes
    assert not active._paused
    assert [p.detached.lane.uid for p in active._plain_ready] == [0, 1, 2, 3]
    assert all(
        p.handoff_receipt["reason"] == "segmented_width_lock"
        and p.detached.segment_transaction is None
        for p in active._plain_ready
    )
    assert scheduler["mtp_ordinary_handoff_events"] == 1
    assert scheduler["mtp_ordinary_handoff_lanes"] == 4
    assert scheduler["mtp_ordinary_handoff_segmented_width_lock"] == 1

    # Another prepared lane joins the same one-way migration while latched.
    next_arrival = MTPGenerationBatch(
        object(),
        [_detached(4)],
        [None],
        [StopSequenceMatcher()],
        segmented_live_tip=True,
        ordinary_handoff_policy=policy,
    )
    active.extend(next_arrival)
    assert not active.state.lanes
    assert [p.detached.lane.uid for p in active._plain_ready] == [0, 1, 2, 3, 4]
    assert active._plain_ready[-1].handoff_receipt["reason"] == (
        "cohort_already_handed_off"
    )
    active.close()


def test_handoff_latch_release_never_raises_with_paused_lanes():
    from mlx2.runtime.adaptive_policy import MTPOrdinaryHandoffPolicy
    from mlx2.runtime.generate import MTPGenerationBatch, StopSequenceMatcher

    policy = MTPOrdinaryHandoffPolicy.from_value(
        {"enabled": True, "max_mtp_width": 1}
    )
    batch = MTPGenerationBatch(
        object(),
        [_detached(0), _detached(1)],
        [None, None],
        [StopSequenceMatcher(), StopSequenceMatcher()],
        segmented_live_tip=True,
        ordinary_handoff_policy=policy,
    )
    batch._maybe_handoff_active_cohort()
    batch.take_plain_fallbacks()
    later = MTPGenerationBatch(
        object(),
        [_detached(5)],
        [None],
        [StopSequenceMatcher()],
        segmented_live_tip=True,
        ordinary_handoff_policy=policy,
    )
    batch._paused[5] = later._detach_packages([0])[0]
    assert batch.release_ordinary_handoff_latch() is False
    assert batch._ordinary_handoff_latched
    batch._paused.clear()
    later.close()
    batch.close()


def test_handoff_keeps_memory_queued_lane_under_admission():
    from mlx2.runtime.adaptive_policy import MTPOrdinaryHandoffPolicy
    from mlx2.runtime.generate import MTPGenerationBatch, StopSequenceMatcher

    class Admission:
        atomic_cohort = False

        def __init__(self):
            self.release = False
            self.ordinary_calls = []

        def __call__(self, rows):
            return {
                uid: 1 if uid == 0 or self.release else "queue"
                for (uid, *_rest) in rows
            }

        def preview(self, rows):
            raise AssertionError("handed-off lanes must not use MTP preview costs")

        def at_depth(self, depth):
            assert depth == 0

            def admit(rows):
                self.ordinary_calls.append(tuple(uid for (uid, *_rest) in rows))
                return {
                    uid: "plain" if self.release else "queue"
                    for (uid, *_rest) in rows
                }

            return admit

    admission = Admission()
    policy = MTPOrdinaryHandoffPolicy.from_value(
        {"enabled": True, "max_mtp_width": 1}
    )
    batch = MTPGenerationBatch(
        object(), [_detached(0), _detached(1)], [None, None],
        [StopSequenceMatcher(), StopSequenceMatcher()],
        segmented_live_tip=True, ordinary_handoff_policy=policy,
        mtp_admission=admission,
    )
    assert batch._apply_admission()
    assert list(batch._memory_queued) == [1]
    assert batch._maybe_handoff_active_cohort()
    assert [package.detached.lane.uid
            for package in batch.take_plain_fallbacks()] == [0]
    assert list(batch._paused) == [1]
    assert batch._paused[1].handoff_receipt is not None

    # The queued lane is reconsidered with ordinary, not MTP, memory costs on
    # every closed boundary even while the migrated cohort is still running.
    batch._optional_reclaim_deadline = float("inf")
    assert batch._apply_admission()
    assert batch._apply_admission()
    assert admission.ordinary_calls == [(1,), (1,)]
    admission.release = True
    assert batch._apply_admission()
    assert not batch._paused
    assert [package.detached.lane.uid
            for package in batch.take_plain_fallbacks()] == [1]
    batch.close()


def test_width_lock_handoff_admits_late_lane_before_plain_migration():
    from mlx2.runtime.adaptive_policy import MTPOrdinaryHandoffPolicy
    from mlx2.runtime.generate import MTPGenerationBatch, StopSequenceMatcher

    class Admission:
        atomic_cohort = False

        def __init__(self):
            self.release = False

        def __call__(self, rows):
            return {
                uid: 1 if uid < 2 or self.release else "queue"
                for (uid, *_rest) in rows
            }

    admission = Admission()
    policy = MTPOrdinaryHandoffPolicy.from_value(
        {"enabled": True, "max_mtp_width": 2}
    )
    active = MTPGenerationBatch(
        object(), [_detached(0), _detached(1)], [None, None],
        [StopSequenceMatcher(), StopSequenceMatcher()],
        segmented_live_tip=True, ordinary_handoff_policy=policy,
        mtp_admission=admission,
    )
    arriving = MTPGenerationBatch(
        object(), [_detached(2)], [None], [StopSequenceMatcher()],
        segmented_live_tip=True, ordinary_handoff_policy=policy,
    )
    active._segmented_compute_width_locked = True
    active.extend(arriving)
    assert not active._plain_ready
    assert list(active._paused) == [2]
    admission.release = True
    assert active._apply_admission()
    assert [package.detached.lane.uid
            for package in active.take_plain_fallbacks()] == [0, 1, 2]
    assert not active._paused
    assert not active._memory_queued
    arriving.close()
    active.close()


# --- per-round verification histograms -------------------------------------


def _depth_proposal(depth, accepted):
    """One self-MTP round of ``depth`` drafts of which ``accepted`` verify."""

    def propose(_model, row_state):
        lane = row_state.lanes[0]
        for cache in row_state.caches.target:
            cache.offset += accepted + 1
        drafts = tuple(31 + index for index in range(depth))
        bonus = 51
        logprobs = mx.zeros((depth + 1, 64))
        hidden = mx.zeros((1, depth + 1, 2))
        outputs = tuple(
            [MTPToken(drafts[index], logprobs[index], True) for index in range(accepted)]
            + [MTPToken(bonus, logprobs[accepted], False)]
        )
        proposal = SelfMTPCycleResult(
            membership_epoch=row_state.membership_epoch,
            lane_uids=(lane.uid,),
            draft_depths=(depth,),
            accepted_lengths=(accepted,),
            target_drops=(depth - accepted,),
            head_drops=(depth,),
            outputs=(outputs,),
            _old_curs=(lane.cur,),
            _old_seed_hs=(lane.seed_h,),
            _drafts=(drafts,),
            _vhidden=(hidden,),
            _logprobs=(logprobs,),
            _bonuses=(bonus,),
        )
        row_state.proposal_open = True
        row_state._open_proposal = proposal
        return proposal

    return propose


def _run_self_mtp_rounds(depth, accept_pattern):
    """Commit one round per entry of ``accept_pattern`` and return the stats."""
    state = attach_segmented_self_mtp_lanes(object(), None, [_detached(0)])
    try:
        for accepted in accept_pattern:
            capped = min(int(accepted), depth)
            with patch(
                "mlx2.runtime.hybrid_speculative._propose_batched_self_mtp_impl",
                side_effect=_depth_proposal(depth, capped),
            ):
                proposal = propose_batched_self_mtp(object(), state)
            commit_batched_self_mtp(
                state,
                proposal,
                emitted_counts=[capped + 1],
                terminal=[False],
            )
        return state.lanes[0].stats
    finally:
        close_segmented_self_mtp_state(state)


def _tau_from_hist(hist, cap):
    """Committed tokens per verification forward at draft cap ``cap``."""
    rounds = sum(hist.values())
    committed = sum((min(int(a), cap) + 1) * n for a, n in hist.items())
    return committed / rounds


def test_self_mtp_verify_histograms_sum_to_the_round_count():
    pattern = [0, 1, 4, 2, 4, 3, 0, 4, 1, 2]
    stats = _run_self_mtp_rounds(4, pattern)

    assert sum(stats.verify_accept_hist.values()) == stats.cycles == len(pattern)
    assert sum(stats.verify_span_hist.values()) == stats.cycles
    # Every round drafted the same depth, so the span histogram is one bucket
    # at depth + 1 -- the positions the target verification forward covered.
    assert stats.verify_span_hist == {5: len(pattern)}
    # The accept histogram's first moment is exactly the existing aggregate.
    assert (
        sum(int(a) * n for a, n in stats.verify_accept_hist.items())
        == stats.draft_accepted
        == sum(pattern)
    )
    assert stats.draft_cycles == len(pattern)


def test_tau_truncated_from_one_deep_run_matches_a_measured_shallow_run():
    """The point of the histogram: a depth sweep from a single deep run.

    A draft accepted to depth k would also have been accepted under any
    shallower cap, so tau at every depth below the one that ran follows from
    the deepest run's accept histogram by truncation. Each shallower depth is
    also actually run here, and its tau is measured from its own aggregates,
    so the identity is checked against measurement rather than restated.
    """
    deepest = 4
    pattern = [0, 1, 4, 2, 4, 3, 0, 4, 1, 2]
    deep = _run_self_mtp_rounds(deepest, pattern)

    for cap in range(1, deepest + 1):
        shallow = _run_self_mtp_rounds(cap, pattern)
        measured = (shallow.draft_accepted + shallow.cycles) / shallow.cycles
        derived = _tau_from_hist(deep.verify_accept_hist, cap)
        assert derived == measured
        # and the shallow run's own histogram agrees with its aggregates
        assert _tau_from_hist(shallow.verify_accept_hist, cap) == measured
        assert sum(shallow.verify_accept_hist.values()) == len(pattern)

    # The deepest point is the one that was actually served.
    assert _tau_from_hist(deep.verify_accept_hist, deepest) == (
        deep.draft_accepted + deep.cycles
    ) / deep.cycles
