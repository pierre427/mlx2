# Mined from unified 1e2bc604, Apache-2.0; see docs/PROVENANCE.md.
import copy
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.runtime.hybrid_speculative import (
    DetachedSelfMTPLane,
    SegmentedSelfMTPState,
    SelfMTPCachePair,
)
from mlx2.runtime.models.cache import ArraysCache
from mlx2.runtime.models.qwen4_exp import QSAKVCache, Qwen4ArraysCache
from mlx2.runtime.segmented_physical_promotion import (
    SegmentedPhysicalPromotionDeclined,
    begin_segmented_physical_promotion,
    begin_shared_prefix_physical_promotion,
)


class _Lineage:
    def __init__(self, position):
        self.generation = 0
        self.position = position
        self.closed = False

    def stats(self):
        return {
            "generation": self.generation,
            "position": self.position,
            "closed": self.closed,
        }


class _Transaction:
    def __init__(self, position):
        self.lineage = _Lineage(position)
        self.closed = False

    @property
    def position(self):
        return self.lineage.position

    def advance(self, count):
        self.lineage.position += count
        self.lineage.generation += 1

    def close(self):
        self.closed = True
        self.lineage.closed = True


def _qsa(length, seed):
    cache = QSAKVCache()
    keys = mx.full((1, 1, length, 2), seed, dtype=mx.float32)
    values = mx.full((1, 1, length, 3), seed + 10, dtype=mx.float32)
    cache.update_and_fetch(keys, values)
    cache.update_index_keys(mx.full((1, length, 2), seed + 20, dtype=mx.float32))
    return cache


def _arrays(seed, *, speculate=False):
    cache = Qwen4ArraysCache(2)
    cache.cache = [
        mx.full((1, 2), seed, dtype=mx.float32),
        mx.full((1, 1, 2), seed + 1, dtype=mx.float32),
    ]
    if speculate:
        cache.start_speculation()
    return cache


def _state():
    rows = []
    for index in range(2):
        rows.append(
            SelfMTPCachePair(
                target=[_qsa(3, index + 1), _arrays(index + 1, speculate=True)],
                draft=[_qsa(2, index + 31), _arrays(index + 31)],
            )
        )
    return SegmentedSelfMTPState(
        lanes=[SimpleNamespace(uid=101), SimpleNamespace(uid=202)],
        row_caches=rows,
        transactions=[_Transaction(3), _Transaction(3)],
        membership_epoch=7,
    )


def _commit(state, advances):
    view = state._segmented_caches
    for index, (pair, count) in enumerate(zip(state.row_caches, advances)):
        qsa = pair.target[0]
        qsa.update_and_fetch(
            mx.full((1, 1, count, 2), 50 + index, dtype=mx.float32),
            mx.full((1, 1, count, 3), 60 + index, dtype=mx.float32),
        )
        qsa.update_index_keys(
            mx.full((1, count, 2), 70 + index, dtype=mx.float32)
        )
        state.transactions[index].advance(count)
    # The true segmented consumer owns the authoritative batched recurrent
    # arrays. Simulate its post-commit state and row split.
    view.target[1][0] = mx.array([[111, 112], [121, 122]], dtype=mx.float32)


def test_shared_prefix_prequeue_binds_and_finishes_exactly():
    pair = SelfMTPCachePair(
        target=[_qsa(3, 7), _arrays(7, speculate=True)],
        draft=[_qsa(2, 37), _arrays(37)],
    )
    canonical = DetachedSelfMTPLane(
        lane=SimpleNamespace(uid=101),
        caches=pair,
        shared_qsa_prefix_id="same-prefix",
    )
    staged = begin_shared_prefix_physical_promotion(
        canonical,
        rows=2,
        reserve_tail=3,
        stream=mx.new_stream(mx.cpu),
    )
    sibling_pair = copy.deepcopy(pair)
    state = SegmentedSelfMTPState(
        lanes=[SimpleNamespace(uid=101), SimpleNamespace(uid=202)],
        row_caches=[pair, sibling_pair],
        transactions=[_Transaction(3), _Transaction(3)],
        membership_epoch=7,
        shared_qsa_prefix_id="same-prefix",
    )
    ticket = staged.bind(state)
    _commit(state, [1, 1])
    physical, receipt = ticket.finish()
    mx.eval(physical.caches.target[0].keys)

    assert receipt.rows == 2
    assert receipt.advance == 1
    assert physical.caches.target[0].offset.tolist() == [4, 4]
    assert physical.caches.target[0].keys[:, :, :3].tolist() == [
        pair.target[0].keys[:, :, :3].tolist()[0],
        sibling_pair.target[0].keys[:, :, :3].tolist()[0],
    ]


def test_shared_prefix_prequeue_rejects_changed_attestation_and_drains():
    pair = SelfMTPCachePair(
        target=[_qsa(3, 7), _arrays(7, speculate=True)],
        draft=[_qsa(2, 37), _arrays(37)],
    )
    canonical = DetachedSelfMTPLane(
        lane=SimpleNamespace(uid=101),
        caches=pair,
        shared_qsa_prefix_id="prefix-a",
    )
    staged = begin_shared_prefix_physical_promotion(
        canonical,
        rows=2,
        reserve_tail=3,
        stream=mx.new_stream(mx.cpu),
    )
    state = SegmentedSelfMTPState(
        lanes=[SimpleNamespace(uid=101), SimpleNamespace(uid=202)],
        row_caches=[pair, copy.deepcopy(pair)],
        transactions=[_Transaction(3), _Transaction(3)],
        membership_epoch=7,
        shared_qsa_prefix_id="prefix-b",
    )
    with pytest.raises(SegmentedPhysicalPromotionDeclined, match="attestation"):
        staged.bind(state)
    staged.cancel_and_drain()
    assert staged.physical is None


@pytest.fixture(autouse=True)
def _cpu_default():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def test_uniform_first_commit_promotes_and_reuses_recurrent_arrays():
    state = _state()
    stream = mx.new_stream(mx.cpu)
    ticket = begin_segmented_physical_promotion(
        state, reserve_tail=4, stream=stream
    )
    assert ticket.pending_bytes == ticket.reserved_bytes > 0
    ticket.settle_for_admission()
    ticket.settle_for_admission()
    assert ticket.pending_bytes == 0
    assert ticket.physical is not None and ticket.state is state
    transactions = list(state.transactions)
    row_key_arrays = [pair.target[0].keys for pair in state.row_caches]
    _commit(state, [2, 2])
    retained = state._segmented_caches.target[1].cache[0]

    batch, receipt = ticket.finish()

    qsa = batch.caches.target[0]
    assert all(qsa.keys is not row_keys for row_keys in row_key_arrays)
    assert qsa._idx == 5
    assert qsa.offset.tolist() == [5, 5]
    assert qsa.keys.shape[2] == 3 + qsa.step
    # Natural physical growth keeps a KV slab but concatenates an exact-width
    # raw-key ledger.  Promotion must publish the same steady-state geometry.
    assert qsa.index_keys.shape[1] == 5
    assert qsa.keys[0, 0, 3:5].tolist() == [[50, 50], [50, 50]]
    assert qsa.keys[1, 0, 3:5].tolist() == [[51, 51], [51, 51]]
    assert qsa.index_keys[0, 3:5].tolist() == [[70, 70], [70, 70]]
    assert qsa.index_keys[1, 3:5].tolist() == [[71, 71], [71, 71]]
    assert batch.caches.target[1].cache[0] is retained
    assert batch.caches.target[1].speculating
    assert all(transaction.closed for transaction in transactions)
    assert state.poisoned and not state.lanes and not state.row_caches
    assert receipt.advance == 2
    assert receipt.reserved_bytes > 0  # Total allocation remains in the receipt.
    assert receipt.recurrent_arrays_reused == 4
    assert receipt.patched_bytes > 0

    # The first physical follow-up must consume the normal growth slab instead
    # of copying the entire promoted prefix into a new allocation.
    keys_before = qsa.keys
    qsa.update_and_fetch(
        mx.full((2, 1, 1, 2), 81, dtype=mx.float32),
        mx.full((2, 1, 1, 3), 82, dtype=mx.float32),
    )
    assert qsa.keys is keys_before


def test_ragged_commit_declines_without_retiring_segmented_state():
    state = _state()
    original_rows = list(state.row_caches)
    transactions = list(state.transactions)
    ticket = begin_segmented_physical_promotion(
        state, reserve_tail=4, stream=mx.new_stream(mx.cpu)
    )
    _commit(state, [1, 2])

    with pytest.raises(SegmentedPhysicalPromotionDeclined, match="must be equal"):
        ticket.finish()

    assert state.row_caches == original_rows
    assert state.transactions == transactions
    assert not state.poisoned
    assert all(not transaction.closed for transaction in transactions)
    assert all(pair.target[1].speculating for pair in state.row_caches)


def test_uid_change_and_open_proposal_fail_before_publication():
    state = _state()
    ticket = begin_segmented_physical_promotion(
        state, reserve_tail=4, stream=mx.new_stream(mx.cpu)
    )
    _commit(state, [1, 1])
    state.lanes[0].uid = 999
    with pytest.raises(SegmentedPhysicalPromotionDeclined, match="UID"):
        ticket.finish()
    assert state.row_caches and not state.poisoned

    state = _state()
    ticket = begin_segmented_physical_promotion(
        state, reserve_tail=4, stream=mx.new_stream(mx.cpu)
    )
    _commit(state, [1, 1])
    state.proposal_open = True
    state._open_proposal = object()
    with pytest.raises(SegmentedPhysicalPromotionDeclined, match="not closed"):
        ticket.finish()
    assert state.row_caches and not state.poisoned


def test_stale_generation_fails_without_retiring_rows():
    state = _state()
    transactions = list(state.transactions)
    ticket = begin_segmented_physical_promotion(
        state, reserve_tail=4, stream=mx.new_stream(mx.cpu)
    )
    _commit(state, [1, 1])
    transactions[0].lineage.generation = 0

    with pytest.raises(SegmentedPhysicalPromotionDeclined, match="generation"):
        ticket.finish()

    assert state.row_caches and not state.poisoned
    assert all(not transaction.closed for transaction in transactions)


def test_destination_reservation_does_not_require_source_tail_capacity():
    state = _state()
    for pair in state.row_caches:
        row = pair.target[0]
        row.keys = mx.array(row.keys[..., : row.offset, :])
        row.values = mx.array(row.values[..., : row.offset, :])
    ticket = begin_segmented_physical_promotion(
        state, reserve_tail=2, stream=mx.new_stream(mx.cpu)
    )
    _commit(state, [1, 1])
    batch, receipt = ticket.finish()
    assert batch.caches.target[0]._idx == 4
    assert receipt.advance == 1


def test_recurrent_unwrap_preserves_plain_arrays_cache_type():
    state = _state()
    for pair in state.row_caches:
        plain = ArraysCache(2)
        plain.cache = list(pair.target[1].cache)
        plain.start_speculation()
        pair.target[1] = plain
    ticket = begin_segmented_physical_promotion(
        state, reserve_tail=2, stream=mx.new_stream(mx.cpu)
    )
    _commit(state, [1, 1])
    batch, _ = ticket.finish()
    assert type(batch.caches.target[1]) is ArraysCache


def test_recurrent_checkpoints_survive_promotion():
    state = _state()
    for index, pair in enumerate(state.row_caches):
        snapshot = [mx.full((1, 2), 90 + index, dtype=mx.float32), None]
        pair.target[1]._checkpoints = [[(3, snapshot)]]
    ticket = begin_segmented_physical_promotion(
        state, reserve_tail=2, stream=mx.new_stream(mx.cpu)
    )
    _commit(state, [1, 1])

    batch, _ = ticket.finish()

    checkpoints = batch.caches.target[1]._checkpoints
    assert len(checkpoints) == 2
    assert [lane[0][0] for lane in checkpoints] == [3, 3]
    assert checkpoints[0][0][1][0].tolist() == [[90, 90]]
    assert checkpoints[1][0][1][0].tolist() == [[91, 91]]


def test_qwen4_ple_history_fill_survives_physical_promotion():
    state = _state()
    for pair in state.row_caches:
        pair.target[1].ple_history_fill = 63
        pair.draft[1].ple_history_fill = 63
    ticket = begin_segmented_physical_promotion(
        state, reserve_tail=2, stream=mx.new_stream(mx.cpu)
    )
    _commit(state, [1, 1])

    batch, _ = ticket.finish()

    assert batch.caches.target[1].ple_history_fill == 63
    assert batch.caches.draft[1].ple_history_fill == 63


def test_cancel_and_drain_is_idempotent_and_prevents_publication():
    state = _state()
    ticket = begin_segmented_physical_promotion(
        state, reserve_tail=2, stream=mx.new_stream(mx.cpu)
    )
    ticket.cancel_and_drain()
    ticket.cancel_and_drain()

    assert state.row_caches and not state.poisoned
    with pytest.raises(RuntimeError, match="already finished"):
        ticket.finish()


def test_cleanup_failure_cannot_republish_partially_retired_rows(monkeypatch):
    state = _state()
    ticket = begin_segmented_physical_promotion(
        state, reserve_tail=2, stream=mx.new_stream(mx.cpu)
    )
    _commit(state, [1, 1])

    def fail_cleanup():
        raise RuntimeError("synthetic old-row cleanup failure")

    monkeypatch.setattr(
        state.row_caches[0].target[0], "stop_speculation", fail_cleanup
    )
    batch, receipt = ticket.finish()

    assert len(batch.lanes) == 2
    assert receipt.cleanup_error_count == 1
    assert state.poisoned
    assert not state.lanes and not state.row_caches and not state.transactions


def test_runtime_pooled_qsa_keys_survive_without_apc_persistence(monkeypatch):
    import mlx2.runtime.models.qwen4_exp as qwen4_exp

    monkeypatch.setattr(qwen4_exp, "_QSA_POOLED_KEY_CACHE", True)
    monkeypatch.setattr(qwen4_exp, "_QSA_APC_SUMMARIES", False)
    state = _state()
    ticket = begin_segmented_physical_promotion(
        state, reserve_tail=2, stream=mx.new_stream(mx.cpu)
    )
    _commit(state, [1, 1])
    for index, pair in enumerate(state.row_caches):
        for cache, length in ((pair.target[0], 4), (pair.draft[0], 2)):
            cache._qsa_pooled_keys = mx.full(
                (1, length, 2), 100 + index, dtype=mx.float32
            )
            cache._qsa_pooled_ratio = 1
            cache._qsa_summary_identity = None

    batch, _ = ticket.finish()

    assert batch.caches.target[0]._qsa_pooled_keys.shape == (2, 4, 2)
    assert batch.caches.draft[0]._qsa_pooled_keys.shape == (2, 2, 2)
    assert batch.caches.target[0]._qsa_summary_identity is None


def test_plain_physical_merge_carries_runtime_pooled_keys_without_apc(monkeypatch):
    import mlx2.runtime.models.qwen4_exp as qwen4_exp

    monkeypatch.setattr(qwen4_exp, "_QSA_POOLED_KEY_CACHE", True)
    monkeypatch.setattr(qwen4_exp, "_QSA_APC_SUMMARIES", False)
    rows = [_qsa(4, 1), _qsa(4, 2)]
    for index, cache in enumerate(rows):
        cache._qsa_pooled_keys = mx.full(
            (1, 2, 2), 120 + index, dtype=mx.float32
        )
        cache._qsa_pooled_ratio = 2
        cache._qsa_summary_identity = None

    batch = qwen4_exp.BatchQSAKVCache.merge(rows)

    assert batch._qsa_pooled_keys.shape == (2, 2, 2)
    assert batch._qsa_pooled_ratio == 2
    assert batch._qsa_summary_identity is None
