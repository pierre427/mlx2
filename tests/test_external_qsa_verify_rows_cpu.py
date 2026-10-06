"""CPU transaction evidence: real QSA attention and exact recurrent replay.

The recurrent arithmetic here is an explicit prefix-sum oracle, not a model
qualification. QSA projections, indexed keys and pooled summaries use the
runtime Attention implementation; each kept prefix is compared to a separate
ordinary forward from the same history.
"""

import mlx.core as mx
import numpy as np
import pytest
from qsa_oracle import tiny_args

from mlx2.runtime.cow_cache import (
    restore_recovery_descriptors,
    snapshot_recovery_descriptors,
)
from mlx2.runtime.hybrid_verify_rows import (
    HybridVerifyRows,
    HybridVerifyUnsupported,
    is_hybrid_rows,
)
from mlx2.runtime.models import qwen4_exp
from mlx2.runtime.models.cache import ArraysCache, KVCache
from mlx2.runtime.models.qwen4_exp import Attention, QSAKVCache


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    before = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(qwen4_exp, "_QSA_POOLED_KEY_CACHE", True)
    monkeypatch.setattr(qwen4_exp, "_QSA_APC_SUMMARIES", True)
    yield
    mx.set_default_device(before)


def _forward(attention, cache, hidden):
    result = attention(
        hidden,
        cache.make_mask(hidden.shape[1], window_size=None, return_array=True),
        cache,
    )
    mx.eval(result, cache.state)
    return result


def _pair(histories, recurrent):
    attention = Attention(tiny_args(indexer_budget=4))
    mx.eval(attention.parameters())
    prefixes = [
        mx.random.normal((1, length, 16), key=mx.random.key(20 + index))
        for index, length in enumerate(histories)
    ]
    rows, oracles = [], []
    for index, prefix in enumerate(prefixes):
        row = QSAKVCache(attention.indexer.summary_identity)
        oracle = QSAKVCache(attention.indexer.summary_identity)
        _forward(attention, row, prefix)
        _forward(attention, oracle, prefix)
        if recurrent:
            state = ArraysCache(1)
            state[0] = mx.array([[float(index + 10)]])
            rows.append([state, row])
        else:
            rows.append([row])
        oracles.append(oracle)
    return attention, rows, oracles


def _recurrent_forward(cache, hidden, lengths):
    """Stage the same exact prefix-sum replay used by the oracle below."""
    snapshot = list(cache.cache)

    def replay(count):
        contributions = [
            mx.sum(hidden[index : index + 1, : min(count, length), 0], axis=1)
            for index, length in enumerate(lengths)
        ]
        return [snapshot[0] + mx.concatenate(contributions)[:, None]]

    cache.record_rollback(hidden.shape[1], replay, snapshot)
    cache[0] = replay(hidden.shape[1])[0]
    mx.eval(cache[0])


def _verify(attention, transaction, hidden, recurrent):
    if recurrent:
        _recurrent_forward(transaction.caches[0], hidden, transaction.lengths)
    return _forward(attention, transaction.caches[-1], hidden)


def _assert_qsa(actual, expected):
    assert actual.offset == expected.offset
    assert actual.index_keys.shape[1] == actual.offset
    visible_actual = (
        actual.keys[:, :, : actual.offset],
        actual.values[:, :, : actual.offset],
        actual.index_keys,
    )
    visible_expected = (
        expected.keys[:, :, : expected.offset],
        expected.values[:, :, : expected.offset],
        expected.index_keys,
    )
    for left, right in zip(visible_actual, visible_expected):
        np.testing.assert_allclose(np.asarray(left), np.asarray(right), atol=2e-6)
    blocks = actual.offset // actual._qsa_pooled_ratio
    assert actual._qsa_pooled_keys.shape[1] == blocks
    assert actual._qsa_summary_identity["complete_blocks"] == blocks
    np.testing.assert_allclose(
        np.asarray(actual._qsa_pooled_keys),
        np.asarray(expected._qsa_pooled_keys),
        atol=2e-6,
    )
    assert not actual._mtp_share_topk and actual._mtp_shared_topk is None


@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.parametrize("kept", [1, 2, 3, 4])
def test_b1_every_kept_prefix_matches_separate_ordinary_qsa(recurrent, kept):
    attention, rows, oracles = _pair([8], recurrent)
    assert is_hybrid_rows(rows)
    initial = rows[0][0][0] if recurrent else None
    hidden = mx.random.normal((1, 4, 16), key=mx.random.key(52))
    transaction = HybridVerifyRows(rows).begin([4])
    result = _verify(attention, transaction, hidden, recurrent)
    expected = _forward(attention, oracles[0], hidden[:, :kept])
    committed = transaction.commit([kept])
    np.testing.assert_allclose(
        np.asarray(result[:, :kept]), np.asarray(expected), atol=3e-6
    )
    _assert_qsa(committed[0][-1], oracles[0])
    if recurrent:
        state = committed[0][0]
        np.testing.assert_allclose(
            np.asarray(state[0]),
            np.asarray(initial + mx.sum(hidden[:, :kept, 0], axis=1)[:, None]),
        )
        assert not state.speculating and not state._rollbacks
    with pytest.raises(RuntimeError, match="closed"):
        transaction.commit([kept])


@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.parametrize("kept", [[1, 1], [2, 2], [4, 3]])
def test_b2_ragged_histories_and_acceptance_preserve_all_qsa_planes(recurrent, kept):
    attention, rows, oracles = _pair([8, 13], recurrent)
    initial = [row[0][0] for row in rows] if recurrent else None
    hidden = mx.random.normal((2, 4, 16), key=mx.random.key(60))
    transaction = HybridVerifyRows(rows).begin([4, 3])
    result = _verify(attention, transaction, hidden, recurrent)
    expected = [
        _forward(attention, oracle, hidden[index : index + 1, :count])
        for index, (oracle, count) in enumerate(zip(oracles, kept))
    ]
    committed = transaction.commit(kept)
    for index, (row, oracle, count) in enumerate(zip(committed, oracles, kept)):
        np.testing.assert_allclose(
            np.asarray(result[index : index + 1, :count]),
            np.asarray(expected[index]),
            atol=3e-6,
        )
        _assert_qsa(row[-1], oracle)
        if recurrent:
            np.testing.assert_allclose(
                np.asarray(row[0][0]),
                np.asarray(
                    initial[index]
                    + mx.sum(hidden[index : index + 1, :count, 0], axis=1)[:, None]
                ),
            )
            assert not row[0].speculating and not row[0]._rollbacks


@pytest.mark.parametrize("histories", [[8], [8, 13]])
def test_aborted_qsa_forward_then_authoritative_recovery_restores_and_retries(
    histories,
):
    attention, rows, oracles = _pair(histories, True)
    snapshots = [snapshot_recovery_descriptors(row) for row in rows]
    before = [row[0][0] for row in rows]
    hidden = mx.random.normal((len(rows), 4, 16), key=mx.random.key(71))
    owner = HybridVerifyRows(rows)
    transaction = owner.begin([4] * len(rows))
    _verify(attention, transaction, hidden, True)
    transaction.abort()
    assert transaction.closed
    assert owner._active is None
    assert transaction.owner is None
    assert transaction.caches == []
    for row, oracle in zip(rows, oracles):
        _assert_qsa(row[-1], oracle)
        assert not row[0].speculating and not row[0]._rollbacks
    recovered = [
        restore_recovery_descriptors(snapshot, sidecar, borrowed)[0]
        for snapshot, sidecar, borrowed in snapshots
    ]
    for row, oracle, state in zip(recovered, oracles, before):
        _assert_qsa(row[-1], oracle)
        assert mx.array_equal(row[0][0], state).item()
    retry = HybridVerifyRows(recovered).begin([4] * len(rows))
    _verify(attention, retry, hidden, True)
    retry.commit([2] * len(rows))
    for index, (row, oracle) in enumerate(zip(recovered, oracles)):
        _forward(attention, oracle, hidden[index : index + 1, :2])
        _assert_qsa(row[-1], oracle)


def test_topology_refuses_unknown_or_inherited_cache_subclass_contracts():
    class UnknownKV(KVCache):
        pass

    class UnknownQSA(QSAKVCache):
        pass

    class Pretend:
        supports_external_verify_transaction = True

    for cache in (UnknownKV(), UnknownQSA(), Pretend()):
        assert not is_hybrid_rows([[cache]])
        with pytest.raises(HybridVerifyUnsupported, match="declared"):
            HybridVerifyRows([[cache]])
    assert is_hybrid_rows([[QSAKVCache()]])
    assert not is_hybrid_rows([[KVCache()]])
    with pytest.raises(HybridVerifyUnsupported, match="declared"):
        HybridVerifyRows([[QSAKVCache()], [KVCache()]])


def test_missing_qsa_verify_layer_fails_closed_before_prefix_commit():
    _, rows, _ = _pair([8], True)
    transaction = HybridVerifyRows(rows).begin([2])
    hidden = mx.ones((1, 2, 16))
    _recurrent_forward(transaction.caches[0], hidden, [2])
    with pytest.raises(RuntimeError, match="every target layer"):
        transaction.commit([1])
    assert transaction.closed
    assert rows[0][-1].offset == 8
    assert not rows[0][0].speculating


@pytest.mark.parametrize("kept", [[1, 1], [2, 2], [4, 3]])
def test_real_qwen4_gdn_ple_qsa_transaction_matches_ordinary_prefix(kept):
    from test_batched_mtp import _tiny_qwen4_model

    mx.random.seed(101)
    model = _tiny_qwen4_model()
    prompts = [[1, 7, 3, 9, 2, 8, 4, 6], [8, 5, 4, 3, 2, 9, 7, 6, 1, 3, 5, 7, 2]]
    rows, oracles = [], []
    for prompt in prompts:
        row, oracle = model.make_cache(), model.make_cache()
        model(mx.array([prompt]), cache=row)
        model(mx.array([prompt]), cache=oracle)
        mx.eval([cache.state for cache in row + oracle])
        rows.append(row)
        oracles.append(oracle)
    transaction = HybridVerifyRows(rows).begin([4, 3])
    tokens = mx.array([[3, 8, 2, 4], [4, 9, 2, 0]])
    result, taps = model.forward_with_taps(tokens, transaction.caches, [0, 1])
    mx.eval(result, taps)
    transaction.commit(kept)
    for index, count in enumerate(kept):
        expected = model(tokens[index : index + 1, :count], cache=oracles[index])
        mx.eval(expected, [cache.state for cache in oracles[index]])
        np.testing.assert_allclose(
            np.asarray(result[index : index + 1, :count]),
            np.asarray(expected),
            atol=3e-5,
        )
        for actual_cache, expected_cache in zip(rows[index], oracles[index]):
            if isinstance(actual_cache, ArraysCache):
                for left, right in zip(actual_cache.cache, expected_cache.cache):
                    if left is not None or right is not None:
                        np.testing.assert_allclose(
                            np.asarray(left), np.asarray(right), atol=3e-5
                        )
                assert not actual_cache.speculating and not actual_cache._rollbacks
            else:
                _assert_qsa(actual_cache, expected_cache)


def test_real_qwen4_aborted_verify_recovery_then_retry_has_no_rejected_state():
    from test_batched_mtp import _tiny_qwen4_model

    mx.random.seed(112)
    model = _tiny_qwen4_model()
    prompts = [[1, 7, 3, 9, 2, 8, 4, 6], [8, 5, 4, 3, 2, 9, 7, 6, 1, 3, 5, 7, 2]]
    rows, snapshots = [], []
    for prompt in prompts:
        row = model.make_cache()
        model(mx.array([prompt]), cache=row)
        mx.eval([cache.state for cache in row])
        rows.append(row)
        snapshots.append(snapshot_recovery_descriptors(row))
    transaction = HybridVerifyRows(rows).begin([4, 3])
    result, taps = model.forward_with_taps(
        mx.array([[3, 8, 2, 4], [4, 9, 2, 0]]), transaction.caches, [0, 1]
    )
    mx.eval(result, taps)
    # This is the failure seam used by the executor: end the live transaction,
    # then restore the authoritative pre-round request-private descriptors.
    transaction.abort()
    recovered = [
        restore_recovery_descriptors(snapshot, sidecar, borrowed)[0]
        for snapshot, sidecar, borrowed in snapshots
    ]
    retry_tokens = mx.array([[7, 1], [5, 3]])
    retry = HybridVerifyRows(recovered).begin([2, 2])
    retry_result, _ = model.forward_with_taps(retry_tokens, retry.caches, [0, 1])
    mx.eval(retry_result)
    retry.commit([1, 2])
    for index, count in enumerate([1, 2]):
        fresh = model.make_cache()
        model(mx.array([prompts[index]]), cache=fresh)
        expected = model(retry_tokens[index : index + 1, :count], cache=fresh)
        mx.eval(expected, [cache.state for cache in fresh])
        np.testing.assert_allclose(
            np.asarray(retry_result[index : index + 1, :count]),
            np.asarray(expected),
            atol=3e-5,
        )
        for actual_cache, expected_cache in zip(recovered[index], fresh):
            if isinstance(actual_cache, ArraysCache):
                for left, right in zip(actual_cache.cache, expected_cache.cache):
                    if left is not None or right is not None:
                        np.testing.assert_allclose(
                            np.asarray(left), np.asarray(right), atol=3e-5
                        )
                assert not actual_cache.speculating and not actual_cache._rollbacks
            else:
                _assert_qsa(actual_cache, expected_cache)
