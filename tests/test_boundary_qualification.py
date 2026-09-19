"""Differential qualification for QSA group closure and APC cancellation."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.runtime.apc_v2 import APCKey, APCv2
from mlx2.runtime.generate import PromptProcessingBatch
from mlx2.runtime.hybrid_speculative import (
    attach_self_mtp_lanes,
    propose_batched_self_mtp,
)
from mlx2.runtime.models.cache import (
    BatchKVCache,
    BatchQuantizedKVCache,
    KVCache,
    _copy_prompt_cache_for_restore,
)
from mlx2.runtime.models.qwen4_exp import (
    BatchQSAKVCache,
    QSAIndexer,
    QSASelection,
    _extend_mtp_shared_topk,
    _record_qsa_mtp_amendment,
    qsa_mtp_amendment_status,
)


def _effective_blocks(selection: QSASelection) -> tuple[int, ...]:
    if selection.kind == "implicit_all":
        return tuple(range(int(selection.n_blocks)))
    if selection.raw_block_ids is None:
        return ()
    return tuple(int(value) for value in selection.raw_block_ids[0, -1].tolist())


def test_qsa_reuse_appends_group_closed_mid_draft_without_acceptance_drift(
    monkeypatch,
):
    """The 3->4 block transition must extend, not silently stale, the reuse set."""
    import mlx2.runtime.models.qwen4_exp as qwen4
    from test_batched_mtp import _prepare_lane, _tiny_qwen4_model

    qsa_mtp_amendment_status(reset=True)
    mx.random.seed(2)
    model = _tiny_qwen4_model()
    arm = ["recompute"]
    selected = {"recompute": [], "reuse": []}
    closures = []
    original_call = QSAIndexer.__call__
    original_extend = qwen4._extend_mtp_shared_topk

    def record_call(indexer, *args, **kwargs):
        result = original_call(indexer, *args, **kwargs)
        if int(result.length) == 1 and int(result.n_blocks) >= 3:
            selected[arm[0]].append(
                (int(result.n_blocks), _effective_blocks(result))
            )
        return result

    def record_closure(cache, shared_topk, n_blocks):
        before_blocks = cache._mtp_shared_topk_n_blocks
        before = tuple(int(value) for value in shared_topk[0].tolist())
        result = original_extend(cache, shared_topk, n_blocks)
        if int(n_blocks) > int(before_blocks):
            closures.append(
                {
                    "before_blocks": int(before_blocks),
                    "after_blocks": int(n_blocks),
                    "before": before,
                    "after": tuple(int(value) for value in result[0].tolist()),
                }
            )
        return result

    monkeypatch.setattr(QSAIndexer, "__call__", record_call)
    monkeypatch.setattr(qwen4, "_extend_mtp_shared_topk", record_closure)

    outcomes = {}
    for share in (False, True):
        arm[0] = "reuse" if share else "recompute"
        lane = _prepare_lane(
            model,
            0,
            list(range(1, 8)),
            share_qsa_indices=share,
        )
        state = attach_self_mtp_lanes(model, None, [lane])
        proposal = propose_batched_self_mtp(model, state)
        outcomes[arm[0]] = (
            proposal.accepted_lengths,
            tuple(tuple(token.token for token in row) for row in proposal.outputs),
        )

    assert outcomes["reuse"] == outcomes["recompute"]
    assert outcomes["reuse"][0] == (1,)
    receipt = next(
        item
        for item in closures
        if item["before_blocks"] == 3 and item["after_blocks"] == 4
    )
    assert set(receipt["after"]) == set(receipt["before"]) | {3}
    assert any(blocks == 4 for blocks, _ids in selected["recompute"])
    assert any(
        blocks == 4 and 3 in ids for blocks, ids in selected["reuse"]
    )
    counters = qsa_mtp_amendment_status()
    assert counters["amendments"] >= 1
    assert counters["appended_blocks"] >= 1
    assert counters["max_appended_blocks"] >= 1


def test_qsa_mtp_amendment_status_is_bounded_and_resettable():
    status = qsa_mtp_amendment_status(reset=True)
    assert set(status) == {
        "calls",
        "noops",
        "amendments",
        "failures",
        "appended_blocks",
        "max_appended_blocks",
    }
    assert qsa_mtp_amendment_status() == {key: 0 for key in status}


def test_qsa_mtp_amendment_status_is_coherent_under_concurrent_updates():
    qsa_mtp_amendment_status(reset=True)

    def record_batch():
        for index in range(500):
            if index % 2:
                _record_qsa_mtp_amendment(noop=True)
            else:
                _record_qsa_mtp_amendment(appended_blocks=2)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(record_batch) for _ in range(8)]
        while any(not future.done() for future in futures):
            status = qsa_mtp_amendment_status()
            assert status["calls"] == (
                status["noops"] + status["amendments"] + status["failures"]
            )
            assert status["appended_blocks"] == status["amendments"] * 2
        for future in futures:
            future.result()

    assert qsa_mtp_amendment_status() == {
        "calls": 4_000,
        "noops": 2_000,
        "amendments": 2_000,
        "failures": 0,
        "appended_blocks": 4_000,
        "max_appended_blocks": 2,
    }


def test_qsa_mtp_amendment_failures_remain_visible_and_coherent():
    qsa_mtp_amendment_status(reset=True)
    shared_topk = SimpleNamespace()

    _record_qsa_mtp_amendment()
    with pytest.raises(RuntimeError, match="missing its captured block count"):
        _extend_mtp_shared_topk(SimpleNamespace(), shared_topk, 3)
    with pytest.raises(RuntimeError, match="outlived a block-grid rewind"):
        _extend_mtp_shared_topk(
            SimpleNamespace(_mtp_shared_topk_n_blocks=3), shared_topk, 2
        )

    assert qsa_mtp_amendment_status() == {
        "calls": 3,
        "noops": 0,
        "amendments": 0,
        "failures": 3,
        "appended_blocks": 0,
        "max_appended_blocks": 0,
    }


class _CacheEchoModel:
    """Small deterministic cache consumer for cancellation parity."""

    def __call__(self, tokens, *, cache):
        values = tokens.astype(mx.float32)[:, None, :, None]
        for entry in cache:
            entry.update_and_fetch(values, values + 0.5)
        return values


def _kv(tokens):
    cache = KVCache()
    values = mx.array(tokens, dtype=mx.float32)[None, None, :, None]
    cache.update_and_fetch(values, values + 0.5)
    mx.eval(cache.state)
    return cache


def _restored_copy(apc, key, tokens):
    hit = apc.lookup(key, tokens + [999])
    assert hit.hit and hit.cached_tokens == len(tokens)
    restored = _copy_prompt_cache_for_restore(hit.cache)
    hit.cache.close()
    return restored


def test_padded_ragged_apc_restore_expansion_cancel_preserves_survivor_and_bytes(
    tmp_path,
):
    key = APCKey("boundary-cancel")
    prefix = [1, 2]
    apc = APCv2(
        max_size=2,
        max_bytes=1 << 20,
        layout_name="boundary-cancel-v1",
        idle_disk_seconds=1,
        idle_disk_dir=str(tmp_path),
    )
    apc.store(key, prefix, [_kv(prefix)])
    entry = apc._trie.get(key, prefix)
    with apc._apc_lock:
        assert apc._spill_entry_locked(key, prefix, entry, reason="pressure")
    restored = _restored_copy(apc, key, prefix)
    reference = _restored_copy(apc, key, prefix)
    assert apc.nbytes == entry.nbytes
    assert apc.nbytes == sum(cache.nbytes for cache in entry.prompt_cache)

    model = _CacheEchoModel()
    batch = PromptProcessingBatch(
        model,
        [10, 11],
        [[_kv([7, 8, 9, 10, 11])], restored],
        tokens=[[7, 8, 9, 10, 11], prefix.copy()],
        prefill_step_size=2,
    )
    single = PromptProcessingBatch(
        model,
        [11],
        [reference],
        tokens=[prefix.copy()],
        prefill_step_size=2,
    )

    batch.prompt([[20, 21, 22], [30]])
    single.prompt([[30]])
    before_cancel_bytes = batch.prompt_cache[0].nbytes
    batch.filter([1])
    assert batch.uids == [11]
    assert batch.tokens == single.tokens

    batch.prompt([[31, 32]])
    single.prompt([[31, 32]])
    actual = batch.extract_cache(0)[0]
    expected = single.extract_cache(0)[0]
    mx.eval(actual.state, expected.state)

    assert batch.tokens == single.tokens == [prefix + [30, 31, 32]]
    assert mx.array_equal(actual.keys, expected.keys).item()
    assert mx.array_equal(actual.values, expected.values).item()
    assert actual.offset == expected.offset == 5
    assert batch.prompt_cache[0].nbytes == (
        batch.prompt_cache[0].keys.nbytes + batch.prompt_cache[0].values.nbytes
    )
    assert batch.prompt_cache[0].nbytes <= before_cancel_bytes
    assert apc.apc_stats["idle_disk"]["restores"] == 1


def test_cancellation_only_removes_padding_that_prefill_already_processed():
    factories = (
        lambda: BatchKVCache([9, 0]),
        lambda: BatchQSAKVCache([9, 0]),
        lambda: BatchQuantizedKVCache([9, 0], group_size=32, bits=4),
    )
    for make_cache in factories:
        cache = make_cache()
        values = mx.ones((2, 1, 3, 32), dtype=mx.float16)
        cache.update_and_fetch(values, values)
        if isinstance(cache, BatchQSAKVCache):
            cache.update_index_keys(values[:, 0])
        cache.filter([0])
        assert cache._idx == 0
        assert cache.offset.tolist() == [-6]
        assert cache.left_padding.tolist() == [6]
        if isinstance(cache, BatchQSAKVCache):
            assert cache.index_keys.shape[1] == 0
