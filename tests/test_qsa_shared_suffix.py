# Mined from unified 1e2bc604, MIT; see docs/PROVENANCE.md.
from unittest.mock import patch

import mlx.core as mx
import pytest

from mlx2.runtime.models.qwen4_exp import QSAIndexer, QSAKVCache
from mlx2.runtime.qsa_shared_suffix import (
    QSAImmutableBase,
    SharedSuffixQSAError,
    SharedSuffixQSAKVCache,
)
from mlx2.runtime.segmented_batch_cache import (
    SegmentedBatchQSAKVCache,
    build_segmented_batch_cache_group,
)


def _identity(complete_blocks):
    return {
        "format_version": 1,
        "model_config_hash": "model",
        "block_size": 4,
        "compress_ratio": 4,
        "producer_version": "test",
        "layer_id": "1",
        "complete_blocks": complete_blocks,
    }


def _source_cache(length=8):
    cache = QSAKVCache(_identity(0))
    cache.keys = mx.arange(length * 6, dtype=mx.float32).reshape(1, 2, length, 3)
    cache.values = cache.keys + 1000
    cache.index_keys = mx.arange(length * 5, dtype=mx.float32).reshape(1, length, 5)
    cache.offset = length
    cache._qsa_pooled_keys = mx.arange(
        (length // 4) * 5, dtype=mx.float32
    ).reshape(1, length // 4, 5)
    cache._qsa_pooled_ratio = 4
    cache._qsa_summary_identity = _identity(length // 4)
    return cache


def _attention_source_and_stock_rows(count):
    from qsa_oracle import tiny_args

    from mlx2.runtime.models.qwen4_exp import Attention

    args = tiny_args(indexer_budget=4)
    attention = Attention(args)
    mx.eval(attention.parameters())
    prefix = mx.random.normal((1, 8, args.hidden_size))
    source = QSAKVCache(attention.indexer.summary_identity)
    output = attention(
        prefix,
        source.make_mask(8, return_array=True, window_size=None),
        source,
    )
    mx.eval(output, source.state)
    ratio = attention.indexer.compress_ratio
    starts = mx.arange(source.offset // ratio) * ratio
    source._qsa_pooled_keys = attention.indexer._pool_blocks(
        source.index_keys, starts
    )
    source._qsa_pooled_ratio = ratio
    source._qsa_summary_identity = dict(attention.indexer.summary_identity)
    source._qsa_summary_identity["complete_blocks"] = source.offset // ratio
    mx.eval(source._qsa_pooled_keys)
    stocks = []
    for _ in range(count):
        stock = QSAKVCache(attention.indexer.summary_identity)
        stock.state = source.state
        stock.meta_state = source.meta_state
        stocks.append(stock)
    return args, attention, source, stocks


def _stock_segmented_output(attention, rows, hidden, lengths):
    segmented = SegmentedBatchQSAKVCache(rows, shared_qsa_prefix=False)
    width = max(lengths)
    segmented.prepare(
        lengths=lengths,
        right_padding=[width - length for length in lengths],
    )
    return segmented.segmented_attention(attention, hidden, None)


def test_rows_share_one_aligned_immutable_base_by_identity():
    source = _source_cache()
    base = QSAImmutableBase.from_cache(source, layout_id="qsa-f32-d3")
    left = SharedSuffixQSAKVCache(base)
    right = SharedSuffixQSAKVCache(base)

    assert left.base is right.base is base
    assert left.base.keys is base.keys
    assert left.base.values is base.values
    assert left.private_nbytes == right.private_nbytes == 0
    assert base.summary_identity["complete_blocks"] == 2
    with pytest.raises(ValueError, match="4-aligned"):
        QSAImmutableBase.from_cache(source, layout_id="qsa-f32-d3", length=7)


def test_pooled_base_records_coverage_without_optional_apc_identity():
    source = _source_cache()
    source._qsa_summary_identity = None

    base = QSAImmutableBase.from_cache(source, layout_id="qsa-f32-d3")

    assert base.pooled_keys.shape[1] == 2
    assert base.summary_identity == {"complete_blocks": 2}


def test_state_identity_uses_stable_suffix_storage_descriptors():
    base = QSAImmutableBase.from_cache(_source_cache(), layout_id="qsa-f32-d3")
    row = SharedSuffixQSAKVCache(base)
    row.append_index_keys(mx.ones((1, 1, 5)))
    row.append_kv(mx.ones((1, 2, 1, 3)), mx.ones((1, 2, 1, 3)))

    first = row.state
    second = row.state

    assert first[3] is second[3] is row._kv.keys
    assert first[4] is second[4] is row._kv.values
    assert first[5] == second[5] == 1


def test_segmented_builder_accepts_uniform_shared_suffix_rows():
    base = QSAImmutableBase.from_cache(_source_cache(), layout_id="qsa-f32-d3")
    rows = [SharedSuffixQSAKVCache(base), SharedSuffixQSAKVCache(base)]

    group = build_segmented_batch_cache_group([[rows[0]], [rows[1]]])

    assert len(group) == 1
    assert isinstance(group[0], SegmentedBatchQSAKVCache)
    assert group[0].rows == rows


def test_ragged_shared_suffix_fallback_matches_stock_rows(monkeypatch):
    monkeypatch.setenv("MLX_LM_QSA_PRIVATE_DELTA", "1")
    mx.random.seed(101)
    args, attention, source, stocks = _attention_source_and_stock_rows(2)
    base = QSAImmutableBase.from_cache(source, layout_id="qsa-ragged-fallback")
    rows = [SharedSuffixQSAKVCache(base) for _ in range(2)]
    lengths = [3, 1]
    hidden = mx.random.normal((2, 3, args.hidden_size))

    expected = _stock_segmented_output(attention, stocks, hidden, lengths)
    segmented = SegmentedBatchQSAKVCache(rows)
    segmented.prepare(lengths=lengths, right_padding=[0, 2])
    actual = segmented.segmented_attention(attention, hidden, None)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()
    assert [row.offset for row in rows] == [11, 9]
    assert [row.index_keys.shape[1] for row in rows] == [3, 1]


def test_preflight_declined_shared_suffix_fallback_matches_stock_rows(monkeypatch):
    monkeypatch.setenv("MLX_LM_QSA_PRIVATE_DELTA", "1")
    mx.random.seed(102)
    args, attention, source, stocks = _attention_source_and_stock_rows(2)
    base = QSAImmutableBase.from_cache(source, layout_id="qsa-declined-fallback")
    rows = [SharedSuffixQSAKVCache(base) for _ in range(2)]
    hidden = mx.random.normal((2, 3, args.hidden_size))

    expected = _stock_segmented_output(attention, stocks, hidden, [3, 3])
    segmented = SegmentedBatchQSAKVCache(rows)
    segmented.prepare(lengths=[3, 3], right_padding=[0, 0])
    with patch(
        "mlx2.runtime.segmented_batch_cache.qwen4_qsa_indexed_private_delta_preflight",
        return_value=(False, "synthetic_decline"),
    ):
        actual = segmented.segmented_attention(attention, hidden, None)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()
    assert [row.offset for row in rows] == [11, 11]


def test_one_surviving_shared_suffix_row_uses_single_row_selector(monkeypatch):
    monkeypatch.setenv("MLX_LM_QSA_PRIVATE_DELTA", "1")
    mx.random.seed(103)
    args, attention, source, stocks = _attention_source_and_stock_rows(1)
    base = QSAImmutableBase.from_cache(source, layout_id="qsa-b1-survivor")
    row = SharedSuffixQSAKVCache(base)
    hidden = mx.random.normal((1, 3, args.hidden_size))

    expected = _stock_segmented_output(attention, stocks, hidden, [3])
    segmented = SegmentedBatchQSAKVCache([row])
    segmented.prepare(lengths=[3], right_padding=[0])
    with (
        patch(
            "mlx2.runtime.segmented_batch_cache."
            "qwen4_qsa_indexed_private_delta_preflight",
            return_value=(True, "engaged"),
        ),
        patch.object(
            attention.indexer,
            "select_shared_suffix_batch",
            side_effect=AssertionError("B1 must not use the batched selector"),
        ),
        patch(
            "mlx2.runtime.segmented_batch_cache."
            "qwen4_qsa_indexed_private_delta_attention",
            side_effect=RuntimeError("synthetic CPU decline"),
        ),
    ):
        actual = segmented.segmented_attention(attention, hidden, None)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()
    assert row.offset == 11
    assert row.index_keys.shape[1] == 3


def test_normal_append_allocates_only_private_suffix_and_never_joins_prefix():
    base = QSAImmutableBase.from_cache(_source_cache(), layout_id="qsa-f32-d3")
    row = SharedSuffixQSAKVCache(base)
    raw = mx.full((1, 2, 5), 7.0)
    keys = mx.full((1, 2, 2, 3), 8.0)
    values = mx.full((1, 2, 2, 3), 9.0)

    returned_raw = row.append_index_keys(raw)
    suffix_k, suffix_v = row.append_kv(keys, values)
    mx.eval(returned_raw, suffix_k, suffix_v)

    assert returned_raw.shape == (1, 2, 5)
    assert suffix_k.shape == (1, 2, 2, 3)
    assert suffix_v.shape == (1, 2, 2, 3)
    assert row.offset == 10
    assert row.base is base
    # The private cache has ordinary physical growth slabs, but its storage
    # starts at suffix position zero rather than allocating base+suffix width.
    assert row._kv.keys.shape[2] == row._kv.step
    assert row._kv.keys.shape[2] < base.length + row._kv.step
    assert mx.array_equal(base.keys, _source_cache().keys).item()


def test_trim_is_exact_inside_suffix_and_refuses_to_cross_base():
    base = QSAImmutableBase.from_cache(_source_cache(), layout_id="qsa-f32-d3")
    row = SharedSuffixQSAKVCache(base)
    row.append_index_keys(mx.ones((1, 3, 5)))
    row.append_kv(mx.ones((1, 2, 3, 3)), mx.ones((1, 2, 3, 3)))

    assert row.trim(2) == 2
    assert row.offset == 9
    assert row.index_keys.shape[1] == 1
    with pytest.raises(SharedSuffixQSAError, match="cross the shared base"):
        row.trim(2)
    assert row.offset == 9


def test_materialization_is_explicit_receipted_and_preserves_qsa_summary():
    events = {}

    def note(key, amount):
        events[key] = events.get(key, 0) + amount

    source = _source_cache()
    base = QSAImmutableBase.from_cache(source, layout_id="qsa-f32-d3")
    row = SharedSuffixQSAKVCache(base, note=note)
    row.append_index_keys(mx.full((1, 4, 5), 11.0))
    row.append_kv(mx.full((1, 2, 4, 3), 12.0), mx.full((1, 2, 4, 3), 13.0))
    row.set_suffix_pooled_keys(mx.full((1, 1, 5), 14.0))

    materialized, receipt = row.materialize_to_qsa()
    mx.eval(materialized.state)

    assert type(materialized) is QSAKVCache
    assert receipt.explicit is True
    assert receipt.base_length == 8
    assert receipt.suffix_length == 4
    assert materialized.offset == 12
    assert materialized.keys.shape[2] == 12
    assert materialized.index_keys.shape[1] == 12
    assert materialized._qsa_pooled_keys.shape == (1, 3, 5)
    assert materialized._qsa_pooled_ratio == 4
    assert materialized._qsa_summary_identity["complete_blocks"] == 3
    assert events == {
        "shared_qsa_materializations": 1,
        "shared_qsa_materialized_bytes": receipt.materialized_bytes,
    }
    assert row.base is base


def test_raw_index_and_kv_append_must_advance_together():
    base = QSAImmutableBase.from_cache(_source_cache(), layout_id="qsa-f32-d3")
    row = SharedSuffixQSAKVCache(base)
    row.append_index_keys(mx.ones((1, 2, 5)))
    with pytest.raises(SharedSuffixQSAError, match="append disagree"):
        row.append_kv(mx.ones((1, 2, 1, 3)), mx.ones((1, 2, 1, 3)))


def test_explicit_unledgered_draft_append_must_be_rewound_before_materialize():
    base = QSAImmutableBase.from_cache(_source_cache(), layout_id="qsa-f32-d3")
    row = SharedSuffixQSAKVCache(base)
    row.append_kv(
        mx.ones((1, 2, 1, 3)),
        mx.ones((1, 2, 1, 3)),
        allow_unledgered=True,
    )
    with pytest.raises(SharedSuffixQSAError, match="ledger and K/V width disagree"):
        row.materialize_to_qsa()
    row.trim(1)
    materialized, _ = row.materialize_to_qsa()
    assert materialized.offset == base.length
    assert materialized.index_keys.shape[1] == base.length


def test_split_aware_indexer_matches_stock_selection_without_joining_raw_prefix():
    from qsa_oracle import tiny_args

    mx.random.seed(7)
    args = tiny_args(indexer_budget=8)
    indexer = QSAIndexer(args)
    prefix_hidden = mx.random.normal((1, 8, args.hidden_size))
    source = QSAKVCache(indexer.summary_identity)
    indexer(
        prefix_hidden,
        source.make_mask(8, return_array=True, window_size=None),
        source,
    )
    source.keys = mx.zeros((1, 1, 8, args.head_dim))
    source.values = mx.zeros((1, 1, 8, args.head_dim))
    source.offset = 8
    starts = mx.arange(2) * indexer.compress_ratio
    source._qsa_pooled_keys = indexer._pool_blocks(source.index_keys, starts)
    source._qsa_pooled_ratio = indexer.compress_ratio
    source._qsa_summary_identity = dict(indexer.summary_identity)
    source._qsa_summary_identity["complete_blocks"] = 2

    stock = QSAKVCache(indexer.summary_identity)
    stock.keys = source.keys
    stock.values = source.values
    stock.index_keys = source.index_keys
    stock.offset = source.offset
    stock._qsa_pooled_keys = source._qsa_pooled_keys
    stock._qsa_pooled_ratio = source._qsa_pooled_ratio
    stock._qsa_summary_identity = source._qsa_summary_identity
    base = QSAImmutableBase.from_cache(source, layout_id="qsa-selector-test")
    split = SharedSuffixQSAKVCache(base)

    hidden = mx.random.normal((1, 8, args.hidden_size))
    mask = stock.make_mask(8, return_array=True, window_size=None)
    projected = indexer.index_qk_proj(hidden)
    stock_selection = indexer(
        hidden, mask, stock, projected_qk=projected
    )
    split_selection = indexer.select_shared_suffix(
        hidden, mask, split, projected_qk=projected
    )
    mx.eval(
        stock_selection.raw_block_ids,
        split_selection.raw_block_ids,
        stock_selection.dense_mask(),
        split_selection.dense_mask(),
    )

    assert mx.array_equal(
        stock_selection.raw_block_ids, split_selection.raw_block_ids
    ).item()
    assert mx.array_equal(
        stock_selection.dense_mask(), split_selection.dense_mask()
    ).item()
    assert split.index_keys.shape[1] == 8
    assert split.base.index_keys.shape[1] == 8


def test_batched_shared_base_scoring_matches_two_row_selection():
    from qsa_oracle import tiny_args

    mx.random.seed(11)
    args = tiny_args(indexer_budget=8)
    indexer = QSAIndexer(args)
    source = _source_cache()
    source.index_keys = mx.random.normal((1, 8, args.indexer_head_dim))
    starts = mx.arange(2) * indexer.compress_ratio
    source._qsa_pooled_keys = indexer._pool_blocks(source.index_keys, starts)
    source._qsa_pooled_ratio = indexer.compress_ratio
    source._qsa_summary_identity = dict(indexer.summary_identity)
    source._qsa_summary_identity["complete_blocks"] = 2
    source.keys = mx.zeros((1, 2, 8, 3))
    source.values = mx.zeros((1, 2, 8, 3))
    base = QSAImmutableBase.from_cache(source, layout_id="qsa-batch-test")
    batched_rows = [SharedSuffixQSAKVCache(base) for _ in range(2)]
    serial_rows = [SharedSuffixQSAKVCache(base) for _ in range(2)]
    hidden = mx.random.normal((2, 8, args.hidden_size))
    projected = indexer.index_qk_proj(hidden)
    masks = [
        row.make_mask(8, return_array=True, window_size=None)
        for row in batched_rows
    ]

    batched = indexer.select_shared_suffix_batch(
        hidden, masks, batched_rows, projected_qk=projected
    )
    serial = [
        indexer.select_shared_suffix(
            hidden[index : index + 1],
            masks[index],
            row,
            projected_qk=projected[index : index + 1],
        )
        for index, row in enumerate(serial_rows)
    ]
    mx.eval(
        *[selection.raw_block_ids for selection in batched],
        *[selection.raw_block_ids for selection in serial],
    )

    for actual, expected in zip(batched, serial):
        assert mx.array_equal(
            actual.raw_block_ids, expected.raw_block_ids
        ).item()
        assert mx.array_equal(actual.dense_mask(), expected.dense_mask()).item()


@pytest.mark.parametrize("lengths", ([3, 1], [3, 3]))
def test_shared_suffix_fallback_uses_stock_indexed_dispatch(lengths, monkeypatch):
    monkeypatch.setenv("MLX_LM_QSA_PRIVATE_DELTA", "1")
    mx.random.seed(104)
    args, attention, source, stocks = _attention_source_and_stock_rows(2)
    base = QSAImmutableBase.from_cache(source, layout_id="stock-dispatch-fallback")
    rows = [SharedSuffixQSAKVCache(base) for _ in range(2)]
    hidden = mx.random.normal((2, 3, args.hidden_size))

    dispatched = []

    def indexed(q, k, v, *_args, **_kwargs):
        dispatched.append((q, k, v))
        return q

    with (
        patch("mlx2.runtime.models.qwen4_exp.decide_qsa_indexed_admission",
              return_value=(True, "test_dispatch")),
        patch("mlx2.runtime.models.qwen4_exp._dispatch_qsa_indexed_with_optional_capture",
              side_effect=indexed) as dispatch,
        patch("mlx2.runtime.segmented_batch_cache.qwen4_qsa_indexed_private_delta_preflight",
              return_value=(False, "test_preflight_decline")),
    ):
        expected = _stock_segmented_output(attention, stocks, hidden, lengths)
        assert dispatch.call_count == 2
        dispatch.reset_mock()
        segmented = SegmentedBatchQSAKVCache(rows)
        segmented.prepare(lengths=lengths, right_padding=[3 - n for n in lengths])
        actual = segmented.segmented_attention(attention, hidden, None)
        mx.eval(expected, actual)
        assert dispatch.call_count == 2
    for stock, shared in zip(dispatched[:2], dispatched[2:]):
        assert all(mx.array_equal(a, b).item() for a, b in zip(stock, shared))
    assert mx.array_equal(actual, expected).item()
    assert [row.offset for row in rows] == [8 + n for n in lengths]
    assert [row.index_keys.shape[1] for row in rows] == list(lengths)
