"""Offset/padding aliases across in-place cache updates (fn-b1-coldwarm-20261029).

A batch cache advances its ``offset`` mx.array in place in update_and_fetch.
A reference held across that update reads the new value in any op built
after it.  The QSA selection and the quantized cache conversions therefore
take copies.
"""
import mlx.core as mx

from mlx2.runtime.models.cache import BatchRotatingKVCache
from mlx2.runtime.models.qwen4_exp import (
    BatchQSAKVCache,
    BatchQSAQuantizedKVCache,
    QSAIndexer,
)
from qsa_oracle import tiny_args


def _kv(rows, length, dim=64, heads=2):
    return mx.random.normal((rows, heads, length, dim), key=mx.random.key(length))


def test_selection_offset_does_not_follow_the_cache_update():
    args = tiny_args()
    indexer = QSAIndexer(args)
    cache = BatchQSAKVCache([0])
    prompt = mx.random.normal((1, 12, args.hidden_size), key=mx.random.key(2))
    indexer(prompt, None, cache)
    cache.update_and_fetch(_kv(1, 12, args.head_dim), _kv(1, 12, args.head_dim))
    hidden = mx.random.normal((1, 1, args.hidden_size), key=mx.random.key(1))
    selection = indexer(hidden, None, cache)
    before = selection.offset.tolist() if isinstance(selection.offset, mx.array) else selection.offset
    assert selection.offset is not cache.offset
    cache.update_and_fetch(_kv(1, 1, args.head_dim), _kv(1, 1, args.head_dim))
    after = selection.offset.tolist() if isinstance(selection.offset, mx.array) else selection.offset
    assert after == before


def test_quantized_conversion_does_not_share_the_offset():
    source = BatchQSAKVCache([2])
    source.update_and_fetch(_kv(1, 70), _kv(1, 70))
    quantized = BatchQSAQuantizedKVCache.from_unquantized(source, group_size=64, bits=8)
    assert quantized.offset is not source.offset
    assert quantized.left_padding is not source.left_padding
    quantized.update_and_fetch(_kv(1, 1), _kv(1, 1))
    assert source.offset.tolist() == [68]
    assert quantized.offset.tolist() == [69]
    assert source._idx == 70


def test_rotating_to_quantized_does_not_share_the_offset():
    source = BatchRotatingKVCache(max_size=128, left_padding=[0])
    source.update_and_fetch(_kv(1, 64), _kv(1, 64))
    quantized = source.to_quantized(group_size=64, bits=8)
    assert quantized.offset is not source.offset
    quantized.offset += 1
    assert source.offset.tolist() == [64]
