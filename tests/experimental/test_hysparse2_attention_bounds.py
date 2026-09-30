"""Exact visibility bounds and fragmented absolute-position contracts."""

import pytest

mx = pytest.importorskip("mlx.core")
from mlx2.experimental.hysparse2.attention import (
    _candidate_tiles,
    _tiles,
    attention,
    sparse_attention,
)


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(42)
    yield
    mx.set_default_device(previous)


def test_fragmented_nonzero_positions_do_not_admit_future_candidates():
    q = mx.ones((1, 1, 1, 1))
    blocks = [
        (mx.array([[[[2.0], [3.0]]]]), mx.array([[[[10.0], [20.0]]]]), 4),
        (mx.array([[[[100.0], [100.0]]]]), mx.array([[[[100.0], [100.0]]]]), 10),
    ]
    candidates = list(_candidate_tiles(blocks, 4))
    assert candidates[1][2].tolist() == [10, 11]
    _, selected = attention(
        q,
        blocks,
        offset=9,
        query_tile=1,
        key_tile=8,
        select=(1, 2),
        block_select=(4, 1),
    )
    positions = selected[2].tolist()[0][0]
    assert {p for p in positions if p != 2147483647} == {4, 5}
    assert (
        float(sparse_attention(q, selected, offset=9, sinks=mx.array([-100.0])).item())
        > 10
    )


def test_gapped_block_positions_remain_absolute():
    k = mx.ones((1, 1, 2, 1))
    blocks = [(k, k, 5), (k[:, :, :1], k[:, :, :1], 7), (k[:, :, :1], k[:, :, :1], 10)]
    actual = [t[2].tolist() for t in _candidate_tiles(blocks, 8)]
    assert actual == [[5, 6, 7], [10]]
    with pytest.raises(ValueError, match="nonoverlapping"):
        list(_candidate_tiles([(k, k, 5), (k, k, 6)], 8))


@pytest.mark.parametrize("window", [None, 16])
def test_visible_attention_output_and_gradient_match_dense(window):
    q = mx.random.normal((2, 3, 48, 8))
    k = mx.random.normal((2, 1, 80, 8))
    v = mx.random.normal((2, 1, 80, 8))

    def actual(q, k, v):
        return attention(
            q, [(k, v, 0)], offset=16, query_tile=8, key_tile=16, window=window
        )[0]

    def dense(q, k, v):
        scores = (q @ k.swapaxes(-1, -2)) * 8**-0.5
        qp, kp = mx.arange(16, 64), mx.arange(80)
        mask = kp[None, :] <= qp[:, None]
        if window is not None:
            mask = mask & (kp[None, :] > qp[:, None] - window)
        scores = mx.where(mask[None, None], scores, -1e30)
        return mx.softmax(scores, axis=-1) @ v

    assert float(mx.max(mx.abs(actual(q, k, v) - dense(q, k, v))).item()) < 3e-5
    ga = mx.grad(lambda q: mx.sum(actual(q, k, v)))(q)
    gb = mx.grad(lambda q: mx.sum(dense(q, k, v)))(q)
    assert float(mx.max(mx.abs(ga - gb)).item()) < 3e-5
    # Host arithmetic: every query tile sees at most window+tile-1 keys.
    if window is not None:
        count = sum(
            t[0].shape[2]
            for t in _tiles([(k, v, 0)], 16, minimum=16 - window + 1, maximum=23)
        )
        assert count == 23


def test_gathered_fine_ranking_keeps_local_unique_and_padding_invalid():
    q = mx.ones((2, 2, 3, 1))
    k = mx.arange(32).astype(mx.float32).reshape(1, 1, 32, 1)
    k = mx.broadcast_to(k, (2, 1, 32, 1))
    _, support = attention(
        q,
        [(k, k, 0)],
        offset=29,
        query_tile=2,
        key_tile=8,
        select=(2, 6),
        block_select=(4, 1),
    )
    assert support[2].shape == (2, 3, 8)
    for row in support[2].tolist():
        for i, positions in enumerate(row):
            valid = [p for p in positions if p != 2147483647]
            assert len(valid) == len(set(valid))
            assert set(valid) == set(range(28, 30 + i))
