"""Exact visibility bounds and fragmented absolute-position contracts."""

import pytest

mx = pytest.importorskip("mlx.core")
from mlx2.experimental.hysparse2.attention import (
    _candidate_tiles,
    _candidate_groups,
    _gather_groups,
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


def test_gather_groups_respect_byte_budget_gaps_and_values():
    k = mx.arange(20).astype(mx.float32).reshape(1, 1, 10, 2)
    blocks = [
        (k[:, :, :3], k[:, :, :3], 0),
        (k[:, :, 3:7], k[:, :, 3:7], 3),
        (k[:, :, 7:], k[:, :, 7:], 9),
    ]
    grouped = list(_gather_groups(blocks, max_bytes=64, max_tokens=5))
    assert [(a.shape[2], start) for a, _, start in grouped] == [(3, 0), (4, 3), (3, 9)]
    assert all(a.nbytes + b.nbytes <= 64 for a, b, _ in grouped)
    merged = list(_gather_groups(blocks, max_bytes=128, max_tokens=8))
    assert [(a.shape[2], start) for a, _, start in merged] == [(7, 0), (3, 9)]
    assert bool(mx.all(merged[0][0] == k[:, :, :7]).item())
@pytest.mark.parametrize("offset", [0, 1024])
def test_coarse_candidates_skip_future_history_with_reference_parity(monkeypatch, offset):
    from mlx2.experimental.hysparse2 import attention as module
    q = mx.random.normal((1, 2, 16, 8))
    k = mx.random.normal((1, 1, 4096, 8))
    v = mx.random.normal(k.shape)
    original = module._candidate_groups
    work = []
    def counted(blocks, block_size, key_tile, *, maximum=None):
        for item in original(blocks, block_size, key_tile, maximum=maximum):
            work.append(item[0].shape[2])
            yield item
    monkeypatch.setattr(module, "_candidate_groups", counted)
    def run(q, k, v):
        dense, selected = module.attention(q, [(k, v, 0)], offset=offset,
            query_tile=4, key_tile=128, select=(4, 8), block_select=(16, 2))
        return dense, selected, module.sparse_attention(q, selected, offset=offset, sinks=mx.zeros((2,)))
    current = run(q, k, v)
    mx.eval(current)
    bounded_work = sum(work)
    def objective(q, k, v):
        dense, _, sparse = run(q, k, v)
        return mx.mean(dense * dense) + mx.mean(sparse * sparse)
    current_grad = mx.grad(objective, argnums=(0, 1, 2))(q, k, v)
    mx.eval(current_grad)
    work.clear()
    def unbounded(blocks, block_size, key_tile, *, maximum=None):
        yield from counted(blocks, block_size, key_tile)
    monkeypatch.setattr(module, "_candidate_groups", unbounded)
    reference = run(q, k, v)
    mx.eval(reference)
    assert bounded_work < sum(work)
    reference_grad = mx.grad(objective, argnums=(0, 1, 2))(q, k, v)
    mx.eval(reference_grad)
    assert bool(mx.all(mx.sort(current[1][2], axis=-1) == mx.sort(reference[1][2], axis=-1)).item())
    for a, b in zip((current[0], current[2], *current_grad), (reference[0], reference[2], *reference_grad)):
        assert float(mx.max(mx.abs(a - b)).item()) < 1e-5


def test_coarse_candidate_groups_never_read_value_tensors():
    class UnreadableValue:
        def __getitem__(self, key):
            raise AssertionError("coarse selection read a value tensor")

    keys = mx.arange(6).astype(mx.float32).reshape(1, 1, 6, 1)
    blocks = [(keys[:, :, :3], UnreadableValue(), 5),
              (keys[:, :, 3:], UnreadableValue(), 8)]
    actual = list(_candidate_groups(blocks, 4, 8, maximum=9))
    reference = list(_candidate_groups([(k, mx.zeros_like(k), start)
                                       for k, _, start in blocks], 4, 8, maximum=9))
    assert len(actual) == len(reference)
    for a, b in zip(actual, reference):
        assert all(bool(mx.all(x == y).item()) for x, y in zip(a, b))
    valid = mx.concatenate([kp for _, kp, _ in actual])
    assert valid.tolist() == [5, 6, 7, 2147483647, 8, 9, 2147483647, 2147483647]
