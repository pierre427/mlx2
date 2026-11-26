"""Multi-lane HybridVerifyRows over sliding-window rings (CPU).

``SegmentedBatchRotatingKVCache`` keeps each lane's own speculating B1 ring
as authoritative storage, so a batched round must leave every lane's window
exactly as a B1 round of the same tokens would: committed prefixes, wrapped
rings, aborts, plain rounds with a released row, and a step that fails
midway.  Attention through the view must equal each lane's B1 attention.
The batched path is opt-in (``batched_rings``) and fails closed without it.
"""

import mlx.core as mx
import pytest

from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows, HybridVerifyUnsupported
from mlx2.runtime.models.cache import KVCache, RotatingKVCache
from mlx2.runtime.segmented_batch_ring import SegmentedBatchRotatingKVCache

WINDOW, HEADS, DIM = 8, 2, 4


@pytest.fixture(autouse=True)
def _cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def _rows(key):
    return mx.random.normal((1, HEADS, 1, DIM), key=mx.random.key(key))


def _token(lane, pos, layer):
    """Deterministic K/V/Q rows for one lane position (no repeated values)."""
    base = 7919 * lane + 104729 * layer + pos
    return _rows(3 * base), _rows(3 * base + 1), _rows(3 * base + 2)


def _block(lane, start, count, layer):
    parts = [_token(lane, p, layer) for p in range(start, start + count)]
    return tuple(mx.concatenate([part[i] for part in parts], axis=2) for i in range(3))


def _lane(lane, prompt):
    """[ring, kv] for one lane after ``prompt`` ordinary tokens."""
    row = [RotatingKVCache(WINDOW), KVCache()]
    for layer, cache in enumerate(row):
        k, v, _ = _block(lane, 0, prompt, layer)
        cache.update_and_fetch(k, v)
    return row


def _window(cache, logical):
    if isinstance(cache, RotatingKVCache):
        keys, values = cache._temporal_order(cache.keys), cache._temporal_order(cache.values)
        if logical:  # what attention can still see: the newest max_size rows
            keep = min(cache.offset, cache.max_size)
            keys, values = keys[..., -keep:, :], values[..., -keep:, :]
        return keys, values
    return cache.keys_and_values()


def _assert_same_lane(row, ref, *, logical=True):
    """Equal offsets and windows; ``logical=False`` also demands the same layout."""
    for cache, want in zip(row, ref):
        assert cache.offset == want.offset
        assert not getattr(cache, "speculating", False)
        if not logical:
            assert getattr(cache, "_idx", None) == getattr(want, "_idx", None)
        for got, exp in zip(_window(cache, logical), _window(want, logical)):
            assert got.shape == exp.shape
            assert mx.array_equal(got, exp).item()


def _b1_round(lane, prompt, count, accepted):
    """The same round on the B1 hybrid path (the layout the batched path must match)."""
    row = _lane(lane, prompt)
    txn = HybridVerifyRows([row]).begin([count])
    for layer, cache in enumerate(txn.caches):
        k, v, _ = _block(lane, prompt, count, layer)
        cache.update_and_fetch(k, v)
    txn.commit([accepted])
    return row


def _verify_forward(caches, starts, lengths, first_layer=0):
    """One batched 'attention' step over the owner's views; returns per-layer outputs."""
    width = max(lengths)
    outs = []
    for layer, view in enumerate(caches, start=first_layer):
        ks, vs, qs = [], [], []
        for lane, (start, count) in enumerate(zip(starts, lengths)):
            k, v, q = _block(lane, start, count, layer)
            pad = [(0, 0), (0, 0), (0, width - count), (0, 0)]
            ks.append(mx.pad(k, pad)), vs.append(mx.pad(v, pad)), qs.append(mx.pad(q, pad))
        k, v, q = (mx.concatenate(a, axis=0) for a in (ks, vs, qs))
        mask = view.make_mask(width, window_size=WINDOW if layer == 0 else None)
        view.update_and_fetch(k, v)
        outs.append(view.bucketed_attention(q, 1.0, mask))
    return outs


def _b1_attention(lane, prompt, count, layer):
    """The lane's B1 verify attention over the same rows."""
    cache = _lane(lane, prompt)[layer]
    k, v, q = _block(lane, prompt, count, layer)
    mask = cache.make_mask(count, window_size=WINDOW if layer == 0 else None, return_array=True)
    keys, values = cache.update_and_fetch(k, v)
    return mx.fast.scaled_dot_product_attention(q, keys, values, scale=1.0, mask=mask)


def _reference(lane, prompt, accepted):
    row = _lane(lane, prompt)
    for layer, cache in enumerate(row):
        for p in range(prompt, prompt + accepted):
            k, v, _ = _token(lane, p, layer)
            cache.update_and_fetch(k, v)
    return row


@pytest.mark.parametrize("prompts,lengths,accepted", [
    ((5, 3), (4, 4), (2, 4)),       # neither ring wrapped
    ((30, 6), (4, 3), (1, 3)),      # lane 0 wrapped, lane 1 fills its window
    ((30, 41), (5, 2), (3, 1)),     # both wrapped, ragged widths
])
def test_batched_commit_equals_b1_per_lane(prompts, lengths, accepted):
    rows = [_lane(i, p) for i, p in enumerate(prompts)]
    owner = HybridVerifyRows(rows, batched_rings=True)
    txn = owner.begin(list(lengths))
    assert txn.batched
    assert isinstance(txn.caches[0], SegmentedBatchRotatingKVCache)
    outs = _verify_forward(txn.caches, prompts, lengths)
    for layer, out in enumerate(outs):
        for lane, (prompt, count) in enumerate(zip(prompts, lengths)):
            want = _b1_attention(lane, prompt, count, layer)
            assert mx.array_equal(out[lane : lane + 1, :, :count], want).item()
            assert not mx.any(out[lane, :, count:]).item()
    committed = txn.commit(list(accepted))
    for lane, (prompt, count, a) in enumerate(zip(prompts, lengths, accepted)):
        _assert_same_lane(committed[lane], _b1_round(lane, prompt, count, a), logical=False)
        _assert_same_lane(committed[lane], _reference(lane, prompt, a))


def test_rounds_back_to_back_keep_lanes_exact():
    prompts, owner_rows = [30, 9], [_lane(0, 30), _lane(1, 9)]
    owner = HybridVerifyRows(owner_rows, batched_rings=True)
    pos = list(prompts)
    for lengths, accepted in [((4, 4), (2, 1)), ((3, 5), (3, 5)), ((5, 2), (1, 2))]:
        txn = owner.begin(list(lengths))
        _verify_forward(txn.caches, pos, lengths)
        txn.commit(list(accepted))
        pos = [p + a for p, a in zip(pos, accepted)]
    for lane in range(2):
        _assert_same_lane(owner_rows[lane], _reference(lane, prompts[lane], pos[lane] - prompts[lane]))


@pytest.mark.parametrize("plain", [False, True])
def test_batched_abort_restores_wrapped_rings(plain):
    prompts = (30, 12)
    rows = [_lane(i, p) for i, p in enumerate(prompts)]
    owner = HybridVerifyRows(rows, batched_rings=True)
    lengths = (1, 1) if plain else (4, 2)
    txn = owner.begin(list(lengths), plain_single_token=plain)
    _verify_forward(txn.caches, prompts, lengths)
    txn.abort()
    for lane, prompt in enumerate(prompts):
        _assert_same_lane(rows[lane], _lane(lane, prompt), logical=False)


def test_plain_round_release_row_rewinds_only_that_lane():
    prompts = (30, 12)
    rows = [_lane(i, p) for i, p in enumerate(prompts)]
    owner = HybridVerifyRows(rows, batched_rings=True)
    txn = owner.begin([1, 1], plain_single_token=True)
    _verify_forward(txn.caches, prompts, (1, 1))
    txn.release_row(0)
    txn.commit([1, 1])
    _assert_same_lane(rows[0], _lane(0, 30), logical=False)
    _assert_same_lane(rows[1], _reference(1, 12, 1))


def test_step_failing_midway_rewinds_every_lane_then_aborts():
    prompts = (30, 12)
    rows = [_lane(i, p) for i, p in enumerate(prompts)]
    owner = HybridVerifyRows(rows, batched_rings=True)
    txn = owner.begin([3, 3])
    ring_view = txn.caches[0]
    original = RotatingKVCache.update_and_fetch

    def boom(self, k, v, **kw):
        if self is rows[1][0]:
            raise RuntimeError("boom")
        return original(self, k, v, **kw)

    RotatingKVCache.update_and_fetch = boom
    try:
        with pytest.raises(RuntimeError, match="boom"):
            _verify_forward(txn.caches, prompts, (3, 3))
    finally:
        RotatingKVCache.update_and_fetch = original
    assert [r.offset for r in ring_view.rows] == list(prompts)  # lane 0 rewound too
    txn.abort()
    for lane, prompt in enumerate(prompts):
        _assert_same_lane(rows[lane], _lane(lane, prompt), logical=False)


def test_row_attention_runs_each_lane_b1_code():
    prompts, lengths = (30, 6), (4, 2)
    rows = [_lane(i, p) for i, p in enumerate(prompts)]
    owner = HybridVerifyRows(rows, batched_rings=True)
    txn = owner.begin(list(lengths))
    view = txn.caches[0]

    def lane_attention(index, ring, valid):
        k, v, q = _block(index, prompts[index], valid, 0)
        mask = ring.make_mask(valid, window_size=WINDOW, return_array=True)
        keys, values = ring.update_and_fetch(k, v)
        return mx.fast.scaled_dot_product_attention(q, keys, values, scale=1.0, mask=mask)

    out = view.row_attention(lane_attention)
    for lane, (prompt, count) in enumerate(zip(prompts, lengths)):
        assert mx.array_equal(out[lane : lane + 1, :, :count], _b1_attention(lane, prompt, count, 0)).item()
    # The plain KV layer still goes through its own view.
    _verify_forward(txn.caches[1:], prompts, lengths, first_layer=1)
    txn.commit([2, 1])
    for lane, (prompt, a) in enumerate(zip(prompts, (2, 1))):
        _assert_same_lane(rows[lane], _reference(lane, prompt, a))


def test_row_attention_that_skips_its_append_fails_and_rewinds():
    rows = [_lane(0, 30), _lane(1, 6)]
    owner = HybridVerifyRows(rows, batched_rings=True)
    txn = owner.begin([2, 2])

    def lazy(index, ring, valid):
        if index == 1:
            return mx.zeros((1, HEADS, valid, DIM))
        k, v, _ = _block(index, 30, valid, 0)
        ring.update_and_fetch(k, v)
        return mx.zeros((1, HEADS, valid, DIM))

    with pytest.raises(RuntimeError, match="did not append"):
        txn.caches[0].row_attention(lazy)
    assert [r[0].offset for r in rows] == [30, 6]
    txn.abort()


def test_deferred_commit_rings_match_rollback_path():
    prompts, lengths, accepted = (30, 12), (4, 3), (2, 3)
    rows = [_lane(i, p) for i, p in enumerate(prompts)]
    owner = HybridVerifyRows(rows, deferred_commit=True, batched_rings=True)
    txn = owner.begin(list(lengths))
    width = max(lengths)
    view = txn.caches[0]
    ks, vs = [], []
    for lane, (start, count) in enumerate(zip(prompts, lengths)):
        k, v, _ = _block(lane, start, count, 0)
        pad = [(0, 0), (0, 0), (0, width - count), (0, 0)]
        ks.append(mx.pad(k, pad)), vs.append(mx.pad(v, pad))
    view.update_and_fetch(mx.concatenate(ks, 0), mx.concatenate(vs, 0), deferred=True)
    assert [r[0].deferred_rows for r in rows] == list(lengths)
    _verify_forward(txn.caches[1:], prompts, lengths, first_layer=1)
    txn.commit(list(accepted))
    for lane, (prompt, a) in enumerate(zip(prompts, accepted)):
        _assert_same_lane(rows[lane], _reference(lane, prompt, a))


def test_batched_rings_are_opt_in_and_exact_type_only():
    rows = [_lane(0, 5), _lane(1, 5)]
    with pytest.raises(HybridVerifyUnsupported, match="supports_batched_ring_verify"):
        HybridVerifyRows(rows)

    class Ring(RotatingKVCache):
        pass

    odd = [[Ring(WINDOW), KVCache()], [Ring(WINDOW), KVCache()]]
    for row in odd:
        for cache in row:
            cache.update_and_fetch(*_block(0, 0, 3, 0)[:2])
    # A subclass is refused (here before the ring check: it declares no contract).
    with pytest.raises(HybridVerifyUnsupported, match="exact"):
        HybridVerifyRows(odd, batched_rings=True)


def test_view_refuses_rings_outside_a_speculation_epoch():
    view = SegmentedBatchRotatingKVCache([_lane(0, 5)[0], _lane(1, 5)[0]])
    with pytest.raises(RuntimeError, match="speculating"):
        view.prepare(lengths=[2, 2])


def test_external_owner_passes_the_target_declaration(monkeypatch):
    from mlx2.runtime import external_speculative as es

    seen = {}

    class Owner:
        def __init__(self, rows, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr("mlx2.runtime.hybrid_verify_rows.HybridVerifyRows", Owner)
    monkeypatch.setattr("mlx2.runtime.hybrid_verify_rows.is_hybrid_rows", lambda rows, **kw: True)
    gen = object.__new__(es.ExternalDraftBatchGenerator)
    gen.scheduler_stats = {}
    for declared in (False, True):
        gen.model = type("M", (), {"supports_speculative_rollback": True,
                                    "supports_batched_ring_verify": declared})()
        gen._target_owner([[object()]])
        assert seen["batched_rings"] is declared


def _ring_step(view, prompts, lengths, layer, *, window, deferred=False, mask=None):
    width = max(lengths)
    ks, vs, qs = [], [], []
    for lane, (start, count) in enumerate(zip(prompts, lengths)):
        k, v, q = _block(lane, start, count, layer)
        pad = [(0, 0), (0, 0), (0, width - count), (0, 0)]
        ks.append(mx.pad(k, pad)), vs.append(mx.pad(v, pad)), qs.append(mx.pad(q, pad))
    k, v, q = (mx.concatenate(a, axis=0) for a in (ks, vs, qs))
    mask = view.make_mask(width, window_size=window) if mask is None else mask
    view.update_and_fetch(k, v, deferred=deferred)
    return view.bucketed_attention(q, 1.0, mask), mask


def _b1_ring_attention(lane, prompt, count, window):
    cache = _lane(lane, prompt)[0]
    k, v, q = _block(lane, prompt, count, 0)
    mask = cache.make_mask(count, window_size=window, return_array=True)
    keys, values = cache.update_and_fetch(k, v)
    return mx.fast.scaled_dot_product_attention(q, keys, values, scale=1.0, mask=mask)


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("prompts,lengths", [((30, 41), (4, 1)), ((30, 41), (1, 5)), ((6, 30), (1, 3))])
def test_one_token_lane_beside_wider_lane_with_window_below_ring(deferred, prompts, lengths):
    """Window 5 < ring 8: a one-token lane's mask must fit how its K/V were fetched."""
    window = 5
    rows = [_lane(i, p) for i, p in enumerate(prompts)]
    owner = HybridVerifyRows(rows, deferred_commit=deferred, batched_rings=True)
    txn = owner.begin(list(lengths))
    out, _ = _ring_step(txn.caches[0], prompts, lengths, 0, window=window, deferred=deferred)
    for lane, (prompt, count) in enumerate(zip(prompts, lengths)):
        want = _b1_ring_attention(lane, prompt, count, window)
        got = out[lane : lane + 1, :, :count]
        if deferred:  # chronological vs physical key order: same values, other sum order
            assert mx.allclose(got, want, atol=1e-5).item()
        else:
            assert mx.array_equal(got, want).item()
    _verify_forward(txn.caches[1:], prompts, lengths, first_layer=1)
    accepted = [max(1, n - 1) for n in lengths]
    txn.commit(accepted)
    for lane, (prompt, a) in enumerate(zip(prompts, accepted)):
        _assert_same_lane(rows[lane], _reference(lane, prompt, a))


def test_one_mask_shared_by_every_ring_layer():
    prompts, lengths = (30, 12), (4, 2)
    rows = [[RotatingKVCache(WINDOW), RotatingKVCache(WINDOW)] for _ in prompts]
    for lane, (row, prompt) in enumerate(zip(rows, prompts)):
        for layer, cache in enumerate(row):
            cache.update_and_fetch(*_block(lane, 0, prompt, layer)[:2])
    owner = HybridVerifyRows(rows, batched_rings=True)
    txn = owner.begin(list(lengths))
    _, mask = _ring_step(txn.caches[0], prompts, lengths, 0, window=WINDOW)
    out, _ = _ring_step(txn.caches[1], prompts, lengths, 1, window=WINDOW, mask=mask)
    assert out.shape[0] == 2
    other = SegmentedBatchRotatingKVCache([RotatingKVCache(WINDOW), RotatingKVCache(WINDOW)])
    for ring in other.rows:
        ring.start_speculation()
    other.prepare(lengths=[1, 1])
    other.update_and_fetch(mx.zeros((2, HEADS, 1, DIM)), mx.zeros((2, HEADS, 1, DIM)))
    with pytest.raises(ValueError, match="lane lengths"):
        other.bucketed_attention(mx.zeros((2, HEADS, 1, DIM)), 1.0, mask)
    txn.commit([2, 2])


def test_deferred_abort_restores_rings():
    prompts, lengths = (30, 12), (4, 3)
    rows = [_lane(i, p) for i, p in enumerate(prompts)]
    owner = HybridVerifyRows(rows, deferred_commit=True, batched_rings=True)
    txn = owner.begin(list(lengths))
    _ring_step(txn.caches[0], prompts, lengths, 0, window=WINDOW, deferred=True)
    _verify_forward(txn.caches[1:], prompts, lengths, first_layer=1)
    txn.abort()
    for lane, prompt in enumerate(prompts):
        _assert_same_lane(rows[lane], _lane(lane, prompt))
