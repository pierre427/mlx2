"""Hy4-flash prefill scaling (CPU): Engram reads stay linear, band attention.

* Engram (exact fix): a prefill chunk whose rows outnumber the reader's LRU
  (``cache_rows``; 65,536 per table by default, 16 rows x 4096+ tokens on
  Hy4-flash) used to land its own prefetched rows into the LRU and evict
  them before the read consumed them, so the foreground read fetched them a
  second time.  Each unique row is now read from disk once per lookup, at
  every size, and the values are unchanged.
* ``sliding_attention_blocks`` (opt-in band attention): numerically
  equivalent to the whole-slice path (bounded difference, not bit-identical),
  unchanged for chunks of at most one block (decode, verify), and through the
  model on the tiny fixture: prefill, chunked prefill and decode stay within
  the bound of the default path.
"""
import time

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.models import hy_v4_flash as H  # noqa: E402
from mlx2.runtime.models import hy_v4_flash_engram as E  # noqa: E402
from test_hy_v4_flash_cpu import TOKENS, build, checkpoint, tiny_config  # noqa: E402


def _reader(tmp_path, rows=4096, dim=8, cache_rows=64):
    from safetensors.numpy import save_file

    values = np.random.default_rng(3).standard_normal((rows, dim)).astype(np.float32)
    save_file({"t": values}, str(tmp_path / "t.safetensors"))
    reader = E.EngramTableRows(tmp_path / "t.safetensors", "t", cache_rows=cache_rows,
                               decode_workers=2, prefill_workers=4)
    return reader, values


def _settle(reader, timeout=10.0):
    """Wait until every prefetch future has finished (they land on the next call)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with reader._lock:
            if all(f.done() for f, _i in reader._inflight.values()):
                return
        time.sleep(0.01)
    raise AssertionError("prefetch did not finish")


@pytest.mark.parametrize("unique", [32, 256, 1024, 3000])
def test_engram_prefetched_rows_are_read_once_even_past_the_lru(tmp_path, unique):
    reader, values = _reader(tmp_path, cache_rows=64)
    try:
        rng = np.random.default_rng(unique)
        picked = rng.choice(values.shape[0], size=unique, replace=False)
        rows = np.concatenate([picked, picked[: unique // 3]])  # repeats, as n-gram heads have
        rng.shuffle(rows)
        reader.prefetch(rows)
        _settle(reader)
        got = reader.read_host(rows)
        np.testing.assert_array_equal(got, values[rows])
        s = reader.stats
        # Linear: every unique row read from disk exactly once, none in the foreground.
        assert s["bytes_read"] == unique * reader.row_bytes
        assert s["foreground_rows_read"] == 0
        assert s["prefetch_rows_consumed"] + s["prefetch_rows_landed"] >= unique
    finally:
        reader.close()


def test_engram_read_waits_on_in_flight_prefetch_and_reads_once(tmp_path, monkeypatch):
    # read_host arrives while the prefetch is still reading: it waits on the
    # in-flight chunks instead of reading their rows again.
    reader, values = _reader(tmp_path, cache_rows=64)
    real = reader._read_chunk
    gate = __import__("threading").Event()

    def slow(fd, rows):
        gate.wait(5.0)
        return real(fd, rows)

    monkeypatch.setattr(reader, "_read_chunk", slow)
    try:
        rows = np.arange(0, 2000, 3)
        reader.prefetch(rows, required=True)
        with reader._lock:
            assert not all(f.done() for f, _i in reader._inflight.values())
        __import__("threading").Timer(0.2, gate.set).start()
        np.testing.assert_array_equal(reader.read_host(rows), values[rows])
        s = reader.stats
        assert s["foreground_rows_read"] == 0 and s["prefetch_rows_consumed"] == rows.size
        assert s["bytes_read"] == rows.size * reader.row_bytes
    finally:
        gate.set()
        reader.close()


@pytest.mark.parametrize("prefetch", [False, True])
def test_engram_cached_rows_own_their_memory(tmp_path, prefetch):
    # Every LRU row is its own copy: the cache stays within cache_rows rows,
    # never a pinned chunk or gathered prefetch array.
    reader, values = _reader(tmp_path, cache_rows=64)
    try:
        rows = np.arange(0, 1500, 7)
        if prefetch:
            reader.prefetch(rows, required=True)
        np.testing.assert_array_equal(reader.read_host(rows), values[rows])
        np.testing.assert_array_equal(reader.read_host(rows[-10:]), values[rows[-10:]])  # hits
        cached = list(reader._cache.values())
        assert len(cached) == reader.cache_rows
        assert all(v.base is None and v.flags.owndata for v in cached)
        assert sum(v.nbytes for v in cached) <= reader.cache_rows * values.shape[1] * 4
        # The latest lookup's hits stay cached (most recent).
        assert set(rows[-10:].tolist()) <= set(reader._cache)
    finally:
        reader.close()


def test_engram_two_prefetches_before_reads_stay_exact(tmp_path):
    # Batched lanes: lane B's prefetch lands lane A's rows; A's read is exact.
    reader, values = _reader(tmp_path, cache_rows=16)
    try:
        a, b = np.arange(0, 200), np.arange(1000, 1300)
        reader.prefetch(a)
        _settle(reader)
        reader.prefetch(b)
        np.testing.assert_array_equal(reader.read_host(a), values[a])
        np.testing.assert_array_equal(reader.read_host(b), values[b])
        assert reader.read_host(np.array([], np.int64)).shape == (0, values.shape[1])
    finally:
        reader.close()


# ------------------------------------------------------------ band attention --

def _band_case(heads, L, start, window, block, dtype, width=24, rank=16, seed=0, square=True):
    rng = np.random.default_rng(seed)
    first = max(0, start - window + 1 - H.ROLLBACK)
    T = start + L - first
    q = mx.array(rng.standard_normal((1, heads, L, width)).astype(np.float32)).astype(dtype)
    kv = mx.array(rng.standard_normal((1, 1, T, width)).astype(np.float32)).astype(dtype)
    sinks = mx.array(rng.standard_normal(heads).astype(np.float32))
    lo = max(first, start - window + 1)
    kv = kv[:, :, lo - first:]
    qpos, kpos = mx.arange(start, start + L), mx.arange(lo, start + L)
    allowed = (kpos[None] <= qpos[:, None]) & (kpos[None] > qpos[:, None] - window)
    band = H.sliding_band_attend(q, kv, lo, start, window, sinks, 0.3, rank, block)
    if not square:
        return None, np.array(band.astype(mx.float32))
    square = H.mla_attend(q, kv, allowed[None, None], sinks, 0.3, rank)
    return np.array(square.astype(mx.float32)), np.array(band.astype(mx.float32))


@pytest.mark.parametrize("L,start,block", [(64, 0, 16), (100, 0, 32), (96, 37, 16), (130, 500, 64),
                                           (33, 0, 32)])
def test_band_attention_is_numerically_equivalent(L, start, block):
    square, band = _band_case(4, L, start, window=24, block=block, dtype=mx.float32)
    np.testing.assert_allclose(band, square, rtol=0, atol=2e-6)


@pytest.mark.parametrize("L,start,block", [(160, 0, 32), (256, 100, 64), (300, 0, 128)])
def test_band_attention_bf16_bound(L, start, block):
    """bf16: at most one ulp of the row's largest output; measured against each
    value's own ulp it reaches tens of ulps near zero (cancellation in the
    value sum), and the error against an fp32 reference is unchanged."""
    square, band = _band_case(4, L, start, window=24, block=block, dtype=mx.bfloat16)
    bf16_ulp = lambda x: np.spacing(np.abs(x).astype(np.float32)) * 2 ** 16
    diff = np.abs(band - square)
    row_ulp = bf16_ulp(np.abs(square).max(axis=-1, keepdims=True))
    assert (diff <= row_ulp).all()
    own = diff[np.abs(square) > 0] / bf16_ulp(square[np.abs(square) > 0])
    assert own.max() <= 64  # measured 0 / 41 / 2 on these cases
    reference, _ = _band_case(4, L, start, window=24, block=block, dtype=mx.float32)
    err_square, err_band = np.abs(square - reference).mean(), np.abs(band - reference).mean()
    assert err_band <= err_square * 1.05


def test_band_attention_single_block_is_bit_identical():
    square, band = _band_case(4, 32, 40, window=24, block=32, dtype=mx.float32)
    np.testing.assert_array_equal(band, square)


def test_band_attention_bounds_live_scores(monkeypatch):
    evals = []
    real = mx.eval
    monkeypatch.setattr(H.mx, "eval", lambda *a: (evals.append(1), real(*a))[1])
    monkeypatch.setattr(H, "SCORE_BUDGET", 4 * 16 * 40 * 2)  # two blocks per evaluation
    _band_case(4, 128, 0, window=24, block=16, dtype=mx.float32, square=False)
    assert 2 <= len(evals) <= 4  # 8 blocks, evaluated in groups; never one sync per block
    evals.clear()
    monkeypatch.setattr(H, "SCORE_BUDGET", 1 << 27)
    _band_case(4, 128, 0, window=24, block=16, dtype=mx.float32, square=False)
    assert evals == []  # fits one budget: no early evaluation


BLOCK = 64  # the tiny window is 8; the model's floor is max(window, ROLLBACK) = 64
LONG = [int(t) for t in np.random.default_rng(11).integers(4, 90, size=260)]


@pytest.fixture(scope="module")
def tiny_pair():
    config = tiny_config()
    W = checkpoint(config)
    base, _ = build(config, W)
    band, _ = build(config, W)
    band.set_sliding_attention_blocks(BLOCK)
    return base, band


def _run(model, chunks, decode):
    cache, outs = model.make_cache(), []
    for a, b in chunks:
        outs.append(model(mx.array([LONG[a:b]]), cache))
    outs += [model(mx.array([[t]]), cache) for t in decode]
    return np.array(mx.concatenate(outs, axis=1)), cache


def test_model_band_prefill_and_decode_match_default(tiny_pair):
    base, band = tiny_pair
    before = sum(l.self_attn.band_forwards for l in band.model.layers)
    a, _ = _run(base, [(0, 200)], LONG[200:210])
    b, _ = _run(band, [(0, 200)], LONG[200:210])
    np.testing.assert_allclose(b, a, rtol=0, atol=2e-5)
    assert band.sliding_attention_blocks == BLOCK
    assert sum(l.self_attn.band_forwards for l in band.model.layers) > before


def test_model_band_second_chunk_reads_ring_and_rollback_history(tiny_pair):
    # The second chunk (150 rows > one block) starts after an earlier chunk:
    # its first blocks read cached ring rows, and the ring also holds the
    # ROLLBACK rows the band must skip.  A rollback then a re-prefill too.
    base, band = tiny_pair
    chunks = [(0, 70), (70, 220)]
    a, cache_a = _run(base, chunks, LONG[220:226])
    b, cache_b = _run(band, chunks, LONG[220:226])
    np.testing.assert_allclose(b, a, rtol=0, atol=2e-5)
    sliding = [c for c in cache_b if c.kind == H.SLIDING]
    assert sliding and all(c.start > 0 for c in sliding)  # the ring has wrapped
    for cache in (cache_a, cache_b):
        for c in cache:
            c.trim(30)
    ra = np.array(base(mx.array([LONG[196:226]]), cache_a))
    rb = np.array(band(mx.array([LONG[196:226]]), cache_b))
    np.testing.assert_allclose(rb, ra, rtol=0, atol=2e-5)


def test_model_band_leaves_short_chunks_bit_identical(tiny_pair):
    base, band = tiny_pair
    # Chunks of at most one block (decode, verify and tree forwards, which
    # are at most ROLLBACK rows) never take the band.
    cache_a, cache_b = base.make_cache(), band.make_cache()
    for a in range(0, 3 * BLOCK, BLOCK):
        x = mx.array([LONG[a:a + BLOCK]])
        np.testing.assert_array_equal(np.array(band(x, cache_b)), np.array(base(x, cache_a)))
    for bad in (-1, 1.5, 4, BLOCK - 1):
        with pytest.raises(ValueError, match="sliding_attention_blocks"):
            band.set_sliding_attention_blocks(bad)
    assert band.sliding_attention_min_block == max(8, H.ROLLBACK)


def test_model_band_media_prefill_matches_default():
    from mlx2.runtime.models import hy_v4_flash_vision as V
    from test_hy_v4_flash_vision_cpu import GRIDS, image_prompt, pixels_for, vision_config, vit_checkpoint

    config = vision_config()
    W, VW = checkpoint(config), vit_checkpoint()
    outs = []
    for rows in (0, BLOCK):
        model, _ = build(config, W)
        model.attach_vision(V.load_vision_tower(config, {k: mx.array(v) for k, v in VW.items()}))
        model.set_sliding_attention_blocks(rows)
        base_prompt = image_prompt()
        # Text after the images so the media chunk is longer than one block.
        prompt = base_prompt[:-3] + LONG[:BLOCK + 20] + base_prompt[-3:]
        cache = model.make_cache()
        parts = [model(mx.array([prompt[:-1]]), cache, pixel_values=mx.array(pixels_for(GRIDS)),
                       image_grid_thw=GRIDS)]
        parts += [model(mx.array([[t]]), cache) for t in [prompt[-1], 30, 31, 33]]
        outs.append(np.array(mx.concatenate(parts, axis=1)))
        if rows:
            assert sum(l.self_attn.band_forwards for l in model.model.layers) > 0
    np.testing.assert_allclose(outs[1], outs[0], rtol=0, atol=2e-5)


def test_adapter_policy_key_and_namespace():
    from mlx2.adapters import hy_v4_flash as A

    assert "sliding_attention_blocks" in A.MODEL_KEYS
    base = A.numerics_layout(A.CACHE_LAYOUT)
    assert base == A.CACHE_LAYOUT
    assert A.numerics_layout(A.CACHE_LAYOUT, sliding_attention_blocks=512) == \
        A.CACHE_LAYOUT + "+swa-band512-v1"
    model, _, _ = A._parse_policy({"sliding_attention_blocks": 512})
    assert model == {"sliding_attention_blocks": 512}
    for bad in (-1, 1.5, "512", True):
        with pytest.raises(ValueError, match="sliding_attention_blocks"):
            A._parse_policy({"sliding_attention_blocks": bad})


def test_sparse_prefill_policy_key_namespace_and_receipt():
    from mlx2.adapters import hy_v4_flash as A

    assert "sparse_prefill_attention" in A.MODEL_KEYS
    assert "sparse_prefill_attention" in A.HYV4FlashAdapter.route_neutral_policy_keys
    assert A.numerics_layout(A.CACHE_LAYOUT, sparse_prefill_attention=True) == \
        A.CACHE_LAYOUT + "+sparse-prefill-v1"
    assert A.numerics_layout(A.CACHE_LAYOUT, sparse_prefill_attention=False) == A.CACHE_LAYOUT
    model, _, _ = A._parse_policy({"sparse_prefill_attention": True})
    assert model == {"sparse_prefill_attention": True}
    for bad in (1, "true", None):
        with pytest.raises(ValueError, match="sparse_prefill_attention"):
            A._parse_policy({"sparse_prefill_attention": bad})
    stub = A.HYV4FlashAdapter.__new__(A.HYV4FlashAdapter)
    receipt = stub._sparse_prefill_receipt()
    assert receipt["selected"] is False and receipt["default"] is False
    assert receipt["observed_used"] is False and receipt["qualified"] is False


def test_sparse_prefill_on_cpu_declines_to_the_bit_identical_dense_path(tiny_pair):
    # The kernel is Metal-only: on the CPU every > 64-row indexed chunk takes
    # the dense reference and counts a decline; outputs are unchanged.
    base, _ = tiny_pair
    a, _ = _run(base, [(0, 120), (120, 200)], LONG[200:204])
    base.set_sparse_prefill_attention(True)
    try:
        b, _ = _run(base, [(0, 120), (120, 200)], LONG[200:204])
        indexed = [l.self_attn for l in base.model.layers if l.self_attn.kind == H.INDEXED]
        assert sum(x.sparse_prefill_forwards for x in indexed) == 0
        assert sum(x.sparse_prefill_declines.get("metal_unavailable", 0) for x in indexed) == 2 * len(indexed)
    finally:
        base.set_sparse_prefill_attention(False)
    np.testing.assert_array_equal(b, a)
    assert not base.sparse_prefill_attention


def test_engram_required_prefetch_is_never_declined(tmp_path):
    # The forward's own rows past MAX_INFLIGHT_ROWS (16 heads x 4096 tokens
    # on Hy4-flash) used to be declined and read in the foreground.
    reader, values = _reader(tmp_path, cache_rows=64)
    try:
        reader.MAX_INFLIGHT_ROWS = 100
        rows = np.arange(0, 1000)
        assert reader.prefetch(rows, required=True) == 1000
        assert reader.stats["prefetch_rows_declined"] == 0
        _settle(reader)
        np.testing.assert_array_equal(reader.read_host(rows), values[rows])
        assert reader.stats["foreground_rows_read"] == 0
        assert reader.stats["bytes_read"] == 1000 * reader.row_bytes
        # Speculative prefetches keep the cap.
        assert reader.prefetch(np.arange(2000, 2300)) <= 100
    finally:
        reader.close()


def test_engram_plan_prefetches_its_rows_as_required(tiny_pair):
    base, _ = tiny_pair
    calls = []

    class Spy:
        def __init__(self, inner):
            self.inner, self.rows = inner, inner.rows

        def prefetch(self, rows, **kw):
            calls.append(kw)
            return 0

        def read(self, rows):
            return self.inner.read(rows)

    saved = dict(base.engram_rows)
    try:
        base.engram_rows = {i: Spy(r) for i, r in saved.items()}
        base(mx.array([TOKENS]))
    finally:
        base.engram_rows = saved
    assert calls and all(kw == {"required": True} for kw in calls)
