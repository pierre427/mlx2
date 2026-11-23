"""Hy4-flash Engram read overlap (CPU): prefetch, fan-out, in-step overlap.

The decode/prefill speedups must not change a single value: prefetched rows
equal foreground reads bit for bit, every Engram layer's reads are submitted
before any is waited on, and ``hidden`` starts the GPU on earlier layers
(``mx.async_eval``) before a later Engram layer waits for its rows.
"""
import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.models import hy_v4_flash as H  # noqa: E402
from mlx2.runtime.models import hy_v4_flash_engram as E  # noqa: E402
from test_hy_v4_flash_cpu import TOKENS, _write_artifact, build, checkpoint, tiny_config  # noqa: E402


def _reader(tmp_path, rows=600, dim=32):
    from safetensors.numpy import save_file

    values = np.random.default_rng(1).standard_normal((rows, dim)).astype(np.float32)
    save_file({"t": values}, str(tmp_path / "t.safetensors"))
    return E.EngramTableRows(tmp_path / "t.safetensors", "t", decode_workers=2, prefill_workers=3), values


def test_prefetched_rows_equal_foreground_reads_and_count_once(tmp_path):
    reader, values = _reader(tmp_path)
    try:
        want = np.array([5, 9, 9, 400, 17])
        assert reader.prefetch(want) == 4
        assert reader.prefetch(want) == 0  # already in flight: not resubmitted
        got = reader.read_host(want)
        np.testing.assert_array_equal(got, values[want])
        s = reader.stats
        assert s["prefetch_rows_submitted"] == 4 and s["prefetch_rows_consumed"] + s["prefetch_rows_landed"] == 4
        assert s["bytes_read"] == 4 * reader.row_bytes and s["cache_hits"] in (0, 4)
        # Landed rows are served from the cache afterwards.
        np.testing.assert_array_equal(reader.read_host(np.array([400])), values[[400]])
        # Speculative rows that are never read land in the LRU and stay exact.
        reader.prefetch(np.array([100, 101]))
        np.testing.assert_array_equal(reader.read_host(np.array([101, 100])), values[[101, 100]])
        reader.MAX_INFLIGHT_ROWS = 1
        assert reader.prefetch(np.array([200, 201, 202])) <= 1
        assert reader.stats["prefetch_rows_declined"] >= 2
        np.testing.assert_array_equal(reader.read_host(np.array([200, 201, 202])), values[[200, 201, 202]])
    finally:
        reader.close()
    with pytest.raises(RuntimeError, match="closed"):
        reader.prefetch(np.array([1]))


class _Spy:
    def __init__(self, inner, log, name):
        self.inner, self.log, self.name = inner, log, name
        self.rows = inner.rows

    def prefetch(self, rows):
        self.log.append(("prefetch", self.name))
        return 0

    def read(self, rows):
        self.log.append(("read", self.name))
        return self.inner.read(rows)


@pytest.fixture(scope="module")
def tiny():
    config = tiny_config()
    W = checkpoint(config)
    model, _ = build(config, W)
    return config, W, model


def test_a_forward_submits_every_layers_reads_before_reading_and_overlaps(tiny, monkeypatch):
    _, _, model = tiny
    want = np.array(model(mx.array([TOKENS])))
    log = []
    original = dict(model.engram_rows)
    model.engram_rows = {i: _Spy(r, log, i) for i, r in original.items()}
    async_calls = []
    real_async = H.mx.async_eval
    monkeypatch.setattr(H.mx, "async_eval", lambda *a: (async_calls.append(1), real_async(*a))[1])
    try:
        got = np.array(model(mx.array([TOKENS])))
    finally:
        model.engram_rows = original
    np.testing.assert_array_equal(got, want)
    first_read = next(i for i, (kind, _n) in enumerate(log) if kind == "read")
    assert all(kind == "prefetch" for kind, _n in log[:first_read])
    assert {n for kind, n in log[:first_read]} == set(original)  # every layer submitted first
    # One right after the first Engram layer, one before every later one.
    assert len(async_calls) == 1 + (len(original) - 1)


def test_ssd_reader_forward_is_bit_identical_with_prefetch(tmp_path, tiny):
    from mlx2.adapters import hy_v4_flash as A

    config, W, model = tiny
    path = _write_artifact(tmp_path, config, W)
    loaded, _, readers = A.load_model(path, A.inspect_artifact(path))
    try:
        cache = loaded.make_cache()
        steps = [np.array(loaded(mx.array([TOKENS[:12]]), cache))[:, -1]]
        for t in TOKENS[12:18]:
            steps.append(np.array(loaded(mx.array([[t]]), cache))[:, -1])
        ref_cache = model.make_cache()
        ref = [np.array(model(mx.array([TOKENS[:12]]), ref_cache))[:, -1]]
        for t in TOKENS[12:18]:
            ref.append(np.array(model(mx.array([[t]]), ref_cache))[:, -1])
        for a, b in zip(steps, ref):
            np.testing.assert_array_equal(a, b)
        assert sum(r.stats["prefetch_rows_submitted"] for r in readers.values()) > 0
    finally:
        for r in readers.values():
            r.close()
