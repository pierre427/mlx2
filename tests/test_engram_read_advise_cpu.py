"""Opt-in Engram read-ahead hint (omlx #5274 / #5464 analogue), CPU only.

Hy4-flash ``EngramTableRows`` hints a verify round's later-layer rows before
the first layer is read; DeepSeek-V4.1 ``EngramRows`` hints each prefetch
chunk's spans before its preads.  Both default off; rows must be
byte-identical with the hint on, off, or refused by the kernel.
"""

import json
import os
import time

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime import read_advise as ra  # noqa: E402
from mlx2.runtime.models import deepseek_v41_engram as v41  # noqa: E402
from mlx2.runtime.models import hy_v4_flash_engram as hy  # noqa: E402

ROWS, DIM = 600, 8


@pytest.fixture
def spans_seen(monkeypatch):
    calls = []

    def fake(fd, backend, spans):
        spans = list(spans)
        calls.append(spans)
        return len(spans)

    monkeypatch.setattr(ra, "advise_spans", fake)
    return calls


def _wait(predicate, timeout=5.0):
    end = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > end:
            raise AssertionError("advice task did not run")
        time.sleep(0.001)


# ---------------------------------------------------------------- Hy4 ----

def _hy_table(tmp_path):
    rng = np.random.default_rng(3)
    table = rng.standard_normal((ROWS, DIM)).astype(np.float32)
    bf16 = (table.view(np.uint32) >> 16).astype(np.uint16)
    pad = np.zeros(5, np.float32).tobytes()
    header = json.dumps({"pad": {"dtype": "F32", "shape": [5], "data_offsets": [0, len(pad)]},
                         "t": {"dtype": "BF16", "shape": [ROWS, DIM],
                               "data_offsets": [len(pad), len(pad) + bf16.nbytes]}}).encode()
    path = tmp_path / "x.safetensors"
    path.write_bytes(len(header).to_bytes(8, "little") + header + pad + bf16.tobytes())
    return path, (bf16.astype(np.uint32) << 16).view(np.float32)


def test_hy4_default_off_issues_no_hint(tmp_path, monkeypatch, spans_seen):
    monkeypatch.delenv(hy.READ_ADVISE_ENV, raising=False)
    path, expect = _hy_table(tmp_path)
    reader = hy.EngramTableRows(path, "t", cache_rows=64)
    try:
        assert reader.read_advise is False
        assert reader.advise(np.array([1, 2, 3])) == 0
        rows = np.array([4, 599, 4, 0])
        np.testing.assert_array_equal(reader.read_host(rows), expect[rows])
        assert spans_seen == []
        status = reader.diagnostics()["read_advise"]
        assert status["configured"] is False and status["calls"] == 0
        assert status["qualified"] is False and status["default"] is False
    finally:
        reader.close()


def test_hy4_hint_offsets_counters_and_identical_rows(tmp_path, monkeypatch, spans_seen):
    monkeypatch.setenv(hy.READ_ADVISE_ENV, "1")
    path, expect = _hy_table(tmp_path)
    reader = hy.EngramTableRows(path, "t", cache_rows=64)
    try:
        reader.read_host(np.array([7]))  # cached rows are not hinted again
        assert reader.advise(np.array([9, 7, 9, 300])) == 2
        _wait(lambda: reader.advisor.status()["calls"] == 1)
        assert spans_seen == [[(reader.start + 9 * reader.row_bytes, reader.row_bytes),
                               (reader.start + 300 * reader.row_bytes, reader.row_bytes)]]
        rows = np.array([9, 300, 7])
        np.testing.assert_array_equal(reader.read_host(rows), expect[rows])
        status = reader.diagnostics()["read_advise"]
        assert status["rows_advised"] == 2 and status["spans_accepted"] == 2
        assert status["failures"] == 0 and "advise_ns" not in status
        # Nothing uncached, out of range, or prefill-sized: no hint.
        assert reader.advise(np.array([9, 300])) == 0
        assert reader.advise(np.array([ROWS])) == 0
        assert reader.advise(np.arange(hy.PREFILL_ROW_THRESHOLD + 3)) == 0
        assert reader.advisor.status()["calls"] == 1
    finally:
        reader.close()
    assert reader.advise(np.array([1])) == 0  # closed: silently nothing


def test_hy4_refused_hint_is_counted_and_harmless(tmp_path, monkeypatch):
    import fcntl

    monkeypatch.setenv(hy.READ_ADVISE_ENV, "1")
    monkeypatch.setenv(ra.TIMING_ENV, "1")
    monkeypatch.setattr(ra, "read_advise_backend", lambda: "f_rdadvise")

    def refuse(*args):
        raise OSError(45, "not supported")

    monkeypatch.setattr(fcntl, "fcntl", refuse)
    path, expect = _hy_table(tmp_path)
    reader = hy.EngramTableRows(path, "t")
    try:
        assert reader.advise(np.array([1, 2, 3])) == 3
        _wait(lambda: reader.advisor.status()["calls"] == 1)
        status = reader.advisor.status()
        assert status["spans_accepted"] == 0 and status["failures"] == 3
        assert "advise_ns" in status
        np.testing.assert_array_equal(reader.read_host(np.array([1, 2, 3])), expect[[1, 2, 3]])
    finally:
        reader.close()


def test_hy4_real_backend_reads_identical(tmp_path, monkeypatch):
    if ra.read_advise_backend() is None:
        pytest.skip("no read-advise backend on this platform")
    monkeypatch.setenv(hy.READ_ADVISE_ENV, "1")
    path, expect = _hy_table(tmp_path)
    reader = hy.EngramTableRows(path, "t")
    try:
        rows = np.array([11, 500, 42, 11])
        reader.advise(rows)
        _wait(lambda: reader.advisor.status()["calls"] == 1)
        np.testing.assert_array_equal(reader.read_host(rows), expect[rows])
        assert reader.advisor.status()["failures"] == 0
    finally:
        reader.close()


def test_bad_flag_values_are_refused(tmp_path, monkeypatch):
    path, _ = _hy_table(tmp_path)
    monkeypatch.setenv(hy.READ_ADVISE_ENV, "yes")
    with pytest.raises(ValueError, match=hy.READ_ADVISE_ENV):
        hy.EngramTableRows(path, "t")
    monkeypatch.setenv(hy.READ_ADVISE_ENV, "1")
    monkeypatch.setattr(ra, "read_advise_backend", lambda: None)
    with pytest.raises(ValueError, match="F_RDADVISE"):
        hy.EngramTableRows(path, "t")


def test_hy4_model_round_hints_later_layers_and_logits_match(tmp_path, monkeypatch, spans_seen):
    from mlx2.adapters import hy_v4_flash as A
    from test_hy_v4_flash_cpu import TOKENS, _write_artifact, checkpoint, tiny_config

    config = tiny_config()
    src = _write_artifact(tmp_path, config, checkpoint(config))
    out = {}
    for arm in ("0", "1"):
        monkeypatch.setenv(hy.READ_ADVISE_ENV, arm)
        model, _, readers = A.load_model(src, A.inspect_artifact(src))
        try:
            prefill = np.array(model(mx.array([TOKENS[:-1]])))
            # The decode step: a fresh token hashes to rows not yet cached.
            decode = np.array(model(mx.array([TOKENS[-1:]])))
            for reader in readers.values():
                _wait(lambda r=reader: r._pool.counters()["weight_read_inflight"] == 0)
            out[arm] = (prefill, decode, {i: r.diagnostics()["read_advise"] for i, r in readers.items()})
        finally:
            for reader in readers.values():
                reader.close()
    np.testing.assert_array_equal(out["0"][0], out["1"][0])
    np.testing.assert_array_equal(out["0"][1], out["1"][1])
    first, *later = sorted(out["1"][2])
    assert later, "tiny config needs two Engram layers"
    assert all(status["calls"] == 0 for status in out["0"][2].values())
    assert out["1"][2][first]["calls"] == 0  # the first layer read is never hinted
    assert sum(out["1"][2][i]["rows_advised"] for i in later) > 0
    assert spans_seen  # only the "1" arm reaches the hint


# --------------------------------------------------------------- V4.1 ----

def _v41_rows(tmp_path, layout):
    rng = np.random.default_rng(5)
    rows = 64
    if layout == "affine":
        codes = rng.integers(0, 2**32, (rows, 32), dtype=np.uint64).astype(np.uint32)
        sc = (rng.uniform(0.01, 0.1, (rows, 8)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
        bi = (rng.uniform(-0.1, 0.1, (rows, 8)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
        blob = codes.tobytes() + sc.tobytes() + bi.tobytes()
        kwargs = dict(offset=0, scale_offset=codes.nbytes, bias_offset=codes.nbytes + sc.nbytes)
        expect = v41.decode_affine4_rows(codes, sc, bi)
    else:
        codes = rng.integers(0, 127, (rows, 256), dtype=np.uint8) | (rng.integers(0, 2, (rows, 256)) << 7).astype(np.uint8)
        scales = rng.integers(120, 134, (rows, 8), dtype=np.uint8)
        packed = np.concatenate([codes, scales], axis=1)
        expect = v41.decode_engram_rows(packed)
        if layout == "row264":
            blob, kwargs = packed.tobytes(), dict(offset=0)
        else:
            blob = codes.tobytes() + scales.tobytes()
            kwargs = dict(offset=0, scale_offset=codes.nbytes)
    path = tmp_path / f"{layout}.bin"
    path.write_bytes(blob)
    return path, rows, kwargs, expect


def _open_v41(path, rows, kwargs):
    return v41.EngramRows(str(path), rows=rows, expected_stat=v41.file_stamp(os.stat(path)),
                          workers=2, **kwargs)


@pytest.mark.parametrize("layout", ["row264", "split", "affine"])
def test_v41_hint_spans_counters_and_identical_rows(tmp_path, monkeypatch, spans_seen, layout):
    path, rows, kwargs, expect = _v41_rows(tmp_path, layout)
    ids = np.array([3, 40, 3, 17, 63])
    monkeypatch.setenv(v41.READ_ADVISE_ENV, "0")
    off = _open_v41(path, rows, kwargs)
    try:
        base = off.read(ids)
        assert spans_seen == [] and off.diagnostics()["read_advise"]["calls"] == 0
    finally:
        off.close()
    monkeypatch.setenv(v41.READ_ADVISE_ENV, "1")
    on = _open_v41(path, rows, kwargs)
    try:
        got = on.read(ids)
        np.testing.assert_array_equal(got, base)
        np.testing.assert_array_equal(got, expect[ids])
        status = on.diagnostics()["read_advise"]
        assert status["calls"] == 2 and status["rows_advised"] == 4  # two workers, four misses
        seen = sorted(span for call in spans_seen for span in call)
        want = []
        for row in (3, 17, 40, 63):
            if layout == "row264":
                want.append((row * 264, 264))
            elif layout == "split":
                want += [(row * 256, 256), (kwargs["scale_offset"] + row * 8, 8)]
            else:
                want += [(row * 128, 128), (kwargs["scale_offset"] + row * 16, 16),
                         (kwargs["bias_offset"] + row * 16, 16)]
        assert seen == sorted(want)
        assert status["spans_requested"] == len(want) == status["spans_accepted"]
        on.read(ids)  # all cached: nothing to hint
        assert on.diagnostics()["read_advise"]["calls"] == 2
    finally:
        on.close()


def test_v41_single_span_chunk_and_wide_lookup_are_not_hinted(tmp_path, monkeypatch, spans_seen):
    path, rows, kwargs, expect = _v41_rows(tmp_path, "row264")
    monkeypatch.setenv(v41.READ_ADVISE_ENV, "1")
    monkeypatch.setattr(v41, "ADVISE_MAX_ROWS", 3)
    table = _open_v41(path, rows, kwargs)
    try:
        np.testing.assert_array_equal(table.read(np.array([5, 6])), expect[[5, 6]])  # 1 row per worker
        np.testing.assert_array_equal(table.read(np.array([7, 8, 9, 10])), expect[[7, 8, 9, 10]])
        assert spans_seen == []
    finally:
        table.close()


def test_v41_real_backend_reads_identical(tmp_path, monkeypatch):
    if ra.read_advise_backend() is None:
        pytest.skip("no read-advise backend on this platform")
    path, rows, kwargs, expect = _v41_rows(tmp_path, "split")
    monkeypatch.setenv(v41.READ_ADVISE_ENV, "1")
    table = _open_v41(path, rows, kwargs)
    try:
        ids = np.array([1, 2, 30, 31, 62])
        np.testing.assert_array_equal(table.read(ids), expect[ids])
        assert table.diagnostics()["read_advise"]["failures"] == 0
    finally:
        table.close()


def test_advise_spans_survives_a_closed_descriptor(tmp_path):
    backend = ra.read_advise_backend()
    if backend is None:
        pytest.skip("no read-advise backend on this platform")
    path = tmp_path / "f"
    path.write_bytes(b"x" * 4096)
    fd = os.open(path, os.O_RDONLY)
    os.close(fd)
    assert ra.advise_spans(fd, backend, [(0, 16), (16, 16)]) == 0
