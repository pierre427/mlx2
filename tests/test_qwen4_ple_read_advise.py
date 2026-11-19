"""Opt-in PLE read-ahead hint (recon-probes-20261113, omlx #5225 analogue)."""
import numpy as np
import pytest

from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.runtime.models import qwen4_ple_nvme as ple
from mlx2.runtime.models.qwen4_ple_nvme import (
    PREFILL_ID_THRESHOLD,
    READ_ADVISE_ENV,
    FileBackedShardedEmbedding,
    ple_table_diagnostics,
)

ROWS = 1024
DIMS = 32


def _table_file(tmp_path):
    row_bytes = DIMS // 2 + 2 * (DIMS // 32) * 2
    path = tmp_path / "ple_rows.bin"
    rng = np.random.default_rng(7)
    path.write_bytes(rng.integers(0, 256, ROWS * row_bytes, dtype=np.uint8).tobytes())
    return path, row_bytes


def _expected(path, row_bytes, ids):
    raw = np.frombuffer(path.read_bytes(), np.uint8).reshape(ROWS, row_bytes)
    return raw[ids]


@pytest.fixture
def advise_calls(monkeypatch):
    calls = []

    def fake(fd, backend, offsets, row_bytes):
        calls.append((backend, list(offsets), row_bytes))
        return len(offsets)

    monkeypatch.setattr(ple, "_advise_rows", fake)
    return calls


def test_default_off_keeps_receipts_and_issues_no_hint(tmp_path, monkeypatch, advise_calls):
    monkeypatch.delenv(READ_ADVISE_ENV, raising=False)
    monkeypatch.delenv("MLX_QWEN4_PLE_NVME_READ_POLICY", raising=False)
    path, row_bytes = _table_file(tmp_path)
    table = FileBackedShardedEmbedding(str(path), ROWS, DIMS, 2)
    try:
        ids = np.array([3, 900, 17], dtype=np.int64)
        assert np.array_equal(table._pread_rows(ids, 1), _expected(path, row_bytes, ids))
        assert advise_calls == []
        assert "advise" not in table.read_policy_status
        assert "read_policy" not in ple_table_diagnostics(table)
    finally:
        table.close()


def test_serial_decode_read_hints_every_row_first(tmp_path, monkeypatch, advise_calls):
    monkeypatch.setenv(READ_ADVISE_ENV, "1")
    monkeypatch.setenv("MLX_QWEN4_PLE_NVME_READ_POLICY", "serial")
    path, row_bytes = _table_file(tmp_path)
    table = FileBackedShardedEmbedding(str(path), ROWS, DIMS, 2)
    try:
        ids = np.array([5, 600, 1023, 0], dtype=np.int64)
        rows = table._pread_rows(ids, 16)
        assert np.array_equal(rows, _expected(path, row_bytes, ids))
        assert len(advise_calls) == 1
        _backend, offsets, rb = advise_calls[0]
        assert rb == row_bytes
        assert offsets == [int(i) * row_bytes for i in ids]
        status = table.read_policy_status
        assert status["advise"]["calls"] == 1
        assert status["advise"]["rows_requested"] == 4
        assert status["advise"]["rows_accepted"] == 4
        assert status["advise"]["qualified"] is False
        assert status["last_receipt"]["advised"] is True
        assert ple_table_diagnostics(table)["read_policy"]["advise"]["calls"] == 1
    finally:
        table.close()


def test_pooled_and_prefill_reads_are_not_hinted(tmp_path, monkeypatch, advise_calls):
    monkeypatch.setenv(READ_ADVISE_ENV, "1")
    monkeypatch.setenv("MLX_QWEN4_PLE_NVME_READ_POLICY", "pooled")
    path, row_bytes = _table_file(tmp_path)
    table = FileBackedShardedEmbedding(str(path), ROWS, DIMS, 2)
    try:
        ids = np.arange(8, dtype=np.int64)
        assert np.array_equal(table._pread_rows(ids, 4), _expected(path, row_bytes, ids))
        assert advise_calls == []  # multi-worker read: hint measured slower there
        wide = np.arange(PREFILL_ID_THRESHOLD, dtype=np.int64)
        assert np.array_equal(table._pread_rows(wide, 1), _expected(path, row_bytes, wide))
        assert advise_calls == []  # prefill-sized read keeps its route
        one = np.array([42], dtype=np.int64)
        table._pread_rows(one, 4)  # one row: nothing to overlap -> not hinted
        assert advise_calls == []
    finally:
        table.close()


def test_real_backend_hint_does_not_change_rows(tmp_path, monkeypatch):
    if ple._read_advise_backend() is None:
        pytest.skip("no read-advise backend on this platform")
    monkeypatch.setenv(READ_ADVISE_ENV, "1")
    monkeypatch.setenv("MLX_QWEN4_PLE_NVME_READ_POLICY", "serial")
    path, row_bytes = _table_file(tmp_path)
    table = FileBackedShardedEmbedding(str(path), ROWS, DIMS, 2)
    try:
        ids = np.array([1, 513, 77], dtype=np.int64)
        assert np.array_equal(table._pread_rows(ids, 1), _expected(path, row_bytes, ids))
        assert table.read_policy_status["advise"]["rows_requested"] == 3
    finally:
        table.close()


def test_bad_flag_value_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv(READ_ADVISE_ENV, "yes")
    path, _ = _table_file(tmp_path)
    with pytest.raises(ValueError, match=READ_ADVISE_ENV):
        FileBackedShardedEmbedding(str(path), ROWS, DIMS, 2)


def test_policy_field_maps_to_env_only_when_on():
    assert "MLX_QWEN4_PLE_NVME_READ_ADVISE" not in FlashNextPolicy().environment()
    on = FlashNextPolicy(ple_read_policy="adaptive", ple_read_advise=True).environment()
    assert on["MLX_QWEN4_PLE_NVME_READ_ADVISE"] == "1"
    serial = FlashNextPolicy(ple_read_policy="serial", ple_read_advise=True).environment()
    assert serial["MLX_QWEN4_PLE_NVME_READ_ADVISE"] == "1"
    with pytest.raises(ValueError, match="ple_read_advise"):
        FlashNextPolicy(ple_read_policy="adaptive", ple_read_advise=1)
    with pytest.raises(ValueError, match="serial or adaptive"):
        FlashNextPolicy(ple_read_advise=True)  # pooled: the hint could never run


def test_serial_single_row_read_is_not_hinted(tmp_path, monkeypatch, advise_calls):
    monkeypatch.setenv(READ_ADVISE_ENV, "1")
    monkeypatch.setenv("MLX_QWEN4_PLE_NVME_READ_POLICY", "serial")
    path, row_bytes = _table_file(tmp_path)
    table = FileBackedShardedEmbedding(str(path), ROWS, DIMS, 2)
    try:
        ids = np.array([9], dtype=np.int64)
        assert np.array_equal(table._pread_rows(ids, 1), _expected(path, row_bytes, ids))
        assert advise_calls == []
        assert table.read_policy_status["last_receipt"]["advised"] is False
    finally:
        table.close()
