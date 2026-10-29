"""A failed forward must not leave an Engram prefetch window owned (CPU only)."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
from mlx2.runtime.models.deepseek_v41_engram import EngramRows, file_stamp


def _table(tmp_path, name="engram.bin"):
    path = tmp_path / name
    raw = np.zeros((16, 264), np.uint8)
    raw[:, 256:] = 127
    raw.tofile(path)
    return EngramRows(str(path), offset=0, rows=16, expected_stat=file_stamp(os.stat(path)))


def test_unread_prefetch_window_is_discarded_and_reader_recovers(tmp_path):
    table = _table(tmp_path)
    try:
        table.prefetch(np.array([1, 2]))
        with pytest.raises(ValueError, match="already owned"):
            table.prefetch(np.array([3]))
        assert table.discard_prefetch() is True
        assert table.discard_prefetch() is False
        assert table.read(np.array([3])).shape == (1, 256)
        table.park()
        table.activate()
        assert table.stats["discarded_prefetches"] == 1
    finally:
        table.close()


def test_model_discard_releases_every_table_and_keeps_original_error(tmp_path):
    first, second = _table(tmp_path, "l1.bin"), _table(tmp_path, "l14.bin")
    try:
        model = SimpleNamespace(engram_tables={1: first, 14: second})
        first.prefetch(np.array([1]))
        second.prefetch(np.array([2]))
        error = RuntimeError("expert read failed mid-forward")
        DeepSeekV41Model.discard_engram_prefetch(model, error)
        assert not getattr(error, "__notes__", None)
        assert first.read(np.array([5])).shape == (1, 256)
        assert second.read(np.array([6])).shape == (1, 256)
    finally:
        first.close()
        second.close()
