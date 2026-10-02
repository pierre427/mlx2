"""adapter.close() releases the early-load weight stream (sweep 2026-10-02 V1).

serving.py skips ``expert_stream.close()`` for an adapter-owned manager on the
promise that ``adapter.close()`` releases it.  The real load paths also put
the manager in ``_tables``, which close() walks; a manager that is only on
``weight_stream`` (the sweep repro) leaked its read pool and descriptors.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest

from mlx2.adapters.qwen35_122b import Qwen35122BA10BAdapter
from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter
from mlx2.adapters.qwen38_27b import Qwen3827BAdapter


class _Manager:
    def __init__(self):
        self.closes = 0
        self._pool = ThreadPoolExecutor(2)
        self._pool.submit(lambda: None).result()

    def close(self):
        self.closes += 1
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None


CLASSES = [Qwen3827BAdapter, Qwen3635BA3BAdapter, Qwen35122BA10BAdapter]


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_close_releases_weight_stream_outside_tables(cls):
    adapter = object.__new__(cls)
    adapter._tables = []
    adapter.weight_stream = manager = _Manager()
    adapter.close()
    assert manager.closes == 1 and manager._pool is None


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_close_does_not_close_a_tabled_stream_twice(cls):
    adapter = object.__new__(cls)
    adapter.weight_stream = manager = _Manager()
    adapter._tables = [manager]
    adapter.close()
    assert manager.closes == 1
