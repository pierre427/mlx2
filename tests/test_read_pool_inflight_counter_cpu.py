"""A finished read is not "in flight" while its done callback is pending.

Future.result() can wake the caller before the pool's done callback retires
the future from _pending.  counters()["weight_read_inflight"] counted that
retired-in-waiting future, so an Engram ``park()`` issued right after a
completed ``read()`` raised "cannot park while Engram reads remain owned"
under load (preflight fd3ea47b, test_deepseek_v41_residency_ownership_cpu).
"""

from mlx2.runtime.io_ownership import ReadPool


def test_completed_read_is_not_counted_before_its_callback_runs(monkeypatch):
    held = []
    monkeypatch.setattr(ReadPool, "_completed", lambda self, future: held.append(future))
    pool = ReadPool(1, "inflight-test")
    try:
        future = pool.submit(lambda: 7)
        assert future.result(timeout=10) == 7
        # The callback has not retired the future: it is still in _pending.
        assert future in pool._pending
        assert pool.counters()["weight_read_inflight"] == 0
    finally:
        monkeypatch.undo()
        for future in held:
            pool._completed(future)
        pool.close()
