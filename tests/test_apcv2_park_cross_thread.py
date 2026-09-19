"""A park that arrives on another thread is completed by the owning worker.

GPU 2026-09-18: MLX streams are thread-local, so the HTTP thread's inline
spill raised "There is no Stream(...) in current thread", every park stayed
pending, the qualifier resumed at once and feature_apc_sessions failed for
every model.
"""

import queue
import tempfile
import threading

import mlx.core as mx

from mlx2.runtime.apc_v2 import APCv2
from mlx2.runtime.models.cache import KVCache


class Worker:
    """One long-lived thread that owns its stream, like the generation worker."""

    def __init__(self):
        self.jobs = queue.Queue()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        stream = mx.new_stream(mx.default_device())
        while True:
            fn, done = self.jobs.get()
            if fn is None:
                return
            with mx.stream(stream):
                done.put(fn())

    def run(self, fn):
        done = queue.Queue()
        self.jobs.put((fn, done))
        return done.get()

    def close(self):
        self.jobs.put((None, None))
        self.thread.join()


def test_cross_thread_park_defers_to_the_worker_and_completes():
    apc = APCv2(layout_name="t", idle_disk_dir=tempfile.mkdtemp(), idle_disk_seconds=600)
    key = apc.key("t", revision="r")
    tag = ("default", "s1")
    worker = Worker()
    try:
        def store():
            cache = [KVCache() for _ in range(2)]
            for layer in cache:
                layer.update_and_fetch(mx.ones((1, 2, 6, 8)) * 2, mx.ones((1, 2, 6, 8)))
            apc.store(key, [1, 2, 3, 4, 5, 6], cache, session_tag=tag)

        worker.run(store)
        parked = apc.park_session(*tag, ttl_seconds=60)  # a foreign (HTTP) thread
        if parked["state"] == "disk":
            return  # an MLX build that lets any thread evaluate the arrays
        stats = apc._disk_stats
        assert parked["park_pending"] == 1
        assert stats["park_deferred_to_worker"] == 1 and stats["spill_failures"] == 0
        # The forced scan is not throttled by the 1 s idle-scan interval.
        assert worker.run(apc.spill_idle_entries) == 1
        assert apc.session_state(*tag)["state"] == "disk"
    finally:
        worker.close()
