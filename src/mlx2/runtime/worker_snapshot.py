"""Collect worker-owned diagnostics outside the admission/publication lock."""

from time import perf_counter_ns


def publish_worker_snapshot(lock, snapshot, collect, *, clock=perf_counter_ns):
    """Publish one complete collection atomically, on the calling worker.

    Collectors keep their existing thread affinity and internal locks. A failed
    collection leaves the previous snapshot intact. Timings are host wall time,
    not device timing; no tensor evaluation is introduced. Publication hold time
    measures through the data update, excluding final timing bookkeeping and
    lock release. It is a lower bound on the complete critical section.
    """
    started = clock()
    values = dict(collect())
    collected = clock()
    with lock:
        acquired = clock()
        previous = snapshot.get("host_snapshot_timing", {})
        timing = {
            "collections": int(previous.get("collections", 0)) + 1,
            "collection_ns": max(0, collected - started),
            "publication_wait_ns": max(0, acquired - collected),
            "publication_hold_ns": 0,
        }
        timing["max_collection_ns"] = max(
            timing["collection_ns"], int(previous.get("max_collection_ns", 0))
        )
        snapshot.update(values)
        # Readers use the same lock, so they cannot observe a partial timing.
        timing["publication_hold_ns"] = max(0, clock() - acquired)
        timing["max_publication_hold_ns"] = max(
            timing["publication_hold_ns"],
            int(previous.get("max_publication_hold_ns", 0)),
        )
        snapshot["host_snapshot_timing"] = timing
