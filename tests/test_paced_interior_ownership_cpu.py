"""Paced interior checkpoints belong to their request's lifecycle (CPU).

9e798f51a queues a prompt's interior checkpoints in the worker and, beside
two or more decoding lanes, publishes one per round.  The queue was not tied
to its owners, so snapshots were stored after the owner's terminal event --
the ownership boundary ``_finish`` documents:

- a client that deleted its APCv2 session on the finish event got the
  session back (``DELETE /v1/apc/sessions/{id}`` removes only what exists);
- a single-LoRA load/unload, which waits for ``engine.jobs`` to empty and
  then clears APCv2, had old-weight snapshots stored after its clear, and a
  later request resumed from them;
- a drain timeout with ``suspend:true`` reported ``suspended`` with no
  resident entry, then the idle loop made the queued snapshots resident;
- a device fault kept the queued snapshots alive through the rebuild's
  release check, outside everything it accounts for, and read a recoverable
  fault as unreleased memory (the worker stopped).

A request's queued snapshots are now published before its successful
terminal event and dropped (``apc_interior_publications_dropped``) when it
ends any other way or the device faults.
"""

import gc
import sys
import threading
import time
import traceback
import weakref
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from route_harness import collect, make_engine, patch_host
from test_06_junction_snapshots import _adapter
from test_apc_hits_hybrid_gdn_self_mtp import (
    Parser,
    make_adapter,
    tiny_qwen38_mtp,
)

from mlx2 import serving
from mlx2.runtime.apc_v2 import APCSessionNotFound


class _Budget:
    """Interior capture is budgeted against the adapter's cache projection."""

    def cache_budget(self, *, mtp):
        from mlx2.adapters.qwen38_memory import Qwen38CacheBudget

        return Qwen38CacheBudget.from_config(dict(vars(self.model.args)), mtp=mtp)


def _budget_adapter(model, vocab):
    return type("Adapter", (_Budget, make_adapter(model, vocab)), {})


class _FaultingParser(Parser):
    @property
    def stop_sequence(self):
        # Read only while the terminal receipt is assembled, after the last
        # token: a receipt defect, as in
        # test_a_failing_terminal_receipt_fails_only_its_own_request.
        raise RuntimeError("injected receipt fault")


class _ReceiptFault:
    def output_parser(self, request):
        if request.get("session_id") == "receipt-fault":
            return _FaultingParser()
        return super().output_parser(request)


def _finished(job, timeout=120):
    while "finish_reason" not in (event := job.events.get(timeout=timeout)):
        assert "error" not in event, event
    return event


def test_paced_interiors_do_not_recreate_a_deleted_session(monkeypatch, tmp_path):
    patch_host(monkeypatch)
    session, tag = "private-session", ("default", "private-session")
    model, vocab = tiny_qwen38_mtp()
    engine = serving.ServingEngine(
        "tiny",
        adapter_factory=_budget_adapter(model, vocab),
        qualification_mode=True,
        mtp=False,
        max_lanes=3,
        prefill_step=16,
        cache_dir=str(tmp_path),
        execution_policy={"apc_interior_checkpoints": {"count": 8, "min_stride": 4}},
    )
    assert engine.ready.wait(60), engine.error
    finished, deleted = threading.Event(), threading.Event()
    interiors = []
    real_emit, real_publish = engine._emit, engine._publish_checkpoint

    def emit(job, event):
        result = real_emit(job, event)
        if "finish_reason" in event and job.request.get("session_id") == session:
            finished.set()
        return result

    def publish(apc, key, tokens, prompt_cache, **kwargs):
        if (
            kwargs.get("retention_role") == "interior_checkpoint"
            and kwargs.get("session_tag") == tag
        ):
            late = finished.is_set()
            interiors.append((len(tokens), late))
            if late:
                # Hold a store made after the finish event until the client's
                # DELETE has returned: one legal interleaving of the threads.
                deleted.wait(5)
        return real_publish(apc, key, tokens, prompt_cache, **kwargs)

    engine._emit = emit
    engine._publish_checkpoint = publish
    try:
        neighbours = [
            engine.submit({"tokens": prompt, "max_tokens": 200, "temperature": 0,
                           "ignore_eos": True})
            for prompt in ([3, 4, 5, 6, 7], [9, 8, 7, 6, 5])
        ]
        for job in neighbours:
            assert "delta" in job.events.get(timeout=60)
        prompt = [(7 * i + 3) % 120 + 1 for i in range(200)]
        owner = engine.submit({"tokens": prompt, "max_tokens": 2, "temperature": 0,
                               "session_id": session})
        _finished(owner)
        removed = engine.apc_session_delete("default", session)
        deleted.set()
        assert removed["removed_entries"] >= 1, removed
        for job in neighbours:
            _finished(job)
        # By this request's end the worker has run rounds with fewer than two
        # lanes, which publish whatever is still queued.
        _finished(engine.submit({"tokens": [5, 6, 7], "max_tokens": 2,
                                 "temperature": 0}))
        with pytest.raises(APCSessionNotFound):
            state = engine.apc_session_state("default", session)
            pytest.fail(f"deleted session is back: {state}; interiors={interiors}")
        counts = dict(engine.counts)
    finally:
        deleted.set()
        engine.close()
    # The scenario ran: the session's prompt captured interiors, and pacing
    # held some of them back beside the two decoding neighbours.
    assert len(interiors) >= 4, interiors
    assert counts["apc_interior_publications_paced"] >= 1, counts
    # The short request keeps its interiors as cache: every one it captured
    # is published, before its finish event.
    assert len(interiors) == counts["apc_interior_checkpoints_captured"], counts
    assert [late for _, late in interiors] == [False] * len(interiors), interiors


@pytest.mark.parametrize("terminal", ["receipt_failure", "overflow_on_finish"])
def test_a_failed_terminal_drops_its_queued_interiors(monkeypatch, terminal):
    """Queued interiors are stored only for a successful terminal.

    The flush ran at the top of the terminal block, before the receipt was
    assembled and the output-overflow claim was made, so a request that
    then ended 500 or 429 had its queued snapshots stored anyway, and
    ``apc_interior_publications_dropped`` did not count them.
    """
    patch_host(monkeypatch)
    session = "receipt-fault" if terminal == "receipt_failure" else "slow-reader"
    tag = ("default", session)
    model, vocab = tiny_qwen38_mtp()
    engine = serving.ServingEngine(
        "tiny",
        adapter_factory=type(
            "Adapter", (_Budget, _ReceiptFault, make_adapter(model, vocab)), {}
        ),
        qualification_mode=True,
        mtp=False,
        max_lanes=3,
        prefill_step=16,
        execution_policy={"apc_interior_checkpoints": {"count": 8, "min_stride": 4}},
    )
    assert engine.ready.wait(60), engine.error
    stored = []
    real_publish = engine._publish_checkpoint

    def publish(apc, key, tokens, prompt_cache, **kwargs):
        if (
            kwargs.get("retention_role") == "interior_checkpoint"
            and kwargs.get("session_tag") == tag
        ):
            stored.append(len(tokens))
        return real_publish(apc, key, tokens, prompt_cache, **kwargs)

    engine._publish_checkpoint = publish
    try:
        neighbours = [
            engine.submit({"tokens": prompt, "max_tokens": 200, "temperature": 0,
                           "ignore_eos": True})
            for prompt in ([3, 4, 5, 6, 7], [9, 8, 7, 6, 5])
        ]
        for job in neighbours:
            assert "delta" in job.events.get(timeout=60)
        prompt = [(7 * i + 3) % 120 + 1 for i in range(200)]
        owner = engine.submit({"tokens": prompt, "max_tokens": 2, "temperature": 0,
                               "session_id": session})
        if terminal == "overflow_on_finish":
            # The overflow harness at the size of this request: a reader that
            # never consumes, and a queue that fills exactly at the finish
            # (one delta per token; nothing is emitted during the prefill).
            owner.events.maxsize = 2
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            with engine.lock:
                if owner.id not in engine.jobs:
                    break
            time.sleep(0.01)
        events = []
        while not owner.events.empty():
            events.append(owner.events.get_nowait())
        for job in neighbours:
            _finished(job)
        # Rounds with fewer than two lanes publish whatever is still queued.
        _finished(engine.submit({"tokens": [5, 6, 7], "max_tokens": 2,
                                 "temperature": 0}))
        counts = dict(engine.counts)
        alive, error = engine.thread.is_alive(), engine.error
    finally:
        engine.close()
    assert alive and error is None
    expected = (
        {"error": "internal error while finishing the request", "status": 500}
        if terminal == "receipt_failure"
        else dict(serving.OUTPUT_OVERFLOW_EVENT)
    )
    assert [e for e in events if "error" in e or "finish_reason" in e] == [
        expected
    ], events
    if terminal == "receipt_failure":
        assert counts["terminal_receipt_failures"] == 1, counts
    captured = counts["apc_interior_checkpoints_captured"]
    # The scenario ran: interiors were captured, and pacing held some back
    # beside the two decoding neighbours until the request's terminal round.
    assert captured >= 4, counts
    assert counts["apc_interior_publications_paced"] >= 1, counts
    # Only what was published before that round is stored; the rest is
    # dropped and counted.
    assert len(stored) < captured, (stored, counts)
    assert counts["apc_interior_publications_dropped"] == captured - len(stored), (
        stored,
        counts,
    )


def test_exclusive_operation_does_not_clear_under_a_queued_interior(monkeypatch):
    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    engine = serving.ServingEngine(
        "tiny",
        adapter_factory=_budget_adapter(model, vocab),
        qualification_mode=True,
        mtp=False,
        max_lanes=2,
        prefill_step=16,
        coalesce_window_ms=200,
        execution_policy={"apc_interior_checkpoints": {"count": 4, "min_stride": 16}},
    )
    assert engine.ready.wait(60), engine.error
    invalidated = threading.Event()
    held = []
    apc = engine.apc
    real_store = apc.store

    def store(key, tokens, prompt_cache, **kwargs):
        if kwargs.get("retention_role") == "interior_checkpoint" and not held:
            with engine.lock:
                live = len(engine.jobs)
            if live == 0:
                # Every owner has its terminal event, so an exclusive
                # operation may run now: hold this store until it has.
                held.append(len(tokens))
                invalidated.wait(3.0)
        return real_store(key, tokens, prompt_cache, **kwargs)

    apc.store = store
    outcome = {}

    def swap_like(_adapter):
        # What a single-LoRA load/unload does once the weights are swapped.
        engine._invalidate_model_state()
        invalidated.set()
        return engine.model_revision

    def operate():
        try:
            outcome["revision"] = engine._exclusive_adapter_operation(
                "lora_unload", swap_like, timeout=60
            )
        except BaseException as error:  # noqa: BLE001 - reported below
            outcome["error"] = error

    def prompt(offset):
        return [(7 * i + 3 + offset) % 126 + 1 for i in range(120)]

    try:
        a = engine.submit({"tokens": prompt(0), "max_tokens": 2, "temperature": 0})
        b = engine.submit({"tokens": prompt(5), "max_tokens": 2, "temperature": 0})
        operator = threading.Thread(target=operate)
        operator.start()
        _finished(a)
        _finished(b)
        operator.join(60)
        assert not operator.is_alive()
        assert "error" not in outcome, outcome
        assert outcome["revision"] == 1
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and held and not invalidated.is_set():
            time.sleep(0.05)
        time.sleep(0.3)  # idle rounds: anything still queued would land now
        counts = dict(engine.counts)
        # A request sharing the first prompt's prefix, after the weights
        # changed, must not resume from a snapshot taken before the change.
        probe = engine.submit(
            {"tokens": prompt(0)[:70] + [9] * 30, "max_tokens": 2, "temperature": 0}
        )
        final = _finished(probe)
    finally:
        apc.store = real_store
        engine.close()
    assert counts["apc_interior_checkpoints_captured"] >= 2, counts
    assert counts["apc_interior_publications_paced"] >= 1, counts
    assert probe.cached_tokens == 0, (
        probe.cached_tokens,
        final["receipt"].get("cache_checkpoint_role"),
        held,
    )


def test_drain_timeout_suspend_leaves_no_interior_to_publish(monkeypatch, tmp_path):
    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    engine = make_engine(
        model, vocab, mtp=False, max_lanes=2, cache_dir=str(tmp_path),
        adapter_mixin=_Budget,
        execution_policy={"apc_interior_checkpoints": {
            "count": 4, "min_stride": 16, "placement": "pow2"}},
    )
    apc = engine.apc
    real_publish = engine._publish_checkpoint
    entered, release = threading.Event(), threading.Event()
    interior_states = []

    def publish(apc_, key, tokens, cache, **kwargs):
        if kwargs.get("retention_role") == "interior_checkpoint":
            interior_states.append(engine.service_state()["state"])
            if len(interior_states) == 1:
                # Hold the first paced publication until the drain deadline
                # has passed: the rest are still queued when it expires.
                entered.set()
                assert release.wait(30)
        return real_publish(apc_, key, tokens, cache, **kwargs)

    engine._publish_checkpoint = publish
    real_spill = apc.spill_idle_entries
    idle_tick = threading.Event()

    def spill(*args, **kwargs):
        idle_tick.set()
        return real_spill(*args, **kwargs)

    apc.spill_idle_entries = spill
    results = {}

    def consume(name, job):
        results[name] = collect(job, 60)

    try:
        decoder = engine.submit({"tokens": [5, 6, 7], "max_tokens": 450,
                                 "temperature": 0})
        threads = [threading.Thread(target=consume, args=("decoder", decoder),
                                    daemon=True)]
        threads[0].start()
        deadline = time.monotonic() + 30
        while engine.counts["cycles"] < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        prompt = [(7 * i + 3) % (vocab - 2) + 1 for i in range(200)]
        long_job = engine.submit({"tokens": prompt, "max_tokens": 300,
                                  "temperature": 0})
        threads.append(threading.Thread(target=consume, args=("long", long_job),
                                        daemon=True))
        threads[1].start()
        assert entered.wait(60), dict(engine.counts)

        engine.quiesce(suspend=True, drain_timeout_seconds=0.1)
        time.sleep(0.25)  # the deadline passes while three checkpoints are queued
        release.set()
        assert engine.wait_for_quiesce(30)
        # The idle branch publishes queued interiors before it spills: one
        # more spill call after the transition proves that branch has run.
        idle_tick.clear()
        assert idle_tick.wait(10)
        for thread in threads:
            thread.join(30)
        state = engine.service_state()
        report = state["last_transition"]["result"]["suspend"]
        with apc._apc_lock:
            resident = apc._resident_entry_count_locked()
        alive, error = engine.thread.is_alive(), engine.error
        counts = dict(engine.counts)
    finally:
        release.set()
        engine.close()

    assert alive and error is None
    assert results["long"]["error"] == "drain timeout"
    assert counts["apc_interior_checkpoints_captured"] == 4
    assert counts["apc_interior_publications_paced"] >= 1
    assert state["state"] == "suspended"
    assert state["last_transition"]["result"]["drain_timed_out"] is True
    assert report["resident_entries_after"] == 0
    assert "suspended" not in interior_states, interior_states
    assert resident == 0, (resident, interior_states)
    assert counts["apc_interior_publications_dropped"] == 3, counts


def test_queued_interiors_are_released_before_a_device_fault_rebuild(monkeypatch):
    """The rebuild's release check allows the load baseline, APCv2's resident
    bytes and known extras.  A queued snapshot is none of these, so on
    Qwen3.8-27B one deep 16K checkpoint alone exceeds the 1 GiB tolerance and
    a recoverable fault stopped the worker.  Its objects must be gone by the
    time the check measures, and the failed request's snapshots must never be
    stored after its 503."""
    from mlx2.runtime import generate as G

    patch_host(monkeypatch)
    snapshots = {}
    real_pop = G.BatchGenerator.pop_interior_checkpoints

    def pop(self, uid):
        checkpoints = real_pop(self, uid)
        snapshots.setdefault(int(uid), []).extend(
            weakref.ref(layer)
            for checkpoint in checkpoints
            for layer in checkpoint["target_cache"]
        )
        return checkpoints

    monkeypatch.setattr(G.BatchGenerator, "pop_interior_checkpoints", pop)

    def prompt(seed, n):
        return [(seed * 13 + 7 * i) % 120 + 1 for i in range(n)]

    model, vocab = tiny_qwen38_mtp()
    engine = serving.ServingEngine(
        "tiny", adapter_factory=_adapter(model, vocab), qualification_mode=True,
        mtp=False, max_lanes=3, max_inflight=4, prefill_step=16,
        coalesce_window_ms=1,
        execution_policy={"apc_interior_checkpoints": {"count": 4, "min_stride": 16}},
    )
    assert engine.ready.wait(120), engine.error
    seen = {}
    rebuilt = threading.Event()
    real_rebuild = engine._rebuild_after_device_oom
    real_inject = engine._inject_device_fault

    def rebuild(exc, build_batch, apc, fault="out_of_memory"):
        # The release check runs after the frames are cleared and cycles are
        # collected; look at what is still alive at that point.
        traceback.clear_frames(exc.__traceback__)
        gc.collect()
        seen["alive_at_check"] = sum(
            ref() is not None for ref in snapshots.get(seen["uid"], ())
        )
        seen["published_at_check"] = engine.counts["apc_interior_checkpoints_published"]
        try:
            return real_rebuild(exc, build_batch, apc, fault=fault)
        finally:
            rebuilt.set()

    def inject(active):
        time.sleep(0.005)  # keep the neighbours decoding through the prefill
        try:
            return real_inject(active)
        except RuntimeError:
            counts = engine.counts
            seen["queued"] = counts["apc_interior_checkpoints_captured"] - sum(
                value for key, value in counts.items()
                if key.startswith("apc_interior_checkpoints_")
                and key != "apc_interior_checkpoints_captured"
            )
            raise

    engine._rebuild_after_device_oom = rebuild
    engine._inject_device_fault = inject
    try:
        neighbours = [
            engine.submit({"tokens": prompt(seed, 20), "max_tokens": 400,
                           "temperature": 0})
            for seed in (1, 2)
        ]
        threads = [threading.Thread(target=collect, args=(job,), daemon=True)
                   for job in neighbours]
        for thread in threads:
            thread.start()
        time.sleep(0.3)
        faulted = engine.submit({
            "tokens": prompt(5, 480), "max_tokens": 30, "temperature": 0,
            "mlx_fault": {"kind": "gpu_timeout", "after_tokens": 1},
        })
        deadline = time.monotonic() + 30
        while faulted.uid is None and time.monotonic() < deadline:
            time.sleep(0.001)
        seen["uid"] = int(faulted.uid)
        failed = collect(faulted)
        for thread in threads:
            thread.join(60)
        assert failed.get("status") == 503, failed
        assert rebuilt.wait(30)
        time.sleep(0.3)  # idle rounds: anything still queued would be stored now
        # Precondition: the fault really landed with interiors still queued.
        assert seen["queued"] >= 1, seen
        assert engine.error is None, engine.error
        assert engine.counts["device_gpu_timeout_recoveries"] == 1
        assert seen["alive_at_check"] == 0, seen
        assert (
            engine.counts["apc_interior_checkpoints_published"]
            == seen["published_at_check"]
        ), seen
        assert engine.counts["apc_interior_publications_dropped"] == seen["queued"]
        served = collect(engine.submit({"tokens": prompt(9, 40), "max_tokens": 4,
                                        "temperature": 0}))
        assert "error" not in served, served
    finally:
        engine.close()
