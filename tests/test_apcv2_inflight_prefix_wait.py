"""CPU contracts for APCv2 in-flight shared-prefix leader/follower reuse."""

import time
from collections import Counter, deque
from types import SimpleNamespace

import mlx.core as mx
import pytest
from test_06_junction_snapshots import _adapter
from test_apc_hits_hybrid_gdn_self_mtp import tiny_qwen38_mtp

from mlx2 import memory, serving
from mlx2.runtime import os_memory
from mlx2.runtime.generate import BatchGenerator
from mlx2.runtime.state_boundaries import BoundaryPurpose, StateBoundary
from mlx2.serving import Job, ServingEngine, apc_inflight_prefix_policy


def collect(job):
    output = []
    while True:
        event = job.events.get(timeout=60)
        if "error" in event:
            return output, event
        if "delta" in event:
            output.extend(
                int(token)
                for token in event["delta"].get("content", "").split()
            )
        if "finish_reason" in event:
            return output, event.get("receipt") or {}


def conversation(vocab, tail_seed, *, shared_tokens=96, tail_tokens=32):
    shared = [(7 * index + 3) % (vocab - 2) + 1 for index in range(shared_tokens)]
    tail = [
        (tail_seed * (index + 1) + 17) % (vocab - 2) + 1
        for index in range(tail_tokens)
    ]
    return shared + tail


def make_engine(
    monkeypatch,
    *,
    enabled,
    mtp=False,
    lanes=4,
    coalesce_ms=50,
    tenant_scoped=True,
):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(
        serving, "runtime_identity", lambda: {"source_sha256": "cpu-inflight-prefix"}
    )
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    model, vocab = tiny_qwen38_mtp()
    policy = (
        {"apc_inflight_prefix_wait": {"enabled": True, "min_shared_tokens": 64}}
        if enabled
        else None
    )
    engine = ServingEngine(
        "tiny",
        adapter_factory=_adapter(model, vocab),
        qualification_mode=True,
        mtp=mtp,
        max_lanes=lanes,
        max_inflight=max(8, lanes),
        tenant_scoped_cache=tenant_scoped,
        prefill_step=16,
        coalesce_window_ms=coalesce_ms,
        execution_policy=policy,
    )
    assert engine.ready.wait(60), engine.error
    return engine, vocab


@pytest.mark.parametrize("mtp", [False, True], ids=["ordinary", "self-mtp"])
def test_four_way_different_tails_prefill_the_shared_prefix_once(monkeypatch, mtp):
    engine, vocab = make_engine(monkeypatch, enabled=True, mtp=mtp)
    prompts = [conversation(vocab, seed) for seed in (5, 9, 13, 17)]
    try:
        jobs = [
            engine.submit(
                {"tokens": prompt, "max_tokens": 6, "temperature": 0},
                tenant_id="alice",
            )
            for prompt in prompts
        ]
        results = [collect(job) for job in jobs]
        assert not any("error" in receipt for _output, receipt in results)
        assert sorted(job.cached_tokens for job in jobs) == [0, 96, 96, 96]
        assert sum(job.prompt_tokens - job.cached_tokens for job in jobs) == 224
        waited = [job for job in jobs if job.apc_prefix_wait_done]
        assert len(waited) == 3
        assert all(
            job.apc_prefix_wait_release == "checkpoint_published"
            for job in waited
        )
        assert all(job.apc_prefix_wait_reuse == "checkpoint_reused" for job in waited)
        assert all(job.apc_prefix_wait_tokens == 96 for job in waited)
        assert engine.counts["apc_inflight_prefix_waits"] == 3
        assert engine.counts["apc_inflight_checkpoints_published"] == 1
        assert not engine.apc_inflight_waiters
        assert not engine.apc_inflight_resolutions
        with engine.apc._apc_lock:
            junctions = [
                (len(tokens), entry)
                for _key, tokens, entry in engine.apc._entry_records_locked()
                if getattr(entry, "_apc_retention_role", None) == "junction"
            ]
        shared_entry = next(entry for length, entry in junctions if length == 96)
        assert (shared_entry.sidecar is not None) is mtp
        assert engine.snapshot["settings"]["apc_inflight_prefix_wait"] == {
            "enabled": True,
            "min_shared_tokens": 64,
            "max_wait_ms": 300_000,
        }
        for job, (_output, receipt) in zip(jobs, results):
            wait = receipt["apcv2_inflight_prefix_wait"]
            assert wait["waited"] is job.apc_prefix_wait_done
            assert wait["shared_tokens"] == job.apc_prefix_wait_tokens
            if job.apc_prefix_wait_done:
                assert wait["release"] == "checkpoint_published"
                assert wait["reuse"] == "checkpoint_reused"
    finally:
        engine.close()

    baseline, baseline_vocab = make_engine(monkeypatch, enabled=False, mtp=mtp)
    assert baseline_vocab == vocab
    try:
        baseline_jobs = [
            baseline.submit(
                {"tokens": prompt, "max_tokens": 6, "temperature": 0},
                tenant_id="alice",
            )
            for prompt in prompts
        ]
        baseline_results = [collect(job) for job in baseline_jobs]
        assert [output for output, _receipt in results] == [
            output for output, _receipt in baseline_results
        ]
        assert sum(
            job.prompt_tokens - job.cached_tokens for job in baseline_jobs
        ) == 512
    finally:
        baseline.close()


def test_short_prefix_unrelated_tenant_and_default_off_do_not_wait(monkeypatch):
    engine, vocab = make_engine(monkeypatch, enabled=True, lanes=3)
    try:
        first = engine.submit(
            {"tokens": conversation(vocab, 5, shared_tokens=32), "max_tokens": 8},
            tenant_id="alice",
        )
        short = engine.submit(
            {"tokens": conversation(vocab, 9, shared_tokens=32), "max_tokens": 8},
            tenant_id="alice",
        )
        for job in (first, short):
            collect(job)
        assert not short.apc_prefix_wait_done
        assert engine.counts.get("apc_inflight_prefix_waits", 0) == 0

        tenant_a = engine.submit(
            {"tokens": conversation(vocab, 13), "max_tokens": 8},
            tenant_id="alice",
        )
        tenant_b = engine.submit(
            {"tokens": conversation(vocab, 17), "max_tokens": 8},
            tenant_id="bob",
        )
        collect(tenant_a)
        collect(tenant_b)
        assert not tenant_b.apc_prefix_wait_done

        session_a = engine.submit(
            {"tokens": conversation(vocab, 19), "max_tokens": 8, "session_id": "a"},
            tenant_id="carol",
        )
        session_b = engine.submit(
            {"tokens": conversation(vocab, 23), "max_tokens": 8, "session_id": "b"},
            tenant_id="carol",
        )
        collect(session_a)
        collect(session_b)
        assert not session_b.apc_prefix_wait_done
    finally:
        engine.close()

    cold, vocab = make_engine(monkeypatch, enabled=False, lanes=2)
    try:
        first = cold.submit({"tokens": conversation(vocab, 5), "max_tokens": 8})
        second = cold.submit({"tokens": conversation(vocab, 9), "max_tokens": 8})
        collect(first)
        collect(second)
        assert not second.apc_prefix_wait_done
        assert "apc_inflight_prefix_wait" not in cold.snapshot["settings"]
    finally:
        cold.close()


def test_cancelled_follower_releases_without_cancelling_leader(monkeypatch):
    engine, vocab = make_engine(
        monkeypatch, enabled=True, lanes=2, coalesce_ms=1000
    )
    try:
        leader = engine.submit({"tokens": conversation(vocab, 5), "max_tokens": 64})
        follower = engine.submit({"tokens": conversation(vocab, 9), "max_tokens": 8})
        deadline = time.monotonic() + 10
        while not follower.apc_prefix_wait_done and time.monotonic() < deadline:
            time.sleep(0.001)
        assert follower.apc_prefix_wait_done
        follower.cancelled.set()
        _output, event = collect(follower)
        assert event["error"] == "cancelled"
        assert follower.apc_prefix_wait_release == "cancelled"
        leader_output, leader_receipt = collect(leader)
        assert leader_output and "error" not in leader_receipt
        assert not engine.apc_inflight_waiters
        assert not engine.apc_inflight_resolutions
    finally:
        engine.close()


def test_drain_timeout_finishes_a_parked_follower_once(monkeypatch):
    """``_fail_drain_timeout`` gave the parked follower its 503, then the
    worker's cancellation sweep over ``prefix_waiting`` found it cancelled
    and queued a second terminal for it.  Its wait must still settle (the
    waiter entry and the outcome counters balance)."""
    engine, vocab = make_engine(
        monkeypatch, enabled=True, lanes=2, coalesce_ms=1000
    )
    try:
        # Slow rounds keep the leader short of the shared boundary until the
        # drain deadline has passed.
        engine._inject_device_fault = lambda active: time.sleep(0.1)
        leader = engine.submit({"tokens": conversation(vocab, 5), "max_tokens": 64})
        follower = engine.submit({"tokens": conversation(vocab, 9), "max_tokens": 8})
        deadline = time.monotonic() + 10
        while not follower.apc_prefix_wait_done and time.monotonic() < deadline:
            time.sleep(0.001)
        assert follower.apc_prefix_wait_done
        engine.quiesce(drain_timeout_seconds=0.1, suspend=False)
        assert engine.wait_for_quiesce(10)
        engine._inject_device_fault = lambda active: None
        engine.resume()
        # A later request proves the worker ran further iterations.
        _output, receipt = collect(
            engine.submit({"tokens": conversation(vocab, 13), "max_tokens": 2},
                          tenant_id="bob")
        )
        assert "error" not in receipt, receipt
        terminals = {}
        for name, job in (("leader", leader), ("follower", follower)):
            events = []
            while not job.events.empty():
                events.append(job.events.get_nowait())
            terminals[name] = [
                event for event in events
                if "error" in event or "finish_reason" in event
            ]
        assert terminals == {
            "leader": [{"error": "drain timeout", "status": 503}],
            "follower": [{"error": "drain timeout", "status": 503}],
        }
        assert follower.apc_prefix_wait_release == "cancelled"
        assert not engine.apc_inflight_waiters
        assert not engine.apc_inflight_resolutions
        counts = engine.counts
        assert counts["apc_inflight_prefix_waits"] == 1
        assert counts["apc_inflight_prefix_waits_cancelled"] == 1
    finally:
        engine.close()


def test_cancelled_leader_releases_follower_to_cold_prefill(monkeypatch):
    # Simulate a capture that was accepted by admission but never reaches the
    # publication queue.  The follower must still be released when the leader
    # disappears, and it may wait only once.
    monkeypatch.setattr(BatchGenerator, "add_state_boundary", lambda *args: True)
    engine, vocab = make_engine(monkeypatch, enabled=True, lanes=4)
    try:
        jobs = [
                engine.submit(
                {"tokens": conversation(vocab, seed), "max_tokens": 64}
            )
            for seed in (5, 9, 13, 17)
        ]
        deadline = time.monotonic() + 10
        while (
            not any(job.apc_prefix_wait_done for job in jobs)
            and time.monotonic() < deadline
        ):
            time.sleep(0.001)
        follower = next(job for job in jobs if job.apc_prefix_wait_done)
        leader = next(
            job for job in jobs
            if job.id == follower.apc_prefix_wait_leader_request_id
        )
        leader.cancelled.set()
        _leader_output, leader_event = collect(leader)
        assert leader_event["error"] == "cancelled"
        follower_output, follower_receipt = collect(follower)
        assert follower_output and "error" not in follower_receipt
        assert follower.cached_tokens == 0
        assert follower.apc_prefix_wait_release == "leader_ended"
        assert follower.apc_prefix_wait_reuse == "not_reused"
        assert engine.counts["apc_inflight_prefix_waits"] >= 1
        for job in jobs:
            if job not in (leader, follower):
                job.cancelled.set()
                collect(job)
    finally:
        engine.close()


def test_declined_capture_releases_when_leader_enters_decode(monkeypatch):
    monkeypatch.setattr(BatchGenerator, "add_state_boundary", lambda *args: True)
    engine, vocab = make_engine(monkeypatch, enabled=True, lanes=4)
    try:
        jobs = [
            engine.submit({"tokens": conversation(vocab, seed), "max_tokens": 8})
            for seed in (29, 31, 37, 41)
        ]
        results = [collect(job) for job in jobs]
        assert all(output and "error" not in receipt for output, receipt in results)
        waited = [job for job in jobs if job.apc_prefix_wait_done]
        assert waited
        assert all(
            job.apc_prefix_wait_release == "leader_entered_decode"
            for job in waited
        )
        assert all(job.apc_prefix_wait_reuse == "not_reused" for job in waited)
        leader = next(job for job in jobs if not job.apc_prefix_wait_done)
        assert not leader.apc_inflight_planned_bytes
    finally:
        engine.close()


def test_publish_failure_releases_followers_to_one_cold_retry(monkeypatch):
    original = ServingEngine._publish_checkpoint

    def fail_inflight(self, apc, key, tokens, prompt_cache, **kwargs):
        if len(tokens) == 96 and kwargs.get("retention_role") == "junction":
            return False
        return original(self, apc, key, tokens, prompt_cache, **kwargs)

    monkeypatch.setattr(ServingEngine, "_publish_checkpoint", fail_inflight)
    engine, vocab = make_engine(monkeypatch, enabled=True, lanes=4)
    try:
        jobs = [
            engine.submit({"tokens": conversation(vocab, seed), "max_tokens": 8})
            for seed in (43, 47, 53, 59)
        ]
        results = [collect(job) for job in jobs]
        assert all(output and "error" not in receipt for output, receipt in results)
        waited = [job for job in jobs if job.apc_prefix_wait_done]
        assert len(waited) == 3
        assert all(
            job.apc_prefix_wait_release == "checkpoint_publish_failed"
            for job in waited
        )
        assert all(job.apc_prefix_wait_reuse == "not_reused" for job in waited)
        assert all(job.cached_tokens == 0 for job in waited)
        assert not engine.apc_inflight_waiters
        assert not engine.apc_inflight_resolutions
    finally:
        engine.close()


def test_capture_failure_releases_followers_before_decode(monkeypatch):
    from mlx2.runtime import cow_cache

    def decline(_target):
        raise RuntimeError("forced exact snapshot decline")

    monkeypatch.setattr(cow_cache, "snapshot_prompt_cache_descriptors", decline)
    engine, vocab = make_engine(monkeypatch, enabled=True, lanes=4)
    try:
        jobs = [
            engine.submit({"tokens": conversation(vocab, seed), "max_tokens": 8})
            for seed in (61, 67, 71, 73)
        ]
        results = [collect(job) for job in jobs]
        assert all(output and "error" not in receipt for output, receipt in results)
        waited = [job for job in jobs if job.apc_prefix_wait_done]
        assert len(waited) == 3
        assert all(
            job.apc_prefix_wait_release == "checkpoint_capture_failed"
            for job in waited
        )
        assert all(job.apc_prefix_wait_reuse == "not_reused" for job in waited)
        assert all(job.cached_tokens == 0 for job in waited)
        assert not engine.apc_inflight_waiters
        assert not engine.apc_inflight_resolutions
        assert not next(
            job for job in jobs if not job.apc_prefix_wait_done
        ).apc_inflight_planned_bytes
    finally:
        engine.close()


def test_junction_capture_failure_releases_followers_before_decode(monkeypatch):
    """A follower may ask for a position the leader already planned as a
    learned JUNCTION; the boundary keeps its junction role.  A declined
    capture there was reported only for INFLIGHT boundaries, so the
    followers waited for the leader's decode instead of the documented
    release on capture failure."""
    from mlx2.runtime import cow_cache

    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(
        serving, "runtime_identity", lambda: {"source_sha256": "cpu-inflight-junction"}
    )
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    original = cow_cache.snapshot_prompt_cache_descriptors
    decline = {"on": False}

    def maybe_decline(*args, **kwargs):
        if decline["on"]:
            raise RuntimeError("forced exact snapshot decline")
        return original(*args, **kwargs)

    monkeypatch.setattr(cow_cache, "snapshot_prompt_cache_descriptors", maybe_decline)
    model, vocab = tiny_qwen38_mtp()
    engine = ServingEngine(
        "tiny",
        adapter_factory=_adapter(model, vocab),
        qualification_mode=True,
        mtp=False,
        max_lanes=4,
        max_inflight=8,
        tenant_scoped_cache=True,
        # One chunk past the shared prefix, so the prior request leaves no
        # exact state at or below 96 and the leader starts cold.
        prefill_step=128,
        coalesce_window_ms=50,
        execution_policy={
            "apc_inflight_prefix_wait": {"enabled": True, "min_shared_tokens": 64},
            "apc_junction_checkpoints": True,
        },
    )
    assert engine.ready.wait(60), engine.error
    try:
        # A stored path the next requests branch from at 96: the leader plans
        # a JUNCTION there before any follower asks for the position.
        _output, receipt = collect(
            engine.submit(
                {"tokens": conversation(vocab, 3, tail_tokens=64), "max_tokens": 4}
            )
        )
        assert "error" not in receipt
        decline["on"] = True
        jobs = [
            engine.submit(
                {"tokens": conversation(vocab, seed, tail_tokens=64), "max_tokens": 8}
            )
            for seed in (61, 67, 71, 73)
        ]
        results = [collect(job) for job in jobs]
        assert all(output and "error" not in receipt for output, receipt in results)
        leader = next(job for job in jobs if not job.apc_prefix_wait_done)
        assert leader.state_boundaries == (
            StateBoundary(96, BoundaryPurpose.JUNCTION),
        )
        assert engine.counts["apc_inflight_checkpoints_existing_plan"] == 3
        waited = [job for job in jobs if job.apc_prefix_wait_done]
        assert len(waited) == 3
        assert [job.apc_prefix_wait_release for job in waited] == [
            "checkpoint_capture_failed"
        ] * 3
        assert engine.counts["apc_inflight_prefix_waits_checkpoint_capture_failed"] == 3
        assert not engine.apc_inflight_waiters
        assert not engine.apc_inflight_resolutions
        assert not leader.apc_inflight_planned_bytes
    finally:
        engine.close()


def test_shared_namespace_receipt_does_not_expose_leader_request_id(monkeypatch):
    engine, vocab = make_engine(
        monkeypatch, enabled=True, lanes=4, tenant_scoped=False
    )
    try:
        jobs = [
            engine.submit(
                {
                    "id": f"client-visible-{seed}",
                    "tokens": conversation(vocab, seed),
                    "max_tokens": 8,
                },
                tenant_id=f"tenant-{seed}",
            )
            for seed in (79, 83, 89, 97)
        ]
        results = [collect(job) for job in jobs]
        waited = [
            receipt["apcv2_inflight_prefix_wait"]
            for job, (_output, receipt) in zip(jobs, results)
            if job.apc_prefix_wait_done
        ]
        assert len(waited) == 3
        assert all(item["leader_request_id"] is None for item in waited)
    finally:
        engine.close()


def test_dynamic_boundary_rejects_passed_and_decoding_frontiers():
    generator = object.__new__(BatchGenerator)
    generator._interior_checkpoint_positions = {}
    generator._state_boundary_purposes = {}
    generator._unprocessed_sequences = deque(
        [(7, (), None, None, list(range(10)))]
    )
    generator._prompt_batch = SimpleNamespace(tokens=[list(range(24))])
    generator._find_uids = lambda _uids: {7: (0, 0)}
    boundary = StateBoundary(32, BoundaryPurpose.INFLIGHT)
    assert generator.add_state_boundary(7, boundary, 64)
    assert tuple(generator._interior_checkpoint_positions[7]) == (32,)
    assert generator._state_boundary_purposes[7][32] == BoundaryPurpose.INFLIGHT
    assert not generator.add_state_boundary(
        7, StateBoundary(10, BoundaryPurpose.INFLIGHT), 64
    )

    generator._find_uids = lambda _uids: {7: (1, 0)}
    assert not generator.add_state_boundary(
        7, StateBoundary(24, BoundaryPurpose.INFLIGHT), 64
    )
    generator._find_uids = lambda _uids: {7: (2, 0)}
    assert not generator.add_state_boundary(
        7, StateBoundary(40, BoundaryPurpose.INFLIGHT), 64
    )

    generator._find_uids = lambda _uids: {7: (0, 0)}
    generator._unprocessed_sequences = deque(
        [(7, (), None, None, list(range(10)))]
    )
    generator._interior_checkpoint_positions[7] = deque((48,))
    generator._state_boundary_purposes[7][48] = BoundaryPurpose.ROLLING
    assert generator.add_state_boundary(
        7, StateBoundary(48, BoundaryPurpose.INFLIGHT), 64
    )
    assert generator._state_boundary_purposes[7][48] == BoundaryPurpose.INFLIGHT


def test_selection_accounts_pending_boundaries_and_declines_missing_projection():
    engine = object.__new__(ServingEngine)
    engine.apc_inflight_prefix_policy = {
        "enabled": True,
        "min_shared_tokens": 64,
        "max_wait_ms": 300_000,
    }
    engine.apc_inflight_route = "hybrid"
    engine.counts = Counter()
    shared = list(range(1, 97))
    leader_tokens = tuple(shared + [101 + index for index in range(32)])
    leader = Job({"tokens": list(leader_tokens)})
    leader.uid = 1
    leader.apc_sequence_key = ("namespace", leader_tokens, "session")
    active = {1: leader}
    hit = SimpleNamespace(cached_tokens=0)

    first_tokens = tuple(shared + [201 + index for index in range(32)])
    first = Job({"tokens": list(first_tokens)})
    first.apc_sequence_key = ("namespace", first_tokens, "session")
    batch = SimpleNamespace(add_state_boundary=lambda *args: True)
    assert engine._select_inflight_prefix_leader(
        batch,
        active,
        first,
        first_tokens,
        hit,
        available_checkpoint_bytes=700,
        cache_projection=lambda position: position * 5,
    ) is leader
    assert leader.apc_inflight_planned_bytes == {96: 480}

    second_tokens = tuple(
        list(range(1, 81)) + [301 + index for index in range(48)]
    )
    second = Job({"tokens": list(second_tokens)})
    second.apc_sequence_key = ("namespace", second_tokens, "session")
    assert engine._select_inflight_prefix_leader(
        batch,
        active,
        second,
        second_tokens,
        hit,
        available_checkpoint_bytes=700,
        cache_projection=lambda position: position * 5,
    ) is None
    assert engine.counts["apc_inflight_prefix_waits_skipped_headroom"] == 1

    third = Job({"tokens": list(second_tokens)})
    third.apc_sequence_key = ("namespace", second_tokens, "session")
    assert engine._select_inflight_prefix_leader(
        batch,
        active,
        third,
        second_tokens,
        hit,
        available_checkpoint_bytes=10_000,
        cache_projection=None,
    ) is None
    assert engine.counts["apc_inflight_prefix_waits_skipped_projection"] == 1

    already_waited = Job({"tokens": list(first_tokens)})
    already_waited.apc_sequence_key = ("namespace", first_tokens, "session")
    already_waited.apc_sequence_waited = True
    assert engine._select_inflight_prefix_leader(
        batch,
        active,
        already_waited,
        first_tokens,
        hit,
        available_checkpoint_bytes=10_000,
        cache_projection=lambda position: position * 5,
    ) is None

    route_mismatch = Job({"tokens": list(first_tokens)})
    route_mismatch.apc_sequence_key = ("namespace", first_tokens, "session")
    engine.apc_inflight_route = "kv"
    assert engine._select_inflight_prefix_leader(
        batch,
        active,
        route_mismatch,
        first_tokens,
        hit,
        available_checkpoint_bytes=10_000,
        cache_projection=lambda position: position * 5,
    ) is None


def test_a_decoding_lane_releases_its_boundary_plan_from_the_ledger(monkeypatch):
    """Selection charges every lane's planned checkpoint bytes against the
    headroom.  A lane's interior/rolling plan (``state_boundary_planned_bytes``)
    was set at admission and never released, so a lane that had long finished
    its prompt -- every planned boundary captured, and resident, or declined --
    kept suppressing in-flight waits for its whole decode."""
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(
        serving, "runtime_identity", lambda: {"source_sha256": "cpu-inflight-ledger"}
    )
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    model, vocab = tiny_qwen38_mtp()
    engine = ServingEngine(
        "tiny",
        adapter_factory=_adapter(model, vocab),
        qualification_mode=True,
        mtp=False,
        max_lanes=2,
        tenant_scoped_cache=True,
        prefill_step=16,
        coalesce_window_ms=1,
        execution_policy={
            "apc_inflight_prefix_wait": {"enabled": True, "min_shared_tokens": 64},
            "apc_interior_checkpoints": {"count": 4, "min_stride": 16},
        },
    )
    assert engine.ready.wait(60), engine.error
    try:
        engine._inject_device_fault = lambda active: time.sleep(0.005)
        decoding = engine.submit(
            {"tokens": conversation(vocab, 5, tail_tokens=200), "max_tokens": 200}
        )
        deadline = time.monotonic() + 30
        while decoding.completion_tokens < 4 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert decoding.completion_tokens >= 4
        # The lane's prompt planned (and captured) interior checkpoints.
        assert engine.counts["apc_interior_checkpoints_captured"] >= 2
        selector = object.__new__(ServingEngine)
        selector.apc_inflight_prefix_policy = {
            "enabled": True,
            "min_shared_tokens": 64,
            "max_wait_ms": 300_000,
        }
        selector.apc_inflight_route = "hybrid"
        selector.counts = Counter()
        shared = list(range(1, 97))
        leader_tokens = tuple(shared + [101 + index for index in range(32)])
        leader = Job({"tokens": list(leader_tokens)})
        leader.uid = decoding.uid + 1
        leader.apc_sequence_key = ("namespace", leader_tokens, "session")
        follower_tokens = tuple(shared + [201 + index for index in range(32)])
        follower = Job({"tokens": list(follower_tokens)})
        follower.apc_sequence_key = ("namespace", follower_tokens, "session")
        selected = selector._select_inflight_prefix_leader(
            SimpleNamespace(add_state_boundary=lambda *args: True),
            {decoding.uid: decoding, leader.uid: leader},
            follower,
            follower_tokens,
            SimpleNamespace(cached_tokens=0),
            # Exactly the new boundary's projection: nothing else is pending.
            available_checkpoint_bytes=96 * 5,
            cache_projection=lambda position: position * 5,
        )
        assert selected is leader, dict(selector.counts)
        assert selector.counts["apc_inflight_prefix_waits_skipped_headroom"] == 0
        decoding.cancelled.set()
        collect(decoding)
    finally:
        engine.close()


def test_kv_only_route_fails_closed_until_prompt_boundary_release_exists():
    from mlx2.runtime.apc_v2 import inspect_apc_capabilities

    engine = object.__new__(ServingEngine)
    engine.mtp = False
    engine.approximate_kv_policy = SimpleNamespace(enabled=False)
    adapter = SimpleNamespace(model=SimpleNamespace(layers=(object(),)))
    with pytest.raises(
        ValueError, match="requires an exact checkpointed-hybrid cache route"
    ):
        engine._select_inflight_route(
            adapter,
            external_draft=False,
            prompt_lookup=False,
            inspect=inspect_apc_capabilities,
        )


@pytest.mark.parametrize(
    "value",
    [
        {"enabled": 1},
        {"min_shared_tokens": 15},
        {"min_shared_tokens": True},
        {"max_wait_ms": 0},
        {"max_wait_ms": 300_001},
        {"max_wait_ms": True},
        {"unknown": 1},
        "enabled",
    ],
)
def test_policy_rejects_ambiguous_or_unbounded_values(value):
    with pytest.raises(ValueError):
        apc_inflight_prefix_policy(value)
