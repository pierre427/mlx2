"""Independent same-prefix requests wait for the first APCv2 publication."""

import time

import mlx.core as mx
import pytest

from mlx2 import memory, serving
from mlx2.runtime import os_memory
from mlx2.serving import ServingEngine
from test_apc_hits_hybrid_gdn_self_mtp import make_adapter, tiny_qwen38_mtp


@pytest.fixture(params=[False, True], ids=["ordinary", "self-mtp"])
def engine(monkeypatch, request):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "cpu-sequence-test"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    model, vocab = tiny_qwen38_mtp()
    value = ServingEngine(
        "tiny", adapter_factory=make_adapter(model, vocab),
        qualification_mode=True, mtp=request.param, max_lanes=2, max_inflight=4,
        tenant_scoped_cache=True, prefill_step=16, coalesce_window_ms=50,
    )
    assert value.ready.wait(60), value.error
    try:
        yield value
    finally:
        value.close()


def collect(job):
    tokens = []
    while True:
        event = job.events.get(timeout=60)
        if "error" in event:
            return tokens, event
        if "delta" in event:
            tokens.extend(int(t) for t in event["delta"].get("content", "").split())
        if "finish_reason" in event:
            return tokens, event.get("receipt") or {}


def prompt(offset=0):
    return [(7 * i + 3 + offset) % 126 + 1 for i in range(128)]


def submit(engine, tokens, tenant="alice", max_tokens=8):
    return engine.submit(
        {"tokens": tokens, "max_tokens": max_tokens, "temperature": 0},
        tenant_id=tenant,
    )


def test_identical_requests_sequence_and_second_reuses_apcv2(engine):
    first = submit(engine, prompt())
    second = submit(engine, prompt())
    first_tokens, first_receipt = collect(first)
    second_tokens, second_receipt = collect(second)
    assert first_tokens == second_tokens
    assert [first.cached_tokens, second.cached_tokens] == [0, 127]
    assert [first.observed_width, second.observed_width] == [1, 1]
    assert first_receipt["apcv2_same_prefix_waited"] is False
    assert second_receipt["apcv2_same_prefix_waited"] is True
    assert engine.counts["apcv2_same_prefix_sequenced"] == 1


def test_different_prompt_and_tenant_do_not_sequence(engine):
    first = submit(engine, prompt(), tenant="alice")
    different = submit(engine, prompt(1), tenant="alice")
    collect(first)
    collect(different)
    assert different.apc_sequence_waited is False
    if not engine.mtp:
        assert different.observed_width == 2

    same_tokens_other_tenant = submit(engine, prompt(), tenant="bob")
    same_tokens_alice = submit(engine, prompt(), tenant="alice")
    collect(same_tokens_other_tenant)
    collect(same_tokens_alice)
    assert same_tokens_other_tenant.apc_sequence_waited is False
    assert same_tokens_other_tenant.cached_tokens == 0
    assert same_tokens_alice.cached_tokens == 127


def test_cancelled_follower_does_not_wait_for_leader_completion(engine):
    first = submit(engine, prompt(), max_tokens=64)
    second = submit(engine, prompt(), max_tokens=8)
    deadline = time.monotonic() + 10
    while not second.apc_sequence_waited and time.monotonic() < deadline:
        time.sleep(0.001)
    assert second.apc_sequence_waited
    second.cancelled.set()
    _, event = collect(second)
    assert event["error"] == "cancelled"
    collect(first)
    assert engine.status()["inflight"] == 0


def test_write_suppressed_leader_and_distinct_sessions_do_not_sequence(engine):
    body = {"tokens": prompt(), "max_tokens": 8, "temperature": 0}
    suppressed = engine.submit(
        {**body, "skip_writing_prefix_cache": True}, tenant_id="alice"
    )
    ordinary = engine.submit(body, tenant_id="alice")
    collect(suppressed)
    collect(ordinary)
    assert ordinary.apc_sequence_waited is False
    assert ordinary.cached_tokens == 0

    first = engine.submit({**body, "session_id": "one"}, tenant_id="bob")
    second = engine.submit({**body, "session_id": "two"}, tenant_id="bob")
    collect(first)
    collect(second)
    assert second.apc_sequence_waited is False


@pytest.mark.parametrize("defer_first", [False, True])
def test_a_memory_deferred_follower_reuses_the_checkpoint_it_waited_for(
    monkeypatch, defer_first
):
    """A follower deferred by memory admission before its leader was admitted
    kept its stale APCv2 miss: it waited out the leader's generation, then
    re-prefilled cold instead of reusing the leader's checkpoint."""
    import time

    from mlx2 import memory, serving
    from mlx2.runtime import os_memory
    from mlx2.serving import ServingEngine
    from test_apc_hits_hybrid_gdn_self_mtp import make_adapter, tiny_qwen38_mtp

    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "cpu-l1"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    monkeypatch.setattr(ServingEngine, "MEMORY_ADMISSION_RETRY", 0.05)
    real = serving.admit_lane_headroom
    calls = {"n": 0}

    def admit(*a, **kw):
        calls["n"] += 1
        if defer_first and calls["n"] == 1:
            return (False, False, 1.0)  # the follower's first reading is short
        return real(*a, **kw)

    monkeypatch.setattr(serving, "admit_lane_headroom", admit)
    model, vocab = tiny_qwen38_mtp()
    engine = ServingEngine(
        "tiny", adapter_factory=make_adapter(model, vocab),
        qualification_mode=True, mtp=False, max_lanes=2, max_inflight=4,
        tenant_scoped_cache=True, prefill_step=16, coalesce_window_ms=1,
    )
    assert engine.ready.wait(60), engine.error
    try:
        follower = submit(engine, prompt(), max_tokens=8)
        deadline = time.monotonic() + 10
        while defer_first and calls["n"] < 1 and time.monotonic() < deadline:
            time.sleep(0.001)
        leader = submit(engine, prompt(), max_tokens=160)
        collect(leader)
        _, receipt = collect(follower)
        if follower.apc_sequence_waited:
            # It waited for the leader's publication; it must reuse it.
            assert follower.cached_tokens == 127
    finally:
        engine.close()
