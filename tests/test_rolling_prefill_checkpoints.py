"""Item 7: rolling disposable prefill checkpoints and the P1 state-boundary plan.

CPU only.  Unit layers (plan/budget, APCv2 roles and retirement, BatchGenerator
capture routing) plus end-to-end ServingEngine runs on tiny random-weight
hybrid (checkpointed) and KV-only models:

* a cancelled long prefill leaves its newest rolling checkpoint, and a retry
  of the same prompt resumes there with output identical to a cold run;
* a successful prefill retires its rolling checkpoints (no ``prefill_rolling``
  entry survives once the committed prompt boundary is published);
* rolling entries are evicted before interior checkpoints;
* a peer lease defers retirement until the lease is released;
* default-off leaves settings and receipts unchanged.
"""

import threading

import mlx.core as mx
import pytest

from mlx2 import memory, serving
from mlx2.runtime import os_memory
from mlx2.runtime.apc_v2 import APCKey, APCv2
from mlx2.runtime.generate import BatchGenerator
from mlx2.runtime.models.cache import ArraysCache, KVCache
from mlx2.runtime.state_boundaries import (
    RETENTION_ROLE,
    BoundaryPurpose,
    StateBoundary,
    budget_state_boundaries,
    plan_state_boundaries,
)
from mlx2.serving import (
    ServingEngine,
    apc_rolling_checkpoint_policy,
    budget_interior_checkpoint_positions,
)

from test_apc_hits_hybrid_gdn_self_mtp import make_adapter, tiny_qwen38_mtp

R, I, J = BoundaryPurpose.ROLLING, BoundaryPurpose.INTERIOR, BoundaryPurpose.JUNCTION


# --------------------------------------------------------------------- P1 plan


def test_plan_merges_sources_dedups_by_max_purpose_and_bounds_positions():
    bounds = plan_state_boundaries(
        prompt_tokens=10_001,
        cached_tokens=1000,
        interior=(512, 1024, 2048, 4096, 8192),
        junction=4096,
        rolling_interval=4096,
    )
    assert bounds == (
        StateBoundary(1024, I),
        StateBoundary(2048, I),
        StateBoundary(4096, J),
        StateBoundary(8192, I),
    )
    # Rolling positions are absolute multiples; P - 1 and the cached prefix
    # are excluded.
    assert plan_state_boundaries(
        prompt_tokens=8193, cached_tokens=4096, rolling_interval=2048
    ) == (StateBoundary(6144, R),)
    assert plan_state_boundaries(
        prompt_tokens=10_000, cached_tokens=0, rolling_interval=0
    ) == ()
    assert plan_state_boundaries(
        prompt_tokens=100, cached_tokens=0, junction=99
    ) == ()
    assert RETENTION_ROLE[R] == "prefill_rolling"
    assert RETENTION_ROLE[J] == "junction"
    with pytest.raises(TypeError):
        plan_state_boundaries(prompt_tokens=True, cached_tokens=0)
    with pytest.raises(ValueError):
        plan_state_boundaries(prompt_tokens=10, cached_tokens=0, rolling_interval=-1)


def test_budget_matches_interior_budget_and_ranks_junction_then_rolling():
    def project(position):
        return position * 10

    interior = (8, 16, 32)
    for available in (0, 79, 80, 319, 320, 400, 480, 560, 10_000):
        expected, charged = budget_interior_checkpoint_positions(
            interior, available_bytes=available, cache_projection=project
        )
        got, got_charged = budget_state_boundaries(
            tuple(StateBoundary(p, I) for p in interior),
            available_bytes=available,
            cache_projection=project,
        )
        assert tuple(b.position for b in got) == expected
        assert got_charged == charged
    bounds = plan_state_boundaries(
        prompt_tokens=100,
        cached_tokens=0,
        interior=(32,),
        junction=40,
        rolling_interval=16,
    )
    # Junction (400) first, then interior 32 (320), then rolling charged as
    # one slot: 16 and 48 fit (slot 480), 64 does not.
    got, charged = budget_state_boundaries(
        bounds, available_bytes=1200, cache_projection=project
    )
    assert [(b.position, b.purpose) for b in got] == [
        (16, R), (32, I), (40, J), (48, R),
    ]
    assert charged == 400 + 320 + 480
    assert budget_state_boundaries(
        bounds, available_bytes=10**9, cache_projection=None
    ) == ((), 0)


def test_rolling_policy_is_default_off_and_strict():
    assert apc_rolling_checkpoint_policy(None) == {"interval_tokens": 0}
    assert apc_rolling_checkpoint_policy({"interval_tokens": 4096}) == {
        "interval_tokens": 4096
    }
    for invalid in ([], {"interval_tokens": True}, {"interval_tokens": -1},
                    {"interval_tokens": 1.5}, {"interval_tokens": 8}, {"other": 1}):
        with pytest.raises(ValueError):
            apc_rolling_checkpoint_policy(invalid)


# ----------------------------------------------------------------- APCv2 roles


def _hybrid(length):
    recurrent = ArraysCache(1)
    recurrent[0] = mx.ones((1, 4), dtype=mx.float32)
    recurrent.lengths = mx.array([length], dtype=mx.int32)
    recurrent._host_lengths = (recurrent.lengths, [length])
    kv = KVCache()
    values = mx.arange(length, dtype=mx.float32).reshape(1, 1, length, 1)
    kv.update_and_fetch(values, values)
    mx.eval(recurrent.state, kv.state)
    return [recurrent, kv]


def test_rolling_entries_are_evicted_before_interior_checkpoints():
    # Interior checkpoints now hold their own resident-entry pool
    # (``max_interior_entries``), so ordinary-pool pressure cannot touch them
    # at all.  Rolling entries share the ordinary pool and, at rank -1, are
    # still the first thing that pool gives up.
    apc = APCv2(max_size=1, layout_name="rolling-order-v1")
    key = APCKey("rolling-order")
    apc.store(key, [1, 2], _hybrid(2), retention_role="interior_checkpoint")
    apc.store(key, [1, 2, 3], _hybrid(3), retention_role="prefill_rolling")
    apc.store(key, [1, 2, 3, 4], _hybrid(4))
    roles = {
        tuple(tokens): entry._apc_retention_role
        for _key, tokens, entry in apc._entry_records_locked()
    }
    assert roles == {(1, 2): "interior_checkpoint", (1, 2, 3, 4): "default"}
    apc.clear(release_memory=False)


def test_rolling_publication_never_downgrades_and_counts_hits():
    apc = APCv2(max_size=4, layout_name="rolling-role-v1")
    key = APCKey("rolling-role")
    apc.store(key, [5, 6, 7], _hybrid(3))
    apc.store(key, [5, 6, 7], _hybrid(3), retention_role="prefill_rolling")
    assert apc.retire(key, [5, 6, 7], role="prefill_rolling") is True
    hit = apc.lookup(key, [5, 6, 7, 8])
    assert hit.hit and hit.retention_role == "default"
    hit.cache.close()
    apc.store(key, [5, 6], _hybrid(2), retention_role="prefill_rolling")
    hit = apc.lookup(key, [5, 6, 9])
    assert hit.hit and hit.retention_role == "prefill_rolling"
    hit.cache.close()
    assert apc.apc_stats["lifetime"]["rolling_hits"] == 1
    assert apc.apc_stats["lifetime"]["junction_hits"] == 0
    apc.clear(release_memory=False)


def test_peer_lease_defers_retirement_until_release():
    apc = APCv2(max_size=4, layout_name="rolling-lease-v1")
    key = APCKey("rolling-lease")
    tokens = [3, 4, 5, 6]
    apc.store(key, tokens, _hybrid(4), retention_role="prefill_rolling")
    lease = apc.lookup(key, tokens + [9])
    assert lease.hit and lease.cached_tokens == 4
    assert apc.retire(key, tokens, role="prefill_rolling") is False
    probe = apc.lookup(key, tokens + [10])
    assert probe.hit, "a deferred retirement keeps the leased entry usable"
    probe.cache.close()
    lease.cache.close()
    # The first APCv2 operation after the last lease is released drops it.
    assert apc.lookup(key, tokens + [11]).hit is False
    assert apc.retire(key, tokens, role="prefill_rolling") is True
    apc.clear(release_memory=False)


def test_republication_cancels_a_pending_retirement():
    apc = APCv2(max_size=4, layout_name="rolling-republish-v1")
    key = APCKey("rolling-republish")
    tokens = [7, 8, 9]
    apc.store(key, tokens, _hybrid(3), retention_role="prefill_rolling")
    lease = apc.lookup(key, tokens + [1])
    assert apc.retire(key, tokens, role="prefill_rolling") is False
    lease.cache.close()
    apc.store(key, tokens, _hybrid(3), retention_role="prefill_rolling")
    hit = apc.lookup(key, tokens + [2])
    assert hit.hit and hit.retention_role == "prefill_rolling"
    hit.cache.close()
    apc.clear(release_memory=False)


# ----------------------------------------------------- BatchGenerator captures


@pytest.fixture(scope="module")
def hybrid():
    return tiny_qwen38_mtp()


def _drive(gen, uid):
    """Run one lane to completion; return (tokens, drained, interiors, stats)."""
    drained, interiors, tokens = [], [], []
    for _ in range(400):
        prompts, responses = gen.next()
        drained.extend(gen.drain_state_checkpoints())
        for response in prompts:
            if response.end_of_prompt:
                interiors.extend(gen.pop_interior_checkpoints(response.uid))
        for response in responses:
            if response.uid == uid:
                tokens.append(int(response.token))
                if response.finish_reason is not None:
                    return tokens, drained, interiors
    raise AssertionError("lane did not finish")


def test_plain_route_routes_rolling_to_drain_and_interior_to_pop(hybrid):
    model, vocab = hybrid
    prompt = [(7 * i + 3) % (vocab - 2) + 1 for i in range(50)]
    plan = (
        StateBoundary(16, R),
        StateBoundary(24, I),
        StateBoundary(32, R),
        StateBoundary(48, R),
    )

    def run(boundaries, pressure=None):
        gen = BatchGenerator(
            model, prefill_step_size=8, memory_pressure_level=pressure
        )
        try:
            uid = gen.insert(
                [prompt], max_tokens=[4],
                state_boundaries=None if boundaries is None else [boundaries],
            )[0]
            return (*_drive(gen, uid), dict(gen.scheduler_stats))
        finally:
            gen.close()

    cold_tokens, cold_drained, cold_interiors, _ = run(None)
    assert cold_drained == [] and cold_interiors == []
    tokens, drained, interiors, stats = run(plan)
    assert tokens == cold_tokens
    assert [(uid, c["covered_tokens"], c["purpose"]) for uid, c in drained] == [
        (0, 16, R), (0, 32, R), (0, 48, R),
    ]
    assert all(len(c["tokens"]) == c["covered_tokens"] for _, c in drained)
    assert [(c["covered_tokens"], c["purpose"]) for c in interiors] == [(24, I)]
    assert stats["apc_rolling_checkpoints_captured"] == 3
    assert stats["apc_interior_checkpoints_captured"] == 1

    # Host pressure at WARN drops only the disposable rolling captures.
    tokens, drained, interiors, stats = run(
        plan, pressure=lambda: os_memory.PressureLevel.WARN
    )
    assert tokens == cold_tokens
    assert drained == []
    assert [c["covered_tokens"] for c in interiors] == [24]
    assert stats["apc_rolling_checkpoints_skipped_pressure"] == 3


def test_removed_lane_drops_undrained_snapshots(hybrid):
    model, vocab = hybrid
    prompt = [(3 * i + 1) % (vocab - 2) + 1 for i in range(40)]
    gen = BatchGenerator(model, prefill_step_size=8)
    try:
        uid = gen.insert(
            [prompt], max_tokens=[2], state_boundaries=[(StateBoundary(8, R),)]
        )[0]
        for _ in range(4):
            gen.next()
        assert gen._state_checkpoints
        gen.remove([uid])
        assert gen.drain_state_checkpoints() == []
    finally:
        gen.close()


# ----------------------------------------------------------- ServingEngine e2e


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)


class _Budget:
    """Linear cache projection so admission can budget checkpoints."""

    def project(self, tokens):
        return int(tokens) * 1024

    def as_dict(self):
        return {"bytes_per_token": 1024}


def _adapter(model, vocab):
    base = make_adapter(model, vocab)

    class Adapter(base):
        def __init__(self, path, execution_policy=None):
            super().__init__(path)

        def cache_budget(self, mtp):
            return _Budget()

    return Adapter


def _engine(model, vocab, *, mtp, interval=None, **kw):
    policy = (
        {"apc_rolling_checkpoints": {"interval_tokens": interval}}
        if interval is not None
        else None
    )
    engine = ServingEngine(
        "tiny", adapter_factory=_adapter(model, vocab), qualification_mode=True,
        mtp=mtp, max_lanes=1, prefill_step=16, execution_policy=policy, **kw,
    )
    assert engine.ready.wait(60), engine.error
    return engine


def _collect(job):
    text = ""
    while True:
        event = job.events.get(timeout=120)
        if "error" in event:
            return None, event
        if "delta" in event:
            text += event["delta"].get("content", "")
        if "finish_reason" in event:
            return [int(t) for t in text.split()], event.get("receipt") or {}


def _run(engine, tokens, max_tokens=6):
    job = engine.submit(
        {"tokens": list(tokens), "max_tokens": max_tokens, "temperature": 0}
    )
    return (*_collect(job), job)


def _roles(engine):
    with engine.apc._apc_lock:
        return sorted(
            (len(tokens), entry._apc_retention_role)
            for _key, tokens, entry in engine.apc._entry_records_locked()
        )


def _cancel_mid_prefill(engine, prompt, should_block):
    """Cancel a request while the worker is blocked entering a scheduler step.

    ``should_block(engine, calls)`` picks the ``BatchGenerator.next`` call to
    hold; the class-level wrapper is restored before returning.
    """
    reached, release = threading.Event(), threading.Event()
    cls = BatchGenerator
    original = cls.next
    state = {"engine": None, "calls": 0}

    def gated(self, *args, **kwargs):
        if state["engine"] is not None and not release.is_set():
            state["calls"] += 1
            if should_block(state["engine"], state["calls"]):
                reached.set()
                release.wait(30)
        return original(self, *args, **kwargs)

    cls.next = gated
    try:
        job = engine.submit(
            {"tokens": list(prompt), "max_tokens": 4, "temperature": 0}
        )
        state["engine"] = engine
        assert reached.wait(60), "prefill never reached the gate"
        job.cancelled.set()
        release.set()
        tokens, event = _collect(job)
    finally:
        release.set()
        cls.next = original
    assert tokens is None and event["error"] == "cancelled"


@pytest.mark.parametrize("mtp", [False, True], ids=["ordinary", "self_mtp"])
def test_cancelled_hybrid_prefill_leaves_rolling_checkpoint_a_retry_hits(
    host, hybrid, mtp
):
    model, vocab = hybrid
    prompt = [(7 * i + 3) % (vocab - 2) + 1 for i in range(200)]
    engine = _engine(model, vocab, mtp=mtp, interval=64)
    try:
        # Rolling points at 64, 128, 192; cancel after two were published.
        _cancel_mid_prefill(
            engine, prompt,
            lambda engine, _calls: (
                engine.counts["apc_rolling_checkpoints_published"] >= 2
            ),
        )
        counts = dict(engine.counts)
        assert counts["apc_rolling_checkpoints_published"] >= 2
        assert counts["apc_rolling_checkpoints_retired"] >= 1
        assert (128, "prefill_rolling") in _roles(engine)
        assert (64, "prefill_rolling") not in _roles(engine)
        warm, receipt, job = _run(engine, prompt)
        assert job.cached_tokens >= 128
        assert receipt["cache_checkpoint_role"] == "prefill_rolling"
        assert receipt["state_boundaries"]["planned"]["rolling"] >= 0
        assert engine.snapshot["settings"]["apc_rolling_checkpoints"] == {"interval_tokens": 64}
        # Success retires every rolling checkpoint, including the restored one.
        assert not [r for r in _roles(engine) if r[1] == "prefill_rolling"]
    finally:
        engine.close()
    cold = _engine(model, vocab, mtp=mtp)
    try:
        cold_tokens, cold_receipt, _ = _run(cold, prompt)
        assert "state_boundaries" not in cold_receipt
        assert "apc_rolling_checkpoints" not in cold.snapshot["settings"]
    finally:
        cold.close()
    assert warm == cold_tokens


def test_successful_prefill_leaves_no_rolling_entry(host, hybrid):
    model, vocab = hybrid
    prompt = [(5 * i + 2) % (vocab - 2) + 1 for i in range(150)]
    engine = _engine(model, vocab, mtp=False, interval=32)
    try:
        tokens, receipt, _job = _run(engine, prompt)
        assert tokens
        counts = dict(engine.counts)
        assert counts["apc_rolling_checkpoints_published"] == 4  # 32..128
        assert counts["apc_rolling_checkpoints_retired"] == 4
        assert receipt["state_boundaries"]["published"] == {"rolling": 4}
        roles = _roles(engine)
        assert (149, "committed_prompt_boundary") in roles
        assert not [role for role in roles if role[1] == "prefill_rolling"]
    finally:
        engine.close()


def test_kv_only_cancel_publishes_partial_prefill(host):
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel

    args = TextModelArgs(
        model_type="qwen3_5", hidden_size=64, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, vocab_size=128, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=3, full_attention_interval=1,
        partial_rotary_factor=0.5, rope_parameters=None, max_position_embeddings=512,
    )
    mx.random.seed(5)
    model = TextModel(args)
    model.eval()
    mx.eval(model.parameters())
    prompt = [(11 * i + 5) % 126 + 1 for i in range(200)]
    engine = _engine(model, 128, mtp=False, interval=32)
    try:
        assert engine.apc_rolling_route == "kv"
        # Hold the sixth scheduler step: several 16-token chunks are done.
        _cancel_mid_prefill(engine, prompt, lambda _engine, calls: calls == 6)
        assert engine.counts["apc_rolling_checkpoints_cancel_published"] == 1
        warm, receipt, warm_job = _run(engine, prompt)
        assert warm_job.cached_tokens >= 32
        assert receipt["cache_checkpoint_role"] == "prefill_rolling"
        # Longer trimmable publications supersede the partial prefix.
        assert not [role for role in _roles(engine) if role[1] == "prefill_rolling"]
    finally:
        engine.close()
    cold = _engine(model, 128, mtp=False)
    try:
        assert _run(cold, prompt)[0] == warm
    finally:
        cold.close()


def test_rolling_policy_fails_closed_on_prompt_lookup(host, hybrid):
    model, vocab = hybrid
    engine = ServingEngine(
        "tiny", adapter_factory=_adapter(model, vocab), qualification_mode=True,
        mtp=False, prompt_lookup=True, max_lanes=1, prefill_step=16,
        execution_policy={"apc_rolling_checkpoints": {"interval_tokens": 64}},
    )
    try:
        engine.thread.join(30)
        assert not engine.ready.is_set()
        assert "APCv2 rolling checkpoints" in engine.error
        assert "prompt lookup" in engine.error
    finally:
        engine.close()
