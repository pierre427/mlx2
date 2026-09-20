"""APCv2 junction snapshots for checkpointed hybrid (recurrent + attention) models.

A hybrid prompt cache cannot branch inside a stored longer path: recurrent
state does not trim.  When request B shares a prefix with a stored path from
request A but diverges, main reported ``untrimmable_branch`` and re-prefilled
from zero, and so did every later request C diverging at the same point.
With ``apc_junction_checkpoints`` B snapshots its exact state at the shared
prefix length during prefill, so C resumes there.

The end-to-end tests drive the real ``ServingEngine`` on CPU with a tiny
random-weight Qwen3.8 hybrid (GDN + full attention) and its real cache budget.
"""

import random
from types import SimpleNamespace


import mlx.core as mx
import pytest

from mlx2 import memory, qualification, serving
from mlx2.prometheus import (
    _ENGINE_EVENTS,
    _OPTIONAL_ENGINE_EVENTS,
    PrometheusBuilder,
    _add_apcv2,
)
from mlx2.runtime import os_memory
from mlx2.runtime.apc_v2 import APCKey, APCv2
from mlx2.runtime.generate import BatchGenerator
from mlx2.runtime.state_boundaries import (
    RETENTION_ROLE,
    BoundaryPurpose,
    StateBoundary,
    budget_state_boundaries,
    plan_state_boundaries,
)
from mlx2.serving import ServingEngine, budget_interior_checkpoint_positions

from test_apc_hits_hybrid_gdn_self_mtp import make_adapter, tiny_qwen38_mtp

J = BoundaryPurpose.JUNCTION
I = BoundaryPurpose.INTERIOR
R = BoundaryPurpose.ROLLING

QWEN38_TINY_CONFIG = dict(
    num_hidden_layers=4, full_attention_interval=4, num_key_value_heads=1,
    head_dim=32, linear_num_value_heads=4, linear_num_key_heads=2,
    linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=3,
    mtp_num_hidden_layers=1,
)


# --------------------------------------------------------------------------
# P1 planning and budgeting (stubbed here; owned by item 7)
# --------------------------------------------------------------------------


def test_plan_keeps_strongest_purpose_and_drops_out_of_range_junctions():
    bounds = plan_state_boundaries(
        prompt_tokens=100, cached_tokens=10, interior=(8, 16, 32, 64),
        junction=32, rolling_interval=32,
    )
    assert bounds == (
        StateBoundary(16, I), StateBoundary(32, J), StateBoundary(64, I),
        StateBoundary(96, R),
    )
    # The junction must lie strictly inside (cached, P - 1): P - 1 is the
    # committed prompt boundary and <= cached is already restored.
    for junction in (10, 5, 99, 100, 150):
        assert plan_state_boundaries(
            prompt_tokens=100, cached_tokens=10, junction=junction
        ) == ()
    assert plan_state_boundaries(
        prompt_tokens=100, cached_tokens=10, junction=98
    ) == (StateBoundary(98, J),)
    with pytest.raises(TypeError):
        plan_state_boundaries(prompt_tokens=100, cached_tokens=0, junction=True)
    assert RETENTION_ROLE[J] == "junction"


def test_budget_prioritizes_junction_and_matches_interior_budget():
    bounds = plan_state_boundaries(
        prompt_tokens=200, cached_tokens=0, interior=(16, 32, 64, 128),
        junction=40,
    )
    # Only one snapshot fits: the junction wins over the deeper interiors.
    kept, charged = budget_state_boundaries(
        bounds, available_bytes=150, cache_projection=lambda p: 100 + p
    )
    assert kept == (StateBoundary(40, J),) and charged == 140
    # The junction alone does not fit; interiors still use what is left.
    kept, _ = budget_state_boundaries(
        bounds, available_bytes=150,
        cache_projection=lambda p: 1000 if p == 40 else p,
    )
    assert kept == (StateBoundary(128, I),)
    assert budget_state_boundaries(
        bounds, available_bytes=10**9, cache_projection=None
    ) == ((), 0)
    # Interior-only input selects exactly what the serving budget selects.
    rng = random.Random(6)
    for _ in range(200):
        interior = tuple(sorted(rng.sample(range(1, 400), rng.randint(0, 6))))
        available = rng.randint(0, 3000)
        projection = lambda p: 50 + 3 * p  # noqa: E731
        legacy = budget_interior_checkpoint_positions(
            interior, available_bytes=available, cache_projection=projection
        )
        kept, charged = budget_state_boundaries(
            plan_state_boundaries(
                prompt_tokens=402, cached_tokens=0, interior=interior
            ),
            available_bytes=available, cache_projection=projection,
        )
        assert (tuple(b.position for b in kept), charged) == legacy


# --------------------------------------------------------------------------
# APCv2 lookup: branch_tokens
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qwen38():
    return tiny_qwen38_mtp()


def _prefill_cache(model, tokens):
    from mlx2.runtime.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    model(mx.array([tokens], mx.uint32), cache=cache)
    mx.eval([c.state for c in cache])
    return cache


def test_lookup_reports_branch_tokens_for_untrimmable_hybrid_paths(qwen38):
    model, vocab = qwen38
    shared = [(3 * i + 1) % (vocab - 2) + 1 for i in range(24)]
    a = shared + [5] * 8
    b = shared + [7] * 8
    apc = APCv2(max_size=8, layout_name="junction-lookup")
    apc.store("k", a, _prefill_cache(model, a))
    miss = apc.lookup("k", b)
    assert not miss.hit and miss.miss_reason == "untrimmable_branch"
    assert miss.branch_tokens == len(shared)
    # A shallower stored entry is a hit, but the shared prefix still extends
    # past it: main dropped this (the hit is shorter than common_prefix).
    apc.store("k", shared[:8], _prefill_cache(model, shared[:8]))
    shallow = apc.lookup("k", b)
    assert shallow.hit and shallow.cached_tokens == 8
    assert shallow.branch_tokens == len(shared)
    # A junction entry at the branch point is a hit with nothing further to
    # plan, and it is counted as a junction hit.
    apc.store(
        "k", shared, _prefill_cache(model, shared), retention_role="junction"
    )
    hit = apc.lookup("k", b)
    assert hit.hit and hit.cached_tokens == len(shared)
    assert hit.retention_role == "junction" and hit.branch_tokens == 0
    assert apc.apc_stats["lifetime"]["junction_hits"] == 1
    # Retained like an ordinary exact entry (between interior and boundary).
    assert APCv2._entry_retention_rank(apc._trie.get("k", shared)) == 1
    apc.clear(release_memory=False)


def test_lookup_on_trimmable_kv_cache_never_reports_a_junction():
    from mlx2.runtime.models.cache import KVCache

    def kv(n):
        cache = [KVCache()]
        cache[0].update_and_fetch(mx.zeros((1, 1, n, 4)), mx.zeros((1, 1, n, 4)))
        return cache

    apc = APCv2(max_size=8, layout_name="junction-kv")
    apc.store("k", list(range(1, 33)), kv(32))
    hit = apc.lookup("k", list(range(1, 17)) + [99] * 8)
    assert hit.hit and hit.cached_tokens == 16 and hit.branch_tokens == 0
    apc.clear(release_memory=False)


# --------------------------------------------------------------------------
# BatchGenerator: explicit boundaries plan without the interior lattice
# --------------------------------------------------------------------------


def test_generator_captures_tagged_junction_with_lattice_off(qwen38):
    model, vocab = qwen38
    prompt = [(5 * i + 2) % (vocab - 2) + 1 for i in range(40)]
    gen = BatchGenerator(model, prefill_step_size=16)
    try:
        uid = gen.insert(
            [prompt], max_tokens=[2],
            state_boundaries=[(StateBoundary(21, J), StateBoundary(30, I))],
        )[0]
        for _ in range(50):
            _prompts, responses = gen.next()
            if any(r.uid == uid and r.finish_reason for r in responses):
                break
        # P1 (item 7) routes every non-interior purpose through
        # drain_state_checkpoints so it can be published as soon as it is
        # captured; the interior lattice still leaves at the prompt end.
        drained = gen.drain_state_checkpoints()
        checkpoints = gen.pop_interior_checkpoints(uid)
    finally:
        gen.close()
    assert [(u, c["covered_tokens"], c["purpose"]) for u, c in drained] == [
        (uid, 21, J)
    ]
    assert [(c["covered_tokens"], c["purpose"]) for c in checkpoints] == [(30, I)]
    assert drained[0][1]["tokens"] == prompt[:21]
    assert checkpoints[0]["tokens"] == prompt[:30]
    assert gen.scheduler_stats.get("apc_junction_checkpoints_captured") == 1
    assert gen.scheduler_stats.get("apc_interior_checkpoints_captured") == 1


# --------------------------------------------------------------------------
# End to end through ServingEngine
# --------------------------------------------------------------------------


def _adapter(model, vocab, *, fail_at=None):
    from mlx2.adapters.qwen38_memory import Qwen38CacheBudget

    base = make_adapter(model, vocab)

    class Budget:
        def __init__(self, inner):
            self.inner = inner
            self.transient_gib_per_lane = inner.transient_gib_per_lane

        def project(self, tokens):
            if fail_at is not None and tokens == fail_at:
                return 1 << 60
            return self.inner.project(tokens)

        def as_dict(self):
            return self.inner.as_dict()

    class Adapter(base):
        def __init__(self, path, execution_policy=None):
            super().__init__(path)

        def cache_budget(self, *, mtp):
            return Budget(Qwen38CacheBudget.from_config(QWEN38_TINY_CONFIG, mtp=mtp))

    return Adapter


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)


def _engine(model, vocab, *, junction, fail_at=None, mtp=False):
    policy = {"apc_junction_checkpoints": True} if junction else None
    engine = ServingEngine(
        "tiny", adapter_factory=_adapter(model, vocab, fail_at=fail_at),
        qualification_mode=True, mtp=mtp, max_lanes=1, prefill_step=16,
        execution_policy=policy,
    )
    assert engine.ready.wait(60), engine.error
    return engine


def _run(engine, tokens, max_tokens=6):
    job = engine.submit(
        {"tokens": list(tokens), "max_tokens": max_tokens, "temperature": 0}
    )
    text = ""
    while True:
        event = job.events.get(timeout=120)
        if "error" in event:
            raise AssertionError(event)
        if "delta" in event:
            text += event["delta"].get("content", "")
        if "finish_reason" in event:
            return [int(t) for t in text.split()], event.get("receipt") or {}, job


def _conversation(vocab, shared_len=45, tail=24):
    shared = [(7 * i + 3) % (vocab - 2) + 1 for i in range(shared_len)]

    def turn(seed):
        return shared + [(seed * (i + 1) + 11) % (vocab - 2) + 1 for i in range(tail)]

    return shared, turn(5), turn(9), turn(13)


def _junction_entries(apc):
    return [
        len(tokens)
        for (_key, tokens, entry) in apc._entry_records_locked()
        if getattr(entry, "_apc_retention_role", None) == "junction"
    ]


@pytest.mark.parametrize("mtp", [False, True], ids=["ordinary", "self_mtp"])
def test_b_captures_junction_and_c_resumes_there_exactly(host, qwen38, mtp):
    model, vocab = qwen38
    shared, a, b, c = _conversation(vocab)
    engine = _engine(model, vocab, junction=True, mtp=mtp)
    try:
        _out_a, _ra, ja = _run(engine, a)
        _out_b, _rb, jb = _run(engine, b)
        # B diverged from A's stored path mid-chunk.  Recurrent state does
        # not trim, so B can only snap back to one of A's recorded prefill
        # chunk checkpoints, short of the divergence; it leaves a junction
        # exactly at the divergence.
        assert int(ja.cached_tokens or 0) == 0
        assert int(jb.cached_tokens or 0) < len(shared)
        assert _junction_entries(engine.apc) == [len(shared)]
        out_c, rc, jc = _run(engine, c)
        counts = dict(engine.counts)
        lifetime = engine.apc.apc_stats["lifetime"]
        settings = dict(engine.snapshot["settings"])
    finally:
        engine.close()
    assert int(jc.cached_tokens) == len(shared)
    assert rc.get("cache_checkpoint_role") == "junction"
    assert lifetime["junction_hits"] == 1
    assert counts["apc_junction_checkpoints_planned"] == 1
    # Capture is a BatchGenerator scheduler event under P1 (item 7) and is
    # asserted directly in the generator test above; the engine's periodic
    # scheduler snapshot lags a request, so it is not asserted here.
    assert counts["apc_junction_checkpoints_published"] == 1
    assert counts.get("apc_junction_checkpoints_degraded", 0) == 0
    assert settings["apc_junction_checkpoints"] is True

    # Exactness: resuming from the junction equals a cold prefill of C.
    cold = _engine(model, vocab, junction=False, mtp=mtp)
    try:
        out_cold, _r, jcold = _run(cold, c)
    finally:
        cold.close()
    assert int(jcold.cached_tokens or 0) == 0
    assert out_c == out_cold


def test_default_off_leaves_c_cold_and_settings_unchanged(host, qwen38):
    model, vocab = qwen38
    shared, a, b, c = _conversation(vocab)
    engine = _engine(model, vocab, junction=False)
    try:
        for prompt in (a, b):
            _run(engine, prompt)
        _out, rc, jc = _run(engine, c)
        counts = dict(engine.counts)
        settings = dict(engine.snapshot["settings"])
        assert _junction_entries(engine.apc) == []
    finally:
        engine.close()
    # Without a junction C also snaps back short of the shared prefix.
    assert int(jc.cached_tokens or 0) < len(shared) - 1
    assert "apc_junction_checkpoints" not in settings
    assert not any(key.startswith("apc_junction_") for key in counts)


def test_junction_degrades_when_its_snapshot_does_not_fit(host, qwen38):
    model, vocab = qwen38
    shared, a, b, c = _conversation(vocab)
    engine = _engine(model, vocab, junction=True, fail_at=len(shared))
    try:
        for prompt in (a, b):
            _run(engine, prompt)
        _out, _rc, jc = _run(engine, c)
        counts = dict(engine.counts)
        assert _junction_entries(engine.apc) == []
    finally:
        engine.close()
    assert int(jc.cached_tokens or 0) < len(shared)
    # B and C both see A's (and B's) longer path and both degrade.
    assert counts["apc_junction_checkpoints_degraded"] == 2
    assert counts.get("apc_junction_checkpoints_captured", 0) == 0


def test_write_suppressed_request_plans_no_junction(host, qwen38):
    model, vocab = qwen38
    _shared, a, b, _c = _conversation(vocab)
    engine = _engine(model, vocab, junction=True)
    try:
        _run(engine, a)
        job = engine.submit({
            "tokens": list(b), "max_tokens": 4, "temperature": 0,
            "skip_writing_prefix_cache": True,
        })
        while "finish_reason" not in job.events.get(timeout=120):
            pass
        counts = dict(engine.counts)
        assert _junction_entries(engine.apc) == []
    finally:
        engine.close()
    assert counts.get("apc_junction_checkpoints_planned", 0) == 0


@pytest.mark.parametrize("value", [1, "yes", None, {"enabled": True}])
def test_policy_is_strictly_boolean(host, qwen38, value):
    model, vocab = qwen38
    with pytest.raises(ValueError, match="apc_junction_checkpoints"):
        ServingEngine(
            "tiny", adapter_factory=_adapter(model, vocab),
            qualification_mode=True, mtp=False,
            execution_policy={"apc_junction_checkpoints": value},
        )


def test_hit_counters_are_exported_live_not_from_the_lagging_snapshot():
    """A junction hit must be visible in /metrics without waiting a second.

    ``snapshot["apcv2"]`` is refreshed at most once a second and seeded
    all-zero at readiness, so exporting hit counts from it read 0 for short
    runs -- the shape behind the GPU gate reporting ``junction_hits: 0`` for
    a junction that demonstrably served.
    """
    from test_rolling_prefill_checkpoints import _hybrid

    apc = APCv2(max_size=8, layout_name="junction-live-v1")
    key = APCKey("junction-live")
    apc.store(key, [1, 2, 3], _hybrid(3), retention_role="junction")
    hit = apc.lookup(key, [1, 2, 3, 4])
    assert hit.hit and hit.retention_role == "junction"
    hit.cache.close()
    assert apc.lifetime_stats()["junction_hits"] == 1

    engine = SimpleNamespace(
        apc=apc,
        snapshot={"apcv2": {"lifetime": {"junction_hits": 0, "hits": 0}}},
    )
    builder = PrometheusBuilder()
    _add_apcv2(
        builder, engine.snapshot["apcv2"], engine.apc.lifetime_stats()
    )
    assert "mlx2_prefix_cache_junction_hits_total 1" in builder.render()
    apc.clear(release_memory=False)


def test_counters_are_exported_and_feature_check_is_selected():
    for event in (
        "planned", "degraded", "captured", "published",
        "skipped_write_suppressed", "skipped_approximate",
        "skipped_publish_failed",
    ):
        # Default-off: the mapping lives in the optional table so a server
        # that never enabled junctions emits no zero series.
        assert _OPTIONAL_ENGINE_EVENTS["apc_junction_checkpoints_" + event] == (
            "apcv2_junction", event
        )
        assert "apc_junction_checkpoints_" + event not in _ENGINE_EVENTS
    assert "feature_apc_junction_checkpoints" in qualification._route_feature_checks(
        {"apc_junction_checkpoints": True}
    )
    assert "feature_apc_junction_checkpoints" not in (
        qualification._route_feature_checks({})
    )
