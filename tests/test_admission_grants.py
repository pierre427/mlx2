"""Lanes attached in one pass must not spend the same measured headroom.

Series 2026-09-24, Qwen3.8-27B-MLX-4bit on a 36 GiB M3 Pro (8 GiB APC cache,
--ordinary): the qualifier's ``shared_warm_requests`` cohort -- two warm
32,547-token prompts -- was admitted, and the first width-2 decode step
failed with ``kIOGPUCommandBufferCallbackErrorOutOfMemory``.  Admission
charged each lane 3.45 GiB above the 4.00 GiB reserve and tested both against
one headroom reading, because an attached lane allocates nothing until the
scheduler steps it (the warm hit is a descriptor-only COW branch).
"""

from types import SimpleNamespace as NS

import mlx.core as mx
import pytest

from mlx2 import memory, serving
from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController as C
from route_harness import collect, make_engine, patch_host, tiny_qwen38_mtp

GIB = 1 << 30
M3 = dict(host_memory_gib=36.0, advisory_gib=28.08)


def _qwen38_m3_controller():
    from mlx2.adapters.qwen38_memory import Qwen38CacheBudget

    budget = Qwen38CacheBudget(
        attention_layers=16, recurrent_layers=48, mtp_layers=0, kv_heads=4,
        head_dim=256, recurrent_heads=48, recurrent_key_heads=16,
        recurrent_key_dim=128, recurrent_value_dim=128, conv_kernel=4,
    )
    return C(**M3, cache_estimator=budget.project,
             transient_gib_per_lane=budget.transient_gib_per_lane,
             saturation_lane_cap=4, verification_row_cap=4)


def _warm_32k_hit():
    """The primed checkpoint: 48 GDN layers of fixed state, 16 bf16 K/V layers
    allocated to 32,768 slots holding 32,546 cached tokens."""
    recurrent = NS(nbytes=1_847_328_768 // 6 // 48)
    kv = NS(nbytes=32768 * 2 * 4 * 256 * 2, keys=NS(shape=(1, 4, 32768, 256)))
    return NS(cache=[recurrent] * 48 + [kv] * 16, cached_tokens=32546,
              remaining_tokens=[1], sidecar=None)


def test_each_cohort_lane_is_charged_its_own_copy_and_state():
    controller = _qwen38_m3_controller()
    context = 32547 + 64
    copy = serving.warm_cache_copy_gib(
        _warm_32k_hit(), context_tokens=context, prefill_step=2048, mtp=False
    )
    # The lane's own copy: all K/V plus the recurrent state, once each.
    assert copy == pytest.approx((1_847_328_768 / 6 + 16 * 32768 * 4096) / GIB)
    lane = serving.lane_admission_required_gib(
        controller, context_tokens=context, draft_depth=0, cache_gib=copy
    ) - controller.hard_reserve_gib
    assert lane == pytest.approx(3.45, abs=0.01)

    # One lane fits a 7.45 GiB reading; the second must see the first's grant.
    headroom = int((controller.hard_reserve_gib + lane + 0.01) * GIB)
    grants = []
    for member in range(2):
        granted = serving.unmaterialized_lane_bytes(grants)
        admitted, _floor, required = serving.admit_lane_headroom(
            controller, context_tokens=context, draft_depth=0, cache_gib=copy,
            headroom=lambda: headroom - granted, reclaim=lambda: False,
            evict=lambda: False, evictable=lambda: 0,
        )
        assert admitted is (member == 0)
        grants.append(NS(admission_reserved_gib=required - controller.hard_reserve_gib))


def test_atomic_cohort_is_refused_not_admitted_into_one_reading(monkeypatch):
    """End to end: a cohort whose lanes fit one at a time answers 429."""
    patch_host(monkeypatch)
    monkeypatch.setattr(memory, "host_memory_gib", lambda: M3["host_memory_gib"])
    monkeypatch.setattr(memory, "metal_advisory_gib", lambda: M3["advisory_gib"])
    reserve = sum(C.host_scaled_reserves(**M3))
    lane = 3.45
    monkeypatch.setattr(
        serving, "lane_admission_required_gib",
        lambda controller, **_kw: controller.hard_reserve_gib + lane,
    )
    # A fixed reading: nothing a queued lane will allocate is visible yet.
    monkeypatch.setattr(
        memory, "execution_headroom", lambda **_kw: int((reserve + 1.5 * lane) * GIB)
    )
    model, vocab = tiny_qwen38_mtp()
    engine = make_engine(model, vocab, mtp=False, max_lanes=4)
    try:
        prompt = [(5 * i + 3) % (vocab - 2) + 1 for i in range(40)]
        solo = collect(engine.submit({"tokens": prompt, "max_tokens": 4,
                                      "temperature": 0}))
        assert "error" not in solo, solo
        cohort = {"id": "m3-shared-b2", "size": 2}
        jobs = [engine.submit({"tokens": prompt, "max_tokens": 4, "temperature": 0,
                               "batch_cohort": dict(cohort)}) for _ in range(2)]
        results = [collect(job) for job in jobs]
        assert all(r.get("status") == 429 for r in results), results
        assert engine.counts["memory_admission_deferred_behind_grants"] >= 1
        # Serialised requests still run: each grant ends at its first token.
        for _ in range(2):
            again = collect(engine.submit({"tokens": prompt, "max_tokens": 4,
                                           "temperature": 0}))
            assert "error" not in again, again
    finally:
        engine.close()
