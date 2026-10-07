"""External-draft and prompt-lookup prefill beside decode is stall-bounded
(2026-10-06 rfix-sched).

GPU (Qwen3.8-27B, 12K prompt prefilling beside one decoding lane): after
the cost-model stall bound landed, ordinary and native-MTP neighbour gaps
fell to p95 464/441 ms while ``--external-draft`` (p95 693-767 ms) and
``--prompt-lookup`` (p95 703-710 ms) were unchanged.  Both generators have
their own prefill loop: one slice per poll at the configured (or
prompt-length autoscaled) step, with no stall bound at all -- 512 rows at
12K, 2048 at 32-64K and 8192 above 64K, the Flash-Next 9.5 s shape.

A fake clock charges each forward ``a + b * rows`` so the cost model sees
what the kernels would pay.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mlx2.runtime import external_speculative as E
from mlx2.runtime import pld as P
from mlx2.serving import decode_fairness_overrides, decode_time_fairness_policy

TARGET_MS = 100.0
PER_ROW = 0.0004  # a 512-row slice costs ~205 ms against a 100 ms target
FIXED = 0.005


class Clock:
    def __init__(self):
        self.t = 100.0
        self.prefill = []  # seconds of each prefill slice

    def __call__(self):
        return self.t


def _fairness(external_draft=False, prompt_lookup=False):
    policy = decode_time_fairness_policy(
        external_draft=external_draft, prompt_lookup=prompt_lookup,
        overrides=decode_fairness_overrides({"stall_target_ms": TARGET_MS}),
    )
    return policy


def _prompt(n, seed=0):
    return [((i * 7 + seed) % 29) + 1 for i in range(n)]


@pytest.fixture
def external_clock(monkeypatch):
    from test_external_dflash2_cpu import tiny

    clock = Clock()
    monkeypatch.setattr(
        E, "time", SimpleNamespace(perf_counter=clock, monotonic=clock, time=clock)
    )
    model, draft = tiny()
    real_body = type(model).prefill_body

    def prefill_body(self, tokens, cache, layers):
        out = real_body(self, tokens, cache, layers)
        seconds = FIXED + PER_ROW * tokens.shape[0] * tokens.shape[1]
        clock.t += seconds
        clock.prefill.append(seconds)
        return out

    real_round = E.ExternalDraftBatchGenerator._round_at_batch_route

    def round_(self, *args, **kwargs):
        out = real_round(self, *args, **kwargs)
        clock.t += 0.020
        return out

    monkeypatch.setattr(type(model), "prefill_body", prefill_body)
    monkeypatch.setattr(E.ExternalDraftBatchGenerator, "_round_at_batch_route", round_)
    return clock, model, draft


def _run_beside_decode(gen, clock, long_prompt, *, polls=400):
    gen.insert([_prompt(6)], max_tokens=[2000])
    for _ in range(3):
        gen.next()
    clock.prefill.clear()
    uid = gen.insert([long_prompt], max_tokens=[2])[0]
    for _ in range(polls):
        gen.next()
        if gen.lanes[uid].anchor is not None:
            return
    raise AssertionError("long prompt never finished prefill")


def test_external_draft_prefill_beside_decode_meets_stall_target(external_clock):
    clock, model, draft = external_clock
    gen = E.ExternalDraftBatchGenerator(
        model, draft_model=draft, binding="test", num_draft=2,
        prefill_step_size=512,
        decode_time_fairness=_fairness(external_draft=True),
    )
    try:
        _run_beside_decode(gen, clock, _prompt(1500, 3))
        assert clock.prefill, "no prefill slice observed"
        assert max(clock.prefill) <= 1.5 * TARGET_MS / 1000, clock.prefill
        stats = gen.scheduler_stats
        assert stats["decode_fairness_prefill_chunks"] >= len(clock.prefill) - 1
        assert stats["decode_fairness_cost_samples"] > 0
        # fair_share 0: the bound only, no debt deferral on this route.
        assert stats["decode_fairness_debt_deferrals"] == 0
    finally:
        gen.close()


def test_external_draft_direct_construction_is_unchanged(external_clock):
    clock, model, draft = external_clock
    gen = E.ExternalDraftBatchGenerator(
        model, draft_model=draft, binding="test", num_draft=2,
        prefill_step_size=512,
    )
    try:
        assert not any(k.startswith("decode_fairness_") for k in gen.scheduler_stats)
        _run_beside_decode(gen, clock, _prompt(1100, 5))
        # Unbounded as before: full 512-row slices beside decode.
        assert max(clock.prefill) == pytest.approx(FIXED + PER_ROW * 512)
    finally:
        gen.close()


@pytest.fixture
def pld_clock(monkeypatch):
    from test_decode_first_publish import tiny_model

    clock = Clock()
    monkeypatch.setattr(
        P, "time", SimpleNamespace(perf_counter=clock, monotonic=clock, time=clock)
    )
    from mlx2.runtime import adaptive_policy

    monkeypatch.setattr(adaptive_policy.time, "monotonic", clock)
    model = tiny_model()

    class Timed:
        """The model with each forward charged ``a + b * rows``."""

        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def __call__(self, inputs, *args, **kwargs):
            out = self._inner(inputs, *args, **kwargs)
            rows = int(inputs.shape[0] * inputs.shape[1])
            seconds = FIXED + PER_ROW * rows
            clock.t += seconds
            if rows > 16:
                clock.prefill.append(seconds)
            return out

    return clock, Timed(model)


def test_prompt_lookup_prefill_beside_decode_meets_stall_target(pld_clock):
    clock, model = pld_clock
    gen = P.PromptLookupBatchGenerator(
        model, completion_batch_size=4, prefill_step_size=512,
        prompt_lookup={}, stop_tokens=[],
        decode_time_fairness=_fairness(prompt_lookup=True),
    )
    try:
        _run_beside_decode(gen, clock, _prompt(1500, 3))
        assert clock.prefill
        assert max(clock.prefill) <= 1.5 * TARGET_MS / 1000, clock.prefill
        stats = gen.scheduler_stats
        assert stats["decode_fairness_prefill_chunks"] >= len(clock.prefill) - 1
        assert stats["decode_fairness_debt_deferrals"] == 0
        # Gap attribution: every gap is a measured slice plus one decode.
        forward = stats["decode_fairness_contended_forward_max_us"]
        assert forward == pytest.approx(1e6 * max(clock.prefill), abs=2)
        assert stats["decode_fairness_contended_gap_max_us"] >= forward
        assert stats["decode_fairness_contended_gap_other_max_us"] <= 2
    finally:
        gen.close()


def test_prompt_lookup_long_prompt_autoscale_step_is_bounded(pld_clock):
    """Above 64K the autoscaled step is 8192 rows (the Flash-Next shape).

    Scaled down: autoscale with a 512-row maximum, as a 12K prompt gets.
    """
    clock, model = pld_clock
    gen = P.PromptLookupBatchGenerator(
        model, completion_batch_size=4, prefill_step_size=512,
        prefill_step_autoscale=True, prompt_lookup={}, stop_tokens=[],
        decode_time_fairness=_fairness(prompt_lookup=True),
    )
    try:
        _run_beside_decode(gen, clock, _prompt(1300, 1))
        assert max(clock.prefill) <= 1.5 * TARGET_MS / 1000, clock.prefill
    finally:
        gen.close()


def test_uncontended_prompt_lookup_prefill_keeps_the_full_step(pld_clock):
    clock, model = pld_clock
    gen = P.PromptLookupBatchGenerator(
        model, completion_batch_size=4, prefill_step_size=512,
        prompt_lookup={}, stop_tokens=[],
        decode_time_fairness=_fairness(prompt_lookup=True),
    )
    try:
        gen.insert([_prompt(1100)], max_tokens=[2])
        gen.next()
        assert clock.prefill[0] == pytest.approx(FIXED + PER_ROW * 512)
        assert gen.scheduler_stats["decode_fairness_prefill_chunks"] == 0
    finally:
        gen.close()


def test_route_identity_records_the_bound_on_every_route():
    ordinary = decode_time_fairness_policy(external_draft=False, prompt_lookup=False)
    assert ordinary == {
        "enabled": True, "fair_share": 0.5, "stall_target_ms": 500.0,
        "estimator": "cost_model",
    }
    for flags in ({"external_draft": True, "prompt_lookup": False},
                  {"external_draft": False, "prompt_lookup": True}):
        policy = decode_time_fairness_policy(**flags)
        assert policy == {
            "enabled": True, "fair_share": 0.0, "stall_target_ms": 500.0,
            "estimator": "cost_model",
        }
        overridden = decode_time_fairness_policy(
            **flags,
            overrides=decode_fairness_overrides({"slice_floor": 1024}),
        )
        assert overridden["slice_floor"] == 1024
