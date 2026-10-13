"""The serving admission gate receives the route's depth and prefill chunk.

Sweep 2026-10-12 L8-5: 086421f7b wired two inputs into ordinary-route
admission -- ``prefill_step_autoscale`` into the ingress prefill estimate and
``native_mtp_arrival_depth`` into ``admit_lane_headroom``'s ``draft_depth``.
The helpers had unit tests, but reverting both call sites left every test
green.  These drive a real ServingEngine (tiny Qwen3.8 MTP model, CPU) and
record what the gate is handed.
"""

import pytest
from route_harness import collect, make_adapter, patch_host, tiny_qwen38_mtp

from mlx2 import serving
from mlx2.serving import ServingEngine

MIB = 1 << 20


class _Budget:
    """Linear cache projection plus a recording prefill-transient estimate."""

    def __init__(self, calls):
        self.calls = calls

    def project(self, tokens):
        return int(tokens) * 1024

    def as_dict(self):
        return {"bytes_per_token": 1024}

    def prefill_transient_bytes(self, context, rows):
        self.calls.append((int(context), int(rows)))
        return int(rows) * MIB


def _engine(monkeypatch, *, max_lanes=1, **kwargs):
    patch_host(monkeypatch)
    gate, estimates = [], []
    real = serving.admit_lane_headroom

    def spy(controller, **options):
        gate.append(options)
        return real(controller, **options)

    monkeypatch.setattr(serving, "admit_lane_headroom", spy)
    model, vocab = tiny_qwen38_mtp()

    class Mixin:
        max_context = 4096

        def cache_budget(self, mtp):
            return _Budget(estimates)

    engine = ServingEngine(
        "tiny",
        adapter_factory=make_adapter(model, vocab, adapter_mixin=Mixin),
        qualification_mode=True, mtp=True, max_lanes=max_lanes,
        max_inflight=32, **kwargs,
    )
    assert engine.ready.wait(120), engine.error
    return engine, vocab, gate, estimates


def _prompt(vocab, length, salt=3):
    return [(7 * i + salt) % (vocab - 2) + 1 for i in range(length)]


def test_idle_static_width_cohort_is_priced_at_its_ordinary_depth(monkeypatch):
    engine, vocab, gate, _ = _engine(
        monkeypatch, max_lanes=2,
        execution_policy={"mtp_ordinary_handoff": {"enabled": True, "max_mtp_width": 1}},
    )
    try:
        body = {"tokens": _prompt(vocab, 24), "max_tokens": 2, "temperature": 0,
                "batch_cohort": {"id": "static-width", "size": 2}}
        jobs = [engine.submit(dict(body)) for _ in range(2)]
        assert all("error" not in collect(job) for job in jobs)
    finally:
        engine.close()
    # Width 2 exceeds the selected max_mtp_width while no lane is decoding:
    # the cohort will run ordinary decode and never verify num_draft rows.
    assert [options["draft_depth"] for options in gate] == [0, 0]


def test_single_mtp_arrival_keeps_its_speculative_depth(monkeypatch):
    engine, vocab, gate, _ = _engine(
        monkeypatch,
        execution_policy={"mtp_ordinary_handoff": {"enabled": True, "max_mtp_width": 1}},
    )
    try:
        body = {"tokens": _prompt(vocab, 24), "max_tokens": 2, "temperature": 0}
        assert "error" not in collect(engine.submit(body))
    finally:
        engine.close()
    assert [options["draft_depth"] for options in gate] == [2]


def test_warm_long_prompt_hit_prices_the_scheduled_prefill_chunk(monkeypatch):
    engine, vocab, gate, estimates = _engine(monkeypatch)
    try:
        # No prefill_step override and no adapter default: prompt-length
        # autoscale is selected (maximum 8192, 512-row chunks up to 32K).
        assert engine.prefill_step_autoscale is True
        prefix = _prompt(vocab, 300)
        first = {"tokens": prefix, "max_tokens": 2, "temperature": 0}
        assert "error" not in collect(engine.submit(first))
        second = dict(first, tokens=prefix + _prompt(vocab, 800, salt=5))
        result = collect(engine.submit(second))
        assert "error" not in result
    finally:
        engine.close()
    cached = result["receipt"].get("cached_tokens")
    assert cached, result["receipt"]
    context = 1100
    # The executor prefills the uncached tail in 512-row chunks; ingress
    # must price that chunk, not min(tail, 8192).
    assert estimates[-1] == (context, 512)
    assert gate[-1]["prefill_gib"] == pytest.approx(512 * MIB / (1 << 30))
