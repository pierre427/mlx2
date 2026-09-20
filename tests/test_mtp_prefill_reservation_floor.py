"""A bounded self-MTP prefill never stalls behind its own reservation.

After the first teacher-forced slice, the generator records the lane's cache
reservation from the adapter's ``cache_budget().project``.  Each later
admission boundary treats ``resident bytes > reservation`` as an append-only
bound violation and withholds the row.  Two defects combined into a silent
hang (roadmap worker rm04, 2026-09-19):

  * the reservation was the adapter projection alone, even when the slice
    had already allocated more (step-rounded K/V plus the draft cache), so the
    very next boundary was a violation;
  * a violating row was neither admitted, memory-queued nor failed, so the
    request emitted no event and was never reported to the watchdog.

The reservation is now floored at the bytes already resident, and a lane
that later outgrows it fails closed with an error and a counter.  CPU only.
"""

import time

import pytest
from test_apc_hits_hybrid_gdn_self_mtp import make_adapter, tiny_qwen38_mtp

from mlx2 import memory, serving
from mlx2.runtime import os_memory


class TinyBudget:
    """Deliberately below the tiny Qwen3.8 resident footprint after a slice."""

    def __init__(self, base, per_token):
        self.base = base
        self.per_token = per_token

    def project(self, context_tokens):
        return self.base + self.per_token * context_tokens

    def as_dict(self):
        return {}


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)


def make_engine(model, vocab, budget=None):
    adapter = make_adapter(model, vocab)
    if budget is not None:

        class Adapter(adapter):
            def cache_budget(self, *, mtp):
                return budget

        adapter = Adapter
    engine = serving.ServingEngine(
        "tiny", adapter_factory=adapter, qualification_mode=True, mtp=True,
        max_lanes=1, prefill_step=16,
    )
    assert engine.ready.wait(60), engine.error
    return engine


def run(engine, tokens, max_tokens=8):
    job = engine.submit({"tokens": list(tokens), "max_tokens": max_tokens,
                         "temperature": 0})
    text = ""
    while True:
        # A stalled lane emits nothing; fail instead of waiting forever.
        event = job.events.get(timeout=60)
        if "error" in event:
            return None, event
        if "delta" in event:
            text += event["delta"].get("content", "")
        if "finish_reason" in event:
            return [int(t) for t in text.split()], event


def scheduler_counter(engine, key, deadline_s=10):
    deadline = time.monotonic() + deadline_s
    while True:
        value = int((engine.status().get("scheduler") or {}).get(key, 0))
        if value or time.monotonic() > deadline:
            return value
        time.sleep(0.2)


def test_underestimating_budget_is_floored_to_resident_bytes(host):
    model, vocab = tiny_qwen38_mtp()
    prompt = [(7 * i + 3) % (vocab - 2) + 1 for i in range(84)]

    reference = make_engine(model, vocab)
    try:
        expected, _ = run(reference, prompt)
    finally:
        reference.close()
    assert expected is not None

    # project(84) = 90,112 B; one 16-token slice already holds ~135 KB
    # (a 256-token fp32 K/V step for the target plus the MTP draft layer).
    engine = make_engine(model, vocab, TinyBudget(4096, 1024))
    try:
        out, event = run(engine, prompt)
        assert out == expected, event
        assert scheduler_counter(
            engine, "mtp_prefill_projection_floored_to_resident"
        ) >= 1
        assert engine.counts.get("mtp_prefill_bound_failures", 0) == 0
    finally:
        engine.close()


def test_prefill_outgrowing_its_reservation_fails_closed(host):
    model, vocab = tiny_qwen38_mtp()
    # Crossing 256 tokens grows the full-attention K/V by a second step,
    # past the reservation floored at the first slice's resident bytes.
    prompt = [(7 * i + 3) % (vocab - 2) + 1 for i in range(300)]
    engine = make_engine(model, vocab, TinyBudget(0, 1))
    try:
        out, event = run(engine, prompt)
        assert out is None, event
        assert event["status"] == 503
        assert "outgrew its admitted reservation" in event["error"]
        assert engine.counts["mtp_prefill_bound_failures"] == 1
        assert scheduler_counter(engine, "mtp_prefill_bound_violations") == 1
        # The engine keeps serving after failing the lane closed.
        short, _ = run(engine, prompt[:40], max_tokens=4)
        assert short is not None and len(short) == 4
    finally:
        engine.close()


def test_real_qwen38_budget_is_not_floored(host):
    from mlx2.adapters.qwen38_memory import Qwen38CacheBudget

    model, vocab = tiny_qwen38_mtp()
    budget = Qwen38CacheBudget.from_config(dict(vars(model.args)), mtp=True)
    engine = make_engine(model, vocab, budget)
    try:
        out, event = run(engine, [(7 * i + 3) % (vocab - 2) + 1 for i in range(84)])
        assert out is not None, event
        assert scheduler_counter(
            engine, "mtp_prefill_projection_floored_to_resident", deadline_s=2
        ) == 0
    finally:
        engine.close()

