"""A memory-admission 429 states required vs available bytes and the blocking term.

Generic (no model): the breakdown is re-derived from the admission
controller, so its terms must sum to exactly what lane admission tested.
"""

import pytest

from mlx2 import memory
from mlx2.admission_diagnostics import lane_admission_breakdown, summarize
from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController as C
from mlx2.serving import ServingEngine, lane_admission_required_gib
from test_serving_deferred_admission import deferred_engine  # noqa: F401 - fixture

GIB = 1 << 30


@pytest.fixture
def reserve8(monkeypatch):
    monkeypatch.setenv("MLX2_SERVICE_RESERVE_GIB", "8")
    return C(host_memory_gib=128.0, advisory_gib=112.0)


def test_breakdown_reproduces_the_hy4_32k_refusal(reserve8):
    """Ladder cell 32K at reserve 8: no cache projection, 24.84 GiB fresh headroom."""
    context = 32995 + 24
    required = lane_admission_required_gib(reserve8, context_tokens=context, draft_depth=0,
                                           cache_gib=0.0)
    terms = {"host_available": 30 * GIB, "host_term": 35 * GIB,
             "advisory_term": 26671970544, "recommended": 112 * GIB}
    b = lane_admission_breakdown(reserve8, context_tokens=context, draft_depth=0, cache_gib=0.0,
                                 tested_required_gib=required, available_bytes=26671970544,
                                 headroom_terms=terms)
    assert sum(b["terms_gib"].values()) == pytest.approx(required, abs=1e-3)
    assert "other" not in b["terms_gib"]
    assert b["required_gib"] == pytest.approx(26.77, abs=0.01)
    assert b["available_gib"] == pytest.approx(24.84, abs=0.01)
    assert b["blocking_term"] == "lane_context_cache"
    assert b["terms_gib"]["lane_context_cache"] == pytest.approx(14.19, abs=0.01)
    assert b["cache_estimate"] == "envelope_0.44_gib_per_1k_tokens"
    # Dropping the reserve to 4 GiB also admits it: the 32K@reserve4 pass.
    assert b["sufficient_terms"][:2] == ["lane_context_cache", "service_reserve"]
    assert b["headroom"]["binding"] == "advisory_term"
    line = summarize(b)
    assert line.startswith("required 26.77 GiB > available 24.84 GiB (short 1.93)")
    assert "blocking term lane_context_cache 14.19 GiB" in line
    assert "headroom bound by advisory_term 24.84 GiB" in line


def test_breakdown_reports_a_shortfall_no_single_term_covers(reserve8):
    b = lane_admission_breakdown(reserve8, context_tokens=1024, draft_depth=2, cache_gib=0.0,
                                 prefill_gib=1.0, available_bytes=0)
    assert not b["single_term_covers_shortfall"] and b["sufficient_terms"] == []
    assert b["blocking_term"] == "service_reserve"  # the largest term
    assert "no single term covers the shortfall" in summarize(b)
    assert b["terms_gib"]["verify_transient"] == pytest.approx(1.76, abs=1e-3)


def test_tested_requirement_difference_is_an_explicit_term(reserve8):
    b = lane_admission_breakdown(reserve8, context_tokens=100, draft_depth=0, cache_gib=0.0,
                                 tested_required_gib=40.0, available_bytes=20 * GIB)
    assert b["required_gib"] == 40.0 and b["terms_gib"]["other"] > 0
    assert sum(b["terms_gib"].values()) == pytest.approx(40.0, abs=1e-3)


def test_deadline_429_names_required_available_and_blocking_term(
        deferred_engine, monkeypatch, caplog):  # noqa: F811
    engine, state = deferred_engine
    monkeypatch.setattr(engine, "MEMORY_ADMISSION_TIMEOUT", .06)
    monkeypatch.setattr(memory, "execution_headroom_terms", lambda host_signals=False: {
        "host_available": 50 * GIB, "host_term": 63 * GIB, "advisory_term": 21 * GIB,
        "recommended": 112 * GIB, "active": 0, "cached": 0, "footprint": 0})
    state["cold"] = True
    caplog.set_level("INFO", logger="mlx2.serving")
    waiting = engine.submit({"max_tokens": 1, "large": True})
    result = waiting.events.get(timeout=5)
    assert result["status"] == 429
    message = result["error"]
    assert message.startswith(ServingEngine.ADMISSION_REFUSAL + ": required ")
    assert "GiB > available 21.00 GiB" in message and "blocking term " in message
    assert "headroom bound by advisory_term 21.00 GiB" in message
    b = waiting.admission_shortfall
    assert b["available_gib"] == 21.0 and b["required_gib"] > 21.0
    assert sum(b["terms_gib"].values()) == pytest.approx(b["required_gib"], abs=1e-3)
    assert b["blocking_term"] in b["terms_gib"]
    assert any("memory admission deferred request" in r.message for r in caplog.records)
    assert any("memory admission refused request" in r.message and "blocking term" in r.message
               for r in caplog.records)
