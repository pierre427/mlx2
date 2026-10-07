"""Batch geometry policy truthfulness (sweep 2026-10-06, SS-5/SS-6)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_decode_first_publish import tiny_model

from mlx2.runtime import generate as G
from mlx2.runtime.batch_geometry import GeometrySchedulerPolicy


@pytest.fixture(scope="module")
def model():
    return tiny_model()


def test_geometry_with_srpt_is_refused(model):
    """SRPT admission bypasses the geometry selector (all counters were 0)."""
    with pytest.raises(ValueError, match="batch_geometry cannot be combined"):
        G.BatchGenerator(
            model, completion_batch_size=8, prefill_batch_size=2,
            prefill_step_size=64, batch_geometry=True,
            prefill_scheduling={"order": "srpt", "max_bypass": 3},
        )


def test_serving_refuses_geometry_with_srpt(monkeypatch):
    from mlx2 import serving

    from test_apc_interior_route_selection import _UnsupportedInteriorAdapter

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "fake"})
    engine = serving.ServingEngine(
        "fixture",
        adapter_factory=_UnsupportedInteriorAdapter,
        qualification_mode=True,
        mtp=False,
        execution_policy={
            "batch_geometry": True,
            "prefill_scheduling": {"order": "srpt"},
        },
    )
    try:
        engine.thread.join(5)
        assert not engine.ready.is_set()
        assert "batch_geometry cannot be combined" in (engine.error or "")
    finally:
        engine.close()


def test_packed_padding_fraction_is_not_a_scheduler_knob():
    """Nothing in the scheduler read it; it was recorded in route identity."""
    with pytest.raises(ValueError, match="unknown batch_geometry"):
        GeometrySchedulerPolicy.from_value({"packed_padding_fraction": 0.3})
    assert "packed_padding_fraction" not in GeometrySchedulerPolicy.from_value(
        True
    ).as_dict()


def test_geometry_judges_rows_at_the_contended_slice(model):
    """600, 2000 and 700 tokens beside decode all execute at the bounded
    slice (no padding between them), so geometry must not pass over the
    2000-token row as incompatible: judged at the uncontended step it was
    3.3x the 600-token row and lost its FIFO turn to the 700-token row."""
    gen = G.BatchGenerator(
        model, completion_batch_size=8, prefill_batch_size=2,
        prefill_batch_window=4, prefill_step_size=2048,
        batch_geometry=True,
        decode_time_fairness={"enabled": True, "stall_target_ms": 500.0},
    )
    try:
        gen.insert([[5, 6, 7, 8]], max_tokens=[200])
        for _ in range(3):
            gen.next()
        uids = gen.insert(
            [
                [(i % 50) + 1 for i in range(600)],
                [(i * 3 % 50) + 1 for i in range(2000)],
                [(i * 7 % 50) + 1 for i in range(700)],
            ],
            max_tokens=[2, 2, 2],
        )
        gen.next()
        assert sorted(gen._prompt_batch.uids) == sorted(uids[:2])
    finally:
        gen.close()
