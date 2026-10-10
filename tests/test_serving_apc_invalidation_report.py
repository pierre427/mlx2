"""A model-state invalidation (LoRA load/unload) clears APCv2, parked
sessions included.  The engine consumes clear()'s loss report instead of
discarding it: the lost parks are counted and logged with the reason."""

import logging
from collections import Counter

import pytest

from mlx2 import serving as serving_mod
from mlx2.serving import ServingEngine

COUNTER = "apc_invalidation_pinned_parks_lost"


class _StubAPC:
    def __init__(self, report):
        self.report = report
        self.calls = 0

    def clear(self):
        self.calls += 1
        return self.report


def _engine(apc):
    engine = ServingEngine.__new__(ServingEngine)
    engine.model_revision = 3
    engine.host_prompt_cache = {}
    engine.incremental_tokenizer_cache = {}
    engine.adapter = object()
    engine.apc = apc
    engine.counts = Counter()
    return engine


def test_invalidation_counts_and_logs_the_parked_sessions_it_destroyed(caplog):
    apc = _StubAPC({
        "entries": 3, "sidecars": 0, "bytes": 10,
        "pinned_parks_lost": 2, "lost_sessions": ["alpha", "beta"],
    })
    engine = _engine(apc)
    with caplog.at_level(logging.ERROR, logger=serving_mod.log.name):
        engine._invalidate_model_state(reason="lora_unload")
    assert apc.calls == 1 and engine.model_revision == 4
    assert engine.counts[COUNTER] == 2
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "lora_unload" in message and "alpha" in message and "beta" in message
        for message in messages
    ), messages


@pytest.mark.parametrize("report", [
    {"entries": 2, "sidecars": 0, "bytes": 10, "pinned_parks_lost": 0, "lost_sessions": []},
    None,  # a cache whose clear() reports nothing
])
def test_invalidation_without_parked_sessions_is_quiet(caplog, report):
    apc = _StubAPC(report)
    engine = _engine(apc)
    with caplog.at_level(logging.WARNING, logger=serving_mod.log.name):
        engine._invalidate_model_state(reason="lora_load")
    assert apc.calls == 1 and engine.counts[COUNTER] == 0
    assert not caplog.records


def test_invalidation_without_a_cache_still_bumps_the_revision():
    engine = _engine(None)
    engine._invalidate_model_state()
    assert engine.model_revision == 4 and engine.counts[COUNTER] == 0


from test_structured_deferral import scripted_engine  # noqa: E402,F401 - fixture


def test_counter_is_declared_at_zero(scripted_engine):
    build, _state = scripted_engine
    engine = build(declare_marker=True)
    try:
        assert COUNTER in engine.counts and engine.counts[COUNTER] == 0
    finally:
        engine.close()
