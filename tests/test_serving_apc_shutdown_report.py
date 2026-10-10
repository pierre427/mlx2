"""The serving worker consumes APCv2's shutdown park report instead of
discarding it: parked sessions left without a disk copy are counted and
logged by the service."""

import logging
from collections import Counter

from mlx2 import serving as serving_mod
from mlx2.serving import ServingEngine


class _StubAPC:
    def __init__(self, report):
        self.report = report
        self.calls = []

    def close(self, **kwargs):
        self.calls.append(kwargs)
        return self.report


def _engine():
    engine = ServingEngine.__new__(ServingEngine)
    engine.apc_persist_on_shutdown = True
    engine.apc_persist_shutdown_seconds = 7
    engine.counts = Counter()
    return engine


def test_shutdown_close_report_is_counted_and_logged(caplog):
    apc = _StubAPC({"spilled": 2, "skipped": 1, "pinned_skipped": 1})
    engine = _engine()
    engine.apc = apc
    with caplog.at_level(logging.ERROR, logger=serving_mod.log.name):
        report = engine._close_apc_on_shutdown(apc)
    assert report is apc.report
    assert apc.calls == [{"persist_resident": True, "time_budget_seconds": 7}]
    assert engine.apc is None
    assert engine.counts["apc_shutdown_pinned_parks_lost"] == 1
    assert engine.counts["apc_shutdown_parks_skipped"] == 1
    assert any(
        "missing after restart" in record.getMessage() and "1 pinned" in record.getMessage()
        for record in caplog.records
    ), [record.getMessage() for record in caplog.records]


def test_shutdown_close_without_losses_is_quiet(caplog):
    apc = _StubAPC({"spilled": 3, "skipped": 0, "pinned_skipped": 0})
    engine = _engine()
    engine.apc = apc
    with caplog.at_level(logging.WARNING, logger=serving_mod.log.name):
        engine._close_apc_on_shutdown(apc)
    assert engine.counts["apc_shutdown_pinned_parks_lost"] == 0
    assert engine.counts["apc_shutdown_parks_skipped"] == 0
    assert not caplog.records


def test_apc_shutdown_counters_are_declared_before_any_shutdown():
    """``/status`` publishes ``dict(self.counts)``: the two shutdown counters
    must exist at zero from construction, not appear on first increment
    (seam review A, N3)."""
    counts = _declared_counts()
    assert counts["apc_shutdown_parks_skipped"] == 0
    assert counts["apc_shutdown_pinned_parks_lost"] == 0
    assert counts["apc_invalidation_pinned_parks_lost"] == 0
    assert counts["apc_suspend_pinned_parks_lost"] == 0


def _declared_counts():
    """The ``Counter({...})`` literal the engine constructor declares."""
    import ast
    import inspect

    import mlx2.serving as serving

    tree = ast.parse(inspect.getsource(serving))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Attribute) and t.attr == "counts"
                for t in node.targets
            )
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None) == "Counter"
            and node.value.args
            and isinstance(node.value.args[0], ast.Dict)
        ):
            literal = node.value.args[0]
            return {
                k.value: v.value
                for k, v in zip(literal.keys, literal.values)
                if k is not None and isinstance(v, ast.Constant)
            }
    raise AssertionError("engine counts literal not found")


def test_shutdown_with_a_clear_only_cache_still_releases_it():
    class _ClearOnly:
        cleared = False

        def clear(self):
            self.cleared = True

    cache = _ClearOnly()
    engine = _engine()
    engine.apc = cache
    assert engine._close_apc_on_shutdown(cache) is None
    assert cache.cleared and engine.apc is None


from types import SimpleNamespace as NS  # noqa: E402

from test_structured_deferral import scripted_engine  # noqa: E402,F401 - fixture


def test_worker_teardown_consumes_the_apc_shutdown_report(scripted_engine, monkeypatch, tmp_path, caplog):
    """Through the real worker finaliser (not the method alone): the report
    APCv2.close() returns at worker exit reaches the engine's counters and
    its log."""
    from mlx2.runtime import apc_v2

    build, _state = scripted_engine
    calls = []

    class APC:  # the fixture's stub surface, plus a close() that reports
        def __init__(self, **kw): self.apc_stats = {}
        def key(self, *a, **kw): return "key"
        def lookup(self, key, tokens, **kw):
            return NS(cache=[NS(nbytes=0)], cached_tokens=len(tokens) - 1, remaining_tokens=[1],
                      sidecar=None, miss_reason=None)
        def store(self, *a, **kw): pass
        def spill_idle_entries(self): pass
        def evict_oldest_unleased(self): return False
        def clear(self): pass
        def __len__(self): return 0

        def close(self, **kw):
            calls.append(kw)
            return {"spilled": 0, "skipped": 1, "pinned_skipped": 1}

    monkeypatch.setattr(apc_v2, "APCv2", APC)
    engine = build(
        declare_marker=True,
        apc_persist_dir=str(tmp_path),
        apc_persist_on_shutdown=True,
        apc_persist_shutdown_seconds=3,
    )
    with caplog.at_level(logging.ERROR, logger=serving_mod.log.name):
        engine.close()
    assert not engine.thread.is_alive() and engine.error is None
    assert calls == [{"persist_resident": True, "time_budget_seconds": 3}]
    assert engine.apc is None
    assert engine.counts["apc_shutdown_pinned_parks_lost"] == 1
    assert engine.counts["apc_shutdown_parks_skipped"] == 1
    assert any(
        "missing after restart" in record.getMessage() for record in caplog.records
    ), [record.getMessage() for record in caplog.records]
