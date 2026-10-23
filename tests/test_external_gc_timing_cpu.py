"""GC timing ownership and clock isolation; synthetic CPU-only fixtures."""

import gc
import time
import weakref
from types import SimpleNamespace

import pytest

from mlx2.runtime import external_speculative as es

from test_dflash_pair_select import generator, tiny


class Owner:
    def __init__(self):
        self.scheduler_stats = {}


def test_gc_timing_uses_real_clock_when_executor_clock_is_replaced(monkeypatch):
    monkeypatch.setattr(es, "time", SimpleNamespace(perf_counter=lambda: 0))
    owner = Owner()
    es._install_gc_timing(owner)
    try:
        gc.collect(1)
        es._publish_gc_timing(owner)
        assert owner.scheduler_stats["external_gc_gen1_calls"] >= 1
        assert owner.scheduler_stats["external_gc_gen1_ns"] > 0
    finally:
        es._remove_gc_timing(owner)


def test_dead_gc_owner_never_reads_clock_or_mutates_callback_list(monkeypatch):
    reads = []
    monkeypatch.setattr(time, "perf_counter_ns", lambda: reads.append(1) or 1)
    owner = Owner()
    es._install_gc_timing(owner)
    callback = owner._gc_callback
    ref = weakref.ref(owner)
    del owner
    assert ref() is None
    reads.clear()
    before = list(gc.callbacks)
    try:
        callback("start", {"generation": 1})
        callback("stop", {"generation": 1})
        assert reads == []
        assert gc.callbacks == before
    finally:
        gc.callbacks.remove(callback)


def test_gc_timing_reinstall_replaces_only_owners_callback():
    owner = Owner()
    before = list(gc.callbacks)
    es._install_gc_timing(owner)
    first = owner._gc_callback
    try:
        es._install_gc_timing(owner)
        assert first not in gc.callbacks
        assert gc.callbacks == before + [owner._gc_callback]
    finally:
        es._remove_gc_timing(owner)
    es._remove_gc_timing(owner)
    assert gc.callbacks == before


def test_timed_generators_close_detaches_only_owned_callbacks(monkeypatch):
    monkeypatch.setenv("MLX2_EXTERNAL_ROUND_TIMING", "1")
    model, draft = tiny()
    before = list(gc.callbacks)
    first, second = generator(model, draft), generator(model, draft)
    first_callback, second_callback = first._gc_callback, second._gc_callback
    try:
        assert gc.callbacks == before + [first_callback, second_callback]
        first.close()
        first.close()
        assert gc.callbacks == before + [second_callback]
        gc.collect(1)
        es._publish_gc_timing(second)
        assert second.scheduler_stats["external_gc_gen1_calls"] >= 1
    finally:
        first.close()
        second.close()
    assert gc.callbacks == before


def test_close_detaches_gc_callback_even_if_lane_cleanup_fails(monkeypatch):
    monkeypatch.setenv("MLX2_EXTERNAL_ROUND_TIMING", "1")
    model, draft = tiny()
    batch = generator(model, draft)
    callback = batch._gc_callback

    def fail(_uids):
        raise RuntimeError("synthetic cleanup failure")

    with monkeypatch.context() as patch:
        patch.setattr(batch, "remove", fail)
        with pytest.raises(RuntimeError, match="synthetic cleanup failure"):
            batch.close()
    assert callback not in gc.callbacks
    batch.close()
