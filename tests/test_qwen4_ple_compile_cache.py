"""Compiled PLE chain cache policy: compile on repeat, LRU-bounded, no overflow.

Before the fix the cache never evicted: once it held ``cache_max`` signatures
every new one was recorded as ``overflow`` and ran eager forever, so a server
fed varied prompt lengths stopped compiling even its hot decode shapes.  These
tests drive the real ``PLELayer`` on CPU with the Metal-only gate patched open
(``mx.compile`` traces on CPU too) and a small cache, so many distinct row
counts stand in for many distinct prompt lengths.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx.utils import tree_map
from qsa_oracle import tiny_args

from mlx2.runtime.models import qwen4_exp as QE

CACHE_MAX = 4


@pytest.fixture
def layer(monkeypatch):
    monkeypatch.setattr(QE, "_PLE_COMPILE", True)
    monkeypatch.setattr(QE, "_PLE_COMPILE_CACHE_MAX", CACHE_MAX)
    monkeypatch.setattr(QE.mx, "default_device", lambda: mx.gpu)
    QE.qwen4_ple_compile_status(reset=True)
    mx.random.seed(0)
    module = QE.PLELayer(tiny_args(ple_layer_ids=[1]), 0, 0)
    module.update(
        tree_map(lambda v: mx.random.normal(v.shape) * 0.2, module.parameters())
    )
    module.set_dtype(mx.bfloat16)
    mx.eval(module.parameters())
    yield module
    QE.qwen4_ple_compile_status(reset=True)


def _inputs(module, rows, *, batch=1, seed=None):
    width = module.hidden_size * module.hc_count
    key = mx.random.key(rows if seed is None else seed)
    (k1, k2, k3) = mx.random.split(key, 3)
    hidden = mx.random.normal((batch, rows, width), key=k1).astype(mx.bfloat16)
    embeddings = mx.random.normal(
        (batch, rows, module.value_proj.weight.shape[1]), key=k2
    ).astype(mx.bfloat16)
    state = mx.random.normal(
        (batch, module.short_conv_state_len, width), key=k3
    ).astype(mx.bfloat16)
    return hidden, embeddings, state


def _run(module, rows, *, check=True):
    (hidden, embeddings, state) = _inputs(module, rows)
    out = module._run_device_chain(hidden, embeddings, None, state, True)
    if check:
        ref = module._device_chain(hidden, embeddings, None, state, True)
        mx.eval(out, ref)
        for (got, want) in zip(out, ref):
            assert mx.array_equal(got, want).item(), rows
    return out


def _counts():
    return QE.qwen4_ple_compile_status()["counts"]


def test_many_distinct_shapes_never_overflow_and_hot_shape_keeps_hitting(layer):
    """More distinct shapes than the cache holds, interleaved with decode."""
    hot_rows = 1
    hits_after_fill = None
    for i, rows in enumerate(range(2, 2 + 10 * CACHE_MAX)):
        _run(layer, rows)  # a recurring prompt length: seen twice
        _run(layer, rows)
        before = _counts().get("hits", 0)
        _run(layer, hot_rows)
        _run(layer, hot_rows)
        if i > 2 * CACHE_MAX:
            # Decode keeps compiling after the cache has been full for a while.
            assert _counts()["hits"] > before
            hits_after_fill = _counts()["hits"]
        assert len(layer._ple_compile_cache) <= CACHE_MAX
    counts = _counts()
    assert counts.get("overflow", 0) == 0
    assert counts["fallbacks"] == 0 and counts["skips"] == 0
    assert hits_after_fill is not None and counts["builds"] > CACHE_MAX
    assert counts["evictions"] > 0
    assert len(layer._ple_compile_cache) <= CACHE_MAX


def test_one_off_shapes_run_eager_without_filling_the_cache(layer):
    for _ in range(3):
        _run(layer, 1)  # hot decode shape: compiled on its second sighting
    assert _counts()["builds"] == 1 and _counts()["hits"] == 1
    for rows in range(2, 2 + 20 * CACHE_MAX):
        _run(layer, rows)  # every prompt tail a different length
    counts = _counts()
    assert counts["cold_eager"] == 1 + 20 * CACHE_MAX
    assert counts["builds"] == 1  # no one-off shape was traced
    assert counts.get("evictions", 0) == 0
    assert counts.get("overflow", 0) == 0
    assert len(layer._ple_compile_cache) == 1
    assert len(layer._ple_compile_seen) <= QE._ple_compile_seen_max()
    _run(layer, 1)
    assert _counts()["hits"] == 2  # the hot entry survived the one-offs


def test_evicted_shape_recompiles_and_lru_keeps_recent(layer):
    for rows in range(1, CACHE_MAX + 2):
        _run(layer, rows)
        _run(layer, rows)  # builds; the last build evicts rows=1
    counts = _counts()
    assert counts["builds"] == CACHE_MAX + 1
    assert counts["evictions"] == 1
    assert len(layer._ple_compile_cache) == CACHE_MAX
    builds = counts["builds"]
    _run(layer, CACHE_MAX + 1)  # still cached
    assert _counts()["builds"] == builds and _counts()["hits"] == 1
    _run(layer, 1)  # evicted: counts as a fresh sighting, runs eager
    _run(layer, 1)  # seen again: rebuilt
    assert _counts()["builds"] == builds + 1
    assert _counts()["evictions"] == 2


def test_failed_trace_is_not_retried(layer, monkeypatch):
    def boom(fn):
        raise RuntimeError("no trace")

    monkeypatch.setattr(QE.mx, "compile", boom)
    for _ in range(4):
        _run(layer, 3)
    counts = _counts()
    assert counts["fallbacks"] == 1
    assert counts["builds"] == 0


def test_status_reports_policy(layer):
    status = QE.qwen4_ple_compile_status()
    assert status["cache_max"] == CACHE_MAX
    assert status["min_seen"] == QE._PLE_COMPILE_MIN_SEEN
    assert "overflow" not in status["counts"]
    for name in ("cold_eager", "evictions"):
        assert status["counts"][name] == 0
