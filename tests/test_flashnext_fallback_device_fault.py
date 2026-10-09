"""The Flash-Next optional-path fallbacks re-raise a Metal device fault
(sweep 2026-10-08 review round 1, siblings of the probe fixes).

Each of these catches turns any exception into the ordinary path, and three
of them cache that answer: the compiled PLE chain caches ``None`` for the
signature, the compiled glue spans demote for the process, and the fused
group-norm self-check marked itself complete before it evaluated.  A device
fault (out-of-memory, GPU timeout) is serving's to recover, not a reason to
fall back; nothing is cached, so the next call compiles or probes again.
Kernel and trace errors keep their fallback.  CPU only.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx.utils import tree_map
from qsa_oracle import tiny_args

from mlx2.runtime.models import qwen3_next as QN
from mlx2.runtime.models import qwen4_exp as QE
from mlx2.runtime.models import qwen4_fused_group_norm as GN
from mlx2.runtime.models import qwen4_gate_inject as GI
from mlx2.serving import SIMULATED_GPU_TIMEOUT

OOM = (
    "[METAL] Command buffer execution failed: Insufficient Memory "
    "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
)
FAULTS = pytest.mark.parametrize(
    "fault", [OOM, SIMULATED_GPU_TIMEOUT], ids=["oom", "gpu_timeout"]
)
FAULT_TEXT = "Command buffer execution failed"


# --- gate/inject candidate --------------------------------------------------


@pytest.fixture
def gate_inject(monkeypatch):
    previous = GI.fused_gate_inject_enabled()
    GI.set_fused_gate_inject_enabled(True)
    GI.reset_qwen4_gate_inject_stats()
    monkeypatch.setattr(
        GI, "admit_qwen4_gate_inject",
        lambda *a, **k: GI.GateInjectAdmission(True, "eligible", "test"),
    )
    yield
    GI.reset_qwen4_gate_inject_stats()
    GI.set_fused_gate_inject_enabled(previous)


def _gate_inputs():
    return (mx.zeros((1, 1, 8)), mx.zeros((1, 1, 2)), mx.zeros((1, 1, 4)))


@FAULTS
def test_gate_inject_device_fault_reaches_the_caller(gate_inject, monkeypatch, fault):
    def launch(*args):
        raise RuntimeError(fault)

    monkeypatch.setattr(GI, "_launch", launch)
    with pytest.raises(RuntimeError, match=FAULT_TEXT):
        GI.try_qwen4_gate_inject(*_gate_inputs())
    assert GI.qwen4_gate_inject_stats()["counts"].get("decline:runtime_error", 0) == 0


def test_gate_inject_kernel_error_still_falls_back(gate_inject, monkeypatch):
    def launch(*args):
        raise RuntimeError("Unable to build metal library from source")

    monkeypatch.setattr(GI, "_launch", launch)
    assert GI.try_qwen4_gate_inject(*_gate_inputs()) is None
    assert GI.qwen4_gate_inject_stats()["counts"]["decline:runtime_error"] == 1


# --- compiled PLE chain -------------------------------------------------------


@pytest.fixture
def ple(monkeypatch):
    monkeypatch.setattr(QE, "_PLE_COMPILE", True)
    monkeypatch.setattr(QE, "_PLE_COMPILE_MIN_SEEN", 2)
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


def _ple_run(module, rows=3):
    width = module.hidden_size * module.hc_count
    (k1, k2, k3) = mx.random.split(mx.random.key(rows), 3)
    hidden = mx.random.normal((1, rows, width), key=k1).astype(mx.bfloat16)
    embeddings = mx.random.normal(
        (1, rows, module.value_proj.weight.shape[1]), key=k2
    ).astype(mx.bfloat16)
    state = mx.random.normal(
        (1, module.short_conv_state_len, width), key=k3
    ).astype(mx.bfloat16)
    out = module._run_device_chain(hidden, embeddings, None, state, True)
    ref = module._device_chain(hidden, embeddings, None, state, True)
    mx.eval(out, ref)
    for (got, want) in zip(out, ref):
        assert mx.array_equal(got, want).item()


def _ple_counts():
    return QE.qwen4_ple_compile_status()["counts"]


def _ple_signatures(module):
    return list(getattr(module, "_ple_compile_cache", {}).items())


@FAULTS
def test_ple_chain_device_fault_is_not_cached_as_eager(ple, monkeypatch, fault):
    real_compile = mx.compile
    pending = [True]

    def compile_faulting_once(fn):
        compiled = real_compile(fn)

        def call(*args):
            if pending[0]:
                pending[0] = False
                raise RuntimeError(fault)
            return compiled(*args)

        return call

    _ple_run(ple)  # first sighting: eager
    monkeypatch.setattr(QE.mx, "compile", compile_faulting_once)
    with pytest.raises(RuntimeError, match=FAULT_TEXT):
        _ple_run(ple)  # built, then the compiled call faults
    assert all(entry[1] is not None for _, entry in _ple_signatures(ple))
    _ple_run(ple)  # recovered: the cached compiled chain runs
    counts = _ple_counts()
    assert counts.get("fallbacks", 0) == 0
    assert counts["builds"] == 1 and counts["hits"] == 1


@FAULTS
def test_ple_chain_build_device_fault_is_not_cached_as_eager(ple, monkeypatch, fault):
    real_compile = mx.compile

    def boom(fn):
        raise RuntimeError(fault)

    _ple_run(ple)
    monkeypatch.setattr(QE.mx, "compile", boom)
    with pytest.raises(RuntimeError, match=FAULT_TEXT):
        _ple_run(ple)
    assert _ple_signatures(ple) == []
    monkeypatch.setattr(QE.mx, "compile", real_compile)
    for _ in range(3):
        _ple_run(ple)
    counts = _ple_counts()
    assert counts.get("fallbacks", 0) == 0 and counts["builds"] == 1


def test_ple_chain_trace_error_still_cached_as_eager(ple, monkeypatch):
    def compile_raising(fn):
        def call(*args):
            raise RuntimeError("trace failed")

        return call

    monkeypatch.setattr(QE.mx, "compile", compile_raising)
    for _ in range(4):
        _ple_run(ple)
    counts = _ple_counts()
    assert counts["fallbacks"] == 1 and counts["builds"] == 1
    assert [entry[1] for _, entry in _ple_signatures(ple)] == [None]


# --- compiled glue spans ------------------------------------------------------


def _span(a):
    return a * 2


def _compile_raising_once(monkeypatch, error):
    real_compile = mx.compile
    pending = [True]

    def compile_(fn, **kwargs):
        compiled = real_compile(fn, **kwargs)

        def call(*args):
            if pending[0]:
                pending[0] = False
                raise error
            return compiled(*args)

        return call

    monkeypatch.setattr(QN.mx, "compile", compile_)


@pytest.fixture
def glue_key():
    key = ("test_glue_span",)
    QN._GLUE_COMPILE_CACHE.pop(key, None)
    yield key
    QN._GLUE_COMPILE_CACHE.pop(key, None)


@FAULTS
def test_glue_span_device_fault_does_not_demote(glue_key, monkeypatch, fault):
    _compile_raising_once(monkeypatch, RuntimeError(fault))
    a = mx.ones((2, 3), dtype=mx.bfloat16)
    with pytest.raises(RuntimeError, match=FAULT_TEXT):
        QN._run_glue(glue_key, lambda: _span, a)
    assert QN._GLUE_COMPILE_CACHE[glue_key] is not None
    out = QN._run_glue(glue_key, lambda: _span, a)
    assert out is not None and mx.array_equal(out, a * 2).item()


def test_glue_span_error_still_demotes(glue_key, monkeypatch):
    _compile_raising_once(monkeypatch, RuntimeError("trace failed"))
    a = mx.ones((2, 3), dtype=mx.bfloat16)
    assert QN._run_glue(glue_key, lambda: _span, a) is None
    assert QN._GLUE_COMPILE_CACHE[glue_key] is None
    assert QN._run_glue(glue_key, lambda: _span, a) is None


# --- fused group-norm self-check ---------------------------------------------


@pytest.fixture
def group_norm_probe(monkeypatch):
    monkeypatch.setattr(GN, "_PROBE_COMPLETE", False)
    monkeypatch.setattr(GN, "_PROBE_OK", False)
    monkeypatch.setattr(GN.mx.metal, "is_available", lambda: True)
    calls = []
    state = {"error": None}

    def fused(x, w, **kwargs):
        calls.append(1)
        if state["error"] is not None:
            error, state["error"] = state["error"], None
            raise error
        return GN.eager_group_norm(x, w)

    monkeypatch.setattr(GN, "fused_group_norm", fused)
    return calls, state


@FAULTS
def test_group_norm_probe_reraises_device_fault_and_probes_again(group_norm_probe, fault):
    calls, state = group_norm_probe
    state["error"] = RuntimeError(fault)
    with pytest.raises(RuntimeError, match=FAULT_TEXT):
        GN.probe_fused_group_norm(rows=2)
    assert GN._PROBE_COMPLETE is False
    assert GN.probe_fused_group_norm(rows=2) is True
    assert GN.probe_fused_group_norm(rows=2) is True
    assert len(calls) == 2


def test_group_norm_probe_still_caches_a_kernel_error(group_norm_probe):
    calls, state = group_norm_probe
    state["error"] = RuntimeError("Unable to build metal library from source")
    assert GN.probe_fused_group_norm(rows=2) is False
    assert GN.probe_fused_group_norm(rows=2) is False
    assert len(calls) == 1
