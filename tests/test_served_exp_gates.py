"""Served-graph exp gates for the routed-decode, HC-decode and QSA kernels.

Follow-up to the fused GDN gate (omlx#4122, tests/test_fused_gdn_served_silu.py):
each kernel below copies one MLX sigmoid spelling and must fall back when the
served op uses the other one. CPU only: probe outcomes are injected; the real
probes run in the opt-in Metal test at the end.
"""

import os
import re

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx2.runtime.models import qwen3_next as QN
from mlx2.runtime.models import qwen4_hc_decode as HCD
from mlx2.runtime.models import qwen4_moe_window as W
from mlx2.runtime.models import qwen4_qsa_indexed as indexed
from mlx2.runtime.models import qwen4_qsa_indexed_merge as merge
from mlx2.runtime.models import qwen4_routed_decode as RD
from mlx2.runtime.models import served_exp as SE

FAST, PRECISE = SE.EXP_SPELLINGS
ALL_GATES = (
    RD.SWIGLU_GATE,
    RD.SHARED_GATE_GATE,
    HCD.SILU_GATE,
    HCD.SIGMOID_GATE,
    merge.OUTPUT_GATE_GATE,
)


@pytest.fixture(autouse=True)
def _fresh_gates():
    for gate in ALL_GATES:
        gate.reset()
    yield
    for gate in ALL_GATES:
        gate.reset()


def _inject(monkeypatch, gate, outcome):
    """Install an injected probe outcome; returns the probe-call log."""
    calls = []

    def probe(dtype):
        calls.append(dtype)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(gate, "probe", probe)
    return calls


# --------------------------------------------------------------------------
# The gate itself
# --------------------------------------------------------------------------
def test_respell_rewrites_only_the_kernel_spelling():
    text = "a = metal::exp(x); b = metal::precise::exp(x);"
    assert SE.respell(text, FAST, PRECISE) == (
        "a = metal::precise::exp(x); b = metal::precise::exp(x);"
    )
    assert SE.respell(text, PRECISE, FAST) == "a = metal::exp(x); b = metal::exp(x);"
    with pytest.raises(ValueError):
        SE.respell("no exp here", FAST, PRECISE)
    with pytest.raises(ValueError):
        SE.respell(text, "metal::fast::exp", FAST)


def test_gate_requires_its_spelling_in_the_probe_text():
    with pytest.raises(ValueError):
        SE.ServedExpGate("x", served_name="s", served=mx.sigmoid, body="y = x;")
    gate = SE.ServedExpGate(
        "x", served_name="s", served=mx.sigmoid,
        body="y = metal::precise::exp(x);", kernel_exp=PRECISE,
    )
    assert gate.candidates() == (PRECISE, FAST)


def test_probe_inputs_cover_every_16bit_encoding_and_sample_float32():
    for dtype in (mx.bfloat16, mx.float16):
        x = SE.probe_inputs(dtype)
        assert x.dtype == dtype and x.size == 1 << 16
    wide = SE.probe_inputs(mx.float32)
    assert wide.dtype == mx.float32 and wide.size == (1 << 16) + (1 << 20)
    assert SE.same_bits(wide, SE.probe_inputs(mx.float32))  # fixed seed
    with pytest.raises(ValueError):
        SE.probe_inputs(mx.int32)


@pytest.mark.parametrize(
    ("kernel_exp", "outcome", "refusal"),
    [
        (FAST, {FAST: True, PRECISE: False}, None),
        (FAST, {FAST: True, PRECISE: True}, None),
        (FAST, {FAST: False, PRECISE: True}, "served op uses metal::precise::exp"),
        (FAST, {FAST: False, PRECISE: False}, "served op matches no kernel form"),
        (FAST, RuntimeError("no Metal"), "served op matches no kernel form"),
        (PRECISE, {FAST: True, PRECISE: True}, None),
        (PRECISE, {FAST: True, PRECISE: False}, "served op uses metal::exp"),
    ],
)
def test_refusal_follows_one_probe_per_process_and_dtype(monkeypatch, kernel_exp, outcome, refusal):
    gate = SE.ServedExpGate(
        "t", served_name="op", served=mx.sigmoid, body=f"y = {kernel_exp}(x);",
        kernel_exp=kernel_exp,
    )
    calls = _inject(monkeypatch, gate, outcome)
    assert gate.refusal(mx.bfloat16) == refusal
    assert gate.refusal(mx.bfloat16) == refusal
    assert len(calls) == 1
    gate.refusal(mx.float16)
    assert calls == [mx.bfloat16, mx.float16]
    gate.reset()
    gate.refusal(mx.bfloat16)
    assert len(calls) == 3


@pytest.mark.parametrize(
    "fault",
    [
        RuntimeError("[METAL] Command buffer execution failed: Insufficient Memory"),
        RuntimeError(
            "[METAL] Command buffer execution failed: Caused GPU Timeout Error "
            "(00000002:kIOGPUCommandBufferCallbackErrorTimeout)"
        ),
    ],
)
def test_device_fault_in_probe_reaches_serving_and_is_not_cached(monkeypatch, fault):
    # Serving recovers a failed command buffer by abandoning its lanes; a probe
    # that swallowed it would hide the fault and refuse the kernel for good.
    gate = SE.ServedExpGate(
        "t", served_name="op", served=mx.sigmoid, body=f"y = {FAST}(x);",
        kernel_exp=FAST,
    )
    calls = _inject(monkeypatch, gate, fault)
    with pytest.raises(RuntimeError):
        gate.refusal(mx.bfloat16)
    calls_after_fault = len(calls)
    _inject(monkeypatch, gate, {FAST: True, PRECISE: False})
    assert gate.refusal(mx.bfloat16) is None
    assert calls_after_fault == 1


def test_real_probe_fails_closed_without_metal():
    """On the CPU test device the kernel cannot run: the gate refuses."""
    gate = RD.SWIGLU_GATE
    if mx.metal.is_available() and mx.default_device() == mx.gpu:
        pytest.skip("Metal default device")
    assert gate.refusal(mx.bfloat16) == "served compiled SwiGLU matches no kernel form"


# --------------------------------------------------------------------------
# Spelling contract: each probe checks the expression its kernels run
# --------------------------------------------------------------------------
def _squash(text):
    """Whitespace-normalised Metal text without ``//`` comments."""
    return re.sub(r"\s+", " ", re.sub(r"//[^\n]*", "", text)).strip()


def test_routed_swiglu_sites_copy_the_probed_epilogue():
    assert "1 / (1 + metal::exp(metal::abs(x)))" in RD.SIGMOID
    assert RD.SWIGLU_GATE.header == RD.SIGMOID
    epilogue = "T t = g * omlx_mlx_sigmoid<T>(g);"
    sources = {
        "gate_up": RD.GATE_UP_SOURCE,
        "split": RD.SPLIT_GATE_UP_SOURCE,
        "shared": RD.SHARED_COMMON,
    }
    for name, source in sources.items():
        assert source.count(epilogue) == 1, name
        assert "t * u;" in source[source.index(epilogue):], name
    # Every omlx_mlx_sigmoid call in the module is this epilogue.
    import inspect

    module = inspect.getsource(RD)
    calls = re.findall(r"omlx_mlx_sigmoid<T>\((\w+)\)", module)
    assert set(calls) == {"g", "x"} and calls.count("x") == 1  # x: the probe
    assert "x * omlx_mlx_sigmoid<T>(x)" in RD.SWIGLU_PROBE_BODY


def test_routed_shared_gate_copies_the_probed_expression():
    body = RD.SHARED_GATE_PROBE_BODY.replace("float(x)", "float(gate[0])")
    body = body.replace("    y = g", "    const T gate = g")
    assert _squash(body) in _squash(RD.SHARED_DOWN_SOURCE)
    # The window's shared rows derive from the same source, one gate per row.
    window = body.replace("float(gate[0])", "float(gate[token])")
    assert _squash(window) in _squash(W._shared_down_window_source())
    assert RD.SHARED_DOWN_SOURCE.count("metal::precise::exp(") == 1


def test_hc_helpers_are_the_probed_ones():
    assert HCD.SILU_GATE.header == SE.metal_helper(HCD.HEADER, "hcd_sigmoid_jit")
    assert HCD.SIGMOID_GATE.header == SE.metal_helper(HCD.HEADER, "hcd_sigmoid_unary")
    assert "metal::exp(metal::abs(x))" in HCD.SILU_GATE.header
    assert "metal::precise::exp(metal::abs(x))" in HCD.SIGMOID_GATE.header
    sources = HCD.NORM_DOWN_SOURCE + HCD.UP_MIX_SOURCE
    # SiLU sites: g = jit(a) then a * g; sigmoid sites use the unary helper.
    assert sources.count("hcd_sigmoid_jit<T>(a)") == sources.count("act[") == 2
    assert sources.count("hcd_sigmoid_unary<T>(") == 3
    assert "metal::exp(" not in sources and "metal::precise::exp(" not in sources


def test_qsa_gate_epilogues_copy_the_probed_expression():
    probe = (
        _squash(merge.OUTPUT_GATE_PROBE_BODY)
        .replace("const T gate_x = x; ", "")
        .replace(" y = gate_sigmoid;", "")
    )
    for source, dtype in ((merge._SOURCE_GATED, "TO"), (indexed._COMBINE_SOURCE_GATED, "T")):
        text = _squash(source)
        if dtype == "TO":
            text = re.sub(r"\bTO\b", "T", text)
        assert probe in text
        assert text.count("metal::exp(") == 1


# --------------------------------------------------------------------------
# Call sites fall back before any kernel runs
# --------------------------------------------------------------------------
def _never(*args, **kwargs):
    raise AssertionError("a kernel ran although the served form refused it")


PRECISE_ONLY = {FAST: False, PRECISE: True}


def test_routed_decode_declines_and_counts(monkeypatch):
    sw = _switch_split()
    QN._enable_routed_decode(sw, "gate_up")
    monkeypatch.setattr(RD, "admit_split_routed_decode",
                        lambda *a: RD.RoutedDecodeAdmission(True, "eligible"))
    monkeypatch.setattr(RD, "runtime_supported", lambda: True)
    monkeypatch.setattr(RD, "split_gate_up_swiglu", _never)
    _inject(monkeypatch, RD.SWIGLU_GATE, PRECISE_ONLY)
    x = mx.zeros((1, 1, 1, 1, 32), dtype=mx.bfloat16)
    inds = mx.zeros((1, 1, 10), dtype=mx.uint32)
    assert QN._try_routed_decode(sw, x, inds, None, False) is None
    reason = "served compiled SwiGLU uses metal::precise::exp"
    assert sw.routed_decode_fallbacks == 1
    assert sw.routed_decode_last_fallback == reason
    assert sw.routed_decode_calls == 0


class _Block(dict):
    shared_folded = False
    sharding_group = None
    fused_expert_kernel_enabled = True
    fused_expert_kernel_mode = "tile4"
    training = False
    moe_router_mode = "stock"
    num_experts = 12
    top_k = 10
    norm_topk_prob = True

    def __init__(self, switch):
        super().__init__(shared_expert=object())
        self.switch_mlp = switch


def _switch_split():
    sw = QN.FusedDownSwitchGLU(32, 16, 12)
    sw.eval()
    return sw


@pytest.mark.parametrize(
    ("gate", "reason"),
    [
        ("SWIGLU_GATE", "served compiled SwiGLU uses metal::precise::exp"),
        ("SHARED_GATE_GATE", "served eager sigmoid uses metal::exp"),
    ],
)
def test_shared_fold_declines(monkeypatch, gate, reason):
    block = _Block(_switch_split())
    monkeypatch.setattr(QN, "_COMPILE_GLUE", False)
    monkeypatch.setattr(RD, "runtime_supported", lambda: True)
    monkeypatch.setattr(RD, "admit_split_routed_decode",
                        lambda *a: RD.RoutedDecodeAdmission(True, "eligible"))
    monkeypatch.setattr(QN, "_served_down_refusal", lambda *a: None)
    monkeypatch.setattr(RD, "admit_shared_fold", lambda *a: RD.RoutedDecodeAdmission(True, "eligible"))
    other = "SHARED_GATE_GATE" if gate == "SWIGLU_GATE" else "SWIGLU_GATE"
    _inject(monkeypatch, getattr(RD, other), {FAST: True, PRECISE: True})
    outcome = PRECISE_ONLY if gate == "SWIGLU_GATE" else {FAST: True, PRECISE: False}
    _inject(monkeypatch, getattr(RD, gate), outcome)
    x = mx.zeros((1, 1, 32), dtype=mx.bfloat16)
    assert QN._shared_fold_refusal(block, x, None, None) == reason
    if gate == "SHARED_GATE_GATE":
        assert not QN._shared_fold_ok(block, x)


def test_window_rows_decline(monkeypatch):
    block = _Block(_switch_split())
    monkeypatch.setattr(QN, "_COMPILE_GLUE", False)
    monkeypatch.setattr(QN, "_MOE_GATE_COMPILE", False)
    monkeypatch.setattr(RD, "admit_split_routed_decode",
                        lambda *a: RD.RoutedDecodeAdmission(True, "eligible"))
    monkeypatch.setattr(QN, "_served_down_refusal", lambda *a: None)
    monkeypatch.setattr(QN, "_base_linear_ok", lambda layer: True)
    monkeypatch.setattr(W, "admit_router_topk", lambda *a: None)
    _inject(monkeypatch, RD.SWIGLU_GATE, PRECISE_ONLY)
    x = mx.zeros((3, 1, 32), dtype=mx.bfloat16)
    assert QN._moe_rows_refusal(block, x, 3) == "served compiled SwiGLU uses metal::precise::exp"


def test_routed_candidate_uses_the_routed_gate(monkeypatch):
    monkeypatch.setattr(RD, "runtime_supported", lambda: True)
    _inject(monkeypatch, RD.SWIGLU_GATE, PRECISE_ONLY)
    assert RD.candidate_runtime_refusal() == "served compiled SwiGLU uses metal::precise::exp"


@pytest.mark.parametrize(
    ("silu", "sigmoid", "reason"),
    [
        (PRECISE_ONLY, {FAST: False, PRECISE: True}, "served compiled nn.silu uses metal::precise::exp"),
        ({FAST: True, PRECISE: False}, {FAST: True, PRECISE: False}, "served eager sigmoid uses metal::exp"),
        ({FAST: True, PRECISE: False}, RuntimeError("x"), "served eager sigmoid matches no kernel form"),
    ],
)
def test_hc_decode_declines_before_any_kernel(monkeypatch, silu, sigmoid, reason):
    from test_qwen4_hc_decode import _module, _x

    HCD.reset_for_tests()
    monkeypatch.setattr(HCD, "runtime_supported", lambda: True)
    monkeypatch.setattr(HCD, "hc_decode_launch", _never)
    _inject(monkeypatch, HCD.SILU_GATE, silu)
    _inject(monkeypatch, HCD.SIGMOID_GATE, sigmoid)
    m = _module()
    x = _x(3)
    want = m(x)
    was = HCD.set_hc_decode_enabled(True)
    try:
        got = m(x)
    finally:
        HCD.set_hc_decode_enabled(was)
    for a, b in zip(got, want):
        assert mx.array_equal(a, b).item()
    assert HCD.hc_decode_status()["last_decline"] == reason
    HCD.reset_for_tests()


def _partials():
    rng = np.random.default_rng(7)
    m = mx.array(rng.normal(0.0, 4.0, (1, 1, 2, 128)).astype(np.float32))
    l = mx.array(np.exp(rng.normal(0.0, 1.0, (1, 1, 2, 128))).astype(np.float32))
    o = mx.array(rng.normal(0.0, 1.0, (1, 1, 2, 128, 32)).astype(np.float32))
    gate = mx.array(rng.normal(0.0, 3.0, (1, 2, 32)).astype(np.float32)).astype(mx.bfloat16)
    return m, l, o, gate


def test_qsa_fused_merge_declines_the_gate_epilogue(monkeypatch):
    m, l, o, gate = _partials()
    seen = []

    def fused(m, l, o, *, output_dtype, output_gate=None):
        seen.append(output_gate)
        return merge.mlx_sequential_merge(m, l, o, output_dtype=output_dtype)

    monkeypatch.setenv("MLX_QWEN4_QSA_INDEXED_FUSED_MERGE", "1")
    monkeypatch.setenv("MLX_QWEN4_QSA_INDEXED_FUSED_GATE", "1")
    monkeypatch.setattr(merge, "fused_merge_available", lambda: True)
    monkeypatch.setattr(merge, "_fused_merge", fused)
    _inject(monkeypatch, merge.OUTPUT_GATE_GATE, PRECISE_ONLY)
    merge.fused_merge_status(reset=True)
    got = merge.combine_indexed_partials(m, l, o, output_dtype=mx.bfloat16, output_gate=gate)
    want = merge.mlx_apply_output_gate(
        merge.mlx_sequential_merge(m, l, o, output_dtype=mx.bfloat16), gate
    )
    assert seen == [None]
    assert mx.array_equal(got, want).item()
    status = merge.fused_merge_status(reset=True)
    assert status["gate_refusals"] == 1
    assert status["gate_last_refusal"] == "served eager sigmoid uses metal::precise::exp"


def test_qsa_native_pass2_declines_the_gate_epilogue(monkeypatch):
    m, l, o, gate = _partials()
    gated_flags = []
    sentinel = mx.ones((1, 1, 2, 32), dtype=mx.bfloat16)

    def combine_kernel(gated=False):
        gated_flags.append(gated)
        return lambda **kw: [sentinel]

    monkeypatch.delenv("MLX_QWEN4_QSA_INDEXED_FUSED_MERGE", raising=False)
    monkeypatch.setattr(indexed, "_combine_kernel", combine_kernel)
    _inject(monkeypatch, merge.OUTPUT_GATE_GATE, PRECISE_ONLY)
    merge.fused_merge_status(reset=True)
    counter = mx.array([3], dtype=mx.uint32)
    (got, returned) = indexed._combine_sdpa_partials(
        m, l, o, counter, output_dtype=mx.bfloat16, output_gate=gate
    )
    assert returned is counter
    assert gated_flags == [False]
    assert mx.array_equal(got, merge.mlx_apply_output_gate(sentinel, gate)).item()
    status = merge.fused_merge_status(reset=True)
    assert status["gate_refusals"] == 1 and not status["gate_engaged"]


def test_qsa_gate_accepted_keeps_the_gated_kernel(monkeypatch):
    m, l, o, gate = _partials()
    gated_flags = []
    sentinel = mx.ones((1, 1, 2, 32), dtype=mx.bfloat16)

    def combine_kernel(gated=False):
        gated_flags.append(gated)
        return lambda **kw: [sentinel]

    monkeypatch.delenv("MLX_QWEN4_QSA_INDEXED_FUSED_MERGE", raising=False)
    monkeypatch.setattr(indexed, "_combine_kernel", combine_kernel)
    _inject(monkeypatch, merge.OUTPUT_GATE_GATE, {FAST: True, PRECISE: False})
    (got, _) = indexed._combine_sdpa_partials(
        m, l, o, mx.array([0], dtype=mx.uint32), output_dtype=mx.bfloat16, output_gate=gate
    )
    assert gated_flags == [True] and got is sentinel


# --------------------------------------------------------------------------
# Real probes (Metal, opt-in)
# --------------------------------------------------------------------------
@pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="set MLX2_RUN_METAL_TESTS=1 (under the GPU lock) for the real-Metal probes",
)
@pytest.mark.parametrize(
    ("gate", "dtype"),
    [
        (RD.SWIGLU_GATE, mx.bfloat16),
        (RD.SHARED_GATE_GATE, mx.bfloat16),
        (HCD.SILU_GATE, mx.bfloat16),
        (HCD.SIGMOID_GATE, mx.bfloat16),
        (merge.OUTPUT_GATE_GATE, mx.bfloat16),
        (merge.OUTPUT_GATE_GATE, mx.float16),
        (merge.OUTPUT_GATE_GATE, mx.float32),
    ],
    ids=lambda v: getattr(v, "name", str(v)),
)
def test_metal_probe_accepts_the_kernel_form_on_the_pinned_build(gate, dtype):
    with mx.stream(mx.gpu):
        matches = gate.probe(dtype)
    print(f"{gate.name} {dtype}: {matches}")
    assert matches[gate.kernel_exp], matches
