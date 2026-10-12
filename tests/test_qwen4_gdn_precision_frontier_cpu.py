"""Qwen4 GDN precision frontier: CPU graph and host-source contracts only."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mlx2.runtime.models import gdn_state, qwen3_5, qwen4_exp
from mlx2.runtime.models import qwen4_fused_gdn as decode
from mlx2.runtime.models import qwen4_fused_gdn_verify as verify
from mlx2.runtime.models.gated_delta import compute_g, compute_g_float32_inputs


def test_qwen4_decay_widens_before_addition_and_softplus_on_cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        a = mx.array([0.1234], dtype=mx.bfloat16)
        dt_bias = mx.array([0.5678], dtype=mx.bfloat16)
        a_log = mx.array([-5.0], dtype=mx.float32)
        legacy = compute_g(a_log, a, dt_bias)
        actual = compute_g_float32_inputs(a_log, a, dt_bias)
        expected = compute_g(a_log, a.astype(mx.float32), dt_bias.astype(mx.float32))
        mx.eval(legacy, actual, expected)
        assert actual.dtype == mx.float32
        assert np.array_equal(np.asarray(actual), np.asarray(expected))
        # This witness crosses the old bf16 softplus boundary, so the guard is
        # able to go red if Qwen4 silently returns to that boundary.
        assert not np.array_equal(np.asarray(actual), np.asarray(legacy))
    finally:
        mx.set_default_device(previous)


def test_only_qwen4_selects_the_float32_decay_contract(monkeypatch):
    qwen4_call = {}
    qwen35_call = {}

    def q4_stub(*args, **kwargs):
        qwen4_call.update(kwargs)
        return ("qwen4", "state")

    def q35_stub(*args, **kwargs):
        qwen35_call.update(kwargs)
        return ("qwen35", "state")

    monkeypatch.setattr(qwen4_exp, "gated_delta_update", q4_stub)
    monkeypatch.setattr(qwen3_5, "gated_delta_update", q35_stub)
    layer = SimpleNamespace(A_log=object(), dt_bias=object())
    args = (object(),) * 7 + (False,)
    assert qwen4_exp.GatedDeltaNet._gated_delta_update(layer, *args) == (
        "qwen4",
        "state",
    )
    assert qwen3_5.GatedDeltaNet._gated_delta_update(layer, *args) == (
        "qwen35",
        "state",
    )
    assert qwen4_call["decay_input_float32"] is True
    assert "decay_input_float32" not in qwen35_call


def test_qwen4_fused_decode_verify_and_replay_producers_use_float32_decay():
    qwen4_sp = "float sp = mlx_softplus_fast<float>(av);"
    qwen4_add = "float av = float(a["
    state_update = "st[j][i] = st[j][i] * shr[2];"
    for source in (
        decode._SOURCE,
        decode._SOURCE_BATCH,
        decode._SOURCE_OUTPROJ,
        verify._SOURCE,
        verify._CATCHUP_SOURCE,
        verify._REPLAY_SOURCE,
    ):
        assert qwen4_add in source
        assert qwen4_sp in source
        assert source.index(qwen4_sp) < source.index(state_update)

    # The fp16 storage class derives from the corrected producer and still
    # rounds only the state after each token; it never narrows the decay gate.
    for source in (decode._SOURCE_ST16, verify._SOURCE_ST16):
        assert qwen4_sp in source
        assert "float(static_cast<half>(st[j][i]))" in source


def test_qwen35_fused_family_keeps_its_existing_decay_boundary():
    qwen35_add = "T av = a["
    qwen35_sp = "T sp = mlx_softplus_fast(av);"
    for source in (
        decode._SOURCE,
        verify._QWEN35_SOURCE,
        verify._QWEN35_CATCHUP_SOURCE,
    ):
        assert qwen35_add in source
        assert qwen35_sp in source


def test_state_storage_and_namespace_contracts_are_unchanged():
    base = "qwen4-exp-layer-segments-v1"
    assert gdn_state.STATE_DTYPES == {"float32": mx.float32, "float16": mx.float16}
    assert gdn_state.layout_with_state_dtype(base, "float32") == base
    assert (
        gdn_state.layout_with_state_dtype(base, "float16")
        == base + ":gdn-state-fp16-v1"
    )
