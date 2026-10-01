"""Fused GDN kernels decline when MLX serves a different SiLU (omlx#4122).

The kernels reproduce compiled ``nn.silu`` with the ``metal::exp`` sigmoid.
MLX #4461 (v0.32.3) serves ``metal::precise::exp`` instead, so on such a
build the fused paths must fall back to the reference.  CPU only: the probe
itself needs Metal and runs only in the opt-in test at the end; everything
else injects its outcome.
"""

import os
import re

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import qwen4_fused_gdn as fused
from mlx2.runtime.models import qwen4_fused_gdn_prefill as prefill
from mlx2.runtime.models import qwen4_fused_gdn_verify as verify


def test_all_16bit_encodings_cover_every_bit_pattern():
    for dtype, nans in ((mx.bfloat16, 254), (mx.float16, 2046)):
        values = fused.all_16bit_encodings(dtype)
        assert values.shape == (1 << 16,) and values.dtype == dtype
        assert (np.array(values.view(mx.uint16)) == np.arange(1 << 16)).all()
        assert int(mx.isnan(values).sum().item()) == nans
        assert int(mx.isinf(values).sum().item()) == 2
    with pytest.raises(ValueError):
        fused.all_16bit_encodings(mx.float32)


def test_same_bits_is_exact_except_that_nan_matches_nan():
    values = fused.all_16bit_encodings(mx.bfloat16)
    assert fused.same_bits(values, values)
    quiet = mx.array([0x7FC0, 0xFFC1], dtype=mx.uint16).view(mx.bfloat16)
    other = mx.array([0x7FC1, 0x7FC0], dtype=mx.uint16).view(mx.bfloat16)
    assert fused.same_bits(quiet, other)
    # One ULP on one finite value is a mismatch; allclose would accept it.
    bits = np.array(values.view(mx.uint16))
    bits[1000] += 1
    assert not fused.same_bits(values, mx.array(bits).view(mx.bfloat16))
    assert not fused.same_bits(mx.array([0.0]), mx.array([-0.0]))
    wide = mx.array([1.0, float("nan")])
    assert fused.same_bits(wide, mx.array([1.0, float("nan")]))
    assert not fused.same_bits(wide, wide.astype(mx.bfloat16))


def test_selector_takes_the_matching_spelling_or_none():
    both = {"metal::exp": True, "metal::precise::exp": True}
    assert fused.select_silu_exp(both) == "metal::exp"
    precise = {"metal::exp": False, "metal::precise::exp": True}
    assert fused.select_silu_exp(precise) == "metal::precise::exp"
    assert fused.select_silu_exp({"metal::exp": False}) is None
    assert fused.select_silu_exp({}) is None


@pytest.fixture
def probe(monkeypatch):
    """Reset the per-process probe and install an injected outcome."""
    monkeypatch.setattr(fused, "_SERVED_SILU_COMPLETE", False)
    monkeypatch.setattr(fused, "_SERVED_SILU_EXP", None)
    calls = []

    def install(outcome):
        def injected():
            calls.append(outcome)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        monkeypatch.setattr(fused, "_probe_served_silu", injected)
        return calls

    return install


@pytest.mark.parametrize(
    ("outcome", "refusal"),
    [
        ({"metal::exp": True, "metal::precise::exp": False}, None),
        (
            {"metal::exp": False, "metal::precise::exp": True},
            "served SiLU uses metal::precise::exp",
        ),
        ({"metal::exp": False, "metal::precise::exp": False},
         "served SiLU matches no kernel form"),
        (RuntimeError("kernel failed to compile"), "served SiLU matches no kernel form"),
    ],
)
def test_refusal_follows_one_probe_per_process(probe, outcome, refusal):
    calls = probe(outcome)
    assert fused.served_silu_refusal() == refusal
    assert fused.served_silu_refusal() == refusal
    assert len(calls) == 1


class _Cache(list):
    speculating = False

    def rollback_spans(self, steps, mask):
        return ()

    def record_rollback(self, *args, **kwargs):
        pass

    def advance(self, steps):
        pass


def _accept(*args, **kwargs):
    return fused.FusedGdnAdmission(True, "eligible")


def _never(*args, **kwargs):
    raise AssertionError("a kernel ran although the served SiLU refused it")


def _inputs(width):
    return (mx.zeros((1, width, 8)), mx.zeros((1, width, 8)),
            mx.zeros((1, width, 4)), mx.zeros((1, width, 4)))


REFUSAL = "served SiLU uses metal::precise::exp"


def test_qwen36_fused_decode_declines_before_any_kernel(monkeypatch):
    from mlx2.runtime.models import qwen36_35b
    from mlx2.runtime.models.qwen3_5 import TextModelArgs

    args = TextModelArgs(
        model_type="qwen3_5_moe_text", hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, vocab_size=128, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=3, full_attention_interval=4,
        partial_rotary_factor=0.5, rope_parameters=None,
        max_position_embeddings=128,
    )
    layer = qwen36_35b.GatedDeltaNet(args)
    layer.set_fused_gdn_decode_mode("fused")
    monkeypatch.setattr(qwen36_35b, "admit_qwen4_fused_gdn_decode", _accept)
    monkeypatch.setattr(qwen36_35b, "fused_gdn_runtime_supported", lambda: True)
    monkeypatch.setattr(qwen36_35b, "served_silu_refusal", lambda: REFUSAL)
    monkeypatch.setattr(qwen36_35b, "probe_qwen4_fused_gdn_decode", _never)
    monkeypatch.setattr(qwen36_35b, "qwen4_fused_gdn_decode", _never)
    cache = _Cache([mx.zeros((1, 2, 8)), mx.zeros((1, 4, 8, 8))])
    assert layer._try_fused_decode(*_inputs(1), None, cache) is None
    assert layer.fused_gdn_decode_last_fallback == REFUSAL
    assert layer.fused_gdn_decode_fallback_reasons == {REFUSAL: 1}


def test_flash_next_fused_paths_decline_before_any_kernel(monkeypatch):
    from mlx2.runtime.models import qwen4_exp
    from test_apc_hits_hybrid_gdn_self_mtp import tiny_qwen4_mtp

    model, _vocab = tiny_qwen4_mtp()
    layer = next(
        layer.linear_attn
        for layer in model.language_model.model.layers
        if getattr(layer, "linear_attn", None) is not None
    )
    layer.set_fused_gdn_decode_mode("fused")
    layer.set_fused_gdn_verify_mode("fused")
    layer.set_fused_gdn_prefill_mode("fused")
    for name in ("admit_qwen4_fused_gdn_decode", "admit_qwen4_fused_gdn_verify"):
        monkeypatch.setattr(qwen4_exp, name, _accept)
    monkeypatch.setattr(prefill, "admit_qwen4_fused_gdn_prefill", _accept)
    monkeypatch.setattr(qwen4_exp, "fused_gdn_runtime_supported", lambda: True)
    monkeypatch.setattr(prefill, "runtime_supported", lambda: True)
    monkeypatch.setattr(qwen4_exp, "served_silu_refusal", lambda: REFUSAL)
    for name in (
        "probe_qwen4_fused_gdn_decode", "probe_qwen4_fused_gdn_verify",
        "probe_qwen4_fused_gdn_replay_verify", "probe_qwen4_fused_gdn_catchup",
        "qwen4_fused_gdn_decode", "qwen4_fused_gdn_verify",
    ):
        monkeypatch.setattr(qwen4_exp, name, _never)
    monkeypatch.setattr(prefill, "qwen4_gdn_prefill_prework", _never)
    state = [mx.zeros((1, 3, 8)), mx.zeros((1, 4, 8, 8))]

    assert layer._try_fused_decode(*_inputs(1), None, _Cache(state)) is None
    assert layer.fused_gdn_decode_last_fallback == REFUSAL
    assert layer._try_fused_verify(*_inputs(3), None, _Cache(state)) is None
    assert layer.fused_gdn_verify_last_fallback == REFUSAL
    assert layer._try_fused_prefill(*_inputs(64), None, _Cache(state)) is None
    assert layer.fused_gdn_prefill_last_fallback == REFUSAL


def _conv_silu_spellings(source):
    """Exp spellings of the conv SiLU sites, which copy compiled nn.silu."""
    return re.findall(r"T\(1\) / \(T\(1\) \+ (metal::(?:precise::)?exp)\(metal::abs\(conv\)\)\)",
                      source)


def test_kernel_sources_keep_one_silu_contract():
    """The probe checks ``mlx_sigmoid_fast``; every conv SiLU site must use it
    (or its exact spelling), and the eager-sigmoid sites stay precise."""
    header = fused._HEADER
    fast = header[header.index("mlx_sigmoid_fast(U x)"):]
    fast = fast[: fast.index("}")]
    assert f"{fused.KERNEL_SILU_EXP}(metal::abs(x))" in fast
    for source in (fused._SOURCE, fused._SOURCE_OUTPROJ, verify._SOURCE):
        assert "T sl = xb * mlx_sigmoid_fast(xb);" in source
        # beta: eager mx.sigmoid, precise on every MLX build
        assert "mlx_sigmoid_precise(b[" in source
    assert _conv_silu_spellings(prefill.prefill_prework_source()) == [
        fused.KERNEL_SILU_EXP
    ]
    # Qwen4's output gate is eager mx.sigmoid: precise on every MLX build.
    assert "metal::precise::exp(metal::abs(zv))" in prefill.prefill_norm_gate_source()


@pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="set MLX2_RUN_METAL_TESTS=1 (under the GPU lock) for the real-Metal probe",
)
def test_metal_probe_selects_the_kernel_form_on_the_pinned_build(monkeypatch):
    monkeypatch.setattr(fused, "_SERVED_SILU_COMPLETE", False)
    monkeypatch.setattr(fused, "_SERVED_SILU_EXP", None)
    with mx.stream(mx.gpu):
        matches = fused._probe_served_silu()
    # MLX without #4461 serves the fast sigmoid; with it, the precise one.
    assert fused.select_silu_exp(matches) in fused._SILU_EXP_CANDIDATES
    assert matches["metal::exp"] != matches["metal::precise::exp"]


def test_device_fault_in_served_silu_probe_is_raised_not_cached(probe):
    calls = probe(
        RuntimeError(
            "[METAL] Command buffer execution failed: Caused GPU Timeout Error "
            "(00000002:kIOGPUCommandBufferCallbackErrorTimeout)"
        )
    )
    with pytest.raises(RuntimeError):
        fused.served_silu_exp()
    assert not fused._SERVED_SILU_COMPLETE
    with pytest.raises(RuntimeError):
        fused.served_silu_exp()
    assert len(calls) == 2

