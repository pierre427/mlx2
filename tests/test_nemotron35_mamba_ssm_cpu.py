import copy
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models.cache import ArraysCache, BatchKVCache
from mlx2.runtime.models.nemotron_h import Model
from mlx2.runtime.models.nemotron_mamba_ssm import (
    admit_fused_mamba_ssm_gate_short,
    admit_fused_mamba_ssm_short,
)


def test_mtp_verify_backbone_uses_selected_layerwise_short_segment():
    tokens = SimpleNamespace(shape=(1, 4))
    hidden = object()
    model = SimpleNamespace(
        _short_segment_policy="layerwise_short_exact",
        _short_segment_counts={"refused": 0},
        _short_segment_refusal_reasons={},
        _short_segment_refusal=lambda _tokens, _cache: None,
        _layerwise_short_hidden=lambda _tokens, _cache, _captures: (hidden, []),
    )

    assert Model.mtp_verify_backbone(model, tokens, []) == (hidden, hidden)


@pytest.mark.parametrize("pad", [0, 2])
def test_layerwise_short_segment_matches_per_token_reference_with_left_padding(pad):
    from tests.test_nemotron_external_taps_cpu import tiny_model

    mx.set_default_device(mx.cpu)
    model = tiny_model()
    for layer in model.layers:
        if layer.block_type == "M":
            layer.mixer.set_conv_policy("fused_short")
            layer.mixer.set_ssm_policy("fused_short")
            layer.mixer.set_projection_policy("short_row_exact")
    cache = [
        ArraysCache(size=2, left_padding=[pad] if pad else None)
        if layer.block_type == "M"
        else BatchKVCache(left_padding=[pad])
        for layer in model.layers
        if layer.block_type in ("M", "*")
    ]
    model(mx.array([[0] * pad + [1, 2, 3, 4, 5]]), cache)
    mx.eval([entry.state for entry in cache])
    inputs = mx.array([[6, 7, 8, 9]])
    logits = {}
    for policy in ("auto", "layerwise_short_exact"):
        branch = copy.deepcopy(cache)
        for entry in branch:
            entry.start_speculation()
        model.set_short_segment_policy(policy)
        output, _taps = model.forward_with_taps(inputs, branch, (0, 1, 4))
        logits[policy] = np.array(output)
    status = model.short_segment_status()
    # Padded rows need the per-token path's mask; the layerwise path is
    # unmasked and must refuse rather than attend to pad keys.
    assert status["counts"]["layerwise_short_exact"] == (0 if pad else 1)
    assert status["refusal_reasons"].get("left padding", 0) == (1 if pad else 0)
    np.testing.assert_allclose(
        logits["layerwise_short_exact"], logits["auto"], atol=1e-6
    )

    # Pending recurrent left padding alone also keeps the masked path.
    model.set_short_segment_policy("layerwise_short_exact")
    recurrent_only = [
        ArraysCache(size=2, left_padding=[pad] if pad else None)
        if layer.block_type == "M"
        else BatchKVCache(left_padding=[0])
        for layer in model.layers
        if layer.block_type in ("M", "*")
    ]
    for entry in recurrent_only:
        entry.start_speculation()
    assert model._short_segment_refusal(inputs, recurrent_only) == (
        "left padding" if pad else None
    )


def _tensor(shape, dtype):
    return SimpleNamespace(shape=shape, dtype=dtype)


def _admit(**overrides):
    steps = overrides.pop("steps", 4)
    values = {
        "x": _tensor((1, steps, 128, 64), mx.bfloat16),
        "B": _tensor((1, steps, 8, 128), mx.bfloat16),
        "C": _tensor((1, steps, 8, 128), mx.bfloat16),
        "dt": _tensor((1, steps, 128), mx.bfloat16),
        "A_log": _tensor((128,), mx.float32),
        "D": _tensor((128,), mx.bfloat16),
        "dt_bias": _tensor((128,), mx.float32),
        "state": _tensor((1, 128, 64, 128), mx.float32),
        "mask": None,
        "lengths": None,
    }
    values.update(overrides)
    return admit_fused_mamba_ssm_short(**values)


def _admit_gate(**overrides):
    steps = overrides.pop("steps", 4)
    values = {
        "x": _tensor((1, steps, 128, 64), mx.bfloat16),
        "B": _tensor((1, steps, 8, 128), mx.bfloat16),
        "C": _tensor((1, steps, 8, 128), mx.bfloat16),
        "dt": _tensor((1, steps, 128), mx.bfloat16),
        "A_log": _tensor((128,), mx.bfloat16),
        "D": _tensor((128,), mx.bfloat16),
        "dt_bias": _tensor((128,), mx.bfloat16),
        "state": _tensor((1, 128, 64, 128), mx.float32),
        "gate": _tensor((1, steps, 8192), mx.bfloat16),
        "mask": None,
        "lengths": None,
    }
    values.update(overrides)
    return admit_fused_mamba_ssm_gate_short(**values)


def test_super_short_ssm_geometry_is_admitted():
    assert _admit(steps=1).accepted
    assert _admit(steps=2).accepted
    assert _admit(steps=8).accepted
    assert _admit(steps=16).accepted


def test_decode_prefill_and_wrong_geometry_fail_closed():
    assert _admit(steps=0).reason.startswith("x shape")
    assert _admit(steps=17).reason.startswith("x shape")
    assert _admit(x=_tensor((1, 4, 64, 64), mx.bfloat16)).reason.startswith(
        "head geometry"
    )
    assert _admit(mask=object()).reason == "masked or padded input"


def test_dtype_contract_is_exact():
    assert (
        _admit(dt=_tensor((1, 4, 128), mx.float32)).reason
        == "requires bf16 activations"
    )
    assert (
        _admit(A_log=_tensor((128,), mx.float16)).reason == "requires bf16/fp32 A_log"
    )
    assert (
        _admit(dt_bias=_tensor((128,), mx.float16)).reason
        == "requires bf16/fp32 dt_bias"
    )
    assert (
        _admit(state=_tensor((1, 128, 64, 128), mx.float16)).reason
        == "requires fp32 recurrent state"
    )


def test_live_checkpoint_bf16_parameters_are_admitted():
    assert _admit(
        A_log=_tensor((128,), mx.bfloat16),
        dt_bias=_tensor((128,), mx.bfloat16),
    ).accepted


def test_short_ssm_gate_accepts_served_geometry():
    assert _admit_gate(steps=1).accepted
    assert _admit_gate(steps=8).accepted
    assert _admit_gate(steps=16).accepted


def test_short_ssm_gate_rejects_wrong_gate_geometry_and_dtype():
    assert (
        _admit_gate(gate=_tensor((1, 4, 8191), mx.bfloat16)).reason
        == "gate shape (1, 4, 8191)"
    )
    assert (
        _admit_gate(gate=_tensor((1, 4, 8192), mx.float32)).reason
        == "requires bf16 gate"
    )
