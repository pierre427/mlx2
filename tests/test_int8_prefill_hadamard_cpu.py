"""Host/component gates for Splash-style rotated approximate int8 prefill."""

import mlx.core as mx
import numpy as np
import pytest
from test_int8_prefill import _Model

from mlx2.runtime import int8_prefill as ip

mx.set_default_device(mx.cpu)


def policy(**overrides):
    return ip.Int8PrefillPolicy(
        enabled=True, scope="mlp", hadamard_rotation=True, **overrides
    )


def test_policy_is_default_off_revision_bound_and_fail_closed():
    assert not ip.Int8PrefillPolicy(enabled=True).hadamard_rotation
    rotated = policy()
    assert rotated.as_dict()["hadamard_rotation"] is True
    assert rotated.revision != ip.Int8PrefillPolicy(enabled=True).revision
    with pytest.raises(ValueError, match="requires an enabled"):
        ip.Int8PrefillPolicy(hadamard_rotation=True)
    with pytest.raises(ValueError, match="mlp scope"):
        ip.Int8PrefillPolicy(
            enabled=True, scope="all", hadamard_rotation=True
        )
    with pytest.raises(ValueError, match="in-place"):
        ip.Int8PrefillPolicy(
            enabled=True,
            hadamard_rotation=True,
            q8_inplace=True,
        )


@pytest.mark.parametrize("block", [128, 512])
def test_signed_block_rotation_is_orthonormal(block):
    rng = np.random.default_rng(block)
    left = mx.array(rng.normal(size=(3, block * 2)).astype(np.float32))
    right = mx.array(rng.normal(size=(5, block * 2)).astype(np.float32))
    before = np.asarray(left @ right.T)
    after = np.asarray(
        ip.rotate_quantizer_blocks(left, block)
        @ ip.rotate_quantizer_blocks(right, block).T
    )
    np.testing.assert_allclose(after, before, rtol=2e-5, atol=2e-5)
    assert np.array_equal(
        np.asarray(ip._rotation_signs(block)),
        np.asarray(ip._rotation_signs(block)),
    )


def test_rotation_geometry_matches_gate_up_and_down():
    assert ip._rotation_block("layers.0.mlp.gate_proj") == 128
    assert ip._rotation_block("layers.0.mlp.up_proj") == 128
    assert ip._rotation_block("layers.0.mlp.down_proj") == 512
    with pytest.raises(ip.Int8PrefillError, match="no declared rotation geometry"):
        ip._rotation_block("layers.0.mlp.fc1")
    with pytest.raises(ip.Int8PrefillError, match="does not divide"):
        ip.rotate_quantizer_blocks(mx.ones((2, 384)), 512)


def test_rotated_weight_and_activation_builds_are_paired(monkeypatch):
    monkeypatch.setattr(ip, "require_supported_device", lambda: "fake-M5")
    monkeypatch.setattr(
        ip,
        "_quantize_rows",
        lambda value: (value, mx.ones((value.shape[0],), dtype=mx.float32)),
    )
    model = _Model(bits=4)
    handle = ip.apply(model, policy())
    try:
        gate = model.layers[0].mlp.gate_proj
        bound = handle._bound[id(gate)]
        weight, _scales = handle._build(bound)
        dense = mx.dequantize(
            gate["weight"],
            gate["scales"],
            gate["biases"],
            group_size=bound.spec.group_size,
            bits=bound.spec.bits,
        )
        np.testing.assert_allclose(
            np.asarray(weight.astype(mx.float32)),
            np.asarray(ip.rotate_quantizer_blocks(dense, 128).astype(mx.float32)),
            rtol=1e-5,
            atol=1e-6,
        )
        activation = mx.arange(2 * bound.spec.k, dtype=mx.float32).reshape(
            2, bound.spec.k
        )
        rotated, _ = handle._activation(activation, bound.spec.k, block=128)
        np.testing.assert_allclose(
            np.asarray(rotated),
            np.asarray(ip.rotate_quantizer_blocks(activation, 128)),
        )
        assert handle._cache_mode(bound) == "resident"
        receipt = handle.receipt()["hadamard_rotation"]
        assert receipt["input_block"] == 128
        assert receipt["intermediate_block"] == 512
        assert receipt["build_transient_bytes_max"] > 0
        assert handle.weight_copy_bytes() == (
            handle.resident_estimate_bytes()
            + handle.rotation_build_transient_bytes_max()
        )
    finally:
        ip.remove(handle)
