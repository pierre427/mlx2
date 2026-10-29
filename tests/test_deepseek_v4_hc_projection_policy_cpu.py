"""Explicit HC projection model policy seams; mock dispatch on CPU only."""

import dataclasses
import sys
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)
from test_deepseek_v4_forward_cpu import SmallWeights, config

from mlx2.adapters.deepseek_v4_gguf import validate_execution_policy
from mlx2.runtime.models.deepseek_v4 import DeepSeekV4Model, HyperConnection


def test_adapter_defaults_reference_and_refuses_unrecognized_or_nonstring():
    assert validate_execution_policy({})["deepseek_hc_projection"] == "reference"
    assert (
        validate_execution_policy({"deepseek_hc_projection": "fixed_fp32"})[
            "deepseek_hc_projection"
        ]
        == "fixed_fp32"
    )
    for bad in ("auto", "fused_fp16", True, 0):
        with pytest.raises(ValueError):
            validate_execution_policy({"deepseek_hc_projection": bad})


def test_model_policy_geometry_fail_closed_before_weight_access():
    c = config((0,))
    provider = SimpleNamespace(
        array=lambda *_: pytest.fail("metadata validation before loading")
    )
    with pytest.raises(ValueError, match="projection arithmetic"):
        DeepSeekV4Model(c, provider, identity="cpu", hc_projection_policy="unknown")
    for bad in (
        dataclasses.replace(c, hc_mult=2),
        dataclasses.replace(c, hidden_size=7),
    ):
        with pytest.raises(ValueError, match="HC4 and K"):
            DeepSeekV4Model(
                bad, provider, identity="cpu", hc_projection_policy="fixed_fp32"
            )


def test_hc_weight_shape_refusal_before_kernel_dispatch():
    c = config((0,))
    w = SmallWeights(c)
    w.data["blk.0.hc_attn_fn.weight"] = mx.zeros((23, 32))
    with pytest.raises(ValueError, match="matrix geometry"):
        HyperConnection(c, w, "blk.0.hc_attn", projection_policy="fixed_fp32")


def test_mock_projection_uses_cached_fp32_weights_and_preserves_other_math(monkeypatch):
    calls = []

    def project(x, w):
        assert x.dtype == w.dtype == mx.float32
        calls.append((x, w))
        return x @ w.T

    monkeypatch.setitem(
        sys.modules,
        "mlx2.runtime.models.deepseek_v4_hc_projection",
        SimpleNamespace(project=project),
    )
    c = config((4, 128))
    w = SmallWeights(c)
    reference = DeepSeekV4Model(c, w, identity="same")
    candidate = DeepSeekV4Model(
        c, w, identity="same", hc_projection_policy="fixed_fp32"
    )
    assert (
        reference.identity == "same"
        and candidate.identity == "same:hc_projection=fixed_fp32"
    )
    for layer in candidate.layers:
        assert (
            layer.attn_hc.projection_policy
            == layer.ffn_hc.projection_policy
            == "fixed_fp32"
        )
    ids = mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32)
    a = reference.make_cache()
    b = candidate.make_cache()
    ao = reference(ids, a)
    bo = candidate(ids, b)
    mx.eval(ao, bo, [x.state for x in a + b])
    np.testing.assert_array_equal(np.array(ao), np.array(bo))
    assert len(calls) == 2 * c.num_hidden_layers + 1
    expected = {
        id(hc.fn_fp32)
        for hc in [candidate.hc_head]
        + [v for l in candidate.layers for v in (l.attn_hc, l.ffn_hc)]
    }
    assert {id(w) for _, w in calls} == expected
    # Changed arithmetic identity cannot reuse an ordinary cache silently.
    with pytest.raises(ValueError, match="cache"):
        candidate(ids, a)
    assert all(
        hc.fn is w.data[prefix + "_fn.weight"]
        for hc, prefix in [(candidate.hc_head, "output_hc")]
    )
