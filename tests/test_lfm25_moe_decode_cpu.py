"""Structural and ordinary-law checks; synthetic checkpoint, no GPU run."""
from types import SimpleNamespace as NS

import mlx.core as mx
import pytest

from mlx2.adapters.lfm25_moe import LFM25MoEAdapter
from mlx2.runtime.models import lfm25_moe_decode as candidate
from mlx2.runtime.models.lfm25_moe import SparseMoE


def block(*, bias=True, normalize=True, scale=1.5):
    result = SparseMoE(NS(num_experts=32, num_experts_per_tok=4,
                          norm_topk_prob=normalize, use_expert_bias=bias,
                          routed_scaling_factor=scale, hidden_size=128,
                          moe_intermediate_size=128))
    result.eval()
    return result


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_reference_is_the_local_sigmoid_law_not_the_donor_softmax(bias, normalize):
    model = block(bias=bias, normalize=normalize)
    if bias:
        model.expert_bias = mx.arange(32, dtype=mx.float32) / 8
    x = mx.random.normal((1, 1, 128), key=mx.random.key(8)).astype(mx.bfloat16)
    probabilities = mx.sigmoid(model.gate(x))
    got = candidate.reference_route(model, probabilities)
    expected = model.routing(x)
    for a, b in zip(got, expected):
        assert mx.array_equal(a, b).item()
    # Ordinary argpartition order is authoritative, not a sorted top-k.


def test_explicit_policy_and_numerical_namespace():
    assert candidate.parse_policy(None) == "off"
    for invalid in (True, 1, "auto", "", {}, []):
        with pytest.raises(ValueError):
            candidate.parse_policy(invalid)
    adapter = LFM25MoEAdapter.__new__(LFM25MoEAdapter)
    adapter.moe_decode_mode = "off"
    assert adapter.execution_numerics_contract() is None
    adapter.moe_decode_mode = candidate.MODE
    receipt = adapter.execution_numerics_contract()
    assert receipt["shader_sha256"] == candidate.identity()
    assert receipt["qualified"] is False


def test_selected_fusion_splits_the_real_apcv2_semantic_namespace():
    from mlx2.serving import adapter_numerical_contract, apc_semantic_namespace

    adapter = LFM25MoEAdapter.__new__(LFM25MoEAdapter)
    adapter.moe_decode_mode = "off"
    assert apc_semantic_namespace("tenant", adapter_execution_numerics=
        adapter_numerical_contract(adapter)) == "tenant"
    adapter.moe_decode_mode = candidate.MODE
    selected = apc_semantic_namespace("tenant", adapter_execution_numerics=
        adapter_numerical_contract(adapter))
    assert selected != "tenant"
    assert selected == apc_semantic_namespace("tenant", adapter_execution_numerics=
        adapter_numerical_contract(adapter))


def test_off_metal_is_a_visible_refusal_and_does_not_mutate_reference():
    model = block()
    x = mx.ones((1, 1, 128), mx.bfloat16)
    assert candidate.admission(model, x) == "Metal unavailable"
    with pytest.raises(ValueError, match="Metal"):
        candidate.install(NS(named_modules=lambda: [("block", model)]))
    assert not hasattr(model, "_mlx2_decode_fusion")


def test_law_and_bias_replacement_decline_without_kernel_execution():
    model = block()
    handle = candidate.Handle()
    handle._laws[id(model)] = candidate._law(model)
    model.expert_bias = mx.ones((32,), mx.float32)
    assert handle.forward(model, mx.ones((1, 1, 128), mx.bfloat16)) is None
    assert handle.status()["counts"]["declined:block law or bias changed"] == 1
    assert not handle.status()["observed_used"]


def test_unsupported_combine_dtype_cannot_launch_a_kernel():
    with pytest.raises(ValueError, match="BF16"):
        candidate.combine(mx.ones((1, 1, 4, 128)), mx.ones((1, 1, 4), mx.bfloat16))


@pytest.mark.parametrize("shape", [(1, 2, 128), (2, 1, 128), (128,)])
def test_prefill_and_batch_shapes_keep_ordinary_path(shape):
    assert candidate.admission(block(), mx.ones(shape, mx.bfloat16)) == "requires B1 one-token decode"
