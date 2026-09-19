"""Laguna q8 fused MoE kernel contracts and Metal equivalence."""

import mlx.core as mx
import pytest

from mlx2.runtime.models.laguna_fused_moe import (
    BITS,
    CANDIDATE_TOKEN_WIDTHS,
    EXPERT_HIDDEN_SIZE,
    GROUP_SIZE,
    HIDDEN_SIZE,
    NUM_EXPERTS,
    TOP_K,
    admit_laguna_router,
    laguna_fused_down,
    laguna_fused_router,
)


def test_laguna_kernel_geometry_is_exact():
    assert (HIDDEN_SIZE, EXPERT_HIDDEN_SIZE) == (2048, 512)
    assert (NUM_EXPERTS, TOP_K) == (256, 8)
    assert (GROUP_SIZE, BITS) == (64, 8)
    assert CANDIDATE_TOKEN_WIDTHS == (1, 2, 4, 8)


def test_router_admission_fails_closed_on_cpu_and_wrong_semantics():
    logits = mx.zeros((1, 1, NUM_EXPERTS), dtype=mx.float32)
    bias = mx.zeros((NUM_EXPERTS,), dtype=mx.bfloat16)
    wrong = admit_laguna_router(
        logits, bias, top_k=7, norm_topk_prob=True, use_sigmoid=True,
        softcap=0.0, candidate_token_widths=(1,),
    )
    assert not wrong.accepted and "top-8" in wrong.reason
    if mx.default_device() == mx.cpu:
        cpu = admit_laguna_router(
            logits, bias, top_k=8, norm_topk_prob=True, use_sigmoid=True,
            softcap=0.0, candidate_token_widths=(1,),
        )
        assert not cpu.accepted and "not GPU" in cpu.reason


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")
@pytest.mark.parametrize("tokens", [1, 2, 4, 8])
def test_fused_router_matches_selected_experts_and_scores(tokens):
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        mx.random.seed(91 + tokens)
        logits = mx.random.normal((1, tokens, NUM_EXPERTS)).astype(mx.float32)
        bias = (mx.arange(NUM_EXPERTS) % 13).astype(mx.bfloat16) * 0.001
        probabilities = mx.sigmoid(logits)
        corrected = probabilities + bias.astype(mx.float32)
        reference_indices = mx.argpartition(
            -corrected, kth=TOP_K - 1, axis=-1
        )[..., :TOP_K]
        reference_scores = mx.take_along_axis(
            probabilities, reference_indices, axis=-1
        )
        reference_scores = reference_scores / mx.sum(
            reference_scores, axis=-1, keepdims=True
        )
        indices, scores = laguna_fused_router(
            logits, bias, output_dtype=mx.bfloat16,
            candidate_token_widths=(tokens,),
        )
        mx.eval(reference_indices, reference_scores, indices, scores)
        assert mx.array_equal(
            mx.sort(reference_indices, axis=-1), mx.sort(indices, axis=-1)
        ).item()
        order = mx.argsort(indices, axis=-1)
        fused_sorted = mx.take_along_axis(scores, order, axis=-1)
        ref_order = mx.argsort(reference_indices, axis=-1)
        ref_sorted = mx.take_along_axis(
            reference_scores.astype(mx.bfloat16), ref_order, axis=-1
        )
        assert mx.allclose(fused_sorted, ref_sorted, atol=2e-3, rtol=2e-3).item()
    finally:
        mx.set_default_device(previous)


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")
def test_fused_q8_down_matches_dequantized_reference_m1():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        mx.random.seed(123)
        hidden = mx.random.normal((1, 1, TOP_K, EXPERT_HIDDEN_SIZE)).astype(
            mx.bfloat16
        )
        indices = mx.array([[[3, 17, 31, 63, 95, 127, 191, 255]]], mx.uint32)
        scores = mx.softmax(mx.arange(TOP_K, dtype=mx.float32)).astype(mx.bfloat16)
        scores = mx.broadcast_to(scores, (1, 1, TOP_K))
        words = EXPERT_HIDDEN_SIZE * BITS // 32
        groups = EXPERT_HIDDEN_SIZE // GROUP_SIZE
        weight = mx.zeros((NUM_EXPERTS, HIDDEN_SIZE, words), dtype=mx.uint32)
        expert_scale = (1 + mx.arange(NUM_EXPERTS, dtype=mx.float32)) * 1e-5
        scales = mx.broadcast_to(
            expert_scale[:, None, None], (NUM_EXPERTS, HIDDEN_SIZE, groups)
        ).astype(mx.bfloat16)
        biases = (-128 * scales).astype(mx.bfloat16)
        fused = laguna_fused_down(
            hidden, indices, scores, weight, scales, biases,
            candidate_token_widths=(1,),
        )
        selected_weight = weight[indices]
        selected_scales = scales[indices]
        selected_biases = biases[indices]
        dense = mx.dequantize(
            selected_weight, selected_scales, selected_biases,
            group_size=GROUP_SIZE, bits=BITS, mode="affine",
        )
        expert = mx.matmul(hidden[..., None, :], dense.swapaxes(-1, -2)).squeeze(-2)
        reference = mx.sum(expert * scores[..., None], axis=-2)
        mx.eval(fused, reference)
        assert mx.allclose(fused, reference, atol=0.25, rtol=2e-2).item()
    finally:
        mx.set_default_device(previous)
