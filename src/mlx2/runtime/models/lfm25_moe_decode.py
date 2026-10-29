"""Opt-in LFM weighting tail and expert combine; ordinary sigmoid law retained.

Original mlx2 Metal composition, motivated by Rapid-MLX #5298. The donor's
softmax, biased mixture weights and epsilon are intentionally not copied.
See provenance/pr-adoption-20261027.json. No shared scheduler state.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from functools import lru_cache
from pathlib import Path

import mlx.core as mx

MODE = "sigmoid_tail_v1"
SCHEMA = "mlx2.lfm25-moe-decode.v1"

_ROUTE = r"""
    const uint lane = thread_index_in_simdgroup;
    // Expert membership AND ordering come from ordinary mx.argpartition.
    // Its partition tail is not a sorted top-k, including on CPU.
    const T p = lane < K ? picked[lane] : T(0);
    const T denom = T(total[0] + T(1.0e-6f));
    if (lane < K) {
        const T normalized = NORMALIZE ? T(p / denom) : p;
        scores[lane] = A(T(normalized * scale[0]));
    }
"""

_COMBINE = r"""
    const uint d = thread_position_in_grid.x;
    if (d >= H) return;
    // Both product and strided reduction materialize in T.
    T value = T(0);
    for (uint k = 0; k < K; ++k) {
        const T product = T(experts[k * H + d] * scores[k]);
        value = T(value + product);
    }
    output[d] = T(value);
"""


@lru_cache(maxsize=1)
def identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def parse_policy(value):
    if value in (None, "off"):
        return "off"
    if value != MODE:
        raise ValueError(f"lfm_moe_decode must be off or {MODE}")
    return MODE


def admission(block, x):
    if tuple(x.shape[:2]) != (1, 1) or x.ndim != 3:
        return "requires B1 one-token decode"
    if x.dtype != mx.bfloat16:
        return "requires BF16 activations"
    if block.training or getattr(block, "sharding_group", None) is not None:
        return "training or sharded block"
    if not 32 <= block.num_experts <= 256 or not 1 <= block.top_k <= 8:
        return "expert geometry"
    if block.use_expert_bias and (
        block.expert_bias.shape != (block.num_experts,)
        or block.expert_bias.dtype != mx.float32
    ):
        return "selection-bias layout"
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return "Metal unavailable"
    return None


@lru_cache(maxsize=2)
def _kernel(kind):
    route = kind == "route"
    return mx.fast.metal_kernel(
        name="mlx2_lfm25_sigmoid_" + kind,
        input_names=["picked", "total", "scale"] if route else ["experts", "scores"],
        output_names=["scores"] if route else ["output"],
        source=_ROUTE if route else _COMBINE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=16)
def _constants(scale, dtype, device):
    # Eight elements keep custom-kernel inputs in the buffer address space.
    return mx.full((8,), scale, dtype)


def route(block, probabilities, activation_dtype=mx.bfloat16):
    scale = _constants(float(block.routed_scaling_factor),
                       probabilities.dtype, mx.default_device())
    selection = probabilities.astype(mx.float32)
    if block.use_expert_bias:
        selection = selection + block.expert_bias
    indices = mx.argpartition(selection, kth=-block.top_k, axis=-1)[..., -block.top_k:]
    picked = mx.take_along_axis(probabilities, indices, axis=-1)
    total = mx.sum(picked, axis=-1, keepdims=True)
    scores = _kernel("route")(
        inputs=[picked, total, scale],
        template=[("T", probabilities.dtype), ("A", activation_dtype),
                  ("K", block.top_k), ("NORMALIZE", block.norm_topk_prob)],
        grid=(32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(1, 1, block.top_k)], output_dtypes=[activation_dtype],
    )[0]
    return indices, scores


def reference_route(block, probabilities, activation_dtype=mx.bfloat16):
    selection = probabilities.astype(mx.float32)
    if block.use_expert_bias:
        selection = selection + block.expert_bias
    indices = mx.argpartition(selection, kth=-block.top_k, axis=-1)[..., -block.top_k:]
    scores = mx.take_along_axis(probabilities, indices, axis=-1)
    if block.norm_topk_prob:
        scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-6)
    return indices, (scores * block.routed_scaling_factor).astype(activation_dtype)


def combine(experts, scores):
    if experts.dtype != mx.bfloat16 or scores.dtype != mx.bfloat16:
        raise ValueError("LFM fused combine requires BF16 experts and scores")
    h = experts.shape[-1]
    return _kernel("combine")(
        inputs=[experts, scores], template=[("T", experts.dtype),
                                          ("K", experts.shape[-2]), ("H", h)],
        grid=(h, 1, 1), threadgroup=(min(h, 256), 1, 1),
        output_shapes=[(1, 1, h)], output_dtypes=[experts.dtype],
    )[0]


def _law(block):
    return (block.num_experts, block.top_k, block.use_expert_bias,
            block.norm_topk_prob, block.routed_scaling_factor,
            getattr(block, "expert_bias", None))


def prepare(block):
    """Check ties, extreme probabilities and actual bias before installation.

    A passed canary is bounded build evidence, not model qualification.
    No canary runs during a served request.
    """
    from ..row_exact.canary import bit_mismatches

    for dtype in (mx.bfloat16, mx.float32):
        for seed in range(4):
            logits = mx.random.normal((1, 1, block.num_experts), key=mx.random.key(seed))
            probabilities = (mx.sigmoid(logits * (seed + 1)) if seed else
                             mx.full(logits.shape, 0.5)).astype(dtype)
            got, want = route(block, probabilities), reference_route(block, probabilities)
            experts = mx.random.normal((1, 1, block.top_k, 128),
                                       key=mx.random.key(100 + seed)).astype(mx.bfloat16)
            output = combine(experts, want[1])
            expected = mx.sum(experts * want[1][..., None], axis=-2)
            mx.eval(got, want, output, expected)
            if any(bit_mismatches(a, b) for a, b in zip(got, want)):
                return False
            if bit_mismatches(output, expected):
                return False
    return True


class Handle:
    def __init__(self):
        self.counts = Counter()
        self._laws = {}

    def forward(self, block, x):
        reason = admission(block, x)
        old = self._laws.get(id(block))
        law = _law(block)
        if old is None or old[:-1] != law[:-1] or old[-1] is not law[-1]:
            reason = "block law or bias changed"
        if reason:
            self.counts["declined:" + reason] += 1
            return None
        probabilities = mx.sigmoid(block.gate(x))
        if probabilities.dtype not in (mx.bfloat16, mx.float32):
            self.counts["declined:router dtype"] += 1
            return None
        indices, scores = route(block, probabilities)
        experts = block.switch_mlp(x, indices)
        if experts.dtype != mx.bfloat16:
            self.counts["declined:expert dtype"] += 1
            return None
        output = combine(experts, scores)
        self.counts["dispatches"] += 1
        return output

    def status(self):
        return {"schema": SCHEMA, "mode": MODE, "selected": True,
                "qualified": False, "observed_used": self.counts["dispatches"] > 0,
                "counts": dict(self.counts)}


def install(model):
    from .lfm25_moe import SparseMoE

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise ValueError("lfm_moe_decode requires Metal")
    blocks = [module for _, module in model.named_modules() if type(module) is SparseMoE]
    if not blocks:
        raise ValueError("lfm_moe_decode found no supported MoE blocks")
    handle = Handle()
    # Prepare all blocks before mutating any of them.
    for block in blocks:
        if not prepare(block):
            raise ValueError("LFM sigmoid routing/combine parity canary failed")
        handle._laws[id(block)] = _law(block)
    for block in blocks:
        object.__setattr__(block, "_mlx2_decode_fusion", handle)
    return handle
