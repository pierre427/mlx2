"""Real-Metal gate: the sorted MoE weighted sum is bit-identical to the eager tail.

Run under the GPU lock with ``MLX2_RUN_METAL_TESTS=1``.  The reference is the
tail exactly as served by ``qwen3_next._routed_tail`` with the lever off:
``_scatter_unsort`` + ``(x * scores[..., None]).sum(axis=-2)``.  Raw output
bits are compared at prefill widths for top-10 (Flash-Next, D=2560) and
top-8 (Qwen3.6-35B, D=2048).  The falsifier: the same kernel with a
different partial count must NOT match, so the gate sees the order.  At K=10
that is one partial (slot-order bf16 sum, ~61% of outputs differ).  At K=8
MLX's eight one-row partials fold in slot order, which *is* the one-partial
sum, so the falsifier there is four partials.

The shapes and the one-partial falsifier follow ddalcu/mlx-serve#653 (MIT;
see docs/PROVENANCE.md).
"""

import os

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import qwen4_moe_weighted_sum as wsum
from mlx2.runtime.models import switch_layers

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="set MLX2_RUN_METAL_TESTS=1 (under the GPU lock) for the real-Metal gate",
)

EXPERTS = 64
# (batch, tokens) per routed slab, as in mlx-serve#653 plus the 64-row floor.
WIDTHS = [(1, 33), (1, 64), (1, 2048), (2, 1024), (1, 8192)]
MODELS = [(10, 2560), (8, 2048)]  # (top_k, hidden)


@pytest.fixture
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(4 << 30)
    try:
        yield
    finally:
        mx.clear_cache()
        mx.set_default_device(previous)


def _slab(batch, tokens, top_k, hidden, seed):
    """Router-shaped bf16 scores, a sorted gather and a sorted down slab."""
    key = mx.random.key(seed)
    k_gate, k_x, k_down = mx.random.split(key, 3)
    gates = mx.random.normal((batch, tokens, EXPERTS), key=k_gate).astype(mx.bfloat16)
    probs = mx.softmax(gates, axis=-1, precise=True)
    inds = mx.argpartition(probs, kth=-top_k, axis=-1)[..., -top_k:]
    scores = mx.take_along_axis(probs, inds, axis=-1)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    x = mx.random.normal((batch, tokens, 8), key=k_x).astype(mx.bfloat16)
    _, idx, inv_order = switch_layers._gather_sort(mx.expand_dims(x, (-2, -3)), inds)
    down = (mx.random.normal((idx.size, 1, hidden), key=k_down) * 0.5).astype(
        mx.bfloat16
    )
    mx.eval(scores, inds, inv_order, down)
    return down, inv_order, scores, inds


def _eager(down, inv_order, scores, inds):
    unsorted = switch_layers._scatter_unsort(down, inv_order, inds.shape).squeeze(-2)
    return (unsorted * scores[..., None]).sum(axis=-2)


def _bits(a):
    a = np.array(a.astype(mx.float32) if a.dtype == mx.bfloat16 else a)
    return a.view(np.uint32)


def _mismatches(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    return int(np.count_nonzero(_bits(a) != _bits(b)))


@pytest.mark.parametrize("top_k,hidden", MODELS)
@pytest.mark.parametrize("batch,tokens", WIDTHS)
def test_weighted_sum_bits_match_eager_tail(gpu, batch, tokens, top_k, hidden):
    down, inv_order, scores, inds = _slab(batch, tokens, top_k, hidden, tokens + top_k)
    admission = wsum.admit_moe_weighted_sum(
        x_sorted=down, inv_order=inv_order, scores=scores, indices=inds,
        do_sort=True, training=False,
    )
    assert admission.accepted, admission.reason
    eager = _eager(down, inv_order, scores, inds)
    fused = wsum.moe_weighted_sum(down, inv_order, scores)
    other = wsum.moe_weighted_sum(
        down, inv_order, scores, partials=1 if top_k > 8 else 4
    )
    mx.eval(eager, fused, other)
    assert fused.dtype == mx.bfloat16
    assert _mismatches(fused, eager) == 0
    # Falsifier: another reduction order must show up as mismatches.
    assert _mismatches(other, eager) > 0


@pytest.mark.parametrize("top_k,hidden", MODELS)
@pytest.mark.parametrize("batch,tokens", [(1, 33), (2, 1024)])
def test_weighted_sum_fp32_scores_bits_match_eager_tail(gpu, batch, tokens, top_k, hidden):
    down, inv_order, scores, inds = _slab(batch, tokens, top_k, hidden, 7 * tokens)
    scores = scores.astype(mx.float32)
    eager = _eager(down, inv_order, scores, inds)
    fused = wsum.moe_weighted_sum(down, inv_order, scores)
    mx.eval(eager, fused)
    assert fused.dtype == mx.float32
    assert _mismatches(fused, eager) == 0
