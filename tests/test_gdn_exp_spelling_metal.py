"""Real-Metal gate: GDN prework kernels equal the served eager ops on a SiLU sweep.

Run under the GPU lock with ``MLX2_RUN_METAL_TESTS=1``.

MLX 0.32.4 serves compiled ``nn.silu`` with ``metal::precise::exp``.  Three
kernels kept the old ``metal::exp`` spelling after the repin (2e9a43bf9): the
Qwen3.8 tree/cohort GDN pre kernel (``qwen38_tree_gdn.gdn_pre``), the fused
prefill prework (Flash-Next default-on, Qwen3.8 opt-in) and its Qwen3.5 swish
norm gate.  The two bf16 sigmoids round apart on one input (-6.84375) and the
fp32 ones on ~1% of bf16-valued inputs, which random draws never reach.  Each
case below feeds conv outputs covering every bf16 value with magnitude in
[2**-6, 32] (an identity last conv tap makes each conv output exactly its qkv
value) and compares the public kernel entry point with the eager chain bit
for bit.  On the pre-fix kernels the tree and prework cases fail on v and on
the q/k of every head that holds -6.84375.
"""

import os

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.runtime.models import qwen38_tree_gdn as tree
from mlx2.runtime.models import qwen4_fused_gdn as fused
from mlx2.runtime.models import qwen4_fused_gdn_prefill as prefill
from mlx2.runtime.models.gated_delta import compute_g, normalize_gdn_qk
from mlx2.runtime.models.precise_ops import gate_sigmoid
from mlx2.runtime.models.qwen3_next import _precise_swiglu
from mlx2.runtime.models.served_exp import covering_rows

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="set MLX2_RUN_METAL_TESTS=1 (under the GPU lock) for the real-Metal gate",
)

NK, NV, DK, DV, TAPS = 16, 48, 128, 128, 4
C = 2 * NK * DK + NV * DV


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _differing(have, want):
    assert have.shape == want.shape and have.dtype == want.dtype
    width = mx.uint16 if have.dtype.size == 2 else mx.uint32
    return int((have.view(width) != want.view(width)).sum().item())


def _sweep(rows, width, seed):
    import numpy as np

    encodings = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    widened = (encodings.astype(np.uint32) << 16).view(np.float32)
    keep = np.isfinite(widened) & (np.abs(widened) >= 2.0 ** -6) & (np.abs(widened) <= 32.0)
    grid = mx.array(encodings[keep]).view(mx.bfloat16)
    return covering_rows(grid, rows, width, seed)[None]


def _identity_conv():
    weight = mx.zeros((C, TAPS, 1), dtype=mx.bfloat16)
    weight[:, TAPS - 1, 0] = 1
    return weight


def _eager_conv(qkv, weight, rows):
    state = mx.zeros((1, TAPS - 1, C), dtype=mx.bfloat16)
    conv = nn.silu(mx.conv1d(mx.concatenate([state, qkv], axis=1), weight, groups=C))
    q, k, v = mx.split(conv, [NK * DK, 2 * NK * DK], -1)
    return q.reshape(1, rows, NK, DK), k.reshape(1, rows, NK, DK), v.reshape(1, rows, NV, DV)


def test_the_sweep_reaches_the_bf16_input_where_the_spellings_differ():
    assert bool(mx.any(_sweep(1, C, 1) == -6.84375).item())


@pytest.mark.parametrize("rows", [1, 4, 9])
def test_tree_pre_kernel_matches_the_eager_gdn_prework(rows):
    """``gdn_pre`` (packed [z | b | a], one chain stream) vs nn.silu(conv1d),
    normalize_gdn_qk, compute_g and the widened gate_sigmoid(b)."""
    qkv = _sweep(rows, C, 100 * rows)
    weight = _identity_conv()
    k1, k2, k3, k4 = mx.random.split(mx.random.key(rows), 4)
    b = (mx.random.normal((1, rows, NV), key=k1) * 3).astype(mx.bfloat16)
    a = (mx.random.normal((1, rows, NV), key=k2) * 3).astype(mx.bfloat16)
    z = mx.random.normal((1, rows, NV * DV), key=k3).astype(mx.bfloat16)
    a_log = mx.linspace(-2.0, 1.0, NV).astype(mx.bfloat16)
    dt_bias = (mx.random.normal((NV,), key=k4) * 0.5).astype(mx.bfloat16)
    plan = tree.ConvPlan([list(range(-1, rows - 1))], TAPS - 1)
    state = mx.zeros((1, TAPS - 1, C), dtype=mx.bfloat16)
    q, k, v, g, beta = tree.gdn_pre(
        qkv, [state], weight, plan, mx.concatenate([z, b, a], axis=-1), a_log, dt_bias,
        nk=NK, nv=NV, dk=DK, dv=DV,
    )
    eq, ek, ev = _eager_conv(qkv, weight, rows)
    eq, ek = normalize_gdn_qk(eq, ek)
    assert _differing(v, ev) == 0
    assert _differing(q, eq) == 0
    assert _differing(k, ek) == 0
    assert _differing(g, compute_g(a_log, a, dt_bias).astype(mx.float32)) == 0
    assert _differing(beta, gate_sigmoid(b.astype(mx.float32))) == 0


@pytest.mark.parametrize("numerics", prefill.PREWORK_NUMERICS)
@pytest.mark.parametrize("rows", [4, 64])
def test_prefill_prework_matches_the_eager_chain(numerics, rows):
    qkv = _sweep(rows, C, 7000 + rows)
    weight = _identity_conv()
    q, k, v, _ = prefill.qwen4_gdn_prefill_prework(qkv, None, weight, numerics=numerics)
    eq, ek, ev = _eager_conv(qkv, weight, rows)
    if numerics == "qwen4":
        eq = eq * mx.rsqrt(mx.sum(mx.square(eq), axis=-1, keepdims=True) + 1e-06)
        ek = ek * mx.rsqrt(mx.sum(mx.square(ek), axis=-1, keepdims=True) + 1e-06)
        eq = eq * DK ** (-0.5)
    else:
        eq, ek = normalize_gdn_qk(eq, ek)
    assert _differing(v, ev) == 0
    assert _differing(q, eq) == 0
    assert _differing(k, ek) == 0


@pytest.mark.parametrize("gate", prefill.NORM_GATES)
def test_prefill_norm_gate_matches_the_eager_gated_norm(gate):
    rows, eps = 16, 1e-6
    y = mx.random.normal((1, rows, NV, DV), key=mx.random.key(5)).astype(mx.bfloat16)
    z = _sweep(rows, NV * DV, 8000)
    w = (1 + 0.1 * mx.random.normal((DV,), key=mx.random.key(6))).astype(mx.bfloat16)
    out = prefill.qwen4_gdn_prefill_norm_gate(y, z, w, eps, gate=gate)
    zr = z.reshape(1, rows, NV, DV)
    if gate == "sigmoid":
        ref = (mx.fast.rms_norm(y, w, eps).astype(mx.float32) * mx.sigmoid(zr.astype(mx.float32)))
        ref = ref.astype(mx.bfloat16)
    else:
        ref = _precise_swiglu(y, zr, mx.fast.rms_norm(y, w, eps))
    assert _differing(out, ref.reshape(1, rows, NV * DV)) == 0


def test_runtime_canaries_pass_and_catch_the_other_spelling():
    """The once-per-process canaries accept the served spelling and refuse
    the other one, so a future spelling change fails closed."""
    served = fused.served_silu_exp()
    assert served in fused._SILU_EXP_CANDIDATES
    assert tree.served_silu_spelling() == served
    assert tree.pre_refusal() is None
    for numerics, gate in prefill.ARCHITECTURE_CONTRACTS.values():
        assert prefill.served_spellings(gate)[0] == served
        assert prefill.prefill_refusal(numerics, gate) is None
    other = next(s for s in fused._SILU_EXP_CANDIDATES if s != served)
    assert tree._decode_parity(other) is not None
    for numerics in prefill.PREWORK_NUMERICS:
        refusal = prefill.eager_parity(numerics, "sigmoid", other, prefill.KERNEL_SILU_EXP)
        assert refusal is not None and "v differs" in refusal
