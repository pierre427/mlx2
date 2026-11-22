# SPDX-License-Identifier: Apache-2.0
"""Fake activation quantization for the Inkling 2 routed experts.

``inkling2_moe_act_quant`` (opt-in, default off) is an accuracy experiment,
not a speed path: Apple GPUs have no FP4 x FP4 (or FP8 x FP8) matrix units,
so the expert matmuls still run BF16 activations against the NVFP4/MXFP8
weights.  The activation is quantized and dequantized in place, so the
matmul sees exactly the values a W4A4 (layers 0-19) or W8A8 (layers 20-23)
kernel would multiply, rounded to the activation dtype.  Every call adds
work; nothing gets faster.

The rules follow the checkpoint's ``hf_quant_config.json``
(``required_activation_quantization``):

**NVFP4, layers 0-19** (``nvfp4_e2m1``, block 16, ``float8_e4m3fn`` block
scales, ``amax_type per_token``, rounding nearest, ``scale_selection_rule
fouroversix_mae``).  The four-over-six rule is MIT Han Lab's Four Over Six
(``mit-han-lab/fouroversix`` @ dadfad69, MIT; arXiv 2512.02010), read from
``quantize/pytorch/reference.py`` and ``kernels/triton/quantize.py``
(``nvfp4_fouroversix_block_scaled_quantization``).  Per row ``r`` (one token)
with ``A = max|x_r|`` and each 16-value block ``b`` with ``a_b = max|x_b|``:

- ``encode = 1536 / A`` (``1536 = 6 * 256``: under four-over-six the E4M3
  scale ceiling is 256, not 448, so the 1.5x candidate still fits);
- ``s6 = e4m3((a_b / 6) * encode)``, ``s4 = e4m3((a_b / 6) * encode * 1.5)``
  (E4M3 round-to-nearest-even; NaN from ``0 * inf`` at ``A = 0`` becomes 0);
- ``q_k = e2m1(x * (1 / ((1 / encode) * s_k)))``, 0 where ``s_k = 0``, with
  E2M1 round-to-nearest-even onto {0, 0.5, 1, 1.5, 2, 3, 4, 6}, saturating
  at 6 (the hardware ``cvt.rn.satfinite.e2m1x2`` and the Triton fallback;
  the PyTorch reference does not clamp, which differs only for |x| >= 7,
  reachable only through subnormal E4M3 scales);
- ``d_k = q_k * s_k * A / 1536``; a token with ``0 < A < 1536 / FLT_MAX``
  (``encode`` overflows; the library would produce NaN) quantizes to zero;
- keep ``d4`` when ``sum|d4 - x_b| < sum|d6 - x_b|`` (strict, so a tie keeps
  the standard scale 6), else ``d6``.

``per_token`` replaces the library's per-tensor ``A`` with each row's own
amax: for ``w13`` a row is a token of the MoE input; for ``w2`` a row is one
(token, expert) row of the SwiGLU output, which is what the down GEMM reads.

**MXFP8, layers 20-23** (``mxfp8_e4m3``, block 32, ``ue8m0`` scales, rounding
nearest): ``scale = 2 ** ceil(log2(a_b / 448))`` (the FP32 exponent of
``a_b / 448``, plus one when its mantissa is nonzero: the round-up rule of
fouroversix's MX path and of ``mx.quantize(mode="mxfp8")``; the contract does
not name the scale rule, and OCP's floor rule would saturate the top binade
instead), ``q = e4m3(x / scale)`` (round-to-nearest-even, saturating at 448),
``d = q * scale``.  E4M3 times a power of two is exact in BF16.

All arithmetic is FP32 and the result is cast back to the input dtype.  The
NVFP4 product ``q * s * A / 1536`` is generally not a BF16 value, so a BF16
activation picks up one more rounding (at most 2^-8 relative) than a true
W4A4 kernel, which keeps ``q * s`` and folds ``A / 1536`` into the output.
"""

from __future__ import annotations

import mlx.core as mx

NVFP4_BLOCK = 16
MXFP8_BLOCK = 32
E2M1_MAX = 6.0
E4M3_MAX = 448.0
E4M3_MAX_FOUROVERSIX = 256.0
_NV_RANGE = E2M1_MAX * E4M3_MAX_FOUROVERSIX  # 1536
FORMATS = ("nvfp4", "mxfp8")


def _blocks(x, size):
    n = x.shape[-1]
    if n % size:
        raise ValueError(f"activation width {n} is not a multiple of block {size}")
    return x.astype(mx.float32).reshape(*x.shape[:-1], n // size, size)


def _e2m1(v):
    """Round to the signed E2M1 grid, nearest-even, saturating at 6."""
    a = mx.abs(v)
    q = mx.where(
        a < 2,
        mx.round(2 * a) / 2,
        mx.where(a < 4, mx.round(a), mx.minimum(2 * mx.round(a / 2), E2M1_MAX)),
    )
    return mx.where(v < 0, -q, q)


def _e4m3(v):
    return mx.from_fp8(mx.to_fp8(v), dtype=mx.float32)


def _nvfp4_candidates(x):
    """``(blocks, d6, d4, pick4)`` for ``x`` (``[..., n]``), FP32."""
    blocks = _blocks(x, NVFP4_BLOCK)
    amax = mx.max(mx.abs(blocks), axis=(-2, -1), keepdims=True)  # per row
    bmax = mx.max(mx.abs(blocks), axis=-1, keepdims=True)
    encode = _NV_RANGE / amax
    decode = 1 / encode
    hp6 = (bmax / E2M1_MAX) * encode
    hp6 = mx.where(mx.isnan(hp6), 0.0, hp6)
    out = []
    for hp in (hp6, hp6 * 1.5):
        s = _e4m3(hp)
        scaled = mx.where(s != 0, blocks * (1 / (decode * s)), 0.0)
        out.append(_e2m1(scaled) * s * amax / _NV_RANGE)
    d6, d4 = out
    err6 = mx.sum(mx.abs(d6 - blocks), axis=-1, keepdims=True)
    err4 = mx.sum(mx.abs(d4 - blocks), axis=-1, keepdims=True)
    return blocks, d6, d4, err4 < err6


def _nvfp4(x):
    blocks, d6, d4, pick4 = _nvfp4_candidates(x)
    out = mx.where(pick4, d4, d6)
    # A nonzero amax below 1536 / FLT_MAX overflows ``encode`` (the library
    # would emit NaN from 0 * inf); such a token quantizes to zero.
    amax = mx.max(mx.abs(blocks), axis=(-2, -1), keepdims=True)
    out = mx.where(mx.isinf(_NV_RANGE / amax) & (amax > 0), 0.0, out)
    return out.reshape(x.shape).astype(x.dtype)


def _mxfp8(x):
    blocks = _blocks(x, MXFP8_BLOCK)
    bits = (mx.max(mx.abs(blocks), axis=-1, keepdims=True) / E4M3_MAX).view(mx.uint32)
    exponent = (bits >> 23) & 0xFF
    exponent = mx.where((bits & 0x7FFFFF) == 0, exponent, exponent + 1)
    scale = (exponent << 23).view(mx.float32)
    q = _e4m3(mx.where(exponent != 0, blocks / scale, 0.0))
    return (q * scale).reshape(x.shape).astype(x.dtype)


_nvfp4_compiled = mx.compile(_nvfp4)
_mxfp8_compiled = mx.compile(_mxfp8)


def fake_quant_nvfp4_act(x):
    """NVFP4 four-over-six (MAE) per-token fake quantization of ``x``."""
    return _nvfp4_compiled(x)


def fake_quant_mxfp8_act(x):
    """MXFP8 (E4M3, block 32, UE8M0 round-up scales) fake quantization."""
    return _mxfp8_compiled(x)


def apply(fmt, x):
    if fmt == "nvfp4":
        return fake_quant_nvfp4_act(x)
    if fmt == "mxfp8":
        return fake_quant_mxfp8_act(x)
    raise ValueError(f"unknown activation format {fmt!r}")


def expert_format(switch):
    """The activation format a layer's expert weights declare, or ``None``.

    NVFP4 experts (``ScaledNVFP4SwitchLinear``, or a quantized switch in
    mode ``nvfp4``) take NVFP4 activations; MXFP8 experts take MXFP8.  Any
    other layout (dense test experts, affine quantization) has no declared
    activation contract."""
    from ...nvfp4_linear import ScaledNVFP4SwitchLinear
    from ..switch_layers import QuantizedSwitchLinear

    kinds = set()
    for name in ("gate_proj", "up_proj", "down_proj"):
        proj = switch[name]
        if isinstance(proj, ScaledNVFP4SwitchLinear):
            kinds.add("nvfp4")
        elif isinstance(proj, QuantizedSwitchLinear) and proj.mode in FORMATS:
            kinds.add(proj.mode)
        else:
            return None
    return kinds.pop() if len(kinds) == 1 else None


def count(counts, key):
    counts[key] = counts.get(key, 0) + 1
