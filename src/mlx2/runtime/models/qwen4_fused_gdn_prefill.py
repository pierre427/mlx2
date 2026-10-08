# SPDX-License-Identifier: Apache-2.0 AND MIT
# Python glue adapted from jundot/omlx (Apache-2.0) at 3e2bdb1f; the Metal
# kernel sources below carry their own scoped MIT notice (mlx-serve).  See
# docs/PROVENANCE.md and provenance/omlx-3903-qwen4-prefill.json.
"""Fused Qwen4 (Flash-Next) GDN prefill prework and gated-norm kernels.

Two ``mx.fast.metal_kernel`` dispatches replace the eager prefill chain in
``qwen3_5.GatedDeltaNet.__call__`` for one admitted geometry:

* **prework** -- conv-state concat + depthwise conv1d (K=4) + SiLU + q/k/v
  split + Qwen4's direct L2 q/k normalization + the ``dk**-0.5`` query scale
  + the next conv state.  One threadgroup of 32 lanes per (row, logical head),
  4 channels per lane; the row count is a run-time input so every prefill
  width shares one pipeline.
* **norm-gate** -- ``RMSNormGated`` (``mx.fast.rms_norm`` with the gain, then
  an fp32 sigmoid output gate) over the recurrence output.

The recurrence itself (``gated_delta_update`` with ``beta_input_dtype=True``)
is untouched and still runs through the layer's own ``_gated_delta_update``.

Admission (``admit_qwen4_fused_gdn_prefill``) is purely structural and fails
closed with a reason: batch 1, rows >= 64, bf16 activations, no mask, no
ragged lengths, no left padding that is pending or unknown on the host (see
"Left padding" below), not speculating, not training, not sharded,
sigmoid output gate, and the Flash-Next geometry (conv_dim 10240, HK 16,
HV 48, DK = DV = 128, conv kernel 4).  The route is opt-in through
``MLX_QWEN4_FUSED_GDN_PREFILL`` (default ``0``) until a GPU bit/ULP gate
qualifies it.

Semantics audit: eager path vs. kernel
--------------------------------------
Checked line by line against ``qwen3_5.GatedDeltaNet.__call__``,
``qwen4_exp.GatedDeltaNet._normalize_qk`` and ``qwen4_exp.RMSNormGated``.

Matches (same rounding sites):

1. q scale: eager ``q_l2 * k.shape[-1] ** -0.5`` multiplies a bf16 array by a
   weak Python scalar, i.e. by ``bf16(128**-0.5)`` with one bf16 rounding.
   The kernel receives ``q_scale = mx.array(dk**-0.5, bf16)`` and rounds
   ``T(float(l2) * float(q_scale))`` once.  k gets no scale (``k_scale`` is
   only read by the non-L2 branch, passed as 1.0).
2. L2 eps: both use a hard-coded ``1e-6`` added in bf16 after the bf16 sum
   (``T(float(T(sum)) + float(T(1e-6)))`` vs eager ``bf16_sum + 1e-06``),
   then ``rsqrt`` rounded to bf16, then ``x * inv`` rounded to bf16.
3. Squares: ``mx.square`` rounds each square to bf16; the kernel does too.
4. SiLU: eager ``nn.silu`` is ``x * sigmoid(x)`` in bf16 with the MLX unary
   sigmoid ``1/(1+exp(|x|))`` and its reflection; the kernel uses the same
   formula (``metal::exp``) and the same two bf16 rounding sites.  mlx2's
   qualified decode kernel (``qwen4_fused_gdn.py``) uses the same fast-exp
   form for this SiLU.
5. Norm weight convention: ``linear_attn.norm.weight`` is NOT in the
   ``sanitize`` zero-centred list, so no ``+1`` fold happens; both paths
   multiply by the stored gain as-is (``w * T(x * inv)``, the
   ``mx.fast.rms_norm`` order).
6. Output gate: sigmoid in fp32 via ``precise::exp`` on the fp32 z, product
   in fp32, one rounding to T -- the same form as mlx2's qualified decode
   kernel's ``mlx_sigmoid_precise<float>``.  A ``silu``/``swish`` output gate
   (``output_gate_type``) is refused, not approximated.
7. Conv state: the kernel writes the last 3 *raw* qkv rows, which equals the
   eager ``conv_input[:, -3:]`` for S >= 3 (admission requires S >= 64); a
   pure copy, so bit-exact.  ``cache[0] is None`` is replaced by bf16 zeros
   of shape (1, 3, C) -- the eager path uses ``inputs.dtype`` for those
   zeros; admission requires bf16 qkv, which equals ``inputs.dtype`` for a
   bf16 model (a model whose projections change dtype would differ).
8. Existing conv state is read as the first three conv taps for rows 0..2,
   exactly the eager concat.

Possible ULP-level differences (not bit-proven; the GPU gate in
``scripts/check_omlx3903_port.py`` measures them):

a. Conv accumulation: the kernel sums the four taps in fp32 in tap order and
   rounds once.  Eager ``nn.Conv1d`` (grouped, depthwise) uses MLX's conv
   implementation, whose accumulation order/precision is not asserted here.
b. L2 sum reduction: per-lane sequential bf16 accumulation of 4 contiguous
   squares, then a 32-lane fp32 ``simd_shuffle_xor`` butterfly, one bf16
   round.  omlx states this mirrors the MLX GPU row-reduce bit-exactly; mlx2's
   own decode kernel mirrors it with ``simd_sum`` instead of the xor tree.
   The two fp32 reduction orders can differ before the bf16 rounding.
c. rms_norm sum: 4 fp32 products per lane then ``simd_sum``, as MLX's
   ``rms_single_row`` for axis 128; not re-derived here.
d. On CPU the eager path's reductions/exp differ from any Metal kernel, so a
   CPU comparison can only be tolerance-level; the CPU tests prove plumbing.

Qwen3.8 27B variant (``architecture="qwen38"``)
----------------------------------------------
The 27B shares Flash-Next's GDN geometry exactly (HK 16, HV 48, DK = DV =
128, conv kernel 4, conv_dim 10240, bf16 conv/norm weights) and the same
eager conv + SiLU (both inherit ``qwen3_5.GatedDeltaNet._recurrent_core``).
It differs in two numerics contracts, each already reproduced bit-exactly by
the qualified 27B decode kernel (``qwen4_fused_gdn`` ``AGNES_NUMERICS``):

* q/k normalization is ``gated_delta.normalize_gdn_qk``: ``mx.fast.rms_norm``
  (fp32 lane sums of exact squares, one ``simd_sum``, ``rsqrt(sum / DK +
  1e-6 / DK)``, one bf16 rounding) then ``q * bf16((DK**-0.5)**2)`` and
  ``k * bf16(DK**-0.5)``.  That is the donor prework's non-L2 branch
  (``L2 = 0``) with the epsilon corrected from ``1e-6`` to ``1e-6 / DK``.
* the output gate is ``Qwen3NextRMSNormGated`` -> compiled
  ``_precise_swiglu``: ``bf16(fp32(rms_norm(y, w)) * (z * sigmoid(z)))``
  with the runtime-compiled (``metal::exp``) sigmoid, instead of Qwen4's
  precise sigmoid.

The recurrence stays the layer's own ``_gated_delta_update`` (the 27B's has
no ``beta_input_dtype``).  The 27B wiring lives in ``qwen38_fused_gdn``.

Left padding:

The eager GDN path never reads ``cache.left_padding`` itself.  Padding acts
only through the SSM mask that ``ArraysCache.make_mask`` builds
(``pos >= left_padding``): ``_recurrent_core`` zeroes masked qkv rows and
hands the mask to ``gated_delta_update``, and the conv-state tail is chosen
by ``cache.lengths``, never by padding.  A row whose padding is ``<= 0``
(zero for an unpadded prompt, negative once ``advance`` has consumed it) has
an all-True mask, and ``make_mask`` already returns ``None`` for it when the
host mirror says so (``_host_all_valid``).  So host-known ``left_padding <=
0`` on every row is exactly the unpadded computation, and admission accepts
it; ``cache.advance`` then updates the padding the same way on both paths.
Admission reads only ``ArraysCache.host_left_padding()``, the identity-keyed
host mirror, so it never syncs the device.  A positive entry is refused as
``left padding (nonzero)``; a padding array with no live mirror is refused as
``left padding (unknown on host)`` (fail closed).

Behavioural differences:

* The eager path writes ``cache[0]`` before the recurrence; the fused path
  writes ``cache[0]`` and ``cache[1]`` together after both kernels were built,
  so a dispatch failure falls back to eager with the cache untouched.
* The recurrence output dtype is checked at graph-build time; if it is not
  bf16 the fused norm-gate is skipped and ``self.norm`` runs eagerly
  (counted in ``fused_gdn_prefill_norm_eager``).
* The fused path returns ``out_proj(flat)`` directly; sharded layers (which
  need ``all_sum``) are refused.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

import mlx.core as mx

from .qwen4_fused_gdn import (
    CONV_DIM,
    CONV_KERNEL,
    KEY_HEAD_DIM,
    NUM_KEY_HEADS,
    NUM_VALUE_HEADS,
    VALUE_DIM,
    VALUE_HEAD_DIM,
    FusedGdnAdmission,
    fused_gdn_runtime_supported,
)

PREFILL_ENV = "MLX_QWEN4_FUSED_GDN_PREFILL"
PREFILL_MIN_ROWS = 64
NKEEP = CONV_KERNEL - 1
L2_EPS = 1e-6


def prefill_enabled_from_env() -> bool:
    """``MLX_QWEN4_FUSED_GDN_PREFILL``; unset or ``0`` means off."""
    raw = os.environ.get(PREFILL_ENV, "0")
    return raw.strip().lower() in {"1", "true", "on", "yes"}


def runtime_supported() -> bool:
    """Metal available and the default device is the GPU."""
    return fused_gdn_runtime_supported()


# ---------------------------------------------------------------------------
# Copyright (c) 2026 David Dalcu.  The GDN prework and Qwen4 norm-gate kernel
# sources below, and their prefill variants, are adapted (via jundot/omlx
# 3e2bdb1f, omlx/patches/qwen35_gdn_prework.py) from ddalcu/mlx-serve's
# MIT-licensed ``src/transformer.zig`` at tag ``v26.8.11-pre-release.1``
# (commit 09970f9b).  Preserve this scoped notice with those kernels.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ---------------------------------------------------------------------------

# Donor verify prework (omlx ``_SOURCE``), kept verbatim so the prefill
# substitution below stays diffable against the donor.
_VERIFY_PREWORK_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint batch_idx = threadgroup_position_in_grid.y / uint(S);
    uint row = threadgroup_position_in_grid.y % uint(S);
    uint logical_head = threadgroup_position_in_grid.z;
    constexpr uint q_heads = uint(HK);
    constexpr uint k_head_base = uint(HK);
    constexpr uint v_head_base = 2 * uint(HK);
    bool is_q = logical_head < q_heads;
    bool is_k = logical_head >= k_head_base && logical_head < v_head_base;
    uint head = is_q ? logical_head
               : (is_k ? logical_head - k_head_base : logical_head - v_head_base);
    uint channel_base = is_q ? head * uint(DK)
                       : (is_k ? uint(HK) * uint(DK) + head * uint(DK)
                               : 2 * uint(HK) * uint(DK) + head * uint(DV));
    T activated[4];
    float sumsq = 0.0f;
    T l2acc = T(0);
    for (uint i = 0; i < 4; ++i) {
        uint channel = channel_base + lane * 4 + i;
        float acc = 0.0f;
        for (uint tap = 0; tap < 4; ++tap) {
            uint input_row = row + tap;
            const T xv = input_row < uint(NKEEP)
                ? conv_state[(batch_idx * uint(NKEEP) + input_row) * uint(C) + channel]
                : qkv[(batch_idx * uint(S) + input_row - uint(NKEEP)) * uint(C) + channel];
            acc += float(xv) * float(conv_w[channel * 4 + tap]);
        }
        const T conv = T(acc);
        T sy = T(1) / (T(1) + metal::exp(metal::abs(conv)));
        const T act = conv * ((conv < T(0)) ? sy : T(1) - sy);
        activated[i] = act;
        if (L2) {
            const T sqv = T(float(act) * float(act));
            l2acc = T(float(l2acc) + float(sqv));
        } else {
            float value = float(act);
            sumsq += value * value;
        }
    }
    if (is_q || is_k) {
        uint out_base = ((batch_idx * uint(S) + row) * uint(HK) + head) * uint(DK) + lane * 4;
        if (L2) {
            // Stock Qwen4 L2 chain: x * rsqrt(sum(square(x), -1) + 1e-6),
            // with dk^-0.5 applied to q only.  mx.square rounds per
            // element; mx.sum accumulates each lane's four contiguous
            // bf16 values sequentially, reduces with an fp32 xor tree and
            // rounds once.  Mirror every rounding site exactly.
            float tv = float(l2acc);
            tv += simd_shuffle_xor(tv, short(16));
            tv += simd_shuffle_xor(tv, short(8));
            tv += simd_shuffle_xor(tv, short(4));
            tv += simd_shuffle_xor(tv, short(2));
            tv += simd_shuffle_xor(tv, short(1));
            const T eps = T(float(T(tv)) + float(T(1e-6f)));
            const T inv = T(metal::precise::rsqrt(float(eps)));
            for (uint i = 0; i < 4; ++i) {
                const T l2 = T(float(activated[i]) * float(inv));
                const T value = is_q ? T(float(l2) * float(q_scale)) : l2;
                if (is_q) {
                    q_out[out_base + i] = value;
                } else {
                    k_out[out_base + i] = value;
                }
            }
        } else {
            sumsq = simd_sum(sumsq);
            float inv = metal::precise::rsqrt(sumsq / float(DK) + 1e-6f);
            const T scale = is_q ? q_scale : k_scale;
            for (uint i = 0; i < 4; ++i) {
                const T rms = T(1) * T(float(activated[i]) * inv);
                const T value = scale * rms;
                if (is_q) {
                    q_out[out_base + i] = value;
                } else {
                    k_out[out_base + i] = value;
                }
            }
        }
    } else {
        uint out_base = ((batch_idx * uint(S) + row) * uint(HV) + head) * uint(DV) + lane * 4;
        for (uint i = 0; i < 4; ++i) {
            v_out[out_base + i] = activated[i];
        }
    }
    if (S < NKEEP && row == 0) {
        for (uint old_row = 0; old_row < uint(NKEEP - S); ++old_row) {
            uint dst = (batch_idx * uint(NKEEP) + old_row) * uint(C)
                       + channel_base + lane * 4;
            uint src = (batch_idx * uint(NKEEP) + old_row + uint(S)) * uint(C)
                       + channel_base + lane * 4;
            for (uint i = 0; i < 4; ++i) {
                conv_out[dst + i] = conv_state[src + i];
            }
        }
    }
    if (row + uint(NKEEP) >= uint(S)) {
        uint state_row = row + uint(NKEEP) - uint(S);
        uint raw_base = (batch_idx * uint(S) + row) * uint(C) + channel_base + lane * 4;
        uint state_base = (batch_idx * uint(NKEEP) + state_row) * uint(C) + channel_base + lane * 4;
        for (uint i = 0; i < 4; ++i) {
            conv_out[state_base + i] = qkv[raw_base + i];
        }
    }
"""


# Donor Qwen4 B1/T1 norm-gate (omlx ``_QWEN4_NORM_GATE_SOURCE``), verbatim.
_DECODE_NORM_GATE_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint head = threadgroup_position_in_grid.z;
    uint base = head * uint(DV) + lane * 4;
    float xs[4];
    float sumsq = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        xs[i] = float(y[base + i]);
        sumsq += xs[i] * xs[i];
    }
    sumsq = simd_sum(sumsq);
    float inv = metal::precise::rsqrt(sumsq / float(DV) + float(eps));
    for (uint i = 0; i < 4; ++i) {
        // mx.fast.rms_norm materializes BF16 before Qwen4 casts it back to
        // FP32 for the sigmoid product.
        const T normed = norm_w[lane * 4 + i] * T(xs[i] * inv);
        float zv = float(z[base + i]);
        float sy = 1.0f / (1.0f + metal::precise::exp(metal::abs(zv)));
        float sig = zv < 0.0f ? sy : 1.0f - sy;
        out[base + i] = T(float(normed) * sig);
    }
"""


#: q/k normalization contracts the prework kernel can reproduce.
#: ``qwen4``: Flash-Next's direct L2 chain (``qwen4_exp._normalize_qk``).
#: ``qwen35``: the shared Qwen3.5 chain (``gated_delta.normalize_gdn_qk``,
#: used by the Qwen3.8 27B): fast RMSNorm with ``eps = 1e-6 / DK`` on the mean,
#: then ``q * (DK**-0.5)**2`` and ``k * DK**-0.5`` in the activation dtype.
PREWORK_NUMERICS = ("qwen4", "qwen35")
#: Output-gate contracts the norm-gate kernel can reproduce.  ``sigmoid`` is
#: Qwen4's ``RMSNormGated`` (precise fp32 sigmoid); ``swish`` is the Qwen3.5
#: ``Qwen3NextRMSNormGated`` (compiled ``_precise_swiglu``: fp32 ``nn.silu``,
#: whose runtime-compiled sigmoid uses ``metal::exp``).
NORM_GATES = ("sigmoid", "swish")

# Qwen3.5 RMS-normalization epsilon on the mean of squares, spelled exactly as
# the qualified 27B decode kernel (qwen4_fused_gdn ``AGNES_NUMERICS``) spells
# it.  float32(1e-6) / 128 is exact and equals float32(1e-06 * (128**-0.5)**2),
# the value ``normalize_gdn_qk`` hands ``mx.fast.rms_norm``.
_QWEN35_RMS_EPS = "1.0e-6f / float(DK)"


def prefill_prework_source(numerics: str = "qwen4") -> str:
    """The verify prework with the row count read at run time.

    ``numerics="qwen4"`` is the L2 variant Flash-Next qualified (unchanged).
    ``numerics="qwen35"`` is the donor's RMS branch (``L2 = 0``) with the
    Qwen3.5 epsilon; the template ``L2`` selects the branch.
    """
    if numerics not in PREWORK_NUMERICS:
        raise ValueError(f"unsupported GDN prework numerics {numerics!r}")
    source = _VERIFY_PREWORK_SOURCE
    substitutions = [
        ("uint(NKEEP - S)", "uint(NKEEP) - S_rt"),
        ("S < NKEEP", "S_rt < uint(NKEEP)"),
        ("uint(S)", "S_rt"),
    ]
    if numerics == "qwen35":
        substitutions.append(
            (
                "rsqrt(sumsq / float(DK) + 1e-6f)",
                f"rsqrt(sumsq / float(DK) + {_QWEN35_RMS_EPS})",
            )
        )
    for old, new in substitutions:
        if old not in source:
            raise RuntimeError(f"GDN prework source changed: {old!r} missing")
        source = source.replace(old, new)
    return "    const uint S_rt = uint(s_len);\n" + source


def prefill_norm_gate_source(gate: str = "sigmoid") -> str:
    """The decode norm-gate addressed by (row, head) instead of head only.

    ``gate="swish"`` replaces the precise sigmoid gate with the Qwen3.5
    ``zv * sigmoid_fast(zv)`` gate, the form the qualified 27B decode kernel
    uses for the compiled ``_precise_swiglu``.
    """
    if gate not in NORM_GATES:
        raise ValueError(f"unsupported GDN output gate {gate!r}")
    old = "uint base = head * uint(DV) + lane * 4;"
    if old not in _DECODE_NORM_GATE_SOURCE:
        raise RuntimeError("GDN norm-gate source changed")
    source = _DECODE_NORM_GATE_SOURCE.replace(
        old,
        "uint base = (threadgroup_position_in_grid.y * uint(HV) + head) * uint(DV)"
        " + lane * 4;",
    )
    if gate == "swish":
        for old, new in (
            (
                "float sy = 1.0f / (1.0f + metal::precise::exp(metal::abs(zv)));",
                "float sy = 1.0f / (1.0f + metal::exp(metal::abs(zv)));",
            ),
            (
                "float sig = zv < 0.0f ? sy : 1.0f - sy;",
                "float sig = zv * (zv < 0.0f ? sy : 1.0f - sy);",
            ),
        ):
            if old not in source:
                raise RuntimeError(f"GDN norm-gate source changed: {old!r} missing")
            source = source.replace(old, new)
    return source


_KERNELS: dict = {}

# Kernel names per variant; the Flash-Next names are the historical ones.
_PREWORK_NAMES = {
    "qwen4": "mlx2_qwen4_gdn_prefill_prework",
    "qwen35": "mlx2_qwen35_gdn_prefill_prework",
}
_NORM_GATE_NAMES = {
    "sigmoid": "mlx2_qwen4_gdn_prefill_norm_gate",
    "swish": "mlx2_qwen35_gdn_prefill_norm_gate",
}


def _kernels(numerics: str = "qwen4", gate: str = "sigmoid"):
    key = (numerics, gate)
    if key not in _KERNELS:
        _KERNELS[key] = (
            mx.fast.metal_kernel(
                name=_PREWORK_NAMES[numerics],
                input_names=["qkv", "conv_state", "conv_w", "q_scale", "k_scale", "s_len"],
                output_names=["q_out", "k_out", "v_out", "conv_out"],
                source=prefill_prework_source(numerics),
                ensure_row_contiguous=True,
            ),
            mx.fast.metal_kernel(
                name=_NORM_GATE_NAMES[gate],
                input_names=["y", "z", "norm_w", "eps"],
                output_names=["out"],
                source=prefill_norm_gate_source(gate),
                ensure_row_contiguous=True,
            ),
        )
    return _KERNELS[key]


def qwen4_gdn_prefill_prework(
    qkv: mx.array,
    conv_state: Optional[mx.array],
    conv_weight: mx.array,
    *,
    num_key_heads: int = NUM_KEY_HEADS,
    num_value_heads: int = NUM_VALUE_HEADS,
    key_head_dim: int = KEY_HEAD_DIM,
    value_head_dim: int = VALUE_HEAD_DIM,
    numerics: str = "qwen4",
):
    """``(q, k, v, next_conv_state)``; callers must run admission first.

    ``qkv`` is the raw ``in_proj_qkv`` output ``(1, S, C)``; ``conv_state``
    ``(1, 3, C)`` or ``None`` (zeros, as the eager path).  Shapes out:
    q/k ``(1, S, HK, DK)``, v ``(1, S, HV, DV)``, conv ``(1, 3, C)``.
    ``numerics`` picks the q/k normalization contract (``PREWORK_NUMERICS``).
    """
    if numerics not in PREWORK_NUMERICS:
        raise ValueError(f"unsupported GDN prework numerics {numerics!r}")
    length = int(qkv.shape[1])
    conv_dim = int(qkv.shape[2])
    if conv_state is None:
        conv_state = mx.zeros((1, NKEEP, conv_dim), dtype=qkv.dtype)
    prework, _ = _kernels(numerics, "sigmoid" if numerics == "qwen4" else "swish")
    hk, hv, dk, dv = num_key_heads, num_value_heads, key_head_dim, value_head_dim
    if numerics == "qwen4":
        # q_l2 * k.shape[-1] ** -0.5 (qwen4_exp._normalize_qk); k unscaled.
        q_scale, k_scale = dk**-0.5, 1.0
    else:
        # normalize_gdn_qk: the same Python expressions, so the weak scalars
        # round to the activation dtype exactly as the eager multiplies do.
        inv_scale = dk ** (-0.5)
        q_scale, k_scale = inv_scale**2, inv_scale
    return tuple(
        prework(
            inputs=[
                qkv,
                conv_state,
                conv_weight,
                mx.array(q_scale, dtype=qkv.dtype),
                mx.array(k_scale, dtype=qkv.dtype),
                mx.array(length, dtype=mx.int32),
            ],
            template=[
                ("T", qkv.dtype),
                ("HK", hk),
                ("HV", hv),
                ("DK", dk),
                ("DV", dv),
                ("NKEEP", NKEEP),
                ("C", conv_dim),
                ("L2", 1 if numerics == "qwen4" else 0),
            ],
            grid=(32, length, 2 * hk + hv),
            threadgroup=(32, 1, 1),
            output_shapes=[
                (1, length, hk, dk),
                (1, length, hk, dk),
                (1, length, hv, dv),
                (1, NKEEP, conv_dim),
            ],
            output_dtypes=[qkv.dtype] * 4,
        )
    )


def qwen4_gdn_prefill_norm_gate(
    y: mx.array,
    z: mx.array,
    norm_weight: mx.array,
    eps: float,
    *,
    num_value_heads: int = NUM_VALUE_HEADS,
    value_head_dim: int = VALUE_HEAD_DIM,
    gate: str = "sigmoid",
) -> mx.array:
    """``RMSNormGated(y, z)`` flattened to ``(1, S, HV*DV)``.

    ``gate`` picks the output-gate contract (``NORM_GATES``).
    """
    if gate not in NORM_GATES:
        raise ValueError(f"unsupported GDN output gate {gate!r}")
    length = int(y.shape[1])
    hv, dv = num_value_heads, value_head_dim
    _, norm_gate = _kernels("qwen4" if gate == "sigmoid" else "qwen35", gate)
    return norm_gate(
        inputs=[y, z, norm_weight, mx.array(eps, dtype=mx.float32)],
        template=[("T", y.dtype), ("HV", hv), ("DV", dv)],
        grid=(32, length, hv),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, length, hv * dv)],
        output_dtypes=[y.dtype],
    )[0]


def _shape(value: Any) -> tuple:
    return tuple(getattr(value, "shape", ()))


#: architecture -> (prework numerics, output gate).  Only the Qwen3.8 27B
#: selects ``qwen38``; Qwen3.5 checkpoints with this geometry are not
#: qualified on this route and stay refused.
ARCHITECTURE_CONTRACTS = {
    "qwen4": ("qwen4", "sigmoid"),
    "qwen38": ("qwen35", "swish"),
}


LEFT_PADDING_NONZERO = "left padding (nonzero)"
LEFT_PADDING_UNKNOWN = "left padding (unknown on host)"


def _left_padding_refusal(
    left_padding: Any, host_left_padding: Optional[Sequence[int]]
) -> Optional[str]:
    """``None`` when the padding is host-known to be consumed on every row.

    Shape metadata only; the values come from the host mirror.  A mirror
    whose length disagrees with the array is treated as unknown.
    """
    if host_left_padding is None:
        return LEFT_PADDING_UNKNOWN
    try:
        host = [int(value) for value in host_left_padding]
    except (TypeError, ValueError):
        return LEFT_PADDING_UNKNOWN
    shape = _shape(left_padding)
    if len(shape) != 1 or shape[0] != len(host):
        return LEFT_PADDING_UNKNOWN
    if any(value > 0 for value in host):
        return LEFT_PADDING_NONZERO
    return None


def admit_qwen4_fused_gdn_prefill(
    *,
    qkv: Any,
    z: Any,
    b: Any,
    a: Any,
    conv_state: Any,
    recurrent_state: Any,
    conv_weight: Any,
    norm_weight: Any,
    mask: Any,
    has_cache: bool,
    lengths: Any,
    left_padding: Any,
    host_left_padding: Optional[Sequence[int]] = None,
    speculating: bool,
    training: bool,
    sharded: bool,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    conv_kernel: int,
    gate_activation: str,
    min_rows: int = PREFILL_MIN_ROWS,
    architecture: str = "qwen4",
) -> FusedGdnAdmission:
    """Pure structural admission; never evaluates an array.

    ``host_left_padding`` is the cache's host mirror of ``left_padding``
    (``ArraysCache.host_left_padding()``), or ``None`` when it is not known
    on the host.  A padding array is admitted only when the mirror covers
    every row and no row has padding still ahead (all ``<= 0``).

    ``architecture`` names the numerics contract: ``qwen4`` (Flash-Next:
    direct L2 q/k, sigmoid gate) or ``qwen38`` (Qwen3.8 27B: Qwen3.5 RMS q/k,
    swish gate, see ``ARCHITECTURE_CONTRACTS``).  Both share one geometry.
    """
    contract = ARCHITECTURE_CONTRACTS.get(architecture)
    if contract is None:
        return FusedGdnAdmission(False, f"unsupported architecture {architecture!r}")
    if training:
        return FusedGdnAdmission(False, "training")
    if sharded:
        return FusedGdnAdmission(False, "distributed sharding")
    if not has_cache:
        return FusedGdnAdmission(False, "no cache")
    if speculating:
        return FusedGdnAdmission(False, "speculative rollback")
    if mask is not None:
        return FusedGdnAdmission(False, "masked prefill")
    if lengths is not None:
        return FusedGdnAdmission(False, "ragged lengths")
    if left_padding is not None:
        refusal = _left_padding_refusal(left_padding, host_left_padding)
        if refusal is not None:
            return FusedGdnAdmission(False, refusal)
    if gate_activation != contract[1]:
        return FusedGdnAdmission(False, f"output gate {gate_activation!r}")
    geometry = (num_key_heads, num_value_heads, key_head_dim, value_head_dim, conv_kernel)
    expected_geometry = (
        NUM_KEY_HEADS,
        NUM_VALUE_HEADS,
        KEY_HEAD_DIM,
        VALUE_HEAD_DIM,
        CONV_KERNEL,
    )
    if geometry != expected_geometry:
        return FusedGdnAdmission(False, f"unsupported geometry {geometry}")
    qkv_shape = _shape(qkv)
    if len(qkv_shape) != 3:
        return FusedGdnAdmission(False, f"qkv rank {len(qkv_shape)}")
    if qkv_shape[0] != 1:
        return FusedGdnAdmission(False, f"batch of {qkv_shape[0]} rows")
    rows = int(qkv_shape[1])
    if rows < min_rows:
        return FusedGdnAdmission(False, f"rows {rows} < {min_rows}")
    expected = {
        "qkv": (1, rows, CONV_DIM),
        "z": (1, rows, VALUE_DIM),
        "b": (1, rows, NUM_VALUE_HEADS),
        "a": (1, rows, NUM_VALUE_HEADS),
        "conv_weight": (CONV_DIM, CONV_KERNEL, 1),
        "norm_weight": (VALUE_HEAD_DIM,),
    }
    values = {
        "qkv": qkv,
        "z": z,
        "b": b,
        "a": a,
        "conv_weight": conv_weight,
        "norm_weight": norm_weight,
    }
    for name, expected_shape in expected.items():
        if _shape(values[name]) != expected_shape:
            return FusedGdnAdmission(
                False, f"{name} shape {_shape(values[name])}, expected {expected_shape}"
            )
    for name in ("qkv", "z", "conv_weight", "norm_weight"):
        dtype = getattr(values[name], "dtype", None)
        if dtype != mx.bfloat16:
            return FusedGdnAdmission(False, f"{name} dtype {dtype}")
    if conv_state is not None:
        if _shape(conv_state) != (1, NKEEP, CONV_DIM):
            return FusedGdnAdmission(False, f"conv_state shape {_shape(conv_state)}")
        if conv_state.dtype != mx.bfloat16:
            return FusedGdnAdmission(False, f"conv_state dtype {conv_state.dtype}")
    if recurrent_state is not None:
        expected_state = (1, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM)
        if _shape(recurrent_state) != expected_state:
            return FusedGdnAdmission(
                False, f"recurrent_state shape {_shape(recurrent_state)}"
            )
        if recurrent_state.dtype not in (mx.float32, mx.float16):
            # The recurrence runs in the layer's own gated_delta_update, which
            # serves the fp16 storage class (gdn_state) too.
            return FusedGdnAdmission(False, "recurrent_state must be float32 or float16")
    return FusedGdnAdmission(True, "eligible")


__all__ = [
    "ARCHITECTURE_CONTRACTS",
    "LEFT_PADDING_NONZERO",
    "LEFT_PADDING_UNKNOWN",
    "NORM_GATES",
    "PREFILL_ENV",
    "PREFILL_MIN_ROWS",
    "PREWORK_NUMERICS",
    "admit_qwen4_fused_gdn_prefill",
    "prefill_enabled_from_env",
    "prefill_norm_gate_source",
    "prefill_prework_source",
    "qwen4_gdn_prefill_norm_gate",
    "qwen4_gdn_prefill_prework",
    "runtime_supported",
]
