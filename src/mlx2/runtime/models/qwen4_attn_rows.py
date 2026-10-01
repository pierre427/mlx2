# SPDX-License-Identifier: Apache-2.0 AND MIT
# Structure adapted from jundot/omlx PR #4052 (Apache-2.0) at d05afc7a; the
# arithmetic inside the Metal sources is transcribed from MLX 39400a0d4
# (``rms_norm.metal``, ``rope.metal``, ``sdpa_vector.h``; MIT).  See
# docs/PROVENANCE.md and provenance/omlx-4052-qwen4-attn-rows.json.
"""Fused Qwen4 (Flash-Next) attention rows for decode and short verify.

A one-token Flash-Next attention layer runs, besides the indexer, about a
dozen dependent launches around its projections: four projection matvecs, a
copy of the strided query half, two RMS norms, two copies and two RoPE
kernels (the rotary dims are a quarter of the head), MLX's SDPA (one or two
launches), the sigmoid of the gate and the gate multiply.  This module
replaces them with the same float operations in the same order:

* the four projections run as ONE stock ``quantized_matmul`` over the
  attention layer's concatenated table (``Attention._fused_projection_table``):
  at M < 8 MLX picks ``qmv``/``qmv_fast``/``qmv_wide`` by M alone and every
  output column accumulates independently of N, so the concatenated launch
  gives each column its separate-launch bits (``PROJECTION_MAX_ROWS``);
* ``prep_qk``: the q and k RMS norms (``rms_single_row``'s 64-thread layout
  for an axis of 256) and the non-traditional partial RoPE (``rope`` /
  ``rope_single``: ``exp2(-d * log2(base))`` frequencies, ``fast::cos`` /
  ``fast::sin``), with the unrotated dims copied, for every head and row in
  one launch that also de-interleaves the query from its gate half;
* ``sdpa_gate``: MLX's vector SDPA plan for this GPU class -- one pass below
  1024 keys, else two passes with ``select_sdpa_blocks`` partitions -- with
  the boolean mask MLX would read, then the multiply by the gate's sigmoid
  (``mx.sigmoid`` itself) folded into the closing reduction, written in the
  ``[B, R, H * D]`` layout ``o_proj`` reads.  The one-pass chains of a head
  run in separate simdgroups spread over the GPU (MLX keeps all 32 in one
  threadgroup) and meet again in the combine launch (omlx #4052's layout).

Only the arithmetic MLX would run is transcribed: MLX sends ``R * GQA > 32``
query rows (R >= 3 at GQA 12) to its unfused fallback, so those rows keep the
stock SDPA and take only the fused projections and ``prep_qk``.

Everything is default off (``MLX_QWEN4_ATTN_FUSED_ROWS``; the Flash-Next
policy field ``attn_fused_rows``) and every stage is admitted strictly; a
refusal is counted by reason and the caller keeps the MLX ops.
"""

from __future__ import annotations

import ctypes
import functools
import math
import os
import threading
from collections import Counter
from typing import Optional

import mlx.core as mx

ENV_NAME = "MLX_QWEN4_ATTN_FUSED_ROWS"
# M < 8: MLX's quantized_matmul stays on qmv/qmv_wide (NAX small-M qmv starts
# at M = 8 for large N, qmm below every vector limit >= 12 on M5).
PROJECTION_MAX_ROWS = 7
# Rows a call may carry into the fused path at all (the widest copy-draft
# verify is 17); SDPA narrows this further (``sdpa_plan``).
PREP_MAX_ROWS = 17
# MLX's vector SDPA serves R rows only while R * GQA <= 32.
_VECTOR_MAX_SIMDS = 32
_ONE_PASS_MAX_KEYS = 1024
# One-pass chains: query heads per simdgroup, simdgroups per threadgroup.
_ONE_PASS_HPT = 2
_ONE_PASS_SG = 4


def _parse_flag(raw: Optional[str]) -> bool:
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in {"1", "true", "on", "yes"}:
        return True
    if value in {"0", "false", "off", "no", ""}:
        return False
    raise ValueError(f"{ENV_NAME} must be 0/off or 1/on; got {raw!r}")


_ENABLED = _parse_flag(os.environ.get(ENV_NAME))
_LOCK = threading.Lock()
_COUNTS: Counter = Counter()
_KERNELS: dict = {}
_KERNEL_FAILED: set = set()


def enabled() -> bool:
    return _ENABLED


def set_enabled(value: bool) -> None:
    """Process switch (the policy sets it through the environment; in-process
    A/B harnesses flip it directly)."""
    global _ENABLED
    _ENABLED = bool(value)


def bump(name: str, amount: int = 1) -> None:
    with _LOCK:
        _COUNTS[name] += amount


def status(*, reset: bool = False) -> dict:
    with _LOCK:
        counts = dict(_COUNTS)
        if reset:
            _COUNTS.clear()
    return {
        "enabled": bool(_ENABLED),
        "device_class": _device_class(),
        "counts": counts,
        "kernel_failures": sorted(_KERNEL_FAILED),
    }


@functools.lru_cache(maxsize=None)
def _device_class() -> str:
    try:
        if not mx.metal.is_available():
            return ""
        return str(mx.device_info().get("architecture", ""))[-1:]
    except Exception:
        return ""


def metal_ready() -> bool:
    return bool(
        mx.default_device() == mx.gpu
        and hasattr(mx.fast, "metal_kernel")
        and _device_class() in ("s", "d")
    )


@functools.lru_cache(maxsize=None)
def _log2f(value: float) -> float:
    """``std::log2(float)`` as MLX's RoPE dispatch computes its base."""
    try:
        libm = ctypes.CDLL(None)
        fn = libm.log2f
        fn.restype = ctypes.c_float
        fn.argtypes = [ctypes.c_float]
        return float(fn(ctypes.c_float(value)))
    except Exception:  # pragma: no cover - libm always has log2f on macOS
        import numpy as np

        return float(np.log2(np.float32(value)))


# --------------------------------------------------------------------------
# SDPA plan (MLX 39400a0d4, scaled_dot_product_attention.cpp)
# --------------------------------------------------------------------------


def select_sdpa_blocks(devc: str, n_keys: int, n_simds: int) -> Optional[int]:
    """``select_sdpa_blocks`` for the 's' and 'd' GPU classes."""
    if devc == "s":
        blocks = 64
        if n_keys > 1024 and n_simds > 4:
            if n_keys <= 8192:
                blocks = 128
            elif n_keys <= 32768:
                blocks = 256
            elif n_keys <= 65536:
                blocks = 512
            else:
                blocks = 1024
        return blocks
    if devc == "d":
        blocks = 128
        if n_simds <= 2 and n_keys > 8192:
            blocks = 256
        elif n_simds >= 6:
            if 16384 <= n_keys < 65536:
                blocks = 512
            elif n_keys >= 65536:
                blocks = 1024
        return blocks
    return None


def sdpa_plan(n_keys: int, rows: int, heads: int, kv_heads: int, head_dim: int):
    """``(1, None)`` / ``(2, blocks)``: the vector kernel MLX runs for these
    ``rows`` query rows over ``n_keys`` keys on this GPU, or ``None`` when MLX
    would not run its vector kernel (or this module does not transcribe it)."""
    devc = _device_class()
    if devc not in ("s", "d") or os.environ.get("MLX_SDPA_BLOCKS"):
        return None
    if not kv_heads or heads % kv_heads:
        return None
    gqa = heads // kv_heads
    if head_dim != 256 or rows < 1 or rows > 8 or rows > n_keys:
        return None
    if rows * gqa > _VECTOR_MAX_SIMDS:
        return None
    if n_keys >= _ONE_PASS_MAX_KEYS:
        return 2, select_sdpa_blocks(devc, n_keys, gqa * rows)
    # (the GQA >= 4096-key clause of MLX's routing cannot fire below 1024)
    if gqa % _ONE_PASS_HPT:
        return None
    return 1, None


# --------------------------------------------------------------------------
# Metal sources
# --------------------------------------------------------------------------

# One threadgroup of 64 threads per (head, row, batch).  The first QH heads are
# queries (read from the interleaved q|gate projection), the rest keys.
# rms_single_row (axis 256, N_READS 4, two simdgroups) then rope / rope_single
# (non-traditional, dims ROT < D, the unrotated dims copied as MLX's rope
# dispatch copies its input first).
_PREP_SOURCE = r"""
    constexpr int SIMD_SIZE = 32;
    constexpr int N_READS = 4;
    const uint axis_size = D;
    const uint lid = thread_position_in_threadgroup.x;
    const uint simd_lane_id = thread_index_in_simdgroup;
    const uint simd_group_id = simdgroup_index_in_threadgroup;
    const int h = threadgroup_position_in_grid.x;
    const int t = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;
    const int rows = threadgroups_per_grid.y;
    const bool is_q = h < QH;

    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[SIMD_SIZE];
    threadgroup T normed[D];

    const device T* x = is_q
        ? qg + b * qg_strides[0] + t * qg_strides[1] + h * (2 * D)
        : kf + b * kf_strides[0] + t * kf_strides[1] + (h - QH) * D;
    const device T* w = is_q ? q_weight : k_weight;

    float acc = 0;
    float thread_x[N_READS];
    x += lid * N_READS;
    w += lid * N_READS;
    for (int i = 0; i < N_READS; i++) {
        thread_x[i] = x[i];
        acc += thread_x[i] * thread_x[i];
    }
    acc = simd_sum(acc);
    if (simd_group_id == 0) {
        local_sums[simd_lane_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        local_sums[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) {
            local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps[0]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < N_READS; i++) {
        normed[lid * N_READS + i] =
            w[i] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device T* out = is_q
        ? q_out + ((size_t)(b * QH + h) * rows + t) * D
        : k_out + ((size_t)(b * KH + (h - QH)) * rows + t) * D;
    constexpr int HALF = ROT / 2;
    if (lid < HALF) {
        const int pos_x = lid;
        float L = scale[0] * static_cast<float>(t + offset[b * OFFSET_STRIDE]);
        float d = static_cast<float>(pos_x) / static_cast<float>(HALF);
        float inv_freq = metal::exp2(-d * base[0]);
        float theta = L * inv_freq;
        float costheta = metal::fast::cos(theta);
        float sintheta = metal::fast::sin(theta);
        float x1 = static_cast<float>(normed[pos_x]);
        float x2 = static_cast<float>(normed[pos_x + HALF]);
        float rx1 = x1 * costheta - x2 * sintheta;
        float rx2 = x1 * sintheta + x2 * costheta;
        out[pos_x] = static_cast<T>(rx1);
        out[pos_x + HALF] = static_cast<T>(rx2);
    }
    for (int j = ROT + int(lid); j < D; j += 64) {
        out[j] = normed[j];
    }
"""

# The per-key update both passes share (sdpa_vector / sdpa_vector_2pass_1).
_KEY_STEP = r"""
#define ATTN_ROWS_KEY_STEP(QV, KP, VP, MAXV, SUMV, OV)                    \
    {                                                                    \
        U score = 0;                                                     \
        for (int e = 0; e < PT; e++) {                                   \
            score += QV[e] * KP[e];                                      \
        }                                                                \
        score = simd_sum(score);                                         \
        U new_max = max(MAXV, score);                                    \
        U factor = fast::exp(MAXV - new_max);                            \
        U exp_score = fast::exp(score - new_max);                        \
        MAXV = new_max;                                                  \
        SUMV = SUMV * factor + exp_score;                                \
        for (int e = 0; e < PT; e++) {                                   \
            OV[e] = OV[e] * factor + exp_score * VP[e];                  \
        }                                                                \
    }
"""

# sdpa_vector: simdgroup s of a head's threadgroup visits keys s, s + 32, ...
# Here chain s of HPT query heads (sharing one KV head) runs in its own
# simdgroup; its state goes to the combine launch.
_ONE_PASS_CHAINS = r"""
    constexpr int BN = 32;
    constexpr int PT = D / 32;
    typedef float U;
    const uint lane = thread_index_in_simdgroup;
    const uint f = threadgroup_position_in_grid.x * SG + simdgroup_index_in_threadgroup;
    const int s = int(f % BN);
    const int hg = int(f / BN);
    const int kv = threadgroup_position_in_grid.y;
    const int row = threadgroup_position_in_grid.z;
    const int rows = threadgroups_per_grid.z;
    const int H = GQA * int(threadgroups_per_grid.y);
    const int N = int(keys_shape[2]);
    const float sc = scale[0];

    U q[HPT][PT];
    U o[HPT][PT];
    U max_score[HPT];
    U sum_exp_score[HPT];
    for (int j = 0; j < HPT; j++) {
        const int head = kv * GQA + hg * HPT + j;
        const device T* qp = queries + (size_t)head * queries_strides[1]
            + (size_t)row * queries_strides[2] + lane * PT;
        for (int e = 0; e < PT; e++) {
            q[j][e] = static_cast<U>(sc) * qp[e];
            o[j][e] = 0;
        }
        max_score[j] = Limits<U>::finite_min;
        sum_exp_score[j] = 0;
    }
    const device T* kp = keys + kv * keys_strides[1] + s * keys_strides[2] + lane * PT;
    const device T* vp = values + kv * values_strides[1] + s * values_strides[2] + lane * PT;
    const int k_step = BN * int(keys_strides[2]);
    const int v_step = BN * int(values_strides[2]);
@MASK_SETUP(s, BN)@
    for (int i = s; i < N; i += BN) {
        if (@USE_KEY@) {
            for (int j = 0; j < HPT; j++) {
                ATTN_ROWS_KEY_STEP(q[j], kp, vp, max_score[j], sum_exp_score[j], o[j])
            }
        }
        kp += k_step;
        vp += v_step;
        @MASK_ADVANCE@
    }

    for (int j = 0; j < HPT; j++) {
        const int head = kv * GQA + hg * HPT + j;
        const size_t c = (size_t)(row * H + head) * BN + s;
        if (lane == 0) {
            maxs[c] = max_score[j];
            sums[c] = sum_exp_score[j];
        }
        // [row, head, dim, chain]: the combine reads one dim's 32 chains.
        for (int e = 0; e < PT; e++) {
            outs[((size_t)(row * H + head) * D + lane * PT + e) * BN + s] = o[j][e];
        }
    }
"""

# sdpa_vector's closing reduction for one (head, row), then the gate.
_ONE_PASS_COMBINE = r"""
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int PT = D / BD;
    typedef float U;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int head = threadgroup_position_in_grid.x;
    const int row = threadgroup_position_in_grid.y;
    const int H = threadgroups_per_grid.x;
    const size_t c0 = (size_t)(row * H + head) * BN;

    U max_score = maxs[c0 + simd_lid];
    U new_max = simd_max(max_score);
    U factor = fast::exp(max_score - new_max);
    U sum_exp_score = simd_sum(sums[c0 + simd_lid] * factor);
    U o[PT];
    for (int i = 0; i < PT; i++) {
        // MLX's outputs[simd_gid * BD + simd_lid]: dim simd_gid * PT + i of
        // chain simd_lid.
        o[i] = simd_sum(
            outs[((size_t)(row * H + head) * D + simd_gid * PT + i) * BN + simd_lid] * factor);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    }
    if (simd_lid == 0) {
        const size_t base = (size_t)row * (H * D) + head * D + simd_gid * PT;
        for (int i = 0; i < PT; i++) {
@STORE@
        }
    }
"""

# sdpa_vector_2pass_1: threadgroup (KV head, block), simdgroup (query head g of
# that KV head, row); keys block, block + BLOCKS, ...
_TWO_PASS_CHAINS = r"""
    constexpr int PT = D / 32;
    typedef float U;
    const uint lane = thread_index_in_simdgroup;
    const int kv = threadgroup_position_in_grid.x;
    const int block = threadgroup_position_in_grid.z;
    const int g = thread_position_in_threadgroup.y;
    const int row = thread_position_in_threadgroup.z;
    const int rows = threads_per_threadgroup.z;
    const int head = GQA * kv + g;
    const int N = int(keys_shape[2]);
    const float sc = scale[0];
    const int o_offset = head * rows + row;

    U q[PT];
    U o[PT] = {0};
    const device T* qp = queries + (size_t)head * queries_strides[1]
        + (size_t)row * queries_strides[2] + lane * PT;
    for (int e = 0; e < PT; e++) {
        q[e] = static_cast<U>(sc) * qp[e];
    }
    U max_score = Limits<U>::finite_min;
    U sum_exp_score = 0;
    const device T* kp = keys + kv * keys_strides[1] + block * keys_strides[2] + lane * PT;
    const device T* vp = values + kv * values_strides[1] + block * values_strides[2] + lane * PT;
    const int k_step = BLOCKS * int(keys_strides[2]);
    const int v_step = BLOCKS * int(values_strides[2]);
@MASK_SETUP(block, BLOCKS)@
    for (int i = block; i < N; i += BLOCKS) {
        if (@USE_KEY@) {
            ATTN_ROWS_KEY_STEP(q, kp, vp, max_score, sum_exp_score, o)
        }
        kp += k_step;
        vp += v_step;
        @MASK_ADVANCE@
    }
    const size_t c = (size_t)o_offset * BLOCKS + block;
    if (lane == 0) {
        sums[c] = sum_exp_score;
        maxs[c] = max_score;
    }
    for (int e = 0; e < PT; e++) {
        partials[c * D + lane * PT + e] = static_cast<T>(o[e]);
    }
"""

# sdpa_vector_2pass_2 for one (head, row), then the gate.
_TWO_PASS_COMBINE = r"""
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int elem_per_thread = D / BD;
    typedef float U;

    thread U o[elem_per_thread] = {0};
    threadgroup U outputs[BN * BD];

    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int head = threadgroup_position_in_grid.x;
    const int row = threadgroup_position_in_grid.y;
    const int H = threadgroups_per_grid.x;
    const int rows = threadgroups_per_grid.y;
    const size_t q_offset = (size_t)head * rows + row;
    const device T* pp = partials + q_offset * BLOCKS * D + simd_gid * D
        + simd_lid * elem_per_thread;
    const device float* sp = sums + q_offset * BLOCKS;
    const device float* mp = maxs + q_offset * BLOCKS;

    U sum_exp_score = 0.0;
    U max_score = Limits<U>::finite_min;

    for (int b = 0; b < BLOCKS / BN; ++b) {
        max_score = max(max_score, mp[simd_lid + BN * b]);
    }
    max_score = simd_max(max_score);

    for (int b = 0; b < BLOCKS / BN; ++b) {
        U factor = fast::exp(mp[simd_lid + BN * b] - max_score);
        sum_exp_score += factor * sp[simd_lid + BN * b];
    }
    sum_exp_score = simd_sum(sum_exp_score);

    for (int b = 0; b < BLOCKS / BN; ++b) {
        U factor = fast::exp(mp[simd_gid] - max_score);
        for (int i = 0; i < elem_per_thread; i++) {
            o[i] += factor * static_cast<U>(pp[i]);
        }
        mp += BN;
        sp += BN;
        pp += BN * D;
    }

    for (int i = 0; i < elem_per_thread; i++) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        const size_t base = (size_t)row * (H * D) + head * D + simd_gid * elem_per_thread;
        for (int i = 0; i < elem_per_thread; i++) {
@STORE@
        }
    }
"""


def _sdpa_source(template: str, *, masked: bool, gated: bool, chain_var: str = "", step: str = "") -> str:
    """Specialize a chain/combine source: MLX's ``use_key = bmask[0]`` walk
    (with its broadcast strides) when masked, and the gate multiply."""
    if masked:
        setup = (
            "const int m_row = mask_shape[2] > 1 ? int(mask_strides[2]) : 0;\n"
            "    const int m_key = mask_shape[3] > 1 ? int(mask_strides[3]) : 0;\n"
            f"    const device bool* mp = mask + row * m_row + {chain_var} * m_key;\n"
            f"    const int m_step = {step} * m_key;"
        )
        use_key, advance = "mp[0]", "mp += m_step;"
    else:
        setup, use_key, advance = "", "true", ""
    store = (
        "out[base + i] = static_cast<T>(o[i]) * gate[base + i];"
        if gated
        else "out[base + i] = static_cast<T>(o[i]);"
    )
    source = template
    if chain_var:
        source = source.replace(f"@MASK_SETUP({chain_var}, {step})@", setup)
    return (
        source.replace("@USE_KEY@", use_key)
        .replace("@MASK_ADVANCE@", advance)
        .replace("@STORE@", store)
    )


def _kernel(name: str):
    kernel = _KERNELS.get(name)
    if kernel is not None:
        return kernel
    masked = name.endswith("_mask")
    gated = name.endswith("_gate")
    if name == "prep":
        kernel = mx.fast.metal_kernel(
            name="mlx2_qwen4_attn_rows_prep_qk",
            input_names=["qg", "kf", "q_weight", "k_weight", "eps", "offset", "base", "scale"],
            output_names=["q_out", "k_out"],
            source=_PREP_SOURCE,
            ensure_row_contiguous=False,
        )
    elif name.startswith("one_chains"):
        kernel = mx.fast.metal_kernel(
            name="mlx2_qwen4_attn_rows_sdpa1_chains" + ("_mask" if masked else ""),
            input_names=["queries", "keys", "values", "scale"] + (["mask"] if masked else []),
            output_names=["maxs", "sums", "outs"],
            header=_KEY_STEP,
            source=_sdpa_source(
                _ONE_PASS_CHAINS, masked=masked, gated=False, chain_var="s", step="BN"
            ),
            ensure_row_contiguous=False,
        )
    elif name.startswith("two_chains"):
        kernel = mx.fast.metal_kernel(
            name="mlx2_qwen4_attn_rows_sdpa2_chains" + ("_mask" if masked else ""),
            input_names=["queries", "keys", "values", "scale"] + (["mask"] if masked else []),
            output_names=["partials", "sums", "maxs"],
            header=_KEY_STEP,
            source=_sdpa_source(
                _TWO_PASS_CHAINS, masked=masked, gated=False, chain_var="block", step="BLOCKS"
            ),
            ensure_row_contiguous=False,
        )
    elif name.startswith("one_combine"):
        kernel = mx.fast.metal_kernel(
            name="mlx2_qwen4_attn_rows_sdpa1_combine" + ("_gate" if gated else ""),
            input_names=["maxs", "sums", "outs"] + (["gate"] if gated else []),
            output_names=["out"],
            source=_sdpa_source(_ONE_PASS_COMBINE, masked=False, gated=gated),
        )
    elif name.startswith("two_combine"):
        kernel = mx.fast.metal_kernel(
            name="mlx2_qwen4_attn_rows_sdpa2_combine" + ("_gate" if gated else ""),
            input_names=["partials", "sums", "maxs"] + (["gate"] if gated else []),
            output_names=["out"],
            source=_sdpa_source(_TWO_PASS_COMBINE, masked=False, gated=gated),
        )
    else:  # pragma: no cover
        raise KeyError(name)
    _KERNELS[name] = kernel
    return kernel


_SCALARS: dict = {}


def _scalar(value, dtype) -> mx.array:
    key = (float(value) if dtype == mx.float32 else int(value), dtype)
    array = _SCALARS.get(key)
    if array is None:
        array = mx.array([value], dtype=dtype)
        _SCALARS[key] = array
    return array


# --------------------------------------------------------------------------
# Stage 2: q/k RMS norms + RoPE
# --------------------------------------------------------------------------


def prep_qk_supported(
    qg: mx.array,
    k_flat: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
    *,
    heads: int,
    kv_heads: int,
    head_dim: int,
    rope,
) -> Optional[str]:
    """``None`` when ``prep_qk`` reproduces ``q_norm``/``k_norm`` + ``rope``
    for these inputs, else the refusal reason."""
    if not metal_ready():
        return "device"
    if qg.dtype not in (mx.bfloat16, mx.float16) or k_flat.dtype != qg.dtype:
        return "dtype"
    if q_weight.dtype != qg.dtype or k_weight.dtype != qg.dtype:
        return "norm_weight_dtype"
    if q_weight.shape != (head_dim,) or k_weight.shape != (head_dim,):
        return "norm_weight_shape"
    if head_dim != 256:
        return "head_dim"
    if qg.ndim != 3 or k_flat.ndim != 3:
        return "rank"
    if qg.shape[-1] != heads * 2 * head_dim or k_flat.shape[-1] != kv_heads * head_dim:
        return "width"
    # The inner (per-row) axis must be unit-stride; row/batch strides are read.
    if rope is None or type(rope).__name__ != "RoPE":
        return "rope_type"
    dims = int(getattr(rope, "dims", 0))
    if getattr(rope, "traditional", True) or dims <= 0 or dims % 2 or dims > head_dim:
        return "rope_layout"
    if dims > 128 or dims < 2:
        return "rope_dims"
    if not math.isfinite(float(getattr(rope, "base", float("nan")))):
        return "rope_base"
    return None


def prep_qk(
    qg: mx.array,
    k_flat: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
    eps: float,
    offset,
    *,
    heads: int,
    kv_heads: int,
    rope,
):
    """``(queries [B, H, R, D], keys [B, KVH, R, D])`` exactly as
    ``rope(q_norm(q).transpose(0, 2, 1, 3), offset)`` and the same for ``k``
    compute them, where ``q`` is the first half of each head's ``2 * D``
    slice of ``qg [B, R, H * 2 * D]`` and ``k_flat`` is ``[B, R, KVH * D]``.
    ``offset`` is an int or an int array of 1 or B entries."""
    batch, rows, _ = qg.shape
    head_dim = k_flat.shape[-1] // kv_heads
    dtype = qg.dtype
    if isinstance(offset, mx.array):
        offsets = offset.astype(mx.int32) if offset.dtype != mx.int32 else offset
        offsets = offsets.reshape(-1)
        stride = 0 if offsets.size == 1 else 1
    else:
        offsets = mx.array([int(offset)], dtype=mx.int32)
        stride = 0
    queries, keys = _kernel("prep")(
        inputs=[
            qg,
            k_flat,
            q_weight,
            k_weight,
            _scalar(float(eps), mx.float32),
            offsets,
            _scalar(_log2f(float(rope.base)), mx.float32),
            _scalar(float(rope.scale), mx.float32),
        ],
        template=[
            ("T", dtype),
            ("QH", heads),
            ("KH", kv_heads),
            ("D", head_dim),
            ("ROT", int(rope.dims)),
            ("OFFSET_STRIDE", stride),
        ],
        grid=(64 * (heads + kv_heads), rows, batch),
        threadgroup=(64, 1, 1),
        output_shapes=[(batch, heads, rows, head_dim), (batch, kv_heads, rows, head_dim)],
        output_dtypes=[dtype, dtype],
    )
    return queries, keys


# --------------------------------------------------------------------------
# Stage 3: SDPA + gate
# --------------------------------------------------------------------------


def sdpa_supported(queries, keys, values, mask) -> Optional[str]:
    """``None`` when ``sdpa_gate`` reproduces MLX's SDPA for these inputs."""
    if not metal_ready():
        return "device"
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return "rank"
    batch, heads, rows, head_dim = queries.shape
    if batch != 1 or keys.shape[0] != 1 or values.shape[0] != 1:
        return "batch"
    if queries.dtype not in (mx.bfloat16, mx.float16):
        return "dtype"
    if keys.dtype != queries.dtype or values.dtype != queries.dtype:
        return "kv_dtype"
    if keys.shape[-1] != head_dim or values.shape[-1] != head_dim:
        return "kv_width"
    if keys.shape[1] != values.shape[1] or keys.shape[2] != values.shape[2]:
        return "kv_shape"
    if sdpa_plan(keys.shape[2], rows, heads, keys.shape[1], head_dim) is None:
        return "plan"
    if mask is not None:
        if isinstance(mask, str) or not isinstance(mask, mx.array):
            return "mask_kind"
        if mask.dtype != mx.bool_:
            return "mask_dtype"
        if mask.ndim > 4:
            return "mask_rank"
        shape = (1,) * (4 - mask.ndim) + tuple(mask.shape)
        if shape[0] != 1 or shape[1] != 1:
            return "mask_heads"
        if shape[2] not in (1, rows) or shape[3] not in (1, keys.shape[2]):
            return "mask_shape"
    return None


def sdpa_gate(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    mask: Optional[mx.array] = None,
    gate: Optional[mx.array] = None,
) -> mx.array:
    """``[B, R, H * D]``: MLX's ``scaled_dot_product_attention(queries, keys,
    values, scale, mask)`` transposed to ``[B, R, H, D]`` and flattened, times
    ``gate`` (``[B, R, H * D]``, already the sigmoid) when given.  ``queries``
    are ``[1, H, R, D]`` with a unit-stride last axis; ``keys``/``values`` are cache
    views ``[1, KVH, N, D]`` with a unit-stride last axis."""
    _, heads, rows, head_dim = queries.shape
    kv_heads = keys.shape[1]
    n_keys = keys.shape[2]
    gqa = heads // kv_heads
    dtype = queries.dtype
    plan, blocks = sdpa_plan(n_keys, rows, heads, kv_heads, head_dim)
    scale_array = _scalar(float(scale), mx.float32)
    mask_inputs = []
    if mask is not None:
        # MLX broadcasts the mask to [B, H, R, N] and reads the row and key
        # strides of the axes that are not singletons.
        mask_inputs = [mask.reshape((1,) * (4 - mask.ndim) + tuple(mask.shape))]
    gate_inputs = [] if gate is None else [gate.reshape(rows, heads * head_dim)]
    gate_suffix = "" if gate is None else "_gate"
    mask_suffix = "_mask" if mask is not None else ""
    if plan == 1:
        hpt, sg = _ONE_PASS_HPT, _ONE_PASS_SG
        chains = 32 * (gqa // hpt)
        maxs, sums, outs = _kernel("one_chains" + mask_suffix)(
            inputs=[queries, keys, values, scale_array] + mask_inputs,
            template=[
                ("T", dtype),
                ("D", head_dim),
                ("GQA", gqa),
                ("HPT", hpt),
                ("SG", sg),
            ],
            grid=(32 * chains, kv_heads, rows),
            threadgroup=(32 * sg, 1, 1),
            output_shapes=[
                (rows * heads * 32,),
                (rows * heads * 32,),
                (rows * heads * head_dim * 32,),
            ],
            output_dtypes=[mx.float32] * 3,
        )
        out = _kernel("one_combine" + gate_suffix)(
            inputs=[maxs, sums, outs] + gate_inputs,
            template=[("T", dtype), ("D", head_dim)],
            grid=(heads * 1024, rows, 1),
            threadgroup=(1024, 1, 1),
            output_shapes=[(rows, heads * head_dim)],
            output_dtypes=[dtype],
        )[0]
        return out.reshape(1, rows, heads * head_dim)
    partials, sums, maxs = _kernel("two_chains" + mask_suffix)(
        inputs=[queries, keys, values, scale_array] + mask_inputs,
        template=[
            ("T", dtype),
            ("D", head_dim),
            ("GQA", gqa),
            ("BLOCKS", blocks),
        ],
        grid=(32 * kv_heads, gqa, blocks * rows),
        threadgroup=(32, gqa, rows),
        output_shapes=[
            (heads * rows * blocks * head_dim,),
            (heads * rows * blocks,),
            (heads * rows * blocks,),
        ],
        output_dtypes=[dtype, mx.float32, mx.float32],
    )
    out = _kernel("two_combine" + gate_suffix)(
        inputs=[partials, sums, maxs] + gate_inputs,
        template=[
            ("T", dtype),
            ("D", head_dim),
            ("BLOCKS", blocks),
        ],
        grid=(heads * 1024, rows, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(rows, heads * head_dim)],
        output_dtypes=[dtype],
    )[0]
    return out.reshape(1, rows, heads * head_dim)
