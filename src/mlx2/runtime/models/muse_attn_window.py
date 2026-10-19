# SPDX-License-Identifier: MIT
"""Muse-owned D128 row-exact attention windows.

The served Muse reference is one MLX vector-SDPA call per token row.  Each
row therefore has a different *physical* reduction domain: row ``r`` reads
exactly ``n_first + r`` keys.  A native multi-query call over the final K/V
width does not preserve that contract merely by masking its future keys; it
changes the kernel specialization and was rejected by the AST geometry gate.

This module transcribes the relevant MLX 0.32.4 vector-SDPA arithmetic into
Muse-owned D128 kernels.  The only window change is to place independent
serial rows on the grid and set ``N = n_first + row`` in the source.  Runs are
split whenever the one-row MLX plan changes.  Optional boolean masks retain
the concatenated one-token mask rows; all-valid rows stay on the unmasked
source, exactly as ordinary Muse decode does.

The source proof is MLX ``scaled_dot_product_attention.cpp`` and
``sdpa_vector.h`` at the revision and hashes recorded below (MIT).  This file
does not import, widen, or dispatch through the separately evidenced Qwen
D256 kernel.  Runtime use remains fail-closed behind Muse's build-, device-
and source-bound bitwise canary, and policy selection remains default off.
"""

from __future__ import annotations

import hashlib
import inspect
import os
from functools import cache

import mlx.core as mx

MLX_SOURCE_REVISION = "39400a0d4cf1641bc6e543cb5e3585757f491854"
MLX_SDPA_DISPATCH_SHA256 = (
    "575e29a36b2d0c653f17b4f55062b644243d2e3dbc362f6725686bef959dd9dd"
)
# Content identity of the MLX vector-SDPA arithmetic transcribed here.  Kept
# separate from the dispatch hash so a source update cannot retain a stale
# numerical verdict by changing only one side of the proof.
MLX_SDPA_VECTOR_SHA256 = (
    "2100a4d1eaa8a524c5147c82c771cad75197495c72daffa03e7ea4c259aebf10"
)
ALGORITHM = "muse-d128-variable-n-v2"
HEAD_DIM = 128
MAX_ROWS = 16
MAX_VECTOR_SIMDS = 32
TWO_PASS_KEYS = 1024
_ONE_PASS_HPT = 2
_ONE_PASS_SG = 4


class ShapeOnly:
    """Array geometry for complete admission before the first live append."""

    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.ndim = len(self.shape)


@cache
def _device_class() -> str:
    try:
        if not mx.metal.is_available():
            return ""
        return str(mx.device_info().get("architecture", ""))[-1:]
    except (AttributeError, RuntimeError, ValueError):
        return ""


def _select_blocks(device_class: str, n_keys: int, n_simds: int) -> int | None:
    """MLX 0.32.4 ``select_sdpa_blocks`` for local s/d device classes."""
    if device_class == "s":
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
    if device_class == "d":
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


def _serial_plan(device_class: str, n_keys: int, gqa: int):
    """The MLX vector plan of one ordinary Muse token row."""
    if device_class not in ("s", "d") or n_keys < 1 or not 1 <= gqa <= 32:
        return None
    if n_keys < TWO_PASS_KEYS:
        # The one-pass transcription assigns two query heads per simdgroup.
        return (1, None) if gqa % _ONE_PASS_HPT == 0 else None
    return (2, _select_blocks(device_class, n_keys, gqa))


def row_plans(
    n_first: int,
    rows: int,
    heads: int,
    kv_heads: int,
    *,
    device_class: str | None = None,
):
    """One-row plan for each exact physical reduction domain."""
    device_class = _device_class() if device_class is None else device_class
    if not kv_heads or heads % kv_heads:
        return ()
    gqa = heads // kv_heads
    return tuple(
        _serial_plan(device_class, n_first + row, gqa) for row in range(rows)
    )


def _plan_runs(plans):
    """Maximal consecutive row runs with the same ordinary one-row plan."""
    runs = []
    start = 0
    for row in range(1, len(plans) + 1):
        if row == len(plans) or plans[row] != plans[start]:
            runs.append((start, row, plans[start]))
            start = row
    return tuple(runs)


def plan_signature(
    n_first: int,
    rows: int,
    heads: int,
    kv_heads: int,
    *,
    device_class: str | None = None,
):
    """Per-row serial plans grouped only at exact plan boundaries."""
    plans = row_plans(
        n_first, rows, heads, kv_heads, device_class=device_class
    )
    if not plans or any(plan is None for plan in plans):
        return ()
    return _plan_runs(plans)


def window_supported(queries, keys, values, n_first: int) -> str | None:
    if mx.default_device() != mx.gpu or _device_class() not in ("s", "d"):
        return "device"
    if not hasattr(mx.fast, "metal_kernel"):
        return "metal_kernel"
    if os.environ.get("MLX_SDPA_BLOCKS"):
        return "mlx_sdpa_blocks"
    # The ordinary Muse call can opt into a different experimental decoder.
    # Do not compare or replace that arithmetic through this capability.
    if os.environ.get("MLX2_FP_DECODE_KERNEL", "0") == "1":
        return "fp_decode_override"
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return "rank"
    batch, heads, rows, head_dim = queries.shape
    if batch != 1 or keys.shape[0] != 1 or values.shape[0] != 1:
        return "batch"
    if head_dim != HEAD_DIM or keys.shape[-1] != HEAD_DIM:
        return "head_dim"
    if not 2 <= rows <= MAX_ROWS:
        return "rows"
    if queries.dtype not in (mx.bfloat16, mx.float16):
        return "dtype"
    if keys.dtype != queries.dtype or values.dtype != queries.dtype:
        return "kv_dtype"
    if (
        keys.shape[1] != values.shape[1]
        or keys.shape[2:] != values.shape[2:]
    ):
        return "kv_shape"
    if n_first < 1 or keys.shape[2] < n_first + rows - 1:
        return "key_count"
    if not plan_signature(n_first, rows, heads, keys.shape[1]):
        return "plan"
    return None


# The per-key update in MLX's one- and two-pass vector-SDPA sources.  The
# operation order and float accumulators are intentionally literal.
_KEY_STEP = r"""
#define MUSE_D128_KEY_STEP(QV, KP, VP, MAXV, SUMV, OV)                   \
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


_ONE_PASS = r"""
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
    const int N = n_first[0] + row;
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
                MUSE_D128_KEY_STEP(q[j], kp, vp, max_score[j], sum_exp_score[j], o[j])
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
        for (int e = 0; e < PT; e++) {
            outs[((size_t)(row * H + head) * D + lane * PT + e) * BN + s] = o[j][e];
        }
    }
"""


_ONE_COMBINE = r"""
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
        o[i] = simd_sum(
            outs[((size_t)(row * H + head) * D + simd_gid * PT + i) * BN + simd_lid] * factor);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    }
    if (simd_lid == 0) {
        const size_t base = (size_t)row * (H * D) + head * D + simd_gid * PT;
        for (int i = 0; i < PT; i++) {
            out[base + i] = static_cast<T>(o[i]);
        }
    }
"""


_TWO_PASS = r"""
    constexpr int PT = D / 32;
    typedef float U;
    const uint lane = thread_index_in_simdgroup;
    const int kv = threadgroup_position_in_grid.x;
    const int block = int(threadgroup_position_in_grid.z) % BLOCKS;
    const int g = thread_position_in_threadgroup.y;
    const int row = int(threadgroup_position_in_grid.z) / BLOCKS;
    const int rows = int(threadgroups_per_grid.z) / BLOCKS;
    const int head = GQA * kv + g;
    const int N = n_first[0] + row;
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
            MUSE_D128_KEY_STEP(q, kp, vp, max_score, sum_exp_score, o)
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


_TWO_COMBINE = r"""
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
            out[base + i] = static_cast<T>(o[i]);
        }
    }
"""


def _masked_source(source: str, chain: str, step: str, masked: bool) -> str:
    if masked:
        setup = (
            "auto mp = masks + mask_base[0] + row * n_first[0]"
            f" + (row * (row - 1)) / 2 + {chain};\n"
            f"    const int m_step = {step};"
        )
        use_key, advance = "mp[0]", "mp += m_step;"
    else:
        setup, use_key, advance = "", "true", ""
    return (
        source.replace(f"@MASK_SETUP({chain}, {step})@", setup)
        .replace("@USE_KEY@", use_key)
        .replace("@MASK_ADVANCE@", advance)
    )


def _one_pass_source(masked: bool) -> str:
    return _masked_source(_ONE_PASS, "s", "BN", masked)


def _two_pass_source(masked: bool) -> str:
    return _masked_source(_TWO_PASS, "block", "BLOCKS", masked)


_KERNELS: dict = {}


def _kernel(name: str):
    kernel = _KERNELS.get(name)
    if kernel is not None:
        return kernel
    masked = name.endswith("_mask")
    extra = ["masks", "mask_base"] if masked else []
    if name.startswith("one_chains"):
        kernel = mx.fast.metal_kernel(
            name="mlx2_muse_d128_window_sdpa1_chains"
            + ("_mask" if masked else ""),
            input_names=["queries", "keys", "values", "scale", "n_first"]
            + extra,
            output_names=["maxs", "sums", "outs"],
            header=_KEY_STEP,
            source=_one_pass_source(masked),
            ensure_row_contiguous=False,
        )
    elif name.startswith("two_chains"):
        kernel = mx.fast.metal_kernel(
            name="mlx2_muse_d128_window_sdpa2_chains"
            + ("_mask" if masked else ""),
            input_names=["queries", "keys", "values", "scale", "n_first"]
            + extra,
            output_names=["partials", "sums", "maxs"],
            header=_KEY_STEP,
            source=_two_pass_source(masked),
            ensure_row_contiguous=False,
        )
    elif name == "one_combine":
        kernel = mx.fast.metal_kernel(
            name="mlx2_muse_d128_window_sdpa1_combine",
            input_names=["maxs", "sums", "outs"],
            output_names=["out"],
            source=_ONE_COMBINE,
        )
    elif name == "two_combine":
        kernel = mx.fast.metal_kernel(
            name="mlx2_muse_d128_window_sdpa2_combine",
            input_names=["partials", "sums", "maxs"],
            output_names=["out"],
            source=_TWO_COMBINE,
        )
    else:  # pragma: no cover - every caller uses a fixed internal name
        raise KeyError(name)
    _KERNELS[name] = kernel
    return kernel


_SCALARS: dict = {}
_INTS: dict = {}


def _scalar(value: float) -> mx.array:
    key = float(value)
    array = _SCALARS.get(key)
    if array is None:
        array = mx.array([key], dtype=mx.float32)
        _SCALARS[key] = array
    return array


def _int(value: int) -> mx.array:
    value = int(value)
    array = _INTS.get(value)
    if array is None:
        array = mx.array([value], dtype=mx.int32)
        if len(_INTS) < 4096:
            _INTS[value] = array
    return array


def window_sdpa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    *,
    n_first: int,
    masks: mx.array | None = None,
) -> mx.array:
    """Attention-only D128 rows over exactly ``n_first + row`` keys."""
    _, heads, rows, _head_dim = queries.shape
    kv_heads = keys.shape[1]
    gqa = heads // kv_heads
    dtype = queries.dtype
    scale_array = _scalar(float(scale))
    plans = plan_signature(n_first, rows, heads, kv_heads)
    suffix = "_mask" if masks is not None else ""
    outputs = []
    for start, stop, plan in plans:
        count = stop - start
        q = queries[:, :, start:stop]
        first = n_first + start
        extra = []
        if masks is not None:
            base = start * n_first + (start * (start - 1)) // 2
            extra = [masks, _int(base)]
        passes, blocks = plan
        if passes == 1:
            chains = 32 * (gqa // _ONE_PASS_HPT)
            maxs, sums, outs = _kernel("one_chains" + suffix)(
                inputs=[q, keys, values, scale_array, _int(first)] + extra,
                template=[
                    ("T", dtype),
                    ("D", HEAD_DIM),
                    ("GQA", gqa),
                    ("HPT", _ONE_PASS_HPT),
                    ("SG", _ONE_PASS_SG),
                ],
                grid=(32 * chains, kv_heads, count),
                threadgroup=(32 * _ONE_PASS_SG, 1, 1),
                output_shapes=[
                    (count * heads * 32,),
                    (count * heads * 32,),
                    (count * heads * HEAD_DIM * 32,),
                ],
                output_dtypes=[mx.float32] * 3,
            )
            out = _kernel("one_combine")(
                inputs=[maxs, sums, outs],
                template=[("T", dtype), ("D", HEAD_DIM)],
                grid=(heads * 1024, count, 1),
                threadgroup=(1024, 1, 1),
                output_shapes=[(count, heads * HEAD_DIM)],
                output_dtypes=[dtype],
            )[0]
        else:
            partials, sums, maxs = _kernel("two_chains" + suffix)(
                inputs=[q, keys, values, scale_array, _int(first)] + extra,
                template=[
                    ("T", dtype),
                    ("D", HEAD_DIM),
                    ("GQA", gqa),
                    ("BLOCKS", blocks),
                ],
                grid=(32 * kv_heads, gqa, blocks * count),
                threadgroup=(32, gqa, 1),
                output_shapes=[
                    (heads * count * blocks * HEAD_DIM,),
                    (heads * count * blocks,),
                    (heads * count * blocks,),
                ],
                output_dtypes=[dtype, mx.float32, mx.float32],
            )
            out = _kernel("two_combine")(
                inputs=[partials, sums, maxs],
                template=[("T", dtype), ("D", HEAD_DIM), ("BLOCKS", blocks)],
                grid=(heads * 1024, count, 1),
                threadgroup=(1024, 1, 1),
                output_shapes=[(count, heads * HEAD_DIM)],
                output_dtypes=[dtype],
            )[0]
        outputs.append(out)
    output = outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=0)
    return output.reshape(1, rows, heads * HEAD_DIM)


@cache
def source_identity() -> str:
    """Bind numerical verdicts to both upstream proof and local Metal text."""
    pieces = [
        ALGORITHM,
        MLX_SOURCE_REVISION,
        MLX_SDPA_DISPATCH_SHA256,
        MLX_SDPA_VECTOR_SHA256,
        repr((HEAD_DIM, MAX_ROWS, TWO_PASS_KEYS, _ONE_PASS_HPT, _ONE_PASS_SG)),
        _KEY_STEP,
        _ONE_PASS,
        _ONE_COMBINE,
        _TWO_PASS,
        _TWO_COMBINE,
        *(
            inspect.getsource(fn)
            for fn in (
                _select_blocks,
                _serial_plan,
                row_plans,
                _plan_runs,
                plan_signature,
                window_supported,
                _masked_source,
                _one_pass_source,
                _two_pass_source,
                _kernel,
                _scalar,
                _int,
                window_sdpa,
            )
        ),
    ]
    return hashlib.sha256("\n".join(pieces).encode()).hexdigest()
