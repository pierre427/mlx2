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

Past the indexer budget (mlx2's own, not in omlx) two more chains of MLX
ops become one launch each: ``qsa_mask`` (``QSASelection.dense_mask()`` for
an explicit selection; integer/bool logic) and ``index_q`` (the indexer
query's RMS norm + position RoPE, products and sums rounded as the composed
ops round them).  The scores matmul, argpartition and pooled keys stay MLX.

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

from .import_env import snapshot as _import_env_snapshot
from ..technologies.window_attention import vector_launch as _VL
from ..technologies.window_attention import vector_sources as _VS

_import_env_snapshot(__name__)

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


# ``select_sdpa_blocks`` for the 's' and 'd' GPU classes: the window-attention
# technology's copy of MLX's block choice (one copy for every transcription).
select_sdpa_blocks = _VL.select_sdpa_blocks


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

# The vector SDPA chains and combines (``sdpa_vector.h``) are generated by the
# window-attention technology: ``_VS.QWEN4_ROWS`` is this module's family
# (cache-view key count, MLX's threadgroup row layout, broadcast bool mask,
# optional gate store); the source text is byte-identical to the copy this
# module carried (tests/test_technologies_window_attention_sources.py).


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
    elif name.startswith(_VS.STAGES):
        stage = "_".join(name.split("_")[:2])
        kernel = mx.fast.metal_kernel(
            **_VS.kernel_kwargs(_VS.QWEN4_ROWS, stage, masked=masked, gated=gated)
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
    plan, blocks = sdpa_plan(keys.shape[2], queries.shape[2], queries.shape[1], keys.shape[1],
                             queries.shape[3])
    mask_suffix = "_mask" if mask is not None else ""
    gate_suffix = "" if gate is None else "_gate"
    return _VL.run_rows(
        lambda passes: _kernel(("one" if passes == 1 else "two") + "_chains" + mask_suffix),
        lambda passes: _kernel(("one" if passes == 1 else "two") + "_combine" + gate_suffix),
        queries,
        keys,
        values,
        _scalar(float(scale), mx.float32),
        plan=plan,
        blocks=blocks,
        hpt=_ONE_PASS_HPT,
        sg=_ONE_PASS_SG,
        mask=mask,
        gate=gate,
    )


# --------------------------------------------------------------------------
# Stage 4 (mlx2's own, past the indexer budget): the QSA mask in one launch
# --------------------------------------------------------------------------

# QSASelection.dense_mask() for an explicit selection, cell for cell: the
# chosen-block bitmap of one (batch, row) is built in threadgroup memory from
# the selected ids (``chosen & valid_blocks``), then every token reads its
# clipped block's bit, ORs the causal tail of its row, ANDs the left-padding
# floor and the causal mask.  Integer/bool logic only (MLX's floor_divide for
# signed ints), so equality is by construction.
_MASK_SOURCE = r"""
    const uint lid = thread_position_in_threadgroup.x;
    const uint lsize = threads_per_threadgroup.x;
    const int l = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;
    const int n_blocks = int(valid_shape[2]);
    const int total = int(tokens_shape[1]);
    const int K = int(selected_shape[2]);
    threadgroup atomic_uint bits[MAX_WORDS];

    for (int w = lid; w < (n_blocks + 31) / 32; w += lsize) {
        atomic_store_explicit(&bits[w], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const size_t vb = (valid_shape[0] > 1 ? b : 0) * valid_strides[0] + l * valid_strides[1];
    const size_t sb = b * selected_strides[0] + l * selected_strides[1];
    for (int k = lid; k < K; k += lsize) {
        const int id = int(selected[sb + k * selected_strides[2]]);
        if (id >= 0 && id < n_blocks && valid[vb + id * valid_strides[2]]) {
            atomic_fetch_or_explicit(&bits[id >> 5], 1u << (id & 31), memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int qp = int(q_pos[(q_pos_shape[0] > 1 ? b : 0) * q_pos_strides[0] + l * q_pos_strides[1]]);
    const int complete = floor_div(qp + 1, BS) * BS;
    const size_t tb = (tokens_shape[0] > 1 ? b : 0) * tokens_strides[0];
    device bool* out = mask + ((size_t)b * threadgroups_per_grid.y + l) * total;
    for (int t = lid; t < total; t += lsize) {
        const int tl = int(tokens[tb + t * tokens_strides[1]]);
        const int blk = clamp(floor_div(tl, BS), 0, n_blocks - 1);
        const bool sel =
            (atomic_load_explicit(&bits[blk >> 5], memory_order_relaxed) >> (blk & 31)) & 1u;
        bool v = sel || (tl >= complete && tl <= qp);
        @VISIBILITY@
        @LEFT_PAD@
        @CAUSAL@
        out[t] = v;
    }
"""

_MASK_HEADER = r"""
inline int floor_div(int x, int y) {
    int q = x / y;
    if (x % y != 0 && (x < 0) != (y < 0)) {
        q -= 1;
    }
    return q;
}
"""

_MASK_MAX_BLOCKS = 131072


def _mask_kernel(left_pad: bool, causal: bool, visibility: bool = False):
    suffix = "_visibility" if visibility else ""
    name = f"mask_{int(left_pad)}{int(causal)}{suffix}"
    kernel = _KERNELS.get(name)
    if kernel is not None:
        return kernel
    source = _MASK_SOURCE.replace(
        "@VISIBILITY@", "v = v && (tl <= qp);" if visibility else ""
    ).replace(
        "@LEFT_PAD@", "v = v && (tl >= 0);" if left_pad else ""
    ).replace(
        "@CAUSAL@",
        (
            "v = v && causal[(causal_shape[0] > 1 ? b : 0) * causal_strides[0]"
            " + (causal_shape[2] > 1 ? l : 0) * causal_strides[2]"
            " + (causal_shape[3] > 1 ? t : 0) * causal_strides[3]];"
        )
        if causal
        else "",
    )
    kernel = mx.fast.metal_kernel(
        name=f"mlx2_qwen4_attn_rows_qsa_mask_{int(left_pad)}{int(causal)}{suffix}",
        input_names=["selected", "valid", "q_pos", "tokens"] + (["causal"] if causal else []),
        output_names=["mask"],
        header=_MASK_HEADER,
        source=source,
        ensure_row_contiguous=False,
    )
    _KERNELS[name] = kernel
    return kernel


def qsa_mask_supported(selection) -> Optional[str]:
    """``None`` when ``qsa_mask`` rebuilds ``selection.dense_mask()``."""
    if not metal_ready():
        return "device"
    if selection.kind != "explicit":
        return "kind"
    if not 0 < selection.n_blocks <= _MASK_MAX_BLOCKS:
        return "blocks"
    arrays = (
        selection.raw_block_ids,
        selection.valid_blocks,
        selection.q_positions,
        selection.token_positions,
    )
    if any(not isinstance(a, mx.array) for a in arrays):
        return "arrays"
    if selection.raw_block_ids.dtype not in (mx.uint32, mx.int32, mx.int64, mx.uint64):
        return "id_dtype"
    if selection.valid_blocks.dtype != mx.bool_:
        return "valid_dtype"
    if selection.q_positions.ndim != 2 or selection.token_positions.ndim != 2:
        return "position_rank"
    for a in (selection.q_positions, selection.token_positions):
        if a.dtype not in (mx.int32, mx.uint32):
            return "position_dtype"
    causal = selection.causal_mask
    from .qwen4_qsa_visibility import QSAVisibility
    if isinstance(causal, QSAVisibility):
        causal = None
    if causal is not None:
        if not isinstance(causal, mx.array) or causal.dtype != mx.bool_ or causal.ndim != 4:
            return "causal"
        if causal.shape[1] != 1 or causal.shape[0] not in (1, selection.batch):
            return "causal_shape"
        if causal.shape[2] not in (1, selection.length) or causal.shape[3] not in (
            1,
            selection.physical_width,
        ):
            return "causal_shape"
    return None


def qsa_mask(selection) -> mx.array:
    """``selection.dense_mask()`` for an explicit selection in one launch:
    ``[B, 1, L, physical_width]`` bool."""
    batch, length, total = selection.batch, selection.length, selection.physical_width
    causal = selection.causal_mask
    from .qwen4_qsa_visibility import QSAVisibility, record
    visibility = isinstance(causal, QSAVisibility)
    if visibility:
        record("fused_mask_calls")
        causal = None
    out_batch = batch if causal is None else max(batch, causal.shape[0])
    selected = selection.raw_block_ids
    if selected.dtype in (mx.int64, mx.uint64):
        selected = selected.astype(mx.int32)
    inputs = [
        selected,
        selection.valid_blocks,
        selection.q_positions,
        selection.token_positions,
    ]
    if causal is not None:
        inputs.append(causal)
    threads = 1024 if total >= 1024 else max(32, ((total + 31) // 32) * 32)
    return _mask_kernel(selection.left_padding is not None, causal is not None, visibility)(
        inputs=inputs,
        template=[("BS", int(selection.block_size)), ("MAX_WORDS", _MASK_MAX_BLOCKS // 32)],
        grid=(threads, length, out_batch),
        threadgroup=(threads, 1, 1),
        output_shapes=[(out_batch, 1, length, total)],
        output_dtypes=[mx.bool_],
    )[0]


# --------------------------------------------------------------------------
# Stage 5 (mlx2's own): the indexer query's RMS norm + position RoPE
# --------------------------------------------------------------------------

# ``_apply_rope_positions(q_layernorm(q), q_pos[..., None], ...)`` for the
# unscaled table: rms_single_row (axis 128: one simdgroup), then the composed
# MLX ops -- every product and sum is its own f32 op there, so contraction is
# off here (an fma would round once instead of twice) and cos/sin are the
# precise ones MLX's unary kernels call.
_INDEX_Q_SOURCE = r"""
#pragma METAL fp contract(off)
    constexpr int SIMD_SIZE = 32;
    constexpr int N_READS = 4;
    const uint axis_size = HD;
    const uint lid = thread_position_in_threadgroup.x;
    const uint simd_lane_id = thread_index_in_simdgroup;
    const uint simd_group_id = simdgroup_index_in_threadgroup;
    const int h = threadgroup_position_in_grid.x;
    const int t = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;
    const int rows = threadgroups_per_grid.y;

    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[SIMD_SIZE];
    threadgroup T normed[HD];

    const device T* x = qk + b * qk_strides[0] + t * qk_strides[1] + h * HD + lid * N_READS;
    const device T* w = weight + lid * N_READS;
    float acc = 0;
    float thread_x[N_READS];
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

    device T* out = q_out + (((size_t)b * rows + t) * NH + h) * HD;
    constexpr int HALF = ROT / 2;
    if (int(lid) < HALF) {
        const int d = lid;
        const int pos = q_pos[(q_pos_shape[0] > 1 ? b : 0) * q_pos_strides[0] + t * q_pos_strides[1]];
        const float angle = static_cast<float>(pos) * freqs[d];
        const float c = metal::precise::cos(angle);
        const float s = metal::precise::sin(angle);
        const float left = static_cast<float>(normed[d]);
        const float right = static_cast<float>(normed[d + HALF]);
        const float a1 = left * c;
        const float b1 = right * s;
        const float a2 = right * c;
        const float b2 = left * s;
        out[d] = static_cast<T>(a1 - b1);
        out[d + HALF] = static_cast<T>(a2 + b2);
    }
    for (int j = ROT + int(lid); j < HD; j += 32) {
        out[j] = normed[j];
    }
"""


def index_q_supported(q_view, weight, q_pos, rotary_dim, n_heads, head_dim, scaling) -> Optional[str]:
    if not metal_ready():
        return "device"
    if scaling is not None:
        return "scaled_rope"
    if q_view.dtype not in (mx.bfloat16, mx.float16) or weight.dtype != q_view.dtype:
        return "dtype"
    if head_dim != 128 or weight.shape != (head_dim,):
        return "head_dim"
    if q_view.ndim != 3 or q_view.shape[-1] < n_heads * head_dim:
        return "shape"
    if not 0 < rotary_dim <= 64 or rotary_dim % 2:
        return "rotary_dim"
    if q_pos.ndim != 2 or q_pos.dtype not in (mx.int32, mx.uint32):
        return "positions"
    return None


def index_q(q_view, weight, eps, q_pos, freqs, *, n_heads, head_dim, rotary_dim) -> mx.array:
    """``[B, L, n_heads, head_dim]``: the indexer query normed and rotated as
    ``_apply_rope_positions(q_layernorm(q), q_pos[..., None], ...)`` computes
    it; ``q_view`` is ``[B, L, >= n_heads * head_dim]`` (the projected
    index_qk, query heads first)."""
    batch, rows, _ = q_view.shape
    kernel = _KERNELS.get("index_q")
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name="mlx2_qwen4_attn_rows_index_q",
            input_names=["qk", "weight", "eps", "q_pos", "freqs"],
            output_names=["q_out"],
            source=_INDEX_Q_SOURCE,
            ensure_row_contiguous=False,
        )
        _KERNELS["index_q"] = kernel
    return kernel(
        inputs=[q_view, weight, _scalar(float(eps), mx.float32), q_pos, freqs],
        template=[("T", q_view.dtype), ("NH", n_heads), ("HD", head_dim), ("ROT", rotary_dim)],
        grid=(32 * n_heads, rows, batch),
        threadgroup=(32, 1, 1),
        output_shapes=[(batch, rows, n_heads, head_dim)],
        output_dtypes=[q_view.dtype],
    )[0]
