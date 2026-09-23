# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
from __future__ import annotations
import logging
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from typing import Any, Optional
import mlx.core as mx

logger = logging.getLogger(__name__)
NUM_KEY_HEADS = 16
NUM_VALUE_HEADS = 48
KEY_HEAD_DIM = 128
VALUE_HEAD_DIM = 128
CONV_KERNEL = 4
KEY_DIM = NUM_KEY_HEADS * KEY_HEAD_DIM
VALUE_DIM = NUM_VALUE_HEADS * VALUE_HEAD_DIM
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
_THREADGROUP_Y_CANDIDATES = (32, 16, 8, 4)


@dataclass(frozen=True)
class FusedGdnAdmission:
    accepted: bool
    reason: str


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(getattr(value, "shape", ()))


def _dtype(value: Any) -> Any:
    return getattr(value, "dtype", None)


def _slab_width(value: Any) -> int:
    """Sequence extent of a ``(B, S, D)`` activation, or 0 when it has none."""
    shape = _shape(value)
    return int(shape[1]) if len(shape) > 1 else 0


def admit_rollback_span(
    spans: Any, mask: Any, width: int, *, masked_reason: str
) -> Optional[FusedGdnAdmission]:
    """Shared mask-free slab predicate for both fused GDN kernels.

    ``spans`` is the cache's host-side rollback geometry for this forward
    (``ArraysCache.rollback_spans``).  Neither kernel reads a mask, so only a
    slab whose single row advances every position is exact: ``()`` (no length
    metadata and no mask) or a one-row span equal to the width, which is how a
    ragged engine describes a fully valid lane -- its mask, derived from that
    same metadata, is then all ones.  A shorter span is right padding, more
    than one row is a batch this kernel does not serve, and ``None`` is
    geometry the cache cannot describe row-wise.

    Returns the refusal, or ``None`` when the slab is admissible.
    """
    if spans is None:
        return FusedGdnAdmission(False, "rollback geometry not describable")
    if spans == ():
        if mask is not None:
            return FusedGdnAdmission(False, masked_reason)
        return None
    if len(spans) != 1:
        # Name the batch: an unpadded B>1 slab is not padding, and calling it
        # that sends a reader of the receipts after the wrong cause.
        return FusedGdnAdmission(False, f"batch of {len(spans)} rows")
    if int(spans[0]) != int(width):
        return FusedGdnAdmission(False, "padded rollback geometry")
    return None


def admit_qwen4_fused_gdn_decode(
    *,
    qkv: Any,
    z: Any,
    b: Any,
    a: Any,
    conv_state: Any,
    recurrent_state: Any,
    conv_weight: Any,
    A_log: Any,
    dt_bias: Any,
    norm_weight: Any,
    mask: Any,
    spans: Any,
    speculating: bool,
    training: bool,
    sharded: bool,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    conv_kernel: int,
    gate_activation: str,
    architecture: str = "qwen4",
) -> FusedGdnAdmission:
    """Pure structural admission check; safe to exercise without MLX eval.

    Agnes shares Qwen4's production recurrent geometry, but not its numerical
    contract: it uses a swish output gate and the shared Qwen3.5 normalization
    boundaries.  ``architecture`` keeps those two contracts explicit without
    weakening Qwen4's sigmoid-only admission.
    """
    if training:
        return FusedGdnAdmission(False, "training")
    if sharded:
        return FusedGdnAdmission(False, "distributed sharding")
    if speculating:
        return FusedGdnAdmission(False, "speculative rollback")
    refusal = admit_rollback_span(
        spans, mask, _slab_width(qkv), masked_reason="masked decode"
    )
    if refusal is not None:
        return refusal
    expected_gate = {
        "qwen4": "sigmoid",
        "agnes": "swish",
        "qwen35": "swish",
    }.get(architecture)
    if expected_gate is None:
        return FusedGdnAdmission(False, f"unsupported architecture {architecture!r}")
    if gate_activation != expected_gate:
        return FusedGdnAdmission(False, f"output gate {gate_activation!r}")
    geometry = (
        num_key_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
        conv_kernel,
    )
    expected_geometry = (
        (16, 32, 128, 128, 4)
        if architecture == "qwen35"
        else (NUM_KEY_HEADS, NUM_VALUE_HEADS, KEY_HEAD_DIM, VALUE_HEAD_DIM, CONV_KERNEL)
    )
    if geometry != expected_geometry:
        return FusedGdnAdmission(False, f"unsupported geometry {geometry}")
    key_dim = num_key_heads * key_head_dim
    value_dim = num_value_heads * value_head_dim
    conv_dim = 2 * key_dim + value_dim
    expected = {
        "qkv": (1, 1, conv_dim),
        "z": (1, 1, value_dim),
        "a": (1, 1, num_value_heads),
        "b": (1, 1, num_value_heads),
        "conv_state": (1, conv_kernel - 1, conv_dim),
        "recurrent_state": (1, num_value_heads, value_head_dim, key_head_dim),
        "conv_weight": (conv_dim, conv_kernel, 1),
        "A_log": (num_value_heads,),
        "dt_bias": (num_value_heads,),
        "norm_weight": (value_head_dim,),
    }
    values = {
        "qkv": qkv,
        "z": z,
        "a": a,
        "b": b,
        "conv_state": conv_state,
        "recurrent_state": recurrent_state,
        "conv_weight": conv_weight,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "norm_weight": norm_weight,
    }
    for name, expected_shape in expected.items():
        if _shape(values[name]) != expected_shape:
            return FusedGdnAdmission(
                False, f"{name} shape {_shape(values[name])}, expected {expected_shape}"
            )
    value_dtype = _dtype(qkv)
    if value_dtype != mx.bfloat16:
        return FusedGdnAdmission(False, f"unsupported activation dtype {value_dtype}")
    for name in ("z", "a", "b", "conv_state", "conv_weight", "dt_bias", "norm_weight"):
        if _dtype(values[name]) != value_dtype:
            return FusedGdnAdmission(False, f"{name} dtype {_dtype(values[name])}")
    if _dtype(recurrent_state) != mx.float32:
        return FusedGdnAdmission(False, "recurrent_state must be float32")
    if _dtype(A_log) not in (value_dtype, mx.float32):
        return FusedGdnAdmission(False, f"A_log dtype {_dtype(A_log)}")
    return FusedGdnAdmission(True, "eligible")


_HEADER = "\n#include <metal_atomic>\ntemplate <typename U>\ninline U mlx_sigmoid_precise(U x) {\n  U e = static_cast<U>(metal::precise::exp(metal::abs(x)));\n  U y = static_cast<U>(1) / (static_cast<U>(1) + e);\n  return (x < 0) ? y : (static_cast<U>(1) - y);\n}\n\ntemplate <typename U>\ninline U mlx_sigmoid_fast(U x) {\n  U e = static_cast<U>(metal::exp(metal::abs(x)));\n  U y = static_cast<U>(1) / (static_cast<U>(1) + e);\n  return (x < 0) ? y : (static_cast<U>(1) - y);\n}\n\ntemplate <typename U>\ninline U mlx_log1p_fast(U x) {\n  float xf = float(x);\n  float xp1 = 1.0f + xf;\n  float out = xp1 == 1.0f ? xf : xf * (metal::log(xp1) / (xp1 - 1.0f));\n  return static_cast<U>(out);\n}\n\ntemplate <typename U>\ninline U mlx_softplus_fast(U x) {\n  if (metal::isnan(x))\n    return metal::numeric_limits<U>::quiet_NaN();\n  constexpr U inf = metal::numeric_limits<U>::infinity();\n  U zero = static_cast<U>(0);\n  U hi = metal::max(x, zero);\n  U lo = metal::min(x, zero);\n  return (lo == -inf || hi == inf)\n      ? hi\n      : (hi + mlx_log1p_fast(static_cast<U>(metal::exp(lo - hi))));\n}\n\n"
_SOURCE = "\n  const uint hv = threadgroup_position_in_grid.z;\n  const uint hk = hv / RATIO;\n  const uint lane = thread_position_in_threadgroup.x;\n  const uint ty = thread_position_in_threadgroup.y;\n  const uint tid = thread_index_in_threadgroup;\n\n  constexpr int NT = 32 * TY;\n  constexpr int NDK = DK / 32;\n  constexpr int NDV = DV / TY;\n  constexpr uint KD = (uint)(HK * DK);\n  constexpr uint VD = (uint)(HV * DV);\n  constexpr uint CD = 2u * KD + VD;\n\n  threadgroup float sq[DK];\n  threadgroup float sk[DK];\n  // Float storage can represent the Qwen4 bf16 squares exactly after their\n  // explicit cast while also serving Agnes' float-accumulating fast RMSNorm.\n  threadgroup float sq_squared[DK];\n  threadgroup float sk_squared[DK];\n  threadgroup float sv[DV];\n  threadgroup float sy[DV];\n  threadgroup float shr[4];\n\n  device const float* si = recurrent_state + (size_t)hv * DV * DK;\n  device float* so = recurrent_state_out + (size_t)hv * DV * DK;\n  float st[NDV][NDK];\n  for (int j = 0; j < NDV; ++j) {\n    uint dv = ty + (uint)TY * (uint)j;\n    for (int i = 0; i < NDK; ++i)\n      st[j][i] = si[(size_t)dv * DK + NDK * lane + i];\n  }\n\n  // q/k channels are shared by three value heads.  All three compute the\n  // identical values locally, but only the first writes their cache channels.\n  for (uint idx = tid; idx < (uint)(2 * DK + DV); idx += NT) {\n    uint part = idx / (uint)DK;\n    uint d = idx - part * (uint)DK;\n    uint c = part == 0u ? hk * DK + d\n           : (part == 1u ? KD + hk * DK + d : 2u * KD + hv * DV + d);\n    device const T* wc = conv_weight + (size_t)c * K;\n    float acc = 0.0f;\n    for (uint tap = 0; tap + 1 < (uint)K; ++tap)\n      acc += float(conv_state[(size_t)tap * CD + c]) * float(wc[tap]);\n    acc += float(qkv[c]) * float(wc[K - 1]);\n    T xb = static_cast<T>(acc);\n    T sl = xb * mlx_sigmoid_fast(xb);\n    if (part == 0u) sq[d] = float(sl);\n    else if (part == 1u) sk[d] = float(sl);\n    else sv[d] = float(sl);\n\n    if (part == 2u || (hv % RATIO) == 0u) {\n      for (uint tap = 0; tap + 2 < (uint)K; ++tap)\n        conv_state_out[(size_t)tap * CD + c] =\n            conv_state[(size_t)(tap + 1) * CD + c];\n      conv_state_out[(size_t)(K - 2) * CD + c] = qkv[c];\n    }\n  }\n\n  if (tid == 0u) {\n    T av = a[hv] + dt_bias[hv];\n    T sp = mlx_softplus_fast(av);\n    shr[2] = metal::precise::exp(\n        -metal::precise::exp(float(A_log[hv])) * float(sp));\n    if constexpr (AGNES_NUMERICS) {\n      // Shared Qwen3.5 widens b before the eager sigmoid.\n      shr[3] = mlx_sigmoid_precise<float>(float(b[hv]));\n    } else {\n      // Qwen4 materializes beta in the activation dtype.\n      shr[3] = float(mlx_sigmoid_precise(b[hv]));\n    }\n  }\n  threadgroup_barrier(mem_flags::mem_threadgroup);\n\n  for (uint d = tid; d < (uint)DK; d += NT) {\n    if constexpr (AGNES_NUMERICS) {\n      float qv = sq[d];\n      float kv = sk[d];\n      sq_squared[d] = qv * qv;\n      sk_squared[d] = kv * kv;\n    } else {\n      T qv = static_cast<T>(sq[d]);\n      T kv = static_cast<T>(sk[d]);\n      sq_squared[d] = float(static_cast<T>(qv * qv));\n      sk_squared[d] = float(static_cast<T>(kv * kv));\n    }\n  }\n  threadgroup_barrier(mem_flags::mem_threadgroup);\n\n  if (simdgroup_index_in_threadgroup == 0u) {\n    uint base = 4u * lane;\n    if constexpr (AGNES_NUMERICS) {\n      // This mirrors MLX rms_single_row for axis_size=128/N_READS=4:\n      // four float products per lane, then one simd_sum in float32.\n      float pq = 0.0f, pk = 0.0f;\n      for (int i = 0; i < 4; ++i) {\n        pq += sq_squared[base + i];\n        pk += sk_squared[base + i];\n      }\n      pq = simd_sum(pq);\n      pk = simd_sum(pk);\n      if (lane == 0u) {\n        constexpr float rms_eps = 1.0e-6f / float(DK);\n        shr[0] = metal::precise::rsqrt(pq / float(DK) + rms_eps);\n        shr[1] = metal::precise::rsqrt(pk / float(DK) + rms_eps);\n      }\n    } else {\n      T pq = static_cast<T>(0), pk = static_cast<T>(0);\n      for (int i = 0; i < 4; ++i) {\n        pq = static_cast<T>(static_cast<T>(sq_squared[base + i]) + pq);\n        pk = static_cast<T>(static_cast<T>(sk_squared[base + i]) + pk);\n      }\n      pq = static_cast<T>(simd_sum(float(pq)));\n      pk = static_cast<T>(simd_sum(float(pk)));\n      if (lane == 0u) {\n        T eps = static_cast<T>(1.0e-6f);\n        T qdenom = pq + eps;\n        T kdenom = pk + eps;\n        shr[0] = float(static_cast<T>(metal::precise::rsqrt(qdenom)));\n        shr[1] = float(static_cast<T>(metal::precise::rsqrt(kdenom)));\n      }\n    }\n  }\n  threadgroup_barrier(mem_flags::mem_threadgroup);\n  for (uint d = tid; d < (uint)DK; d += NT) {\n    if constexpr (AGNES_NUMERICS) {\n      // normalize_gdn_qk expresses L2 normalization through fast RMSNorm,\n      // then applies 1/DK to q and 1/sqrt(DK) to k in activation dtype.\n      T q_rms = static_cast<T>(sq[d] * shr[0]);\n      T k_rms = static_cast<T>(sk[d] * shr[1]);\n      sq[d] = float(static_cast<T>(q_rms * static_cast<T>(1.0f / float(DK))));\n      sk[d] = float(static_cast<T>(\n          k_rms * static_cast<T>(0.08838834764831845f)));\n    } else {\n      T qscale = static_cast<T>(0.08838834764831845f);\n      T q_normalized = static_cast<T>(\n          static_cast<T>(sq[d]) * static_cast<T>(shr[0]));\n      T k_normalized = static_cast<T>(\n          static_cast<T>(sk[d]) * static_cast<T>(shr[1]));\n      sq[d] = float(static_cast<T>(q_normalized * qscale));\n      sk[d] = float(k_normalized);\n    }\n  }\n  threadgroup_barrier(mem_flags::mem_threadgroup);\n\n  for (int j = 0; j < NDV; ++j) {\n    uint dv = ty + (uint)TY * (uint)j;\n    float kv = 0.0f;\n    for (int i = 0; i < NDK; ++i) {\n      uint s = NDK * lane + i;\n      st[j][i] = st[j][i] * shr[2];\n      kv += st[j][i] * sk[s];\n    }\n    kv = simd_sum(kv);\n    float delta = (sv[dv] - kv) * shr[3];\n    float out = 0.0f;\n    for (int i = 0; i < NDK; ++i) {\n      uint s = NDK * lane + i;\n      st[j][i] = st[j][i] + sk[s] * delta;\n      out += st[j][i] * sq[s];\n    }\n    out = simd_sum(out);\n    if (thread_index_in_simdgroup == 0u)\n      sy[dv] = float(static_cast<T>(out));\n    for (int i = 0; i < NDK; ++i)\n      so[(size_t)dv * DK + NDK * lane + i] = st[j][i];\n  }\n  threadgroup_barrier(mem_flags::mem_threadgroup);\n\n  if (simdgroup_index_in_threadgroup == 0u) {\n    float po = 0.0f;\n    uint base = 4u * lane;\n    for (int i = 0; i < 4; ++i) po += sy[base + i] * sy[base + i];\n    po = simd_sum(po);\n    if (lane == 0u)\n      shr[0] = metal::precise::rsqrt(po / (float)DV + norm_eps);\n  }\n  threadgroup_barrier(mem_flags::mem_threadgroup);\n  for (uint d = tid; d < (uint)DV; d += NT) {\n    T normalized = static_cast<T>(sy[d] * shr[0]);\n    normalized = norm_weight[d] * normalized;\n    float zv = float(z[hv * DV + d]);\n    float gate = mlx_sigmoid_precise<float>(zv);\n    if constexpr (AGNES_NUMERICS) {\n      // Agnes' compiled _precise_swiglu uses nn.silu in float32. Runtime-built\n      // compiled spans lower that sigmoid to metal::exp, hence the fast form.\n      gate = zv * mlx_sigmoid_fast<float>(zv);\n    }\n    float x = float(normalized) * gate;\n    output[hv * DV + d] = static_cast<T>(x);\n  }\n"
_SOURCE_OUTPROJ = "\n  const uint lane = thread_position_in_threadgroup.x;\n  const uint ty = thread_position_in_threadgroup.y;\n  const uint tid = thread_index_in_threadgroup;\n  const uint sg = simdgroup_index_in_threadgroup;\n\n  constexpr int NT = 32 * TY;\n  constexpr int NDK = DK / 32;\n  constexpr int NDV = DV / TY;\n  constexpr uint KD = (uint)(HK * DK);\n  constexpr uint VD = (uint)(HV * DV);\n  constexpr uint CD = 2u * KD + VD;\n\n  threadgroup float sq[DK];\n  threadgroup float sk[DK];\n  threadgroup T sq_squared[DK];\n  threadgroup T sk_squared[DK];\n  threadgroup float sv[DV];\n  threadgroup float sy[DV];\n  threadgroup float shr[4];\n  constexpr uint HEADS_PER_BLOCK = 4u;\n  constexpr uint QBLOCK = HEADS_PER_BLOCK * DV;\n  constexpr uint NBLOCKS = HV / HEADS_PER_BLOCK;\n  const uint block = threadgroup_position_in_grid.z;\n  const uint epoch_value = epoch[0];\n  device atomic_uint* barrier =\n      reinterpret_cast<device atomic_uint*>(const_cast<device uint*>(control));\n\n  // The control array is resident per layer.  Epochs are supplied by the\n  // Python caller, so no stale value can release a later invocation.\n  if (block == 0u) {\n    if (tid == 0u) {\n      atomic_store_explicit(barrier + 1, 0u, memory_order_relaxed);\n      atomic_store_explicit(barrier, epoch_value, memory_order_relaxed);\n    }\n  } else if (tid == 0u) {\n    while (atomic_load_explicit(barrier, memory_order_relaxed)\n           != epoch_value) {}\n  }\n  threadgroup_barrier(mem_flags::mem_threadgroup);\n\n  threadgroup T all_y[QBLOCK];\n\n  for (uint local_hv = 0; local_hv < HEADS_PER_BLOCK; ++local_hv) {\n    const uint hv = block * HEADS_PER_BLOCK + local_hv;\n    const uint hk = hv / RATIO;\n    device const float* si = recurrent_state + (size_t)hv * DV * DK;\n    device float* so = recurrent_state_out + (size_t)hv * DV * DK;\n    float st[NDV][NDK];\n    for (int j = 0; j < NDV; ++j) {\n      uint dv = ty + (uint)TY * (uint)j;\n      for (int i = 0; i < NDK; ++i)\n        st[j][i] = si[(size_t)dv * DK + NDK * lane + i];\n    }\n\n    for (uint idx = tid; idx < (uint)(2 * DK + DV); idx += NT) {\n      uint part = idx / (uint)DK;\n      uint d = idx - part * (uint)DK;\n      uint c = part == 0u ? hk * DK + d\n             : (part == 1u ? KD + hk * DK + d : 2u * KD + hv * DV + d);\n      device const T* wc = conv_weight + (size_t)c * K;\n      float acc = 0.0f;\n      for (uint tap = 0; tap + 1 < (uint)K; ++tap)\n        acc += float(conv_state[(size_t)tap * CD + c]) * float(wc[tap]);\n      acc += float(qkv[c]) * float(wc[K - 1]);\n      T xb = static_cast<T>(acc);\n      T sl = xb * mlx_sigmoid_fast(xb);\n      if (part == 0u) sq[d] = float(sl);\n      else if (part == 1u) sk[d] = float(sl);\n      else sv[d] = float(sl);\n\n      if (part == 2u || (hv % RATIO) == 0u) {\n        for (uint tap = 0; tap + 2 < (uint)K; ++tap)\n          conv_state_out[(size_t)tap * CD + c] =\n              conv_state[(size_t)(tap + 1) * CD + c];\n        conv_state_out[(size_t)(K - 2) * CD + c] = qkv[c];\n      }\n    }\n\n    if (tid == 0u) {\n      T av = a[hv] + dt_bias[hv];\n      T sp = mlx_softplus_fast(av);\n      shr[2] = metal::precise::exp(\n          -metal::precise::exp(float(A_log[hv])) * float(sp));\n      // Exhaustive bf16 sweep: mlx_sigmoid_precise<T> equals mx.sigmoid on\n      // every finite bf16 input; the fast form differs on one (x ~ -6.85).\n      shr[3] = float(mlx_sigmoid_precise(b[hv]));\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n    for (uint d = tid; d < (uint)DK; d += NT) {\n      T qv = static_cast<T>(sq[d]);\n      T kv = static_cast<T>(sk[d]);\n      sq_squared[d] = static_cast<T>(qv * qv);\n      sk_squared[d] = static_cast<T>(kv * kv);\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n    if (sg == 0u) {\n      T pq = static_cast<T>(0), pk = static_cast<T>(0);\n      uint base = 4u * lane;\n      for (int i = 0; i < 4; ++i) {\n        pq = static_cast<T>(sq_squared[base + i] + pq);\n        pk = static_cast<T>(sk_squared[base + i] + pk);\n      }\n      pq = static_cast<T>(simd_sum(float(pq)));\n      pk = static_cast<T>(simd_sum(float(pk)));\n      if (lane == 0u) {\n        T eps = static_cast<T>(1.0e-6f);\n        T qdenom = pq + eps;\n        T kdenom = pk + eps;\n        shr[0] = float(static_cast<T>(metal::precise::rsqrt(qdenom)));\n        shr[1] = float(static_cast<T>(metal::precise::rsqrt(kdenom)));\n      }\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    T qscale = static_cast<T>(0.08838834764831845f);\n    for (uint d = tid; d < (uint)DK; d += NT) {\n      T q_normalized = static_cast<T>(static_cast<T>(sq[d]) * static_cast<T>(shr[0]));\n      T k_normalized = static_cast<T>(static_cast<T>(sk[d]) * static_cast<T>(shr[1]));\n      sq[d] = float(static_cast<T>(q_normalized * qscale));\n      sk[d] = float(k_normalized);\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n    for (int j = 0; j < NDV; ++j) {\n      uint dv = ty + (uint)TY * (uint)j;\n      float kv = 0.0f;\n      for (int i = 0; i < NDK; ++i) {\n        uint s = NDK * lane + i;\n        st[j][i] = st[j][i] * shr[2];\n        kv += st[j][i] * sk[s];\n      }\n      kv = simd_sum(kv);\n      float delta = (sv[dv] - kv) * shr[3];\n      float out = 0.0f;\n      for (int i = 0; i < NDK; ++i) {\n        uint s = NDK * lane + i;\n        st[j][i] = st[j][i] + sk[s] * delta;\n        out += st[j][i] * sq[s];\n      }\n      out = simd_sum(out);\n      if (thread_index_in_simdgroup == 0u)\n        sy[dv] = float(static_cast<T>(out));\n      for (int i = 0; i < NDK; ++i)\n        so[(size_t)dv * DK + NDK * lane + i] = st[j][i];\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n    if (sg == 0u) {\n      float po = 0.0f;\n      uint base = 4u * lane;\n      for (int i = 0; i < 4; ++i) po += sy[base + i] * sy[base + i];\n      po = simd_sum(po);\n      if (lane == 0u)\n        shr[0] = metal::precise::rsqrt(po / (float)DV + norm_eps);\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    for (uint d = tid; d < (uint)DV; d += NT) {\n      T normalized = static_cast<T>(sy[d] * shr[0]);\n      normalized = norm_weight[d] * normalized;\n      // Exhaustive bf16 sweep: the precise float32 sigmoid matches mx.sigmoid on\n      // every finite bf16-valued gate; the fast form differs on ~1% of them.\n      float x = float(normalized) * mlx_sigmoid_precise<float>(float(z[hv * DV + d]));\n      all_y[local_hv * DV + d] = static_cast<T>(x);\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n  }\n\n  // One exact affine_qmv_fast K-block.  Retaining every lane's value lets\n  // block 0 reproduce MLX's 12-block accumulation and final simd_sum order.\n  constexpr uint VALUES = 16u;\n  constexpr uint ROWS_PER_SIMD = 4u;\n  constexpr uint ROWS_PER_WAVE = 32u * ROWS_PER_SIMD;\n  for (uint row0 = sg * ROWS_PER_SIMD; row0 < (uint)OUT;\n       row0 += ROWS_PER_WAVE) {\n    float result[ROWS_PER_SIMD] = {0.0f};\n    float xv[VALUES];\n    float xsum = 0.0f;\n    uint xbase = lane * VALUES;\n    for (uint i = 0; i < VALUES; i += 4) {\n      T tx0 = all_y[xbase + i];\n      T tx1 = all_y[xbase + i + 1];\n      T tx2 = all_y[xbase + i + 2];\n      T tx3 = all_y[xbase + i + 3];\n      float x0 = float(tx0), x1 = float(tx1);\n      float x2 = float(tx2), x3 = float(tx3);\n      xsum += float(tx0 + tx1 + tx2 + tx3);\n      xv[i] = x0;\n      xv[i + 1] = x1 / 16.0f;\n      xv[i + 2] = x2 / 256.0f;\n      xv[i + 3] = x3 / 4096.0f;\n    }\n    for (uint r = 0; r < ROWS_PER_SIMD; ++r) {\n      uint row = row0 + r;\n      if (row >= (uint)OUT) continue;\n      const device uchar* wb =\n          reinterpret_cast<const device uchar*>(out_weight)\n          + (size_t)row * (VD / 2) + block * (QBLOCK / 2) + lane * 8;\n      const device ushort* wp = reinterpret_cast<const device ushort*>(wb);\n      float accum = 0.0f;\n      for (uint i = 0; i < 4; ++i) {\n        ushort packed = wp[i];\n        accum += xv[4 * i] * float(packed & 0x000f)\n                 + xv[4 * i + 1] * float(packed & 0x00f0)\n                 + xv[4 * i + 2] * float(packed & 0x0f00)\n                 + xv[4 * i + 3] * float(packed & 0xf000);\n      }\n      uint group = block * (QBLOCK / OGS) + lane / 4;\n      float os = float(out_scales[(size_t)row * (VD / OGS) + group]);\n      float ob = float(out_biases[(size_t)row * (VD / OGS) + group]);\n      result[r] = os * accum + xsum * ob;\n      partials[((size_t)block * OUT + row) * 32 + lane] = result[r];\n    }\n  }\n\n  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);\n  if (tid == 0u)\n    atomic_fetch_add_explicit(\n        barrier + 1, 1u, memory_order_relaxed);\n  threadgroup_barrier(mem_flags::mem_threadgroup);\n  if (block != 0u) return;\n  if (tid == 0u) {\n    while (atomic_load_explicit(barrier + 1, memory_order_relaxed)\n           != NBLOCKS) {}\n  }\n  threadgroup_barrier(mem_flags::mem_threadgroup);\n\n  for (uint row0 = sg * ROWS_PER_SIMD; row0 < (uint)OUT;\n       row0 += ROWS_PER_WAVE) {\n    for (uint r = 0; r < ROWS_PER_SIMD; ++r) {\n      uint row = row0 + r;\n      if (row >= (uint)OUT) continue;\n      float value = 0.0f;\n      for (uint qb = 0; qb < NBLOCKS; ++qb)\n        value += partials[((size_t)qb * OUT + row) * 32 + lane];\n      value = simd_sum(value);\n      if (lane == 0u) output[row] = static_cast<T>(value);\n    }\n  }\n"


@lru_cache(maxsize=None)
def _kernel():
    return mx.fast.metal_kernel(
        name="qwen4_fused_gdn_decode",
        input_names=[
            "qkv",
            "z",
            "b",
            "a",
            "conv_state",
            "conv_weight",
            "A_log",
            "dt_bias",
            "recurrent_state",
            "norm_weight",
            "norm_eps",
        ],
        output_names=["output", "conv_state_out", "recurrent_state_out"],
        header=_HEADER,
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=None)
def _kernel_outproj():
    return mx.fast.metal_kernel(
        name="qwen4_fused_gdn_decode_outproj_q4",
        input_names=[
            "qkv",
            "z",
            "b",
            "a",
            "conv_state",
            "conv_weight",
            "A_log",
            "dt_bias",
            "recurrent_state",
            "norm_weight",
            "out_weight",
            "out_scales",
            "out_biases",
            "control",
            "epoch",
            "norm_eps",
        ],
        output_names=["output", "conv_state_out", "recurrent_state_out", "partials"],
        header=_HEADER,
        source=_SOURCE_OUTPROJ,
        ensure_row_contiguous=True,
    )


def qwen4_fused_gdn_decode(
    qkv,
    z,
    b,
    a,
    conv_state,
    conv_weight,
    A_log,
    dt_bias,
    recurrent_state,
    norm_weight,
    norm_eps: float,
    *,
    threadgroup_y: int,
    architecture: str = "qwen4",
    num_key_heads: int = NUM_KEY_HEADS,
    num_value_heads: int = NUM_VALUE_HEADS,
    key_head_dim: int = KEY_HEAD_DIM,
    value_head_dim: int = VALUE_HEAD_DIM,
    conv_kernel: int = CONV_KERNEL,
):
    """Build the fused graph.  Callers must run structural admission first."""
    if threadgroup_y not in _THREADGROUP_Y_CANDIDATES:
        raise ValueError(
            f"unsupported threadgroup_y {threadgroup_y}; expected one of {_THREADGROUP_Y_CANDIDATES}"
        )
    if architecture not in ("qwen4", "agnes", "qwen35"):
        raise ValueError(f"unsupported fused GDN architecture {architecture!r}")
    key_dim = num_key_heads * key_head_dim
    value_dim = num_value_heads * value_head_dim
    conv_dim = 2 * key_dim + value_dim
    outputs = _kernel()(
        inputs=[
            qkv,
            z,
            b,
            a,
            conv_state,
            conv_weight,
            A_log,
            dt_bias,
            recurrent_state,
            norm_weight,
            float(norm_eps),
        ],
        template=[
            ("T", qkv.dtype),
            ("HK", num_key_heads),
            ("HV", num_value_heads),
            ("DK", key_head_dim),
            ("DV", value_head_dim),
            ("K", conv_kernel),
            ("TY", threadgroup_y),
            ("RATIO", num_value_heads // num_key_heads),
            ("AGNES_NUMERICS", int(architecture in ("agnes", "qwen35"))),
        ],
        grid=(32, threadgroup_y, num_value_heads),
        threadgroup=(32, threadgroup_y, 1),
        output_shapes=[
            (1, 1, value_dim),
            (1, conv_kernel - 1, conv_dim),
            (1, num_value_heads, value_head_dim, key_head_dim),
        ],
        output_dtypes=[qkv.dtype, qkv.dtype, mx.float32],
    )
    return tuple(outputs)


def qwen4_fused_gdn_decode_outproj(
    qkv,
    z,
    b,
    a,
    conv_state,
    conv_weight,
    A_log,
    dt_bias,
    recurrent_state,
    norm_weight,
    norm_eps: float,
    out_weight,
    out_scales,
    out_biases,
    control,
    epoch: int,
    *,
    output_dim: int,
    output_group_size: int,
):
    """One-dispatch GDN recurrence, gated norm, and affine-q4 QMV."""
    if output_dim != 2560 or output_group_size != 64:
        raise ValueError("only the production 2560x6144 affine-q4 epilogue")
    outputs = _kernel_outproj()(
        inputs=[
            qkv,
            z,
            b,
            a,
            conv_state,
            conv_weight,
            A_log,
            dt_bias,
            recurrent_state,
            norm_weight,
            out_weight,
            out_scales,
            out_biases,
            control,
            mx.array([epoch], dtype=mx.uint32),
            float(norm_eps),
        ],
        template=[
            ("T", qkv.dtype),
            ("HK", NUM_KEY_HEADS),
            ("HV", NUM_VALUE_HEADS),
            ("DK", KEY_HEAD_DIM),
            ("DV", VALUE_HEAD_DIM),
            ("K", CONV_KERNEL),
            ("TY", 32),
            ("RATIO", NUM_VALUE_HEADS // NUM_KEY_HEADS),
            ("OUT", output_dim),
            ("OGS", output_group_size),
        ],
        grid=(32, 32, NUM_VALUE_HEADS // 4),
        threadgroup=(32, 32, 1),
        output_shapes=[
            (1, 1, output_dim),
            (1, CONV_KERNEL - 1, CONV_DIM),
            (1, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM),
            (NUM_VALUE_HEADS // 4, output_dim, 32),
        ],
        output_dtypes=[qkv.dtype, qkv.dtype, mx.float32, mx.float32],
    )
    return tuple(outputs[:3])


def fused_gdn_runtime_supported() -> bool:
    """Report capability without constructing or launching a kernel."""
    return bool(
        hasattr(mx, "fast")
        and hasattr(mx.fast, "metal_kernel")
        and hasattr(mx, "metal")
        and mx.metal.is_available()
        and (mx.default_device() == mx.gpu)
    )


_PROBED_THREADGROUP_Y: Optional[int] = None
_PROBE_COMPLETE = False
_PROBE_LOCK = Lock()


def probe_qwen4_fused_gdn_decode(dtype) -> Optional[int]:
    """Compile candidates once and publish the supported geometry atomically."""
    global _PROBE_COMPLETE, _PROBED_THREADGROUP_Y
    if _PROBE_COMPLETE:
        return _PROBED_THREADGROUP_Y
    with _PROBE_LOCK:
        if _PROBE_COMPLETE:
            return _PROBED_THREADGROUP_Y
        if not fused_gdn_runtime_supported():
            _PROBE_COMPLETE = True
            return None
        qkv = mx.zeros((1, 1, CONV_DIM), dtype=dtype)
        z = mx.zeros((1, 1, VALUE_DIM), dtype=dtype)
        gates = mx.zeros((1, 1, NUM_VALUE_HEADS), dtype=dtype)
        conv_state = mx.zeros((1, CONV_KERNEL - 1, CONV_DIM), dtype=dtype)
        conv_weight = mx.zeros((CONV_DIM, CONV_KERNEL, 1), dtype=dtype)
        recurrent_state = mx.zeros(
            (1, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM), dtype=mx.float32
        )
        vector = mx.zeros((NUM_VALUE_HEADS,), dtype=dtype)
        A_log = mx.zeros((NUM_VALUE_HEADS,), dtype=mx.float32)
        norm_weight = mx.ones((VALUE_HEAD_DIM,), dtype=dtype)
        for threadgroup_y in _THREADGROUP_Y_CANDIDATES:
            try:
                outputs = qwen4_fused_gdn_decode(
                    qkv,
                    z,
                    gates,
                    gates,
                    conv_state,
                    conv_weight,
                    A_log,
                    vector,
                    recurrent_state,
                    norm_weight,
                    1e-06,
                    threadgroup_y=threadgroup_y,
                )
                mx.eval(*outputs)
                _PROBED_THREADGROUP_Y = threadgroup_y
                break
            except ValueError as exc:
                if "threads per threadgroup" in str(exc):
                    continue
                logger.info("Qwen4 fused GDN probe failed: %s", exc)
                break
            except RuntimeError as exc:
                logger.info(
                    "Qwen4 fused GDN threadgroup_y=%d is unavailable: %s",
                    threadgroup_y,
                    exc,
                )
                continue
        _PROBE_COMPLETE = True
        return _PROBED_THREADGROUP_Y
