# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
from __future__ import annotations
import logging
from functools import lru_cache
from threading import Lock
from typing import Any, Optional
import mlx.core as mx
from .qwen4_fused_gdn import (
    _HEADER,
    _THREADGROUP_Y_CANDIDATES,
    CONV_DIM,
    CONV_KERNEL,
    KEY_HEAD_DIM,
    NUM_KEY_HEADS,
    NUM_VALUE_HEADS,
    VALUE_DIM,
    VALUE_HEAD_DIM,
    FusedGdnAdmission,
    _dtype,
    _shape,
    _slab_width,
    admit_rollback_span,
    fused_gdn_runtime_supported,
    probe_qwen4_fused_gdn_decode,
)

logger = logging.getLogger(__name__)
MAX_VERIFY_WIDTH_PROVEN = 17
MAX_VERIFY_STEPS = 8


def admit_qwen4_fused_gdn_verify(
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
    catchup: bool = False,
    training: bool,
    sharded: bool,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    conv_kernel: int,
    gate_activation: str,
) -> FusedGdnAdmission:
    """Pure structural admission for a B=1 verify block; no MLX evaluation."""
    if training:
        return FusedGdnAdmission(False, "training")
    if sharded:
        return FusedGdnAdmission(False, "distributed sharding")
    if catchup:
        if speculating:
            return FusedGdnAdmission(False, "catch-up cache is speculating")
        refusal = admit_rollback_span(
            spans, mask, _slab_width(qkv), masked_reason="masked catch-up"
        )
        if refusal is not None:
            return refusal
    else:
        if not speculating:
            return FusedGdnAdmission(False, "not a speculative verify")
        refusal = admit_rollback_span(
            spans, mask, _slab_width(qkv), masked_reason="masked verify"
        )
        if refusal is not None:
            return refusal
    if gate_activation != "sigmoid":
        return FusedGdnAdmission(False, f"output gate {gate_activation!r}")
    geometry = (
        num_key_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
        conv_kernel,
    )
    if geometry != (
        NUM_KEY_HEADS,
        NUM_VALUE_HEADS,
        KEY_HEAD_DIM,
        VALUE_HEAD_DIM,
        CONV_KERNEL,
    ):
        return FusedGdnAdmission(False, f"unsupported geometry {geometry}")
    qkv_shape = _shape(qkv)
    if len(qkv_shape) != 3 or qkv_shape[0] != 1 or qkv_shape[2] != CONV_DIM:
        return FusedGdnAdmission(
            False, f"qkv shape {qkv_shape}, expected (1, S, {CONV_DIM})"
        )
    steps = qkv_shape[1]
    if steps < 2:
        return FusedGdnAdmission(False, f"verify width {steps} below 2")
    if steps > MAX_VERIFY_STEPS:
        return FusedGdnAdmission(
            False, f"verify width {steps} above {MAX_VERIFY_STEPS}"
        )
    expected = {
        "z": (1, steps, VALUE_DIM),
        "a": (1, steps, NUM_VALUE_HEADS),
        "b": (1, steps, NUM_VALUE_HEADS),
        "conv_state": (1, CONV_KERNEL - 1, CONV_DIM),
        "recurrent_state": (1, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM),
        "conv_weight": (CONV_DIM, CONV_KERNEL, 1),
        "A_log": (NUM_VALUE_HEADS,),
        "dt_bias": (NUM_VALUE_HEADS,),
        "norm_weight": (VALUE_HEAD_DIM,),
    }
    values = {
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


_SOURCE = "\n  const uint hv = threadgroup_position_in_grid.z;\n  const uint hk = hv / RATIO;\n  const uint lane = thread_position_in_threadgroup.x;\n  const uint ty = thread_position_in_threadgroup.y;\n  const uint tid = thread_index_in_threadgroup;\n\n  constexpr int NT = 32 * TY;\n  constexpr int NDK = DK / 32;\n  constexpr int NDV = DV / TY;\n  constexpr uint KD = (uint)(HK * DK);\n  constexpr uint VD = (uint)(HV * DV);\n  constexpr uint CD = 2u * KD + VD;\n  constexpr uint KEEP = (uint)K - 1u;\n  constexpr uint SNAPS = (uint)S - 1u;\n\n  threadgroup float sq[DK];\n  threadgroup float sk[DK];\n  threadgroup T sq_squared[DK];\n  threadgroup T sk_squared[DK];\n  threadgroup float sv[DV];\n  threadgroup float sy[DV];\n  threadgroup float shr[4];\n\n  device const float* si = recurrent_state + (size_t)hv * DV * DK;\n  device float* so = recurrent_state_out + (size_t)hv * DV * DK;\n  float st[NDV][NDK];\n  for (int j = 0; j < NDV; ++j) {\n    uint dv = ty + (uint)TY * (uint)j;\n    for (int i = 0; i < NDK; ++i)\n      st[j][i] = si[(size_t)dv * DK + NDK * lane + i];\n  }\n\n  const bool owns_shared = (hv % RATIO) == 0u;\n\n  // Convolution window bookkeeping is token independent: publish the final\n  // window (the next conv cache) and every intermediate window the layer\n  // records as a restore point.\n  for (uint idx = tid; idx < (uint)(2 * DK + DV); idx += NT) {\n    uint part = idx / (uint)DK;\n    uint d = idx - part * (uint)DK;\n    uint c = part == 0u ? hk * DK + d\n           : (part == 1u ? KD + hk * DK + d : 2u * KD + hv * DV + d);\n    if (part == 2u || owns_shared) {\n      for (uint tap = 0; tap < KEEP; ++tap) {\n        uint row = (uint)S + tap;\n        conv_state_out[(size_t)tap * CD + c] =\n            row < KEEP ? conv_state[(size_t)row * CD + c]\n                       : qkv[(size_t)(row - KEEP) * CD + c];\n      }\n      for (uint p = 1; p <= SNAPS; ++p) {\n        for (uint tap = 0; tap < KEEP; ++tap) {\n          uint row = p + tap;\n          conv_snapshots[((size_t)(p - 1u) * KEEP + tap) * CD + c] =\n              row < KEEP ? conv_state[(size_t)row * CD + c]\n                         : qkv[(size_t)(row - KEEP) * CD + c];\n        }\n      }\n    }\n  }\n\n  for (uint t = 0; t < (uint)S; ++t) {\n    for (uint idx = tid; idx < (uint)(2 * DK + DV); idx += NT) {\n      uint part = idx / (uint)DK;\n      uint d = idx - part * (uint)DK;\n      uint c = part == 0u ? hk * DK + d\n             : (part == 1u ? KD + hk * DK + d : 2u * KD + hv * DV + d);\n      device const T* wc = conv_weight + (size_t)c * K;\n      float acc = 0.0f;\n      for (uint tap = 0; tap < (uint)K; ++tap) {\n        uint row = t + tap;\n        T xv = row < KEEP ? conv_state[(size_t)row * CD + c]\n                          : qkv[(size_t)(row - KEEP) * CD + c];\n        acc += float(xv) * float(wc[tap]);\n      }\n      T xb = static_cast<T>(acc);\n      // nn.silu is reproduced by the fast sigmoid form on every finite bf16.\n      T sl = xb * mlx_sigmoid_fast(xb);\n      if (part == 0u) sq[d] = float(sl);\n      else if (part == 1u) sk[d] = float(sl);\n      else sv[d] = float(sl);\n    }\n\n    if (tid == 0u) {\n      T av = a[t * HV + hv] + dt_bias[hv];\n      T sp = mlx_softplus_fast(av);\n      shr[2] = metal::precise::exp(\n          -metal::precise::exp(float(A_log[hv])) * float(sp));\n      // mx.sigmoid on bf16 is the precise form on every finite bf16 input;\n      // the fast form differs on one (x ~ -6.85), which real activations reach.\n      shr[3] = float(mlx_sigmoid_precise(b[t * HV + hv]));\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n    for (uint d = tid; d < (uint)DK; d += NT) {\n      T qv = static_cast<T>(sq[d]);\n      T kv = static_cast<T>(sk[d]);\n      sq_squared[d] = static_cast<T>(qv * qv);\n      sk_squared[d] = static_cast<T>(kv * kv);\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n    if (simdgroup_index_in_threadgroup == 0u) {\n      T pq = static_cast<T>(0), pk = static_cast<T>(0);\n      uint base = 4u * lane;\n      for (int i = 0; i < 4; ++i) {\n        pq = static_cast<T>(sq_squared[base + i] + pq);\n        pk = static_cast<T>(sk_squared[base + i] + pk);\n      }\n      pq = static_cast<T>(simd_sum(float(pq)));\n      pk = static_cast<T>(simd_sum(float(pk)));\n      if (lane == 0u) {\n        T eps = static_cast<T>(1.0e-6f);\n        T qdenom = pq + eps;\n        T kdenom = pk + eps;\n        shr[0] = float(static_cast<T>(metal::precise::rsqrt(qdenom)));\n        shr[1] = float(static_cast<T>(metal::precise::rsqrt(kdenom)));\n      }\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    T qscale = static_cast<T>(0.08838834764831845f);\n    for (uint d = tid; d < (uint)DK; d += NT) {\n      T q_normalized = static_cast<T>(static_cast<T>(sq[d]) * static_cast<T>(shr[0]));\n      T k_normalized = static_cast<T>(static_cast<T>(sk[d]) * static_cast<T>(shr[1]));\n      sq[d] = float(static_cast<T>(q_normalized * qscale));\n      sk[d] = float(k_normalized);\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n    device float* state_dst =\n        t < SNAPS ? state_snapshots + ((size_t)t * HV + hv) * DV * DK : so;\n    for (int j = 0; j < NDV; ++j) {\n      uint dv = ty + (uint)TY * (uint)j;\n      float kv = 0.0f;\n      for (int i = 0; i < NDK; ++i) {\n        uint s = NDK * lane + i;\n        st[j][i] = st[j][i] * shr[2];\n        kv += st[j][i] * sk[s];\n      }\n      kv = simd_sum(kv);\n      float delta = (sv[dv] - kv) * shr[3];\n      float out = 0.0f;\n      for (int i = 0; i < NDK; ++i) {\n        uint s = NDK * lane + i;\n        st[j][i] = st[j][i] + sk[s] * delta;\n        out += st[j][i] * sq[s];\n      }\n      out = simd_sum(out);\n      if (thread_index_in_simdgroup == 0u)\n        sy[dv] = float(static_cast<T>(out));\n      for (int i = 0; i < NDK; ++i)\n        state_dst[(size_t)dv * DK + NDK * lane + i] = st[j][i];\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n    if (simdgroup_index_in_threadgroup == 0u) {\n      float po = 0.0f;\n      uint base = 4u * lane;\n      for (int i = 0; i < 4; ++i) po += sy[base + i] * sy[base + i];\n      po = simd_sum(po);\n      if (lane == 0u)\n        shr[0] = metal::precise::rsqrt(po / (float)DV + norm_eps);\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    for (uint d = tid; d < (uint)DV; d += NT) {\n      T normalized = static_cast<T>(sy[d] * shr[0]);\n      normalized = norm_weight[d] * normalized;\n      // float32 sigmoid of a bf16-valued gate: the precise form matches\n      // mx.sigmoid on every finite bf16 input; the fast form differs on ~1%.\n      float x = float(normalized) *\n                mlx_sigmoid_precise<float>(float(z[t * VD + hv * DV + d]));\n      output[t * VD + hv * DV + d] = static_cast<T>(x);\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n  }\n"


def _derive_catchup_source() -> str:
    """Remove rollback-only stores from the exact verify kernel body.

    Catch-up always commits the full token block, so it needs the output and
    the two final cache states but none of the per-token restore points.  Keep
    this as a checked derivation of ``_SOURCE`` so the arithmetic and rounding
    sequence cannot drift from the already-proven kernel accidentally.
    """
    source = _SOURCE
    conv_snapshot_loop = "      for (uint p = 1; p <= SNAPS; ++p) {\n        for (uint tap = 0; tap < KEEP; ++tap) {\n          uint row = p + tap;\n          conv_snapshots[((size_t)(p - 1u) * KEEP + tap) * CD + c] =\n              row < KEEP ? conv_state[(size_t)row * CD + c]\n                         : qkv[(size_t)(row - KEEP) * CD + c];\n        }\n      }\n"
    state_destination = "    device float* state_dst =\n        t < SNAPS ? state_snapshots + ((size_t)t * HV + hv) * DV * DK : so;\n"
    state_store = "      for (int i = 0; i < NDK; ++i)\n        state_dst[(size_t)dv * DK + NDK * lane + i] = st[j][i];\n"
    final_state_store = "      if (t + 1u == (uint)S) {\n        for (int i = 0; i < NDK; ++i)\n          so[(size_t)dv * DK + NDK * lane + i] = st[j][i];\n      }\n"
    replacements = (
        (conv_snapshot_loop, ""),
        (state_destination, ""),
        (state_store, final_state_store),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError("fused GDN catch-up source derivation drifted")
        source = source.replace(old, new, 1)
    return source


_CATCHUP_SOURCE = _derive_catchup_source()


def _derive_replay_source() -> str:
    """Replace intermediate state snapshots with a compact correction tape.

    The verify arithmetic remains byte-for-byte derived from ``_SOURCE``.  The
    final state is still materialized for full acceptance; only partial
    rollback uses the compact ``(key, correction, decay)`` tape.
    """
    source = _SOURCE
    conv_snapshot_loop = "      for (uint p = 1; p <= SNAPS; ++p) {\n        for (uint tap = 0; tap < KEEP; ++tap) {\n          uint row = p + tap;\n          conv_snapshots[((size_t)(p - 1u) * KEEP + tap) * CD + c] =\n              row < KEEP ? conv_state[(size_t)row * CD + c]\n                         : qkv[(size_t)(row - KEEP) * CD + c];\n        }\n      }\n"
    state_destination = "    device float* state_dst =\n        t < SNAPS ? state_snapshots + ((size_t)t * HV + hv) * DV * DK : so;\n"
    state_store = "      for (int i = 0; i < NDK; ++i)\n        state_dst[(size_t)dv * DK + NDK * lane + i] = st[j][i];\n"
    final_state_store = "      if (t + 1u == (uint)S) {\n        for (int i = 0; i < NDK; ++i)\n          so[(size_t)dv * DK + NDK * lane + i] = st[j][i];\n      }\n"
    normalized_barrier = "    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n    device float* state_dst =\n"
    tape_preamble = "    threadgroup_barrier(mem_flags::mem_threadgroup);\n    if (t < SNAPS && owns_shared) {\n      for (uint d = tid; d < (uint)DK; d += NT)\n        replay_keys[(size_t)t * KD + hk * DK + d] = static_cast<T>(sk[d]);\n    }\n    if (t < SNAPS && tid == 0u)\n      replay_decay[(size_t)t * HV + hv] = shr[2];\n\n    device float* state_dst =\n"
    delta_line = "      float delta = (sv[dv] - kv) * shr[3];\n"
    tape_delta = "      float delta = (sv[dv] - kv) * shr[3];\n      if (t < SNAPS && thread_index_in_simdgroup == 0u)\n        replay_corrections[((size_t)t * HV + hv) * DV + dv] = delta;\n"
    replacements = (
        (conv_snapshot_loop, ""),
        (normalized_barrier, tape_preamble),
        (state_destination, ""),
        (delta_line, tape_delta),
        (state_store, final_state_store),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError("fused GDN compact-replay source derivation drifted")
        source = source.replace(old, new, 1)
    return source


_REPLAY_SOURCE = _derive_replay_source()

_RECONSTRUCT_SOURCE = r"""
  const uint hv = threadgroup_position_in_grid.z;
  const uint hk = hv / RATIO;
  const uint lane = thread_position_in_threadgroup.x;
  const uint ty = thread_position_in_threadgroup.y;

  constexpr int NDK = DK / 32;
  constexpr int NDV = DV / TY;
  constexpr uint KD = (uint)(HK * DK);

  device const float* si = recurrent_state + (size_t)hv * DV * DK;
  device float* so = recurrent_state_out + (size_t)hv * DV * DK;
  for (int j = 0; j < NDV; ++j) {
    uint dv = ty + (uint)TY * (uint)j;
    for (int i = 0; i < NDK; ++i) {
      uint dk = NDK * lane + i;
      float st = si[(size_t)dv * DK + dk];
      for (uint t = 0; t < (uint)M; ++t) {
        float decay = replay_decay[(size_t)t * HV + hv];
        float correction =
            replay_corrections[((size_t)t * HV + hv) * DV + dv];
        float key = float(replay_keys[(size_t)t * KD + hk * DK + dk]);
        st = st * decay;
        st = st + key * correction;
      }
      so[(size_t)dv * DK + dk] = st;
    }
  }
"""


@lru_cache(maxsize=None)
def _kernel():
    return mx.fast.metal_kernel(
        name="qwen4_fused_gdn_verify",
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
        output_names=[
            "output",
            "conv_state_out",
            "recurrent_state_out",
            "state_snapshots",
            "conv_snapshots",
        ],
        header=_HEADER,
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=None)
def _catchup_kernel():
    return mx.fast.metal_kernel(
        name="qwen4_fused_gdn_catchup",
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
        source=_CATCHUP_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=None)
def _replay_verify_kernel():
    return mx.fast.metal_kernel(
        name="qwen4_fused_gdn_replay_verify",
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
        output_names=[
            "output",
            "conv_state_out",
            "recurrent_state_out",
            "replay_keys",
            "replay_corrections",
            "replay_decay",
        ],
        header=_HEADER,
        source=_REPLAY_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=None)
def _reconstruct_kernel():
    return mx.fast.metal_kernel(
        name="qwen4_fused_gdn_reconstruct",
        input_names=[
            "recurrent_state",
            "replay_keys",
            "replay_corrections",
            "replay_decay",
        ],
        output_names=["recurrent_state_out"],
        source=_RECONSTRUCT_SOURCE,
        ensure_row_contiguous=True,
    )


def qwen4_fused_gdn_verify(
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
):
    """Build the fused verify graph. Callers must run structural admission first.

    Returns ``(output, conv_state_out, recurrent_state_out, state_snapshots,
    conv_snapshots)``; ``state_snapshots[:, p]`` and ``conv_snapshots[:, p]``
    are the recurrent state and convolution window after ``p + 1`` tokens for
    ``p`` in ``range(S - 1)``.
    """
    if threadgroup_y not in _THREADGROUP_Y_CANDIDATES:
        raise ValueError(
            f"unsupported threadgroup_y {threadgroup_y}; expected one of {_THREADGROUP_Y_CANDIDATES}"
        )
    steps = int(qkv.shape[1])
    if not 2 <= steps <= MAX_VERIFY_WIDTH_PROVEN:
        raise ValueError(
            f"unsupported verify width {steps}; expected 2..{MAX_VERIFY_WIDTH_PROVEN}"
        )
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
            ("HK", NUM_KEY_HEADS),
            ("HV", NUM_VALUE_HEADS),
            ("DK", KEY_HEAD_DIM),
            ("DV", VALUE_HEAD_DIM),
            ("K", CONV_KERNEL),
            ("S", steps),
            ("TY", threadgroup_y),
            ("RATIO", NUM_VALUE_HEADS // NUM_KEY_HEADS),
        ],
        grid=(32, threadgroup_y, NUM_VALUE_HEADS),
        threadgroup=(32, threadgroup_y, 1),
        output_shapes=[
            (1, steps, VALUE_DIM),
            (1, CONV_KERNEL - 1, CONV_DIM),
            (1, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM),
            (1, steps - 1, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM),
            (1, steps - 1, CONV_KERNEL - 1, CONV_DIM),
        ],
        output_dtypes=[qkv.dtype, qkv.dtype, mx.float32, mx.float32, qkv.dtype],
    )
    return tuple(outputs)


def qwen4_fused_gdn_replay_verify(
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
):
    """Build the compact-tape verify graph selected by qualified profiles."""
    if threadgroup_y not in _THREADGROUP_Y_CANDIDATES:
        raise ValueError(
            f"unsupported threadgroup_y {threadgroup_y}; expected one of {_THREADGROUP_Y_CANDIDATES}"
        )
    steps = int(qkv.shape[1])
    if not 2 <= steps <= MAX_VERIFY_WIDTH_PROVEN:
        raise ValueError(
            f"unsupported verify width {steps}; expected 2..{MAX_VERIFY_WIDTH_PROVEN}"
        )
    outputs = _replay_verify_kernel()(
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
            ("HK", NUM_KEY_HEADS),
            ("HV", NUM_VALUE_HEADS),
            ("DK", KEY_HEAD_DIM),
            ("DV", VALUE_HEAD_DIM),
            ("K", CONV_KERNEL),
            ("S", steps),
            ("TY", threadgroup_y),
            ("RATIO", NUM_VALUE_HEADS // NUM_KEY_HEADS),
        ],
        grid=(32, threadgroup_y, NUM_VALUE_HEADS),
        threadgroup=(32, threadgroup_y, 1),
        output_shapes=[
            (1, steps, VALUE_DIM),
            (1, CONV_KERNEL - 1, CONV_DIM),
            (1, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM),
            (1, steps - 1, NUM_KEY_HEADS, KEY_HEAD_DIM),
            (1, steps - 1, NUM_VALUE_HEADS, VALUE_HEAD_DIM),
            (1, steps - 1, NUM_VALUE_HEADS),
        ],
        output_dtypes=[
            qkv.dtype,
            qkv.dtype,
            mx.float32,
            qkv.dtype,
            mx.float32,
            mx.float32,
        ],
    )
    return tuple(outputs)


def validate_qwen4_gdn_replay_acceptance(accepted: int, tape_steps: int) -> int:
    """Validate a partial-acceptance request before dispatching reconstruction."""
    accepted = int(accepted)
    tape_steps = int(tape_steps)
    if tape_steps < 1:
        raise ValueError("compact replay tape must contain at least one step")
    if not 1 <= accepted <= tape_steps:
        raise ValueError(
            f"accepted prefix {accepted} outside compact replay tape 1..{tape_steps}"
        )
    return accepted


def qwen4_fused_gdn_reconstruct(
    recurrent_state,
    replay_keys,
    replay_corrections,
    replay_decay,
    accepted: int,
    *,
    threadgroup_y: int,
):
    """Reconstruct one partial-acceptance state from the pre-verify checkpoint."""
    if threadgroup_y not in _THREADGROUP_Y_CANDIDATES:
        raise ValueError(
            f"unsupported threadgroup_y {threadgroup_y}; expected one of {_THREADGROUP_Y_CANDIDATES}"
        )
    accepted = validate_qwen4_gdn_replay_acceptance(
        accepted, int(replay_decay.shape[1])
    )
    return _reconstruct_kernel()(
        inputs=[recurrent_state, replay_keys, replay_corrections, replay_decay],
        template=[
            ("T", replay_keys.dtype),
            ("HK", NUM_KEY_HEADS),
            ("HV", NUM_VALUE_HEADS),
            ("DK", KEY_HEAD_DIM),
            ("DV", VALUE_HEAD_DIM),
            ("M", accepted),
            ("TY", threadgroup_y),
            ("RATIO", NUM_VALUE_HEADS // NUM_KEY_HEADS),
        ],
        grid=(32, threadgroup_y, NUM_VALUE_HEADS),
        threadgroup=(32, threadgroup_y, 1),
        output_shapes=[
            (1, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM),
        ],
        output_dtypes=[mx.float32],
    )[0]


def qwen4_fused_gdn_catchup(
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
):
    """Build the no-rollback catch-up graph for an always-committed block."""
    if threadgroup_y not in _THREADGROUP_Y_CANDIDATES:
        raise ValueError(
            f"unsupported threadgroup_y {threadgroup_y}; expected one of {_THREADGROUP_Y_CANDIDATES}"
        )
    steps = int(qkv.shape[1])
    if not 2 <= steps <= MAX_VERIFY_WIDTH_PROVEN:
        raise ValueError(
            f"unsupported catch-up width {steps}; expected 2..{MAX_VERIFY_WIDTH_PROVEN}"
        )
    outputs = _catchup_kernel()(
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
            ("HK", NUM_KEY_HEADS),
            ("HV", NUM_VALUE_HEADS),
            ("DK", KEY_HEAD_DIM),
            ("DV", VALUE_HEAD_DIM),
            ("K", CONV_KERNEL),
            ("S", steps),
            ("TY", threadgroup_y),
            ("RATIO", NUM_VALUE_HEADS // NUM_KEY_HEADS),
        ],
        grid=(32, threadgroup_y, NUM_VALUE_HEADS),
        threadgroup=(32, threadgroup_y, 1),
        output_shapes=[
            (1, steps, VALUE_DIM),
            (1, CONV_KERNEL - 1, CONV_DIM),
            (1, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM),
        ],
        output_dtypes=[qkv.dtype, qkv.dtype, mx.float32],
    )
    return tuple(outputs)


_PROBED_STEPS: dict[int, Optional[int]] = {}
_PROBED_CATCHUP_STEPS: dict[int, Optional[int]] = {}
_PROBED_REPLAY_STEPS: dict[int, Optional[int]] = {}
_PROBE_LOCK = Lock()


def probe_qwen4_fused_gdn_replay_verify(dtype, steps: int) -> Optional[int]:
    """Compile the compact verify and every partial reconstruction width."""
    steps = int(steps)
    if steps in _PROBED_REPLAY_STEPS:
        return _PROBED_REPLAY_STEPS[steps]
    with _PROBE_LOCK:
        if steps in _PROBED_REPLAY_STEPS:
            return _PROBED_REPLAY_STEPS[steps]
        if (
            not 2 <= steps <= MAX_VERIFY_WIDTH_PROVEN
            or not fused_gdn_runtime_supported()
        ):
            _PROBED_REPLAY_STEPS[steps] = None
            return None
        start = probe_qwen4_fused_gdn_decode(dtype)
        if start is None:
            _PROBED_REPLAY_STEPS[steps] = None
            return None
        qkv = mx.zeros((1, steps, CONV_DIM), dtype=dtype)
        z = mx.zeros((1, steps, VALUE_DIM), dtype=dtype)
        gates = mx.zeros((1, steps, NUM_VALUE_HEADS), dtype=dtype)
        conv_state = mx.zeros((1, CONV_KERNEL - 1, CONV_DIM), dtype=dtype)
        conv_weight = mx.zeros((CONV_DIM, CONV_KERNEL, 1), dtype=dtype)
        recurrent_state = mx.zeros(
            (1, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM), dtype=mx.float32
        )
        vector = mx.zeros((NUM_VALUE_HEADS,), dtype=dtype)
        A_log = mx.zeros((NUM_VALUE_HEADS,), dtype=mx.float32)
        norm_weight = mx.ones((VALUE_HEAD_DIM,), dtype=dtype)
        result: Optional[int] = None
        for threadgroup_y in [c for c in _THREADGROUP_Y_CANDIDATES if c <= start]:
            try:
                outputs = qwen4_fused_gdn_replay_verify(
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
                replays = [
                    qwen4_fused_gdn_reconstruct(
                        recurrent_state,
                        outputs[3],
                        outputs[4],
                        outputs[5],
                        accepted,
                        threadgroup_y=threadgroup_y,
                    )
                    for accepted in range(1, steps)
                ]
                mx.eval(*outputs, *replays)
                result = threadgroup_y
                break
            except ValueError as exc:
                if "threads per threadgroup" in str(exc):
                    continue
                logger.info("Qwen4 compact GDN replay probe failed: %s", exc)
                break
            except RuntimeError as exc:
                logger.info(
                    "Qwen4 compact GDN replay width %d unavailable at threadgroup_y=%d: %s",
                    steps,
                    threadgroup_y,
                    exc,
                )
                continue
        _PROBED_REPLAY_STEPS[steps] = result
        return result


def probe_qwen4_fused_gdn_verify(dtype, steps: int) -> Optional[int]:
    """Compile-and-run one verify specialization once per width.

    The decode probe's published ``threadgroup_y`` is tried first, then every
    smaller candidate, like the decode ladder. A width whose specializations
    all fail stays on the stock path. Cached widths are read lock-free.
    """
    steps = int(steps)
    if steps in _PROBED_STEPS:
        return _PROBED_STEPS[steps]
    with _PROBE_LOCK:
        if steps in _PROBED_STEPS:
            return _PROBED_STEPS[steps]
        if (
            not 2 <= steps <= MAX_VERIFY_WIDTH_PROVEN
            or not fused_gdn_runtime_supported()
        ):
            _PROBED_STEPS[steps] = None
            return None
        start = probe_qwen4_fused_gdn_decode(dtype)
        if start is None:
            _PROBED_STEPS[steps] = None
            return None
        qkv = mx.zeros((1, steps, CONV_DIM), dtype=dtype)
        z = mx.zeros((1, steps, VALUE_DIM), dtype=dtype)
        gates = mx.zeros((1, steps, NUM_VALUE_HEADS), dtype=dtype)
        conv_state = mx.zeros((1, CONV_KERNEL - 1, CONV_DIM), dtype=dtype)
        conv_weight = mx.zeros((CONV_DIM, CONV_KERNEL, 1), dtype=dtype)
        recurrent_state = mx.zeros(
            (1, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM), dtype=mx.float32
        )
        vector = mx.zeros((NUM_VALUE_HEADS,), dtype=dtype)
        A_log = mx.zeros((NUM_VALUE_HEADS,), dtype=mx.float32)
        norm_weight = mx.ones((VALUE_HEAD_DIM,), dtype=dtype)
        result: Optional[int] = None
        for threadgroup_y in [c for c in _THREADGROUP_Y_CANDIDATES if c <= start]:
            try:
                outputs = qwen4_fused_gdn_verify(
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
                result = threadgroup_y
                break
            except ValueError as exc:
                if "threads per threadgroup" in str(exc):
                    continue
                logger.info("Qwen4 fused GDN verify probe failed: %s", exc)
                break
            except RuntimeError as exc:
                logger.info(
                    "Qwen4 fused GDN verify width %d unavailable at threadgroup_y=%d: %s",
                    steps,
                    threadgroup_y,
                    exc,
                )
                continue
        _PROBED_STEPS[steps] = result
        return result


def probe_qwen4_fused_gdn_catchup(dtype, steps: int) -> Optional[int]:
    """Compile-and-run the no-snapshot catch-up specialization once."""
    steps = int(steps)
    if steps in _PROBED_CATCHUP_STEPS:
        return _PROBED_CATCHUP_STEPS[steps]
    with _PROBE_LOCK:
        if steps in _PROBED_CATCHUP_STEPS:
            return _PROBED_CATCHUP_STEPS[steps]
        if (
            not 2 <= steps <= MAX_VERIFY_WIDTH_PROVEN
            or not fused_gdn_runtime_supported()
        ):
            _PROBED_CATCHUP_STEPS[steps] = None
            return None
        start = probe_qwen4_fused_gdn_decode(dtype)
        if start is None:
            _PROBED_CATCHUP_STEPS[steps] = None
            return None
        qkv = mx.zeros((1, steps, CONV_DIM), dtype=dtype)
        z = mx.zeros((1, steps, VALUE_DIM), dtype=dtype)
        gates = mx.zeros((1, steps, NUM_VALUE_HEADS), dtype=dtype)
        conv_state = mx.zeros((1, CONV_KERNEL - 1, CONV_DIM), dtype=dtype)
        conv_weight = mx.zeros((CONV_DIM, CONV_KERNEL, 1), dtype=dtype)
        recurrent_state = mx.zeros(
            (1, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM), dtype=mx.float32
        )
        vector = mx.zeros((NUM_VALUE_HEADS,), dtype=dtype)
        A_log = mx.zeros((NUM_VALUE_HEADS,), dtype=mx.float32)
        norm_weight = mx.ones((VALUE_HEAD_DIM,), dtype=dtype)
        result: Optional[int] = None
        for threadgroup_y in [c for c in _THREADGROUP_Y_CANDIDATES if c <= start]:
            try:
                outputs = qwen4_fused_gdn_catchup(
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
                result = threadgroup_y
                break
            except ValueError as exc:
                if "threads per threadgroup" in str(exc):
                    continue
                logger.info("Qwen4 fused GDN catch-up probe failed: %s", exc)
                break
            except RuntimeError as exc:
                logger.info(
                    "Qwen4 fused GDN catch-up width %d unavailable at threadgroup_y=%d: %s",
                    steps,
                    threadgroup_y,
                    exc,
                )
                continue
        _PROBED_CATCHUP_STEPS[steps] = result
        return result
