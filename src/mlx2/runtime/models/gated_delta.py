# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
import os
from functools import partial
from typing import Optional, Tuple
import mlx.core as mx
import mlx.nn as nn
from .precise_ops import gate_sigmoid

_ENABLE_GDN_PACKED = os.environ.get("MLX_GDN_PACKED", "1") != "0"
_ENABLE_GDN_CORE = os.environ.get("MLX_GDN_CORE", "0") == "1"
_CORE_GDN_CHUNK_SIZE = 8
_CORE_GDN_MIN_T = 17
_CORE_GDN_MAX_T = 256
_CORE_GDN_HEADS = frozenset(
    {(24, 24), (32, 32), (16, 16), (16, 32), (16, 48), (16, 64)}
)
_core_gated_delta_update = getattr(mx.fast, "gated_delta_update", None)


def _readout_needs_widening(input_type, state_type) -> bool:
    """Whether the fp32 state's range can overflow the activation dtype.

    A dtype with the state's exponent width holds its maximum to within a
    factor of two (bfloat16: 3.39e38 vs float32 3.40e38); a narrower one is
    smaller by many orders of magnitude (float16: 65504). So this comparison
    is the exponent-width test.
    """
    if input_type == state_type:
        return False
    return mx.finfo(input_type).max < mx.finfo(state_type).max / 2


_SATURATING_CASTS = {}


def _saturating_cast(dtype):
    """Compiled clamp-then-cast, so the narrowing is one elementwise pass."""
    fn = _SATURATING_CASTS.get(dtype)
    if fn is None:
        info = mx.finfo(dtype)
        fn = mx.compile(
            lambda y, lo=info.min, hi=info.max: mx.clip(y, lo, hi).astype(dtype),
            shapeless=True,
        )
        _SATURATING_CASTS[dtype] = fn
    return fn


def _cast_readout(y, input_type, state_type):
    """Narrow the readout to the activation dtype, saturating on overflow.

    Every GDN caller feeds the readout straight into a scale-invariant
    RMSNorm, so one infinite component gives inf/inf = NaN and destroys the
    layer. Clamping keeps the vector finite and signed. NaN still propagates.
    """
    if not _readout_needs_widening(input_type, state_type):
        return y.astype(input_type)
    return _saturating_cast(input_type)(y)


def _can_use_core_gated_delta(q, k, v, g, state, mask):
    """Whether the fixed MLX primitive covers this exact recurrence layout."""
    if not _ENABLE_GDN_CORE or _core_gated_delta_update is None:
        return False
    if (
        mask is not None
        or g.ndim != 3
        or (not _CORE_GDN_MIN_T <= q.shape[1] <= _CORE_GDN_MAX_T)
    ):
        return False
    (Hk, Dk) = q.shape[2:]
    (Hv, Dv) = v.shape[2:]
    return (
        k.shape == q.shape
        and (Hk, Hv) in _CORE_GDN_HEADS
        and (Dk == 128)
        and (Dv == 128)
        and (g.dtype == mx.float32)
        and (state.dtype == mx.float32)
        and (q.dtype in (mx.float32, mx.bfloat16, mx.float16))
        and (k.dtype in (mx.float32, mx.bfloat16, mx.float16))
        and (v.dtype in (mx.float32, mx.bfloat16, mx.float16))
        and (not _readout_needs_widening(q.dtype, state.dtype))
    )


@partial(mx.compile, shapeless=True)
def compute_g(A_log, a, dt_bias):
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias))


@partial(mx.compile, shapeless=True)
def compute_lower_bound_g(A_log, a, dt_bias, lower_bound):
    return mx.exp(
        lower_bound
        * mx.sigmoid(
            mx.exp(A_log.astype(mx.float32)) * (a.astype(mx.float32) + dt_bias)
        )
    )


def _make_gated_delta_kernel(has_mask=False, vectorized=False):
    if not mx.metal.is_available():
        return None
    mask_source = "mask[b_idx * T + t]" if has_mask else "true"
    if vectorized:
        g_comment = "// g: [B, T, Hv, Dk]"
        g_setup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        g_access = "g_[s_idx]"
        g_advance = "g_ += Hv * Dk;"
    else:
        g_comment = "// g: [B, T, Hv]"
        g_setup = "auto g_ = g + b_idx * T * Hv;"
        g_access = "g_[hv_idx]"
        g_advance = "g_ += Hv;"
    source = f"\n        auto n = thread_position_in_grid.z;\n        auto b_idx = n / Hv;\n        auto hv_idx = n % Hv;\n        auto hk_idx = hv_idx / (Hv / Hk);\n        constexpr int n_per_t = Dk / 32;\n\n        // q, k: [B, T, Hk, Dk]\n        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;\n        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;\n\n        // v, y: [B, T, Hv, Dv]\n        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;\n        y += b_idx * T * Hv * Dv + hv_idx * Dv;\n\n        auto dk_idx = thread_position_in_threadgroup.x;\n        auto dv_idx = thread_position_in_grid.y;\n\n        // state_in, state_out: [B, Hv, Dv, Dk]\n        auto i_state = state_in + (n * Dv + dv_idx) * Dk;\n        auto o_state = state_out + (n * Dv + dv_idx) * Dk;\n\n        float state[n_per_t];\n        for (int i = 0; i < n_per_t; ++i) {{\n          auto s_idx = n_per_t * dk_idx + i;\n          state[i] = static_cast<float>(i_state[s_idx]);\n        }}\n\n        {g_comment}\n        {g_setup}\n        auto beta_ = beta + b_idx * T * Hv;\n\n        for (int t = 0; t < T; ++t) {{\n          if ({mask_source}) {{\n            float kv_mem = 0.0f;\n            for (int i = 0; i < n_per_t; ++i) {{\n              auto s_idx = n_per_t * dk_idx + i;\n              state[i] = state[i] * {g_access};\n              kv_mem += state[i] * k_[s_idx];\n            }}\n            kv_mem = simd_sum(kv_mem);\n\n            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];\n\n            float out = 0.0f;\n            for (int i = 0; i < n_per_t; ++i) {{\n              auto s_idx = n_per_t * dk_idx + i;\n              state[i] = state[i] + k_[s_idx] * delta;\n              out += state[i] * q_[s_idx];\n            }}\n            out = simd_sum(out);\n            if (thread_index_in_simdgroup == 0) {{\n              y[dv_idx] = static_cast<InT>(out);\n            }}\n          }} else {{\n            y[dv_idx] = static_cast<InT>(0);\n          }}\n          // Increment data pointers to next time step\n          q_ += Hk * Dk;\n          k_ += Hk * Dk;\n          v_ += Hv * Dv;\n          y += Hv * Dv;\n          {g_advance}\n          beta_ += Hv;\n        }}\n        for (int i = 0; i < n_per_t; ++i) {{\n          auto s_idx = n_per_t * dk_idx + i;\n          o_state[s_idx] = static_cast<StT>(state[i]);\n        }}\n    "
    inputs = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if has_mask:
        inputs.append("mask")
    suffix = ""
    if vectorized:
        suffix += "_vec"
    if has_mask:
        suffix += "_mask"
    return mx.fast.metal_kernel(
        name=f"gated_delta_step{suffix}",
        input_names=inputs,
        output_names=["y", "state_out"],
        source=source,
    )


def _make_gated_delta_packed_kernel():
    """Make the scalar-gate Dk=128 prefill specialization.

    The generic kernel assigns one 32-lane SIMD-group to each value row. For
    Dk=128 that leaves every lane with only four state elements and performs
    two full-SIMD reductions per row and token. This kernel instead packs
    eight independent value rows into a SIMD-group: four lanes own each row
    and each lane keeps 32 contiguous state elements in registers.

    The reduction reproduces the unpacked comparator
    (_make_gated_delta_kernel_xtree) bitwise BY CONSTRUCTION: both use the
    same explicitly-written ascending butterfly (shuffle_xor 1,2,4,8,16)
    rather than relying on how simd_sum lowers. The butterfly's first three
    levels combine partials that live in a single packed lane (IEEE addition
    is commutative, so the local pairwise tree is bit-identical), and the
    last two levels map onto shuffle_xor(1) and shuffle_xor(2) within the
    four-lane row group. Each 4-element partial keeps the comparator's
    sequential order, so y and the state are bit-identical to it on any
    device. On current Apple GPUs the explicit tree is also bit-identical
    to the simd_sum-based generic kernel.
    """
    if not mx.metal.is_available():
        return None
    source = "\n        constexpr int lanes_per_row = 4;\n        constexpr int rows_per_simdgroup = 32 / lanes_per_row;\n        constexpr int values_per_lane = Dk / lanes_per_row;\n        constexpr int partials_per_lane = values_per_lane / 4;\n\n        auto n = thread_position_in_grid.z;\n        auto b_idx = n / Hv;\n        auto hv_idx = n % Hv;\n        auto hk_idx = hv_idx / (Hv / Hk);\n\n        auto lane = thread_index_in_simdgroup;\n        auto row_in_simdgroup = lane / lanes_per_row;\n        auto lane_in_row = lane & (lanes_per_row - 1);\n        auto row_group = thread_position_in_grid.y;\n        auto dv_idx = row_group * rows_per_simdgroup + row_in_simdgroup;\n\n        // q, k: [B, T, Hk, Dk]\n        auto q_ = q + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;\n        auto k_ = k + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;\n\n        // v, y: [B, T, Hv, Dv]\n        auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;\n        y += (b_idx * T * Hv + hv_idx) * Dv;\n\n        // state_in, state_out: [B, Hv, Dv, Dk]\n        auto i_state = state_in + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;\n        auto o_state = state_out + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;\n\n        float state[values_per_lane];\n        for (int i = 0; i < values_per_lane; ++i) {\n          state[i] = static_cast<float>(i_state[i]);\n        }\n\n        // g, beta: [B, T, Hv]\n        auto g_ = g + b_idx * T * Hv;\n        auto beta_ = beta + b_idx * T * Hv;\n\n        for (int t = 0; t < T; ++t) {\n          float gt = static_cast<float>(g_[hv_idx]);\n\n          // Partials mirror the generic kernel: each 4-element chain is one\n          // original lane's sequential accumulation.\n          float part[partials_per_lane];\n          for (int pb = 0; pb < partials_per_lane; ++pb) {\n            float acc = 0.0f;\n            for (int i = 0; i < 4; ++i) {\n              int e = pb * 4 + i;\n              state[e] = state[e] * gt;\n              acc += state[e] * static_cast<float>(k_[e]);\n            }\n            part[pb] = acc;\n          }\n          // Butterfly levels xor 1,2,4 stay inside this lane (commutative\n          // pairwise tree); levels xor 8,16 become the row-group shuffles.\n          float kv_mem =\n              ((part[0] + part[1]) + (part[2] + part[3])) +\n              ((part[4] + part[5]) + (part[6] + part[7]));\n          kv_mem += simd_shuffle_xor(kv_mem, 1);\n          kv_mem += simd_shuffle_xor(kv_mem, 2);\n\n          auto delta =\n              (static_cast<float>(v_[dv_idx]) - kv_mem) *\n              static_cast<float>(beta_[hv_idx]);\n\n          for (int pb = 0; pb < partials_per_lane; ++pb) {\n            float acc = 0.0f;\n            for (int i = 0; i < 4; ++i) {\n              int e = pb * 4 + i;\n              state[e] = state[e] + static_cast<float>(k_[e]) * delta;\n              acc += state[e] * static_cast<float>(q_[e]);\n            }\n            part[pb] = acc;\n          }\n          float out =\n              ((part[0] + part[1]) + (part[2] + part[3])) +\n              ((part[4] + part[5]) + (part[6] + part[7]));\n          out += simd_shuffle_xor(out, 1);\n          out += simd_shuffle_xor(out, 2);\n          if (lane_in_row == 0) {\n            y[dv_idx] = static_cast<InT>(out);\n          }\n\n          q_ += Hk * Dk;\n          k_ += Hk * Dk;\n          v_ += Hv * Dv;\n          y += Hv * Dv;\n          g_ += Hv;\n          beta_ += Hv;\n        }\n\n        for (int i = 0; i < values_per_lane; ++i) {\n          o_state[i] = static_cast<StT>(state[i]);\n        }\n    "
    return mx.fast.metal_kernel(
        name="gated_delta_step_packed_btree",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"],
        source=source,
    )


_gated_delta_kernel = _make_gated_delta_kernel(has_mask=False, vectorized=False)
_gated_delta_kernel_masked = _make_gated_delta_kernel(has_mask=True, vectorized=False)
_gated_delta_kernel_vec = _make_gated_delta_kernel(has_mask=False, vectorized=True)
_gated_delta_kernel_vec_masked = _make_gated_delta_kernel(
    has_mask=True, vectorized=True
)
_gated_delta_kernel_packed = _make_gated_delta_packed_kernel()


@mx.compile
def _gated_delta_step_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Ops-based reference implementation for a single recurrent step.

    Shapes:
      - q, k: [B, H, Dk]
      - v: [B, H, Dv]
      - g: [B, H] or [B, H, Dk]
      - beta: [B, H]
      - state: [B, H, Dv, Dk]
    Returns:
      - y: [B, H, Dv]
      - new_state: [B, H, Dv, Dk]
    """
    old_state = state
    if g.ndim == 2:
        decay = g[..., None, None]
    elif g.ndim == 3:
        decay = g[..., None, :]
    else:
        raise ValueError(f"Unsupported gating shape {g.shape}")
    state = state * decay
    kv_mem = (state * k[..., None, :]).sum(axis=-1)
    delta = (v - kv_mem) * beta[..., None]
    state = state + k[..., None, :] * delta[..., None]
    y = (state * q[..., None, :]).sum(axis=-1)
    if mask is not None:
        mask = mx.expand_dims(mask, axis=(1, 2, 3))
        state = mx.where(mask, state, old_state)
    return (_cast_readout(y, q.dtype, old_state.dtype), state)


def _gated_delta_kernel_impl(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
    *,
    allow_packed: bool,
) -> Tuple[mx.array, mx.array]:
    (B, T, Hk, Dk) = k.shape
    (Hv, Dv) = v.shape[2:]
    input_type = q.dtype
    state_type = state.dtype
    widen = _readout_needs_widening(input_type, state_type)
    readout_type = mx.float32 if widen else input_type
    packed_eligible = (
        mask is None
        and g.ndim == 3
        and (Dk == 128)
        and (Dv % 8 == 0)
        and (g.dtype == mx.float32)
        and (state.dtype == mx.float32)
    )
    if packed_eligible and allow_packed and _ENABLE_GDN_PACKED:
        kernel = _gated_delta_kernel_packed
        inputs = [q, k, v, g, beta, state, T]
        grid = (32, Dv // 8, B * Hv)
        threadgroup = (32, 2, 1)
    elif g.ndim == 4:
        kernel = _gated_delta_kernel_vec
        inputs = [q, k, v, g, beta, state, T]
        if mask is not None:
            kernel = _gated_delta_kernel_vec_masked
            inputs.append(mask)
        grid = (32, Dv, B * Hv)
        threadgroup = (32, 4, 1)
    else:
        kernel = _gated_delta_kernel
        inputs = [q, k, v, g, beta, state, T]
        if mask is not None:
            kernel = _gated_delta_kernel_masked
            inputs.append(mask)
        grid = (32, Dv, B * Hv)
        threadgroup = (32, 4, 1)
    (y, new_state) = kernel(
        inputs=inputs,
        template=[
            ("InT", readout_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[readout_type, state_type],
    )
    if widen:
        y = _cast_readout(y, input_type, state_type)
    return (y, new_state)


def gated_delta_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    return _gated_delta_kernel_impl(q, k, v, g, beta, state, mask, allow_packed=True)


def gated_delta_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Ops-based reference implementation for prompt prefill (sequential loop).
    Supports both scalar and vectorized gating.

    Shapes:
      - q, k: [B, T, Hk, Dk]
      - v: [B, T, Hv, Dv]
      - g: [B, T, Hv] (scalar) or [B, T, Hv, Dk] (vectorized)
      - beta: [B, T, Hv]
      - state: [B, Hv, Dv, Dk]
    Returns:
      - y: [B, T, Hv, Dv]
      - state: [B, Hv, Dv, Dk]
    """
    (B, T, Hk, Dk) = q.shape
    (Hv, Dv) = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    if (repeat_factor := (Hv // Hk)) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)
    ys = []
    for t in range(T):
        (y, state) = _gated_delta_step_ops(
            q[:, t],
            k[:, t],
            v[:, t],
            g[:, t],
            beta[:, t],
            state,
            None if mask is None else mask[:, t],
        )
        ys.append(y)
    y = mx.stack(ys, axis=1)
    return (y, state)


def normalize_gdn_qk(q: mx.array, k: mx.array) -> Tuple[mx.array, mx.array]:
    """L2-normalize GDN q/k and fold in the delta-rule query scale.

    Matches the reference FLA l2norm, ``x * rsqrt(sum(x^2) + 1e-6)``, followed
    by the delta rule's ``scale = head_dim ** -0.5`` applied to the query only.

    ``mx.fast.rms_norm(x, None, eps)`` computes ``x / sqrt(mean(x^2) + eps)``,
    i.e. it adds eps to the MEAN of squares, so the equivalent epsilon is
    ``1e-6 / head_dim``. Passing ``1e-6`` straight through inflates the
    effective epsilon on the sum by head_dim (128x at head_dim=128), which
    systematically shrinks q/k whenever their norm is small.
    """
    inv_scale = k.shape[-1] ** (-0.5)
    eps = 1e-06 * inv_scale**2
    q = inv_scale**2 * mx.fast.rms_norm(q, None, eps)
    k = inv_scale * mx.fast.rms_norm(k, None, eps)
    return (q, k)


def gated_delta_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    use_kernel: bool = True,
    lower_bound: float | None = None,
    beta_input_dtype: bool = False,
) -> Tuple[mx.array, mx.array]:
    beta = gate_sigmoid(b) if beta_input_dtype else gate_sigmoid(b.astype(mx.float32))
    if lower_bound is None:
        g = compute_g(A_log, a, dt_bias)
    else:
        g = compute_lower_bound_g(A_log, a, dt_bias, lower_bound)
    if state is None:
        (B, _, Hk, Dk) = q.shape
        (Hv, Dv) = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    if not use_kernel or mx.default_device() != mx.gpu or (not mx.metal.is_available()):
        return gated_delta_ops(q, k, v, g, beta, state, mask)
    if _can_use_core_gated_delta(q, k, v, g, state, mask):
        return _core_gated_delta_update(
            q,
            k,
            v,
            g,
            beta,
            initial_state=state,
            stream=mx.gpu,
            chunk_size=_CORE_GDN_CHUNK_SIZE,
        )
    return gated_delta_kernel(q, k, v, g, beta, state, mask)
