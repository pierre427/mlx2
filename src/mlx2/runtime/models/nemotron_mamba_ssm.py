"""Geometry-locked Nemotron-H short SSM scan with in-kernel dt preparation.

This candidate preserves the ordinary short-scan recurrence and only folds the
FP32 ``dt + bias -> softplus -> clamp`` graph into the same Metal dispatch.
It is selected explicitly by the Super adapter and refuses every geometry
outside B=1, S=1..16, H=128, Dh=64, G=8, Ds=128.  S=1 is the ordinary decode
case; S=2..16 covers bounded verification tiles and accepted-prefix replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx import nn

from .served_exp import ServedExpGate, adapted_helper_header, kernel_dtype_refusal


@dataclass(frozen=True)
class MambaSsmAdmission:
    accepted: bool
    reason: str


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(getattr(value, "shape", ()))


def admit_fused_mamba_ssm_short(
    *, x, B, C, dt, A_log, D, dt_bias, state, mask, lengths
) -> MambaSsmAdmission:
    shape = _shape(x)
    if len(shape) != 4 or shape[0] != 1 or not 1 <= shape[1] <= 16:
        return MambaSsmAdmission(False, f"x shape {shape}")
    if shape[2:] != (128, 64):
        return MambaSsmAdmission(False, f"head geometry {shape[2:]}")
    steps = shape[1]
    if _shape(B) != (1, steps, 8, 128) or _shape(C) != _shape(B):
        return MambaSsmAdmission(False, "B/C geometry")
    if _shape(dt) != (1, steps, 128):
        return MambaSsmAdmission(False, f"dt shape {_shape(dt)}")
    if _shape(A_log) != (128,) or _shape(D) != (128,) or _shape(dt_bias) != (128,):
        return MambaSsmAdmission(False, "per-head geometry")
    if _shape(state) != (1, 128, 64, 128):
        return MambaSsmAdmission(False, f"state shape {_shape(state)}")
    if mask is not None or lengths is not None:
        return MambaSsmAdmission(False, "masked or padded input")
    if any(value.dtype != mx.bfloat16 for value in (x, B, C, D, dt)):
        return MambaSsmAdmission(False, "requires bf16 activations")
    if A_log.dtype not in (mx.bfloat16, mx.float32):
        return MambaSsmAdmission(False, "requires bf16/fp32 A_log")
    if dt_bias.dtype not in (mx.bfloat16, mx.float32):
        return MambaSsmAdmission(False, "requires bf16/fp32 dt_bias")
    if state.dtype != mx.float32:
        return MambaSsmAdmission(False, "requires fp32 recurrent state")
    return MambaSsmAdmission(True, "eligible")


def admit_fused_mamba_ssm_gate_short(
    *, x, B, C, dt, A_log, D, dt_bias, state, gate, mask, lengths
) -> MambaSsmAdmission:
    admission = admit_fused_mamba_ssm_short(
        x=x,
        B=B,
        C=C,
        dt=dt,
        A_log=A_log,
        D=D,
        dt_bias=dt_bias,
        state=state,
        mask=mask,
        lengths=lengths,
    )
    if not admission.accepted:
        return admission
    if _shape(gate) != (1, x.shape[1], 128 * 64):
        return MambaSsmAdmission(False, f"gate shape {_shape(gate)}")
    if gate.dtype != mx.bfloat16:
        return MambaSsmAdmission(False, "requires bf16 gate")
    return MambaSsmAdmission(True, "eligible")


_HEADER = r"""
template <typename U>
inline U mlx_log1p_fast(U x) {
  float xf = float(x);
  float xp1 = 1.0f + xf;
  float out = xp1 == 1.0f ? xf : xf * (metal::log(xp1) / (xp1 - 1.0f));
  return static_cast<U>(out);
}

template <typename U>
inline U mlx_softplus_fast(U x) {
  if (metal::isnan(x))
    return metal::numeric_limits<U>::quiet_NaN();
  constexpr U inf = metal::numeric_limits<U>::infinity();
  U zero = static_cast<U>(0);
  U hi = metal::max(x, zero);
  U lo = metal::min(x, zero);
  return (lo == -inf || hi == inf)
      ? hi
      : (hi + mlx_log1p_fast(static_cast<U>(metal::exp(lo - hi))));
}

template <typename U>
inline U mlx_sigmoid_fast(U x) {
  U e = static_cast<U>(metal::exp(metal::abs(x)));
  U v = static_cast<U>(1) / (static_cast<U>(1) + e);
  return (x < 0) ? v : (static_cast<U>(1) - v);
}
"""


_SOURCE = r"""
  auto n = thread_position_in_grid.z;
  auto h_idx = n % H;
  auto b_idx = n / H;
  auto grp = h_idx / G;
  constexpr int n_per_t = Ds / 32;
  constexpr int HB = H / G;

  auto i_state = state_in + n * Dh * Ds;
  auto o_state = state_out + n * Dh * Ds;
  auto ds_idx = thread_position_in_threadgroup.x;
  auto d_idx = thread_position_in_grid.y;
  auto A = -fast::exp(static_cast<float>(A_log[h_idx]));

  float st[n_per_t];
  for (int i = 0; i < n_per_t; ++i)
    st[i] = i_state[d_idx * Ds + n_per_t * ds_idx + i];

  for (int s = 0; s < S; ++s) {
    auto x_ = static_cast<float>(
        X[((b_idx * S + s) * H + h_idx) * Dh + d_idx]);
    float raw_dt = static_cast<float>(dt[(b_idx * S + s) * H + h_idx]);
    float dt_ = mlx_softplus_fast<float>(raw_dt + float(dt_bias[h_idx]));
    if constexpr (CLAMP_DT)
      dt_ = metal::clamp(dt_, 0.001f, 100.0f);
    auto dA = fast::exp(A * dt_);
    auto B_ = B + ((b_idx * S + s) * HB + grp) * Ds;
    auto C_ = C + ((b_idx * S + s) * HB + grp) * Ds;

    float acc = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * ds_idx + i;
      auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
      st[i] = dA * st[i] + dB_by_x;
      acc += st[i] * C_[s_idx];
    }
    acc = simd_sum(acc);
    if (thread_index_in_simdgroup == 0) {
      auto out_idx = ((b_idx * S + s) * H + h_idx) * Dh + d_idx;
      auto y = static_cast<T>(acc + x_ * D[h_idx]);
      if constexpr (APPLY_GATE) {
        // Preserve the served BF16 boundaries: materialize SSM output, then
        // nn.silu(gate), then the pointwise product consumed by group RMS.
        auto g = gate[out_idx];
        auto silu = static_cast<T>(g * mlx_sigmoid_fast<T>(g));
        out[out_idx] = static_cast<T>(silu * y);
      } else {
        out[out_idx] = y;
      }
    }
  }

  for (int i = 0; i < n_per_t; ++i)
    o_state[d_idx * Ds + n_per_t * ds_idx + i] = st[i];
"""


_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="mlx2_nemotron_mamba_ssm_short_dt" + _served_header()[1],
            input_names=[
                "X",
                "A_log",
                "B",
                "C",
                "D",
                "dt",
                "dt_bias",
                "state_in",
                "gate",
            ],
            output_names=["out", "state_out"],
            header=_served_header()[0],
            source=_SOURCE,
            ensure_row_contiguous=True,
        )
    return _KERNEL


def fused_mamba_ssm_short(x, A_log, B, C, D, dt, dt_bias, state, time_step_limit):
    limits = tuple(float(value) for value in time_step_limit)
    if limits == (0.001, 100.0):
        clamp_dt = True
    elif limits == (0.0, float("inf")):
        clamp_dt = False
    else:
        raise ValueError(
            "fused Nemotron short SSM requires served time_step_limit "
            "(0, inf) or (0.001, 100.0)"
        )
    batch, steps, heads, head_dim = x.shape
    groups, state_dim = B.shape[-2:]
    out, state_out = _kernel()(
        inputs=[x, A_log, B, C, D, dt, dt_bias, state, x],
        template=[
            ("T", x.dtype),
            ("Dh", head_dim),
            ("Ds", state_dim),
            ("H", heads),
            ("G", heads // groups),
            ("S", steps),
            ("CLAMP_DT", clamp_dt),
            ("APPLY_GATE", False),
        ],
        grid=(32, head_dim, heads * batch),
        threadgroup=(32, 8, 1),
        output_shapes=[x.shape, state.shape],
        output_dtypes=[x.dtype, state.dtype],
    )
    return out, state_out


_SILU_GATE = ServedExpGate(
    "nemotron_mamba_ssm_gate_silu",
    served_name="compiled nn.silu",
    served=nn.silu,
    header=_HEADER,
    body="    y = x * mlx_sigmoid_fast<T>(x);",
)


def fused_mamba_ssm_gate_refusal(dtype=mx.bfloat16) -> str | None:
    return kernel_dtype_refusal(_SILU_GATE, dtype)


def _served_header() -> tuple[str, str]:
    """Kernel header with the SiLU sigmoid spelled like the served op."""
    return adapted_helper_header(_SILU_GATE, _HEADER, "mlx_sigmoid_fast")


def fused_mamba_ssm_gate_short(
    x, A_log, B, C, D, dt, dt_bias, state, gate, time_step_limit
):
    limits = tuple(float(value) for value in time_step_limit)
    if limits == (0.001, 100.0):
        clamp_dt = True
    elif limits == (0.0, float("inf")):
        clamp_dt = False
    else:
        raise ValueError(
            "fused Nemotron short SSM+gate requires served time_step_limit "
            "(0, inf) or (0.001, 100.0)"
        )
    batch, steps, heads, head_dim = x.shape
    groups, state_dim = B.shape[-2:]
    out, state_out = _kernel()(
        inputs=[x, A_log, B, C, D, dt, dt_bias, state, gate],
        template=[
            ("T", x.dtype),
            ("Dh", head_dim),
            ("Ds", state_dim),
            ("H", heads),
            ("G", heads // groups),
            ("S", steps),
            ("CLAMP_DT", clamp_dt),
            ("APPLY_GATE", True),
        ],
        grid=(32, head_dim, heads * batch),
        threadgroup=(32, 8, 1),
        output_shapes=[x.shape, state.shape],
        output_dtypes=[x.dtype, state.dtype],
    )
    return out, state_out
