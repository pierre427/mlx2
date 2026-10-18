"""Geometry-locked Nemotron-H B=1/S=1..8 gate-before-group-RMS candidate.

The served graph materializes ``nn.silu(gate) * y`` in BF16, performs eight
independent 1024-element RMS normalizations, materializes that BF16 result,
then multiplies by the BF16 learned gain.  This kernel preserves those
boundaries and is selected only through the Super adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx import nn

from .served_exp import ServedExpGate, adapted_helper_header, kernel_dtype_refusal

WIDTH = 8192
GROUP_SIZE = 1024
GROUPS = WIDTH // GROUP_SIZE
EPS = 1e-5
THREADS = 256
ELEMS_PER_THREAD = GROUP_SIZE // THREADS


@dataclass(frozen=True)
class MambaNormAdmission:
    accepted: bool
    reason: str


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(getattr(value, "shape", ()))


def admit_fused_mamba_norm_short(
    *, y, gate, weight, eps: float, group_size: int, training: bool
) -> MambaNormAdmission:
    if training:
        return MambaNormAdmission(False, "training")
    shape = _shape(y)
    if len(shape) != 3 or shape[0] != 1 or not 1 <= shape[1] <= 8 or shape[2] != WIDTH:
        return MambaNormAdmission(False, f"y shape {_shape(y)}")
    if _shape(gate) != shape:
        return MambaNormAdmission(False, f"gate shape {_shape(gate)}")
    if _shape(weight) != (WIDTH,):
        return MambaNormAdmission(False, f"weight shape {_shape(weight)}")
    if group_size != GROUP_SIZE:
        return MambaNormAdmission(False, f"group size {group_size}")
    if eps != EPS:
        return MambaNormAdmission(False, f"eps {eps}")
    if any(value.dtype != mx.bfloat16 for value in (y, gate, weight)):
        return MambaNormAdmission(False, "requires bf16 operands")
    return MambaNormAdmission(True, "eligible")


def admit_fused_mamba_norm_b1(**kwargs) -> MambaNormAdmission:
    """Compatibility name retained for callers that only request S=1."""
    if _shape(kwargs["y"]) != (1, 1, WIDTH):
        return MambaNormAdmission(False, f"y shape {_shape(kwargs['y'])}")
    admission = admit_fused_mamba_norm_short(**kwargs)
    return admission


_HEADER = """
template <typename U>
inline U mlx_sigmoid_fast(U x) {
  U e = static_cast<U>(metal::exp(metal::abs(x)));
  U v = static_cast<U>(1) / (static_cast<U>(1) + e);
  return (x < 0) ? v : (static_cast<U>(1) - v);
}
"""

_SOURCE = r"""
  const uint tid = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint slab = threadgroup_position_in_grid.z;
  const uint group = slab % uint(GROUPS);
  const uint row = slab / uint(GROUPS);
  const uint local_base = group * uint(GS) + tid * uint(EPT);
  const uint base = row * uint(WIDTH) + local_base;
  threadgroup float partials[8];

  T gated[EPT];
  float ss = 0.0f;
  for (uint i = 0; i < uint(EPT); ++i) {
    const uint idx = base + i;
    const T g = gate[idx];
    // Preserve the two served BF16 boundaries: nn.silu(g), then * y.
    const T silu = g * mlx_sigmoid_fast(g);
    gated[i] = silu * y[idx];
    const float v = float(gated[i]);
    ss += v * v;
  }

  ss = simd_sum(ss);
  if (lane == 0) partials[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float part = (lane < 8) ? partials[lane] : 0.0f;
    part = simd_sum(part);
    if (lane == 0) partials[0] = part;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float mean = partials[0] / float(GS);
  const float inv = metal::precise::rsqrt(mean + 1.0e-5f);
  for (uint i = 0; i < uint(EPT); ++i) {
    const uint idx = base + i;
    // fast.rms_norm returns BF16 here; the gain multiply is a separate BF16 op.
    const T normed = T(float(gated[i]) * inv);
    out[idx] = weight[local_base + i] * normed;
  }
"""

_KERNEL = None

_SILU_GATE = ServedExpGate(
    "nemotron_mamba_gate_norm_silu",
    served_name="compiled nn.silu",
    served=nn.silu,
    header=_HEADER,
    body="    y = x * mlx_sigmoid_fast<T>(x);",
)


def fused_mamba_norm_refusal(dtype=mx.bfloat16) -> str | None:
    return kernel_dtype_refusal(_SILU_GATE, dtype)


def _served_header() -> tuple[str, str]:
    """Kernel header with the SiLU sigmoid spelled like the served op."""
    return adapted_helper_header(_SILU_GATE, _HEADER, "mlx_sigmoid_fast")


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="mlx2_nemotron_mamba_gate_group_rms_short" + _served_header()[1],
            input_names=["y", "gate", "weight"],
            output_names=["out"],
            header=_served_header()[0],
            source=_SOURCE,
            ensure_row_contiguous=True,
        )
    return _KERNEL


def fused_mamba_norm_short(y, gate, weight):
    (out,) = _kernel()(
        inputs=[y, gate, weight],
        template=[
            ("T", y.dtype),
            ("GS", GROUP_SIZE),
            ("GROUPS", GROUPS),
            ("WIDTH", WIDTH),
            ("EPT", ELEMS_PER_THREAD),
        ],
        grid=(THREADS, 1, GROUPS * y.shape[1]),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[y.shape],
        output_dtypes=[y.dtype],
    )
    return out


def fused_mamba_norm_b1(y, gate, weight):
    if y.shape[1] != 1:
        raise ValueError("fused_mamba_norm_b1 requires S=1")
    return fused_mamba_norm_short(y, gate, weight)
