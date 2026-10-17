"""Geometry-locked Nemotron Mamba Q8 output projection plus residual add.

The projection kernel structure is derived from TensorFold's row QMV.  This
candidate additionally owns the following BF16 residual add, eliminating the
4096-element projection materialization at the Mamba block boundary.  It is
adapter-selected and admits only the exact Super B=1/S=1 decode geometry.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx import nn

INPUT_WIDTH = 8192
OUTPUT_WIDTH = 4096
GROUP_SIZE = 64
ROWS_PER_GROUP = 4


@dataclass(frozen=True)
class MambaOutprojResidualAdmission:
    accepted: bool
    reason: str


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(getattr(value, "shape", ()))


def admit_fused_mamba_outproj_residual_b1(
    *, y, residual, projection, training: bool
) -> MambaOutprojResidualAdmission:
    if training:
        return MambaOutprojResidualAdmission(False, "training")
    if _shape(y) != (1, 1, INPUT_WIDTH):
        return MambaOutprojResidualAdmission(False, f"input shape {_shape(y)}")
    if _shape(residual) != (1, 1, OUTPUT_WIDTH):
        return MambaOutprojResidualAdmission(
            False, f"residual shape {_shape(residual)}"
        )
    if not isinstance(projection, nn.QuantizedLinear):
        return MambaOutprojResidualAdmission(False, "requires QuantizedLinear")
    if int(projection.bits) != 8 or int(projection.group_size) != GROUP_SIZE:
        return MambaOutprojResidualAdmission(False, "requires affine q8 group64")
    if _shape(projection.weight) != (OUTPUT_WIDTH, INPUT_WIDTH // 4):
        return MambaOutprojResidualAdmission(
            False, f"weight shape {_shape(projection.weight)}"
        )
    groups = INPUT_WIDTH // GROUP_SIZE
    if _shape(projection.scales) != (OUTPUT_WIDTH, groups):
        return MambaOutprojResidualAdmission(
            False, f"scales shape {_shape(projection.scales)}"
        )
    if _shape(projection.biases) != (OUTPUT_WIDTH, groups):
        return MambaOutprojResidualAdmission(
            False, f"biases shape {_shape(projection.biases)}"
        )
    if getattr(projection, "bias", None) is not None:
        return MambaOutprojResidualAdmission(False, "linear bias")
    if any(
        value.dtype != mx.bfloat16
        for value in (y, residual, projection.scales, projection.biases)
    ):
        return MambaOutprojResidualAdmission(False, "requires bf16 operands")
    return MambaOutprojResidualAdmission(True, "eligible")


_HEADER = r"""
inline float load16_q8(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(d)));
    xt[i] = float(a); xt[i + 1] = float(b);
    xt[i + 2] = float(c); xt[i + 3] = float(d);
  }
  return sum;
}
inline float qdot16_q8(const device uint8_t* w, const thread float* xt,
                       float scale, float bias, float sum) {
  float accum = 0.0f;
  for (int i = 0; i < 16; i++) accum += xt[i] * float(w[i]);
  return scale * accum + sum * bias;
}
"""

_SOURCE = r"""
  const uint lane = thread_index_in_simdgroup;
  const int row0 = int(threadgroup_position_in_grid.y) * RPS;
  constexpr int KB = K;
  constexpr int KG = K / GS;
  const device uint8_t* w = (const device uint8_t*)W + size_t(row0) * KB + lane * 16;
  const device bfloat* sc = S + size_t(row0) * KG + (lane * 16) / GS;
  const device bfloat* bi = B + size_t(row0) * KG + (lane * 16) / GS;
  const device bfloat* x = X + lane * 16;
  float acc[RPS];
  for (int j = 0; j < RPS; j++) acc[j] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += 512) {
    float xt[16];
    const float sum = load16_q8(x, xt);
    for (int j = 0; j < RPS; j++)
      acc[j] += qdot16_q8(w + j * KB, xt, float(sc[j * KG]),
                         float(bi[j * KG]), sum);
    w += 512; sc += 512 / GS; bi += 512 / GS; x += 512;
  }
  for (int j = 0; j < RPS; j++) {
    const float value = simd_sum(acc[j]);
    if (lane == 0) {
      // Stock materializes QuantizedLinear's BF16 result before the block add.
      const bfloat projected = bfloat(value);
      const bfloat summed = RES[row0 + j] + projected;
      OUT[row0 + j] = summed;
    }
  }
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        digest = hashlib.sha256((_HEADER + _SOURCE).encode()).hexdigest()[:16]
        _KERNEL = mx.fast.metal_kernel(
            name=f"mlx2_nemotron_mamba_q8_outproj_residual_b1_{digest}",
            input_names=["X", "RES", "W", "S", "B"],
            output_names=["OUT"],
            header=_HEADER,
            source=_SOURCE,
            ensure_row_contiguous=True,
        )
    return _KERNEL


def fused_mamba_outproj_residual_b1(y, residual, projection):
    """Return ``residual + projection(y)`` after exact admission."""
    (out,) = _kernel()(
        inputs=[
            y,
            residual,
            projection.weight,
            projection.scales,
            projection.biases,
        ],
        template=[
            ("K", INPUT_WIDTH),
            ("N", OUTPUT_WIDTH),
            ("RPS", ROWS_PER_GROUP),
            ("GS", GROUP_SIZE),
        ],
        grid=(32, OUTPUT_WIDTH // ROWS_PER_GROUP, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, 1, OUTPUT_WIDTH)],
        output_dtypes=[mx.bfloat16],
    )
    return out
