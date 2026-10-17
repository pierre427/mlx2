"""Row-tiled ModelOpt NVFP4 projection candidate.

The source layout is ModelOpt's row-major E2M1 bytes plus one E4M3 scale per
16 weights and a separate FP32 tensor scale.  One SIMD group owns four output
channels and reuses each decoded weight across up to four activation rows.
Rows above four are split into independent four-row blocks so the live
accumulator set stays bounded.

This is an explicitly selected component candidate.  It does not replace the
ordinary :class:`ModelOptNVFP4Linear` path.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

import mlx.core as mx

MAX_ROWS = 16
ROW_TILE = 4


@dataclass(frozen=True)
class NVFP4TileAdmission:
    accepted: bool
    reason: str


_HEADER = r"""
METAL_FUNC float mlx2_e2m1(uint8_t raw) {
  float value = 0.0f;
  switch (raw & 0x07) {
    case 1: value = 0.5f; break;
    case 2: value = 1.0f; break;
    case 3: value = 1.5f; break;
    case 4: value = 2.0f; break;
    case 5: value = 3.0f; break;
    case 6: value = 4.0f; break;
    case 7: value = 6.0f; break;
  }
  return (raw & 0x08) ? -value : value;
}

METAL_FUNC float mlx2_e4m3(uint8_t raw) {
  const int exponent = int((raw >> 3) & 0x0f);
  const int mantissa = int(raw & 0x07);
  float value = exponent == 0
      ? float(mantissa) * 0.001953125f
      : ldexp(1.0f + float(mantissa) * 0.125f, exponent - 7);
  return (raw & 0x80) ? -value : value;
}
"""


_SOURCE = r"""
  const uint lid = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int output_blocks = (N_SIZE + 7) / 8;
  const int packed_block = int(threadgroup_position_in_grid.z);
  const int row_block = packed_block / output_blocks;
  const int output_block = packed_block - row_block * output_blocks;
  const int row0 = row_block * ROW_TILE;
  const int out0 = output_block * 8 + int(sg) * 4;
  const device uint8_t* wb = (const device uint8_t*)w;
  // Keep each row's reduction recurrence scalar and ordered like the native
  // one-row qmv.  Without volatile, Metal vectorizes adjacent row
  // accumulators and changes a handful of BF16 boundary roundings at R>1.
  thread volatile float accum[ROW_TILE][4];
  for (int rr = 0; rr < ROW_TILE; ++rr)
    for (int oo = 0; oo < 4; ++oo)
      accum[rr][oo] = 0.0f;

  for (int k = int(lid); k < K_SIZE; k += 32) {
    float inputs[ROW_TILE];
    for (int rr = 0; rr < ROW_TILE; ++rr) {
      const int row = row0 + rr;
      inputs[rr] = row < ROWS ? float(x[row * K_SIZE + k]) : 0.0f;
    }
    for (int oo = 0; oo < 4; ++oo) {
      const int out = out0 + oo;
      if (out >= N_SIZE)
        continue;
      const uint8_t packed = wb[out * (K_SIZE / 2) + k / 2];
      const uint8_t q = (k & 1) ? (packed >> 4) : (packed & 0x0f);
      const uint8_t sb = scales[out * (K_SIZE / 16) + k / 16];
      const float weight = mlx2_e4m3(sb) * mlx2_e2m1(q);
      for (int rr = 0; rr < ROW_TILE; ++rr)
        accum[rr][oo] += inputs[rr] * weight;
    }
  }

  for (int rr = 0; rr < ROW_TILE; ++rr) {
    const int row = row0 + rr;
    if (row >= ROWS)
      continue;
    for (int oo = 0; oo < 4; ++oo) {
      const int out = out0 + oo;
      if (out >= N_SIZE)
        continue;
      float value = simd_sum(float(accum[rr][oo]));
      if (lid == 0) {
        value *= float(global_scale);
        if (HAS_BIAS)
          value += float(bias[out]);
        y[row * N_SIZE + out] = static_cast<T>(value);
      }
    }
  }
"""


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(getattr(value, "shape", ()))


def admit(linear, x, *, require_gpu: bool = True) -> NVFP4TileAdmission:
    if require_gpu and (mx.default_device() != mx.gpu or not mx.metal.is_available()):
        return NVFP4TileAdmission(False, "device")
    if x.ndim < 2:
        return NVFP4TileAdmission(False, "rank")
    rows = int(x.size // x.shape[-1])
    if not 1 <= rows <= MAX_ROWS or x.size != rows * x.shape[-1]:
        return NVFP4TileAdmission(False, "rows")
    if x.dtype not in (mx.bfloat16, mx.float16):
        return NVFP4TileAdmission(False, "activation_dtype")
    if int(x.shape[-1]) % 32:
        return NVFP4TileAdmission(False, "input_width")
    if linear.weight.dtype != mx.uint32 or linear.weight.ndim != 2:
        return NVFP4TileAdmission(False, "weight_layout")
    n, packed = map(int, linear.weight.shape)
    k = int(x.shape[-1])
    if packed * 8 != k or _shape(linear.scales) != (n, k // 16):
        return NVFP4TileAdmission(False, "weight_layout")
    if linear.scales.dtype != mx.uint8 or linear.global_scale.dtype != mx.float32:
        return NVFP4TileAdmission(False, "scale_layout")
    if "bias" in linear and (
        linear.bias.dtype != mx.bfloat16 or _shape(linear.bias) != (n,)
    ):
        return NVFP4TileAdmission(False, "bias_layout")
    return NVFP4TileAdmission(True, "eligible")


@cache
def _kernel(rows: int, row_tile: int, has_bias: bool):
    return mx.fast.metal_kernel(
        name=(f"mlx2_modelopt_nvfp4_tile_r{rows}_t{row_tile}_b{int(has_bias)}"),
        input_names=["x", "w", "scales", "global_scale", "bias"],
        output_names=["y"],
        header=_HEADER,
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


def _project(linear, x, *, row_tile: int):
    admission = admit(linear, x)
    if not admission.accepted:
        raise ValueError(f"ModelOpt NVFP4 row tile refused: {admission.reason}")
    lead = x.shape[:-1]
    rows = int(x.size // x.shape[-1])
    k = int(x.shape[-1])
    n = int(linear.weight.shape[0])
    has_bias = "bias" in linear
    bias = linear.bias if has_bias else mx.zeros((n,), dtype=mx.bfloat16)
    row_blocks = (rows + row_tile - 1) // row_tile
    output_blocks = (n + 7) // 8
    result = _kernel(rows, row_tile, has_bias)(
        inputs=[
            x.reshape(rows, k),
            linear.weight,
            linear.scales,
            linear.global_scale,
            bias,
        ],
        template=[
            ("T", x.dtype),
            ("K_SIZE", k),
            ("N_SIZE", n),
            ("ROWS", rows),
            ("ROW_TILE", row_tile),
            ("HAS_BIAS", int(has_bias)),
        ],
        grid=(32, 2, output_blocks * row_blocks),
        threadgroup=(32, 2, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[x.dtype],
    )[0]
    return result.reshape(*lead, n)


def project(linear, x):
    """Project packed rows with four-row weight reuse.

    This candidate is faster than serial B1 at larger row counts, but remains
    a separately measured arithmetic mode because Metal may vectorize the
    adjacent row accumulators.
    """

    return _project(linear, x, row_tile=ROW_TILE)


def project_row_exact(linear, x):
    """Launch native-one-row arithmetic for 1..16 rows in one grid.

    Each threadgroup owns one row, so the compiler sees the same scalar
    accumulator structure as the bit-exact one-row kernel while launch and
    scheduling overhead can still be amortized across the grid.
    """

    return _project(linear, x, row_tile=1)
