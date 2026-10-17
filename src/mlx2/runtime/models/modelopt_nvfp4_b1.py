"""Geometry-locked ModelOpt NVFP4 B1 projection component candidate.

This is not installed by default.  It is first exercised by the owned GPU
component probe named in ``provenance/nemotron35-nvfp4-b1.json``.
"""
from __future__ import annotations

from functools import cache

import mlx.core as mx

_HEADER = r"""
#include <metal_simdgroup>
using namespace metal;

METAL_FUNC float mlx2_e2m1(uint8_t raw) {
  const uint8_t magnitude = raw & 0x07;
  float value = 0.0f;
  switch (magnitude) {
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
    const int out0 = int(threadgroup_position_in_grid.z) * 8 + int(sg) * 4;
    const device uint8_t* wb = (const device uint8_t*)w;
    float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = int(lid); k < K_SIZE; k += 32) {
      const float input = x[k];
      for (int r = 0; r < 4; r++) {
        const int out = out0 + r;
        if (out >= N_SIZE) {
          continue;
        }
        uint8_t packed = wb[out * (K_SIZE / 2) + k / 2];
        uint8_t q = (k & 1) ? (packed >> 4) : (packed & 0x0f);
        uint8_t sb = scales[out * (K_SIZE / 16) + k / 16];
        float value = mlx2_e2m1(q);
        float scale = mlx2_e4m3(sb);
        accum[r] += input * (scale * value);
      }
    }
    for (int r = 0; r < 4; r++) {
      const int out = out0 + r;
      if (out >= N_SIZE) {
        continue;
      }
      float value = simd_sum(accum[r]);
      if (lid == 0) {
        value *= float(global_scale);
        if (HAS_BIAS) {
          value += float(bias[out]);
        }
        y[out] = static_cast<T>(value);
      }
    }
"""


@cache
def _kernel(has_bias: bool):
    return mx.fast.metal_kernel(
        name=f"mlx2_modelopt_nvfp4_b1_{int(has_bias)}",
        input_names=["x", "w", "scales", "global_scale", "bias"],
        output_names=["y"],
        header=_HEADER,
        source=_SOURCE,
    )


def refusal(linear, x) -> str | None:
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return "device"
    if x.ndim < 2 or x.size != x.shape[-1]:
        return "not_b1"
    if x.dtype not in (mx.bfloat16, mx.float16):
        return "activation_dtype"
    if x.shape[-1] % 32:
        return "input_width"
    if linear.weight.dtype != mx.uint32 or linear.weight.ndim != 2:
        return "weight_layout"
    n, packed = map(int, linear.weight.shape)
    k = int(x.shape[-1])
    if packed * 8 != k or tuple(linear.scales.shape) != (n, k // 16):
        return "weight_layout"
    if linear.scales.dtype != mx.uint8 or linear.global_scale.dtype != mx.float32:
        return "scale_layout"
    if "bias" in linear and (
        linear.bias.dtype != mx.bfloat16 or tuple(linear.bias.shape) != (n,)
    ):
        return "bias_layout"
    return None


def project(linear, x):
    """Run the fused B1 component or fail closed with its refusal reason."""
    reason = refusal(linear, x)
    if reason is not None:
        raise ValueError(f"ModelOpt NVFP4 B1 fused projection refused: {reason}")
    lead = x.shape[:-1]
    k = int(x.shape[-1])
    n = int(linear.weight.shape[0])
    has_bias = "bias" in linear
    bias = linear.bias if has_bias else mx.zeros((n,), dtype=mx.bfloat16)
    result = _kernel(has_bias)(
        inputs=[
            x.reshape(k),
            linear.weight,
            linear.scales,
            linear.global_scale,
            bias,
        ],
        template=[
            ("T", x.dtype),
            ("K_SIZE", k),
            ("N_SIZE", n),
            ("HAS_BIAS", int(has_bias)),
        ],
        grid=(32, 2, (n + 7) // 8),
        threadgroup=(32, 2, 1),
        output_shapes=[(n,)],
        output_dtypes=[x.dtype],
    )[0]
    return result.reshape(*lead, n)
