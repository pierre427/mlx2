"""Opt-in TensorFold Flash row matvec for affine q4/q8 group-64 artifacts.

TensorFold bb4b4a35 (MIT) is the source for the row kernel structure and
quantized dot; see provenance/flashnext-tensorfold-qmv.json. PLE and expert
tables are intentionally outside this installer.
"""

from __future__ import annotations

import hashlib
from collections import Counter

import mlx.core as mx
from mlx import nn


_HEADER = r"""
inline float load16(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(d)));
    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f;
    xt[i + 2] = float(c) / 256.0f; xt[i + 3] = float(d) / 4096.0f;
  }
  return sum;
}
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
inline float qdot16_q4(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0.0f;
  for (int i = 0; i < 4; i++)
    accum += xt[4 * i] * float(ws[i] & 0x000f) + xt[4 * i + 1] * float(ws[i] & 0x00f0) +
             xt[4 * i + 2] * float(ws[i] & 0x0f00) + xt[4 * i + 3] * float(ws[i] & 0xf000);
  return scale * accum + sum * bias;
}
inline float qdot16_q8(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
  float accum = 0.0f;
  for (int i = 0; i < 16; i++) accum += xt[i] * float(w[i]);
  return scale * accum + sum * bias;
}
"""

_SOURCE_Q4 = r"""
  const uint lane = thread_index_in_simdgroup;
  const int r = int(simdgroup_index_in_threadgroup);
  const int row0 = int(threadgroup_position_in_grid.y) * RPS;
  constexpr int KB = K / 2;
  constexpr int KG = K / GS;
  const device uint8_t* w = (const device uint8_t*)W + size_t(row0) * KB + lane * 8;
  const device bfloat* sc = S + size_t(row0) * KG + (lane * 16) / GS;
  const device bfloat* bi = B + size_t(row0) * KG + (lane * 16) / GS;
  const device bfloat* x = X + r * K + lane * 16;
  float acc[RPS];
  for (int j = 0; j < RPS; j++) acc[j] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += 512) {
    float xt[16];
    const float sum = load16(x, xt);
    for (int j = 0; j < RPS; j++)
      acc[j] += qdot16_q4(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum);
    w += 256; sc += 512 / GS; bi += 512 / GS; x += 512;
  }
  for (int j = 0; j < RPS; j++) {
    const float v = simd_sum(acc[j]);
    if (lane == 0) OUT[r * N + row0 + j] = bfloat(v);
  }
"""

_SOURCE_Q8 = r"""
  const uint lane = thread_index_in_simdgroup;
  const int r = int(simdgroup_index_in_threadgroup);
  const int row0 = int(threadgroup_position_in_grid.y) * RPS;
  constexpr int KB = K;
  constexpr int KG = K / GS;
  const device uint8_t* w = (const device uint8_t*)W + size_t(row0) * KB + lane * 16;
  const device bfloat* sc = S + size_t(row0) * KG + (lane * 16) / GS;
  const device bfloat* bi = B + size_t(row0) * KG + (lane * 16) / GS;
  const device bfloat* x = X + r * K + lane * 16;
  float acc[RPS];
  for (int j = 0; j < RPS; j++) acc[j] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += 512) {
    float xt[16];
    const float sum = load16_q8(x, xt);
    for (int j = 0; j < RPS; j++)
      acc[j] += qdot16_q8(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum);
    w += 512; sc += 512 / GS; bi += 512 / GS; x += 512;
  }
  for (int j = 0; j < RPS; j++) {
    const float v = simd_sum(acc[j]);
    if (lane == 0) OUT[r * N + row0 + j] = bfloat(v);
  }
"""

_KERNELS = {}
_COUNTS = Counter()


def _kernel(bits):
    source = _SOURCE_Q4 if bits == 4 else _SOURCE_Q8
    if bits not in _KERNELS:
        digest = hashlib.sha256((_HEADER + source).encode()).hexdigest()[:16]
        _KERNELS[bits] = mx.fast.metal_kernel(
            name=f"mlx2_flash_tensorfold_qmv_q{bits}_{digest}",
            input_names=["X", "W", "S", "B"], output_names=["OUT"],
            source=source, header=_HEADER,
        )
    return _KERNELS[bits]


def eligible(module) -> bool:
    if not isinstance(module, nn.QuantizedLinear):
        return False
    bits = int(module.bits)
    if bits not in (4, 8) or int(module.group_size) != 64:
        return False
    n = int(module.weight.shape[0])
    k = int(module.weight.shape[1]) * (32 // bits)
    return k % 512 == 0 and n % 4 == 0 and module.scales.dtype == mx.bfloat16


def qmv_rows(x, module):
    """Compute 1..17 rows with the same arithmetic for every row width."""
    shape = x.shape
    rows = x.size // shape[-1]
    if x.dtype != mx.bfloat16 or not 1 <= rows <= 17 or not eligible(module):
        raise ValueError("TensorFold Flash row matvec requires bf16, affine q4/q8-g64 and 1..17 rows")
    n, k = int(module.weight.shape[0]), int(shape[-1])
    bits = int(module.bits)
    if k != int(module.weight.shape[1]) * (32 // bits):
        raise ValueError("TensorFold Flash row matvec activation width mismatch")
    out = _kernel(bits)(
        inputs=[x.reshape(rows, k), module.weight, module.scales, module.biases],
        template=[("K", k), ("N", n), ("RPS", 4), ("GS", 64)],
        grid=(32 * rows, n // 4, 1), threadgroup=(32 * rows, 1, 1),
        output_shapes=[(rows, n)], output_dtypes=[mx.bfloat16],
    )[0].reshape(*shape[:-1], n)
    if "bias" in module:
        out = out + module.bias
    return out


class TensorFoldQMVLinear(nn.QuantizedLinear):
    def __call__(self, x):
        rows = x.size // x.shape[-1]
        if (
            getattr(self, "_tensorfold_qmv_enabled", True)
            and 1 <= rows <= 17
            and x.dtype == mx.bfloat16
        ):
            _COUNTS["kernel_calls"] += 1
            _COUNTS["kernel_rows"] += rows
            return qmv_rows(x, self)
        _COUNTS["stock_calls"] += 1
        return nn.QuantizedLinear.__call__(self, x)


def install(model) -> dict:
    """Install on affine q4/q8-g64 dense projections without copying weights."""
    installed = []
    for name, module in model.named_modules():
        if type(module) is not nn.QuantizedLinear:
            continue
        if any(part in name.split(".") for part in (
            "switch_mlp", "shared_expert", "ple", "lm_head", "mtp_draft_head"
        )):
            continue
        if eligible(module):
            module.__class__ = TensorFoldQMVLinear
            object.__setattr__(module, "_tensorfold_qmv_enabled", True)
            installed.append(name)
    if not installed:
        raise ValueError("TensorFold Flash qmv found no eligible affine q4/q8-g64 projections")
    return {"source_revision": "bb4b4a35863af562fc4ccb2586300d8f94b5d6de",
            "kernel": "qmv_rows_q4q8g64", "installed": len(installed),
            "names_sha256": hashlib.sha256("\n".join(sorted(installed)).encode()).hexdigest()}


def counters() -> dict:
    return dict(_COUNTS)
