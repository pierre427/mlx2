# SPDX-License-Identifier: Apache-2.0
# Adapted from IronMLX, Copyright 2026 Xin Liu.
"""Opt-in fused BF16 affine4 prefill/SwiGLU candidate, without a second weight bank.

The Metal source can be compiled offline. Import and CPU reference functions
never import MLX. Hardware parity/performance remain unqualified.
See provenance/ironmlx-tiled-prefill-20261104.json.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

ENV = "MLX2_IRONMLX_FUSED_PREFILL"
LAW = "ironmlx-fused-prefill-bf16-v1"

_BODY = r"""

  constexpr int RS = 32, NSG = 4, BM = RS * NSG, BN = 64, BK = 64;
  const int M = dims[0], NW = dims[1], K = dims[2], NB = dims[3], NO = dims[4], KW = dims[5];
  const int KG = KW / 64, S = K / BK;
  const int m0 = threadgroup_position_in_grid.x * BM;
#if !SWIGLU
  const int c0 = threadgroup_position_in_grid.y * BN;
#endif
  const uint tid = thread_index_in_threadgroup, sg = simdgroup_index_in_threadgroup;
  // Thread pairs share a column: half h of its 64 codes per step.
  const int c = int(tid) >> 1, h = int(tid) & 1;
#if SWIGLU
  const int g0 = threadgroup_position_in_grid.y * 32;
  const int n = NB + (c < 32 ? g0 + c : NW / 2 + g0 + c - 32);
#if CHECK
  const bool live = g0 + (c & 31) < NO;
#else
  constexpr bool live = true;
#endif
#else
#if CHECK
  const bool live = c0 + c < NO;
  const int n = NB + c0 + c;
#else
  constexpr bool live = true;
#if CLAMP
  // Columns past NW (a 64-padded output) recompute the last column.
  const int n = min(NB + c0 + c, NW - 1);
#else
  const int n = NB + c0 + c;
#endif
#endif
#endif
  threadgroup bfloat stage[2][BN * BK];
  // Column n's 32 codes of half h in group g: tile n / NTW, row n % NTW.
  const device uint4* wbase = (const device uint4*)W + (int64_t(n / NTW) * KG * NTW + (n % NTW)) * 2 + h;
  auto fetch = [&](int g, thread uint4& q, thread float2& sbf) {
    if (live) {
      q = wbase[int64_t(g) * NTW * 2];
      sbf = float2(float(Scale[int64_t(n) * KG + g]), float(B[int64_t(n) * KG + g]));
    } else {
      q = uint4(0);
      sbf = float2(0.0f);
    }
  };
  // MLX's dequantized values: scale * code + bias in f32, rounded once to bf16.
  auto dequant = [&](uint4 q, float2 sbf, threadgroup bfloat* dst) {
    for (int w = 0; w < 4; w++) {
      const uint word = q[w];
      vec<bfloat, 8> out;
      for (int e = 0; e < 8; e++) out[e] = bfloat(fma(float((word >> (4 * e)) & 0xFu), sbf.x, sbf.y));
      *(threadgroup vec<bfloat, 8>*)(dst + c * BK + (h * 4 + w) * 8) = out;
    }
  };
  constexpr auto desc = matmul2d_descriptor(RS, BN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<1>> op;
  const int mr = m0 + int(sg) * RS;
#if CHECK
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X, dextents<int32_t, 2>(K, M));
  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tB0(stage[0], dextents<int32_t, 2>(BK, BN));
  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tB1(stage[1], dextents<int32_t, 2>(BK, BN));
  auto P = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB0), float>();
#else
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + int64_t(mr) * K, dextents<int32_t, 2>(K, RS));
  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> sB0(stage[0], dextents<int32_t, 2>(BK, BN));
  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> sB1(stage[1], dextents<int32_t, 2>(BK, BN));
  auto tB0 = sB0.template slice<BK, BN>(0, 0);
  auto tB1 = sB1.template slice<BK, BN>(0, 0);
  auto a0 = tA.template slice<BK, RS>(0, 0);
  auto P = op.template get_destination_cooperative_tensor<decltype(a0), decltype(tB0), float>();
#endif
  for (uint16_t i = 0; i < P.get_capacity(); ++i) P[i] = 0.0f;
  uint4 q;
  float2 sbf;
  fetch(0, q, sbf);
  for (int st = 0; st < S; st++) {
    dequant(q, sbf, stage[st & 1]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (st + 1 < S) fetch(st + 1, q, sbf);
#if CHECK
    auto a = tA.slice(st * BK, mr);
#else
    auto a = tA.template slice<BK, RS>(st * BK, 0);
#endif
    if (st & 1) op.run(a, tB1, P); else op.run(a, tB0, P);
  }
#if SWIGLU
  // Every simdgroup has read its last stage; each takes a 32 x 64 quarter.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup bfloat16_t* prod = (threadgroup bfloat16_t*)&stage[0][0] + int(sg) * (RS * BN);
  for (uint16_t i = 0; i < P.get_capacity(); ++i) {
    if (!P.is_valid_element(i)) continue;
    const auto idx = P.get_multidimensional_index(i);
    prod[int(idx[1]) * BN + int(idx[0])] = bfloat16_t(P[i]);
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  const int l = int(thread_index_in_simdgroup);
  for (int r = 0; r < RS; r++) {
    const bfloat16_t gate = prod[r * BN + l], up = prod[r * BN + 32 + l];
    const bfloat16_t sig = IronSigmoid()(gate);
    const bfloat16_t silu = gate * sig;
    const bfloat16_t out = silu * up;
    const int m = mr + r;
#if CHECK
    if (m < M && g0 + l < NO) Y[int64_t(m) * NO + g0 + l] = out;
#else
    Y[int64_t(m) * NO + g0 + l] = out;
#endif
  }
#else
  for (uint16_t i = 0; i < P.get_capacity(); ++i) {
    if (!P.is_valid_element(i)) continue;
    const auto idx = P.get_multidimensional_index(i);
    const int cc = c0 + int(idx[0]), m = mr + int(idx[1]);
#if CHECK
    if (m < M && cc < NO) Y[int64_t(m) * NO + cc] = bfloat(P[i]);
#else
    Y[int64_t(m) * NO + cc] = bfloat(P[i]);
#endif
  }
#endif
"""

_HEADER = r"""
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
typedef bfloat bfloat16_t;
struct IronSigmoid {
  template <typename T> T operator()(T x) thread {
    float y = 1.0f / (1.0f + metal::exp(metal::abs(float(x))));
    return T((float(x) < 0) ? y : 1.0f - y);
  }
};
"""


def selected():
    value = os.environ.get(ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{ENV} must be 0 or 1")
    return value == "1"


def status():
    return {"implemented": True, "selected": selected(), "qualified": False,
            "observed_used": False, "law": LAW, "native_validation": "pending"}


def _header(ntcol):
    if type(ntcol) is not int or ntcol not in (32, 64):
        raise ValueError("tiled q4 layout must use 32 or 64 columns")
    return f"#define CHECK 1\n#define SWIGLU 1\n#define CLAMP 0\n#define NTW {ntcol}\n" + _HEADER


def metal_source(ntcol=32):
    """Standalone compile-only shader; no Metal device is created."""
    return _header(ntcol) + r"""
kernel void mlx2_ironmlx_fused_prefill(
 const device bfloat* X [[buffer(0)]], const device uint* W [[buffer(1)]],
 const device bfloat* Scale [[buffer(2)]], const device bfloat* B [[buffer(3)]],
 const device int* dims [[buffer(4)]], device bfloat* Y [[buffer(5)]],
 uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
 uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
 uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
 uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
""" + _BODY + "\n}\n"


def bf16(values):
    """Round finite FP32 values to BF16, ties to even; return FP32 storage."""
    x = np.asarray(values, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("finite values required")
    u = x.view(np.uint32)
    result = ((u + np.uint32(0x7fff) + ((u >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("BF16 overflow")
    return result


def reference(x, packed, scales, biases, *, ntcol=32):
    """CPU mathematical oracle for tiled q4 + rounded gate/up + SwiGLU.

    NumPy GEMM reduction is not a bit reference for the MPP tensor unit.
    Inputs use the existing lane layout [N/ntcol,K/64,ntcol,8] uint32;
    scales and affine biases remain [N,K/64] BF16 (FP32 storage here).
    """
    _header(ntcol)
    if any(type(value) is not np.ndarray for value in (x, packed, scales, biases)):
        raise ValueError("CPU reference requires NumPy arrays; device protocols are refused")
    x, s, b = bf16(x), bf16(scales), bf16(biases)
    if x.ndim < 2 or s.ndim != 2 or s.shape != b.shape:
        raise ValueError("activation and affine metadata geometry")
    n, groups = s.shape
    k = x.shape[-1]
    p = np.asarray(packed)
    if (p.dtype != np.uint32 or n < 64 or k < 64 or groups < 1
            or n % 64 or k % 64 or groups * 64 != k
            or p.size != n * k // 8 or x.size == 0):
        raise ValueError("affine4/g64 tiled geometry")
    words = p.reshape(n // ntcol, groups, ntcol, 8).transpose(0, 2, 1, 3).reshape(n, k // 8)
    codes = ((words[..., None] >> (np.arange(8, dtype=np.uint32) * 4)) & 15).reshape(n, k)
    # FP64 product/add followed by FP32 approximates a float32 fused affine
    # dequantization before the explicit BF16 store.
    weight = bf16(codes.astype(np.float64) * np.repeat(s, 64, axis=1)
                  + np.repeat(b, 64, axis=1))
    gate, up = np.split(bf16(x @ weight.T), 2, axis=-1)
    # Round each BF16 arithmetic boundary as in the shader epilogue.
    with np.errstate(over="ignore"):
        exp = np.exp(np.abs(gate)).astype(np.float32)
        inv = np.float32(1) / (np.float32(1) + exp)
    sig = bf16(np.where(gate < 0, inv, np.float32(1) - inv))
    return bf16(bf16(gate * sig) * up)


@lru_cache(maxsize=2)
def _kernel(ntcol):
    import mlx.core as mx
    return mx.fast.metal_kernel(name=f"mlx2_ironmlx_fused_prefill_nt{ntcol}",
        input_names=["X", "W", "Scale", "B", "dims"], output_names=["Y"],
        header=_header(ntcol), source=_BODY)


def project(x, packed, scales, biases, *, ntcol=32):
    """Explicit experimental backend entry; input is a fused gate/up weight.

    Callers must bind LAW into their adapter/cache identity. No serving
    adapter installs this entry automatically. Unknown devices refuse.
    """
    if not selected():
        raise RuntimeError(f"select {ENV}=1 to request the unqualified candidate")
    _header(ntcol)
    import mlx.core as mx
    if (mx.default_device() != mx.gpu or not mx.metal.is_available()
            or "M5" not in mx.device_info().get("device_name", "")):
        raise RuntimeError("unqualified fused prefill requires an M5 GPU")
    if len(x.shape) < 2 or len(scales.shape) != 2 or scales.shape != biases.shape:
        raise ValueError("activation and affine metadata geometry")
    n, groups = scales.shape
    k = x.shape[-1]
    if k < 64 or n < 64 or groups < 1 or x.size == 0:
        raise ValueError("requires nonempty affine4/g64 activations and weights")
    m = x.size // k
    if (n % 64 or k % 64 or groups * 64 != k or m < 1
            or packed.size != n * k // 8 or packed.dtype != mx.uint32
            or any(a.dtype != mx.bfloat16 for a in (x, scales, biases))):
        raise ValueError("requires BF16 affine4/g64 tiled fused gate/up weights")
    dims = mx.array([m, n, k, 0, n // 2, k], dtype=mx.int32)
    return _kernel(ntcol)(inputs=[mx.contiguous(x), mx.contiguous(packed),
        mx.contiguous(scales), mx.contiguous(biases), dims],
        grid=((m + 127) // 128 * 128, (n // 2 + 31) // 32, 1),
        threadgroup=(128, 1, 1), output_shapes=[(*x.shape[:-1], n // 2)],
        output_dtypes=[mx.bfloat16])[0]
