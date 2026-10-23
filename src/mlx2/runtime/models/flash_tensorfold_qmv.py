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


def _shared_source(bits):
    """Two rows share each decoded block; per-row arithmetic stays ordered.

    Original mlx2 candidate over the existing TensorFold dot-product law.
    Register reuse avoids a threadgroup staging buffer and barriers. A lone
    tail row performs exactly the same arithmetic without an out-of-bounds
    activation read. This is not a measured performance improvement.
    """
    if bits == 4:
        decode = """
      const device uint16_t* packed = (const device uint16_t*)(w + j * KB);
      for (int i = 0; i < 4; i++) {
        const uint16_t word = packed[i];
        qw[4*i] = float(word & 0x000f); qw[4*i+1] = float(word & 0x00f0);
        qw[4*i+2] = float(word & 0x0f00); qw[4*i+3] = float(word & 0xf000);
      }
"""
        dot = """
        for (int i = 0; i < 4; i++)
          accum += xt[r][4*i] * qw[4*i] + xt[r][4*i+1] * qw[4*i+1] +
                   xt[r][4*i+2] * qw[4*i+2] + xt[r][4*i+3] * qw[4*i+3];
"""
    elif bits == 8:
        decode = "for (int i = 0; i < 16; i++) qw[i] = float(w[j * KB + i]);"
        dot = "for (int i = 0; i < 16; i++) accum += xt[r][i] * qw[i];"
    else:
        raise ValueError("shared Flash row matvec supports q4 and q8")
    return """
  const uint lane = thread_index_in_simdgroup;
  const int first = int(simdgroup_index_in_threadgroup) * 2;
  const int row0 = int(threadgroup_position_in_grid.y) * RPS;
  constexpr int KB = K * BITS / 8;
  constexpr int KG = K / GS;
  const device uint8_t* w = (const device uint8_t*)W + size_t(row0) * KB + lane * (2 * BITS);
  const device bfloat* sc = S + size_t(row0) * KG + (lane * 16) / GS;
  const device bfloat* bi = B + size_t(row0) * KG + (lane * 16) / GS;
  float acc[2][RPS];
  for (int r = 0; r < 2; r++)
    for (int j = 0; j < RPS; j++) acc[r][j] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += 512) {
    float xt[2][16], sums[2];
    for (int r = 0; r < 2; r++) {
      if (first + r < ROWS) {
        const device bfloat* x = X + (first + r) * K + k0 + lane * 16;
        sums[r] = LOAD(x, xt[r]);
      }
    }
    for (int j = 0; j < RPS; j++) {
      float qw[16];
      DECODE
      const float scale = float(sc[j * KG]), bias = float(bi[j * KG]);
      for (int r = 0; r < 2; r++) {
        if (first + r >= ROWS) continue;
        float accum = 0.0f;
        DOT
        acc[r][j] += scale * accum + sums[r] * bias;
      }
    }
    w += 64 * BITS; sc += 512 / GS; bi += 512 / GS;
  }
  for (int r = 0; r < 2; r++) {
    if (first + r >= ROWS) continue;
    for (int j = 0; j < RPS; j++) {
      const float value = simd_sum(acc[r][j]);
      if (lane == 0) OUT[(first + r) * N + row0 + j] = bfloat(value);
    }
  }
""".replace("BITS", str(bits)).replace("LOAD", "load16" if bits == 4 else "load16_q8").replace(
        "DECODE", decode).replace("DOT", dot)


def _kernel(bits, *, shared_weights=False):
    source = _shared_source(bits) if shared_weights else _SOURCE_Q4 if bits == 4 else _SOURCE_Q8
    key = (bits, bool(shared_weights))
    if key not in _KERNELS:
        digest = hashlib.sha256((_HEADER + source).encode()).hexdigest()[:16]
        _KERNELS[key] = mx.fast.metal_kernel(
            name=f"mlx2_flash_tensorfold_qmv_q{bits}_{digest}",
            input_names=["X", "W", "S", "B"], output_names=["OUT"],
            source=source, header=_HEADER,
        )
    return _KERNELS[key]


def eligible(module) -> bool:
    if not isinstance(module, nn.QuantizedLinear):
        return False
    bits = int(module.bits)
    if bits not in (4, 8) or int(module.group_size) != 64:
        return False
    n = int(module.weight.shape[0])
    k = int(module.weight.shape[1]) * (32 // bits)
    return k % 512 == 0 and n % 4 == 0 and module.scales.dtype == mx.bfloat16


def qmv_rows(x, module, *, shared_weights=False):
    """Compute 1..17 rows with the same arithmetic for every row width."""
    shape = x.shape
    rows = x.size // shape[-1]
    if x.dtype != mx.bfloat16 or not 1 <= rows <= 17 or not eligible(module):
        raise ValueError("TensorFold Flash row matvec requires bf16, affine q4/q8-g64 and 1..17 rows")
    n, k = int(module.weight.shape[0]), int(shape[-1])
    bits = int(module.bits)
    if k != int(module.weight.shape[1]) * (32 // bits):
        raise ValueError("TensorFold Flash row matvec activation width mismatch")
    if type(shared_weights) is not bool:
        raise ValueError("shared_weights must be boolean")
    groups = (rows + 1) // 2 if shared_weights else rows
    template = [("K", k), ("N", n), ("RPS", 4), ("GS", 64)]
    if shared_weights:
        template.append(("ROWS", rows))
    out = _kernel(bits, shared_weights=shared_weights)(
        inputs=[x.reshape(rows, k), module.weight, module.scales, module.biases],
        template=template,
        grid=(32 * groups, n // 4, 1), threadgroup=(32 * groups, 1, 1),
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
            shared = getattr(self, "_tensorfold_qmv_shared_weights", False)
            if shared:
                _COUNTS["shared_weight_calls"] += 1
            return qmv_rows(x, self, shared_weights=shared)
        _COUNTS["stock_calls"] += 1
        return nn.QuantizedLinear.__call__(self, x)


def hc_owned_projection_ids(model) -> set[int]:
    """Ids of the projections HC's fused decode kernels own.

    HC admission requires these to be plain ``nn.QuantizedLinear``; a class
    swap by any installer silently makes every HC call decline, despite the
    selected HC profile.  Shared by every TensorFold installer.
    """
    return {
        id(projection)
        for _, owner in model.named_modules()
        if getattr(owner, "hc_count", None) is not None
        for field in ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight")
        if (projection := getattr(owner, field, None)) is not None
    }


def install(model, *, shared_weights=False) -> dict:
    """Install on affine q4/q8-g64 dense projections without copying weights."""
    if type(shared_weights) is not bool:
        raise ValueError("shared_weights must be boolean")
    installed = []
    hc_owned = hc_owned_projection_ids(model)
    for name, module in model.named_modules():
        if type(module) is not nn.QuantizedLinear or id(module) in hc_owned:
            continue
        if any(part in name.split(".") for part in (
            "switch_mlp", "shared_expert", "ple", "lm_head", "mtp_draft_head"
        )):
            continue
        if eligible(module):
            module.__class__ = TensorFoldQMVLinear
            object.__setattr__(module, "_tensorfold_qmv_enabled", True)
            object.__setattr__(module, "_tensorfold_qmv_shared_weights", shared_weights)
            installed.append(name)
    if not installed:
        raise ValueError("TensorFold Flash qmv found no eligible affine q4/q8-g64 projections")
    return {"source_revision": "bb4b4a35863af562fc4ccb2586300d8f94b5d6de",
            "kernel": "qmv_rows_q4q8g64_shared2" if shared_weights else "qmv_rows_q4q8g64",
            **({"shared_weights": True, "qualified": False} if shared_weights else {}),
            "installed": len(installed),
            "names_sha256": hashlib.sha256("\n".join(sorted(installed)).encode()).hexdigest()}


def counters() -> dict:
    return dict(_COUNTS)
