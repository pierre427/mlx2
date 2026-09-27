# SPDX-License-Identifier: Apache-2.0
# Adapted from jundot/omlx PR #3912 (Apache-2.0); see docs/PROVENANCE.md and
# provenance/omlx-fn-2026-09-25.json. The quantized dot products are omlx's
# transcription of MLX 0.32.2 ``quantized.h`` (MIT); see that record.
"""One-token routed experts for Flash-Next decode in fewer launches.

After routing, a one-token Flash-Next MoE block runs its routed experts as the
gate+up ``gather_qmm``, the compiled SwiGLU, and the down projection with the
router-weighted top-k sum (mlx2's ``qwen4_fused_down`` tile4 kernel in the
Flash-Next profile, or ``gather_qmm`` + multiply + sum on the stock path).

Two kernels from omlx #3912:

``gate_up_swiglu``
    gate+up with a SwiGLU epilogue. Each simdgroup computes gate rows and the
    matching up rows of one expert with MLX's ``qmv_fast`` lane partition and
    add order, rounds both to the activation dtype, then applies MLX's
    ``Sigmoid`` and the two multiplies of the compiled ``swiglu`` in order.
    Intended to equal ``gather_qmm`` + ``swiglu`` bit for bit.
``down_combine``
    down with the router-weighted sum: simdgroup ``j`` runs the stock ``qmv``
    work of expert ``j``, rounds, multiplies by the score in the activation
    dtype and sums the ten products in ``col_reduce_small`` order. Intended to
    equal the stock ``gather_qmm`` + ``(x * scores).sum`` tail bit for bit, which
    is NOT the Flash-Next profile's tile4 fused down (that one accumulates the
    weighted products in fp32).

Modes (``MLX_QWEN4_MOE_ROUTED_DECODE``, or the Flash-Next policy field
``moe_routed_decode``; ``set_moe_routed_decode_mode`` switches live):

``off``        (default) nothing changes.
``gate_up``    the gate+up kernel replaces ``gather_qmm`` + SwiGLU; the down
               path is whatever the block would run anyway.
``two_launch`` both kernels, the literal omlx pairing.

Admission is structural and exact-shape: one token (so the gather is
unsorted), bf16 activations and scales, top-k 10, affine 4-bit group-64,
hidden % 512 == 0 and intermediate % 512 != 0 (MLX then picks ``qmv_fast`` for
gate+up and ``qmv`` for down), resident (not streamed) expert tables.
Everything else keeps the composed body. The MLX transcription is from 0.32.2;
the mlx2 venv runs a fork (39400a0d4), so bit-exactness is a GPU-checked
claim (``scripts/check_fn_routed_decode.py``), not an assumption.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import mlx.core as mx

ROUTED_DECODE_ENV = "MLX_QWEN4_MOE_ROUTED_DECODE"
MODES = ("off", "gate_up", "two_launch")
TOP_K = 10
BITS = 4
GROUP_SIZE = 64
GATE_UP_ROWS = 2  # gate rows (and as many up rows) per simdgroup
GATE_UP_SIMDGROUPS = 2
DOWN_ROWS = 4


def mode_from_env() -> str:
    raw = os.environ.get(ROUTED_DECODE_ENV, "off").strip().lower()
    aliases = {"": "off", "0": "off", "false": "off", "1": "gate_up"}
    raw = aliases.get(raw, raw)
    if raw not in MODES:
        raise ValueError(
            f"{ROUTED_DECODE_ENV}={raw!r}: expected one of {MODES}"
        )
    return raw


@dataclass(frozen=True)
class RoutedDecodeAdmission:
    accepted: bool
    reason: str


def _quantized_ok(layer) -> str | None:
    if layer is None:
        return "missing projection"
    if "weight" not in layer or "scales" not in layer:
        return "projection is not resident affine-quantized"
    if getattr(layer, "bits", None) != BITS:
        return f"bits {getattr(layer, 'bits', None)} != {BITS}"
    if getattr(layer, "group_size", None) != GROUP_SIZE:
        return f"group size {getattr(layer, 'group_size', None)} != {GROUP_SIZE}"
    if getattr(layer, "mode", "affine") != "affine":
        return "not affine"
    if "biases" not in layer or "bias" in layer:
        return "needs affine biases and no linear bias"
    if layer["scales"].dtype != mx.bfloat16 or layer["biases"].dtype != mx.bfloat16:
        return "scales/biases must be bfloat16"
    if layer["weight"].dtype != mx.uint32:
        return "weight must be packed uint32"
    if getattr(layer, "training", False):
        return "training"
    return None


def admit_routed_decode(x, indices, scores, gate_up, down) -> RoutedDecodeAdmission:
    """Structural check; never evaluates an array (no device sync)."""
    hidden = x.shape[-1]
    if x.size != hidden:
        return RoutedDecodeAdmission(False, "not one token")
    if x.dtype != mx.bfloat16:
        return RoutedDecodeAdmission(False, "activation must be bfloat16")
    if indices.size != TOP_K or indices.shape[-1] != TOP_K:
        return RoutedDecodeAdmission(False, f"top-k must be {TOP_K}")
    if scores is not None and (scores.size != TOP_K or scores.dtype != x.dtype):
        return RoutedDecodeAdmission(False, "scores must be [10] in the activation dtype")
    for name, layer in (("gate_up", gate_up), ("down", down)):
        reason = _quantized_ok(layer)
        if reason is not None:
            return RoutedDecodeAdmission(False, f"{name}: {reason}")
    inter = down["weight"].shape[-1] * 32 // BITS
    if hidden % 512:
        return RoutedDecodeAdmission(False, "hidden % 512 != 0 (MLX would not pick qmv_fast)")
    if inter % 512 == 0 or inter % GROUP_SIZE:
        return RoutedDecodeAdmission(False, "intermediate must be a non-multiple of 512, multiple of 64")
    if tuple(gate_up["weight"].shape[1:]) != (2 * inter, hidden * BITS // 32):
        return RoutedDecodeAdmission(False, f"gate_up weight shape {tuple(gate_up['weight'].shape)}")
    if tuple(down["weight"].shape[1:]) != (hidden, inter * BITS // 32):
        return RoutedDecodeAdmission(False, f"down weight shape {tuple(down['weight'].shape)}")
    if (2 * inter) % (GATE_UP_ROWS * GATE_UP_SIMDGROUPS) or hidden % DOWN_ROWS:
        return RoutedDecodeAdmission(False, "row tiling")
    return RoutedDecodeAdmission(True, "eligible")


def runtime_supported() -> bool:
    return mx.default_device() == mx.gpu and mx.metal.is_available()


# --------------------------------------------------------------------------
# Metal sources: from omlx e3d4a213 (qwen35_moe_routed_decode.py and the
# _HEADER of moe_verify_gather.py), kept diffable. One adaptation: the input
# sums in load_vector/load_vector_safe cast each element to float before
# adding, as the mlx2 venv's MLX fork does since 2d6705923 ("Accumulate affine
# QMV input sums in float"); omlx transcribes 0.32.2, which adds in T. With the
# verbatim header the gate/up rows differed from gather_qmm in ~78% of
# elements (GPU, 2026-09-25).
# --------------------------------------------------------------------------
QMV_HEADER = r"""
using namespace metal;

constant constexpr int SIMD_SIZE = 32;
constant constexpr int BITS = __BITS__;
constant constexpr int GS = __GS__;
constant constexpr int FAST = __FAST__;
constant constexpr int PACK_FACTOR =
    (BITS == 5) ? 8 : (BITS == 6 ? 4 : 32 / BITS);
constant constexpr int BYTES_PER_PACK =
    ((BITS & (BITS - 1)) == 0) ? 4 : (BITS == 5 ? 5 : 3);
constant constexpr int PACKS_PER_THREAD = FAST ? 2 : 1;
constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
constant constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
constant constexpr int RESULTS_PER_SIMDGROUP = 4;
constant constexpr int BN = 8;

template <typename T>
inline float load_vector(const device T* x, thread float* x_thread) {
  float sum = 0;
  if (BITS == 4) {
    for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  } else if (BITS == 5) {
    for (int i = 0; i < VALUES_PER_THREAD; i += 8) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]) + float(x[i + 4]) + float(x[i + 5]) +
          float(x[i + 6]) + float(x[i + 7]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  } else if (BITS == 6) {
    for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  } else if (BITS == 8) {
    for (int i = 0; i < VALUES_PER_THREAD; i++) {
      sum += float(x[i]);
      x_thread[i] = x[i];
    }
  }
  return sum;
}

template <typename T>
inline float load_vector_safe(const device T* x, thread float* x_thread, int N) {
  float sum = 0;
  if (BITS == 4) {
    for (int i = 0; i < N; i += 4) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  } else if (BITS == 5) {
    for (int i = 0; i < N; i += 8) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]) + float(x[i + 4]) + float(x[i + 5]) +
          float(x[i + 6]) + float(x[i + 7]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  } else if (BITS == 6) {
    for (int i = 0; i < N; i += 4) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  } else if (BITS == 8) {
    for (int i = 0; i < N; i++) {
      sum += float(x[i]);
      x_thread[i] = x[i];
    }
  }
  for (int i = N; i < VALUES_PER_THREAD; i++) {
    x_thread[i] = 0;
  }
  return sum;
}

inline float qdot_n(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum,
    int N) {
  float accum = 0;
  if (BITS == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  } else if (BITS == 5) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  } else if (BITS == 6) {
    for (int i = 0; i < (N / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  } else if (BITS == 8) {
    for (int i = 0; i < N; i++) {
      accum += x_thread[i] * w[i];
    }
  }
  return scale * accum + sum * bias;
}
"""

SIGMOID = r"""
// MLX 0.32.2 Sigmoid, evaluated in T as the compiled swiglu does.
template <typename U>
inline U omlx_mlx_sigmoid(U x) {
  auto y = 1 / (1 + metal::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
"""

GATE_UP_SOURCE = r"""
    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
    const int in_vec_size_g = K / GS;
    const int out_row = int(tid.y) * (NSG * RPS) + int(simd_gid) * RPS;
    const size_t expert = size_t(rhs[tid.z]);

    const device uint8_t* ws = (const device uint8_t*)w +
        expert * N * in_vec_size_w + out_row * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* sc = scales + expert * N * in_vec_size_g +
        out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* bs = biases + expert * N * in_vec_size_g +
        out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* xp = x + int(simd_lid) * VALUES_PER_THREAD;

    float x_thread[VALUES_PER_THREAD];
    float result[2 * RPS] = {0};

    for (int k = 0; k < K; k += BLOCK_SIZE) {
      float sum = load_vector<T>(xp, x_thread);
      for (int row = 0; row < 2 * RPS; row++) {
        const int r = row < RPS ? row : N / 2 + row - RPS;
        const device uint8_t* wl = ws + r * in_vec_size_w;
        float s = sc[r * in_vec_size_g];
        float b = bs[r * in_vec_size_g];
        result[row] += qdot_n(wl, x_thread, s, b, sum, VALUES_PER_THREAD);
      }
      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK_SIZE / GS;
      bs += BLOCK_SIZE / GS;
      xp += BLOCK_SIZE;
    }

    for (int row = 0; row < 2 * RPS; row++) {
      result[row] = simd_sum(result[row]);
    }
    if (simd_lid == 0) {
      device T* yp = y + size_t(tid.z) * (N / 2) + out_row;
      for (int row = 0; row < RPS; row++) {
        T g = static_cast<T>(result[row]);
        T u = static_cast<T>(result[row + RPS]);
        T t = g * omlx_mlx_sigmoid<T>(g);
        yp[row] = t * u;
      }
    }
"""

DOWN_SOURCE = r"""
    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
    const int in_vec_size_g = K / GS;
    const int out_row = int(tid.y) * RPS;
    const int slot = int(simd_gid);
    const size_t expert = size_t(rhs[slot]);
    threadgroup T part[10 * RPS];

    const device uint8_t* ws = (const device uint8_t*)w +
        expert * N * in_vec_size_w + out_row * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* sc = scales + expert * N * in_vec_size_g +
        out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* bs = biases + expert * N * in_vec_size_g +
        out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* xp = x + slot * K + int(simd_lid) * VALUES_PER_THREAD;

    float x_thread[VALUES_PER_THREAD];
    float result[RPS] = {0};

    int k = 0;
    for (; k < K - BLOCK_SIZE; k += BLOCK_SIZE) {
      float sum = load_vector<T>(xp, x_thread);
      for (int row = 0; row < RPS; row++) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        float s = sc[row * in_vec_size_g];
        float b = bs[row * in_vec_size_g];
        result[row] += qdot_n(wl, x_thread, s, b, sum, VALUES_PER_THREAD);
      }
      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK_SIZE / GS;
      bs += BLOCK_SIZE / GS;
      xp += BLOCK_SIZE;
    }
    const int remaining = clamp(
        int(K - k - int(simd_lid) * VALUES_PER_THREAD), 0, VALUES_PER_THREAD);
    if (remaining > 0) {
      float sum = load_vector_safe<T>(xp, x_thread, remaining);
      for (int row = 0; row < RPS; row++) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        float s = sc[row * in_vec_size_g];
        float b = bs[row * in_vec_size_g];
        result[row] += qdot_n(wl, x_thread, s, b, sum, remaining);
      }
    }

    for (int row = 0; row < RPS; row++) {
      result[row] = simd_sum(result[row]);
    }
    if (simd_lid == 0) {
      for (int row = 0; row < RPS; row++) {
        part[slot * RPS + row] = static_cast<T>(result[row]) * scores[slot];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0 && int(simd_lid) < RPS) {
      // col_reduce_small over 10 rows with 8 lanes: rows j and j + 8 fold
      // first, then lanes 0..7 accumulate in order, all in T.
      const int row = int(simd_lid);
      T t[8];
      for (int j = 0; j < 8; ++j) {
        t[j] = part[j * RPS + row] + T(0);
      }
      t[0] = part[8 * RPS + row] + t[0];
      t[1] = part[9 * RPS + row] + t[1];
      T acc = t[0];
      for (int j = 1; j < 8; ++j) {
        acc = t[j] + acc;
      }
      y[out_row + row] = acc;
    }
"""


def _header(fast: bool) -> str:
    return (
        QMV_HEADER.replace("__BITS__", str(BITS))
        .replace("__GS__", str(GROUP_SIZE))
        .replace("__FAST__", "1" if fast else "0")
        + SIGMOID
    )


_KERNELS = None


def _kernels():
    global _KERNELS
    if _KERNELS is None:
        gate_up = mx.fast.metal_kernel(
            name="mlx2_qwen4_moe_gate_up_swiglu_decode",
            input_names=["x", "w", "scales", "biases", "rhs"],
            output_names=["y"],
            header=_header(fast=True),
            source=GATE_UP_SOURCE,
        )
        down = mx.fast.metal_kernel(
            name="mlx2_qwen4_moe_down_combine_decode",
            input_names=["x", "w", "scales", "biases", "rhs", "scores"],
            output_names=["y"],
            header=_header(fast=False),
            source=DOWN_SOURCE,
        )
        _KERNELS = (gate_up, down)
    return _KERNELS


def gate_up_swiglu(x, indices, gate_up):
    """``swiglu(gate, up)`` of the ten routed experts for one token: [10, inter]."""
    hidden = x.shape[-1]
    inter = gate_up["weight"].shape[1] // 2
    rows = GATE_UP_ROWS * GATE_UP_SIMDGROUPS
    kernel = _kernels()[0]
    return kernel(
        inputs=[
            x.reshape(hidden),
            gate_up["weight"],
            gate_up["scales"],
            gate_up["biases"],
            indices.reshape(TOP_K).astype(mx.uint32),
        ],
        template=[
            ("T", x.dtype),
            ("K", hidden),
            ("N", 2 * inter),
            ("RPS", GATE_UP_ROWS),
            ("NSG", GATE_UP_SIMDGROUPS),
        ],
        grid=(32, GATE_UP_SIMDGROUPS * inter // rows, TOP_K),
        threadgroup=(32, GATE_UP_SIMDGROUPS, 1),
        output_shapes=[(TOP_K, inter)],
        output_dtypes=[x.dtype],
    )[0]


def down_combine(h, indices, scores, down):
    """``sum_j scores[j] * down_j(h[j])`` for one token: [hidden]."""
    inter = h.shape[-1]
    hidden = down["weight"].shape[1]
    kernel = _kernels()[1]
    return kernel(
        inputs=[
            h.reshape(TOP_K, inter),
            down["weight"],
            down["scales"],
            down["biases"],
            indices.reshape(TOP_K).astype(mx.uint32),
            scores.reshape(TOP_K),
        ],
        template=[("T", h.dtype), ("K", inter), ("N", hidden), ("RPS", DOWN_ROWS)],
        grid=(32, TOP_K * hidden // DOWN_ROWS, 1),
        threadgroup=(32, TOP_K, 1),
        output_shapes=[(hidden,)],
        output_dtypes=[h.dtype],
    )[0]
