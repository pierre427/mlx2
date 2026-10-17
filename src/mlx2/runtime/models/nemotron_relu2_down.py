# SPDX-License-Identifier: MIT
"""Geometry-locked Super ReLU2/down/FP32 top-22 decode candidate."""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from .qwen4_routed_decode import _header, expert_operands
from .switch_layers import QuantizedSwitchLinear

TOP_K = 22
INTERMEDIATE = 2688
HIDDEN = 1024
RPS = 4
SIMDGROUPS = 11


@dataclass(frozen=True)
class Admission:
    accepted: bool
    reason: str


def admit(x, indices, scores, down, *, training: bool) -> Admission:
    """Fail closed outside the exact served Super B1/S1 affine4 geometry."""
    if training:
        return Admission(False, "training")
    if tuple(x.shape) != (1, 1, HIDDEN) or x.dtype != mx.bfloat16:
        return Admission(False, "needs B1/S1 BF16 latent input")
    if indices.size != TOP_K or scores.size != TOP_K:
        return Admission(False, "needs one top-22 routed row")
    if scores.dtype != mx.float32:
        return Admission(False, "router scores must be float32")
    if not isinstance(down, QuantizedSwitchLinear):
        return Admission(False, "down projection is not QuantizedSwitchLinear")
    if down.bits != 4 or down.group_size != 64 or down.mode != "affine":
        return Admission(False, "down projection is not affine4 group64")
    if "bias" in down or down.get("biases") is None:
        return Admission(False, "down projection bias layout differs")
    if down["weight"].dtype != mx.uint32:
        return Admission(False, "packed weights must be uint32")
    if down["scales"].dtype != mx.bfloat16 or down["biases"].dtype != mx.bfloat16:
        return Admission(False, "affine parameters must be bfloat16")
    if tuple(down["weight"].shape[1:]) != (HIDDEN, INTERMEDIATE // 8):
        return Admission(False, "down geometry differs")
    if tuple(down["scales"].shape[1:]) != (HIDDEN, INTERMEDIATE // 64):
        return Admission(False, "down scale geometry differs")
    return Admission(True, "eligible")


SOURCE = r"""
    const uint3 tid = threadgroup_position_in_grid;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
    const int in_vec_size_g = K / GS;
    const int out_row = int(tid.y) * RPS;
    threadgroup float part[TOPK * RPS];

    for (int pair = 0; pair < 2; ++pair) {
      const int slot = int(sg) * 2 + pair;
      if (slot >= TOPK) continue;
      const size_t expert = size_t(rhs[slot]);
      const device uint8_t* ws = (const device uint8_t*)w +
          expert * N * in_vec_size_w + out_row * in_vec_size_w +
          int(lane) * PACKS_PER_THREAD * BYTES_PER_PACK;
      const device T* sc = scales + expert * N * in_vec_size_g +
          out_row * in_vec_size_g + int(lane) / SCALE_STEP_PER_THREAD;
      const device T* bs = biases + expert * N * in_vec_size_g +
          out_row * in_vec_size_g + int(lane) / SCALE_STEP_PER_THREAD;
      const device T* xp = x + slot * K + int(lane) * VALUES_PER_THREAD;
      float result[RPS] = {0};
      float x_thread[VALUES_PER_THREAD];

      int k = 0;
      for (; k < K - BLOCK_SIZE; k += BLOCK_SIZE) {
        float sum = 0.0f;
        for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
          T a0 = metal::max(xp[i], T(0)); a0 = a0 * a0;
          T a1 = metal::max(xp[i + 1], T(0)); a1 = a1 * a1;
          T a2 = metal::max(xp[i + 2], T(0)); a2 = a2 * a2;
          T a3 = metal::max(xp[i + 3], T(0)); a3 = a3 * a3;
          sum += float(a0) + float(a1) + float(a2) + float(a3);
          x_thread[i] = a0;
          x_thread[i + 1] = a1 / 16.0f;
          x_thread[i + 2] = a2 / 256.0f;
          x_thread[i + 3] = a3 / 4096.0f;
        }
        for (int row = 0; row < RPS; ++row) {
          result[row] += qdot_n(ws + row * in_vec_size_w, x_thread,
              sc[row * in_vec_size_g], bs[row * in_vec_size_g], sum,
              VALUES_PER_THREAD);
        }
        ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
        sc += BLOCK_SIZE / GS; bs += BLOCK_SIZE / GS; xp += BLOCK_SIZE;
      }
      const int remaining = clamp(
          int(K - k - int(lane) * VALUES_PER_THREAD), 0, VALUES_PER_THREAD);
      if (remaining > 0) {
        float sum = 0.0f;
        for (int i = 0; i < remaining; ++i) {
          T a = metal::max(xp[i], T(0)); a = a * a;
          sum += float(a);
          const int shift = (i & 3) * 4;
          x_thread[i] = float(a) / float(1 << shift);
        }
        for (int i = remaining; i < VALUES_PER_THREAD; ++i) x_thread[i] = 0.0f;
        for (int row = 0; row < RPS; ++row) {
          result[row] += qdot_n(ws + row * in_vec_size_w, x_thread,
              sc[row * in_vec_size_g], bs[row * in_vec_size_g], sum, remaining);
        }
      }
      for (int row = 0; row < RPS; ++row) result[row] = simd_sum(result[row]);
      if (lane == 0) {
        for (int row = 0; row < RPS; ++row) {
          const T expert_value = static_cast<T>(result[row]);
          part[slot * RPS + row] = float(expert_value) * scores[slot];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && int(lane) < RPS) {
      const int row = int(lane);
      float lanes[8] = {0.0f};
      for (int slot = 0; slot < TOPK; ++slot) {
        lanes[slot % 8] = part[slot * RPS + row] + lanes[slot % 8];
      }
      float acc = lanes[0];
      for (int l = 1; l < 8; ++l) acc = lanes[l] + acc;
      y[out_row + row] = static_cast<T>(acc);
    }
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="mlx2_nemotron_relu2_down_fp32_top22",
            input_names=["x", "w", "scales", "biases", "rhs", "scores"],
            output_names=["y"],
            header=_header(fast=False),
            source=SOURCE,
        )
    return _KERNEL


def relu2_down_combine(preactivation, indices, scores, down):
    """Exact intended served tail: ReLU2, expert down, FP32 top-22 sum."""
    return _kernel()(
        inputs=[
            preactivation.reshape(TOP_K, INTERMEDIATE),
            *expert_operands(down),
            indices.reshape(TOP_K).astype(mx.uint32),
            scores.reshape(TOP_K),
        ],
        template=[
            ("T", preactivation.dtype), ("K", INTERMEDIATE), ("N", HIDDEN),
            ("RPS", RPS), ("TOPK", TOP_K),
        ],
        grid=(32, SIMDGROUPS * HIDDEN // RPS, 1),
        threadgroup=(32, SIMDGROUPS, 1),
        output_shapes=[(HIDDEN,)],
        output_dtypes=[preactivation.dtype],
    )[0].reshape(1, 1, HIDDEN)
