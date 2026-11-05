# SPDX-License-Identifier: Apache-2.0
# The per-row quantized dot products are qwen4_routed_decode's QMV_HEADER
# (adapted from jundot/omlx PR #3912, Apache-2.0; MLX 0.32.2 quantized.h, MIT)
# plus MLX's 2-bit qmv branch. See provenance/deepseek-v41-omlx.json.
"""Device-routed one-token routed experts for the omlx-layout V4.1 route.

The per-expert reference (and the bank path's gather_qmm at one row) runs
each selected expert's gate, up and down as MLX ``quantized_matmul``. These
kernels run the same per-row arithmetic for all selected experts in one
launch each, reading the expert IDs from a device buffer, so the route never
leaves the GPU:

``gate_up``  MLX ``qmv_fast`` lane partition and add order (K=5120 is a
             multiple of 512) for the gate and up rows of every slot; raw F32.
``down``     MLX ``qmv`` with its tail-safe loop (K=2304 is not); raw F32.

Activations are F32 (the FP8-rounded inputs) and scales/biases BF16, read as
float exactly as MLX promotes them. The BF16 rounds, the clamped SwiGLU, the
FP8 middle and the expert-ID-ordered sum stay ordinary MLX ops in the caller.
Bit equality with ``quantized_matmul`` is a GPU-checked claim (self_check and
the route's oracle), not an assumption.
"""
import hashlib
from functools import lru_cache
from pathlib import Path

import mlx.core as mx

GATE_UP_ROWS = 2
GATE_UP_SIMDGROUPS = 2
DOWN_ROWS = 4

_TWO_BIT_LOAD = r"""if (BITS == 2) {
    for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  } else """
_TWO_BIT_LOAD_SAFE = _TWO_BIT_LOAD.replace("i < VALUES_PER_THREAD", "i < N")
_TWO_BIT_DOT = r"""if (BITS == 2) {
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  } else """

GATE_UP_SOURCE = r"""
    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
    const int in_vec_size_g = K / GS;
    const int out_row = int(tid.y) * (NSG * RPS) + int(simd_gid) * RPS;
    const size_t row0 = size_t(rhs[tid.z]) * NI + out_row;
    const int lane_w = int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const int lane_g = int(simd_lid) / SCALE_STEP_PER_THREAD;

    const device uint8_t* gw = (const device uint8_t*)wg + row0 * in_vec_size_w + lane_w;
    const device uint8_t* uw = (const device uint8_t*)wu + row0 * in_vec_size_w + lane_w;
    const device S* gs = sg + row0 * in_vec_size_g + lane_g;
    const device S* gb = bg + row0 * in_vec_size_g + lane_g;
    const device S* us = su + row0 * in_vec_size_g + lane_g;
    const device S* ub = bu + row0 * in_vec_size_g + lane_g;
    const device T* xp = x + int(simd_lid) * VALUES_PER_THREAD;

    float x_thread[VALUES_PER_THREAD];
    float result[2 * RPS] = {0};

    for (int k = 0; k < K; k += BLOCK_SIZE) {
      float sum = load_vector<T>(xp, x_thread);
      for (int row = 0; row < 2 * RPS; row++) {
        const bool g = row < RPS;
        const int r = g ? row : row - RPS;
        const device uint8_t* wl = (g ? gw : uw) + r * in_vec_size_w;
        float s = float((g ? gs : us)[r * in_vec_size_g]);
        float b = float((g ? gb : ub)[r * in_vec_size_g]);
        result[row] += qdot_n(wl, x_thread, s, b, sum, VALUES_PER_THREAD);
      }
      gw += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      uw += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      gs += BLOCK_SIZE / GS;
      gb += BLOCK_SIZE / GS;
      us += BLOCK_SIZE / GS;
      ub += BLOCK_SIZE / GS;
      xp += BLOCK_SIZE;
    }

    for (int row = 0; row < 2 * RPS; row++) {
      result[row] = simd_sum(result[row]);
    }
    if (simd_lid == 0) {
      device T* gp = yg + size_t(tid.z) * NI + out_row;
      device T* up = yu + size_t(tid.z) * NI + out_row;
      for (int row = 0; row < RPS; row++) {
        gp[row] = static_cast<T>(result[row]);
        up[row] = static_cast<T>(result[row + RPS]);
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

    const device uint8_t* ws = (const device uint8_t*)w +
        expert * N * in_vec_size_w + out_row * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device S* sc = scales + expert * N * in_vec_size_g +
        out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device S* bs = biases + expert * N * in_vec_size_g +
        out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* xp = x + slot * K + int(simd_lid) * VALUES_PER_THREAD;

    float x_thread[VALUES_PER_THREAD];
    float result[RPS] = {0};

    int k = 0;
    for (; k < K - BLOCK_SIZE; k += BLOCK_SIZE) {
      float sum = load_vector<T>(xp, x_thread);
      for (int row = 0; row < RPS; row++) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        float s = float(sc[row * in_vec_size_g]);
        float b = float(bs[row * in_vec_size_g]);
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
        float s = float(sc[row * in_vec_size_g]);
        float b = float(bs[row * in_vec_size_g]);
        result[row] += qdot_n(wl, x_thread, s, b, sum, remaining);
      }
    }

    for (int row = 0; row < RPS; row++) {
      result[row] = simd_sum(result[row]);
    }
    if (simd_lid == 0) {
      for (int row = 0; row < RPS; row++) {
        y[size_t(slot) * N + out_row + row] = static_cast<T>(result[row]);
      }
    }
"""


def _once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("V4.1 routed decode: pinned QMV anchor changed: " + old[:60])
    return text.replace(old, new)


def header(bits, group_size, fast):
    from .qwen4_routed_decode import QMV_HEADER
    text = QMV_HEADER.replace("__BITS__", str(bits)).replace("__GS__", str(group_size))
    text = text.replace("__FAST__", "1" if fast else "0")
    # MLX qmv_fast reads one pack per thread at 2 bits (packs_per_thread =
    # bits == 2 ? 1 : 2); the shared header assumes two for every fast width.
    text = _once(text, "constant constexpr int PACKS_PER_THREAD = FAST ? 2 : 1;",
                 "constant constexpr int PACKS_PER_THREAD = (FAST && BITS != 2) ? 2 : 1;")
    load = "inline float load_vector(const device T* x, thread float* x_thread) {\n  float sum = 0;\n  if (BITS == 4) {"
    text = _once(text, load, load.replace("if (BITS == 4) {", _TWO_BIT_LOAD + "if (BITS == 4) {"))
    safe = ("inline float load_vector_safe(const device T* x, thread float* x_thread, int N) {\n"
            "  float sum = 0;\n  if (BITS == 4) {")
    text = _once(text, safe, safe.replace("if (BITS == 4) {", _TWO_BIT_LOAD_SAFE + "if (BITS == 4) {"))
    dot = "  float accum = 0;\n  if (BITS == 4) {"
    text = _once(text, dot, dot.replace("if (BITS == 4) {", _TWO_BIT_DOT + "if (BITS == 4) {"))
    return text


@lru_cache(maxsize=1)
def implementation_identity():
    from . import qwen4_routed_decode
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(qwen4_routed_decode.__file__)):
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
    digest.update(mx.__version__.encode())
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _kernel(kind, bits, group_size):
    if kind == "gate_up":
        return mx.fast.metal_kernel(
            name=f"mlx2_v41_routed_gate_up_b{bits}_g{group_size}",
            input_names=["x", "wg", "sg", "bg", "wu", "su", "bu", "rhs"], output_names=["yg", "yu"],
            header=header(bits, group_size, True), source=GATE_UP_SOURCE)
    return mx.fast.metal_kernel(
        name=f"mlx2_v41_routed_down_b{bits}_g{group_size}",
        input_names=["x", "w", "scales", "biases", "rhs"], output_names=["y"],
        header=header(bits, group_size, False), source=DOWN_SOURCE)


def supports(hidden, inter, top, specs):
    """Structural gate: the qmv_fast / qmv choice MLX makes for these shapes."""
    gate, up, down = (specs[k] for k in ("gate", "up", "down"))
    same = (gate["bits"], gate["group_size"]) == (up["bits"], up["group_size"])
    return (same and gate["bits"] in (2, 4, 6, 8) and down["bits"] in (2, 4, 6, 8)
            and hidden % 512 == 0 and inter % 512 != 0 and inter % (GATE_UP_ROWS*GATE_UP_SIMDGROUPS) == 0
            and hidden % DOWN_ROWS == 0 and 1 <= top <= 32)


def gate_up(x, ids, gate, up, spec):
    """Raw F32 gate and up rows of every routed slot: two [top, inter] arrays."""
    top = ids.size
    inter = gate[0].shape[1]
    hidden = x.shape[-1]
    rows = GATE_UP_ROWS * GATE_UP_SIMDGROUPS
    yg, yu = _kernel("gate_up", spec["bits"], spec["group_size"])(
        inputs=[x.reshape(hidden).astype(mx.float32), *gate, *up, ids.reshape(top).astype(mx.uint32)],
        template=[("T", mx.float32), ("S", gate[1].dtype), ("K", hidden), ("NI", inter),
                  ("RPS", GATE_UP_ROWS), ("NSG", GATE_UP_SIMDGROUPS)],
        grid=(32, GATE_UP_SIMDGROUPS * inter // rows, top), threadgroup=(32, GATE_UP_SIMDGROUPS, 1),
        output_shapes=[(top, inter), (top, inter)], output_dtypes=[mx.float32, mx.float32])
    return yg, yu


def down(h, ids, weights, spec):
    """Raw F32 down rows of every routed slot: [top, hidden]."""
    top, inter = h.shape
    hidden = weights[0].shape[1]
    return _kernel("down", spec["bits"], spec["group_size"])(
        inputs=[h.astype(mx.float32), *weights, ids.reshape(top).astype(mx.uint32)],
        template=[("T", mx.float32), ("S", weights[1].dtype), ("K", inter), ("N", hidden), ("RPS", DOWN_ROWS)],
        grid=(32, top * hidden // DOWN_ROWS, 1), threadgroup=(32, top, 1),
        output_shapes=[(top, hidden)], output_dtypes=[mx.float32])[0]
