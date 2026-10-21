# SPDX-License-Identifier: Apache-2.0
# Mechanism after Kapitsa (nace-ai/kapitsa 8b190f5, Apache-2.0) moe8_plan /
# moe8e expert tiles; arithmetic is mlx2's qwen4_routed_decode transcription
# of MLX quantized.h (MIT).  See provenance/qwen36-moe-expert-grouped-rows.json.
"""Expert-grouped multi-row routed experts for Qwen3.6 verify tiles.

A tree or chain verify of R rows routes R * top_k (row, slot) assignments.
The stock gather (``gather_qmm`` -> ``gather_qmv`` below MLX's sorted-rhs
threshold) runs one quantized matvec per assignment, so an expert chosen by
several rows has its weights streamed once per row.  Kapitsa's verify reads
each distinct expert once for all rows routed to it (``moe8_plan`` builds the
expert tiles on the GPU) and writes per-(row, slot) outputs so the top-k sum
order never depends on the grouping.

Here the same mechanism keeps MLX's per-row arithmetic.  For every distinct
expert, one threadgroup per block of output rows runs MLX's ``qmv_fast`` (or
``qmv``) lane partition, block order and ``simd_sum`` for each member row in
turn, with the expert's weight rows shared across the members.  Each
(row, slot) output is therefore the value ``gather_qmv`` computes for that
assignment alone, independent of which other rows share the tile (row
invariant), and the router-weighted combine stays the stock MLX ops on the
per-slot outputs.

Grouping runs on the GPU with no sort and no host sync: a one-threadgroup
plan launch elects, for every distinct expert, its lowest flat assignment as
leader, ranks the leaders and writes each group's expert and member list
(ascending flat order).  Grid z of the two expert launches enumerates group
slots; slots past the group count exit after one load.  A row picks an
expert at most once, so a group holds at most R members.

Default off; selected per block by ``Qwen36SparseMoeBlock.set_grouped_rows``
(adapter policy key ``moe_grouped_rows``).  Bitwise identity with the stock
gathers is a GPU-checked claim (``scripts/check_qwen36_moe_grouped_rows.py``,
``scripts/probe_qwen36_tree_moe_rows.py``), not an assumption.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

from . import qwen4_routed_decode as RD

# Formats the grouped kernels serve: the routed tables of the Qwen3.6 4-bit
# artifacts are affine g64 at 4 or 6 bits per layer; 8-bit for the Q8 one.
GROUPED_FORMATS = ((4, 64), (6, 64), (8, 64))
# Widest window (rows) a call admits; also the member-array bound.
MAX_ROWS = 32
# Launch geometry: output rows per simdgroup, simdgroups per threadgroup, and
# members served per pass over the shared weight rows.  Geometry never changes
# a member's arithmetic: every output element is one lane-partitioned qmv
# reduction over its own activation row.
DEFAULT_GEOMETRY = {"gu_rps": 2, "gu_nsg": 4, "dn_rps": 4, "dn_nsg": 4, "chunk": 4, "hoist": 0}

# One threadgroup, one thread per assignment: leaders, their ranks, and each
# group's [expert, count, members...] row of ``groups`` (stride 2 + MAXR);
# rows past the group count get expert -1.
PLAN_SOURCE = r"""
    const uint a = thread_position_in_threadgroup.x;
    threadgroup uint flags[BP];
    const uint e = a < uint(B) ? rhs[a] : 0xffffffffu;
    bool lead = a < uint(B);
    for (uint i = 0; lead && i < a; ++i) {
      lead = rhs[i] != e;
    }
    flags[a] = lead ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint rank = 0;
    uint total = 0;
    for (uint i = 0; i < uint(BP); ++i) {
      const uint f = flags[i];
      total += f;
      rank += i < a ? f : 0u;
    }
    if (lead) {
      device int* gp = groups + rank * (2 + MAXR);
      gp[0] = int(e);
      int n = 0;
      for (uint i = a; i < uint(B) && n < MAXR; ++i) {
        if (rhs[i] == e) {
          gp[2 + n] = int(i);
          n++;
        }
      }
      gp[1] = n;
    }
    if (a >= total && a < uint(B)) {
      groups[a * (2 + MAXR)] = -1;
    }
"""

_GROUP = r"""
    const device int* gp = groups + threadgroup_position_in_grid.z * (2 + MAXR);
    const int expert = gp[0];
    if (expert < 0) {
      return;
    }
    const uint e = uint(expert);
    const int count = gp[1];
    const device int* members = gp + 2;
"""

GATE_UP_SOURCE = (
    r"""
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
    const int in_vec_size_g = K / GS;
    const int out_row =
        int(threadgroup_position_in_grid.y) * (NSG * RPS) + int(simd_gid) * RPS;
"""
    + _GROUP
    + r"""
    const size_t row0 = size_t(e) * NI + out_row;
    const int lane_w = int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const int lane_g = int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device uint8_t* gw0 = (const device uint8_t*)wg + row0 * in_vec_size_w + lane_w;
    const device uint8_t* uw0 = (const device uint8_t*)wu + row0 * in_vec_size_w + lane_w;
    const device T* gs0 = sg + row0 * in_vec_size_g + lane_g;
    const device T* gb0 = bg + row0 * in_vec_size_g + lane_g;
    const device T* us0 = su + row0 * in_vec_size_g + lane_g;
    const device T* ub0 = bu + row0 * in_vec_size_g + lane_g;
    const int lane_x = int(simd_lid) * VALUES_PER_THREAD;

    for (int c0 = 0; c0 < count; c0 += CH) {
      const int cn = min(CH, count - c0);
      float result[CH][2 * RPS];
      for (int j = 0; j < CH; ++j) {
        for (int row = 0; row < 2 * RPS; ++row) {
          result[j][row] = 0;
        }
      }
      const device uint8_t* gw = gw0;
      const device uint8_t* uw = uw0;
      const device T* gs = gs0;
      const device T* gb = gb0;
      const device T* us = us0;
      const device T* ub = ub0;
      int k = 0;
      for (; k < (FAST ? K : K - BLOCK_SIZE); k += BLOCK_SIZE) {
        for (int j = 0; j < CH; ++j) {
          if (j < cn) {
            const device T* xp = x + size_t(members[c0 + j] / TOPK) * K + lane_x + k;
            float x_thread[VALUES_PER_THREAD];
            float sum = load_vector<T>(xp, x_thread);
            for (int row = 0; row < 2 * RPS; row++) {
              const bool g = row < RPS;
              const int r = g ? row : row - RPS;
              const device uint8_t* wl = (g ? gw : uw) + r * in_vec_size_w;
              float s = (g ? gs : us)[r * in_vec_size_g];
              float b = (g ? gb : ub)[r * in_vec_size_g];
              result[j][row] += qdot_n(wl, x_thread, s, b, sum, VALUES_PER_THREAD);
            }
          }
        }
        gw += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
        uw += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
        gs += BLOCK_SIZE / GS;
        gb += BLOCK_SIZE / GS;
        us += BLOCK_SIZE / GS;
        ub += BLOCK_SIZE / GS;
      }
      if (!FAST) {
        const int remaining = clamp(
            int(K - k - int(simd_lid) * VALUES_PER_THREAD), 0, VALUES_PER_THREAD);
        if (remaining > 0) {
          for (int j = 0; j < CH; ++j) {
            if (j < cn) {
              const device T* xp = x + size_t(members[c0 + j] / TOPK) * K + lane_x + k;
              float x_thread[VALUES_PER_THREAD];
              float sum = load_vector_safe<T>(xp, x_thread, remaining);
              for (int row = 0; row < 2 * RPS; row++) {
                const bool g = row < RPS;
                const int r = g ? row : row - RPS;
                const device uint8_t* wl = (g ? gw : uw) + r * in_vec_size_w;
                float s = (g ? gs : us)[r * in_vec_size_g];
                float b = (g ? gb : ub)[r * in_vec_size_g];
                result[j][row] += qdot_n(wl, x_thread, s, b, sum, remaining);
              }
            }
          }
        }
      }
      for (int j = 0; j < CH; ++j) {
        if (j < cn) {
          for (int row = 0; row < 2 * RPS; row++) {
            result[j][row] = simd_sum(result[j][row]);
          }
          if (simd_lid == 0) {
            device T* yp = y + size_t(members[c0 + j]) * NI + out_row;
            for (int row = 0; row < RPS; row++) {
              T g = static_cast<T>(result[j][row]);
              T u = static_cast<T>(result[j][row + RPS]);
              T t = g * omlx_mlx_sigmoid<T>(g);
              yp[row] = t * u;
            }
          }
        }
      }
    }
"""
)

DOWN_SOURCE = (
    r"""
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
    const int in_vec_size_g = K / GS;
    const int out_row =
        int(threadgroup_position_in_grid.y) * (NSG * RPS) + int(simd_gid) * RPS;
"""
    + _GROUP
    + r"""
    const size_t row0 = size_t(e) * N + out_row;
    const device uint8_t* ws0 = (const device uint8_t*)w + row0 * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* sc0 = scales + row0 * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* bs0 = biases + row0 * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const int lane_x = int(simd_lid) * VALUES_PER_THREAD;

    for (int c0 = 0; c0 < count; c0 += CH) {
      const int cn = min(CH, count - c0);
      float result[CH][RPS];
      for (int j = 0; j < CH; ++j) {
        for (int row = 0; row < RPS; ++row) {
          result[j][row] = 0;
        }
      }
      const device uint8_t* ws = ws0;
      const device T* sc = sc0;
      const device T* bs = bs0;
      int k = 0;
      for (; k < (FAST ? K : K - BLOCK_SIZE); k += BLOCK_SIZE) {
        for (int j = 0; j < CH; ++j) {
          if (j < cn) {
            const device T* xp = x + size_t(members[c0 + j]) * K + lane_x + k;
            float x_thread[VALUES_PER_THREAD];
            float sum = load_vector<T>(xp, x_thread);
            for (int row = 0; row < RPS; row++) {
              const device uint8_t* wl = ws + row * in_vec_size_w;
              float s = sc[row * in_vec_size_g];
              float b = bs[row * in_vec_size_g];
              result[j][row] += qdot_n(wl, x_thread, s, b, sum, VALUES_PER_THREAD);
            }
          }
        }
        ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
        sc += BLOCK_SIZE / GS;
        bs += BLOCK_SIZE / GS;
      }
      if (!FAST) {
        const int remaining = clamp(
            int(K - k - int(simd_lid) * VALUES_PER_THREAD), 0, VALUES_PER_THREAD);
        if (remaining > 0) {
          for (int j = 0; j < CH; ++j) {
            if (j < cn) {
              const device T* xp = x + size_t(members[c0 + j]) * K + lane_x + k;
              float x_thread[VALUES_PER_THREAD];
              float sum = load_vector_safe<T>(xp, x_thread, remaining);
              for (int row = 0; row < RPS; row++) {
                const device uint8_t* wl = ws + row * in_vec_size_w;
                float s = sc[row * in_vec_size_g];
                float b = bs[row * in_vec_size_g];
                result[j][row] += qdot_n(wl, x_thread, s, b, sum, remaining);
              }
            }
          }
        }
      }
      for (int j = 0; j < CH; ++j) {
        if (j < cn) {
          for (int row = 0; row < RPS; row++) {
            result[j][row] = simd_sum(result[j][row]);
          }
          if (simd_lid == 0) {
            device T* yp = y + size_t(members[c0 + j]) * N + out_row;
            for (int row = 0; row < RPS; row++) {
              yp[row] = static_cast<T>(result[j][row]);
            }
          }
        }
      }
    }
"""
)


# Hoisted decode: each lane converts a weight row's packed codes to floats
# once per block and every member of the group reuses them.  The per-member
# expression is MLX's ``qdot`` term for term (same operands, same order);
# only the integer-to-float conversions, which are exact, are shared.
HOIST_HEADER = r"""
constant constexpr int NWF = BITS == 6 ? VALUES_PER_THREAD / 4 * 6 : VALUES_PER_THREAD;

inline void wconv(const device uint8_t* w, thread float* wf) {
  if (BITS == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < VALUES_PER_THREAD / 4; i++) {
      wf[4 * i] = float(ws[i] & 0x000f);
      wf[4 * i + 1] = float(ws[i] & 0x00f0);
      wf[4 * i + 2] = float(ws[i] & 0x0f00);
      wf[4 * i + 3] = float(ws[i] & 0xf000);
    }
  } else if (BITS == 6) {
    for (int i = 0; i < VALUES_PER_THREAD / 4; i++) {
      const device uint8_t* wp = w + 3 * i;
      wf[6 * i] = float(wp[0] & 0x3f);
      wf[6 * i + 1] = float(wp[0] & 0xc0);
      wf[6 * i + 2] = float(wp[1] & 0x0f);
      wf[6 * i + 3] = float(wp[1] & 0xf0);
      wf[6 * i + 4] = float(wp[2] & 0x03);
      wf[6 * i + 5] = float(wp[2] & 0xfc);
    }
  } else if (BITS == 8) {
    for (int i = 0; i < VALUES_PER_THREAD; i++) {
      wf[i] = float(w[i]);
    }
  }
}

inline float qdot_f(
    const thread float* wf,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum) {
  float accum = 0;
  if (BITS == 4) {
    for (int i = 0; i < VALUES_PER_THREAD / 4; i++) {
      accum +=
          (x_thread[4 * i] * wf[4 * i] +
           x_thread[4 * i + 1] * wf[4 * i + 1] +
           x_thread[4 * i + 2] * wf[4 * i + 2] +
           x_thread[4 * i + 3] * wf[4 * i + 3]);
    }
  } else if (BITS == 6) {
    for (int i = 0; i < VALUES_PER_THREAD / 4; i++) {
      const thread float* xt = x_thread + 4 * i;
      const thread float* w6 = wf + 6 * i;
      accum += w6[0] * xt[0];
      accum += w6[1] * xt[1];
      accum += w6[2] * (xt[1] * 256.0f);
      accum += w6[3] * xt[2];
      accum += w6[4] * (xt[2] * 256.0f);
      accum += w6[5] * xt[3];
    }
  } else if (BITS == 8) {
    for (int i = 0; i < VALUES_PER_THREAD; i++) {
      accum += x_thread[i] * wf[i];
    }
  }
  return scale * accum + sum * bias;
}
"""

GATE_UP_HOIST_SOURCE = (
    r"""
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
    const int in_vec_size_g = K / GS;
    const int out_row =
        int(threadgroup_position_in_grid.y) * (NSG * RPS) + int(simd_gid) * RPS;
"""
    + _GROUP
    + r"""
    const size_t row0 = size_t(e) * NI + out_row;
    const int lane_w = int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const int lane_g = int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device uint8_t* gw0 = (const device uint8_t*)wg + row0 * in_vec_size_w + lane_w;
    const device uint8_t* uw0 = (const device uint8_t*)wu + row0 * in_vec_size_w + lane_w;
    const device T* gs0 = sg + row0 * in_vec_size_g + lane_g;
    const device T* gb0 = bg + row0 * in_vec_size_g + lane_g;
    const device T* us0 = su + row0 * in_vec_size_g + lane_g;
    const device T* ub0 = bu + row0 * in_vec_size_g + lane_g;
    const int lane_x = int(simd_lid) * VALUES_PER_THREAD;

    for (int c0 = 0; c0 < count; c0 += CH) {
      const int cn = min(CH, count - c0);
      float result[CH][2 * RPS];
      for (int j = 0; j < CH; ++j) {
        for (int row = 0; row < 2 * RPS; ++row) {
          result[j][row] = 0;
        }
      }
      for (int k = 0; k < K; k += BLOCK_SIZE) {
        float xt[CH][VALUES_PER_THREAD];
        float xs[CH];
        for (int j = 0; j < CH; ++j) {
          if (j < cn) {
            xs[j] = load_vector<T>(
                x + size_t(members[c0 + j] / TOPK) * K + lane_x + k, xt[j]);
          }
        }
        const int wk = k / BLOCK_SIZE * (BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR);
        const int gk = k / GS;
        for (int row = 0; row < 2 * RPS; row++) {
          const bool g = row < RPS;
          const int r = g ? row : row - RPS;
          float wf[NWF];
          wconv((g ? gw0 : uw0) + r * in_vec_size_w + wk, wf);
          const float s = (g ? gs0 : us0)[r * in_vec_size_g + gk];
          const float b = (g ? gb0 : ub0)[r * in_vec_size_g + gk];
          for (int j = 0; j < CH; ++j) {
            if (j < cn) {
              result[j][row] += qdot_f(wf, xt[j], s, b, xs[j]);
            }
          }
        }
      }
      for (int j = 0; j < CH; ++j) {
        if (j < cn) {
          for (int row = 0; row < 2 * RPS; row++) {
            result[j][row] = simd_sum(result[j][row]);
          }
          if (simd_lid == 0) {
            device T* yp = y + size_t(members[c0 + j]) * NI + out_row;
            for (int row = 0; row < RPS; row++) {
              T g = static_cast<T>(result[j][row]);
              T u = static_cast<T>(result[j][row + RPS]);
              T t = g * omlx_mlx_sigmoid<T>(g);
              yp[row] = t * u;
            }
          }
        }
      }
    }
"""
)

DOWN_HOIST_SOURCE = (
    r"""
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
    const int in_vec_size_g = K / GS;
    const int out_row =
        int(threadgroup_position_in_grid.y) * (NSG * RPS) + int(simd_gid) * RPS;
"""
    + _GROUP
    + r"""
    const size_t row0 = size_t(e) * N + out_row;
    const device uint8_t* ws0 = (const device uint8_t*)w + row0 * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* sc0 = scales + row0 * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* bs0 = biases + row0 * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const int lane_x = int(simd_lid) * VALUES_PER_THREAD;

    for (int c0 = 0; c0 < count; c0 += CH) {
      const int cn = min(CH, count - c0);
      float result[CH][RPS];
      for (int j = 0; j < CH; ++j) {
        for (int row = 0; row < RPS; ++row) {
          result[j][row] = 0;
        }
      }
      for (int k = 0; k < K; k += BLOCK_SIZE) {
        float xt[CH][VALUES_PER_THREAD];
        float xs[CH];
        for (int j = 0; j < CH; ++j) {
          if (j < cn) {
            xs[j] = load_vector<T>(x + size_t(members[c0 + j]) * K + lane_x + k, xt[j]);
          }
        }
        const int wk = k / BLOCK_SIZE * (BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR);
        const int gk = k / GS;
        for (int row = 0; row < RPS; row++) {
          float wf[NWF];
          wconv(ws0 + row * in_vec_size_w + wk, wf);
          const float s = sc0[row * in_vec_size_g + gk];
          const float b = bs0[row * in_vec_size_g + gk];
          for (int j = 0; j < CH; ++j) {
            if (j < cn) {
              result[j][row] += qdot_f(wf, xt[j], s, b, xs[j]);
            }
          }
        }
      }
      for (int j = 0; j < CH; ++j) {
        if (j < cn) {
          for (int row = 0; row < RPS; row++) {
            result[j][row] = simd_sum(result[j][row]);
          }
          if (simd_lid == 0) {
            device T* yp = y + size_t(members[c0 + j]) * N + out_row;
            for (int row = 0; row < RPS; row++) {
              yp[row] = static_cast<T>(result[j][row]);
            }
          }
        }
      }
    }
"""
)


@lru_cache(maxsize=1)
def _plan_kernel():
    return mx.fast.metal_kernel(
        name="qwen36_grouped_plan",
        input_names=["rhs"],
        output_names=["groups"],
        source=PLAN_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=32)
def _kernel(kind: str, bits: int, group_size: int, fast: bool, hoist: bool = False):
    if hoist and not fast:
        raise ValueError("the hoisted grouped kernels take qmv_fast layouts only")
    if kind == "gate_up":
        names = ["x", "wg", "sg", "bg", "wu", "su", "bu", "groups"]
        source = GATE_UP_HOIST_SOURCE if hoist else GATE_UP_SOURCE
    else:
        names = ["x", "w", "scales", "biases", "groups"]
        source = DOWN_HOIST_SOURCE if hoist else DOWN_SOURCE
    header = RD._header(fast=fast, bits=bits, group_size=group_size)
    return mx.fast.metal_kernel(
        name=f"qwen36_grouped_{kind}_q{bits}g{group_size}"
        + ("" if fast else "_qmv")
        + ("_hoist" if hoist else ""),
        input_names=names,
        output_names=["y"],
        header=header + (HOIST_HEADER if hoist else ""),
        source=source,
        ensure_row_contiguous=True,
    )


def _format(layer):
    return (getattr(layer, "bits", None), getattr(layer, "group_size", None))


def admission_refusal(x, indices, gate, up, down) -> str | None:
    """Why the grouped kernels may not serve this call, or ``None``.

    Structural only: the per-row numerics are a GPU-checked claim.
    """
    if not RD.runtime_supported():
        return "Metal runtime unavailable"
    if x.dtype != mx.bfloat16:
        return "grouped rows take bf16 activations"
    hidden = int(x.shape[-1])
    rows = x.size // hidden
    if not 2 <= rows <= MAX_ROWS:
        return f"window of {rows} rows outside 2..{MAX_ROWS}"
    if indices.size != rows * int(indices.shape[-1]):
        return "indices do not cover the window rows"
    fmt = _format(gate)
    if fmt not in GROUPED_FORMATS:
        return f"unsupported routed expert format q{fmt[0]}g{fmt[1]}"
    if _format(up) != fmt or _format(down) != fmt:
        return "routed gate/up/down formats differ"
    for layer in (gate, up, down):
        if getattr(layer, "mode", "affine") != "affine" or layer.get("biases") is None:
            return "grouped rows take affine tables with biases"
        if "bias" in layer:
            return "grouped rows take bias-free expert tables"
        if layer["scales"].dtype != mx.bfloat16:
            return "grouped rows take bf16 scales"
    inter = int(gate.weight.shape[1])
    if int(down.weight.shape[1]) != hidden or gate["scales"].shape[-1] * fmt[1] != hidden:
        return "expert table shapes do not match the activation"
    geo = DEFAULT_GEOMETRY
    if inter % (geo["gu_rps"] * geo["gu_nsg"]) or hidden % (geo["dn_rps"] * geo["dn_nsg"]):
        return "expert table rows do not tile"
    return RD.served_swiglu_refusal()


def grouped_rows(x, indices, gate, up, down, *, geometry=None):
    """Per-slot routed expert outputs ``[rows, top_k, hidden]`` (``x.dtype``).

    Equal, assignment by assignment, to SwitchGLU's ``gather_qmm`` gate/up,
    compiled SwiGLU and ``gather_qmm`` down; callers combine with the stock
    router-weighted sum.  ``geometry`` overrides ``DEFAULT_GEOMETRY`` (launch
    shape only).
    """
    geo = {**DEFAULT_GEOMETRY, **(geometry or {})}
    hidden = int(x.shape[-1])
    rows = x.size // hidden
    top_k = int(indices.shape[-1])
    bits, group_size = _format(gate)
    inter = int(gate.weight.shape[1])
    flat = x.reshape(rows, hidden)
    rhs = indices.reshape(-1).astype(mx.uint32)
    assignments = rows * top_k
    padded = -(-assignments // 32) * 32
    groups = _plan_kernel()(
        inputs=[rhs],
        template=[("B", assignments), ("BP", padded), ("MAXR", rows)],
        grid=(padded, 1, 1),
        threadgroup=(padded, 1, 1),
        output_shapes=[(assignments * (2 + rows),)],
        output_dtypes=[mx.int32],
    )[0]
    chunk = max(1, min(int(geo["chunk"]), rows))
    gu_rps, gu_nsg = int(geo["gu_rps"]), int(geo["gu_nsg"])
    dn_rps, dn_nsg = int(geo["dn_rps"]), int(geo["dn_nsg"])
    if inter % (gu_rps * gu_nsg) or hidden % (dn_rps * dn_nsg):
        raise ValueError("grouped geometry does not tile the expert tables")
    hoist = bool(geo["hoist"]) and RD.qmv_fast_layout(hidden, inter, bits) and RD.qmv_fast_layout(
        inter, hidden, bits
    )
    h = _kernel("gate_up", bits, group_size, RD.qmv_fast_layout(hidden, inter, bits), hoist)(
        inputs=[flat, *RD.expert_operands(gate), *RD.expert_operands(up), groups],
        template=[
            ("T", x.dtype),
            ("K", hidden),
            ("NI", inter),
            ("RPS", gu_rps),
            ("NSG", gu_nsg),
            ("TOPK", top_k),
            ("MAXR", rows),
            ("CH", chunk),
        ],
        grid=(32, inter // gu_rps, assignments),
        threadgroup=(32, gu_nsg, 1),
        output_shapes=[(assignments, inter)],
        output_dtypes=[x.dtype],
    )[0]
    d = _kernel("down", bits, group_size, RD.qmv_fast_layout(inter, hidden, bits), hoist)(
        inputs=[h, *RD.expert_operands(down), groups],
        template=[
            ("T", x.dtype),
            ("K", inter),
            ("N", hidden),
            ("RPS", dn_rps),
            ("NSG", dn_nsg),
            ("MAXR", rows),
            ("CH", chunk),
        ],
        grid=(32, hidden // dn_rps, assignments),
        threadgroup=(32, dn_nsg, 1),
        output_shapes=[(assignments, hidden)],
        output_dtypes=[x.dtype],
    )[0]
    return d.reshape(*indices.shape, hidden)


# ---------------------------------------------------------------------------
# Tensor-unit expert tiles ("lane" law).  NOT bitwise equal to the stock
# gathers: every (row, slot) output follows mlx2's lane-matmul law (MPP
# 16-row fragments, fp32 group products in a fixed K-slice order, the
# epilogue ``s * P + b * sum(x)``), the law the dense Qwen3.8 tree tiles
# already run.  It is row invariant (a row's bits never depend on the other
# rows of its expert tile or of the window) and reads each distinct expert's
# weights once per output tile, but it is a different numerical law from the
# one-row decode, so it is a separately selected, unqualified candidate.
# ---------------------------------------------------------------------------

# One threadgroup: groups as [expert, start, count] in leader order (expert -1
# past the group count), ``tok`` the activation row of every sorted position
# and ``inv`` the sorted position of every flat assignment.
LANE_PLAN_SOURCE = r"""
    const uint a = thread_position_in_threadgroup.x;
    threadgroup uint flags[BP];
    threadgroup uint counts[BP];
    const uint e = a < uint(B) ? rhs[a] : 0xffffffffu;
    bool lead = a < uint(B);
    for (uint i = 0; lead && i < a; ++i) {
      lead = rhs[i] != e;
    }
    uint n = 0;
    if (lead) {
      for (uint i = a; i < uint(B); ++i) {
        n += rhs[i] == e ? 1u : 0u;
      }
    }
    flags[a] = lead ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint rank = 0;
    uint total = 0;
    for (uint i = 0; i < uint(BP); ++i) {
      const uint f = flags[i];
      total += f;
      rank += i < a ? f : 0u;
    }
    if (lead) {
      counts[rank] = n;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lead) {
      uint start = 0;
      for (uint r = 0; r < rank; ++r) {
        start += counts[r];
      }
      grp[rank * 3] = int(e);
      grp[rank * 3 + 1] = int(start);
      grp[rank * 3 + 2] = int(n);
      uint j = 0;
      for (uint i = a; i < uint(B); ++i) {
        if (rhs[i] == e) {
          tok[start + j] = i / uint(TOPK);
          inv[i] = start + j;
          j++;
        }
      }
    }
    if (a >= total && a < uint(B)) {
      grp[a * 3] = -1;
    }
"""

_LANE_GROUP_HEADER = r"""  const device int* gq = grp + threadgroup_position_in_grid.y * 3;
  const int expert = gq[0];
  if (expert < 0) {
    return;
  }
  const int gstart = gq[1];
  const int M = gq[2];
  const device XT* Xg = (const device XT*)X + (int64_t)gstart * K;
  const device uchar* We = (const device uchar*)W + (int64_t)expert * N * (K * BITS / 8);
  const device ST* Se = (const device ST*)SC + (int64_t)expert * N * (K / GS);
  const device ST* Be = (const device ST*)BI + (int64_t)expert * N * (K / GS);
"""

_LANE_SB_LOAD = r"""
    float s[NF][4], bb[NF][4];
    for (int f = 0; f < NF; f++) {
      for (int j = 0; j < 4; j++) {
        const int64_t at = (int64_t)(n0 + f * 16 + fn + j) * (K / GS) + g;
        s[f][j] = colok[f] ? float(Se[at]) : 0.0f;
        bb[f][j] = colok[f] ? float(Be[at]) : 0.0f;
      }
    }
"""


def _patch(source: str, old: str, new: str, count: int | None = 1) -> str:
    found = source.count(old)
    if (count is None and found == 0) or (count is not None and found != count):
        raise AssertionError(f"lane expert anchor changed: {old!r} x{found}")
    return source.replace(old, new)


def lane_expert_source(bits: int, group_size: int) -> str:
    """The lane-matmul main body re-addressed to one expert of a table and a
    contiguous group of sorted activation rows (patched, anchors asserted)."""
    from ..lane import matmul as L

    register_b = L.REGISTER_B[0] and (bits, group_size) in L.REGISTER_B_FORMATS
    src = L.main_source(bits, group_size, register_b)
    # Re-address the dense operands first; the group header (which itself
    # reads W, X, SC and BI) goes in last so no rewrite can touch it.
    src = _patch(src, "  const int rb = threadgroup_position_in_grid.y * 16 * TMR;", "  const int rb = 0;")
    src = _patch(src, "(device XT*)X + (int64_t)rb * K", "(device XT*)Xg + (int64_t)rb * K")
    src = _patch(src, "(const device XT*)X + (int64_t)m0 * K", "Xg + (int64_t)m0 * K")
    src = _patch(src, "(const device XT*)X + (int64_t)(m0 + 8) * K", "Xg + (int64_t)(m0 + 8) * K")
    src = _patch(src, "(const device uchar*)W", "We", None) if "(const device uchar*)W" in src else src
    src = _patch(src, "(device WP*)W", "(device WP*)We") if "(device WP*)W" in src else src
    src = _patch(src, L._SB_DECL.split("\n")[1] + "\n", "")
    src = _patch(src, L._SB_LOAD, _LANE_SB_LOAD)
    src = _patch(src, "Y[m * N + n + j]", "Y[(int64_t)(gstart + m) * N + n + j]")
    if "SBt" in src or "(const device uchar*)W" in src or "(device WP*)W " in src:
        raise AssertionError("lane expert source still addresses the dense operands")
    src = _patch(src, "  const int M = mdims[0], MP = mdims[1];\n", _LANE_GROUP_HEADER)
    if "mdims" in src:
        raise AssertionError("lane expert source still reads the dense row count")
    return src


@lru_cache(maxsize=1)
def _lane_plan_kernel():
    return mx.fast.metal_kernel(
        name="qwen36_grouped_lane_plan",
        input_names=["rhs"],
        output_names=["grp", "tok", "inv"],
        source=LANE_PLAN_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=16)
def _lane_kernel(bits: int, group_size: int):
    import hashlib

    from ..lane import matmul as L

    source = lane_expert_source(bits, group_size)
    wtype = L._NATIVE.get(bits, "uchar")
    header = (L._HEADER + f"typedef bfloat XT;\ntypedef {wtype} WT;\ntypedef uchar WP;\n"
              f"typedef bfloat ST;\nconstexpr constant int BITS = {bits};\n")
    digest = hashlib.sha256((header + source).encode()).hexdigest()[:16]
    return mx.fast.metal_kernel(
        name=f"qwen36_lane_expert_q{bits}g{group_size}_{digest}",
        input_names=["X", "W", "SC", "BI", "grp"],
        output_names=["Y"],
        header=header,
        source=source,
        ensure_row_contiguous=True,
    )


def lane_refusal(gate, up, down) -> str | None:
    """Why the tensor-unit expert tiles may not run here, or ``None``."""
    from ..lane import matmul as L

    if L.backend() != "mpp":
        return "lane expert tiles need the M5 tensor units"
    for layer in (gate, up, down):
        if int(layer.bits) not in L.QUANT_BITS or int(layer.group_size) not in L.GROUP_SIZES:
            return f"no lane format for q{layer.bits}g{layer.group_size}"
        if layer["weight"].dtype != mx.uint32:
            return "packed expert weights must be uint32"
    return None


def _lane_project(xs, layer, grp, rows_cap, split_k=None):
    from ..lane import matmul as L

    bits, group_size = int(layer.bits), int(layer.group_size)
    n = int(layer["weight"].shape[1])
    k = int(xs.shape[-1])
    sk = L.split_k(n, k, group_size, bits) if split_k is None else int(split_k)
    slots = int(xs.shape[0])
    return _lane_kernel(bits, group_size)(
        inputs=[xs, layer["weight"], layer["scales"], layer["biases"], grp],
        template=[("TMR", -(-rows_cap // 16)), ("N", n), ("K", k), ("NT", L.NT),
                  ("SK", sk), ("GS", group_size)],
        grid=(-(-n // L.NT) * 32 * sk, slots, 1),
        threadgroup=(32 * sk, 1, 1),
        output_shapes=[(slots, n)],
        output_dtypes=[xs.dtype],
    )[0]


def lane_grouped_rows(x, indices, gate, up, down, *, split_k=None):
    """Per-slot routed expert outputs under the lane law (see above)."""
    from .activations import swiglu

    hidden = int(x.shape[-1])
    rows = x.size // hidden
    top_k = int(indices.shape[-1])
    rhs = indices.reshape(-1).astype(mx.uint32)
    assignments = rows * top_k
    padded = -(-assignments // 32) * 32
    grp, tok, inv = _lane_plan_kernel()(
        inputs=[rhs],
        template=[("B", assignments), ("BP", padded), ("TOPK", top_k)],
        grid=(padded, 1, 1),
        threadgroup=(padded, 1, 1),
        output_shapes=[(assignments * 3,), (assignments,), (assignments,)],
        output_dtypes=[mx.int32, mx.uint32, mx.uint32],
    )
    xs = mx.take(x.reshape(rows, hidden), tok, axis=0)
    g = _lane_project(xs, gate, grp, rows, split_k)
    u = _lane_project(xs, up, grp, rows, split_k)
    h = swiglu(g, u)
    d = _lane_project(h, down, grp, rows, split_k)
    return mx.take(d, inv, axis=0).reshape(*indices.shape, hidden)


def plan(indices) -> list[tuple[int, int, list[int]]]:
    """Host transcription of the plan launch: ``(leader, expert, members)``.

    Groups are in leader order; members are flat (row, slot) assignment ids
    in ascending order and the leader is the lowest one.  Used by CPU tests
    and receipts, never by serving.
    """
    flat = [int(v) for v in indices.reshape(-1).tolist()]
    groups, seen = [], set()
    for a, e in enumerate(flat):
        if e in seen:
            continue
        seen.add(e)
        groups.append((a, e, [i for i in range(a, len(flat)) if flat[i] == e]))
    return groups


__all__ = [
    "DEFAULT_GEOMETRY",
    "GROUPED_FORMATS",
    "MAX_ROWS",
    "admission_refusal",
    "grouped_rows",
    "plan",
]
