# SPDX-License-Identifier: Apache-2.0
# Mechanism (an expert-grouped plan, then products that read each distinct
# expert once for every row that chose it) from MTPLX (Apache-2.0,
# https://github.com/youssofal/MTPLX @ 3890be556160af48887d5bfbf5304c4ad5b9417f,
# mtplx/models/inkling/fast/experts.py); no MTPLX code copied.  The per-pair
# lane orders reproduce MLX 0.32.4 (MIT) fp_qmv_impl / fp_qmv_fast_impl.
# See docs/PROVENANCE.md and provenance/inkling2-mtplx-moe-panel.json.
"""``inkling2_moe_panel_kernel``: the routed-expert forward of a 2-32 row
Inkling 2 ``SparseMoE`` call (a verify panel) with each distinct expert read
once (default off).

Today a multi-row call runs MLX's gathered product per (row, expert) pair:
below MLX's streaming floor (``B >= 16`` and ``B / E >= 4`` sorted rows) every
pair is its own ``gather_qmv`` threadgroup set that re-reads its expert's
codes, so a 4-row verify panel at top-8 reads 32 expert weight sets per
visit although it names only about 19 distinct experts (MTPLX's figure; 8
rows: 64 pair reads).  Here, after the shipped router ops:

1. ``plan`` (one threadgroup): from the first ``top_k`` entries of every row's
   order, each expert's member pairs (``row * top_k + k``) and the list of the
   panel's distinct experts in ascending id.
2. ``gate_up``: one threadgroup per (distinct expert, 4 output rows), two
   simdgroups (the 4 gate rows and the 4 up rows of the stacked table, as
   ``moe_decode_kernel``); each weight word and group scale is read and
   decoded once and multiplied into every member row of that expert, in up
   to ``MEMBERS_PER_PASS`` members at a time.  Epilogue: the gather's BF16
   rounding, the NVFP4 per-expert ``weight_scale_2`` law (MXFP8 has none) and
   the softcapped SwiGLU, written per pair.
3. ``down``: one threadgroup per (distinct expert, 8 output rows), the same
   member loop over the pairs' activation rows; BF16 rounding and the down
   scale law, written per pair.
4. ``combine``: per (row, column) the routing normalization and the weighted
   8-way sum in the order ``moe_decode_kernel`` replays (MLX's
   ``row_reduce_small`` / ``col_reduce_small``), BF16 out.

Exactness target: bitwise equal to the shipped fused route (``moe_fast``,
W4A16: BF16 activations, no activation quantization) at the same width.  The
argument: at 2-32 rows the shipped route never streams (it would only at
``B / E >= 4``, declined below), so each pair's product is ``fp_gather_qmv``
-> ``fp_qmv_impl`` (or ``fp_qmv_fast_impl`` where ``N % 8 == 0`` and ``K`` is
a multiple of the fast block), whose per-pair arithmetic does not depend on
how many pairs the launch holds.  This module repeats that arithmetic per
pair: the same lane slices (values per lane, block size, the K tail), the
same statement forms (``qdot``: FP4 quads, FP8 one product per value,
``scale * accum`` then the add into the lane's sum) and ``simd_sum``.
Decoding a weight once and using it for several rows changes nothing: the
decoded value is the same float.  The router is the shipped op sequence at
the shipped shape (``moe_decode_kernel.route_compiled``).

Two MLX dispatch details cannot be read from Python, so they are module
switches the Metal parity test sweeps (as ``moe_decode_kernel`` does):

- ``QDOT_CONTRACT`` (shared with ``moe_decode_kernel``): the FP contraction
  mode of the dot products.
- ``FAST_K_RULE``: when MLX picks ``gather_qmv_fast``.  ``"bits"`` (default)
  is the bits-aware alignment ``(32 / bits) * 2 * 32`` of MLX 0.32.2+ and
  0.32.5 source (512 for FP4, 256 for FP8: so the MXFP8 gate/up product at
  K = 2304 is the fast kernel, as MTPLX records for 0.32.2); ``"fixed512"``
  is the ``K % 512`` rule of 0.32.0/0.32.1 sources.

Declines (counted in ``moe_fast_counts`` as ``panel_kernel_declined:<why>``;
the caller runs the shipped route): not on the GPU; fewer than 2 or more
than ``MAX_ROWS`` rows; non-BF16 activations; act-quant on; experts other
than the stacked NVFP4 or MXFP8 tables; top_k outside 1..8; more than 1024
experts; widths the tiles do not cover; and any call where the shipped route
would take MLX's streaming ``gather_qmm_rhs`` (natively or through the rhs
pad), whose order this module does not reproduce (``rhs_route``,
``pad_policy``); and, unless ``PANEL_OVER_WEIGHTED_SUM``, the
calls the shipped route combines through
``qwen4_moe_weighted_sum`` (``weighted_sum_order_unproven``).

Counters: an engaged call counts ``panel_kernel``, ``panel_kernel:<fmt>``
and ``fused`` (so the receipt's ``moe_fused_calls`` still counts every
fused-route call), and the switch's sort receipt gets a ``panel_kernel``
bucket (the panel neither sorts nor gathers).
"""

from __future__ import annotations

import struct
import zlib

import mlx.core as mx

from . import moe_decode_kernel as _mdk

MAX_ROWS = 32
MAX_TOP_K = 8
MAX_EXPERTS = 1024
MEMBERS_PER_PASS = 4
# Calls the shipped route combines through qwen4_moe_weighted_sum
# (``shipped_weighted_sum_engages``: inkling2_moe_weighted_sum on, a sorted
# multi-token call, the GPU, top_k in its verified set and at least its
# MIN_ROUTED_ROWS routed rows, i.e. B8+ at the Mini's top 8); the panel's
# combine replays moe_decode_kernel's col_reduce order instead.  False
# (default): such calls decline (``weighted_sum_order_unproven``) and keep
# the weighted-sum route.  True: the panel takes them.
# GPU evidence (run w1122-3ee59f64e-mtplx-smoke-moe-a1, main 3ee59f64e, M5
# Max, MLX 0.32.4, tests/test_inkling2_moe_panel_kernel_metal.py real
# shape): the panel was bitwise against the shipped weighted-sum route at
# B8/B16/B32, NVFP4 and MXFP8 (max |diff| 0.0), and the checkpoint's 8-row
# verify forward was bitwise with this on.  The default stays False until
# the full test (Pierre's call).
PANEL_OVER_WEIGHTED_SUM = False
FAST_K_RULE = "bits"  # "bits" (MLX 0.32.2+ / 0.32.5) or "fixed512" (0.32.0/0.32.1)
_FAST_RULES = ("bits", "fixed512")
FORMATS = ("nvfp4", "mxfp8")
# (bits, group size) of the two expert formats.
_GEOMETRY = {"nvfp4": (4, 16), "mxfp8": (8, 32)}

route_compiled = _mdk.route_compiled


def _f32_literal(value):
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    return f"as_type<float>(0x{bits:08x}u)"


def fast_qmv(bits, n_out, k, rule=None):
    """Whether MLX's ``gather_qmv`` dispatch picks ``gather_qmv_fast``."""
    rule = FAST_K_RULE if rule is None else rule
    if rule not in _FAST_RULES:
        raise ValueError(f"moe_panel_kernel: fast rule {rule!r}")
    align = 512 if rule == "fixed512" else (32 // bits) * 2 * 32
    return n_out % 8 == 0 and k % align == 0


def _header(contract):
    pragma = "" if contract == "default" else f"#pragma METAL fp contract({contract})\n"
    return pragma + r"""
// MLX fp4_e2m1::operator float16_t (fp4.h), via Dequantize<4>.
inline float mlx2p_dq4(uint8_t bits) {
  half converted = as_type<half>(ushort((bits & 7) << 9));
  converted *= 16384.0;
  return static_cast<float>(bits & 8 ? -converted : converted);
}

// MLX fp8_e4m3::operator float16_t (fp8.h), via Dequantize<8> and
// dequantize_scale<float, 16>.
inline float mlx2p_e4m3(uint8_t bits) {
  uint16_t v = bits & 127;
  uint16_t sign_bit = ((uint16_t)((bits >> 7) & 1)) << 15;
  uint16_t u = (v << 7) | (((v + 1) >> 7) << 14) | sign_bit;
  half converted = as_type<half>(u);
  return static_cast<float>(converted * 256.0);
}

// MLX fp8_e8m0::operator float (fp8.h), via dequantize_scale<float, 32>.
inline float mlx2p_e8m0(uint8_t bits) {
  uint32_t out = (bits == 0 ? 0x400000u : (static_cast<uint32_t>(bits) << 23));
  return as_type<float>(out);
}

template <int GS>
inline float mlx2p_scale(uint8_t s) {
  if (GS == 16) {
    return mlx2p_e4m3(s);
  } else {
    return mlx2p_e8m0(s);
  }
}

// One lane's VPT weight values of one row, decoded as MLX's qdot reads them
// (FP4: uint16 words, four nibbles each; FP8: one byte a value).
template <int BITS, int VPT>
inline void mlx2p_decode(const device uint8_t* w, thread float* d) {
  if (BITS == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < VPT / 4; i++) {
      d[4 * i] = mlx2p_dq4(ws[i]);
      d[4 * i + 1] = mlx2p_dq4(ws[i] >> 4);
      d[4 * i + 2] = mlx2p_dq4(ws[i] >> 8);
      d[4 * i + 3] = mlx2p_dq4(ws[i] >> 12);
    }
  } else {
    for (int i = 0; i < VPT; i++) {
      d[i] = mlx2p_e4m3(w[i]);
    }
  }
}

// MLX qdot / qdot_safe<float, VPT, BITS> on decoded values (the tail is
// always a whole lane slice here: K is a multiple of the group size).
template <int BITS, int VPT>
inline float mlx2p_qdot(const thread float* d, const thread float* x, float scale) {
  float accum = 0;
  if (BITS == 4) {
    for (int i = 0; i < (VPT / 4); i++) {
      accum +=
          (x[4 * i] * d[4 * i] +
           x[4 * i + 1] * d[4 * i + 1] +
           x[4 * i + 2] * d[4 * i + 2] +
           x[4 * i + 3] * d[4 * i + 3]);
    }
  } else {
    for (int i = 0; i < VPT; i++) {
      accum += x[i] * d[i];
    }
  }
  return scale * accum;
}

// MLX fp_qmv_impl (FAST false: packs_per_thread 1, blocks while
// k < K - block, then a tail) or fp_qmv_fast_impl (FAST true: packs_per_thread
// 2, K a multiple of the block) for 4 consecutive weight rows of one expert
// and nm <= NM member vectors (x + xoff[m]); result[m][row] is the lane's sum
// before simd_sum.  ``w`` / ``s`` point at the first row.
template <int BITS, int GS, bool FAST, int KDIM, int NM>
inline void mlx2p_rows4(const device uint8_t* w, const device uint8_t* s,
                        const device bfloat16_t* x, const thread uint* xoff, uint nm,
                        uint lid, thread float (&result)[NM][4]) {
  constexpr int PACK = 32 / BITS;
  constexpr int VPT = PACK * (FAST ? 2 : 1);
  constexpr int BLOCK = VPT * 32;
  constexpr int ROW_BYTES = KDIM * BITS / 8;
  constexpr int ROW_GROUPS = KDIM / GS;
  constexpr int SCALE_STEP = GS / VPT;
  w += lid * (VPT * BITS / 8);
  s += lid / SCALE_STEP;
  uint xo = lid * VPT;
  float xt[NM][VPT];
  float d[VPT];
  int k = 0;
  const int full_end = FAST ? KDIM : KDIM - BLOCK;
  for (; k < full_end; k += BLOCK) {
    for (uint m = 0; m < nm; m++) {
      for (int i = 0; i < VPT; i++) xt[m][i] = x[xoff[m] + xo + i];
    }
    for (int row = 0; row < 4; row++) {
      const float sc = mlx2p_scale<GS>(s[row * ROW_GROUPS]);
      mlx2p_decode<BITS, VPT>(w + row * ROW_BYTES, d);
      for (uint m = 0; m < nm; m++) {
        result[m][row] += mlx2p_qdot<BITS, VPT>(d, xt[m], sc);
      }
    }
    w += BLOCK * BITS / 8;
    s += BLOCK / GS;
    xo += BLOCK;
  }
  if (!FAST) {
    const int remaining = clamp(static_cast<int>(KDIM - k - lid * VPT), 0, VPT);
    if (remaining > 0) {
      for (uint m = 0; m < nm; m++) {
        for (int i = 0; i < VPT; i++) xt[m][i] = x[xoff[m] + xo + i];
      }
      for (int row = 0; row < 4; row++) {
        const float sc = mlx2p_scale<GS>(s[row * ROW_GROUPS]);
        mlx2p_decode<BITS, VPT>(w + row * ROW_BYTES, d);
        for (uint m = 0; m < nm; m++) {
          result[m][row] += mlx2p_qdot<BITS, VPT>(d, xt[m], sc);
        }
      }
    }
  }
}
""" + r"""
#pragma METAL fp contract(off)

// The gather's BF16 rounding, then (NVFP4) moe_fast._scale_law: FP32
// multiply of the BF16 product by the per-expert scale, BF16 round.
template <int HAS_SCALE2>
inline float mlx2p_round(float raw, float scale) {
  bfloat16_t y = static_cast<bfloat16_t>(raw);
  if (HAS_SCALE2) {
    return static_cast<float>(static_cast<bfloat16_t>(static_cast<float>(y) * scale));
  }
  return static_cast<float>(y);
}

// MLX Sigmoid (unary_ops.h).
inline float mlx2p_sigmoid(float x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}

// moe_fast._swiglu_law on BF16-valued inputs.
inline float mlx2p_swiglu(float up, float gate, float gate_cap, float up_cap) {
  float a = metal::precise::tanh(gate / gate_cap);
  a = a * gate_cap;
  a = a * mlx2p_sigmoid(gate);
  a = a * metal::precise::tanh(up / up_cap);
  return a * up_cap;
}

inline float mlx2p_add(float a, float b) { return a + b; }
"""


_PLAN_SOURCE = r"""
    const uint e = thread_position_in_threadgroup.x;
    threadgroup uint used[NE];
    if (e < NE) {
      uint c = 0;
      for (uint t = 0; t < W; t++) {
        for (uint k = 0; k < TOPK; k++) {
          if (order[t * OS + k] == e) {
            plan[1 + 2 * NE + e * W + c] = t * TOPK + k;
            c++;
          }
        }
      }
      plan[1 + NE + e] = c;
      used[e] = c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == 0) {
      uint n = 0;
      for (uint i = 0; i < NE; i++) {
        if (used[i] != 0) {
          plan[1 + n] = i;
          n++;
        }
      }
      for (uint i = n; i < NE; i++) plan[1 + i] = 0;
      plan[0] = n;
    }
"""

_GATE_UP_SOURCE = r"""
    const uint tile = threadgroup_position_in_grid.x;
    const uint slot = threadgroup_position_in_grid.y;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lid = thread_index_in_simdgroup;
    if (slot >= plan[0]) return;
    const uint e = plan[1 + slot];
    const uint count = plan[1 + NE + e];
    const device uint32_t* mem = plan + 1 + 2 * NE + e * W;
    const uint row0 = sg * I + tile * 4;
    const size_t first = (size_t)e * (2 * I) + row0;
    const device uint8_t* wr = (const device uint8_t*)w + first * (H * BITS / 8);
    const device uint8_t* sr = s + first * (H / GS);
    const float sc2 = HAS_SCALE2 ? scale2[e * 2] : 1.0f;
    threadgroup float up_vals[NM * 4];
    for (uint m0 = 0; m0 < count; m0 += NM) {
      const uint nm = min((uint)NM, count - m0);
      uint xoff[NM];
      for (uint m = 0; m < NM; m++) xoff[m] = m < nm ? (mem[m0 + m] / TOPK) * H : 0;
      float result[NM][4];
      for (uint m = 0; m < NM; m++) {
        for (int r = 0; r < 4; r++) result[m][r] = 0;
      }
      mlx2p_rows4<BITS, GS, (FAST != 0), H, NM>(wr, sr, x, xoff, nm, lid, result);
      for (uint m = 0; m < nm; m++) {
        for (int r = 0; r < 4; r++) result[m][r] = simd_sum(result[m][r]);
      }
      if (sg == 1 && lid < 4) {
        for (uint m = 0; m < nm; m++) {
          up_vals[m * 4 + lid] = mlx2p_round<HAS_SCALE2>(result[m][lid], sc2);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0 && lid < 4) {
        for (uint m = 0; m < nm; m++) {
          float gate = mlx2p_round<HAS_SCALE2>(result[m][lid], sc2);
          hidden[(size_t)mem[m0 + m] * I + tile * 4 + lid] = static_cast<bfloat16_t>(
              mlx2p_swiglu(up_vals[m * 4 + lid], gate, GATE_CAP, UP_CAP));
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

_DOWN_SOURCE = r"""
    const uint tile = threadgroup_position_in_grid.x;
    const uint slot = threadgroup_position_in_grid.y;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lid = thread_index_in_simdgroup;
    if (slot >= plan[0]) return;
    const uint e = plan[1 + slot];
    const uint count = plan[1 + NE + e];
    const device uint32_t* mem = plan + 1 + 2 * NE + e * W;
    const uint row0 = tile * 8 + sg * 4;
    const size_t first = (size_t)e * H + row0;
    const device uint8_t* wr = (const device uint8_t*)w + first * (I * BITS / 8);
    const device uint8_t* sr = s + first * (I / GS);
    const float sc2 = HAS_SCALE2 ? scale2[e * 2 + 1] : 1.0f;
    for (uint m0 = 0; m0 < count; m0 += NM) {
      const uint nm = min((uint)NM, count - m0);
      uint xoff[NM];
      for (uint m = 0; m < NM; m++) xoff[m] = m < nm ? mem[m0 + m] * I : 0;
      float result[NM][4];
      for (uint m = 0; m < NM; m++) {
        for (int r = 0; r < 4; r++) result[m][r] = 0;
      }
      mlx2p_rows4<BITS, GS, (FAST != 0), I, NM>(wr, sr, hidden, xoff, nm, lid, result);
      for (uint m = 0; m < nm; m++) {
        for (int r = 0; r < 4; r++) result[m][r] = simd_sum(result[m][r]);
      }
      if (lid < 4) {
        for (uint m = 0; m < nm; m++) {
          part[(size_t)mem[m0 + m] * H + row0 + lid] = static_cast<bfloat16_t>(
              mlx2p_round<HAS_SCALE2>(result[m][lid], sc2));
        }
      }
    }
"""

_COMBINE_SOURCE = r"""
    const uint j = thread_position_in_grid.x;
    const uint t = thread_position_in_grid.y;
    if (j >= H) return;
    const device uint32_t* ord = order + t * OS;
    const device float* sc = scores + t * OS;
    float total_s = 0;
    if (ROUTE_HALVES) {
      float a = 0, b = 0;
      for (int k = 0; k < TOPK / 2; k++) a = mlx2p_add(sc[ord[k]], a);
      for (int k = TOPK / 2; k < TOPK; k++) b = mlx2p_add(sc[ord[k]], b);
      total_s = mlx2p_add(a, b);
    } else {
      for (int k = 0; k < TOPK; k++) total_s = mlx2p_add(sc[ord[k]], total_s);
    }
    const float denom = mlx2p_add(total_s, EPS);
    float total = 0;
    for (int k = 0; k < TOPK; k++) {
      float weight = sc[ord[k]] / denom;
      weight = weight * ROUTE_SCALE;
      float p = static_cast<float>(part[((size_t)t * TOPK + k) * H + j]) * weight;
      total = mlx2p_add(p, total);
    }
    y[(size_t)t * H + j] = static_cast<bfloat16_t>(total);
"""

_kernels: dict = {}


def _kernel(kind, contract, route_sum, caps):
    key = (kind, contract, route_sum, caps)
    kernel = _kernels.get(key)
    if kernel is not None:
        return kernel
    gate_cap, up_cap, route_scale = caps
    consts = (f"#define GATE_CAP {_f32_literal(gate_cap)}\n"
              f"#define UP_CAP {_f32_literal(up_cap)}\n"
              f"#define ROUTE_SCALE {_f32_literal(route_scale)}\n"
              f"#define EPS {_f32_literal(1e-20)}\n"
              f"#define ROUTE_HALVES {1 if route_sum == 'halves' else 0}\n")
    header = consts + _header(contract)
    # The caps are compiled in: name them, so two cap sets never share a
    # kernel name (and a cached library).
    tag = f"{contract}_{route_sum}_{zlib.crc32(consts.encode()):08x}"
    if kind == "plan":
        kernel = mx.fast.metal_kernel(
            name="inkling2_moe_panel_plan", input_names=["order"], output_names=["plan"],
            source=_PLAN_SOURCE)
    elif kind == "gate_up":
        kernel = mx.fast.metal_kernel(
            name=f"inkling2_moe_panel_gate_up_{tag}",
            input_names=["x", "plan", "w", "s", "scale2"], output_names=["hidden"],
            source=_GATE_UP_SOURCE, header=header)
    elif kind == "down":
        kernel = mx.fast.metal_kernel(
            name=f"inkling2_moe_panel_down_{tag}",
            input_names=["hidden", "plan", "w", "s", "scale2"], output_names=["part"],
            source=_DOWN_SOURCE, header=header)
    else:
        kernel = mx.fast.metal_kernel(
            name=f"inkling2_moe_panel_combine_{tag}",
            input_names=["part", "order", "scores"], output_names=["y"],
            source=_COMBINE_SOURCE, header=header)
    _kernels[key] = kernel
    return kernel


def _count(moe, key):
    counts = moe.__dict__.setdefault("moe_fast_counts", {})
    counts[key] = counts.get(key, 0) + 1


def device_ok():
    """The kernels run on the GPU only (CPU tests patch this together with
    ``launch`` to run the plumbing against ``law``)."""
    return mx.default_device() == mx.gpu


def expert_format(experts):
    """``"nvfp4"`` / ``"mxfp8"`` for the stacked tables this module reads,
    else None."""
    from ...nvfp4_linear import ScaledNVFP4SwitchLinear
    from ..switch_layers import QuantizedSwitchLinear

    if experts is None:
        return None
    gate_up, down = experts.gate_up, experts.down
    if (type(gate_up) is ScaledNVFP4SwitchLinear and type(down) is ScaledNVFP4SwitchLinear
            and experts.scales is not None):
        return "nvfp4"
    if (type(gate_up) is QuantizedSwitchLinear and type(down) is QuantizedSwitchLinear
            and experts.scales is None
            and all(m.mode == "mxfp8" and m.bits == 8 and m.group_size == 32
                    and m.get("biases") is None and "bias" not in m
                    for m in (gate_up, down))):
        return "mxfp8"
    return None


def _shipped_sorted(moe, rows, token_length):
    """The shipped fused route's sort decision for this call (no counter)."""
    from .. import switch_layers as sl

    sort, _ = sl._switch_sort_decision(
        moe.switch_mlp._sort_policy, assignments=rows * int(moe.top_k),
        token_length=token_length, invariant=False)
    return sort


def _shipped_streams(moe, experts, fmt, rows, token_length):
    """Whether the shipped fused route would run MLX's streaming
    ``gather_qmm_rhs`` (natively or through the rhs pad) for this call: its
    order is not the per-pair ``gather_qmv`` order this module replays."""
    from .. import switch_layers as sl

    n = rows * int(moe.top_k)
    if not _shipped_sorted(moe, rows, token_length):
        return None
    num_experts = int(experts.gate_up["weight"].shape[0])
    if n >= sl._RHS_MIN_ROWS and n // num_experts >= sl._RHS_ROWS_PER_EXPERT:
        return "rhs_route"
    if fmt == "mxfp8" and sl._RHS_PAD_MIN_ROWS_PER_EXPERT:
        # QuantizedSwitchLinear pads sorted gathers toward the streaming
        # kernel; ScaledNVFP4SwitchLinear has no pad.
        if sl._RHS_PAD_POLICY != "floor":
            return "pad_policy"
        if sl._rhs_stream_pad(n, num_experts):
            return "rhs_route"
    return None


def decline_reason(moe, experts, x, weighted_sum=False):
    """None when the kernels take ``moe(x)``; ``weighted_sum`` is the
    caller's ``inkling2_moe_weighted_sum`` request (see
    ``PANEL_OVER_WEIGHTED_SUM``)."""
    if not device_ok():
        return "not_gpu"
    hidden = int(x.shape[-1])
    rows = int(x.size) // hidden if hidden else 0
    if not 2 <= rows <= MAX_ROWS:
        return "rows"
    if x.dtype != mx.bfloat16:
        return "dtype"
    if moe.__dict__.get("_act_quant"):
        return "act_quant"
    fmt = expert_format(experts)
    if fmt is None:
        return "expert_format"
    top_k = int(moe.top_k)
    if not 1 <= top_k <= MAX_TOP_K or (_mdk.ROUTE_SUM == "halves" and top_k % 2):
        # The combine replays MLX col_reduce_small with one y-thread per
        # routed row (top_k <= 8), as moe_decode_kernel.
        return "top_k"
    num_experts = int(experts.gate_up["weight"].shape[0])
    if not top_k <= num_experts <= MAX_EXPERTS:
        return "experts"
    _, group = _GEOMETRY[fmt]
    inter = int(experts.down["scales"].shape[-1]) * group
    if (hidden % group or inter % group or hidden % 8 or inter % 4
            or hidden < 8 or inter < 8
            or int(experts.gate_up["weight"].shape[1]) != 2 * inter
            or int(experts.gate_up["scales"].shape[-1]) * group != hidden
            or int(experts.down["weight"].shape[1]) != hidden):
        return "geometry"
    token_length = int(x.shape[-2]) if x.ndim >= 2 else 1
    streams = _shipped_streams(moe, experts, fmt, rows, token_length)
    if streams is not None:
        return streams
    if not PANEL_OVER_WEIGHTED_SUM and shipped_weighted_sum_engages(
            moe, rows, token_length, weighted_sum):
        # The shipped route would combine through qwen4_moe_weighted_sum.
        return "weighted_sum_order_unproven"
    return None


def shipped_weighted_sum_engages(moe, rows, token_length, weighted_sum):
    """Whether ``moe_fast.fused_call`` would combine this call through
    ``qwen4_moe_weighted_sum``: its own condition (``do_sort and
    weighted_sum and token_length > 1``) and ``admit_moe_weighted_sum``'s
    gates.  Its remaining structural checks (score/index/inverse shapes, the
    sorted down rows' layout and dtypes) hold by construction for the
    fused route's BF16 calls this module admits."""
    from .. import qwen4_moe_weighted_sum as ws

    if not (weighted_sum and token_length > 1 and _shipped_sorted(moe, rows, token_length)):
        return False
    if not ws.runtime_supported():
        return False
    top_k = int(moe.top_k)
    return top_k in tuple(ws.SUPPORTED_TOP_K) and rows * top_k >= ws.MIN_ROUTED_ROWS


def plan_reference(order, top_k, num_experts):
    """The plan kernel's content in Python (tests): ``(distinct experts in
    ascending id, {expert: member pairs in ascending pair id})``."""
    rows = order.reshape(-1, order.shape[-1])[:, :top_k].tolist()
    members = {}
    for t, row in enumerate(rows):
        for k, e in enumerate(row):
            members.setdefault(int(e), []).append(t * top_k + k)
    return sorted(members), {e: members[e] for e in sorted(members)}


def launch(moe, experts, x, order, scores, *, contract=None, route_sum=None,
           fast_rule=None, return_parts=False):
    """The four kernels: ``y`` with ``x``'s shape (BF16).  With
    ``return_parts`` also ``(plan, hidden_rows, down_rows)``."""
    contract = _mdk.QDOT_CONTRACT if contract is None else contract
    route_sum = _mdk.ROUTE_SUM if route_sum is None else route_sum
    if contract not in _mdk._CONTRACTS or route_sum not in _mdk._ROUTE_SUMS:
        raise ValueError(f"moe_panel_kernel: contract {contract!r} / route_sum {route_sum!r}")
    fmt = expert_format(experts)
    bits, group = _GEOMETRY[fmt]
    hidden = int(x.shape[-1])
    rows = int(x.size) // hidden
    inter = int(experts.down["scales"].shape[-1]) * group
    top_k = int(moe.top_k)
    num_experts = int(experts.gate_up["weight"].shape[0])
    stride = int(order.shape[-1])
    act = moe.switch_mlp.activation
    caps = (float(act.gate_cap), float(act.up_cap), float(moe.route_scale))
    order = order.reshape(rows, stride)
    scores = scores.reshape(rows, stride)
    slots = min(num_experts, rows * top_k)
    has_scale2 = 1 if fmt == "nvfp4" else 0
    scale2 = experts.scales if has_scale2 else mx.zeros((1,), mx.float32)
    plan_threads = (num_experts + 31) // 32 * 32
    (plan,) = _kernel("plan", contract, route_sum, caps)(
        inputs=[order],
        template=[("NE", num_experts), ("W", rows), ("TOPK", top_k), ("OS", stride)],
        grid=(plan_threads, 1, 1), threadgroup=(plan_threads, 1, 1),
        output_shapes=[(1 + 2 * num_experts + num_experts * rows,)],
        output_dtypes=[mx.uint32],
    )
    common = [("H", hidden), ("I", inter), ("NE", num_experts), ("W", rows), ("TOPK", top_k),
              ("BITS", bits), ("GS", group), ("HAS_SCALE2", has_scale2),
              ("NM", MEMBERS_PER_PASS)]
    fast_gu = int(fast_qmv(bits, 2 * inter, hidden, fast_rule))
    (hidden_rows,) = _kernel("gate_up", contract, route_sum, caps)(
        inputs=[x.reshape(rows, hidden), plan, experts.gate_up["weight"],
                experts.gate_up["scales"], scale2],
        template=common + [("FAST", fast_gu)],
        grid=(inter // 4 * 64, slots, 1), threadgroup=(64, 1, 1),
        output_shapes=[(rows * top_k, inter)], output_dtypes=[mx.bfloat16],
    )
    fast_dn = int(fast_qmv(bits, hidden, inter, fast_rule))
    (down_rows,) = _kernel("down", contract, route_sum, caps)(
        inputs=[hidden_rows, plan, experts.down["weight"], experts.down["scales"], scale2],
        template=common + [("FAST", fast_dn)],
        grid=(hidden // 8 * 64, slots, 1), threadgroup=(64, 1, 1),
        output_shapes=[(rows * top_k, hidden)], output_dtypes=[mx.bfloat16],
    )
    width = 256 if hidden >= 256 else (hidden + 31) // 32 * 32
    (y,) = _kernel("combine", contract, route_sum, caps)(
        inputs=[down_rows, order, scores],
        template=[("H", hidden), ("TOPK", top_k), ("OS", stride)],
        grid=(hidden, rows, 1), threadgroup=(width, 1, 1),
        output_shapes=[(rows, hidden)], output_dtypes=[mx.bfloat16],
    )
    y = y.reshape(x.shape)
    return (y, (plan, hidden_rows, down_rows)) if return_parts else y


def law(moe, experts, x, order, scores):
    """The kernels' arithmetic in MLX ops (rounding points, slot order and
    epilogues; the dot products in MLX's own gather, unsorted).  The CPU
    tests run the plumbing against it; the Metal test uses it as a third
    reference."""
    from .moe_fast import _combine, _gate_up_act, _scaled_combine, _scaled_gate_up_act

    top_k = int(moe.top_k)
    hidden = int(x.shape[-1])
    rows = int(x.size) // hidden
    idx = order.reshape(rows, -1)[:, :top_k]
    s = mx.take_along_axis(scores.reshape(rows, -1), idx, axis=-1)
    if _mdk.ROUTE_SUM == "halves":
        half = top_k // 2
        total = s[:, :half].sum(axis=-1, keepdims=True) + s[:, half:].sum(axis=-1, keepdims=True)
    else:
        total = s.sum(axis=-1, keepdims=True)
    weights = s / (total + 1e-20) * moe.route_scale
    xe = mx.expand_dims(x.reshape(rows, hidden), (-2, -3))
    gate_up = experts.gate_up(xe, idx)
    act = moe.switch_mlp.activation
    # The shipped compiled epilogues (on the CPU a compiled tanh/sigmoid can
    # round differently from the eager ops; the kernels follow MLX's GPU
    # functors, which both use there).
    if experts.scales is None:
        h = _gate_up_act(gate_up, act.gate_cap, act.up_cap)
    else:
        table = experts.scales[idx]
        h = _scaled_gate_up_act(gate_up, table[..., 0:1, None], act.gate_cap, act.up_cap)
    down = experts.down(h, idx).squeeze(-2)
    if experts.scales is None:
        out = _combine(down, weights)
    else:
        out = _scaled_combine(down, table[..., 1:2], weights)
    return out.reshape(x.shape)


def call(moe, experts, x, router_weight=None, weighted_sum=False):
    """``moe(x)`` through the kernels, or None (counted) for the shipped route."""
    reason = decline_reason(moe, experts, x, weighted_sum)
    if reason is not None:
        _count(moe, "panel_kernel_declined:" + reason)
        return None
    if router_weight is None:
        from .language import _fp32

        router_weight = _fp32(moe, "gate_weight")
    order, scores = route_compiled(x, router_weight, moe.e_score_correction_bias, moe.top_k)
    y = launch(moe, experts, x, order, scores)
    _count(moe, "panel_kernel")
    _count(moe, "panel_kernel:" + expert_format(experts))
    _count(moe, "fused")
    # The switch's sort receipt: the panel neither sorts nor gathers, so it
    # has its own bucket (the shipped route would have counted auto_*).
    counts = moe.switch_mlp._sort_counts
    counts["panel_kernel"] = counts.get("panel_kernel", 0) + 1
    return y
