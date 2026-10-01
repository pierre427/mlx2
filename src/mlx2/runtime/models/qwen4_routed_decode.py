# SPDX-License-Identifier: Apache-2.0
# Adapted from jundot/omlx PR #3912 (Apache-2.0); see docs/PROVENANCE.md and
# provenance/omlx-fn-2026-09-25.json. The quantized dot products are omlx's
# transcription of MLX 0.32.2 ``quantized.h`` (MIT); see that record.
"""One-token routed experts for Flash-Next decode in fewer launches.

After routing, a one-token Flash-Next MoE block runs its routed experts as the
gate and up ``gather_qmm`` (one launch on a fused [gate|up] table, two on the
split tables the served artifact loads), the compiled SwiGLU, and the down
projection with the router-weighted top-k sum (mlx2's ``qwen4_fused_down``
tile4 kernel in the Flash-Next profile, or ``gather_qmm`` + multiply + sum on
the stock path).

Kernels (#3912 and #4113 from jundot/omlx, #4039/#4055 scheduling ideas):

``gate_up_swiglu`` / ``split_gate_up_swiglu``
    gate+up with a SwiGLU epilogue, reading one fused [gate|up] table or the
    separate gate and up tables (omlx #4113's two-table ``qmv_rows``). Each
    simdgroup computes gate rows and the matching up rows of one expert with
    MLX's ``qmv_fast`` lane partition and add order, rounds both to the
    activation dtype, then applies MLX's ``Sigmoid`` and the two multiplies of
    the compiled ``swiglu`` in order. Intended to equal ``gather_qmm`` (x2 on
    split tables) + ``swiglu`` bit for bit. Grid z is ``token * TOPK + slot``.
``down_combine``
    down with the router-weighted sum: simdgroup ``j`` runs the stock ``qmv``
    work of expert ``j``, rounds, multiplies by the score in the activation
    dtype and sums the ten products in ``col_reduce_small`` order. Intended to
    equal the stock ``gather_qmm`` + ``(x * scores).sum`` tail bit for bit, which
    is NOT the Flash-Next profile's tile4 fused down (that one accumulates the
    weighted products in fp32).
``served_down``
    the Flash-Next profile's tile4 fused down (``qwen4_fused_moe``) with the
    same per-row arithmetic (lane-strided words, ``simd_sum``, two T-valued
    roundings, fp32 slot sum in slot order) but ``DOWN_ROWS_SERVED`` output
    rows per threadgroup (omlx #4055: two rows, twice the threadgroups) and
    one-expert views. Intended to equal tile4 bit for bit.

One-expert views (omlx #4039): MLX commits a command buffer once the inputs
bound to it pass its size cap (50 MB on this GPU class), counting each input
array whole, so every launch that binds a 400+ MB expert table ends a command
buffer. The kernels bind ``w[:1]``, a view sharing the table's buffer at its
offset, and index the other experts from it. The view is used only when it
provably aliases the row-contiguous table; otherwise the whole table is bound.
Scheduling only: the bytes read are the same.

Modes (``MLX_QWEN4_MOE_ROUTED_DECODE``, or the Flash-Next policy field
``moe_routed_decode``; ``set_moe_routed_decode_mode`` switches live):

``off``          (default) nothing changes.
``gate_up``      the gate+up kernel replaces the gate/up ``gather_qmm`` +
                 SwiGLU; the down path is whatever the block would run anyway.
``gate_up_down`` gate+up kernel, then ``served_down`` in place of the tile4
                 fused down. Only where the block would run tile4; otherwise
                 the down half declines (counted) and the block's down runs.
``two_launch``   gate+up kernel and ``down_combine``, the literal omlx pairing
                 (stock-down numerics, not the Flash-Next default's).

Batched rows (extension point for omlx #4106 / #4052): the split gate+up
kernel already addresses x row ``token = z / TOPK`` and output row ``z``, so
an M-row window is grid z = M * TOPK with ``indices`` [M, 10]; served_down
already takes the token from grid z. Admission (``x.size == hidden``) and the
``do_sort`` decline in qwen3_next are what keep them one-token today. For M>1
the reference is gather_qmm's unsorted per-(row, expert) qmv_fast (and tile4
at width 3 is only a candidate), so each width needs its own Metal gate. A
router top-k folded into gate+up (#4052) would replace ``rhs`` by a device
top-k computed in the same launch.

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

from .served_exp import ServedExpGate

ROUTED_DECODE_ENV = "MLX_QWEN4_MOE_ROUTED_DECODE"
VIEWS_ENV = "MLX_QWEN4_MOE_ROUTED_VIEWS"
DOWN_ROWS_ENV = "MLX_QWEN4_MOE_ROUTED_DOWN_ROWS"
MODES = ("off", "gate_up", "gate_up_down", "gate_up_down_shared", "two_launch")
TOP_K = 10
BITS = 4
GROUP_SIZE = 64
GATE_UP_ROWS = 2  # gate rows (and as many up rows) per simdgroup
GATE_UP_SIMDGROUPS = 2
DOWN_ROWS = 4
# served_down: output rows per threadgroup. 4 is the tile4 partition; 2 is
# omlx #4055's one-token choice (twice the threadgroups). Grid-only change.
DOWN_ROWS_SERVED_CHOICES = (2, 4)


def mode_from_env() -> str:
    raw = os.environ.get(ROUTED_DECODE_ENV, "off").strip().lower()
    aliases = {"": "off", "0": "off", "false": "off", "1": "gate_up"}
    raw = aliases.get(raw, raw)
    if raw not in MODES:
        raise ValueError(
            f"{ROUTED_DECODE_ENV}={raw!r}: expected one of {MODES}"
        )
    return raw


def views_from_env() -> bool:
    raw = os.environ.get(VIEWS_ENV, "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def down_rows_from_env() -> int:
    raw = os.environ.get(DOWN_ROWS_ENV, "2").strip() or "2"
    try:
        rows = int(raw)
    except ValueError:
        rows = -1
    if rows not in DOWN_ROWS_SERVED_CHOICES:
        raise ValueError(
            f"{DOWN_ROWS_ENV}={raw!r}: expected one of {DOWN_ROWS_SERVED_CHOICES}"
        )
    return rows


# Live scheduling switches (both bit-neutral); read once at import, settable
# in process so an A/B can rotate them without a reload.
_VIEWS = views_from_env()
_DOWN_ROWS_SERVED = down_rows_from_env()


def set_expert_views(enabled: bool) -> bool:
    global _VIEWS
    _VIEWS = bool(enabled)
    return _VIEWS


def expert_views_enabled() -> bool:
    return _VIEWS


def set_served_down_rows(rows: int) -> int:
    global _DOWN_ROWS_SERVED
    if rows not in DOWN_ROWS_SERVED_CHOICES:
        raise ValueError(f"served down rows must be one of {DOWN_ROWS_SERVED_CHOICES}")
    _DOWN_ROWS_SERVED = int(rows)
    return _DOWN_ROWS_SERVED


def served_down_rows() -> int:
    return _DOWN_ROWS_SERVED


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


def admit_split_routed_decode(x, indices, scores, gate, up, down) -> RoutedDecodeAdmission:
    """``admit_routed_decode`` for separate gate and up expert tables.

    Same geometry; the gate and up tables must each be ``[E, inter, hidden]``
    packed like the down table's experts. Never evaluates an array.
    """
    hidden = x.shape[-1]
    if x.size != hidden:
        return RoutedDecodeAdmission(False, "not one token")
    if x.dtype != mx.bfloat16:
        return RoutedDecodeAdmission(False, "activation must be bfloat16")
    if indices.size != TOP_K or indices.shape[-1] != TOP_K:
        return RoutedDecodeAdmission(False, f"top-k must be {TOP_K}")
    if scores is not None and (scores.size != TOP_K or scores.dtype != x.dtype):
        return RoutedDecodeAdmission(False, "scores must be [10] in the activation dtype")
    from .switch_layers import QuantizedSwitchLinear

    for name, layer in (("gate", gate), ("up", up), ("down", down)):
        # Exactly the class whose call is a bare gather_qmm: a subclass
        # (streamed or repacked tables) may hold no rows or compute otherwise.
        if type(layer) is not QuantizedSwitchLinear:
            return RoutedDecodeAdmission(False, f"{name}: not a resident QuantizedSwitchLinear")
        reason = _quantized_ok(layer)
        if reason is not None:
            return RoutedDecodeAdmission(False, f"{name}: {reason}")
    experts = down["weight"].shape[0]
    inter = down["weight"].shape[-1] * 32 // BITS
    if hidden % 512:
        return RoutedDecodeAdmission(False, "hidden % 512 != 0 (MLX would not pick qmv_fast)")
    if inter % 512 == 0 or inter % GROUP_SIZE:
        return RoutedDecodeAdmission(False, "intermediate must be a non-multiple of 512, multiple of 64")
    table = (experts, inter, hidden * BITS // 32)
    if tuple(gate["weight"].shape) != table or tuple(up["weight"].shape) != table:
        return RoutedDecodeAdmission(False, "gate/up tables do not match down")
    if tuple(down["weight"].shape) != (experts, hidden, inter * BITS // 32):
        return RoutedDecodeAdmission(False, f"down weight shape {tuple(down['weight'].shape)}")
    if inter % (GATE_UP_ROWS * GATE_UP_SIMDGROUPS) or hidden % DOWN_ROWS:
        return RoutedDecodeAdmission(False, "row tiling")
    return RoutedDecodeAdmission(True, "eligible")


def _address(a) -> int:
    import numpy as np

    return np.frombuffer(memoryview(a).cast("B"), dtype=np.uint8).ctypes.data


def _expert_view(a):
    """``a[:1]`` when it provably aliases ``a`` (same buffer address, both
    C-contiguous), else ``a``. The kernels index every expert from the bound
    pointer, so a copied or offset view would read the wrong bytes."""
    view = a[:1]
    mx.eval(a, view)
    whole, first = memoryview(a), memoryview(view)
    if whole.c_contiguous and first.c_contiguous and _address(view) == _address(a):
        return view
    return a


def expert_operands(layer):
    """(weight, scales, biases) to bind for a resident expert table.

    With views on, one-expert views cached on the module and rebuilt when any
    of its arrays is replaced (the cache holds the originals, so an id cannot
    be reused under it). The first build evaluates the table once.
    """
    arrays = (layer["weight"], layer["scales"], layer["biases"])
    if not _VIEWS:
        return arrays
    cached = layer.__dict__.get("_mlx2_expert_views")
    if cached is not None and all(c is a for c, a in zip(cached[0], arrays)):
        return cached[1]
    views = tuple(_expert_view(a) for a in arrays)
    layer.__dict__["_mlx2_expert_views"] = (arrays, views)
    return views


def expert_view_count(layer) -> int:
    """How many of the layer's cached operands are true one-expert views."""
    cached = layer.__dict__.get("_mlx2_expert_views")
    if cached is None:
        return 0
    return sum(v is not a for a, v in zip(*cached))


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


# Split-table gate+up (omlx #4113 @1335263e ``qmv_rows`` with gate rows from
# (wg, sg, bg) and up rows from (wu, su, bu)); the per-row arithmetic is
# GATE_UP_SOURCE's. Grid z = token * TOPK + slot: x row ``token``, output row
# ``tid.z``. Only one token is admitted today; the z axis is the extension
# point for batched rows (see docs in the report / provenance).
SPLIT_GATE_UP_SOURCE = r"""
    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
    const int in_vec_size_g = K / GS;
    const int out_row = int(tid.y) * (NSG * RPS) + int(simd_gid) * RPS;
    const int token = int(tid.z) / TOPK;
    const size_t row0 = size_t(rhs[tid.z]) * NI + out_row;
    const int lane_w = int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const int lane_g = int(simd_lid) / SCALE_STEP_PER_THREAD;

    const device uint8_t* gw = (const device uint8_t*)wg + row0 * in_vec_size_w + lane_w;
    const device uint8_t* uw = (const device uint8_t*)wu + row0 * in_vec_size_w + lane_w;
    const device T* gs = sg + row0 * in_vec_size_g + lane_g;
    const device T* gb = bg + row0 * in_vec_size_g + lane_g;
    const device T* us = su + row0 * in_vec_size_g + lane_g;
    const device T* ub = bu + row0 * in_vec_size_g + lane_g;
    const device T* xp = x + size_t(token) * K + int(simd_lid) * VALUES_PER_THREAD;

    float x_thread[VALUES_PER_THREAD];
    float result[2 * RPS] = {0};

    for (int k = 0; k < K; k += BLOCK_SIZE) {
      float sum = load_vector<T>(xp, x_thread);
      for (int row = 0; row < 2 * RPS; row++) {
        const bool g = row < RPS;
        const int r = g ? row : row - RPS;
        const device uint8_t* wl = (g ? gw : uw) + r * in_vec_size_w;
        float s = (g ? gs : us)[r * in_vec_size_g];
        float b = (g ? gb : ub)[r * in_vec_size_g];
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
      device T* yp = y + size_t(tid.z) * NI + out_row;
      for (int row = 0; row < RPS; row++) {
        T g = static_cast<T>(result[row]);
        T u = static_cast<T>(result[row + RPS]);
        T t = g * omlx_mlx_sigmoid<T>(g);
        yp[row] = t * u;
      }
    }
"""

# The Flash-Next profile's tile4 fused down (qwen4_fused_moe
# _DOWN_REDUCE_TILE4_SOURCE, mlx2's own kernel) with RPS output rows per
# threadgroup. Per (slot, row) the arithmetic is tile4's: lane-strided packed
# words, fp32 accum_x / accum_q in nibble order, simd_sum, the expert value
# and the weighted value rounded to T, then the fp32 sum over slots 0..9 in
# order. Five simdgroups own two slots each, as in tile4. hidden, rhs and
# scores are row-contiguous here (the caller passes this module's own
# gate+up output and a contiguous routing row). Grid z is the token.
SERVED_DOWN_SOURCE = r"""
    constexpr uint DOWN_WORDS = EH / 8;
    constexpr uint DOWN_GROUPS = EH / 64;

    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint row_base = threadgroup_position_in_grid.y * RPS;
    uint token = threadgroup_position_in_grid.z;
    uint slot_base = sg * 2;
    const device uint32_t* packed = w;
    threadgroup float partials[TOPK * RPS];

    float values[2 * RPS];
#pragma unroll
    for (uint i = 0; i < 2 * RPS; ++i) {
        values[i] = 0.0f;
    }

#pragma unroll
    for (uint local_slot = 0; local_slot < 2; ++local_slot) {
        uint slot = slot_base + local_slot;
        uint elem = token * TOPK + slot;
        uint expert = uint(rhs[elem]);
        const device T* hrow = hidden + size_t(elem) * EH;

        for (uint word = lane; word < DOWN_WORDS; word += 32) {
            size_t hbase = size_t(word) * 8;
            float xv[8];
            float accum_x = 0.0f;
#pragma unroll
            for (uint nibble = 0; nibble < 8; ++nibble) {
                xv[nibble] = float(hrow[hbase + nibble]);
                accum_x += xv[nibble];
            }

#pragma unroll
            for (uint local_row = 0; local_row < RPS; ++local_row) {
                uint row = row_base + local_row;
                size_t wrow = size_t(expert) * H + row;
                uint32_t p = packed[wrow * DOWN_WORDS + word];
                uint group = word >> 3;
                float scale = float(scales[wrow * DOWN_GROUPS + group]);
                float bias = float(biases[wrow * DOWN_GROUPS + group]);
                float accum_q = 0.0f;
#pragma unroll
                for (uint nibble = 0; nibble < 8; ++nibble) {
                    accum_q += xv[nibble] *
                        float((p >> (4 * nibble)) & 0xFu);
                }
                values[local_slot * RPS + local_row] +=
                    scale * accum_q + bias * accum_x;
            }
        }
    }

#pragma unroll
    for (uint i = 0; i < 2 * RPS; ++i) {
        values[i] = simd_sum(values[i]);
    }
    if (lane == 0) {
#pragma unroll
        for (uint local_slot = 0; local_slot < 2; ++local_slot) {
            uint slot = slot_base + local_slot;
            float score = float(scores[token * TOPK + slot]);
#pragma unroll
            for (uint local_row = 0; local_row < RPS; ++local_row) {
                T expert_value = static_cast<T>(
                    values[local_slot * RPS + local_row]);
                T weighted_value = static_cast<T>(
                    float(expert_value) * score);
                partials[slot * RPS + local_row] = float(weighted_value);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0 && lane < RPS) {
        float routed = 0.0f;
#pragma unroll
        for (uint slot = 0; slot < TOPK; ++slot) {
            routed += partials[slot * RPS + lane];
        }
        out[size_t(token) * H + row_base + lane] = static_cast<T>(routed);
    }
"""
SERVED_DOWN_SIMDGROUPS = 5  # two slots each, as tile4


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
        split_gate_up = mx.fast.metal_kernel(
            name="mlx2_qwen4_moe_split_gate_up_swiglu_decode",
            input_names=["x", "wg", "sg", "bg", "wu", "su", "bu", "rhs"],
            output_names=["y"],
            header=_header(fast=True),
            source=SPLIT_GATE_UP_SOURCE,
        )
        served_down = mx.fast.metal_kernel(
            name="mlx2_qwen4_moe_served_down_decode",
            input_names=["hidden", "w", "scales", "biases", "rhs", "scores"],
            output_names=["out"],
            source=SERVED_DOWN_SOURCE,
        )
        _KERNELS = (gate_up, down, split_gate_up, served_down)
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
            *expert_operands(gate_up),
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
            *expert_operands(down),
            indices.reshape(TOP_K).astype(mx.uint32),
            scores.reshape(TOP_K),
        ],
        template=[("T", h.dtype), ("K", inter), ("N", hidden), ("RPS", DOWN_ROWS)],
        grid=(32, TOP_K * hidden // DOWN_ROWS, 1),
        threadgroup=(32, TOP_K, 1),
        output_shapes=[(hidden,)],
        output_dtypes=[h.dtype],
    )[0]


def split_gate_up_swiglu(x, indices, gate, up):
    """``swiglu(gate_j(x), up_j(x))`` of the ten routed experts from separate
    gate and up tables for one token: [10, inter]."""
    hidden = x.shape[-1]
    inter = gate["weight"].shape[1]
    rows = GATE_UP_ROWS * GATE_UP_SIMDGROUPS
    kernel = _kernels()[2]
    return kernel(
        inputs=[
            x.reshape(hidden),
            *expert_operands(gate),
            *expert_operands(up),
            indices.reshape(TOP_K).astype(mx.uint32),
        ],
        template=[
            ("T", x.dtype),
            ("K", hidden),
            ("NI", inter),
            ("RPS", GATE_UP_ROWS),
            ("NSG", GATE_UP_SIMDGROUPS),
            ("TOPK", TOP_K),
        ],
        grid=(32, GATE_UP_SIMDGROUPS * inter // rows, TOP_K),
        threadgroup=(32, GATE_UP_SIMDGROUPS, 1),
        output_shapes=[(TOP_K, inter)],
        output_dtypes=[x.dtype],
    )[0]


def served_down(h, indices, scores, down, rows=None):
    """The tile4 fused down for one token with ``rows`` output rows per
    threadgroup: [hidden]. Admission is the caller's (``admit_qwen4_fused_down``
    at width 1 with the tile4 variant)."""
    rows = _DOWN_ROWS_SERVED if rows is None else rows
    if rows not in DOWN_ROWS_SERVED_CHOICES:
        raise ValueError(f"served down rows must be one of {DOWN_ROWS_SERVED_CHOICES}")
    inter = h.shape[-1]
    hidden = down["weight"].shape[1]
    kernel = _kernels()[3]
    return kernel(
        inputs=[
            h.reshape(TOP_K, inter),
            *expert_operands(down),
            indices.reshape(TOP_K),
            scores.reshape(TOP_K),
        ],
        template=[
            ("T", h.dtype),
            ("H", hidden),
            ("EH", inter),
            ("TOPK", TOP_K),
            ("RPS", rows),
        ],
        grid=(32 * SERVED_DOWN_SIMDGROUPS, hidden // rows, 1),
        threadgroup=(32 * SERVED_DOWN_SIMDGROUPS, 1, 1),
        output_shapes=[(hidden,)],
        output_dtypes=[h.dtype],
    )[0]


# --------------------------------------------------------------------------
# Shared-expert fold (omlx #4039 idea, matched to mlx2's served block).
# The Flash-Next block ends with
#   y = tile4_down(routed); s = shared_expert(x)   (4-bit qmv_fast gate/up,
#   compiled swiglu, 4-bit qmv down); g = mx.sigmoid(shared_expert_gate(x))
#   (8-bit qmv, one row; eager sigmoid from the -fno-fast-math metallib);
#   out = y + g * s   (eager bf16 multiply, then add; glue compile off)
# The fold computes the shared gate/up rows in the gate+up launch (grid z 0:
# shared rows, z 1..10: routed slots) and the shared down rows plus that
# combine in the down launch (a sixth simdgroup). The shared gate logit stays
# the block's own launch and enters the down launch as an input: an in-kernel
# 8-bit qmv row matched MLX on 64/64 gate checks but differed by one bf16 ulp
# on 1 of 36960 full-model calls (layer 16, -1.9766 vs -1.96875), so it is not
# folded. Each weight format gets its own namespaced copy of the QMV header
# (omlx's per-format namespaces and qmv_rows helper).
# --------------------------------------------------------------------------
QMV_ROWS = r"""
template <typename T, int K, int NA, int NB>
METAL_FUNC void qmv_rows(
    const device uint8_t* wa, const device T* sa, const device T* ba, size_t row_a,
    const device uint8_t* wb, const device T* sb, const device T* bb, size_t row_b,
    const device T* x, uint simd_lid, thread float* result) {
  constexpr int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
  constexpr int in_vec_size_g = K / GS;
  const int lane_w = int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
  const int lane_g = int(simd_lid) / SCALE_STEP_PER_THREAD;
  wa += row_a * in_vec_size_w + lane_w;
  sa += row_a * in_vec_size_g + lane_g;
  ba += row_a * in_vec_size_g + lane_g;
  wb += row_b * in_vec_size_w + lane_w;
  sb += row_b * in_vec_size_g + lane_g;
  bb += row_b * in_vec_size_g + lane_g;
  x += int(simd_lid) * VALUES_PER_THREAD;
  float x_thread[VALUES_PER_THREAD];
  for (int row = 0; row < NA + NB; row++) {
    result[row] = 0;
  }
  int k = 0;
  for (; k < (FAST ? K : K - BLOCK_SIZE); k += BLOCK_SIZE) {
    float sum = load_vector<T>(x, x_thread);
    for (int row = 0; row < NA + NB; row++) {
      const bool a = row < NA;
      const int r = a ? row : row - NA;
      const device uint8_t* wl = (a ? wa : wb) + r * in_vec_size_w;
      float s = (a ? sa : sb)[r * in_vec_size_g];
      float b = (a ? ba : bb)[r * in_vec_size_g];
      result[row] += qdot_n(wl, x_thread, s, b, sum, VALUES_PER_THREAD);
    }
    wa += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    wb += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    sa += BLOCK_SIZE / GS;
    ba += BLOCK_SIZE / GS;
    sb += BLOCK_SIZE / GS;
    bb += BLOCK_SIZE / GS;
    x += BLOCK_SIZE;
  }
  if (!FAST) {
    const int remaining = clamp(
        int(K - k - int(simd_lid) * VALUES_PER_THREAD), 0, VALUES_PER_THREAD);
    if (remaining > 0) {
      float sum = load_vector_safe<T>(x, x_thread, remaining);
      for (int row = 0; row < NA + NB; row++) {
        const bool a = row < NA;
        const int r = a ? row : row - NA;
        const device uint8_t* wl = (a ? wa : wb) + r * in_vec_size_w;
        float s = (a ? sa : sb)[r * in_vec_size_g];
        float b = (a ? ba : bb)[r * in_vec_size_g];
        result[row] += qdot_n(wl, x_thread, s, b, sum, remaining);
      }
    }
  }
  for (int row = 0; row < NA + NB; row++) {
    result[row] = simd_sum(result[row]);
  }
}
"""


def _format_header(namespace: str, bits: int, fast: bool) -> str:
    body = (
        QMV_HEADER.replace("__BITS__", str(bits))
        .replace("__GS__", str(GROUP_SIZE))
        .replace("__FAST__", "1" if fast else "0")
        .replace("using namespace metal;", "")
    )
    return f"namespace {namespace} {{\n{body}\n{QMV_ROWS}\n}}  // namespace {namespace}\n"


SHARED_COMMON = "using namespace metal;\n" + SIGMOID + r"""
template <typename T, int RPS>
METAL_FUNC void swiglu_store(thread const float* result, device T* yp, uint simd_lid) {
  if (simd_lid == 0) {
    for (int row = 0; row < RPS; row++) {
      T g = static_cast<T>(result[row]);
      T u = static_cast<T>(result[row + RPS]);
      T t = g * omlx_mlx_sigmoid<T>(g);
      yp[row] = t * u;
    }
  }
}
"""

SHARED_GATE_UP_SOURCE = r"""
    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int out_row = int(tid.y) * (NSG * RPS) + int(simd_gid) * RPS;
    float result[2 * RPS];
    if (tid.z == 0) {
      q4f::qmv_rows<T, K, RPS, RPS>(
          (const device uint8_t*)shw, shs, shb, out_row,
          (const device uint8_t*)suw, sus, sub, out_row, x, simd_lid, result);
      swiglu_store<T, RPS>(result, y + TOPK * NI + out_row, simd_lid);
      return;
    }
    const int slot = int(tid.z) - 1;
    const size_t row0 = size_t(rhs[slot]) * NI + out_row;
    q4f::qmv_rows<T, K, RPS, RPS>(
        (const device uint8_t*)wg, sg, bg, row0,
        (const device uint8_t*)wu, su, bu, row0, x, simd_lid, result);
    swiglu_store<T, RPS>(result, y + size_t(slot) * NI + out_row, simd_lid);
"""

SHARED_DOWN_SOURCE = r"""
    constexpr uint DOWN_WORDS = EH / 8;
    constexpr uint DOWN_GROUPS = EH / 64;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint row_base = threadgroup_position_in_grid.y * RPS;
    threadgroup float partials[TOPK * RPS];
    threadgroup T shared_part[RPS];
    if (sg == 5) {
      float result[RPS];
      q4s::qmv_rows<T, EH, RPS, 0>(
          (const device uint8_t*)sdw, sds, sdb, row_base,
          (const device uint8_t*)sdw, sds, sdb, row_base,
          hidden + TOPK * EH, lane, result);
      if (lane == 0) {
        for (int r = 0; r < RPS; r++) shared_part[r] = static_cast<T>(result[r]);
      }
    } else {
    const device uint32_t* packed = w;
    uint slot_base = sg * 2;
    float values[2 * RPS];
#pragma unroll
    for (uint i = 0; i < 2 * RPS; ++i) {
        values[i] = 0.0f;
    }
#pragma unroll
    for (uint local_slot = 0; local_slot < 2; ++local_slot) {
        uint slot = slot_base + local_slot;
        uint expert = uint(rhs[slot]);
        const device T* hrow = hidden + size_t(slot) * EH;
        for (uint word = lane; word < DOWN_WORDS; word += 32) {
            size_t hbase = size_t(word) * 8;
            float xv[8];
            float accum_x = 0.0f;
#pragma unroll
            for (uint nibble = 0; nibble < 8; ++nibble) {
                xv[nibble] = float(hrow[hbase + nibble]);
                accum_x += xv[nibble];
            }
#pragma unroll
            for (uint local_row = 0; local_row < RPS; ++local_row) {
                uint row = row_base + local_row;
                size_t wrow = size_t(expert) * H + row;
                uint32_t p = packed[wrow * DOWN_WORDS + word];
                uint group = word >> 3;
                float scale = float(scales[wrow * DOWN_GROUPS + group]);
                float bias = float(biases[wrow * DOWN_GROUPS + group]);
                float accum_q = 0.0f;
#pragma unroll
                for (uint nibble = 0; nibble < 8; ++nibble) {
                    accum_q += xv[nibble] *
                        float((p >> (4 * nibble)) & 0xFu);
                }
                values[local_slot * RPS + local_row] +=
                    scale * accum_q + bias * accum_x;
            }
        }
    }
#pragma unroll
    for (uint i = 0; i < 2 * RPS; ++i) {
        values[i] = simd_sum(values[i]);
    }
    if (lane == 0) {
#pragma unroll
        for (uint local_slot = 0; local_slot < 2; ++local_slot) {
            uint slot = slot_base + local_slot;
            float score = float(scores[slot]);
#pragma unroll
            for (uint local_row = 0; local_row < RPS; ++local_row) {
                T expert_value = static_cast<T>(
                    values[local_slot * RPS + local_row]);
                T weighted_value = static_cast<T>(
                    float(expert_value) * score);
                partials[slot * RPS + local_row] = float(weighted_value);
            }
        }
    }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane < RPS) {
        float routed = 0.0f;
#pragma unroll
        for (uint slot = 0; slot < TOPK; ++slot) {
            routed += partials[slot * RPS + lane];
        }
        const T yr = static_cast<T>(routed);
        const float g = float(gate[0]);
        const T e = T(1.0f + float(T(metal::precise::exp(metal::abs(g)))));
        const T sy = T(metal::precise::divide(1.0f, float(e)));
        const T gate = g < 0.0f ? sy : T(1.0f - float(sy));
        const T sh = T(float(gate) * float(shared_part[lane]));
        out[row_base + lane] = T(float(yr) + float(sh));
    }
"""

_SHARED_KERNELS = None


def _shared_kernels():
    global _SHARED_KERNELS
    if _SHARED_KERNELS is None:
        gate_up = mx.fast.metal_kernel(
            name="mlx2_qwen4_moe_gate_up_shared_decode",
            input_names=["x", "wg", "sg", "bg", "wu", "su", "bu", "rhs",
                         "shw", "shs", "shb", "suw", "sus", "sub"],
            output_names=["y"],
            header=SHARED_COMMON + _format_header("q4f", 4, True),
            source=SHARED_GATE_UP_SOURCE,
        )
        down = mx.fast.metal_kernel(
            name="mlx2_qwen4_moe_served_down_shared_decode",
            input_names=["hidden", "w", "scales", "biases", "rhs", "scores", "sdw", "sds", "sdb", "gate"],
            output_names=["out"],
            header="using namespace metal;\n" + _format_header("q4s", 4, False),
            source=SHARED_DOWN_SOURCE,
        )
        _SHARED_KERNELS = (gate_up, down)
    return _SHARED_KERNELS


def _linear_ok(layer, bits: int, out_dims: int, in_dims: int) -> str | None:
    import mlx.nn as nn

    # The row-exact verify route (qwen4_row_exact) swaps in a subclass whose
    # one-row call is the plain one; the kernels read only the arrays.
    if getattr(type(layer), "_mlx2_row_exact_base", type(layer)) is not nn.QuantizedLinear:
        return "not a plain QuantizedLinear"
    if "_lane_prepared" in layer.__dict__:
        return "lane matmul installed"
    if (layer.bits, layer.group_size, getattr(layer, "mode", "affine")) != (bits, GROUP_SIZE, "affine"):
        return f"format b{layer.bits}g{layer.group_size} != b{bits}g{GROUP_SIZE}"
    if "bias" in layer or "biases" not in layer:
        return "needs affine biases and no linear bias"
    if layer["scales"].dtype != mx.bfloat16 or layer["biases"].dtype != mx.bfloat16:
        return "scales/biases must be bfloat16"
    if tuple(layer["weight"].shape) != (out_dims, in_dims * bits // 32):
        return f"weight shape {tuple(layer['weight'].shape)}"
    return None


def admit_shared_fold(shared, hidden: int, inter: int) -> RoutedDecodeAdmission:
    """Structural check of the shared expert for the fold (its gate logit is
    computed by the block's own module and only has to be [.., 1] in T)."""
    if shared is None:
        return RoutedDecodeAdmission(False, "no shared expert")
    if hasattr(shared, "_prefill_counts"):
        return RoutedDecodeAdmission(False, "tensorfold prefill MLP installed")
    for name, layer, out_dims, in_dims in (
        ("shared gate_proj", shared.get("gate_proj"), inter, hidden),
        ("shared up_proj", shared.get("up_proj"), inter, hidden),
        ("shared down_proj", shared.get("down_proj"), hidden, inter),
    ):
        reason = _linear_ok(layer, BITS, out_dims, in_dims)
        if reason is not None:
            return RoutedDecodeAdmission(False, f"{name}: {reason}")
    return RoutedDecodeAdmission(True, "eligible")


def _dense_operands(layer):
    return (layer["weight"], layer["scales"], layer["biases"])


def shared_fold_decode(x, indices, scores, gate, up, down, shared, gate_logit, rows=None):
    """``tile4 routed + sigmoid(gate_logit) * shared_expert(x)`` for one token
    in two launches, from split gate/up tables. ``gate_logit`` is the block's
    own ``shared_expert_gate(x)`` ([..., 1] in the activation dtype).
    Admission is the caller's (split routed + tile4 + ``admit_shared_fold``)."""
    rows = _DOWN_ROWS_SERVED if rows is None else rows
    if rows not in DOWN_ROWS_SERVED_CHOICES:
        raise ValueError(f"served down rows must be one of {DOWN_ROWS_SERVED_CHOICES}")
    hidden = x.shape[-1]
    inter = gate["weight"].shape[1]
    gate_up_kernel, down_kernel = _shared_kernels()
    h = gate_up_kernel(
        inputs=[
            x.reshape(hidden),
            *expert_operands(gate),
            *expert_operands(up),
            indices.reshape(TOP_K).astype(mx.uint32),
            *_dense_operands(shared.gate_proj),
            *_dense_operands(shared.up_proj),
        ],
        template=[("T", x.dtype), ("K", hidden), ("NI", inter),
                  ("RPS", GATE_UP_ROWS), ("NSG", GATE_UP_SIMDGROUPS), ("TOPK", TOP_K)],
        grid=(32, inter // GATE_UP_ROWS, TOP_K + 1),
        threadgroup=(32, GATE_UP_SIMDGROUPS, 1),
        output_shapes=[((TOP_K + 1) * inter,)],
        output_dtypes=[x.dtype],
    )[0]
    return down_kernel(
        inputs=[
            h,
            *expert_operands(down),
            indices.reshape(TOP_K),
            scores.reshape(TOP_K),
            *_dense_operands(shared.down_proj),
            gate_logit.reshape(1),
        ],
        template=[("T", x.dtype), ("H", hidden), ("EH", inter), ("TOPK", TOP_K), ("RPS", rows)],
        grid=(32 * (SERVED_DOWN_SIMDGROUPS + 1), hidden // rows, 1),
        threadgroup=(32 * (SERVED_DOWN_SIMDGROUPS + 1), 1, 1),
        output_shapes=[(hidden,)],
        output_dtypes=[x.dtype],
    )[0]


# --------------------------------------------------------------------------
# Candidate: top-8/10 routed decode over split gate/up tables with MLX's own
# down traversal (omlx #4113 @1335263e, Apache-2.0; see docs/PROVENANCE.md and
# provenance/omlx-4113-routed-candidate.json). DEFAULT OFF, unqualified.
#
# Only the structural generalisation is mined: the top-k template, the
# two-table ``qmv_rows`` gate+up (gate and up rows read from separate expert
# tables, as Qwen3.6 keeps them with fused gate/up pinned off) and a down pass
# that follows ``qmv_fast`` when MLX would pick it (intermediate % 512 == 0,
# e.g. Qwen3.6's 512) and ``qmv`` with its guarded tail otherwise. The shared
# expert fold, verify windows, expert views and self-disable are not taken.
# ``qmv_fast`` down is different arithmetic and dispatch from the qualified
# top-10 ``qmv`` down, so nothing here is presumed bit-exact: the candidate is
# refused unless ``scripts/check_fn_routed_decode.py`` passes on Metal at the
# served geometry and build.
# --------------------------------------------------------------------------
CANDIDATE_MODES = ("off", "two_launch")
CANDIDATE_TOP_K = (8, 10)


def qmv_fast_layout(k: int, n: int, bits: int = BITS) -> bool:
    """Whether MLX runs ``qmv_fast`` (else ``qmv``) for a one-row affine
    product with ``k`` inputs and ``n`` outputs (omlx #4113's reading of MLX
    0.32.2 ``qmv_fast_k_alignment``). A GPU-checked claim on the lab fork."""
    pack_factor = 8 if bits in (3, 5) else (4 if bits == 6 else 32 // bits)
    return n % 8 == 0 and k % (pack_factor * (1 if bits == 2 else 2) * 32) == 0


def _candidate_table_ok(layer) -> str | None:
    from .switch_layers import QuantizedSwitchLinear

    # Exactly the class whose call is a bare gather_qmm: a subclass (streamed
    # tables, repacked layers) may hold no resident rows or compute otherwise.
    if type(layer) is not QuantizedSwitchLinear:
        return "not a resident QuantizedSwitchLinear"
    return _quantized_ok(layer)


def admit_routed_candidate(x, indices, scores, gate, up, down) -> RoutedDecodeAdmission:
    """Structural check for the candidate; never evaluates an array."""
    hidden = x.shape[-1]
    if x.size != hidden:
        return RoutedDecodeAdmission(False, "not one token")
    if x.dtype != mx.bfloat16:
        # fp16 has no served-SiLU contract probe; bf16 only until one exists.
        return RoutedDecodeAdmission(False, "activation must be bfloat16")
    top_k = indices.size
    if top_k not in CANDIDATE_TOP_K or indices.shape[-1] != top_k:
        return RoutedDecodeAdmission(False, "top-k must be 8 or 10")
    if scores is None:
        return RoutedDecodeAdmission(False, "no scores")
    if scores.size != top_k or scores.dtype != x.dtype:
        return RoutedDecodeAdmission(False, "scores must be [top-k] in the activation dtype")
    for name, layer in (("gate", gate), ("up", up), ("down", down)):
        reason = _candidate_table_ok(layer)
        if reason is not None:
            return RoutedDecodeAdmission(False, f"{name}: {reason}")
    experts = down["weight"].shape[0]
    inter = down["weight"].shape[-1] * 32 // BITS
    table = (experts, inter, hidden * BITS // 32)
    if tuple(gate["weight"].shape) != table or tuple(up["weight"].shape) != table:
        return RoutedDecodeAdmission(False, "gate/up tables do not match down")
    if tuple(down["weight"].shape) != (experts, hidden, inter * BITS // 32):
        return RoutedDecodeAdmission(False, f"down weight shape {tuple(down['weight'].shape)}")
    if not qmv_fast_layout(hidden, inter):
        return RoutedDecodeAdmission(False, "gate/up would not take qmv_fast")
    if inter % GROUP_SIZE:
        return RoutedDecodeAdmission(False, "intermediate must be a multiple of 64")
    if inter % (GATE_UP_ROWS * GATE_UP_SIMDGROUPS) or hidden % DOWN_ROWS:
        return RoutedDecodeAdmission(False, "row tiling")
    return RoutedDecodeAdmission(True, "eligible")


# gate rows from (wg, sg, bg), up rows from (wu, su, bu); slot = grid z.
CANDIDATE_GATE_UP_SOURCE = r"""
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
    const device T* gs = sg + row0 * in_vec_size_g + lane_g;
    const device T* gb = bg + row0 * in_vec_size_g + lane_g;
    const device T* us = su + row0 * in_vec_size_g + lane_g;
    const device T* ub = bu + row0 * in_vec_size_g + lane_g;
    const device T* xp = x + int(simd_lid) * VALUES_PER_THREAD;

    float x_thread[VALUES_PER_THREAD];
    float result[2 * RPS] = {0};

    for (int k = 0; k < K; k += BLOCK_SIZE) {
      float sum = load_vector<T>(xp, x_thread);
      for (int row = 0; row < 2 * RPS; row++) {
        const bool g = row < RPS;
        const int r = g ? row : row - RPS;
        const device uint8_t* wl = (g ? gw : uw) + r * in_vec_size_w;
        float s = (g ? gs : us)[r * in_vec_size_g];
        float b = (g ? gb : ub)[r * in_vec_size_g];
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
      device T* yp = y + size_t(tid.z) * NI + out_row;
      for (int row = 0; row < RPS; row++) {
        T g = static_cast<T>(result[row]);
        T u = static_cast<T>(result[row + RPS]);
        T t = g * omlx_mlx_sigmoid<T>(g);
        yp[row] = t * u;
      }
    }
"""

# The top-10 down/combine with TOPK slots and MLX's traversal for the shape:
# FAST walks whole blocks (qmv_fast), otherwise full blocks then the guarded
# tail (qmv). The k-sum is col_reduce_small's lane j % 8 fold, then lanes in
# order, all in T; for TOPK 10 it is the qualified top-10 order.
CANDIDATE_DOWN_SOURCE = r"""
    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
    const int in_vec_size_g = K / GS;
    const int out_row = int(tid.y) * RPS;
    const int slot = int(simd_gid);
    const size_t expert = size_t(rhs[slot]);
    threadgroup T part[TOPK * RPS];

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
    for (; k < (FAST ? K : K - BLOCK_SIZE); k += BLOCK_SIZE) {
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
    if (!FAST) {
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
      const int row = int(simd_lid);
      T lane[8];
      for (int l = 0; l < 8; ++l) {
        lane[l] = T(0);
      }
      for (int j = 0; j < TOPK; ++j) {
        lane[j % 8] = part[j * RPS + row] + lane[j % 8];
      }
      T acc = lane[0];
      for (int l = 1; l < 8; ++l) {
        acc = lane[l] + acc;
      }
      y[out_row + row] = acc;
    }
"""

_CANDIDATE_KERNELS: dict = {}


def _candidate_kernel(kind: str, fast: bool):
    key = (kind, fast)
    if key not in _CANDIDATE_KERNELS:
        if kind == "gate_up":
            inputs = ["x", "wg", "sg", "bg", "wu", "su", "bu", "rhs"]
            source = CANDIDATE_GATE_UP_SOURCE
        else:
            inputs = ["x", "w", "scales", "biases", "rhs", "scores"]
            source = CANDIDATE_DOWN_SOURCE
        _CANDIDATE_KERNELS[key] = mx.fast.metal_kernel(
            name=f"mlx2_routed_candidate_{kind}_{'fast' if fast else 'qmv'}",
            input_names=inputs,
            output_names=["y"],
            header=_header(fast=fast),
            source=source,
        )
    return _CANDIDATE_KERNELS[key]


def candidate_gate_up_swiglu(x, indices, gate, up):
    """``swiglu(gate_j(x), up_j(x))`` for the top-k routed experts: [top_k, inter]."""
    hidden = x.shape[-1]
    inter = gate["weight"].shape[1]
    top_k = indices.size
    rows = GATE_UP_ROWS * GATE_UP_SIMDGROUPS
    return _candidate_kernel("gate_up", True)(
        inputs=[
            x.reshape(hidden),
            gate["weight"], gate["scales"], gate["biases"],
            up["weight"], up["scales"], up["biases"],
            indices.reshape(top_k).astype(mx.uint32),
        ],
        template=[
            ("T", x.dtype), ("K", hidden), ("NI", inter),
            ("RPS", GATE_UP_ROWS), ("NSG", GATE_UP_SIMDGROUPS),
        ],
        grid=(32, GATE_UP_SIMDGROUPS * inter // rows, top_k),
        threadgroup=(32, GATE_UP_SIMDGROUPS, 1),
        output_shapes=[(top_k, inter)],
        output_dtypes=[x.dtype],
    )[0]


def candidate_down_combine(h, indices, scores, down):
    """``sum_j scores[j] * down_j(h[j])`` with MLX's traversal for the shape."""
    top_k = indices.size
    inter = h.shape[-1]
    hidden = down["weight"].shape[1]
    return _candidate_kernel("down", qmv_fast_layout(inter, hidden))(
        inputs=[
            h.reshape(top_k, inter),
            down["weight"], down["scales"], down["biases"],
            indices.reshape(top_k).astype(mx.uint32),
            scores.reshape(top_k),
        ],
        template=[
            ("T", h.dtype), ("K", inter), ("N", hidden),
            ("RPS", DOWN_ROWS), ("TOPK", top_k),
        ],
        grid=(32, top_k * hidden // DOWN_ROWS, 1),
        threadgroup=(32, top_k, 1),
        output_shapes=[(hidden,)],
        output_dtypes=[h.dtype],
    )[0]


# Served-graph gates (omlx #4122 follow-up; served_exp). Every SwiGLU epilogue
# here (gate_up, split, shared fold, window rows, candidate) copies the
# compiled ``swiglu`` with SIGMOID's ``metal::exp``; the shared fold's gate
# copies the eager ``mx.sigmoid`` unary in ``metal::precise::exp``. Each runs
# only while its spelling reproduces the served op on this build (bf16, the
# only admitted dtype). Callers check ``runtime_supported()`` first.
SWIGLU_PROBE_BODY = "    T t = x * omlx_mlx_sigmoid<T>(x);\n    y = t * T(1);"
SHARED_GATE_PROBE_BODY = (
    "    const float g = float(x);\n"
    "    const T e = T(1.0f + float(T(metal::precise::exp(metal::abs(g)))));\n"
    "    const T sy = T(metal::precise::divide(1.0f, float(e)));\n"
    "    y = g < 0.0f ? sy : T(1.0f - float(sy));"
)


def _served_swiglu(x):
    from .activations import swiglu

    return swiglu(x, mx.ones_like(x))


SWIGLU_GATE = ServedExpGate(
    "routed_swiglu",
    served_name="compiled SwiGLU",
    served=_served_swiglu,
    header=SIGMOID,
    body=SWIGLU_PROBE_BODY,
)
SHARED_GATE_GATE = ServedExpGate(
    "routed_shared_gate",
    served_name="eager sigmoid",
    served=mx.sigmoid,
    body=SHARED_GATE_PROBE_BODY,
    kernel_exp="metal::precise::exp",
)


def served_swiglu_refusal() -> str | None:
    """Why the SwiGLU epilogues may not run under this MLX build, or None."""
    return SWIGLU_GATE.refusal(mx.bfloat16)


def served_shared_gate_refusal() -> str | None:
    """Why the folded shared-expert gate may not run here, or None."""
    return SHARED_GATE_GATE.refusal(mx.bfloat16)


def candidate_runtime_refusal() -> str | None:
    """Why the candidate may not run in this process, or None.

    The kernels spell sigmoid with ``metal::exp``; they run only while the
    served SwiGLU is probed to use that form on this build.
    """
    if not runtime_supported():
        return "Metal runtime unavailable"
    return served_swiglu_refusal()
