# SPDX-License-Identifier: Apache-2.0
# Adapted from jundot/omlx PRs #4041, #4052, #4105 and #4106 (Apache-2.0); see
# docs/PROVENANCE.md and provenance/omlx-4106-moe-window.json.
"""Routed MoE row windows and the router top-k fold for Flash-Next.

A row window runs the routed experts of ``R`` rows (2..``WINDOW_MAX_ROWS``:
the tokens of one verify window, or one token from each of ``B`` decode
lanes) in the launches of the one-token routed decode
(``qwen4_routed_decode``), with every row computing exactly the arithmetic of
its own one-token call:

``routed_rows``
    the split gate+up/SwiGLU kernel and ``served_down`` (the tile4 fused
    down's arithmetic) with the token on grid z. These are the one-token
    kernel objects themselves; only the grid grows (the kernels already
    address x row ``z / TOPK`` and output row ``z``, and served_down takes
    the token from grid z).
``shared_rows``
    the shared-expert fold of ``qwen4_routed_decode.shared_fold_decode``
    for every row: grid z of gate+up is ``token * (TOPK + 1) + j``, the
    down launch takes the token from grid z. The sources are the one-token
    sources with the token's pointer offsets substituted (each substitution
    is asserted to occur exactly once). The shared gate logit stays an input
    computed outside the kernel (an in-kernel 8-bit row was 1 ulp off on 1
    of 36960 full-model calls in lane L2).

Router top-k (omlx #4052; here matched to the served stock routing, not to
omlx's own fused router):

The served block routes with ``mx.softmax(precise=True)``,
``mx.argpartition`` (a full stable ascending sort on Metal: slot j is the
j-th of the ten largest probabilities in ascending order, equal values in
ascending expert order), ``take_along_axis`` and ``scores / scores.sum()``
(a sequential bf16 sum over slots 0..9, then a bf16 divide). A Metal probe
on 60000 rows found that order and that sum exactly
(qualification/runs/omlx-w2a-moe-window-20260930/). ``router_topk`` does all
of it in one launch, one simdgroup per row: lane ``l`` holds what thread
``s * 32 + l`` of MLX's 128-thread ``softmax_single_row`` holds, so every
``simd_max``/``simd_sum`` sees the same values in the same lanes; each
rounded probability packs with its expert into a unique key
``((bf16 bits + 1) << 16) | expert``, so the descending selection is ten
``simd_max`` steps and slot ``s`` is selection ``9 - s``.

``fold`` runs that routing inside the gate+up launch: every simdgroup
recomputes its row's selection from the router logits before streaming its
expert rows, and the first threadgroup of each (row, slot) stores the slot's
expert and score for the down launch. The router gemv (8-bit) stays the
block's own launch.

Admission is structural (no device sync); the caller (``qwen3_next``) checks
the block's one-token reference (split tables, tile4 down at width 1,
stock routing) and counts every decline.
"""

from __future__ import annotations

import os

import mlx.core as mx

from . import qwen4_routed_decode as RD

WINDOW_ENV = "MLX_QWEN4_MOE_WINDOW"
TOPK_ENV = "MLX_QWEN4_MOE_TOPK_FOLD"
WINDOW_SHARED_ENV = "MLX_QWEN4_MOE_WINDOW_SHARED"
# Who may run a row window: row-exact verify windows (lane L5), batched
# one-token decode (B lanes x 1 token) and plain (stock) MTP verify windows.
CONSUMERS = ("row_exact", "batch_decode", "verify")
TOPK_MODES = ("off", "launch", "fold")
# 17 covers the 16-token copy drafts plus the pending token.
WINDOW_MAX_ROWS = 17
NUM_EXPERTS = 512
TOP_K = RD.TOP_K


def consumers_from_env() -> frozenset:
    raw = os.environ.get(WINDOW_ENV, "").strip().lower()
    if raw in ("", "0", "off", "false", "none"):
        return frozenset()
    if raw in ("1", "all", "on", "true"):
        return frozenset(CONSUMERS)
    names = frozenset(part.strip() for part in raw.split(",") if part.strip())
    unknown = names - set(CONSUMERS)
    if unknown:
        raise ValueError(f"{WINDOW_ENV}={raw!r}: unknown consumers {sorted(unknown)}; expected {CONSUMERS}")
    return names


def topk_mode_from_env() -> str:
    raw = os.environ.get(TOPK_ENV, "off").strip().lower()
    raw = {"": "off", "0": "off", "false": "off", "1": "fold"}.get(raw, raw)
    if raw not in TOPK_MODES:
        raise ValueError(f"{TOPK_ENV}={raw!r}: expected one of {TOPK_MODES}")
    return raw


def window_shared_from_env() -> bool:
    raw = os.environ.get(WINDOW_SHARED_ENV, "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


# Live switches (an A/B rotates them in process). Rows above this take the
# routing launch instead of the in-kernel fold: every gate+up simdgroup
# recomputes its row's routing, so the fold's ALU grows with the rows while
# the launch it saves does not. Launch-chain microbench on real layers (M5
# Max, w2a evidence): the fold beats the launch at 1-3 rows, ties at 4 and
# loses at 8 (+5%) and 16 (+11%); omlx #4106 also stops at 3.
_TOPK_FOLD_MAX_ROWS = 3
_WINDOW_SHARED = window_shared_from_env()


def set_topk_fold_max_rows(rows: int) -> int:
    global _TOPK_FOLD_MAX_ROWS
    if not 1 <= int(rows) <= WINDOW_MAX_ROWS:
        raise ValueError(f"fold max rows must be in 1..{WINDOW_MAX_ROWS}")
    _TOPK_FOLD_MAX_ROWS = int(rows)
    return _TOPK_FOLD_MAX_ROWS


def topk_fold_max_rows() -> int:
    return _TOPK_FOLD_MAX_ROWS


def set_window_shared(enabled: bool) -> bool:
    global _WINDOW_SHARED
    _WINDOW_SHARED = bool(enabled)
    return _WINDOW_SHARED


def window_shared_enabled() -> bool:
    return _WINDOW_SHARED


def admit_router_topk(logits, top_k: int, norm_topk_prob: bool) -> str | None:
    """Why the routing kernels may not stand in for the stock routing of
    ``logits`` ([..., NE] router gemv output), or None. Never evaluates."""
    ne = logits.shape[-1]
    rows = logits.size // ne if ne else 0
    if logits.dtype != mx.bfloat16:
        return "router logits must be bfloat16"
    if ne != NUM_EXPERTS:
        return f"expert count {ne} != {NUM_EXPERTS}"
    if top_k != TOP_K or not norm_topk_prob:
        return "only normalized top-10 routing"
    if not 1 <= rows <= WINDOW_MAX_ROWS:
        return f"{rows} rows outside 1..{WINDOW_MAX_ROWS}"
    if not RD.runtime_supported():
        return "Metal runtime unavailable"
    return None


# --------------------------------------------------------------------------
# Routing source. MLX's softmax_single_row (precise: float accumulation,
# N_READS 4, NE / 4 threads in NE / 128 simdgroups) emulated by one
# simdgroup (omlx #4052 ``omlx_router_softmax_row``), then the stock
# argpartition order and normalization (mlx2's: ascending slots, sequential
# bf16 sum, bf16 divide).
# --------------------------------------------------------------------------
ROUTER_HEADER = r"""
template <typename T, int NE>
METAL_FUNC void mlx2_router_keys(const device T* logits, uint lane, thread uint* key) {
  constexpr int S = NE / 128;
  float ld[S][4];
  for (int s = 0; s < S; s++) {
    for (int i = 0; i < 4; i++) {
      ld[s][i] = float(logits[(s * 32 + int(lane)) * 4 + i]);
    }
  }
  // MLX: per-thread max from finite_min, simd_max, the S partials in lanes
  // 0..S-1 of a -inf row (Limits<float>::min), simd_max.
  float lane_max = -INFINITY;
  for (int s = 0; s < S; s++) {
    float m = -metal::numeric_limits<float>::max();
    for (int i = 0; i < 4; i++) {
      m = (m < ld[s][i]) ? ld[s][i] : m;
    }
    m = simd_max(m);
    lane_max = int(lane) == s ? m : lane_max;
  }
  const float maxval = simd_max(lane_max);
  float lane_sum = 0;
  for (int s = 0; s < S; s++) {
    float n = 0;
    for (int i = 0; i < 4; i++) {
      const float e = metal::fast::exp(ld[s][i] - maxval);
      ld[s][i] = e;
      n += e;
    }
    n = simd_sum(n);
    lane_sum = int(lane) == s ? n : lane_sum;
  }
  const float normalizer = 1 / simd_sum(lane_sum);
  for (int s = 0; s < S; s++) {
    for (int i = 0; i < 4; i++) {
      const float v = float(static_cast<T>(ld[s][i] * normalizer));
      const uint expert = uint((s * 32 + int(lane)) * 4 + i);
      // Non-negative bf16 probabilities order like their bit patterns; the
      // expert in the low half breaks ties toward the higher index (the one
      // a stable ascending sort places later). NaN never selects.
      key[s * 4 + i] = metal::isnan(v) ? 0u : ((((as_type<uint>(v) >> 16) + 1u) << 16) | expert);
    }
  }
}

// The ten largest keys of the row, descending (sel[0] largest).
template <int NE>
METAL_FUNC void mlx2_router_select(thread const uint* key, thread uint* sel) {
  constexpr int PER = NE / 32;
  uint below = 0xffffffffu;
  for (int j = 0; j < 10; j++) {
    uint best = 0u;
    for (int k = 0; k < PER; k++) {
      const uint v = key[k];
      best = (v < below && v > best) ? v : best;
    }
    below = simd_max(best);
    sel[j] = below;
  }
}

inline uint mlx2_router_expert(uint sel) {
  return sel == 0u ? 0u : (sel & 0xffffu);
}

inline float mlx2_router_prob(uint sel) {
  return sel == 0u ? 0.0f : as_type<float>(((sel >> 16) - 1u) << 16);
}

// Slot s is selection 9 - s (argpartition's ascending suffix). The stock
// normalization sums the ten bf16 probabilities in slot order in bf16 and
// divides in bf16.
template <typename T>
METAL_FUNC T mlx2_router_score(thread const uint* sel, int slot) {
  T total = T(0.0f);
  for (int s = 0; s < 10; s++) {
    total = static_cast<T>(float(static_cast<T>(mlx2_router_prob(sel[9 - s]))) + float(total));
  }
  const T p = static_cast<T>(mlx2_router_prob(sel[9 - slot]));
  return static_cast<T>(float(p) / float(total));
}
"""

ROUTER_TOPK_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.y;
    uint key[NE / 32];
    mlx2_router_keys<T, NE>(logits + size_t(row) * NE, lane, key);
    uint sel[10];
    mlx2_router_select<NE>(key, sel);
    if (lane < 10) {
      const int slot = int(lane);
      indices[row * 10 + slot] = mlx2_router_expert(sel[9 - slot]);
      scores[row * 10 + slot] = mlx2_router_score<T>(sel, slot);
    }
"""

# Prefix for a gate+up threadgroup serving routed slot ``SLOT_EXPR`` of
# token ``TOKEN_EXPR`` with output element ``ELEM_EXPR``: recompute the
# row's selection, take the slot's expert, and (first threadgroup of the
# element, first simdgroup, first lane) store the expert and its score.
FOLD_PREFIX = r"""
    uint fold_key[NE / 32];
    mlx2_router_keys<T, NE>(logits + size_t(TOKEN_EXPR) * NE, simd_lid, fold_key);
    uint fold_sel[10];
    mlx2_router_select<NE>(fold_key, fold_sel);
    const uint fold_expert = mlx2_router_expert(fold_sel[9 - (SLOT_EXPR)]);
    if (tid.y == 0 && simd_gid == 0 && simd_lid == 0) {
      indices[ELEM_EXPR] = fold_expert;
      scores[ELEM_EXPR] = mlx2_router_score<T>(fold_sel, SLOT_EXPR);
    }
"""


def _replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise AssertionError(f"expected one {old!r} in the one-token source, found {count}")
    return source.replace(old, new)


# Split gate+up with the routing folded in: the one-token source with
# ``rhs[tid.z]`` replaced by the recomputed selection.
FOLD_SPLIT_GATE_UP_SOURCE = _replace_once(
    RD.SPLIT_GATE_UP_SOURCE,
    "    const size_t row0 = size_t(rhs[tid.z]) * NI + out_row;\n",
    FOLD_PREFIX.replace("TOKEN_EXPR", "token")
    .replace("SLOT_EXPR", "int(tid.z) % TOPK")
    .replace("ELEM_EXPR", "tid.z")
    + "    const size_t row0 = size_t(fold_expert) * NI + out_row;\n",
)


def _shared_gate_up_window_source(fold: bool) -> str:
    """The shared-fold gate+up for every row of a window: grid z is
    ``token * (TOPK + 1) + j`` (j 0: shared rows, j >= 1: slot j - 1)."""
    src = RD.SHARED_GATE_UP_SOURCE
    src = _replace_once(
        src,
        "    const uint3 tid = threadgroup_position_in_grid;\n",
        "    const uint3 tid = uint3(threadgroup_position_in_grid.xy,"
        " threadgroup_position_in_grid.z % (TOPK + 1));\n"
        "    const int token = int(threadgroup_position_in_grid.z) / (TOPK + 1);\n"
        "    const device T* xr = x + size_t(token) * K;\n"
        "    device T* yr = y + size_t(token) * (TOPK + 1) * NI;\n",
    )
    src = src.replace(" x, simd_lid, result);", " xr, simd_lid, result);")
    if src.count(" xr, simd_lid, result);") != 2:
        raise AssertionError("shared gate+up source changed: expected two qmv_rows calls")
    src = _replace_once(src, "swiglu_store<T, RPS>(result, y + TOPK * NI + out_row, simd_lid);",
                        "swiglu_store<T, RPS>(result, yr + TOPK * NI + out_row, simd_lid);")
    src = _replace_once(src, "swiglu_store<T, RPS>(result, y + size_t(slot) * NI + out_row, simd_lid);",
                        "swiglu_store<T, RPS>(result, yr + size_t(slot) * NI + out_row, simd_lid);")
    if fold:
        src = _replace_once(
            src,
            "    const size_t row0 = size_t(rhs[slot]) * NI + out_row;\n",
            FOLD_PREFIX.replace("TOKEN_EXPR", "token")
            .replace("SLOT_EXPR", "slot")
            .replace("ELEM_EXPR", "token * TOPK + slot")
            + "    const size_t row0 = size_t(fold_expert) * NI + out_row;\n",
        )
    else:
        src = _replace_once(src, "size_t(rhs[slot])", "size_t(rhs[token * TOPK + slot])")
    return src


def _shared_down_window_source() -> str:
    """The shared-fold down for every row of a window: grid z is the token."""
    src = RD.SHARED_DOWN_SOURCE
    src = _replace_once(
        src,
        "    uint row_base = threadgroup_position_in_grid.y * RPS;\n",
        "    uint row_base = threadgroup_position_in_grid.y * RPS;\n"
        "    const uint token = threadgroup_position_in_grid.z;\n"
        "    const device T* hr = hidden + size_t(token) * (TOPK + 1) * EH;\n",
    )
    src = _replace_once(src, "hidden + TOPK * EH, lane, result);", "hr + TOPK * EH, lane, result);")
    src = _replace_once(src, "uint expert = uint(rhs[slot]);", "uint expert = uint(rhs[token * TOPK + slot]);")
    src = _replace_once(src, "const device T* hrow = hidden + size_t(slot) * EH;",
                        "const device T* hrow = hr + size_t(slot) * EH;")
    src = _replace_once(src, "float score = float(scores[slot]);",
                        "float score = float(scores[token * TOPK + slot]);")
    src = _replace_once(src, "const float g = float(gate[0]);", "const float g = float(gate[token]);")
    src = _replace_once(src, "out[row_base + lane] =", "out[size_t(token) * H + row_base + lane] =")
    return src


_KERNELS: dict = {}


def _kernel(name: str):
    kernel = _KERNELS.get(name)
    if kernel is not None:
        return kernel
    if name == "router_topk":
        kernel = mx.fast.metal_kernel(
            name="mlx2_qwen4_moe_router_topk_rows",
            input_names=["logits"],
            output_names=["indices", "scores"],
            header="using namespace metal;\n" + ROUTER_HEADER,
            source=ROUTER_TOPK_SOURCE,
        )
    elif name == "fold_gate_up":
        kernel = mx.fast.metal_kernel(
            name="mlx2_qwen4_moe_split_gate_up_topk_fold",
            input_names=["x", "wg", "sg", "bg", "wu", "su", "bu", "logits"],
            output_names=["y", "indices", "scores"],
            header=RD._header(fast=True) + ROUTER_HEADER,
            source=FOLD_SPLIT_GATE_UP_SOURCE,
        )
    elif name in ("shared_gate_up", "shared_gate_up_fold"):
        fold = name.endswith("_fold")
        inputs = ["x", "wg", "sg", "bg", "wu", "su", "bu", "logits" if fold else "rhs",
                  "shw", "shs", "shb", "suw", "sus", "sub"]
        kernel = mx.fast.metal_kernel(
            name="mlx2_qwen4_moe_gate_up_shared_window" + ("_topk_fold" if fold else ""),
            input_names=inputs,
            output_names=["y", "indices", "scores"] if fold else ["y"],
            header=RD.SHARED_COMMON + RD._format_header("q4f", 4, True)
            + (ROUTER_HEADER if fold else ""),
            source=_shared_gate_up_window_source(fold),
        )
    elif name == "shared_down":
        kernel = mx.fast.metal_kernel(
            name="mlx2_qwen4_moe_served_down_shared_window",
            input_names=["hidden", "w", "scales", "biases", "rhs", "scores", "sdw", "sds", "sdb", "gate"],
            output_names=["out"],
            header="using namespace metal;\n" + RD._format_header("q4s", 4, False),
            source=_shared_down_window_source(),
        )
    else:
        raise KeyError(name)
    _KERNELS[name] = kernel
    return kernel


def _rows(x) -> int:
    return int(x.size // x.shape[-1])


def router_topk(logits):
    """Stock-identical ``(indices uint32, scores)`` [rows, 10] of bf16 router
    logits [..., 512] in one launch. Admission is the caller's."""
    ne = logits.shape[-1]
    rows = _rows(logits)
    return _kernel("router_topk")(
        inputs=[logits.reshape(rows, ne)],
        template=[("T", logits.dtype), ("NE", ne)],
        grid=(32, rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, TOP_K), (rows, TOP_K)],
        output_dtypes=[mx.uint32, logits.dtype],
    )


def _gate_up_template(x, inter):
    return [
        ("T", x.dtype),
        ("K", x.shape[-1]),
        ("NI", inter),
        ("RPS", RD.GATE_UP_ROWS),
        ("NSG", RD.GATE_UP_SIMDGROUPS),
        ("TOPK", TOP_K),
    ]


def _served_down(h, indices, scores, down, rows, hidden, inter, dtype, rps):
    return RD._kernels()[3](
        inputs=[h, *RD.expert_operands(down), indices.reshape(rows * TOP_K),
                scores.reshape(rows * TOP_K)],
        template=[("T", dtype), ("H", hidden), ("EH", inter), ("TOPK", TOP_K), ("RPS", rps)],
        grid=(32 * RD.SERVED_DOWN_SIMDGROUPS, hidden // rps, rows),
        threadgroup=(32 * RD.SERVED_DOWN_SIMDGROUPS, 1, 1),
        output_shapes=[(rows, hidden)],
        output_dtypes=[dtype],
    )[0]


def routed_rows(x, indices, scores, gate, up, down, *, logits=None, rows_per_tg=None):
    """``tile4_down(swiglu(gate_j(x_r), up_j(x_r)))`` weighted by the routing,
    for every row ``r`` of ``x`` ([..., hidden]) in two launches: [rows, hidden].

    With ``logits`` (the rows' router gemv output), the routing is computed
    inside the gate+up launch and ``indices``/``scores`` must be None; the
    result is then ``(y, indices, scores)``. Admission is the caller's
    (split routed + tile4 at width 1, per row)."""
    rps = RD.served_down_rows() if rows_per_tg is None else rows_per_tg
    hidden = x.shape[-1]
    rows = _rows(x)
    inter = gate["weight"].shape[1]
    xr = x.reshape(rows, hidden)
    grid = (32, RD.GATE_UP_SIMDGROUPS * inter // (RD.GATE_UP_ROWS * RD.GATE_UP_SIMDGROUPS),
            rows * TOP_K)
    if logits is None:
        h = RD._kernels()[2](
            inputs=[xr, *RD.expert_operands(gate), *RD.expert_operands(up),
                    indices.reshape(rows * TOP_K).astype(mx.uint32)],
            template=_gate_up_template(x, inter),
            grid=grid,
            threadgroup=(32, RD.GATE_UP_SIMDGROUPS, 1),
            output_shapes=[(rows * TOP_K, inter)],
            output_dtypes=[x.dtype],
        )[0]
        return _served_down(h, indices, scores, down, rows, hidden, inter, x.dtype, rps)
    h, indices, scores = _kernel("fold_gate_up")(
        inputs=[xr, *RD.expert_operands(gate), *RD.expert_operands(up),
                logits.reshape(rows, logits.shape[-1])],
        template=_gate_up_template(x, inter) + [("NE", logits.shape[-1])],
        grid=grid,
        threadgroup=(32, RD.GATE_UP_SIMDGROUPS, 1),
        output_shapes=[(rows * TOP_K, inter), (rows * TOP_K,), (rows * TOP_K,)],
        output_dtypes=[x.dtype, mx.uint32, x.dtype],
    )
    y = _served_down(h, indices, scores, down, rows, hidden, inter, x.dtype, rps)
    return y, indices.reshape(rows, TOP_K), scores.reshape(rows, TOP_K)


def shared_rows(x, indices, scores, gate, up, down, shared, gate_logit, *, logits=None,
                rows_per_tg=None):
    """``tile4 routed + sigmoid(gate_logit) * shared_expert(x)`` for every
    row of ``x`` in two launches: [rows, hidden]. ``gate_logit`` ([rows, 1]
    or [rows]) is computed outside (one-row arithmetic per row). With
    ``logits`` the routing runs inside gate+up and the result is
    ``(y, indices, scores)``. Admission is the caller's (split routed + tile4
    + ``admit_shared_fold``)."""
    rps = RD.served_down_rows() if rows_per_tg is None else rows_per_tg
    if rps not in RD.DOWN_ROWS_SERVED_CHOICES:
        raise ValueError(f"served down rows must be one of {RD.DOWN_ROWS_SERVED_CHOICES}")
    hidden = x.shape[-1]
    rows = _rows(x)
    inter = gate["weight"].shape[1]
    xr = x.reshape(rows, hidden)
    routing = (
        [logits.reshape(rows, logits.shape[-1])]
        if logits is not None
        else [indices.reshape(rows * TOP_K).astype(mx.uint32)]
    )
    template = [("T", x.dtype), ("K", hidden), ("NI", inter), ("RPS", RD.GATE_UP_ROWS),
                ("NSG", RD.GATE_UP_SIMDGROUPS), ("TOPK", TOP_K)]
    common = dict(
        grid=(32, inter // RD.GATE_UP_ROWS, rows * (TOP_K + 1)),
        threadgroup=(32, RD.GATE_UP_SIMDGROUPS, 1),
    )
    operands = [xr, *RD.expert_operands(gate), *RD.expert_operands(up), *routing,
                *RD._dense_operands(shared.gate_proj), *RD._dense_operands(shared.up_proj)]
    if logits is None:
        h = _kernel("shared_gate_up")(
            inputs=operands, template=template,
            output_shapes=[(rows * (TOP_K + 1) * inter,)], output_dtypes=[x.dtype], **common,
        )[0]
    else:
        h, indices, scores = _kernel("shared_gate_up_fold")(
            inputs=operands, template=template + [("NE", logits.shape[-1])],
            output_shapes=[(rows * (TOP_K + 1) * inter,), (rows * TOP_K,), (rows * TOP_K,)],
            output_dtypes=[x.dtype, mx.uint32, x.dtype], **common,
        )
    y = _kernel("shared_down")(
        inputs=[h, *RD.expert_operands(down), indices.reshape(rows * TOP_K),
                scores.reshape(rows * TOP_K), *RD._dense_operands(shared.down_proj),
                gate_logit.reshape(rows)],
        template=[("T", x.dtype), ("H", hidden), ("EH", inter), ("TOPK", TOP_K), ("RPS", rps)],
        grid=(32 * (RD.SERVED_DOWN_SIMDGROUPS + 1), hidden // rps, rows),
        threadgroup=(32 * (RD.SERVED_DOWN_SIMDGROUPS + 1), 1, 1),
        output_shapes=[(rows, hidden)],
        output_dtypes=[x.dtype],
    )[0]
    if logits is None:
        return y
    return y, indices.reshape(rows, TOP_K), scores.reshape(rows, TOP_K)
