# SPDX-License-Identifier: Apache-2.0
"""Default-off Qwen3.6 MoE slices with the stock qmv_fast down contract.

The Flash-Next served_down accumulates weighted slots in float32; Qwen3.6
stock reduces bf16 products. This specialization retains the latter, including
its slot order. See provenance/qwen36-flash-next-decode.json. No speed claim.
"""

import os
from functools import lru_cache

import mlx.core as mx

from . import qwen3_next as QN
from . import qwen4_moe_window as W
from . import qwen4_routed_decode as RD
from .import_env import snapshot
from .precise_ops import gate_sigmoid

snapshot(__name__)
_WINDOW = os.environ.get("MLX_QWEN36_MOE_WINDOW", "0") == "1"


def shared_admission(shared, hidden, inter):
    """Qwen3.6 shared down is qmv_fast at 4/8 bits, group 64/128."""
    if shared is None or hasattr(shared, "_prefill_counts"):
        return "shared expert missing or prefill replacement installed"
    gate, up, down = (
        shared.get("gate_proj"),
        shared.get("up_proj"),
        shared.get("down_proj"),
    )
    for name, layer, n, k in [
        ("gate", gate, inter, hidden),
        ("up", up, inter, hidden),
        ("down", down, hidden, inter),
    ]:
        bits = getattr(layer, "bits", None)
        if bits not in RD.SHARED_FORMAT_BITS:
            return f"shared {name}: only 4/8-bit affine tables"
        gs = getattr(layer, "group_size", None)
        if gs not in (64, 128):
            return f"shared {name}: group size must be 64 or 128"
        reason = RD._linear_ok(layer, bits, n, k, group_size=gs)
        if reason:
            return f"shared {name}: {reason}"
        if not RD.qmv_fast_layout(k, n, bits):
            return f"shared {name}: not qmv_fast"
    if (gate.bits, gate.group_size) != (up.bits, up.group_size):
        return "shared gate/up formats differ"
    return None


def down_source(shared=False):
    """Stock-down candidate body with row offsets and optional shared qmv.
    The historical candidate and every Flash-Next kernel stay unchanged."""
    source = RD.CANDIDATE_DOWN_SOURCE
    if shared:
        source = source.replace("    const size_t expert = size_t(rhs[slot]);", "")
        marker = "    const device uint8_t* ws ="
        begin = source.index(marker)
        end = source.index("    threadgroup_barrier(mem_flags::mem_threadgroup);")
        routed = source[begin:end]
        source = (
            source[:begin]
            + """    threadgroup T shared_part[RPS];
    if (slot == TOPK) {
      float result[RPS];
      shared_qmv::qmv_rows<T, K, RPS, 0>(
          (const device uint8_t*)sdw, sds, sdb, out_row,
          (const device uint8_t*)sdw, sds, sdb, out_row,
          x + TOPK * K, simd_lid, result);
      if (simd_lid == 0) {
        for (int r = 0; r < RPS; ++r) shared_part[r] = static_cast<T>(result[r]);
      }
    } else {
      const size_t expert = size_t(rhs[slot]);
"""
            + routed
            + "    }\n"
            + source[end:]
        )
        source = source.replace(
            "      y[out_row + row] = acc;",
            """      T weighted_shared = shared_part[row] * gate[0];
      y[out_row + row] = acc + weighted_shared;""",
        )
    extent = "TOPK + 1" if shared else "TOPK"
    buffers = {"x": f"({extent}) * K", "rhs": "TOPK", "scores": "TOPK", "y": "N"}
    if shared:
        buffers["gate"] = "1"
    prefix = "  const uint token = threadgroup_position_in_grid.z;\n"
    prefix += "".join(
        f"  const auto {n}_row = {n} + (size_t)token * ({v});\n"
        for n, v in buffers.items()
    )
    prefix += "  {\n" + "".join(f"  const auto {n} = {n}_row;\n" for n in buffers)
    return prefix + source + "\n  }\n"


@lru_cache(maxsize=8)
def down_kernel(shared_format=None):
    shared = shared_format is not None
    shared_bits, shared_gs = shared_format if shared else (None, None)
    names = ["x", "w", "scales", "biases", "rhs", "scores"]
    header = RD._header(fast=True)
    if shared:
        names += ["sdw", "sds", "sdb", "gate"]
        header += RD._format_header("shared_qmv", shared_bits, True, shared_gs)
    return mx.fast.metal_kernel(
        name="qwen36_stock_down_rows"
        + (f"_shared{shared_bits}g{shared_gs}" if shared else ""),
        input_names=names,
        output_names=["y"],
        header=header,
        source=down_source(shared),
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=4)
def shared_gate_up_kernel(bits, group_size):
    # A separate namespace keeps routed g64 and shared g128 affine scales
    # independent even when both projections are 4-bit.
    source = W._shared_gate_up_window_source(False).replace(
        "q4f::qmv_rows<", "shared_gate_qmv::qmv_rows<", 1
    )
    header = (
        RD.SHARED_COMMON
        + RD._format_header("q4f", 4, True)
        + RD._format_header("shared_gate_qmv", bits, True, group_size)
    )
    return mx.fast.metal_kernel(
        name=f"qwen36_gate_up_shared_rows_b{bits}g{group_size}",
        input_names=[
            "x",
            "wg",
            "sg",
            "bg",
            "wu",
            "su",
            "bu",
            "rhs",
            "shw",
            "shs",
            "shb",
            "suw",
            "sus",
            "sub",
        ],
        output_names=["y"],
        header=header,
        source=source,
        ensure_row_contiguous=True,
    )


def routed_rows(x, indices, scores, gate, up, down, *, shared=None, shared_gate=None):
    """Exact one-token MoE arithmetic for each window row (Metal gate pending)."""
    hidden = x.shape[-1]
    inter = gate.weight.shape[1]
    rows = x.size // hidden
    top_k = 8
    flat = x.reshape(rows, hidden)
    template = [
        ("T", x.dtype),
        ("K", hidden),
        ("NI", inter),
        ("RPS", RD.GATE_UP_ROWS),
        ("NSG", RD.GATE_UP_SIMDGROUPS),
        ("TOPK", top_k),
    ]
    if shared is None:
        h = RD._kernels()[2](
            inputs=[
                flat,
                *RD.expert_operands(gate),
                *RD.expert_operands(up),
                indices.reshape(-1).astype(mx.uint32),
            ],
            template=template,
            grid=(32, inter // RD.GATE_UP_ROWS, rows * top_k),
            threadgroup=(32, RD.GATE_UP_SIMDGROUPS, 1),
            output_shapes=[(rows, top_k, inter)],
            output_dtypes=[x.dtype],
        )[0]
        extra = []
        shared_format = None
    else:
        h = shared_gate_up_kernel(shared.gate_proj.bits, shared.gate_proj.group_size)(
            inputs=[
                flat,
                *RD.expert_operands(gate),
                *RD.expert_operands(up),
                indices.reshape(-1).astype(mx.uint32),
                *RD._dense_operands(shared.gate_proj),
                *RD._dense_operands(shared.up_proj),
            ],
            template=template,
            grid=(32, inter // RD.GATE_UP_ROWS, rows * (top_k + 1)),
            threadgroup=(32, RD.GATE_UP_SIMDGROUPS, 1),
            output_shapes=[(rows, top_k + 1, inter)],
            output_dtypes=[x.dtype],
        )[0]
        extra = [*RD._dense_operands(shared.down_proj), shared_gate.reshape(rows)]
        shared_format = (shared.down_proj.bits, shared.down_proj.group_size)
    rps = RD.served_down_rows()
    return down_kernel(shared_format)(
        inputs=[
            h,
            *RD.expert_operands(down),
            indices.reshape(-1).astype(mx.uint32),
            scores.reshape(-1),
            *extra,
        ],
        template=[
            ("T", x.dtype),
            ("K", inter),
            ("N", hidden),
            ("RPS", rps),
            ("TOPK", top_k),
        ],
        grid=(32, hidden // rps, rows),
        threadgroup=(32, top_k + int(shared is not None), 1),
        output_shapes=[(rows, hidden)],
        output_dtypes=[x.dtype],
    )[0]


class Qwen36SparseMoeBlock(QN.Qwen3NextSparseMoeBlock):
    """Adapter-owned specialization. Ordinary reference is the parent block."""

    def __init__(self, args):
        super().__init__(args)
        self.qwen36_window_consumers = (
            frozenset(("batch_decode", "verify", "row_exact"))
            if _WINDOW
            else frozenset()
        )
        self.qwen36_decode_calls = 0
        self.qwen36_decode_fallbacks = 0
        self.qwen36_decode_last_fallback = None
        object.__setattr__(self, "qwen36_decode_reasons", {})

    def set_moe_window_consumers(self, consumers):
        consumers = frozenset(consumers)
        if consumers - set(W.CONSUMERS):
            raise ValueError("unknown MoE window consumer")
        self.qwen36_window_consumers = consumers
        return consumers

    def _decline(self, reason):
        self.qwen36_decode_fallbacks += 1
        self.qwen36_decode_last_fallback = reason
        reasons = self.qwen36_decode_reasons
        if reason not in reasons and len(reasons) >= 16:
            reason = "other"
        reasons[reason] = reasons.get(reason, 0) + 1
        return None

    def __call__(self, x):
        rows = x.size // x.shape[-1]
        consumer = QN._moe_window_consumer(x) if rows > 1 else None
        routed_decode = rows == 1 and self.switch_mlp.routed_decode_mode in (
            "gate_up_down",
            "gate_up_down_shared",
        )
        shared_decode = (
            routed_decode
            and self.switch_mlp.routed_decode_mode == "gate_up_down_shared"
        )
        window = rows > 1 and consumer in self.qwen36_window_consumers
        if not (routed_decode or window):
            return super().__call__(x)
        if not 1 <= rows <= W.WINDOW_MAX_ROWS:
            self._decline("window width outside 1..17")
            return super().__call__(x)
        sw = self.switch_mlp
        reason = None
        if self.shared_folded or self.sharding_group is not None:
            reason = "shared row folded or sharded"
        elif self.training or sw.training:
            reason = "training"
        elif self.fused_expert_kernel_mode != "stock":
            reason = "Qwen3.6 rows require stock down"
        elif (
            self.moe_router_mode != "stock" or QN._MOE_GATE_COMPILE or QN._COMPILE_GLUE
        ):
            reason = "alternative router or compiled combine selected"
        elif not self.norm_topk_prob:
            reason = "only normalized top-8 routing"
        elif self.top_k != 8 or self.num_experts != 256:
            reason = "unsupported topology"
        elif "gate_up_proj" in sw:
            reason = "requires split gate/up tables"
        else:
            admit = RD.admit_split_routed_decode(
                QN._ShapeOnly((x.shape[-1],), x.dtype),
                QN._ShapeOnly((8,), mx.uint32),
                QN._ShapeOnly((8,), x.dtype),
                sw.gate_proj,
                sw.up_proj,
                sw.down_proj,
            )
            if not admit.accepted:
                reason = admit.reason
        if reason is None and not RD.runtime_supported():
            reason = "Metal runtime unavailable"
        if reason is None:
            reason = RD.served_swiglu_refusal()
        if reason:
            self._decline(reason)
            return super().__call__(x)
        # Router and shared projections retain one-token launch geometry.
        flat = x.reshape(rows, -1)
        logits = mx.concatenate(
            [
                self.gate(flat[r : r + 1].reshape(1, 1, -1)).reshape(1, -1)
                for r in range(rows)
            ]
        )
        refusal = None
        if self.moe_topk_mode == "launch":
            # The same structural admission the parent's launch applies; the
            # kernel transcribes stock routing for bf16 logits only.
            refusal = W.admit_router_topk(logits, 8, bool(self.norm_topk_prob))
            if refusal is None and logits.dtype != x.dtype:
                refusal = "router logits dtype differs from the activation"
            if refusal is not None:
                self.moe_topk_fallbacks += 1
                self.moe_topk_last_fallback = refusal
        if self.moe_topk_mode == "launch" and refusal is None:
            inds, scores = W.router_topk(logits, top_k=8)
            self.moe_topk_calls["launch"] += 1
            self.moe_topk_last_fallback = None
        else:
            inds, scores = QN._stock_routing(self, logits)
        shared = self.shared_expert
        shared_reason = shared_admission(
            shared, x.shape[-1], sw.gate_proj.weight.shape[1]
        )
        fold = (shared_decode or window) and shared_reason is None
        if (shared_decode or window) and shared_reason is not None:
            self.shared_fold_fallbacks += 1
            self.shared_fold_last_fallback = shared_reason
        shared_gate = mx.concatenate(
            [
                gate_sigmoid(
                    self.shared_expert_gate(flat[r : r + 1].reshape(1, 1, -1))
                ).reshape(1, 1)
                for r in range(rows)
            ]
        )
        y = routed_rows(
            x,
            inds,
            scores,
            sw.gate_proj,
            sw.up_proj,
            sw.down_proj,
            shared=shared if fold else None,
            shared_gate=shared_gate if fold else None,
        )
        if not fold:
            shared_y = mx.concatenate(
                [
                    shared(flat[r : r + 1].reshape(1, 1, -1)).reshape(1, -1)
                    for r in range(rows)
                ]
            )
            y = y + shared_gate * shared_y
        self.qwen36_decode_calls += 1
        self.qwen36_decode_last_fallback = None
        if window:
            self.moe_window_calls[consumer] += 1
            self.moe_window_rows += rows
        if fold:
            self.shared_fold_calls += 1
        return y.reshape(x.shape)
