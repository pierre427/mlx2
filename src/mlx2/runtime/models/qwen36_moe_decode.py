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
from . import qwen36_moe_grouped as G
from . import qwen4_routed_decode as RD
from .import_env import snapshot
from .qwen_moe_contracts import QWEN36_E256_K8_ROUTING
from .precise_ops import gate_sigmoid

snapshot(__name__)
_WINDOW = os.environ.get("MLX_QWEN36_MOE_WINDOW", "0") == "1"
# Expert-grouped multi-row routed experts (``qwen36_moe_grouped``), default
# off; ``Qwen36SparseMoeBlock.set_grouped_rows`` switches it live.  "1" is
# the stock-exact law ("qmv"); "lane" selects the tensor-unit law.
_GROUPED_ROWS = {"0": "off", "1": "qmv", "qmv": "qmv", "lane": "lane"}.get(
    os.environ.get("MLX_QWEN36_MOE_GROUPED_ROWS", "0"), "off"
)

# Routed expert table formats the row kernels serve.  The kernels are written
# over the shared QMV header, which carries MLX's 6-bit unpack; 6-bit g64
# tables (the mixed-precision 4-bit artifacts keep half their MoE layers at
# 6 bits) are admitted for the one-token decode only, where the GPU check
# (``qwen36-q6-routed-decode``) compared them bitwise with the stock block.
# Multi-row window consumers keep the 4/8-bit formats they were checked on.
ROUTED_FORMATS = ((4, 64), (8, 64))
DECODE_ONLY_ROUTED_FORMATS = ((6, 64),)
# Widest stock-block call that takes the short multi-row glue (see
# ``Qwen36SparseMoeBlock.__init__``).
MULTI_ROW_GLUE_ROWS = 4
# Widest call that takes the one-launch routing while the grouped rows serve
# the experts (the widths that routing was checked on).
GROUPED_TOPK_LAUNCH_ROWS = 16
GROUPED_ROWS_LAWS = ("off", "qmv", "lane")


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


@lru_cache(maxsize=16)
def down_kernel(
    routed_bits=4, routed_group_size=64, shared_format=None, fast=True
):
    """``fast``: MLX's qmv_fast layout (K in whole 512-wide blocks); else the
    qmv layout with a guarded K tail, as MLX picks for the same shape."""
    shared = shared_format is not None
    shared_bits, shared_gs = shared_format if shared else (None, None)
    names = ["x", "w", "scales", "biases", "rhs", "scores"]
    header = RD._header(
        fast=fast, bits=routed_bits, group_size=routed_group_size
    )
    if shared:
        names += ["sdw", "sds", "sdb", "gate"]
        header += RD._format_header("shared_qmv", shared_bits, True, shared_gs)
    return mx.fast.metal_kernel(
        name=f"qwen36_stock_down_rows_q{routed_bits}g{routed_group_size}"
        + (f"_shared{shared_bits}g{shared_gs}" if shared else "")
        + ("" if fast else "_qmv"),
        input_names=names,
        output_names=["y"],
        header=header,
        source=down_source(shared),
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=8)
def routed_gate_up_kernel(bits, group_size=64):
    return mx.fast.metal_kernel(
        name=f"qwen36_split_gate_up_rows_q{bits}g{group_size}",
        input_names=["x", "wg", "sg", "bg", "wu", "su", "bu", "rhs"],
        output_names=["y"],
        header=RD._header(fast=True, bits=bits, group_size=group_size),
        source=RD.SPLIT_GATE_UP_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=8)
def shared_gate_up_kernel(routed_bits, bits, group_size):
    # A separate namespace keeps routed g64 and shared g128 affine scales
    # independent even when both projections are 4-bit.
    source = W._shared_gate_up_window_source(False).replace(
        "q4f::qmv_rows<", "shared_gate_qmv::qmv_rows<", 1
    )
    source = source.replace("q4f::", "routed_qmv::")
    common = RD.SHARED_COMMON.replace(
        RD.SIGMOID, RD.SWIGLU_GATE.adapt(RD.SIGMOID, mx.bfloat16)
    )
    header = (
        common
        + RD._format_header("routed_qmv", routed_bits, True)
        + RD._format_header("shared_gate_qmv", bits, True, group_size)
    )
    return mx.fast.metal_kernel(
        name=(
            f"qwen36_gate_up_shared_rows_q{routed_bits}"
            f"_shared{bits}g{group_size}"
        ),
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
    """Exact one-token MoE arithmetic for each window row.

    Focused Q4/Q8 Metal checks pass; the full qualification gate remains pending.
    """
    hidden = x.shape[-1]
    inter = gate.weight.shape[1]
    rows = x.size // hidden
    top_k = 8
    routed_format = (gate.bits, gate.group_size)
    if routed_format not in ROUTED_FORMATS + DECODE_ONLY_ROUTED_FORMATS:
        raise ValueError(
            f"unsupported routed expert format q{gate.bits}g{gate.group_size}"
        )
    if (up.bits, up.group_size) != routed_format or (
        down.bits,
        down.group_size,
    ) != routed_format:
        raise ValueError("routed gate/up/down formats differ")
    routed_bits, routed_group_size = routed_format
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
        h = routed_gate_up_kernel(routed_bits, routed_group_size)(
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
        h = shared_gate_up_kernel(
            routed_bits, shared.gate_proj.bits, shared.gate_proj.group_size
        )(
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
    # One simdgroup per slot; the grid counts threads, so it spans every slot.
    slots = top_k + int(shared is not None)
    # The split admission takes non-512 intermediates too; a qmv_fast walk
    # over those reads past each down row (no guarded tail).
    fast = RD.qmv_fast_layout(inter, hidden, routed_bits)
    return down_kernel(
        routed_bits, routed_group_size, shared_format, fast
    )(
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
        grid=(32, slots * hidden // rps, rows),
        threadgroup=(32, slots, 1),
        output_shapes=[(rows, hidden)],
        output_dtypes=[x.dtype],
    )[0]


def _grouped_rows_hook(switch, x, indices):
    """``FusedDownSwitchGLU`` hook: per-slot outputs from the grouped kernels.

    Declines (counted, ``None``) for one row and every call the structural
    admission refuses, so the module's stock gathers run unchanged.  Law
    ``qmv`` is bitwise the stock gathers; ``lane`` is the tensor-unit law.
    """
    rows = x.size // x.shape[-1]
    if rows < 2:
        return None
    stats = switch.__dict__["grouped_rows_stats"]
    law = switch.__dict__.get("grouped_rows_law", "qmv")
    gate, up, down = switch.gate_proj, switch.up_proj, switch.down_proj
    reason = G.admission_refusal(x, indices, gate, up, down)
    if reason is None and law == "lane":
        reason = G.lane_refusal(gate, up, down)
    if reason is not None:
        stats["fallbacks"] += 1
        reasons = stats["reasons"]
        if reason not in reasons and len(reasons) >= 16:
            reason = "other"
        reasons[reason] = reasons.get(reason, 0) + 1
        return None
    stats["calls"] += 1
    stats["rows"] += rows
    if law == "lane":
        return G.lane_grouped_rows(x, indices, gate, up, down)
    return G.grouped_rows(x, indices, gate, up, down)


def qwen36_grouped_rows_stats(model, *, reset=False):
    """Aggregate grouped-rows calls, rows and counted refusals of a model."""
    report = {"selected_blocks": 0, "calls": 0, "rows": 0, "fallbacks": 0, "reasons": {}}
    for _, module in model.named_modules():
        if isinstance(module, Qwen36SparseMoeBlock):
            law = module.grouped_rows_selected()
            report["selected_blocks"] += int(law != "off")
            report.setdefault("laws", {})[law] = report.get("laws", {}).get(law, 0) + 1
            stats = module.grouped_rows_stats(reset=reset)
            for key in ("calls", "rows", "fallbacks"):
                report[key] += stats[key]
            for reason, count in stats["reasons"].items():
                report["reasons"][reason] = report["reasons"].get(reason, 0) + count
    return report


def router_and_shared_gate_rows(block, flat):
    """Project a Qwen3.6 MoE window with one-row arithmetic per row.

    The Q8 artifact stores both the E256 router and the scalar shared-expert
    gate as affine quantized linears with the same input and format.  The
    shared row-exact package can therefore read both tables in one grouped
    launch while retaining an independent one-row accumulator for every
    token.  Its canary owns admission and falls back member-by-member (and
    ultimately row-by-row) for dense, mixed, lane-prepared, or mismatched
    layouts.  A one-token decode keeps its established calls byte-for-byte.
    """
    rows = int(flat.shape[0])
    if rows <= 1:
        x = flat.reshape(1, 1, -1)
        return block.gate(x).reshape(1, -1), block.shared_expert_gate(x).reshape(
            1, 1
        )
    return QN._row_projections(
        None, (block.gate, block.shared_expert_gate), flat
    )


class Qwen36SparseMoeBlock(QN.Qwen3NextSparseMoeBlock):
    """Adapter-owned specialization. Ordinary reference is the parent block."""

    def __init__(self, args):
        super().__init__(args)
        self.moe_routing_contract = QWEN36_E256_K8_ROUTING
        self.switch_mlp.moe_routing_contract = self.moe_routing_contract
        self.qwen36_window_consumers = (
            frozenset(("batch_decode", "verify", "row_exact"))
            if _WINDOW
            else frozenset()
        )
        # Short multi-row forwards (2..4 tokens: MTP verify, small batches)
        # that reach the stock block: the routing runs as the one-launch
        # stock-identical kernel and the expert gathers stay unsorted.  Both
        # are bitwise equal to the stock block on the 35B at 2..16 rows (GPU
        # check ``qwen36-moe-rows-glue``) and 3-7% faster per forward at 2..4
        # rows; from 6 rows the timing was within noise or slower (the sort's
        # expert reuse wins at 16), so wider calls keep the stock glue.
        object.__setattr__(self, "moe_topk_launch_max_rows", MULTI_ROW_GLUE_ROWS)
        object.__setattr__(
            self.switch_mlp,
            "moe_sort_min_assignments",
            MULTI_ROW_GLUE_ROWS * self.top_k + 1,
        )
        self.qwen36_decode_calls = 0
        self.qwen36_decode_fallbacks = 0
        self.qwen36_decode_last_fallback = None
        object.__setattr__(self, "qwen36_decode_reasons", {})
        object.__setattr__(
            self.switch_mlp,
            "grouped_rows_stats",
            {"calls": 0, "rows": 0, "fallbacks": 0, "reasons": {}},
        )
        self.set_grouped_rows(_GROUPED_ROWS)

    def set_grouped_rows(self, enabled):
        """Select the expert-grouped multi-row routed experts for every
        multi-row call of this block's switch module: ``"qmv"`` (or True,
        bitwise the stock gathers), ``"lane"`` (tensor-unit law, a separate
        numerical law) or ``"off"`` (or False, the stock gathers)."""
        law = {True: "qmv", False: "off", None: "off"}.get(enabled, enabled)
        if law not in GROUPED_ROWS_LAWS:
            raise ValueError(f"grouped rows law must be one of {GROUPED_ROWS_LAWS}")
        enabled = law != "off"
        sw = self.switch_mlp
        if enabled:
            object.__setattr__(sw, "grouped_rows_hook", _grouped_rows_hook)
            object.__setattr__(sw, "grouped_rows_law", law)
        else:
            sw.__dict__.pop("grouped_rows_hook", None)
            sw.__dict__.pop("grouped_rows_law", None)
        # The stock-identical one-launch routing was checked bit-equal on the
        # 35B at 2..16 rows (``qwen36-moe-rows-glue``); the default stops at
        # MULTI_ROW_GLUE_ROWS only because wider stock calls preferred the
        # sorted gathers, which the grouped rows replace.
        object.__setattr__(
            self,
            "moe_topk_launch_max_rows",
            GROUPED_TOPK_LAUNCH_ROWS if enabled else MULTI_ROW_GLUE_ROWS,
        )
        return enabled

    def grouped_rows_selected(self):
        return self.switch_mlp.__dict__.get("grouped_rows_law", "off")

    def grouped_rows_stats(self, reset=False):
        stats = self.switch_mlp.__dict__["grouped_rows_stats"]
        report = {**stats, "reasons": dict(stats["reasons"])}
        if reset:
            stats.update(calls=0, rows=0, fallbacks=0)
            stats["reasons"].clear()
        return report

    def set_moe_window_consumers(self, consumers):
        consumers = frozenset(consumers)
        if consumers - self.moe_routing_contract.allowed_row_consumers:
            raise ValueError("unknown MoE window consumer")
        self.qwen36_window_consumers = consumers
        return consumers

    def row_exact_window_selected(self):
        """Declaration for the row-exact target executor: whether a
        multi-row window call runs this block's ``row_exact`` window kernels
        (explicitly selected consumer) over a routed one-token decode route.
        Otherwise the executor runs one one-token block call per row."""
        return (
            "row_exact" in self.qwen36_window_consumers
            and self.switch_mlp.routed_decode_mode
            in ("gate_up_down", "gate_up_down_shared")
        )

    def row_exact_window_refusal(self, x):
        """Why the selected multi-row exact window cannot run, or ``None``.

        Selection is a policy statement; admission also depends on the loaded
        artifact's tables and the live runtime.  The row-exact executor uses
        this probe to retain its exact per-row fallback when those differ.
        """
        return self._routed_refusal(x, consumer="row_exact")

    def _decline(self, reason):
        self.qwen36_decode_fallbacks += 1
        self.qwen36_decode_last_fallback = reason
        reasons = self.qwen36_decode_reasons
        if reason not in reasons and len(reasons) >= 16:
            reason = "other"
        reasons[reason] = reasons.get(reason, 0) + 1
        return None

    def _routed_refusal(self, x, *, consumer):
        rows = x.size // x.shape[-1]
        if not 1 <= rows <= W.WINDOW_MAX_ROWS:
            return "window width outside 1..17"
        sw = self.switch_mlp
        reason = self.moe_routing_contract.routing_refusal(
            expert_count=self.num_experts,
            top_k=self.top_k,
            normalize_topk=bool(self.norm_topk_prob),
            activation_dtype=(
                "bfloat16" if x.dtype == mx.bfloat16 else str(x.dtype)
            ),
            projection_layout="split_gate_up",
            consumer=consumer,
        )
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
        elif reason is not None:
            pass
        elif "gate_up_proj" in sw:
            reason = "requires split gate/up tables"
        else:
            bits = getattr(sw.gate_proj, "bits", None)
            group_size = getattr(sw.gate_proj, "group_size", None)
            if (bits, group_size) in DECODE_ONLY_ROUTED_FORMATS:
                if consumer is not None or rows != 1:
                    return (
                        f"routed q{bits}g{group_size} tables are admitted for "
                        "the one-token decode only"
                    )
            elif (bits, group_size) not in ROUTED_FORMATS:
                return f"unsupported routed expert format q{bits}g{group_size}"
            admit = RD.admit_split_routed_decode(
                QN._ShapeOnly((x.shape[-1],), x.dtype),
                QN._ShapeOnly((8,), mx.uint32),
                QN._ShapeOnly((8,), x.dtype),
                sw.gate_proj,
                sw.up_proj,
                sw.down_proj,
                contract=self.moe_routing_contract,
                bits=bits,
                group_size=group_size,
            )
            if not admit.accepted:
                reason = admit.reason
        if reason is None and not RD.runtime_supported():
            reason = "Metal runtime unavailable"
        if reason is None:
            reason = RD.served_swiglu_refusal()
        return reason

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
        sw = self.switch_mlp
        reason = self._routed_refusal(
            x, consumer=consumer if window else None
        )
        if reason:
            self._decline(reason)
            return super().__call__(x)
        # Router and shared-gate projections retain one-token arithmetic.
        # When their capabilities match, the shared row-exact package groups
        # the weight reads into one launch; its canary-preserving fallback is
        # still the same independent one-token call for every row.
        flat = x.reshape(rows, -1)
        logits, shared_gate_logit = router_and_shared_gate_rows(self, flat)
        refusal = None
        if self.moe_topk_mode == "launch":
            # The same structural admission the parent's launch applies; the
            # kernel transcribes stock routing for bf16 logits only.
            refusal = W.admit_router_topk(
                logits, self.top_k, bool(self.norm_topk_prob),
                contract=self.moe_routing_contract,
                consumer=consumer if window else None,
            )
            if refusal is None and logits.dtype != x.dtype:
                refusal = "router logits dtype differs from the activation"
            if refusal is not None:
                self.moe_topk_fallbacks += 1
                self.moe_topk_last_fallback = refusal
        if self.moe_topk_mode == "launch" and refusal is None:
            inds, scores = W.router_topk(
                logits,
                top_k=self.top_k,
                contract=self.moe_routing_contract,
                consumer=consumer if window else None,
            )
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
        shared_gate = gate_sigmoid(shared_gate_logit).reshape(rows, 1)
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
