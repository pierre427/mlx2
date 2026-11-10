"""EXL3 trellis linears and expert banks on the Sushi-derived kernels.

MIT Sushi ad562c4 Metal bodies; adapter-owned model math, no scheduler/cache
changes. See provenance/sushi-integration-20261012.json. The explicit Sushi
pack marker stays 2.5-bit MCG; upstream checkpoints (runtime/trellis) reuse
these modules at any even tile word count. Unqualified format.
"""

from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu

FORMAT_REVISION = "mlx2-exl3-mcg-w16-n40-v1"
ASSETS = Path(__file__).with_name("exl3_kernels")


def validate_metadata(meta):
    if not isinstance(meta, dict) or set(meta) != {
        "version",
        "codebook",
        "window_bits",
        "bits_per_weight",
        "execution",
    }:
        raise ValueError(
            "EXL3 requires explicit version/codebook/window_bits/bits_per_weight/execution"
        )
    if (
        type(meta["version"]) is not int
        or meta["version"] != 1
        or meta["codebook"] != "mcg"
        or type(meta["window_bits"]) is not int
        or meta["window_bits"] != 16
        or type(meta["bits_per_weight"]) not in (float, int)
        or meta["bits_per_weight"] != 2.5
        or meta["execution"] not in ("reference", "fused")
    ):
        raise ValueError(
            "only version-1 2.5-bit MCG/w16 EXL3 reference or fused is implemented"
        )
    return dict(meta)


def _patched(name, edits):
    """``name``'s source with each (old, new) edit applied exactly once."""
    text = (ASSETS / name).read_text()
    for old, new in edits:
        if text.count(old) != 1:
            raise RuntimeError(f"EXL3 kernel patch does not match {name}: {old!r}")
        text = text.replace(old, new)
    return text


# Shared-expert slots: a call over T tokens has T*K routed slots then T shared
# slots (threads_per_grid.y = T*(K+1)); a shared slot reads the shared bank
# (expert 0) at its token row. Routed rows keep the base kernels' arithmetic.
_SHARED_ROW = (
    "const uint ntok = uint(threads_per_grid.y) / uint(TOPK + 1);\n"
    "const bool shared_row = {slot} >= ntok * uint(TOPK);\n"
)


def _shared_gate_up_source():
    return "// mlx2: FUSED_GATE_UP with the shared expert's slots appended.\n" + _patched(
        "FUSED_GATE_UP_SOURCE.metal",
        [
            ("const uint e = uint(slots[slot]);",
             _SHARED_ROW.format(slot="slot") + "const uint e = shared_row ? 0u : uint(slots[slot]);"),
            ("const size_t xr = (size_t)(slot / uint(TOPK)) * (size_t)(IDIM);",
             "const size_t xr = (size_t)(shared_row ? slot - ntok * uint(TOPK) : slot / uint(TOPK)) * (size_t)(IDIM);"),
            ("(proj == 0u ? gsuh : usuh)", "(shared_row ? (proj == 0u ? sgsuh : susuh) : (proj == 0u ? gsuh : usuh))"),
            ("(gt + (size_t)e", "((shared_row ? sgt : gt) + (size_t)e"),
            ("(ut + (size_t)e", "((shared_row ? sut : ut) + (size_t)e"),
            ("(sg == 0u ? gsvh : usvh)", "(shared_row ? (sg == 0u ? sgsvh : susvh) : (sg == 0u ? gsvh : usvh))"),
            ("""  const float g = float(T(half(val[lid])));
  const float u = float(T(half(val[128u + lid])));
  const float y = float(T(1.0f / (1.0f + metal::exp(-metal::abs(g)))));
  const float sig = g < 0.0f ? float(T(1.0f - y)) : y;
  const float gs = float(T(g * sig));
  h[(size_t)slot * (size_t)(ODIM) + block * 128u + lid] = T(gs * u);""",
             """  // Routed slots in T (half); shared slots in ST (the trunk dtype), stored
  // as half: exact for normal-range values; below 2^-14 half subnormals
  // drop bits and above 65504 it overflows (the routed slots' range too).
  h[(size_t)slot * (size_t)(ODIM) + block * 128u + lid] = half(shared_row
      ? exl3_swiglu<ST>(val[lid], val[128u + lid])
      : exl3_swiglu<T>(val[lid], val[128u + lid]));"""),
        ],
    )


# FUSED_LINEAR with KS input splits: 8*KS simdgroups, simdgroup s takes
# output tile s % 8 over input tiles [ks*IT/KS, (ks+1)*IT/KS) (ks = s / 8),
# and the KS partial sums add in ks order before the finish. KS = 1 is the
# base kernel's arithmetic; any KS is the same program for every row.
_KSPLIT_EDITS = [
    ("threadgroup float inner[128];", "threadgroup float inner[128u * uint(KS)];"),
    ("constexpr uint SGS = 8u;", "constexpr uint SGS = 8u * uint(KS);"),
    ("const uint ot = block * 8u + sg;", "const uint ks = sg >> 3u;\n  const uint ot = block * 8u + (sg & 7u);"),
    ("for (uint tk = 0u; tk < IT; tk += 2u) {",
     "for (uint tk = ks * (IT / uint(KS)); tk < (ks + 1u) * (IT / uint(KS)); tk += 2u) {"),
    ("  float4 v = float4(inner[lane], inner[lane + 32u], inner[lane + 64u], inner[lane + 96u]);\n",
     "  float4 v = float4(inner[lane], inner[lane + 32u], inner[lane + 64u], inner[lane + 96u]);\n"
     "  for (uint k = 1u; k < uint(KS); k++) {\n"
     "    v += float4(inner[k * 128u + lane], inner[k * 128u + lane + 32u],\n"
     "                inner[k * 128u + lane + 64u], inner[k * 128u + lane + 96u]);\n"
     "  }\n"),
]


def _ksplit_linear_source():
    return "// mlx2: FUSED_LINEAR with KS input splits.\n" + _patched("FUSED_LINEAR_SOURCE.metal", _KSPLIT_EDITS)


def _default_ks(input_dims):
    return FUSED_KS_BY_INPUT.get(input_dims, FUSED_KS)


def fused_ksplit(input_dims, ks):
    """The largest admitted split <= ks: KS must leave an even tile count."""
    tiles = input_dims // 16
    while ks > 1 and (tiles % (2 * ks) or 256 * ks > 1024):
        ks //= 2
    return ks


# Up to three same-input dense linears in one launch: threadgroup blocks
# [0, NB0) run linear 0, [NB0, NB0+NB1) linear 1, the rest linear 2; each
# block runs FUSED_LINEAR (KS)'s program for its own linear.
_GROUP_EDITS = [
    ("const uint block = uint(threadgroup_position_in_grid.x);",
     "const uint gblock = uint(threadgroup_position_in_grid.x);\n"
     "const uint which = gblock < uint(NB0) ? 0u : (gblock < uint(NB0 + NB1) ? 1u : 2u);\n"
     "const uint block = gblock - (which == 0u ? 0u : (which == 1u ? uint(NB0) : uint(NB0 + NB1)));\n"
     "const uint ODIM_ = which == 0u ? uint(ODIM0) : (which == 1u ? uint(ODIM1) : uint(ODIM2));"),
    ("constexpr uint OT = uint(ODIM) / TILE;", "const uint OT = ODIM_ / TILE;"),
    ("const uint e = uint(slots[row]);", "const uint e = 0u;"),
    ("const device half *suh_e = suh +", "const device half *suh_e = (which == 0u ? suh0 : (which == 1u ? suh1 : suh2)) +"),
    ("const device half *svh_e = svh + (size_t)e * (size_t)(ODIM);",
     "const device half *svh_e = (which == 0u ? svh0 : (which == 1u ? svh1 : svh2)) + (size_t)e * (size_t)ODIM_;"),
    ("(trellis + (size_t)e", "((which == 0u ? trellis0 : (which == 1u ? trellis1 : trellis2)) + (size_t)e"),
    ("constexpr size_t STEP = (size_t)OT * PACKED_W;", "const size_t STEP = (size_t)OT * PACKED_W;"),
    ("const size_t yb = (size_t)row * (size_t)(ODIM) + base;",
     "const size_t yb = (size_t)row * (size_t)ODIM_ + base;\n"
     "  device OUT *y = which == 0u ? y0 : (which == 1u ? y1 : y2);"),
]


def _group_linear_source():
    return "// mlx2: FUSED_LINEAR (KS splits) over up to three same-input linears.\n" + _patched(
        "FUSED_LINEAR_SOURCE.metal", _KSPLIT_EDITS + _GROUP_EDITS)


def fused_group(linears, x):
    """FUSED_LINEAR of 2-3 dense one-expert banks reading the same rows, one
    launch; outputs in x's dtype, each row the same program as fused_rows."""
    banks = [lin.bank for lin in linears]
    first = banks[0]
    if not 2 <= len(banks) <= 3 or any(
            (b.input_dims, b.words, b.trellis.dtype) != (first.input_dims, first.words, first.trellis.dtype)
            for b in banks):
        raise ValueError("EXL3 grouped linears need 2-3 banks of one input width, rate and storage")
    padded = banks + [banks[-1]] * (3 - len(banks))
    blocks = [b.output_dims // 128 for b in banks] + [0] * (3 - len(banks))
    ks = fused_ksplit(first.input_dims, _default_ks(first.input_dims))
    rows = x.shape[0]
    inputs = [x]
    for b in padded:
        inputs += [b.trellis, b.suh, b.svh]
    outs = _kernels()[14](
        inputs=inputs,
        template=[
            ("IDIM", first.input_dims), ("NHW", first.words), ("OUT", x.dtype), ("KS", ks),
            ("NB0", blocks[0]), ("NB1", blocks[1]),
            ("ODIM0", padded[0].output_dims), ("ODIM1", padded[1].output_dims), ("ODIM2", padded[2].output_dims),
        ],
        grid=(sum(blocks) * 256 * ks, rows, 1),
        threadgroup=(256 * ks, 1, 1),
        output_shapes=[(rows, b.output_dims) if i < len(banks) else (1, 128) for i, b in enumerate(padded)],
        output_dtypes=[x.dtype] * 3,
    )
    return outs[: len(banks)]



def _int_counters(*roots):
    """(module, attribute, value) for every integer counter under ``roots``."""
    out = []
    for root in roots:
        modules = root.modules() if hasattr(root, "modules") else [root]
        for module in modules:
            for name, value in vars(module).items():
                if type(value) is int:
                    out.append((module, name, value))
    return out

class Exl3ProjectionGroup:
    """Same-input dense EXL3 linears of one module run as one FUSED_LINEAR
    launch at fused widths (below NAX_MIN_ROWS on Metal). ``__call__``
    returns {name: output} or None to decline, and credits each linear's
    call counters as its own call would."""

    def __init__(self, linears):
        self.names = tuple(linears)
        self.linears = tuple(linears.values())
        object.__setattr__(self, "calls", 0)

    def __call__(self, x):
        flat = x.reshape(-1, x.shape[-1])
        rows = flat.shape[0]
        if mx.default_device() == mx.cpu or not 0 < rows < NAX_MIN_ROWS:
            return None
        if any(rows >= lin.__dict__.get("resident_min_rows", NAX_MIN_ROWS) for lin in self.linears):
            return None  # a resident member serves this width from fp16
        outs = fused_group(self.linears, flat)
        self.calls += 1
        result = {}
        for name, lin, out in zip(self.names, self.linears, outs):
            object.__setattr__(lin, "calls", lin.calls + 1)
            object.__setattr__(lin, "rows", lin.rows + rows)
            result[name] = out.reshape(*x.shape[:-1], lin.output_dims)
        return result


def group_refusal(linears):
    """Why these linears cannot share a launch, or None."""
    if not 2 <= len(linears) <= 3:
        return "a group has 2-3 linears"
    first = linears[0]
    for lin in linears:
        if not isinstance(lin, Exl3DenseLinear) or lin.law != "fused_nax":
            return "grouped projections need fused_nax EXL3 linears"
        if "bias" in lin:
            return "grouped projections carry no bias"
        if (lin.input_dims, lin.words, lin.bank.trellis.dtype) != (
                first.input_dims, first.words, first.bank.trellis.dtype):
            return "grouped projections differ in input width, rate or storage"
    return None


# The decoder residual added in the finish: OUT(r + OUT(result)), the stock
# bf16 add of the linear's output and the residual, rounded once.
_RESIDUAL_EDITS = [
    (f"  y[yb + lane{off}] = OUT(float(half(({a}) * sc * float(svh_e[base + lane{off}]))));",
     f"  y[yb + lane{off}] = OUT(float(res[yb + lane{off}]) + float(OUT(float(half(({a}) * sc * float(svh_e[base + lane{off}]))))));")
    for a, off in (("s0 + s2", ""), ("s1 + s3", " + 32u"), ("s0 - s2", " + 64u"), ("s1 - s3", " + 96u"))
]


def _residual_linear_source():
    return "// mlx2: FUSED_LINEAR (KS splits) with the decoder residual added.\n" + _patched(
        "FUSED_LINEAR_SOURCE.metal", _KSPLIT_EDITS + _RESIDUAL_EDITS)


class ResidualSlot:
    """One-shot decoder residual handed to the module that can add it in its
    last launch. The decoder sets ``value``; a consumer takes it (``value``
    back to None, ``used`` True) only when its output has the residual's
    shape; otherwise the decoder adds it as before."""

    __slots__ = ("value", "used")

    def __init__(self):
        self.value, self.used = None, False

    def take(self, shape, dtype):
        value = self.value
        if value is None or value.shape != tuple(shape) or value.dtype != dtype:
            return None
        self.value, self.used = None, True
        return value


def _shared_gate_up_logit_source():
    # One simdgroup of each shared slot's first block also forms the shared
    # expert gate's logit, dot(x_row, w) in float in a fixed lane order,
    # rounded to ST (the 2048->1 gate GEMV, not bitwise MLX's).
    return _shared_gate_up_source().replace(
        "// mlx2: FUSED_GATE_UP with the shared expert's slots appended.",
        "// mlx2: FUSED_GATE_UP (shared slots) with the shared gate logit.", 1) + """
if (shared_row && block == 0u && sg == 0u) {
  float acc = 0.0f;
  for (uint i = uint(lane); i < uint(IDIM); i += 32u) {
    acc = fma(float(x[xr + i]), float(wsg[i]), acc);
  }
  acc = simd_sum(acc);
  if (lane == 0u) {
    glo[slot - ntok * uint(TOPK)] = ST(acc);
  }
}
"""


def _shared_linear_source():
    return "// mlx2: FUSED_LINEAR (KS splits) with the shared expert's rows appended.\n" + _patched(
        "FUSED_LINEAR_SOURCE.metal",
        _KSPLIT_EDITS + [
            ("const uint e = uint(slots[row]);",
             _SHARED_ROW.format(slot="row") + "const uint e = shared_row ? 0u : uint(slots[row]);"),
            ("const device half *suh_e = suh +", "const device half *suh_e = (shared_row ? ssuh : suh) +"),
            ("const device half *svh_e = svh +", "const device half *svh_e = (shared_row ? ssvh : svh) +"),
            ("(trellis + (size_t)e", "((shared_row ? strellis : trellis) + (size_t)e"),
        ],
    )


_SHARED_COMBINE_EDIT = (
    "float(shared[(size_t)row * (size_t)(D) + col])",
    "float(T(d[((size_t)uint(threads_per_grid.y) * uint(K) + row) * (size_t)(D) + col]))")


def _shared_combine_source():
    # The shared expert's down rows follow the routed rows in d (half, as the
    # unfused shared down rounds before its trunk cast).
    return "// mlx2: MOE_COMBINE reading the shared rows from d.\n" + _patched(
        "MOE_COMBINE_SOURCE.metal", [_SHARED_COMBINE_EDIT])


def _shared_combine_residual_source():
    return "// mlx2: MOE_COMBINE (shared rows in d) with the decoder residual added.\n" + _patched(
        "MOE_COMBINE_SOURCE.metal",
        [_SHARED_COMBINE_EDIT,
         ("out[(size_t)row * (size_t)(D) + col] = T(y + p);",
          "out[(size_t)row * (size_t)(D) + col] = T(float(res[(size_t)row * (size_t)(D) + col]) + float(T(y + p)));")])


@lru_cache(maxsize=1)
def _kernels():
    def kernel(name, inputs, outputs, file, header=""):
        return mx.fast.metal_kernel(
            name="mlx2_exl3_" + name,
            input_names=inputs,
            output_names=outputs,
            source=(ASSETS / file).read_text(),
            header=header,
        )

    def kernel_text(name, inputs, outputs, source, header=""):
        return mx.fast.metal_kernel(
            name="mlx2_exl3_" + name, input_names=inputs, output_names=outputs,
            source=source, header=header,
        )

    header = (ASSETS / "header_mcg_w16.metal").read_text()
    return (
        kernel("prepare", ["x", "suh", "slots"], ["y"], "PREPARE_SOURCE.metal"),
        kernel(
            "inner",
            ["x", "trellis", "slots"],
            ["y"],
            "INDEXED_COOP_SOURCE.metal",
            header,
        ),
        kernel("finish", ["inner", "svh", "slots"], ["y"], "FINISH_SOURCE.metal"),
        kernel(
            "pair",
            ["x", "suhg", "suhu", "tg", "tu", "slots"],
            ["yg", "yu"],
            "PAIR_GEMV_SOURCE.metal",
            header,
        ),
        kernel("rows", ["x", "trellis", "order", "eids"], ["y"], "ROWS_GEMV_SOURCE.metal", header),
        kernel_text("fused_linear", ["x", "trellis", "suh", "svh", "slots"], ["y"], _ksplit_linear_source(), header),
        kernel("reconstruct", ["trellis", "experts"], ["inner"], "RECONSTRUCT_SOURCE.metal", header),
        kernel(
            "fused_gate_up",
            ["x", "slots", "gt", "gsuh", "gsvh", "ut", "usuh", "usvh"],
            ["h"],
            "FUSED_GATE_UP_SOURCE.metal",
            header + (ASSETS / "header_fused_tiles.metal").read_text(),
        ),
        kernel("moe_combine", ["d", "scores", "shared", "gl"], ["out"], "MOE_COMBINE_SOURCE.metal"),
        kernel_text(
            "fused_gate_up_shared",
            ["x", "slots", "gt", "gsuh", "gsvh", "ut", "usuh", "usvh",
             "sgt", "sgsuh", "sgsvh", "sut", "susuh", "susvh"],
            ["h"],
            _shared_gate_up_source(),
            header + (ASSETS / "header_fused_tiles.metal").read_text(),
        ),
        kernel_text(
            "fused_linear_shared",
            ["x", "trellis", "suh", "svh", "slots", "strellis", "ssuh", "ssvh"],
            ["y"],
            _shared_linear_source(),
            header,
        ),
        kernel_text("moe_combine_shared", ["d", "scores", "gl"], ["out"], _shared_combine_source()),
        kernel("prepare_map", ["x", "suh", "src", "slots"], ["y"], "PREPARE_MAP_SOURCE.metal"),
        kernel("finish_map", ["inner", "svh", "slots", "dst"], ["y"], "FINISH_MAP_SOURCE.metal"),
        kernel_text(
            "fused_linear_group",
            ["x", "trellis0", "suh0", "svh0", "trellis1", "suh1", "svh1", "trellis2", "suh2", "svh2"],
            ["y0", "y1", "y2"],
            _group_linear_source(),
            header,
        ),
        kernel_text("fused_linear_res", ["x", "trellis", "suh", "svh", "slots", "res"], ["y"],
                    _residual_linear_source(), header),
        kernel_text("moe_combine_shared_res", ["d", "scores", "gl", "res"], ["out"],
                    _shared_combine_residual_source()),
        kernel_text(
            "fused_gate_up_shared_logit",
            ["x", "slots", "gt", "gsuh", "gsvh", "ut", "usuh", "usvh",
             "sgt", "sgsuh", "sgsvh", "sut", "susuh", "susvh", "wsg"],
            ["h", "glo"],
            _shared_gate_up_logit_source(),
            header + (ASSETS / "header_fused_tiles.metal").read_text(),
        ),
    )


# Row-blocked law (ROWS_GEMV_SOURCE): block b covers rows order[b*R..b*R+R)
# sorted by expert; a row's arithmetic is the same at every R and width.
# Four rows per block: eight measured slower on an M5 Max (register
# pressure; lm_head at 512 rows 377 vs 185 ms), same per-row arithmetic.
ROWS_PER_BLOCK = 4
# Bank calls this wide use eight-row blocks (rows of one expert share a tile
# decode); narrower bank calls, mostly distinct experts, use one-row blocks.
GROUP_MIN_ROWS = 32
# fused_nax: dense calls this wide reconstruct fp16 weights and run
# mx.matmul (the M5 tensor units); narrower calls keep the fused kernel.
NAX_MIN_ROWS = 64
# Columns reconstructed per matmul: bounds the fp16 weight transient
# (lm_head: 2048 x 16384 x 2 B = 64 MiB per slice).
NAX_SLICE_COLUMNS = 16384
# rows_nax: bank calls this wide (128 tokens at top-8) reconstruct their
# experts in chunks and run sorted gather_mm on the tensor units.
NAX_BANK_MIN_ROWS = 1024
# Experts reconstructed per gather_mm (64 x 2048 x 512 x 2 B = 128 MiB).
NAX_BANK_CHUNK = 64


class NaxLayout(NamedTuple):
    order: mx.array
    inverse: mx.array
    sorted_ids: mx.array
    bounds: tuple
    src: mx.array = None  # token row of each sorted slot (order // topk)


def nax_layout(ids, num_experts, topk=1):
    """Expert-sorted rows and each expert's segment (one host sync)."""
    import numpy as np

    host = np.asarray(ids.astype(mx.uint32))
    order = np.argsort(host, kind="stable").astype(np.uint32)
    ranked = host[order]
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size, dtype=np.uint32)
    bounds = tuple(int(b) for b in np.searchsorted(ranked, np.arange(num_experts + 1)))
    return NaxLayout(mx.array(order), mx.array(inverse), mx.array(ranked), bounds,
                     mx.array(order // np.uint32(topk)))


def nax_bank(proj, rows, ids, layout):
    """One bank projection at prefill width: prepare, per expert chunk
    reconstruct + sorted gather_mm, finish (rows in caller order)."""
    prepared = proj.prepare(rows, ids)[layout.order]
    inner = _nax_sorted_inner(proj, prepared, layout)
    return proj.finish(inner[layout.inverse], ids)


def nax_switch(switch, flat, ids, topk, out_dtype):
    """Routed gate/up/down at prefill width, kept in expert-sorted order:
    gate and up prepare straight from the token rows (no repeated or
    permuted input copy), their finishes and SwiGLU stay sorted, and the
    down finish scatters to caller order in ``out_dtype``. Row for row the
    arithmetic of three ``nax_bank`` calls, so bitwise their result."""
    layout = nax_layout(ids, switch.gate_proj.num_experts, topk)
    g, u = (proj.finish(_nax_sorted_inner(proj, proj.prepare_map(flat, layout.src, layout.sorted_ids), layout),
                        layout.sorted_ids)
            for proj in (switch.gate_proj, switch.up_proj))
    down = switch.down_proj
    inner = _nax_sorted_inner(down, down.prepare(swiglu(g, u), layout.sorted_ids), layout)
    return down.finish_map(inner, layout.sorted_ids, layout.order, ids.size, out_dtype)


def _nax_sorted_inner(proj, prepared, layout):
    """Inner products of expert-sorted prepared rows, sorted order."""
    cpu = mx.default_device() == mx.cpu
    parts = []
    for e0 in range(0, proj.num_experts, NAX_BANK_CHUNK):
        e1 = min(e0 + NAX_BANK_CHUNK, proj.num_experts)
        s0, s1 = layout.bounds[e0], layout.bounds[e1]
        if s0 == s1:
            continue
        weights = proj.reconstruct(mx.arange(e0, e1, dtype=mx.uint32))
        lhs = prepared[s0:s1][:, None, :]
        if cpu:  # MLX's CPU gather GEMM has no half kernel
            lhs, weights = lhs.astype(mx.float32), weights.astype(mx.float32)
        part = mx.gather_mm(lhs, weights, rhs_indices=layout.sorted_ids[s0:s1] - e0,
                            sorted_indices=True)
        parts.append(part.squeeze(-2).astype(mx.float16))
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
# Sushi's fused gate/up pair (PAIR_GEMV_SOURCE) was validated at 40-word tiles
# only; at 32 and 64 words it measured 82-86% of output scale off the reference
# switch on Qwen3.6-35B-A3B layer-10 experts (exl3-phase3-metal-20261104).
PAIR_VALIDATED_WORDS = 40
# Revision of the row law; part of the install receipt and identity.
ROWS_LAW_REVISION = "mlx2-exl3-rows-v4"


class RowPlan(NamedTuple):
    order: mx.array
    eids: mx.array
    rows: int
    dense: bool = False


_CONSTANTS = {}


def _constant(kind, count):
    """Immutable per-width index arrays, built once (they are MLX constants)."""
    key = (kind, count)
    if key not in _CONSTANTS:
        _CONSTANTS[key] = (mx.zeros if kind == "zeros" else mx.arange)(count, dtype=mx.uint32)
    return _CONSTANTS[key]


def row_plan(ids, num_experts, *, rows=None):
    """Device-side plan: rows sorted by expert, block size by call width.

    ``rows`` overrides the block size (timing studies); a row's arithmetic
    is the same at every block size.
    """
    if rows is not None and not 1 <= rows <= ROWS_PER_BLOCK:
        raise ValueError(f"row blocks hold 1..{ROWS_PER_BLOCK} rows")
    ids = ids.astype(mx.uint32)
    count = ids.size
    if num_experts == 1:
        size = rows or (ROWS_PER_BLOCK if count > 1 else 1)
        return RowPlan(_constant("arange", count), ids, size, True)
    size = rows or (ROWS_PER_BLOCK if count >= GROUP_MIN_ROWS else 1)
    order = mx.argsort(ids).astype(mx.uint32) if size > 1 else _constant("arange", count)
    return RowPlan(order, ids, size)


def _h128(x):
    shape = x.shape
    if mx.default_device() == mx.cpu:
        import numpy as np

        a = np.asarray(x, dtype=np.float32).reshape(-1, 128).copy()
        for bit in (1, 2, 4, 8, 16, 32, 64):
            block = a.reshape(-1, 128 // (2 * bit), 2, bit)
            lo, hi = block[:, :, 0, :].copy(), block[:, :, 1, :].copy()
            block[:, :, 0, :], block[:, :, 1, :] = lo + hi, lo - hi
        return mx.array((a * np.float32(1 / (128**0.5))).reshape(shape))
    return mx.hadamard_transform(x.reshape(-1, 128), scale=1 / (128**0.5)).reshape(
        shape
    )


LAWS = ("reference", "rows")


class Exl3Linear(nn.Module):
    def __init__(self, trellis, suh, svh, *, law="reference"):
        super().__init__()
        if law not in LAWS:
            raise ValueError(f"EXL3 law must be one of {LAWS}")
        if (
            trellis.dtype != mx.uint16
            or trellis.ndim != 4
            or trellis.shape[-1] % 2
            or not 16 <= trellis.shape[-1] <= 128
            or suh.dtype != mx.float16
            or svh.dtype != mx.float16
            or suh.shape != (trellis.shape[0], trellis.shape[1] * 16)
            or svh.shape != (trellis.shape[0], trellis.shape[2] * 16)
            or trellis.shape[0] < 1
            or suh.shape[-1] % 128
            or svh.shape[-1] % 128
        ):
            raise ValueError(
                "EXL3 requires [E,I/16,O/16,n] U16 (even n in 16..128) "
                "and matching H128 F16 vectors"
            )
        if not bool(mx.all(mx.isfinite(suh)).item()) or not bool(
            mx.all(mx.isfinite(svh)).item()
        ):
            raise ValueError("EXL3 scale vectors must be finite")
        self.trellis, self.suh, self.svh = trellis, suh, svh
        self.law = law
        self.freeze()

    @property
    def input_dims(self):
        return self.suh.shape[-1]

    @property
    def output_dims(self):
        return self.svh.shape[-1]

    @property
    def num_experts(self):
        return self.trellis.shape[0]

    @property
    def words(self):
        return self.trellis.shape[-1]

    @property
    def packed_nbytes(self):
        return self.trellis.nbytes + self.suh.nbytes + self.svh.nbytes

    def prepare(self, x, ids):
        if mx.default_device() == mx.cpu:
            return _h128(
                x.astype(mx.float32) * self.suh[ids].astype(mx.float32)
            ).astype(mx.float16)
        return _kernels()[0](
            inputs=[x, self.suh, ids],
            template=[("IDIM", self.input_dims)],
            grid=(self.input_dims // 128 * 32, ids.size, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(ids.size, self.input_dims)],
            output_dtypes=[mx.float16],
        )[0]

    def finish(self, inner, ids):
        if mx.default_device() == mx.cpu:
            return (
                _h128(inner.astype(mx.float32)) * self.svh[ids].astype(mx.float32)
            ).astype(mx.float16)
        return _kernels()[2](
            inputs=[inner, self.svh, ids],
            template=[("ODIM", self.output_dims)],
            grid=(self.output_dims // 128 * 32, ids.size, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(ids.size, self.output_dims)],
            output_dtypes=[mx.float16],
        )[0]

    def prepare_map(self, x, src, ids):
        """Slot s prepared from row src[s] of x against expert ids[s]."""
        if mx.default_device() == mx.cpu:
            return self.prepare(x[src], ids)
        return _kernels()[12](
            inputs=[x, self.suh, src, ids],
            template=[("IDIM", self.input_dims)],
            grid=(self.input_dims // 128 * 32, ids.size, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(ids.size, self.input_dims)],
            output_dtypes=[mx.float16],
        )[0]

    def finish_map(self, inner, ids, dst, rows, dtype):
        """finish of slot s against expert ids[s] written to row dst[s] in
        ``dtype``; dst must be a permutation of range(rows)."""
        if mx.default_device() == mx.cpu:
            out = self.finish(inner, ids)
            return mx.zeros((rows, self.output_dims), mx.float16).at[dst].add(out).astype(dtype)
        return _kernels()[13](
            inputs=[inner, self.svh, ids, dst],
            template=[("ODIM", self.output_dims), ("OUT", dtype)],
            grid=(self.output_dims // 128 * 32, ids.size, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(rows, self.output_dims)],
            output_dtypes=[dtype],
        )[0]

    def fused_rows(self, x, ids, ks=None, residual=None):
        """FUSED_LINEAR_SOURCE: prepare, inner product and finish of row i
        against expert ids[i] in one launch; output in x's dtype. ``ks``
        input splits (default FUSED_KS, admitted by shape only); ``residual``
        ([rows, output_dims], x's dtype) added in the finish."""
        ks = fused_ksplit(self.input_dims, _default_ks(self.input_dims) if ks is None else ks)
        extra = [] if residual is None else [residual]
        return _kernels()[5 if residual is None else 15](
            inputs=[x, self.trellis, self.suh, self.svh, ids.astype(mx.uint32), *extra],
            template=[
                ("IDIM", self.input_dims),
                ("ODIM", self.output_dims),
                ("NHW", self.words),
                ("OUT", x.dtype),
                ("KS", ks),
            ],
            grid=(self.output_dims // 128 * 256 * ks, x.shape[0], 1),
            threadgroup=(256 * ks, 1, 1),
            output_shapes=[(x.shape[0], self.output_dims)],
            output_dtypes=[x.dtype],
        )[0]

    def reconstruct(self, experts, ot0=0, columns=None, *, transposed=False):
        """Decoded fp16 inner weights [U, I, columns] ([U, columns, I] when
        transposed) from output tile ot0."""
        columns = self.output_dims - ot0 * 16 if columns is None else columns
        if columns % 128 or ot0 * 16 + columns > self.output_dims:
            raise ValueError("EXL3 reconstruction slices whole 128-column blocks")
        experts = experts.astype(mx.uint32)
        if mx.default_device() == mx.cpu:
            import numpy as np

            from .exl3_reference import decode_inner

            tiles = np.asarray(self.trellis[:, :, ot0 : ot0 + columns // 16])
            out = np.stack([decode_inner(tiles[int(e)]) for e in np.asarray(experts)])
            return mx.array(np.ascontiguousarray(out.transpose(0, 2, 1)) if transposed else out)
        return _kernels()[6](
            inputs=[self.trellis, experts],
            template=[
                ("IDIM", self.input_dims),
                ("ODIM", self.output_dims),
                ("NHW", self.words),
                ("OT0", ot0),
                ("OSLICE", columns),
                ("TRANS", transposed),
            ],
            grid=(columns // 16 * 32, self.input_dims // 16, experts.size),
            threadgroup=(32, 1, 1),
            output_shapes=[(experts.size, columns, self.input_dims) if transposed
                           else (experts.size, self.input_dims, columns)],
            output_dtypes=[mx.float16],
        )[0]

    def _rows(self, prepared, plan):
        """Row-blocked inner product; float32 rows in the caller's order."""
        rows = prepared.shape[0]
        if mx.default_device() == mx.cpu:
            import numpy as np

            from .exl3_reference import decode_inner

            host = np.asarray(prepared.astype(mx.float32))
            eids = np.asarray(plan.eids)
            value = np.zeros((rows, self.output_dims), np.float32)
            for eid in np.unique(eids):
                inner = decode_inner(np.asarray(self.trellis[int(eid)])).astype(np.float32)
                value[eids == eid] = host[eids == eid] @ inner
            return mx.array(value)
        if rows == 0:
            return mx.zeros((0, self.output_dims), mx.float32)
        return _kernels()[4](
            inputs=[prepared, self.trellis, plan.order, plan.eids],
            template=[
                ("IDIM", self.input_dims),
                ("ODIM", self.output_dims),
                ("NHW", self.words),
                ("ROWS", plan.rows),
                ("DENSE", plan.dense),
            ],
            grid=(self.output_dims // 16 * 128, -(-rows // plan.rows), 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(rows, self.output_dims)],
            output_dtypes=[mx.float32],
        )[0]

    def __call__(self, x, ids, *, trusted_indices=False, plan=None):
        if x.ndim != 2 or x.shape != (ids.size, self.input_dims) or ids.ndim != 1:
            raise ValueError(
                "EXL3 projection requires one input vector per routed slot"
            )
        if ids.dtype != mx.uint32:
            raise ValueError("EXL3 slot IDs must be uint32")
        if (
            not trusted_indices
            and ids.size
            and int(mx.max(ids).item()) >= self.num_experts
        ):
            raise ValueError("EXL3 expert index outside the bound bank")
        prepared = self.prepare(x, ids)
        if self.law == "rows":
            return self.finish(self._rows(prepared, plan or row_plan(ids, self.num_experts)), ids)
        if mx.default_device() == mx.cpu:
            import numpy as np
            from .exl3_reference import decode_inner

            inner = mx.array(decode_inner(np.asarray(self.trellis)))
            # MLX's CPU gather backend has no half GEMM. Accumulate float32
            # then round at the packed projection's F16 boundary.
            value = (
                mx.gather_mm(
                    prepared.astype(mx.float32)[:, None, :],
                    inner.astype(mx.float32),
                    rhs_indices=ids,
                )
                .squeeze(-2)
                .astype(mx.float16)
            )
        else:
            value = _kernels()[1](
                inputs=[prepared, self.trellis, ids],
                template=[
                    ("IDIM", self.input_dims),
                    ("ODIM", self.output_dims),
                    ("NHW", self.words),
                ],
                grid=(self.output_dims // 16 * 128, ids.size, 1),
                threadgroup=(128, 1, 1),
                output_shapes=[(ids.size, self.output_dims)],
                output_dtypes=[mx.float16],
            )[0]
        return self.finish(value, ids)


# FUSED_GATE_UP holds two prepared input rows in threadgroup memory
# (4 * IDIM + 3 KiB); wider inputs keep the unfused projections.
FUSED_GATE_UP_MAX_INPUT = 6144
# Input splits of the fused dense/down kernel (FUSED_LINEAR KS); 1 was the
# rev-1..3 law. Whole-model B1 decode on Qwen3.6-35B-A3B 4.00 bpw with shared
# slots (exl3-fuse4-metal-20261105/bench): KS 1 130.1, KS 2 137.3, KS 4
# 117.9-119.0 tok/s. FUSED_KS_BY_INPUT overrides by input width. Interleaved
# x3 (bench3): rev 3 118.2, shared slots 128.3, KS 2 135.3, KS 2 + KS 4 at
# 4096-wide inputs (out_proj/o_proj) 140.7 median; KS 4 at 512 (down) slower.
# KS 4 at 4096 is NOT selected: with it the whole-model check
# (exl3-fuse5-metal-20261105/400) found a full-vs-step flip outside the
# near-tie contract (exact tie in the full forward, step margin 0.625 nat);
# KS 2 everywhere passed it (exl3-fuse4-metal-20261105/400-ks2).
FUSED_KS = 2
FUSED_KS_BY_INPUT = {}


def fused_gate_up(gate, up, x, ids, topk, dtype=mx.float16):
    """FUSED_GATE_UP_SOURCE: h[slot] = swiglu(gate(x[slot // topk]),
    up(x[slot // topk])) against expert ids[slot], one launch; SwiGLU and h
    in ``dtype``."""
    if gate.input_dims != up.input_dims or gate.output_dims != up.output_dims:
        raise ValueError("fused gate/up needs one geometry")
    if gate.input_dims > FUSED_GATE_UP_MAX_INPUT:
        raise ValueError("fused gate/up input exceeds its threadgroup memory")
    return _kernels()[7](
        inputs=[x, ids.astype(mx.uint32), gate.trellis, gate.suh, gate.svh, up.trellis, up.suh, up.svh],
        template=[
            ("IDIM", gate.input_dims),
            ("ODIM", gate.output_dims),
            ("NG", gate.words),
            ("NU", up.words),
            ("TOPK", topk),
            ("T", dtype),
        ],
        grid=(gate.output_dims // 128 * 1024, ids.size, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(ids.size, gate.output_dims)],
        output_dtypes=[dtype],
    )[0]


def moe_combine(d, scores, shared, gl, topk, dtype):
    """MOE_COMBINE_SOURCE: the routed decode tail in one launch,
    ``sum_k(d[:, k] * scores[:, k]) + sigmoid(gl) * shared`` in ``dtype``."""
    rows, dims = shared.shape
    if d.shape != (rows * topk, dims) or scores.shape != (rows, topk) or gl.size != rows:
        raise ValueError("EXL3 MoE combine geometry mismatch")
    return _kernels()[8](
        inputs=[d, scores, shared, gl.reshape(rows)],
        template=[("D", dims), ("K", topk), ("T", dtype)],
        grid=(dims, rows, 1),
        threadgroup=(min(256, dims), 1, 1),
        output_shapes=[(rows, dims)],
        output_dtypes=[dtype],
    )[0]


def shared_slot_refusal(routed, shared):
    """Why the shared MLP cannot ride as extra slots of the routed launches
    (``None`` when it can): same geometry, tile words and storage dtypes."""
    if not isinstance(shared, Exl3FusedMLP):
        return "shared expert is not a fused EXL3 MLP"
    for name in ("gate_proj", "up_proj", "down_proj"):
        if "bias" in getattr(shared, name):
            # The slot launches read only the banks; a bias would be dropped.
            return f"shared {name} carries a bias"
        a, b = getattr(routed, name), getattr(shared, name).bank
        if (a.input_dims, a.output_dims, a.words) != (b.input_dims, b.output_dims, b.words):
            return f"shared {name} geometry or rate differs from the routed bank"
        if a.trellis.dtype != b.trellis.dtype or a.suh.dtype != b.suh.dtype or a.svh.dtype != b.svh.dtype:
            return f"shared {name} storage dtype differs from the routed bank"
    return None


def shared_gate_up(gate, up, sgate, sup, x, ids, topk, shared_dtype, gate_weight=None):
    """FUSED_GATE_UP over T*K routed slots then T shared slots (h in half;
    the shared SwiGLU rounds in ``shared_dtype``, the trunk dtype). With
    ``gate_weight`` ([input_dims] in ``shared_dtype``) also returns the
    shared gate's logits [T] in ``shared_dtype``: (h, logits)."""
    rows = x.shape[0]
    if ids.size != rows * topk:
        raise ValueError("EXL3 shared gate/up geometry mismatch")
    if gate_weight is not None:
        return _kernels()[17](
            inputs=[x, ids, gate.trellis, gate.suh, gate.svh, up.trellis, up.suh, up.svh,
                    sgate.trellis, sgate.suh, sgate.svh, sup.trellis, sup.suh, sup.svh, gate_weight],
            template=[
                ("IDIM", gate.input_dims), ("ODIM", gate.output_dims),
                ("NG", gate.words), ("NU", up.words), ("TOPK", topk),
                ("T", mx.float16), ("ST", shared_dtype),
            ],
            grid=(gate.output_dims // 128 * 1024, rows * (topk + 1), 1),
            threadgroup=(1024, 1, 1),
            output_shapes=[(rows * (topk + 1), gate.output_dims), (rows,)],
            output_dtypes=[mx.float16, shared_dtype],
        )
    return _kernels()[9](
        inputs=[x, ids, gate.trellis, gate.suh, gate.svh, up.trellis, up.suh, up.svh,
                sgate.trellis, sgate.suh, sgate.svh, sup.trellis, sup.suh, sup.svh],
        template=[
            ("IDIM", gate.input_dims), ("ODIM", gate.output_dims),
            ("NG", gate.words), ("NU", up.words), ("TOPK", topk),
            ("T", mx.float16), ("ST", shared_dtype),
        ],
        grid=(gate.output_dims // 128 * 1024, rows * (topk + 1), 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(rows * (topk + 1), gate.output_dims)],
        output_dtypes=[mx.float16],
    )[0]


def shared_down(down, sdown, h, ids, topk, ks=None):
    """FUSED_LINEAR over T*K routed rows then T shared rows, half out."""
    ks = fused_ksplit(down.input_dims, _default_ks(down.input_dims) if ks is None else ks)
    if h.shape[0] != ids.size // topk * (topk + 1) or ids.size % topk:
        raise ValueError("EXL3 shared down geometry mismatch")
    return _kernels()[10](
        inputs=[h, down.trellis, down.suh, down.svh, ids, sdown.trellis, sdown.suh, sdown.svh],
        template=[
            ("IDIM", down.input_dims), ("ODIM", down.output_dims),
            ("NHW", down.words), ("OUT", mx.float16), ("TOPK", topk), ("KS", ks),
        ],
        grid=(down.output_dims // 128 * 256 * ks, h.shape[0], 1),
        threadgroup=(256 * ks, 1, 1),
        output_shapes=[(h.shape[0], down.output_dims)],
        output_dtypes=[mx.float16],
    )[0]


def shared_combine(d, scores, gl, topk, dtype, residual=None):
    """MOE_COMBINE with the shared rows read from the tail of ``d``;
    ``residual`` ([rows, dims], ``dtype``) added last."""
    rows = scores.shape[0]
    dims = d.shape[1]
    if d.shape[0] != rows * (topk + 1) or gl.size != rows:
        raise ValueError("EXL3 shared combine geometry mismatch")
    extra = [] if residual is None else [residual]
    return _kernels()[11 if residual is None else 16](
        inputs=[d, scores, gl.reshape(rows), *extra],
        template=[("D", dims), ("K", topk), ("T", dtype)],
        grid=(dims, rows, 1),
        threadgroup=(min(256, dims), 1, 1),
        output_shapes=[(rows, dims)],
        output_dtypes=[dtype],
    )[0]


class Exl3FusedMLP(nn.Module):
    """A dense SwiGLU MLP of three Exl3DenseLinear projections: at decode
    widths one fused gate/up/SwiGLU launch and one fused down launch; wider
    calls keep the projections' own laws."""

    def __init__(self, gate_proj, up_proj, down_proj):
        super().__init__()
        if any("bias" in p for p in (gate_proj, up_proj)):
            raise ValueError("the fused gate/up chain carries no bias")
        self.gate_proj, self.up_proj, self.down_proj = gate_proj, up_proj, down_proj
        object.__setattr__(self, "fused_calls", 0)

    def __call__(self, x):
        rows = x.reshape(-1, x.shape[-1])
        if (mx.default_device() != mx.cpu and 0 < rows.shape[0] < NAX_MIN_ROWS
                and self.gate_proj.input_dims <= FUSED_GATE_UP_MAX_INPUT):
            # SwiGLU in the trunk dtype, as the unfused path casts g and u.
            h = fused_gate_up(self.gate_proj.bank, self.up_proj.bank, rows,
                              _constant("zeros", rows.shape[0]), 1, x.dtype)
            object.__setattr__(self, "fused_calls", self.fused_calls + 1)
            for proj in (self.gate_proj, self.up_proj):  # observed through the chain
                object.__setattr__(proj, "calls", proj.calls + 1)
                object.__setattr__(proj, "rows", proj.rows + rows.shape[0])
            return self.down_proj(h).astype(x.dtype).reshape(*x.shape[:-1], self.down_proj.output_dims)
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Exl3SwitchGLU(nn.Module):
    def __init__(self, gate, up, down, *, execution, trusted_router=False):
        super().__init__()
        if (
            gate.input_dims != up.input_dims
            or gate.output_dims != up.output_dims
            or down.input_dims != gate.output_dims
            or down.output_dims != gate.input_dims
            or len({gate.num_experts, up.num_experts, down.num_experts}) != 1
        ):
            raise ValueError("EXL3 gate/up/down geometry mismatch")
        if execution not in ("reference", "fused", "rows", "rows_nax", "fused_chain_nax"):
            raise ValueError(
                "EXL3 switch execution must be reference, fused, rows, rows_nax or fused_chain_nax"
            )
        row_law = execution in ("rows", "rows_nax")
        if row_law != all(p.law == "rows" for p in (gate, up, down)):
            raise ValueError("rows execution needs row-law projections, and only it")
        self.gate_proj, self.up_proj, self.down_proj = gate, up, down
        self.execution = execution
        self.trusted_router = trusted_router
        # The Qwen adapter's one-token dispatcher reads this switch contract.
        # Affine routed kernels cannot consume trellis storage.
        self.routed_decode_mode = "off"
        # Current adapter diagnostics read the ordinary switch counter too.
        object.__setattr__(self, "routed_decode_calls", 0)
        object.__setattr__(self, "fused_calls", 0)
        object.__setattr__(self, "reference_calls", 0)
        object.__setattr__(self, "rows_calls", 0)
        object.__setattr__(self, "nax_calls", 0)
        object.__setattr__(self, "chain_calls", 0)
        object.__setattr__(self, "tail_calls", 0)
        object.__setattr__(self, "shared_slot_calls", 0)
        object.__setattr__(self, "residual_calls", 0)
        object.__setattr__(self, "gate_logit_calls", 0)
        # The shared gate's 2048->1 logit inside the gate/up launch: correct
        # (0.0 max err vs the GEMV, row-invariant) but OFF: slower than the
        # GEMV launch it saves (interleaved x3 B1 decode, residual folds on:
        # 145.4 without, 141.8 with; exl3-fuse6-metal-20261105/bench). Its dot
        # product lands on one threadgroup and lengthens the critical path.
        object.__setattr__(self, "shared_gate_logit", False)
        # Shared expert as extra slots of the routed decode launches (rev 4).
        object.__setattr__(self, "shared_slots", True)
        object.__setattr__(self, "tail_declines", {})
        self.freeze()

    def _tail_decline(self, reason):
        declines = self.tail_declines
        declines[reason] = declines.get(reason, 0) + 1
        return None

    def decode_tail(self, block, x, inds, scores):
        """The routed MoE block from routing on, at decode/verify widths:
        fused gate/up/SwiGLU, fused down, the shared expert and its gate, and
        one combine launch. ``None`` (counted) declines to the block's own
        tail, which runs this module's ordinary call."""
        if self.execution != "fused_chain_nax":
            return None
        if mx.default_device() == mx.cpu:
            return self._tail_decline("cpu")
        ids = inds.reshape(-1)
        topk = inds.shape[-1]
        if ids.size >= NAX_BANK_MIN_ROWS:
            return self._tail_decline("prefill width")
        if self.gate_proj.input_dims > FUSED_GATE_UP_MAX_INPUT:
            return self._tail_decline("input exceeds fused gate/up threadgroup memory")
        if x.dtype not in (mx.bfloat16, mx.float16) or scores.dtype != x.dtype:
            return self._tail_decline("activation or score dtype")
        if (inds.shape[:-1] != x.shape[:-1] or not 1 <= topk <= 32 or ids.size == 0
                or x.shape[-1] != self.gate_proj.input_dims):
            return self._tail_decline("routing shape")
        if not self.trusted_router:
            return self._tail_decline("untrusted router")
        gate_weight = getattr(block.shared_expert_gate, "weight", None)
        if gate_weight is None or gate_weight.dtype != x.dtype:
            return self._tail_decline("shared expert gate dtype")
        flat = x.reshape(-1, x.shape[-1])
        ids = ids.astype(mx.uint32)
        shared_mlp = block.shared_expert
        cache = self.__dict__.setdefault("_shared_slot_admission", {})
        if id(shared_mlp) not in cache:  # fixed after load: admit once per module
            cache[id(shared_mlp)] = shared_slot_refusal(self, shared_mlp)
        if self.shared_slots and cache[id(shared_mlp)] is None:
            # Shared expert as T extra slots of the two routed launches.
            sg, su, sd = (getattr(shared_mlp, n).bank for n in ("gate_proj", "up_proj", "down_proj"))
            head = block.shared_expert_gate
            in_kernel = (self.shared_gate_logit and type(head) is nn.Linear and "bias" not in head
                         and head.weight.shape == (1, flat.shape[1]) and head.weight.dtype == x.dtype)
            if in_kernel:  # the shared gate's logit rides in the gate/up launch
                h, gl = shared_gate_up(self.gate_proj, self.up_proj, sg, su, flat, ids, topk, x.dtype,
                                       gate_weight=head.weight.reshape(-1))
                object.__setattr__(self, "gate_logit_calls", self.gate_logit_calls + 1)
            else:
                h = shared_gate_up(self.gate_proj, self.up_proj, sg, su, flat, ids, topk, x.dtype)
                gl = head(x)
            d = shared_down(self.down_proj, sd, h, ids, topk)
            slot = self.__dict__.get("residual_slot")
            residual = slot.take(x.shape, x.dtype) if slot is not None else None
            y = shared_combine(d, scores.reshape(-1, topk), gl, topk, x.dtype,
                               residual=None if residual is None else residual.reshape(flat.shape))
            if residual is not None:
                object.__setattr__(self, "residual_calls", self.residual_calls + 1)
            rows = flat.shape[0]
            object.__setattr__(shared_mlp, "fused_calls", shared_mlp.fused_calls + 1)
            for name in ("gate_proj", "up_proj", "down_proj"):
                proj = getattr(shared_mlp, name)
                object.__setattr__(proj, "calls", proj.calls + 1)
                object.__setattr__(proj, "rows", proj.rows + rows)
            object.__setattr__(self, "chain_calls", self.chain_calls + 1)
            object.__setattr__(self, "tail_calls", self.tail_calls + 1)
            object.__setattr__(self, "shared_slot_calls", self.shared_slot_calls + 1)
            return y.reshape(x.shape)
        h = fused_gate_up(self.gate_proj, self.up_proj, flat, ids, topk)
        d = self.down_proj.fused_rows(h, ids)
        counters = _int_counters(block.shared_expert, block.shared_expert_gate)
        shared = block.shared_expert(x).reshape(flat.shape)
        gl = block.shared_expert_gate(x)
        if shared.dtype != x.dtype or gl.dtype != x.dtype:
            # The block runs the shared expert itself after a decline; the lazy
            # outputs above are dropped unevaluated, so their call counters
            # must not stand either (sweep 2026-11-06 EXL3-6).
            for module, name, value in counters:
                object.__setattr__(module, name, value)
            return self._tail_decline("shared expert dtype")
        y = moe_combine(d, scores.reshape(-1, topk), shared, gl, topk, x.dtype)
        object.__setattr__(self, "chain_calls", self.chain_calls + 1)
        object.__setattr__(self, "tail_calls", self.tail_calls + 1)
        return y.reshape(x.shape)

    def __call__(self, x, indices):
        if self.training:
            raise ValueError("EXL3 experts are inference-only")
        if (
            indices.ndim < 1
            or x.ndim < 2
            or x.shape[-1] != self.gate_proj.input_dims
            or x.shape[:-1] != indices.shape[:-1]
            or not 1 <= indices.shape[-1] <= 32
            or indices.size == 0
        ):
            raise ValueError("EXL3 expects matching token rows and top-k 1..32")
        ids = indices.reshape(-1).astype(mx.uint32)
        if ids.size > 65535:
            raise ValueError("EXL3 call exceeds the implemented kernel grid bound")
        if (
            not self.trusted_router
            and ids.size
            and int(mx.max(ids).item()) >= self.gate_proj.num_experts
        ):
            raise ValueError("EXL3 expert index outside the bound bank")
        topk = indices.shape[-1]
        flat = x.reshape(-1, x.shape[-1])
        # Lazy: the fused chain reads token rows directly and never evaluates it.
        rows = mx.repeat(flat, topk, axis=0)
        cpu = mx.default_device() == mx.cpu
        fused = self.execution == "fused" and not cpu
        if fused and not self.gate_proj.words == self.up_proj.words == PAIR_VALIDATED_WORDS:
            raise ValueError(
                f"the Sushi fused gate/up pair is validated only at {PAIR_VALIDATED_WORDS}-word tiles"
            )
        if fused:
            i, o = self.gate_proj.input_dims, self.gate_proj.output_dims
            splits = 2 if i % 256 == 0 else 1
            partials = _kernels()[3](
                inputs=[
                    flat,
                    self.gate_proj.suh,
                    self.up_proj.suh,
                    self.gate_proj.trellis,
                    self.up_proj.trellis,
                    ids,
                ],
                template=[
                    ("IDIM", i),
                    ("ODIM", o),
                    ("NSPLIT", splits),
                    ("NHW", self.gate_proj.words),
                    ("TOPK", topk),
                    ("FUNNEL", True),
                    ("OTPT", 2),
                ],
                grid=(o // 32 * 128, ids.size, splits),
                threadgroup=(128, 1, 1),
                output_shapes=[(ids.size * splits, o)] * 2,
                output_dtypes=[mx.float32] * 2,
            )
            g, u = [
                proj.finish(value.reshape(ids.size, splits, o).sum(axis=1), ids)
                for value, proj in zip(partials, (self.gate_proj, self.up_proj))
            ]
            object.__setattr__(self, "fused_calls", self.fused_calls + 1)
        elif (self.execution == "fused_chain_nax" and ids.size < NAX_BANK_MIN_ROWS and not cpu
              and self.gate_proj.input_dims <= FUSED_GATE_UP_MAX_INPUT):
            # Decode/verify widths: gate/up/SwiGLU and down, two launches.
            h = fused_gate_up(self.gate_proj, self.up_proj, flat, ids, topk)
            out = self.down_proj.fused_rows(h, ids).astype(x.dtype)
            object.__setattr__(self, "chain_calls", self.chain_calls + 1)
            return out.reshape(*indices.shape, self.down_proj.output_dims)
        elif self.execution in ("rows_nax", "fused_chain_nax") and ids.size >= NAX_BANK_MIN_ROWS:
            # Prefill widths: expert-sorted reconstruction + gather_mm, one
            # layout for all three projections.
            out = nax_switch(self, flat, ids, topk, x.dtype)
            object.__setattr__(self, "nax_calls", self.nax_calls + 1)
            return out.reshape(*indices.shape, self.down_proj.output_dims)
        elif self.execution in ("rows", "rows_nax"):
            # One plan for all three projections: they share the routed rows.
            plan = row_plan(ids, self.gate_proj.num_experts)
            g, u = (
                self.gate_proj(rows, ids, trusted_indices=True, plan=plan),
                self.up_proj(rows, ids, trusted_indices=True, plan=plan),
            )
            object.__setattr__(self, "rows_calls", self.rows_calls + 1)
        else:
            plan = None
            g, u = (
                self.gate_proj(rows, ids, trusted_indices=True),
                self.up_proj(rows, ids, trusted_indices=True),
            )
            object.__setattr__(self, "reference_calls", self.reference_calls + 1)
        # Keep output in the trunk activation dtype so the existing router's
        # weighted tail and shared expert retain their dtype contract.
        down_plan = plan if self.execution in ("rows", "rows_nax") else None
        out = self.down_proj(
            swiglu(g, u), ids, trusted_indices=True, plan=down_plan
        ).astype(x.dtype)
        return out.reshape(*indices.shape, self.down_proj.output_dims)

    def receipt(self):
        return {
            "format_revision": FORMAT_REVISION,
            "execution": self.execution,
            "qualification": "unqualified",
            "fused_calls": self.fused_calls,
            "reference_calls": self.reference_calls,
            "rows_calls": self.rows_calls,
            "nax_calls": self.nax_calls,
            "chain_calls": self.chain_calls,
            "tail_calls": self.tail_calls,
            "shared_slot_calls": self.shared_slot_calls,
            "residual_calls": self.residual_calls,
            "gate_logit_calls": self.gate_logit_calls,
            "tail_declines": dict(self.tail_declines),
            "packed_weight_bytes": sum(
                p.packed_nbytes for p in (self.gate_proj, self.up_proj, self.down_proj)
            ),
        }


def install_exl3_experts(model, weights, metadata):
    """Adapter load hook: admit all split expert banks before replacing any.

    Keys use Sushi's split ``switch_mlp.{gate,up,down}_proj.{trellis,suh,svh}``.
    Unmarked trellis tensors and mixed affine/EXL3 blocks fail closed.
    """
    meta = validate_metadata(metadata)
    # The marker declares one rate; every bank must carry exactly its tiles.
    words = int(meta["bits_per_weight"] * 16)
    if any(v.shape[-1] != words for k, v in weights.items() if k.endswith(".trellis")):
        raise ValueError(f"EXL3 pack marker declares {words}-word tiles")
    replacements = install_banks(model, weights, execution=meta["execution"])
    return {
        "format_revision": FORMAT_REVISION,
        **meta,
        "qualification": "unqualified",
        "blocks": len(replacements),
        "packed_weight_bytes": sum(m.receipt()["packed_weight_bytes"] for m in replacements),
    }


def install_banks(model, weights, *, execution):
    """Replace every routed ``switch_mlp`` with an ``Exl3SwitchGLU``.

    All banks are admitted before any module is swapped; returns the new
    switches. Callers own format metadata and its identity.
    """
    modules = dict(model.named_modules())
    roots = {key.rsplit(".", 2)[0] for key in weights if key.endswith(".trellis")}
    if not roots:
        raise ValueError("EXL3 metadata has no packed expert tensors")
    replacements = []
    for root in sorted(roots):
        old = modules.get(root)
        if old is None or not root.endswith(".switch_mlp") or "down_proj" not in old:
            raise ValueError(f"EXL3 tensor has no supported switch module: {root}")
        parent = modules.get(root.rpartition(".")[0])
        if parent is None or getattr(parent, "shared_folded", False):
            raise ValueError(
                "EXL3 does not support shared experts folded into routed tables"
            )
        down = old.down_proj
        allowed = {
            f"{root}.{name}.{part}"
            for name in ("gate_proj", "up_proj", "down_proj")
            for part in ("trellis", "suh", "svh")
        }
        if any(key.startswith(root + ".") and key not in allowed for key in weights):
            raise ValueError("mixed EXL3/affine expert tensors are unsupported")
        projs = []
        for name in ("gate_proj", "up_proj", "down_proj"):
            keys = [f"{root}.{name}.{k}" for k in ("trellis", "suh", "svh")]
            if any(key not in weights for key in keys):
                raise ValueError(f"incomplete EXL3 block: {root}")
            proj = Exl3Linear(
                *(weights[key] for key in keys),
                law="rows" if execution in ("rows", "rows_nax") else "reference",
            )
            expected = (
                (down.input_dims, down.output_dims)
                if name == "down_proj"
                else (down.output_dims, down.input_dims)
            )
            if (
                proj.input_dims,
                proj.output_dims,
            ) != expected or proj.num_experts != down.num_experts:
                raise ValueError(
                    f"EXL3 pack disagrees with model topology: {root}.{name}"
                )
            if any(
                key.startswith(f"{root}.{name}.") and key not in keys for key in weights
            ):
                raise ValueError("mixed EXL3/affine expert tensors are unsupported")
            projs.append(proj)
        replacements.append(
            (
                root,
                parent,
                Exl3SwitchGLU(*projs, execution=execution, trusted_router=True),
            )
        )
    # Refuse partial trunks: loading a mixed model must be an explicit future
    # format, not an accidental omission of one packed layer.
    supported = {
        p for p, m in modules.items() if p.endswith(".switch_mlp") and "down_proj" in m
    }
    if roots != supported:
        raise ValueError("EXL3 metadata must cover every routed switch block")
    from ..weight_stream import _swap_module

    for root, parent, replacement in replacements:
        _swap_module(model, root, replacement)
        parent.fused_gate_up = False
        parent.fused_expert_kernel_mode = "stock"
        parent.moe_weighted_sum = False
    return [replacement for _, _, replacement in replacements]


class Exl3DenseLinear(nn.Module):
    """One stored EXL3 linear, run on the indexed kernels as a one-expert bank.

    Replaces an ``nn.Linear`` of the same geometry; output keeps the input
    dtype. Ordinary per-row projection only (no wide-row reconstruct yet).
    """

    # One kernel launch addresses at most this many rows (grid y).
    ROWS_PER_LAUNCH = 65535

    LAWS = ("reference", "rows", "fused", "fused_nax", "fused_w8a8")

    def __init__(self, trellis, suh, svh, bias=None, *, law="reference"):
        super().__init__()
        if law not in self.LAWS:
            raise ValueError(f"EXL3 dense law must be one of {self.LAWS}")
        # "fused": FUSED_LINEAR_SOURCE, one launch per call at any width.
        # "fused_nax": fused below NAX_MIN_ROWS rows, fp16 reconstruction and
        # mx.matmul at or above (two laws split by width, like affine qmv/qmm).
        # "fused_w8a8": fused below NAX_MIN_ROWS, approximate W8A8 above: the
        # prepared rows and the reconstructed weights quantized per row and per
        # output channel to int8, int8 GEMM on the tensor units (int8_prefill).
        self.law = law
        self.bank = Exl3Linear(trellis[None], suh[None], svh[None],
                               law="rows" if law == "rows" else "reference")
        if bias is not None:
            if bias.shape != (self.bank.output_dims,):
                raise ValueError("EXL3 bias must match the output width")
            self.bias = bias
        object.__setattr__(self, "calls", 0)
        object.__setattr__(self, "rows", 0)
        self.freeze()

    @property
    def input_dims(self):
        return self.bank.input_dims

    @property
    def output_dims(self):
        return self.bank.output_dims

    def _nax(self, flat):
        """prepare, then per column slice reconstruct + matmul, then finish."""
        bank = self.bank
        ids = _constant("zeros", flat.shape[0])
        prepared = bank.prepare(flat, ids)
        zero = _constant("zeros", 1)
        inner = []
        for lo in range(0, bank.output_dims, NAX_SLICE_COLUMNS):
            columns = min(NAX_SLICE_COLUMNS, bank.output_dims - lo)
            weight = bank.reconstruct(zero, lo // 16, columns)[0]
            inner.append(mx.matmul(prepared, weight))
        inner = inner[0] if len(inner) == 1 else mx.concatenate(inner, axis=1)
        # The trunk-dtype cast rides in the finish launch (bitwise astype).
        return bank.finish_map(inner, ids, _constant("arange", flat.shape[0]), flat.shape[0], flat.dtype)

    def make_resident(self, min_rows):
        """Keep the fp16 reconstruction (4x the trellis bytes) and serve every
        call of ``min_rows``+ rows from it: the _nax arithmetic without the
        per-call reconstruction (exl3-nax-survey-20261105: lm_head at 16 rows
        2.4 vs 6.6 ms fused). Only for the fused_nax law."""
        if self.law != "fused_nax":
            raise ValueError("resident EXL3 weights extend the fused_nax law only")
        zero = mx.zeros((1,), mx.uint32)
        slices = []
        for lo in range(0, self.bank.output_dims, NAX_SLICE_COLUMNS):
            columns = min(NAX_SLICE_COLUMNS, self.bank.output_dims - lo)
            slices.append(self.bank.reconstruct(zero, lo // 16, columns)[0])
        mx.eval(slices)
        object.__setattr__(self, "_resident", slices)
        object.__setattr__(self, "resident_min_rows", int(min_rows))
        object.__setattr__(self, "resident_calls", 0)
        return sum(w.nbytes for w in slices)

    def _nax_resident(self, flat):
        bank = self.bank
        ids = _constant("zeros", flat.shape[0])
        prepared = bank.prepare(flat, ids)
        inner = [mx.matmul(prepared, w) for w in self._resident]
        inner = inner[0] if len(inner) == 1 else mx.concatenate(inner, axis=1)
        return bank.finish_map(inner, ids, _constant("arange", flat.shape[0]), flat.shape[0], flat.dtype)

    def _w8a8(self, flat):
        """Approximate: int8 rows x int8 output channels, then finish."""
        from ..int8_prefill import (
            MPP_SLICE_OFFSET_LIMIT, _int8_gemm, _quantize_rows, max_tensor_slice_offset,
        )

        bank = self.bank
        width = min(NAX_SLICE_COLUMNS, bank.output_dims)
        if max_tensor_slice_offset(rows=flat.shape[0], k=bank.input_dims, n=width) >= MPP_SLICE_OFFSET_LIMIT:
            # MPP tensor slices past 2 GiB land low on M5 (see int8_prefill).
            object.__setattr__(self, "w8a8_fallbacks", getattr(self, "w8a8_fallbacks", 0) + 1)
            return self._nax(flat)
        ids = _constant("zeros", flat.shape[0])
        xq, xs = _quantize_rows(bank.prepare(flat, ids))
        zero = _constant("zeros", 1)
        inner = []
        for lo in range(0, bank.output_dims, NAX_SLICE_COLUMNS):
            columns = min(NAX_SLICE_COLUMNS, bank.output_dims - lo)
            wq, ws = _quantize_rows(bank.reconstruct(zero, lo // 16, columns, transposed=True)[0])
            inner.append(_int8_gemm(xq, xs, wq, ws))
        inner = inner[0] if len(inner) == 1 else mx.concatenate(inner, axis=1)
        return bank.finish(inner, ids)

    def _fused(self, flat):
        return self.bank.fused_rows(flat, _constant("zeros", flat.shape[0]))

    @property
    def words(self):
        return self.bank.words

    def __call__(self, x):
        if self.training:
            raise ValueError("EXL3 linears are inference-only")
        if x.shape[-1] != self.input_dims:
            raise ValueError("EXL3 linear input width mismatch")
        flat = x.reshape(-1, self.input_dims)
        rows = flat.shape[0]
        if rows == 0:
            # nn.Linear semantics for an empty batch; the launch loop below
            # would build no parts (sweep 2026-11-06 EXL3-5).
            out = mx.zeros((*x.shape[:-1], self.output_dims), x.dtype)
            return out + self.bias.astype(x.dtype) if "bias" in self else out
        resident = self.__dict__.get("_resident")
        if resident is not None and rows >= self.resident_min_rows and mx.default_device() != mx.cpu:
            out = self._nax_resident(flat).reshape(*x.shape[:-1], self.output_dims)
            if "bias" in self:
                out = out + self.bias.astype(x.dtype)
            object.__setattr__(self, "calls", self.calls + 1)
            object.__setattr__(self, "rows", self.rows + rows)
            object.__setattr__(self, "resident_calls", self.resident_calls + 1)
            return out
        if self.law in ("fused_nax", "fused_w8a8") and rows >= NAX_MIN_ROWS:
            wide = self._nax(flat) if self.law == "fused_nax" else self._w8a8(flat)
            out = wide.astype(x.dtype).reshape(*x.shape[:-1], self.output_dims)
            if "bias" in self:
                out = out + self.bias.astype(x.dtype)
            object.__setattr__(self, "calls", self.calls + 1)
            object.__setattr__(self, "rows", self.rows + rows)
            object.__setattr__(self, "nax_calls", getattr(self, "nax_calls", 0) + 1)
            return out
        if self.law in ("fused", "fused_nax", "fused_w8a8") and mx.default_device() != mx.cpu and rows:
            slot = self.__dict__.get("residual_slot")
            residual = (slot.take((*x.shape[:-1], self.output_dims), x.dtype)
                        if slot is not None and "bias" not in self else None)
            if residual is not None:
                out = self.bank.fused_rows(flat, _constant("zeros", rows),
                                           residual=residual.reshape(rows, self.output_dims))
                object.__setattr__(self, "calls", self.calls + 1)
                object.__setattr__(self, "rows", self.rows + rows)
                object.__setattr__(self, "residual_calls", getattr(self, "residual_calls", 0) + 1)
                return out.reshape(*x.shape[:-1], self.output_dims)
            out = self._fused(flat).reshape(*x.shape[:-1], self.output_dims)
            if "bias" in self:
                out = out + self.bias.astype(x.dtype)
            object.__setattr__(self, "calls", self.calls + 1)
            object.__setattr__(self, "rows", self.rows + rows)
            return out
        outs = []
        for lo in range(0, rows, self.ROWS_PER_LAUNCH):
            part = flat[lo : lo + self.ROWS_PER_LAUNCH]
            ids = _constant("zeros", part.shape[0])
            outs.append(self.bank(part, ids, trusted_indices=True))
        out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
        out = out.astype(x.dtype).reshape(*x.shape[:-1], self.output_dims)
        if "bias" in self:
            out = out + self.bias.astype(x.dtype)
        object.__setattr__(self, "calls", self.calls + 1)
        object.__setattr__(self, "rows", self.rows + rows)
        return out
