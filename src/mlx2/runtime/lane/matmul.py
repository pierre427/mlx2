"""Row-invariant small-M matmul on the M5 tensor units, for every weight format.

A speculative verify runs several rows through one forward.  MLX's quantized
matmul picks a different kernel (and summation order) by row count, and on
M5 its per-row cost climbs steeply from four rows.  This "lane" matmul uses
one arithmetic for every row count, reads each weight once for all rows, and
costs about the same for 1 and 16 rows.

For weight group ``g`` (``GS`` inputs, one scale ``s`` and bias ``b`` per
output column), with ``q`` the unsigned integer weights:

    P[m, n, g] = x[m, g-block] . q[n, g-block]        tensor units, fp32 result
    y[m, n]    = sum over g, in order, of  s[n, g] * P + b[n, g] * xs[m, g]

``xs[m, g]`` is the fp32 sum of the group's inputs, added sequentially.  The
K groups are split into ``SK`` slices chosen from the weight shape and format
only, never from the row count, and the slices are added in slice order.  Every
row therefore gets the same bits whether it is computed alone or with others.
Unquantized weights (``bits == 16``) accumulate ``P`` directly.

The weight format only changes how ``q`` reaches the tensor units:

* 4-bit: MPP's ``uint4b_format`` reads MLX's packed nibbles in place.
* 8-bit: MPP's ``uint8_t`` reads MLX's packed bytes in place.
* bf16/fp16 (``bits == 16``): read in place.
* 2/3/5/6-bit: no native format.  Each simdgroup unpacks its group's
  ``NT x GS`` values from MLX's LSB-first bitstream into threadgroup memory
  as ``uint8`` and runs the same ``uint8`` product.

This is not bitwise equal to MLX's own kernels.  It defines a numerical law,
and a caller that needs serial/verify equality must use it for one-row decode
as well.  The q4 structure (fragment mapping, (s, b) pairing, slice
reduction) is adapted from TensorFold ``bb4b4a35`` ``lane_qmm`` (MIT); see
provenance/lane-matmul.json.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

QUANT_BITS = (2, 3, 4, 5, 6, 8)
UNQUANTIZED_BITS = 16
GROUP_SIZES = (32, 64, 128)
NT = 32               # output columns per simdgroup tile (one column per lane when unpacking)
ROW_BLOCK = 32        # rows per threadgroup; 16-row fragments (TMR = 1 or 2)
MAX_ROWS = 128        # rows accepted by one call
_NATIVE = {4: "uint4b_format", 8: "uint8_t"}
_TYPES = {mx.bfloat16: "bfloat", mx.float16: "half"}

_HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""

# fp32 group sums of x, added in index order.
_XSUM = r"""
  const int M = mdims[0], MP = mdims[1];
  const uint m = thread_position_in_grid.y;
  const uint g = thread_position_in_grid.x;
  if (g >= K / GS || int(m) >= MP) return;
  float acc = 0.0f;
  if (int(m) < M) for (int i = 0; i < GS; i++) acc += float(X[m * K + g * GS + i]);
  XS[g * MP + m] = acc;
"""

_MAIN = r"""
  const ushort lane = thread_index_in_simdgroup;
  const ushort sg = simdgroup_index_in_threadgroup;     // K slice
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);       // fragment row of this lane (and fm + 8)
  const short fn = ((qid & 2) | (lane & 1)) * 4;        // first of its four fragment columns
  const int M = mdims[0], MP = mdims[1];
  constexpr int KG = K / GS;
  constexpr int NF = NT / 16;
  const int n0 = threadgroup_position_in_grid.x * NT;
  const int rb = threadgroup_position_in_grid.y * 16 * TMR;
  const int g_begin = (sg * KG) / SK;
  const int g_end = ((sg + 1) * KG) / SK;
  constexpr auto desc = matmul2d_descriptor(16 * TMR, NT, GS, false, true, false,
                                            matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device XT, dextents<int32_t, 2>, tensor_inline> tA(
      (device XT*)X + (int64_t)rb * K, dextents<int32_t, 2>(K, M - rb));
@@B_DECL@@
  float C[TMR][NF * 8];
  for (int t = 0; t < TMR; t++) for (int i = 0; i < NF * 8; i++) C[t][i] = 0.0f;
@@SB_DECL@@
  for (int g = g_begin; g < g_end; g++) {
@@SB_LOAD@@
    auto a = tA.slice(g * GS, 0);
@@B_GROUP@@
    auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    op.run(a, b, P);
@@ACCUM@@
@@AFTER@@
  }
  // K slices are added in slice order, one 16-row block at a time.
@@PART@@
  for (int t = 0; t < TMR; t++) {
    if (SK > 1) {
      if (sg > 0) for (int i = 0; i < NF * 8; i++) part[((sg - 1) * NF * 8 + i) * 32 + lane] = C[t][i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0)
        for (int s2 = 1; s2 < SK; s2++)
          for (int i = 0; i < NF * 8; i++) C[t][i] += part[((s2 - 1) * NF * 8 + i) * 32 + lane];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == 0)
      for (int f = 0; f < NF; f++)
        for (int r = 0; r < 2; r++) {
          const int m = rb + t * 16 + fm + 8 * r;
          const int n = n0 + f * 16 + fn;
          if (m < M && n < N)
            for (int j = 0; j < 4; j++) Y[m * N + n + j] = static_cast<XT>(C[t][f * 8 + r * 4 + j]);
        }
  }
"""

_SB_DECL = r"""
  const device uint4* sbv = (const device uint4*)SBt;   // (s, b) pairs, [g][n], ST precision
  bool colok[NF];
  for (int f = 0; f < NF; f++) colok[f] = n0 + f * 16 + fn < N;
"""

_SB_LOAD = r"""
    float s[NF][4], bb[NF][4];
    for (int f = 0; f < NF; f++) {
      const uint4 q = colok[f] ? sbv[(g * N + n0 + f * 16 + fn) / 4] : uint4(0);
      const vec<ST, 8> v = as_type<vec<ST, 8>>(q);
      for (int j = 0; j < 4; j++) { s[f][j] = float(v[2 * j]); bb[f][j] = float(v[2 * j + 1]); }
    }
"""

_ACCUM_AFFINE = r"""
    for (int t = 0; t < TMR; t++) {
      const float xs0 = XS[g * MP + rb + t * 16 + fm];
      const float xs1 = XS[g * MP + rb + t * 16 + fm + 8];
      for (int f = 0; f < NF; f++)
        for (int r = 0; r < 2; r++)
          for (int j = 0; j < 4; j++) {
            const int i = f * 8 + r * 4 + j;
            C[t][i] = fma(s[f][j], P[t * NF * 8 + i], fma(bb[f][j], r ? xs1 : xs0, C[t][i]));
          }
    }
"""

_ACCUM_PLAIN = r"""
    for (int t = 0; t < TMR; t++)
      for (int i = 0; i < NF * 8; i++) C[t][i] += P[t * NF * 8 + i];
"""

# Native formats read MLX's weight layout in place: row n, K contiguous.  Each
# column tile's view starts at its own rows (64-bit offset), so weights over
# 2 GB stay within the tensor's 32-bit indexing.
_B_DECL_NATIVE = r"""
  constexpr int64_t ROW_UNITS = (BITS == 4) ? K / 2 : K;   // WP elements per weight row
  tensor<device WT, dextents<int32_t, 2>, tensor_inline> tB(
      (device WP*)W + (int64_t)n0 * ROW_UNITS, dextents<int32_t, 2>(K, N - n0));
"""
_B_GROUP_NATIVE = "    auto b = tB.slice(g * GS, 0);\n"

# 2/3/5/6-bit: lane l unpacks output column n0 + l of this group into uint8.
# The group's packed bits are loaded as whole words and decoded with static
# shifts (GS and BITS are compile-time), four values per threadgroup store.
_B_DECL_UNPACK = r"""
  constexpr int ROW_BYTES = K * BITS / 8;
  constexpr int GROUP_BYTES = GS * BITS / 8;
  constexpr int GROUP_WORDS = GROUP_BYTES / 4;
  // Staging and the slice-reduction buffer share memory: staging is dead
  // once every slice leaves the group loop (keeps two threadgroups per core).
  constexpr int STAGE_WORDS = SK * NT * GS / 4;
  constexpr int PART_WORDS = (SK > 1 ? SK - 1 : 1) * (NT / 16) * 8 * 32;
  threadgroup uint wbuf32[STAGE_WORDS > PART_WORDS ? STAGE_WORDS : PART_WORDS];
  threadgroup uchar* mine = (threadgroup uchar*)wbuf32 + sg * NT * GS;
"""
def _unpack_group_source(bits: int, group_size: int) -> str:
    """Fully unrolled decode of one group (static word indices and shifts).

    Lane ``l`` loads column ``n0 + l``'s packed words for the group and writes
    four uint8 values per threadgroup word, [n][k] with K contiguous.
    """
    mask = (1 << bits) - 1
    words = group_size * bits // 32
    loads = "\n".join(f"        const uint w{i} = src[{i}];" for i in range(words))
    stores = []
    for q in range(group_size // 4):
        terms = []
        for e in range(4):
            bit = (q * 4 + e) * bits
            wi, sh = bit >> 5, bit & 31
            value = f"(w{wi} >> {sh})" if sh else f"w{wi}"
            if sh + bits > 32:
                value = f"({value} | (w{wi + 1} << {32 - sh}))"
            value = f"({value} & {mask}u)"
            terms.append(value if e == 0 else f"({value} << {8 * e})")
        stores.append(f"        dst[{q}] = " + " | ".join(terms) + ";")
    return (
        "    {\n"
        "      threadgroup uint* dst = (threadgroup uint*)(mine + lane * GS);\n"
        "      const int n = n0 + lane;\n"
        "      if (n < N) {\n"
        "        const device uint* src = (const device uint*)(\n"
        "            (const device uchar*)W + (int64_t)n * ROW_BYTES + (int64_t)g * GROUP_BYTES);\n"
        f"{loads}\n" + "\n".join(stores) + "\n"
        "      } else {\n"
        "        for (int q = 0; q < GS / 4; q++) dst[q] = 0u;\n"
        "      }\n"
        "    }\n"
        "    simdgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    tensor<threadgroup uchar, dextents<int32_t, 2>, tensor_inline> b(mine, dextents<int32_t, 2>(GS, NT));\n"
    )


_AFTER_UNPACK = "    simdgroup_barrier(mem_flags::mem_threadgroup);\n"


class LaneUnsupported(ValueError):
    """This projection cannot use the lane matmul without changing its contract."""


@dataclass(frozen=True)
class LaneWeights:
    """Per-projection preparation; the source module's arrays are shared, not copied."""

    bits: int
    group_size: int
    n: int
    k: int
    split_k: int
    weight: Any                 # MLX packed uint32 (quantized) or bf16/fp16 (N, K)
    scale_bias: Any = None      # (K/GS, N, 2) in the scales' dtype; None when unquantized
    bias: Any = None            # optional additive bias (N,)

    @property
    def format(self) -> str:
        return "unquantized" if self.bits == UNQUANTIZED_BITS else f"affine-q{self.bits}-g{self.group_size}"


def split_k(n: int, k: int, group_size: int, bits: int) -> int:
    """K slices for one weight: fixed by shape and format, never by row count."""
    tiles = -(-n // NT)
    groups = k // group_size
    # Unpacking stages SK x NT x GS bytes; keep it within 16 KB so two
    # threadgroups fit on a core (the slice buffer reuses the same memory).
    cap = 8
    if bits not in _NATIVE and bits != UNQUANTIZED_BITS:
        while cap > 1 and cap * NT * group_size > 16 * 1024:
            cap //= 2
    sk = 1
    while sk < cap and tiles * sk < 1024 and groups // (sk * 2) >= 8:
        sk *= 2
    return sk


def check_geometry(*, bits: int, group_size: int, mode: str, n: int, k: int,
                   weight_dtype, scales_dtype=None) -> None:
    """Raise LaneUnsupported unless the lane kernels cover this projection."""
    if bits == UNQUANTIZED_BITS:
        if weight_dtype not in _TYPES:
            raise LaneUnsupported("unquantized weights must be bf16 or fp16")
        if k % 64:
            raise LaneUnsupported("unquantized K must be a multiple of 64")
    else:
        if mode != "affine":
            raise LaneUnsupported(f"quantization mode {mode!r} is not affine")
        if bits not in QUANT_BITS:
            raise LaneUnsupported(f"{bits}-bit affine weights are not supported")
        if group_size not in GROUP_SIZES:
            raise LaneUnsupported(f"group size {group_size} is not supported")
        if weight_dtype != mx.uint32:
            raise LaneUnsupported("packed weights must be uint32")
        if scales_dtype not in _TYPES:
            raise LaneUnsupported("scales and biases must be bf16 or fp16")
        if k % group_size:
            raise LaneUnsupported("K must be a multiple of the group size")
    if n % 4 or n < 4:
        raise LaneUnsupported("N must be a positive multiple of 4")


def prepare(module) -> LaneWeights:
    """Prepare an ``nn.QuantizedLinear`` or ``nn.Linear`` for the lane matmul."""
    from mlx import nn

    if isinstance(module, nn.QuantizedLinear):
        weight, scales, biases = module["weight"], module["scales"], module.get("biases")
        bits, group_size = int(module.bits), int(module.group_size)
        mode = getattr(module, "mode", "affine")
        if biases is None:
            raise LaneUnsupported("affine lane matmul needs quantization biases")
        n = int(weight.shape[0])
        k = int(weight.shape[1]) * 32 // bits
        check_geometry(bits=bits, group_size=group_size, mode=mode, n=n, k=k,
                       weight_dtype=weight.dtype, scales_dtype=scales.dtype)
        if tuple(scales.shape) != (n, k // group_size) or biases.shape != scales.shape:
            raise LaneUnsupported("scale/bias geometry does not match the packed weight")
        pairs = mx.contiguous(mx.stack([scales.T, biases.T], axis=-1).astype(scales.dtype))
        return LaneWeights(bits, group_size, n, k, split_k(n, k, group_size, bits),
                           weight, pairs, module.get("bias"))
    if isinstance(module, nn.Linear):
        weight = module["weight"]
        if weight.ndim != 2:
            raise LaneUnsupported("linear weight must be rank 2")
        n, k = map(int, weight.shape)
        check_geometry(bits=UNQUANTIZED_BITS, group_size=64, mode="none", n=n, k=k,
                       weight_dtype=weight.dtype)
        return LaneWeights(UNQUANTIZED_BITS, 64, n, k, split_k(n, k, 64, UNQUANTIZED_BITS),
                           weight, None, module.get("bias"))
    raise LaneUnsupported(f"{type(module).__name__} is not a linear projection")


def main_source(bits: int, group_size: int) -> str:
    """The main kernel body for one weight format (placeholders substituted)."""
    affine = bits != UNQUANTIZED_BITS
    if bits in _NATIVE or not affine:
        b_decl, b_group, after = _B_DECL_NATIVE, _B_GROUP_NATIVE, ""
    else:
        b_decl, after = _B_DECL_UNPACK, _AFTER_UNPACK
        b_group = _unpack_group_source(bits, group_size)
    part = ("  threadgroup float part[(SK > 1 ? SK - 1 : 1) * NF * 8 * 32];\n"
            if b_decl is _B_DECL_NATIVE else
            "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "  threadgroup float* part = (threadgroup float*)wbuf32;\n")
    parts = {
        "@@PART@@": part,
        "@@B_DECL@@": b_decl,
        "@@SB_DECL@@": _SB_DECL if affine else "",
        "@@SB_LOAD@@": _SB_LOAD if affine else "",
        "@@B_GROUP@@": b_group,
        "@@ACCUM@@": _ACCUM_AFFINE if affine else _ACCUM_PLAIN,
        "@@AFTER@@": after,
    }
    source = _MAIN
    for marker, text in parts.items():
        if source.count(marker) != 1:
            raise AssertionError(f"lane kernel anchor {marker} changed")
        source = source.replace(marker, text)
    return source


_KERNELS: dict[tuple, Any] = {}
_MDIMS: dict[tuple[int, int], Any] = {}
_XS_CACHE: dict[tuple[int, int], tuple[Any, Any]] = {}


def _named(base: str, source: str) -> str:
    # MLX caches compiled kernels by name, so the name carries the source hash.
    return f"mlx2_lane_{base}_{hashlib.sha256((_HEADER + source).encode()).hexdigest()[:16]}"


def _kernel(bits: int, group_size: int, xt: str, wt: str, st: str) -> Any:
    key = (bits, group_size, xt, wt, st)
    if key not in _KERNELS:
        source = main_source(bits, group_size)
        wp = "uchar" if bits != UNQUANTIZED_BITS else wt
        wtype = _NATIVE.get(bits, wt if bits == UNQUANTIZED_BITS else "uchar")
        header = (_HEADER + f"typedef {xt} XT;\ntypedef {wtype} WT;\ntypedef {wp} WP;\n"
                  f"typedef {st} ST;\nconstexpr constant int BITS = {bits};\n")
        _KERNELS[key] = mx.fast.metal_kernel(
            name=_named(f"q{bits}g{group_size}_{xt}_{wt}_{st}", header + source),
            input_names=["X", "XS", "W", "SBt", "mdims"], output_names=["Y"],
            source=source, header=header, ensure_row_contiguous=True,
        )
    return _KERNELS[key]


def _xsum_kernel(xt: str) -> Any:
    key = ("xsum", xt)
    if key not in _KERNELS:
        header = _HEADER + f"typedef {xt} XT;\n"
        _KERNELS[key] = mx.fast.metal_kernel(
            name=_named(f"xsum_{xt}", header + _XSUM),
            input_names=["X", "mdims"], output_names=["XS"],
            source=_XSUM, header=header,
        )
    return _KERNELS[key]


def _mdims(m: int, mp: int):
    key = (m, mp)
    if key not in _MDIMS:
        _MDIMS[key] = mx.array(key, dtype=mx.int32)
    return _MDIMS[key]


def _group_sums(x2, k: int, group_size: int, mdims, mp: int, xt: str):
    """Group sums of x, shared by projections that read the same input."""
    key = (id(x2), group_size)
    hit = _XS_CACHE.get(key)
    if hit is not None and hit[0] is x2:
        return hit[1]
    groups = k // group_size
    xs = _xsum_kernel(xt)(
        inputs=[x2, mdims], template=[("K", k), ("GS", group_size)],
        grid=(groups, mp, 1), threadgroup=(min(groups, 256), 1, 1),
        output_shapes=[(groups, mp)], output_dtypes=[mx.float32],
    )[0]
    _XS_CACHE[key] = (x2, xs)
    while len(_XS_CACHE) > 8:
        _XS_CACHE.pop(next(iter(_XS_CACHE)))
    return xs


def lane_matmul(x, lw: LaneWeights):
    """``x @ W.T`` (+ bias) for ``x`` of shape (..., K) with at most MAX_ROWS rows."""
    xt = _TYPES.get(x.dtype)
    if xt is None:
        raise LaneUnsupported("activations must be bf16 or fp16")
    k = int(x.shape[-1])
    if k != lw.k:
        raise LaneUnsupported("activation width differs from the weight")
    lead = x.shape[:-1]
    x2 = x.reshape(-1, k)
    m = int(x2.shape[0])
    if not 1 <= m <= MAX_ROWS:
        raise LaneUnsupported(f"lane matmul takes 1-{MAX_ROWS} rows, got {m}")
    mp = 16 * ((m + 15) // 16)
    block = min(mp, ROW_BLOCK)
    mdims = _mdims(m, mp)
    if lw.bits == UNQUANTIZED_BITS:
        wt = _TYPES[lw.weight.dtype]
        st = "bfloat"
        xs = mdims                       # unused by the plain accumulator
        sbt = mdims
    else:
        wt = "uchar"
        st = _TYPES[lw.scale_bias.dtype]
        xs = _group_sums(x2, k, lw.group_size, mdims, mp, xt)
        sbt = lw.scale_bias
    y = _kernel(lw.bits, lw.group_size, xt, wt, st)(
        inputs=[x2, xs, lw.weight, sbt, mdims],
        template=[("TMR", block // 16), ("N", lw.n), ("K", k), ("NT", NT),
                  ("SK", lw.split_k), ("GS", lw.group_size)],
        grid=(-(-lw.n // NT) * 32 * lw.split_k, -(-mp // block), 1),
        threadgroup=(32 * lw.split_k, 1, 1),
        output_shapes=[(m, lw.n)], output_dtypes=[x.dtype],
    )[0].reshape(*lead, lw.n)
    if lw.bias is not None:
        y = y + lw.bias
    return y


_M5: list[bool] = []


def available() -> bool:
    """True when the default device can run the M5 tensor-unit kernels."""
    if mx.default_device() != mx.gpu:
        return False
    if not _M5:
        try:
            info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
            _M5.append(mx.metal.is_available() and "M5" in str(info.get("device_name", "")))
        except Exception:  # noqa: BLE001 - absent Metal means unavailable
            _M5.append(False)
    return _M5[0]
