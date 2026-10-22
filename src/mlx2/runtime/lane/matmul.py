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

``xs[m, g]`` is the fp32 sum of the group's inputs, added sequentially inside
the same kernel (one launch per projection).  The
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
reduction) is adapted from TensorFold ``bb4b4a35`` ``lane_qmm``
(MIT License, Copyright (c) 2026 TensorFold contributors; full notice in the
repository's NOTICE file; provenance in provenance/lane-matmul.json).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from . import simd

QUANT_BITS = (2, 3, 4, 5, 6, 8)
UNQUANTIZED_BITS = 16
GROUP_SIZES = (32, 64, 128)
NT = 32               # output columns per simdgroup tile (one column per lane when unpacking)
# One 16-row MPP fragment per threadgroup.  Interleaved M5 measurements on the
# packed DFlash shapes found that TMR=1 is faster than a 32-row/TMR=2 block at
# the stable 32/64/128-row cells while preserving the established one-row lane
# law; the small 16-row cell is shape/noise dependent and B1 is neutral.
ROW_BLOCK = 16
MAX_ROWS = 128        # rows accepted by one call
_NATIVE = {4: "uint4b_format", 8: "uint8_t"}
_TYPES = {mx.bfloat16: "bfloat", mx.float16: "half"}

_HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
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
@@RUN@@
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

# The group sums of x are computed in the kernel (no second launch).  The four
# lanes that share fragment row fm (lane bits 0 and 3) each add a quarter of
# the group in index order, then combine as (q0 + q1) + (q2 + q3) with two
# shuffles.  The order is fixed, independent of the row count.
_ACCUM_AFFINE = r"""
    for (int t = 0; t < TMR; t++) {
      const int m0 = rb + t * 16 + fm;
      const int quarter = (((lane >> 3) & 1) << 1) | (lane & 1);
      float xs0 = 0.0f, xs1 = 0.0f;
      if (m0 < M) {
        const device XT* xp = (const device XT*)X + (int64_t)m0 * K + g * GS + quarter * (GS / 4);
        for (int i = 0; i < GS / 4; i++) xs0 += float(xp[i]);
      }
      if (m0 + 8 < M) {
        const device XT* xp = (const device XT*)X + (int64_t)(m0 + 8) * K + g * GS + quarter * (GS / 4);
        for (int i = 0; i < GS / 4; i++) xs1 += float(xp[i]);
      }
      xs0 += simd_shuffle_xor(xs0, ushort(1));
      xs1 += simd_shuffle_xor(xs1, ushort(1));
      xs0 += simd_shuffle_xor(xs0, ushort(8));
      xs1 += simd_shuffle_xor(xs1, ushort(8));
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

# 4-bit group-64 weights: each lane loads its own share of the group's
# integer codes straight from MLX's layout and hands them to the tensor op as
# a right-input cooperative tensor of half, in place of MPP reading a
# ``uint4b_format`` device tensor.  Codes become half exactly through
# ``(q | 0x6400) - 1024`` on nibble pairs, and the next group's codes are
# loaded while this group runs.  Same products, same op, same K order: the
# result is bitwise identical to the device-tensor read (GPU gate:
# register/device parity, 1-128 rows, bf16 and fp16), so the law is
# unchanged.  q4 g64 only, where it was measured: in-model Qwen3.8-27B
# 16-row forward -6% on an M5 Max (docs/experiments/LANE-REGISTER-B-2026-10-21.md).
#
# The fragment layout of matmul2d(16, NT, GS, B transposed) right inputs on
# M5: lane l holds k in {4 qi .. 4 qi + 3} + 16 kb (kb < GS / 16) of the
# columns nb + 8 j (j < 4), element kb * 16 + j * 4 + kk, where
# qi = (l & 1) | ((l >> 3) & 1) << 1 and nb = ((l >> 1) & 3) | ((l >> 4) & 1) << 2:
# one aligned ushort of four nibbles per (column, kb).
REGISTER_B = [True]
REGISTER_B_FORMATS = frozenset({(4, 64)})


def _register_sources(group_size: int) -> tuple[str, str]:
    """(declarations, per-group body) of the register-fed q4 path."""
    chunks = group_size // 16
    names = [f"bw{j}_{kb}" for j in range(4) for kb in range(chunks)]

    def load(gexpr: str, indent: str) -> str:
        return "".join(
            f"{indent}bw{j}_{kb} = bok{j} ? bwp{j}[({gexpr}) * (GS / 4) + {kb * 4}] : ushort(0);\n"
            for j in range(4) for kb in range(chunks))

    decl = [
        "  const int bqi = (lane & 1) | (((lane >> 3) & 1) << 1);     // k quarter of this lane's codes\n",
        "  const int bnb = ((lane >> 1) & 3) | (((lane >> 4) & 1) << 2);   // first of its four columns\n",
    ]
    for j in range(4):
        decl.append(f"  const bool bok{j} = n0 + bnb + {8 * j} < N;\n")
        decl.append(f"  const device ushort* bwp{j} = (const device ushort*)((const device uchar*)W\n"
                    f"      + (int64_t)(bok{j} ? n0 + bnb + {8 * j} : 0) * (K / 2)) + bqi;\n")
    decl.append("  ushort " + ", ".join(names) + ";\n")
    decl.append("  if (g_begin < g_end) {\n" + load("g_begin", "    ") + "  }\n")
    # ``cb``, not ``b``: the device-read path binds ``b`` to a tensor slice
    # and spells a bare ``decltype(b)`` (valid for a tensor); keeping the
    # cooperative operand distinct keeps the address-space guard exact.
    body = ["    auto cb = op.template get_right_input_cooperative_tensor<XT, half, float>();\n"]
    for j in range(4):
        for kb in range(0, chunks, 2):
            body.append(f"    {{ const uint v = uint(bw{j}_{kb}) | (uint(bw{j}_{kb + 1}) << 16);\n")
            for kk in range(4):
                body.append(
                    f"      const half2 h{kk} = as_type<half2>(((v >> {4 * kk}) & 0x000F000Fu) | 0x64006400u)"
                    " - half2(1024.0h);\n"
                    f"      cb[{kb * 16 + j * 4 + kk}] = h{kk}.x; cb[{(kb + 1) * 16 + j * 4 + kk}] = h{kk}.y;\n")
            body.append("    }\n")
    body.append("    if (g + 1 < g_end) {\n" + load("g + 1", "      ") + "    }\n")
    body.append("    auto P = op.template get_destination_cooperative_tensor<decltype(a), "
                "metal::remove_addrspace_t<decltype(cb)>, float>();\n")
    body.append("    op.run(a, cb, P);\n")
    return "".join(decl), "".join(body)


_RUN_DEVICE = (
    "    auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();\n"
    "    op.run(a, b, P);\n"
)


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
    scale_bias: Any = None      # (K/GS, N, 2) in the scales' dtype; None when unquantized or simd
    bias: Any = None            # optional additive bias (N,)
    backend: str = "mpp"        # "mpp" (M5 tensor units) or "simd" (M1-M4, see simd.py)
    scales: Any = None          # simd only: MLX's own (N, K/GS) scales and biases, shared
    biases: Any = None

    @property
    def scales_dtype(self):
        source = self.scale_bias if self.scale_bias is not None else self.scales
        return None if source is None else source.dtype

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


def prepare(module, backend_name: str | None = None) -> LaneWeights:
    """Prepare an ``nn.QuantizedLinear`` or ``nn.Linear`` for the lane matmul.

    ``backend_name`` defaults to this device's ``backend()``.  The simd
    backend reads MLX's scales and biases in place (no second copy) and does
    not cover unquantized weights.
    """
    from mlx import nn

    backend_name = backend_name or backend() or "mpp"
    if isinstance(module, nn.QuantizedLinear):
        weight, scales, biases = module["weight"], module["scales"], module.get("biases")
        bits, group_size = int(module.bits), int(module.group_size)
        mode = getattr(module, "mode", "affine")
        if biases is None:
            raise LaneUnsupported("affine lane matmul needs quantization biases")
        n = int(weight.shape[0])
        k = int(weight.shape[1]) * 32 // bits
        if backend_name == "simd":
            try:
                simd.check_geometry(bits=bits, group_size=group_size, mode=mode, n=n, k=k,
                                    weight_dtype=weight.dtype, scales_dtype=scales.dtype,
                                    biases_dtype=biases.dtype, weight_ndim=weight.ndim)
            except simd.SimdUnsupported as exc:
                raise LaneUnsupported(str(exc)) from None
            if scales.dtype != mx.bfloat16:
                # The simd kernels read bf16 activations only: an fp16/fp32
                # checkpoint would fall back to stock on every call while its
                # receipt claimed the lane law.
                raise LaneUnsupported("the simd kernels need a bf16 checkpoint (bf16 activations)")
            if tuple(scales.shape) != (n, k // group_size) or biases.shape != scales.shape:
                raise LaneUnsupported("scale/bias geometry does not match the packed weight")
            return LaneWeights(bits, group_size, n, k, simd.splits(n, k), weight, None,
                               module.get("bias"), "simd", scales, biases)
        check_geometry(bits=bits, group_size=group_size, mode=mode, n=n, k=k,
                       weight_dtype=weight.dtype, scales_dtype=scales.dtype)
        if tuple(scales.shape) != (n, k // group_size) or biases.shape != scales.shape:
            raise LaneUnsupported("scale/bias geometry does not match the packed weight")
        pairs = mx.contiguous(mx.stack([scales.T, biases.T], axis=-1).astype(scales.dtype))
        return LaneWeights(bits, group_size, n, k, split_k(n, k, group_size, bits),
                           weight, pairs, module.get("bias"))
    if isinstance(module, nn.Linear):
        if backend_name == "simd":
            raise LaneUnsupported("unquantized weights need the M5 tensor units")
        weight = module["weight"]
        if weight.ndim != 2:
            raise LaneUnsupported("linear weight must be rank 2")
        n, k = map(int, weight.shape)
        check_geometry(bits=UNQUANTIZED_BITS, group_size=64, mode="none", n=n, k=k,
                       weight_dtype=weight.dtype)
        return LaneWeights(UNQUANTIZED_BITS, 64, n, k, split_k(n, k, 64, UNQUANTIZED_BITS),
                           weight, None, module.get("bias"))
    raise LaneUnsupported(f"{type(module).__name__} is not a linear projection")


def main_source(bits: int, group_size: int, register_b: bool | None = None) -> str:
    """The main kernel body for one weight format (placeholders substituted).

    ``register_b`` (default ``REGISTER_B[0]``) feeds the codes of the formats
    in ``REGISTER_B_FORMATS`` to the tensor op from registers; False keeps
    MPP's device-tensor read.  Both give the same bits.
    """
    affine = bits != UNQUANTIZED_BITS
    register_b = REGISTER_B[0] if register_b is None else register_b
    run = _RUN_DEVICE
    if register_b and (bits, group_size) in REGISTER_B_FORMATS:
        (b_decl, b_group), after, run = _register_sources(group_size), "", ""
    elif bits in _NATIVE or not affine:
        b_decl, b_group, after = _B_DECL_NATIVE, _B_GROUP_NATIVE, ""
    else:
        b_decl, after = _B_DECL_UNPACK, _AFTER_UNPACK
        b_group = _unpack_group_source(bits, group_size)
    part = ("  threadgroup float part[(SK > 1 ? SK - 1 : 1) * NF * 8 * 32];\n"
            if b_decl is not _B_DECL_UNPACK else
            "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "  threadgroup float* part = (threadgroup float*)wbuf32;\n")
    parts = {
        "@@PART@@": part,
        "@@B_DECL@@": b_decl,
        "@@SB_DECL@@": _SB_DECL if affine else "",
        "@@SB_LOAD@@": _SB_LOAD if affine else "",
        "@@B_GROUP@@": b_group,
        "@@RUN@@": run,
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


def _named(base: str, source: str) -> str:
    # MLX caches compiled kernels by name, so the name carries the source hash.
    return f"mlx2_lane_{base}_{hashlib.sha256((_HEADER + source).encode()).hexdigest()[:16]}"


def _kernel(bits: int, group_size: int, xt: str, wt: str, st: str) -> Any:
    register_b = REGISTER_B[0] and (bits, group_size) in REGISTER_B_FORMATS
    key = (bits, group_size, xt, wt, st, register_b)
    if key not in _KERNELS:
        source = main_source(bits, group_size, register_b)
        wp = "uchar" if bits != UNQUANTIZED_BITS else wt
        wtype = _NATIVE.get(bits, wt if bits == UNQUANTIZED_BITS else "uchar")
        header = (_HEADER + f"typedef {xt} XT;\ntypedef {wtype} WT;\ntypedef {wp} WP;\n"
                  f"typedef {st} ST;\nconstexpr constant int BITS = {bits};\n")
        _KERNELS[key] = mx.fast.metal_kernel(
            name=_named(f"q{bits}g{group_size}_{xt}_{wt}_{st}", header + source),
            input_names=["X", "W", "SBt", "mdims"], output_names=["Y"],
            source=source, header=header, ensure_row_contiguous=True,
        )
    return _KERNELS[key]


def _mdims(m: int, mp: int):
    key = (m, mp)
    if key not in _MDIMS:
        _MDIMS[key] = mx.array(key, dtype=mx.int32)
    return _MDIMS[key]


def _row_tiling(m: int) -> tuple[int, int]:
    """(padded rows, rows a threadgroup) of one ``m``-row tensor-unit launch."""
    mp = 16 * ((m + 15) // 16)
    return mp, min(mp, ROW_BLOCK)


def weight_passes(lw: LaneWeights, rows: int) -> int:
    """Full reads of ``lw``'s weight that one ``rows``-row ``lane_matmul`` performs.

    Published kernel geometry for cost planning, not a timing: every
    threadgroup along the launch's row axis streams the whole (split-K
    sliced) weight once, so this is that axis's grid size.  A simd lane
    answers with its own tiling (``simd.weight_passes``).
    """
    if not 1 <= rows <= MAX_ROWS:
        raise LaneUnsupported(f"lane matmul takes 1-{MAX_ROWS} rows, got {rows}")
    if lw.backend == "simd":
        if lw.scales is None or lw.biases is None:
            raise LaneUnsupported("simd lane requires affine scales and biases")
        return simd.weight_passes(lw.bits, lw.group_size, lw.n, lw.k,
                                  lw.scales.dtype, lw.biases.dtype, rows)
    mp, block = _row_tiling(rows)
    return -(-mp // block)


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
    if lw.backend == "simd":
        try:
            y = simd.qmm(x2, lw.weight, lw.scales, lw.biases, lw.group_size, lw.bits)
        except simd.SimdUnsupported as exc:
            raise LaneUnsupported(str(exc)) from None
        y = y.reshape(*lead, lw.n)
        return y + lw.bias if lw.bias is not None else y
    mp, block = _row_tiling(m)
    mdims = _mdims(m, mp)
    if lw.bits == UNQUANTIZED_BITS:
        wt = _TYPES[lw.weight.dtype]
        st = "bfloat"
        sbt = mdims                      # unused by the plain accumulator
    else:
        wt = "uchar"
        st = _TYPES[lw.scale_bias.dtype]
        sbt = lw.scale_bias
    y = _kernel(lw.bits, lw.group_size, xt, wt, st)(
        inputs=[x2, lw.weight, sbt, mdims],
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
BACKENDS = ("mpp", "simd")
_FORCED: list[str | None] = [None]


def _tensor_units() -> bool:
    if not _M5:
        try:
            info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
            arch = re.match(r"applegpu_g(\d+)", str(info.get("architecture", "")))
            _M5.append(mx.metal.is_available() and (
                "M5" in str(info.get("device_name", "")) or (arch is not None and int(arch.group(1)) >= 17)))
        except Exception:  # noqa: BLE001 - absent Metal means unavailable
            _M5.append(False)
    return _M5[0]


def backend() -> str | None:
    """The lane backend for the default device.

    "mpp" on GPUs with Metal 4 tensor units (M5), "simd" on every other Apple
    GPU (M1-M4, see ``simd``), None off the GPU.  The two are different
    numerical laws; a route binds the one it installed.
    """
    if mx.default_device() != mx.gpu:
        return None
    if _FORCED[0] is not None:
        return _FORCED[0] if (_FORCED[0] == "simd" and simd.available()) or _tensor_units() else None
    if _tensor_units():
        return "mpp"
    return "simd" if simd.available() else None


def force_backend(name: str | None) -> None:
    """Pin the backend (gates and paired A/B; "simd" also runs on an M5), None for auto."""
    if name is not None and name not in BACKENDS:
        raise ValueError(f"lane backend must be one of {BACKENDS}")
    _FORCED[0] = name


def available() -> bool:
    """True when the default device can run a lane backend."""
    return backend() is not None
