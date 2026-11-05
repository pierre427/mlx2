# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Shinyoung Kang (derived Metal kernel bodies).
# Modified for mlx2; see provenance/integer-exact-microscaling.json and its license.
"""Model-neutral SC-MXFP4 dense and grouped INT8 prefill primitives.

Fixed immutable SC32 representation. Imports and host geometry validation never
compile or dispatch Metal. Dense calls allocate only one projection's unpack;
grouped calls unpack tiles in threadgroup memory and never retain INT8 experts.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import math

from .format import SC_FORMAT

DENSE_TILE = 128
GROUPED_TILE = 64
COLUMN_TILE = 64
MPP_OFFSET_LIMIT = 2**31
MAX_EXPERTS = 2048

_HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
typedef vec<char, 8> char8_t;
"""

_UNPACK = r"""
    const uint i = thread_position_in_grid.x;
    const uint word = codes[i];
    const uint e = i * 8, n = e / K, k = e % K;
    const uint blk = k / SC_BLOCK;
    const uint bc = (bcodes[n * (K / (SC_BLOCK * SC_PER_BYTE)) + blk / SC_PER_BYTE]
                     >> (SC_BC_BITS * (blk % SC_PER_BYTE))) & ((1u << SC_BC_BITS) - 1);
    const int mult = sc_mult(bc);
    char8_t v;
#pragma unroll
    for (uint j = 0; j < 8; ++j) {
        const uint c = (word >> (4 * j)) & 0xF;
        const int mag = sc_mag(c & 7) * mult;
        v[j] = char((c & 8) ? -mag : mag);
    }
    ((device char8_t *)wq)[i] = v;
"""

_ACT_QUANT = r"""
    const uint row = threadgroup_position_in_grid.x, lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    threadgroup float red[32];
    threadgroup float shared_amax;
    constexpr int J = K / (4 * NT);
    float v[J][4];
    float amax = 0;
    for (int j = 0; j < J; ++j)
        for (int i = 0; i < 4; ++i) {
            v[j][i] = row < M ? float(x[size_t(row) * K + 4 * (j * NT + lid) + i]) : 0.0f;
            amax = max(amax, abs(v[j][i]));
        }
    amax = simd_max(amax);
    if (lane == 0) red[sg] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        amax = simd_max(lane < NT / 32 ? red[lane] : 0.0f);
        if (lane == 0) shared_amax = amax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float s = shared_amax / 127.0f;
    const float qi = s > 0 ? 1.0f / s : 0.0f;
    for (int j = 0; j < J; ++j) {
        char4 q;
        for (int i = 0; i < 4; ++i) q[i] = char(clamp(rint(v[j][i] * qi), -127.0f, 127.0f));
        *(device char4 *)(xq + size_t(row) * K + 4 * (j * NT + lid)) = q;
    }
    if (lid == 0) xs[row] = s;
"""

_INT8_MM = r"""
    const uint num_m = threadgroups_per_grid.y, num_n = threadgroups_per_grid.x;
    const uint pid = threadgroup_position_in_grid.y * num_n + threadgroup_position_in_grid.x;
    const uint per_group = GM * num_n;
    const uint first_m = (pid / per_group) * GM;
    const uint gsz = min(num_m - first_m, uint(GM));
    const uint r = pid % per_group;
    const int m0 = (first_m + r % gsz) * BM, n0 = (r / gsz) * BN;
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tA((device int8_t *)xq, dextents<int32_t, 2>(K, M));
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tB((device int8_t *)wq, dextents<int32_t, 2>(K, N));
    tensor<device int32_t, dextents<int32_t, 2>, tensor_inline> tC((device int32_t *)acc, dextents<int32_t, 2>(N, M));
    constexpr auto desc = matmul2d_descriptor(BM, BN, dynamic_length_v<int>, false, true, false,
                                              matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto mA = tA.slice(0, m0);
    auto mB = tB.slice(0, n0);
    auto mC = tC.slice(n0, m0);
    op.run(mA, mB, mC);
"""

_SCALED_MM = r"""
    const uint num_m = threadgroups_per_grid.y, num_n = threadgroups_per_grid.x;
    const uint pid = threadgroup_position_in_grid.y * num_n + threadgroup_position_in_grid.x;
    const uint per_group = GM * num_n;
    const uint first_m = (pid / per_group) * GM;
    const uint gsz = min(num_m - first_m, uint(GM));
    const uint r = pid % per_group;
    const int m0 = (first_m + r % gsz) * BM, n0 = (r / gsz) * BN;
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tA((device int8_t *)xq, dextents<int32_t, 2>(K, M));
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tB((device int8_t *)wq, dextents<int32_t, 2>(K, N));
    constexpr auto desc = matmul2d_descriptor(BM, BN, dynamic_length_v<int>, false, true, false,
                                              matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto mA = tA.slice(0, m0);
    auto mB = tB.slice(0, n0);
    auto c = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), int32_t>();
    op.run(mA, mB, c);
#pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); i += 4) {
        if (!c.is_valid_element(i)) continue;
        auto idx = c.get_multidimensional_index(i);
        const uint n = n0 + idx[0], m = m0 + idx[1];
        const float4 a = float4(c[i], c[i + 1], c[i + 2], c[i + 3]) * xscale[m] * *(device const float4 *)(wscale + n);
        device T *o = out + size_t(m) * N + n;
        for (int j = 0; j < 4; ++j) o[j] = static_cast<T>(a[j]);
    }
"""

_GROUPED_MM = r"""
    // ---- which expert / M-tile this threadgroup computes
    const uint s = threadgroup_position_in_grid.y;
    const uint n0 = threadgroup_position_in_grid.x * BN;
    const uint tid = thread_position_in_threadgroup.x;
    if (s >= uint(tstart[E])) return;
    uint lo = 0, hi = E;  // largest e with tstart[e] <= s
    while (hi - lo > 1) { const uint mid = (lo + hi) / 2; if (uint(tstart[mid]) <= s) lo = mid; else hi = mid; }
    const uint e = lo;
    const uint row0 = off[e] + (s - tstart[e]) * BM, row_end = off[e + 1];

    threadgroup int8_t sB[BN * BK];
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tA((device int8_t *)xq, dextents<int32_t, 2>(K, RP));
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tB(sB, dextents<int32_t, 2>(BK, BN));
    constexpr auto desc = matmul2d_descriptor(BM, BN, BK, false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto a0 = tA.template slice<BK, BM>(0, row0);
    auto c = op.template get_destination_cooperative_tensor<decltype(a0), decltype(tB), int32_t>();
#pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); ++i)
        if (c.is_valid_element(i)) c[i] = 0;

    // ---- K loop: unpack expert e's BN x BK weight tile into threadgroup memory, then accumulate
    device const uint *cw = codes + (size_t(e) * N + n0) * (K / 8);
    device const uchar *bw = bcodes + (size_t(e) * N + n0) * (K / (SC_BLOCK * SC_PER_BYTE));
    constexpr uint WPR = BK / 8;                 // code words per tile row
    constexpr uint WPT = BN * WPR / (32 * SG);   // words per thread
    for (uint k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
        for (uint j = 0; j < WPT; ++j) {
            const uint w = tid + j * 32 * SG, rr = w / WPR, cc = w % WPR;
            const uint word = cw[size_t(rr) * (K / 8) + k0 / 8 + cc];
            const uint blk = (k0 + 8 * cc) / SC_BLOCK;
            const uint bc = (bw[size_t(rr) * (K / (SC_BLOCK * SC_PER_BYTE)) + blk / SC_PER_BYTE]
                             >> (SC_BC_BITS * (blk % SC_PER_BYTE))) & ((1u << SC_BC_BITS) - 1);
            const int mult = sc_mult(bc);
            char8_t v;
#pragma unroll
            for (uint q = 0; q < 8; ++q) {
                const uint cd = (word >> (4 * q)) & 0xF;
                const int mag = sc_mag(cd & 7) * mult;
                v[q] = char((cd & 8) ? -mag : mag);
            }
            *(threadgroup char8_t *)(sB + rr * BK + 8 * cc) = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto a = tA.template slice<BK, BM>(k0, row0);
        op.run(a, tB, c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- epilogue (rows past this expert's segment belong to the next expert: not stored)
    device const float *wse = wscale + size_t(e) * N;
#pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); i += 4) {
        if (!c.is_valid_element(i)) continue;
        auto idx = c.get_multidimensional_index(i);
        const uint n = n0 + idx[0], m = row0 + idx[1];
        if (m >= row_end) continue;
        if (!SCALED) {
            for (int j = 0; j < 4; ++j) out[size_t(m) * N + n + j] = c[i + j];
        } else {
            const float4 v = float4(c[i], c[i + 1], c[i + 2], c[i + 3]) * xscale[m] * *(device const float4 *)(wse + n);
            for (int j = 0; j < 4; ++j) out[size_t(m) * N + n + j] = static_cast<T>(v[j]);
        }
    }
"""

# One binary search per expert avoids materializing [assignments, experts].
_SEGMENTS = r"""
    const uint e = thread_position_in_grid.x;
    if (e >= E) return;
    uint lo = 0, hi = R;
    while (lo < hi) {
        uint mid = (lo + hi) / 2;
        if (int(indices[mid]) < int(e)) lo = mid + 1; else hi = mid;
    }
    const uint start = lo;
    hi = R;
    while (lo < hi) {
        uint mid = (lo + hi) / 2;
        if (int(indices[mid]) <= int(e)) lo = mid + 1; else hi = mid;
    }
    counts[e] = int(lo - start);
"""

_SOURCES = {
    "unpack": (("codes", "bcodes"), ("wq",), _UNPACK),
    "quantize": (("x",), ("xq", "xs"), _ACT_QUANT),
    "int8_mm": (("xq", "wq"), ("acc",), _INT8_MM),
    "scaled_mm": (("xq", "wq", "xscale", "wscale"), ("out",), _SCALED_MM),
    "grouped_mm": (("xq", "codes", "bcodes", "wscale", "xscale", "off", "tstart"), ("out",), _GROUPED_MM),
    "segments": (("indices",), ("counts",), _SEGMENTS),
}
KERNEL_REVISION = hashlib.sha256(
    (SC_FORMAT.revision + _HEADER + "".join(v[2] for v in _SOURCES.values())).encode()
).hexdigest()


def _shape(shape, *, rank=None, name="array"):
    shape = tuple(shape)
    if (rank is not None and len(shape) != rank) or not shape or any(type(d) is not int or d <= 0 for d in shape):
        raise ValueError(f"{name} must have positive dimensions" + (f" and rank {rank}" if rank else ""))
    return shape


def _weight_geometry(codes_shape, bcodes_shape, row_scales_shape, *, grouped=False):
    codes = _shape(codes_shape, rank=3 if grouped else 2, name="codes")
    n, k = codes[-2], codes[-1] * 8
    if n % COLUMN_TILE or k % 128 or k > SC_FORMAT.max_reduction:
        raise ValueError("SC prefill requires N divisible by 64 and K divisible by 128 within the INT32 bound")
    if tuple(bcodes_shape) != (*codes[:-1], k // 128) or tuple(row_scales_shape) != codes[:-1]:
        raise ValueError("SC metadata shapes do not match packed weights")
    if grouped:
        if codes[0] > MAX_EXPERTS:
            raise ValueError(f"SC grouped prefill supports at most {MAX_EXPERTS} experts")
    elif n * k >= MPP_OFFSET_LIMIT:
        raise ValueError("SC dense weight tensor exceeds MPP's signed byte-offset range")
    return n, k


def validate_dense_geometry(x_shape, codes_shape, bcodes_shape, row_scales_shape):
    """Return (flattened rows, N, K), validating only host metadata."""
    x = _shape(x_shape, name="x")
    n, k = _weight_geometry(codes_shape, bcodes_shape, row_scales_shape)
    if x[-1] != k:
        raise ValueError("activation width differs from SC weight reduction width")
    rows = math.prod(x[:-1])
    padded = (rows + DENSE_TILE - 1) // DENSE_TILE * DENSE_TILE
    if max(padded * k, padded * n * 4) >= MPP_OFFSET_LIMIT:
        raise ValueError("SC dense activation/output tensor exceeds MPP's signed byte-offset range")
    return rows, n, k


def _broadcast_shapes(left, right):
    result = []
    for i in range(1, max(len(left), len(right)) + 1):
        a = left[-i] if i <= len(left) else 1
        b = right[-i] if i <= len(right) else 1
        if a != b and a != 1 and b != 1:
            raise ValueError("activation batches and expert indices cannot broadcast")
        result.append(max(a, b))
    return tuple(reversed(result))


def validate_grouped_geometry(x_shape, indices_shape, codes_shape, bcodes_shape, row_scales_shape):
    """Return (broadcast batch shape, E, assignments, N, K), with no device work.

    As in SwitchLinear/gather_qmm, x is [..., 1, K] and indices selects
    weight tables over the broadcast batch axes. This admits both repeated
    input rows for gate/up and a distinct input row per down assignment.
    """
    x = _shape(x_shape, name="x")
    indices = _shape(indices_shape, name="indices")
    n, k = _weight_geometry(codes_shape, bcodes_shape, row_scales_shape, grouped=True)
    if len(x) < 2 or x[-2:] != (1, k):
        raise ValueError("SC grouped prefill requires activation shape [...,1,K]")
    batch = _broadcast_shapes(x[:-2], indices)
    rows = math.prod(batch)
    # Tile unpack addresses weights with size_t; only the MPP input tensor
    # uses signed 32-bit byte offsets. Reserve a complete final tile.
    if max((rows + GROUPED_TILE) * k, rows * n * 4) >= MPP_OFFSET_LIMIT:
        raise ValueError("SC grouped activation/output tensor exceeds MPP's signed byte-offset range")
    return batch, codes_shape[0], rows, n, k


@lru_cache(maxsize=128)
def _compiled(name, dtype_name, constants):
    import mlx.core as mx

    ins, outs, source = _SOURCES[name]
    prelude = "".join(f"constexpr int {key} = {int(value)};\n" for key, value in constants)
    if dtype_name:
        prelude += f"using T = {dtype_name};\n"
    tag = hashlib.sha256((name + dtype_name + repr(constants) + KERNEL_REVISION).encode()).hexdigest()[:16]
    return mx.fast.metal_kernel(
        name="mlx2_sc_" + name + "_" + tag,
        input_names=list(ins), output_names=list(outs),
        header=_HEADER + SC_FORMAT.metal_header(), source=prelude + source,
        ensure_row_contiguous=True,
    )


def _dispatch(name, inputs, constants, shapes, dtypes, grid, threadgroup, dtype_name=""):
    kernel = _compiled(name, dtype_name, tuple(constants.items()))
    return kernel(inputs=inputs, grid=grid, threadgroup=threadgroup,
                  output_shapes=shapes, output_dtypes=dtypes)


def _packed_dtypes(codes, bcodes, row_scales=None):
    import mlx.core as mx

    if codes.dtype != mx.uint32 or bcodes.dtype != mx.uint8:
        raise ValueError("SC packed weights require uint32 codes and uint8 block shifts")
    if row_scales is not None and row_scales.dtype != mx.float32:
        raise ValueError("SC row scales require float32")


def quantize_activations(x, padded_rows=None):
    """Per-row symmetric INT8, rint/F32 scale, with zero-filled padding."""
    import mlx.core as mx

    if x.ndim != 2 or x.dtype not in (mx.bfloat16, mx.float16):
        raise ValueError("SC activation quantization requires a BF16/FP16 matrix")
    m, k = _shape(x.shape, rank=2, name="x")
    if k % 128 or k > SC_FORMAT.max_reduction:
        raise ValueError("SC activation width must be divisible by 128 within the INT32 bound")
    mp = (m + DENSE_TILE - 1) // DENSE_TILE * DENSE_TILE if padded_rows is None else padded_rows
    if type(mp) is not int or mp < m or mp * k >= MPP_OFFSET_LIMIT:
        raise ValueError("invalid or excessive SC activation padding")
    nt = min(256, k // 4)
    while nt % 32 or (k // 4) % nt:
        nt -= 1
    return _dispatch("quantize", [x], {"K": k, "M": m, "NT": nt},
                     [(mp, k), (mp,)], [mx.int8, mx.float32], (mp * nt, 1, 1), (nt, 1, 1))


def unpack_i8(codes, bcodes):
    """One dense matrix's exact INT8 embedding; no persistent expert copy."""
    import mlx.core as mx

    _packed_dtypes(codes, bcodes)
    n, k = _weight_geometry(codes.shape, bcodes.shape, codes.shape[:-1])
    return _dispatch("unpack", [codes, bcodes], {"K": k}, [(n, k)], [mx.int8],
                     (n * k // 8, 1, 1), (256, 1, 1))[0]


def int8_mm(xq, wq, xscale=None, row_scales=None, *, output_dtype=None):
    """INT8 matrix product, or BF16 product with the in-register row scales.

    xq rows must already be padded to DENSE_TILE. The INT32 mode is for
    bounded reference probes; operands must follow the SC |weight|<=96,
    |activation|<=127 contract established by the checked codec/quantizer.
    """
    import mlx.core as mx

    m, k = _shape(xq.shape, rank=2, name="xq")
    n, wk = _shape(wq.shape, rank=2, name="wq")
    if xq.dtype != mx.int8 or wq.dtype != mx.int8 or wk != k:
        raise ValueError("SC INT8 GEMM operands must be matching INT8 matrices")
    validate_dense_geometry((m, k), (n, k // 8), (n, k // 128), (n,))
    if m % DENSE_TILE:
        raise ValueError("SC INT8 GEMM rows must be padded to 128")
    constants = {"M": m, "N": n, "K": k, "BM": DENSE_TILE, "BN": COLUMN_TILE, "SG": 4, "GM": 8}
    grid = (n // COLUMN_TILE * 128, m // DENSE_TILE, 1)
    if xscale is None and row_scales is None:
        return _dispatch("int8_mm", [xq, wq], constants, [(m, n)], [mx.int32], grid, (128, 1, 1))[0]
    if xscale is None or row_scales is None or xscale.shape != (m,) or row_scales.shape != (n,):
        raise ValueError("SC scaling requires one activation and one weight scale per row")
    if xscale.dtype != mx.float32 or row_scales.dtype != mx.float32:
        raise ValueError("SC GEMM scales must have float32 dtype")
    output_dtype = mx.bfloat16 if output_dtype is None else output_dtype
    if output_dtype not in (mx.bfloat16, mx.float16):
        raise ValueError("SC GEMM output dtype must be BF16 or FP16")
    metal_type = "bfloat16_t" if output_dtype == mx.bfloat16 else "half"
    return _dispatch("scaled_mm", [xq, wq, xscale, row_scales], constants,
                     [(m, n)], [output_dtype], grid, (128, 1, 1), metal_type)[0]


def dense(x, codes, bcodes, row_scales):
    """BF16/FP16 x[...,K] times one SC matrix; INT8 activation law is approximate."""
    import mlx.core as mx

    rows, n, k = validate_dense_geometry(x.shape, codes.shape, bcodes.shape, row_scales.shape)
    _packed_dtypes(codes, bcodes, row_scales)
    if x.dtype not in (mx.bfloat16, mx.float16):
        raise ValueError("SC dense prefill requires BF16/FP16 activations")
    xq, xs = quantize_activations(x.reshape(rows, k))
    wq = unpack_i8(codes, bcodes)
    out = int8_mm(xq, wq, xs, row_scales, output_dtype=x.dtype)
    return out[:rows].reshape(*x.shape[:-1], n)


def segments(sorted_indices, n_experts):
    """GPU prefix descriptors for trusted sorted expert indices."""
    import mlx.core as mx

    if sorted_indices.ndim != 1 or sorted_indices.dtype not in (mx.uint32, mx.int32):
        raise ValueError("SC segment descriptors require flat int32/uint32 indices")
    if type(n_experts) is not int or not 0 < n_experts <= MAX_EXPERTS or sorted_indices.size <= 0:
        raise ValueError("invalid SC expert/assignment count")
    r = sorted_indices.size
    counts = _dispatch("segments", [sorted_indices], {"E": n_experts, "R": r},
                       [(n_experts,)], [mx.int32], (n_experts, 1, 1), (min(256, n_experts), 1, 1))[0]
    zero = mx.zeros((1,), mx.int32)
    off = mx.concatenate([zero, mx.cumsum(counts)])
    tiles = (counts + GROUPED_TILE - 1) // GROUPED_TILE
    starts = mx.concatenate([zero, mx.cumsum(tiles)])
    return off, starts, (r + GROUPED_TILE - 1) // GROUPED_TILE + n_experts


def grouped_int8_mm(xq, codes, bcodes, row_scales, off, starts, max_tiles, rows, xscale=None, *, output_dtype=None):
    """Low-level grouped INT32/BF16 product over already sorted rows.

    Descriptor arrays must come from segments() and padded activations from
    quantize_activations(). Only model-internal validated expert IDs enter
    this contract; the public grouped() handles arbitrary ordering safely.
    """
    import mlx.core as mx

    _packed_dtypes(codes, bcodes, row_scales)
    n, k = _weight_geometry(codes.shape, bcodes.shape, row_scales.shape, grouped=True)
    e = codes.shape[0]
    if xq.ndim != 2 or xq.dtype != mx.int8 or xq.shape[1] != k or xq.shape[0] < rows + GROUPED_TILE:
        raise ValueError("SC grouped INT8 activations require a full padding tile")
    if type(rows) is not int or rows <= 0 or max(xq.shape[0] * k, rows * n * 4) >= MPP_OFFSET_LIMIT:
        raise ValueError("invalid SC grouped row/offset range")
    if off.shape != (e + 1,) or starts.shape != (e + 1,) or off.dtype != mx.int32 or starts.dtype != mx.int32:
        raise ValueError("SC grouped segment descriptors must be int32 [E+1]")
    if type(max_tiles) is not int or max_tiles <= 0:
        raise ValueError("invalid SC grouped tile bound")
    scaled = xscale is not None
    if scaled and (xscale.shape != (rows,) or xscale.dtype != mx.float32):
        raise ValueError("SC grouped activation scales require float32 [rows]")
    if not scaled:
        xscale = mx.zeros((1,), mx.float32)
    output_dtype = mx.bfloat16 if output_dtype is None else output_dtype
    if output_dtype not in (mx.bfloat16, mx.float16):
        raise ValueError("SC grouped output dtype must be BF16 or FP16")
    metal_type = "bfloat16_t" if output_dtype == mx.bfloat16 else "half"
    const = {"K": k, "R": rows, "RP": xq.shape[0], "N": n, "E": e,
             "BM": GROUPED_TILE, "BN": COLUMN_TILE, "BK": 128, "SG": 4, "SCALED": int(scaled)}
    return _dispatch("grouped_mm", [xq, codes, bcodes, row_scales, xscale, off, starts], const,
                     [(rows, n)], [output_dtype if scaled else mx.int32],
                     (n // COLUMN_TILE * 128, max_tiles, 1), (128, 1, 1), metal_type if scaled else "int")[0]


def grouped(x, indices, codes, bcodes, row_scales, sorted_indices=False):
    """BF16/FP16 switched projection with the gather_qmm batch/broadcast contract.

    Invalid expert IDs are clipped before memory addressing and produce NaN
    for their assignments, never an out-of-bounds read or a different expert's
    apparently valid output. Values originate in the adapter's model router.
    """
    import mlx.core as mx

    batch, e, rows, n, k = validate_grouped_geometry(
        x.shape, indices.shape, codes.shape, bcodes.shape, row_scales.shape)
    _packed_dtypes(codes, bcodes, row_scales)
    if x.dtype not in (mx.bfloat16, mx.float16) or indices.dtype not in (mx.int32, mx.uint32):
        raise ValueError("SC grouped prefill requires BF16/FP16 x and int32/uint32 expert IDs")
    if type(sorted_indices) is not bool:
        raise ValueError("sorted_indices must be a boolean")
    flat = mx.broadcast_to(indices, batch).reshape(rows)
    valid = (flat >= 0) & (flat < e)
    safe = mx.clip(flat, 0, e - 1).astype(mx.int32)
    order = None if sorted_indices else mx.argsort(safe)
    expert_sorted = safe if order is None else safe[order]
    # Gate/up inputs carry a singleton top-k axis. Quantize each source
    # token once, then broadcast its codes/scales; down inputs already have
    # one row per assignment. This avoids repeated dynamic row reductions.
    source_batch = x.shape[:-2]
    source_rows = math.prod(source_batch)
    aq, asc = quantize_activations(x.reshape(source_rows, k), padded_rows=source_rows)
    xq = mx.broadcast_to(aq.reshape(*source_batch, k), (*batch, k)).reshape(rows, k)
    xs = mx.broadcast_to(asc.reshape(source_batch), batch).reshape(rows)
    if order is not None:
        xq, xs = xq[order], xs[order]
    xq = mx.concatenate([xq, mx.zeros((GROUPED_TILE, k), mx.int8)])
    off, starts, max_tiles = segments(expert_sorted, e)
    out = grouped_int8_mm(xq, codes, bcodes, row_scales, off, starts, max_tiles, rows, xs, output_dtype=x.dtype)
    if order is not None:
        out = out[mx.argsort(order)]
    out = mx.where(valid[:, None], out, mx.array(float("nan"), x.dtype))
    return out.reshape(*batch, 1, n)
