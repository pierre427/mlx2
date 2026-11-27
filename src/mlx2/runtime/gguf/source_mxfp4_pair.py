# SPDX-License-Identifier: MIT
"""Explicit Source MXFP4 pair; bounded component parity, unqualified profile.

Reads the pinned, installed MIT MLX helpers (audited 0.32.4 and 0.32.5
builds); no upstream repository access. Original wrapper preserves the stock virtual grid and calls unchanged
helper bodies. See docs/research/2026-10-25-ds41-original-pair-design.md.
Nothing runs on import. Dense weights and ordinary QMM stay unchanged.
"""
from functools import lru_cache
import hashlib
from pathlib import Path


def installed_root():
    import mlx.core as mx
    return Path(mx.__file__).resolve().parent


HASHES = {
    'fp_quantized.h': 'fb4bf0919642717c5e36cfa7be0628e651844586cae1b405c27b2057a17dd1c8',
    'fp4.h': 'e3159945d7e5ae24f7a1ca9c4875b2a5d15cfb6952405df24a0df1bb2622a93a',
    'fp8.h': '9a9d77a6295f66b4ada4311c3b855b6ecb7a2b38f28623109e3c61d2e17e81ae',
}
LIBRARY_SHA256 = 'd8c9393d584e781393c3effccfccd8ba85ad0c70264edaf9f897dca9875375c6'
# HASHES/LIBRARY_SHA256 above are the original 0.32.4 pins; generation uses
# installed_pins(). Each build pins libmlx.dylib, the three headers and the
# 0-based fp_quantized.h line spans that are transcribed: ``shared`` (fp4/fp8
# decode, scale, load and qdot laws), ``fast`` (fp_qmv_fast_impl), ``qmv``
# (fp_qmv_impl, used by source_down_fusion) and ``wide`` (fp_qmv_wide_impl).
# 0.32.5 re-audit (2026-11-27, v0.32.4..v0.32.5): fp4.h is byte-identical.
# fp_quantized.h inserts Fp4HalfLoader/use_fp4_half_loader (89 lines, not
# transcribed) before fp_qmv_quad_impl; ``shared``, ``qmv`` and ``wide`` are
# byte-identical at the shifted spans, and ``fast`` differs only in moving
# ``has_global_scale`` to the 4th template parameter before ``partial_rows``.
# The helper spelling fp_qmv_fast_impl<float,32,4,false,false,4> sets both to
# false, so the instantiation is unchanged. fp8.h changes only the NaN
# encodings: an e4m3 encode of NaN now yields 0x7F instead of saturating
# (never called here), and an e8m0 scale byte 0xFF now decodes to NaN instead
# of +inf (the stock kernel reads the same installed fp8.h, so parity with
# stock holds; finite scales are unchanged). quantized.cpp keeps the stock
# float32 route for this geometry (K 5120, N 2304, M 1..12 on gen-17 M5 Max,
# vector limit 13): the new qmv_mma and NAX split-K branches require
# tf32 or a non-float32 input, and qmv/qmv_wide keep their grids and fp tile
# cap of 5. Bitwise parity to the stock dispatch stays a GPU-gate claim.
PINS_BY_VERSION = {
    '0.32.4': dict(library=LIBRARY_SHA256, headers=HASHES,
                   spans=dict(shared=(19, 147), fast=(332, 421), qmv=(421, 578), wide=(582, 697))),
    '0.32.5': dict(library='0b8158bf29ecac60049f9163dda9537aab10d6fd8aa3d7614c646c3e6f810750',
                   headers={
                       'fp_quantized.h': '553777467b49cb1f1b498cf24a5870ea57b65cf8b68536d17dd64fe56583de45',
                       'fp4.h': 'e3159945d7e5ae24f7a1ca9c4875b2a5d15cfb6952405df24a0df1bb2622a93a',
                       'fp8.h': '0d212a9fa1d371bd85c8d7cf8159dbe0e702a4bbdf0a334f7fe3032c55743a1c',
                   },
                   spans=dict(shared=(19, 147), fast=(421, 510), qmv=(510, 667), wide=(671, 786))),
}


def installed_pins():
    """Audited pins for the installed MLX build; raises on an unaudited one."""
    import importlib.metadata
    pinned = PINS_BY_VERSION.get(importlib.metadata.version('mlx'))
    if pinned is None:
        raise RuntimeError('installed stock dispatch changed; re-audit before compiling')
    return pinned


@lru_cache(maxsize=1)
def _verify_library():
    pinned = installed_pins()
    path = installed_root()/'lib/libmlx.dylib'
    if hashlib.sha256(path.read_bytes()).hexdigest() != pinned['library']:
        raise RuntimeError('installed stock dispatch changed; re-audit before compiling')
    return pinned


def installed_texts():
    """Verified installed header texts and the build's pinned line spans."""
    pinned = _verify_library()
    root = installed_root()/'include/mlx/backend/metal/kernels'
    text = {}
    for name, expected in pinned['headers'].items():
        raw = (root/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise RuntimeError('installed stock helper changed; re-audit before compiling')
        text[name] = raw.decode()
    return text, pinned['spans']


def descriptor(rows, *, verified_m5_vector_band=False):
    """CPU-only descriptor. Wider row counts require a verified stock route."""
    if type(rows) is not int or not (1 <= rows <= (12 if verified_m5_vector_band else 3)):
        raise ValueError('unverified Source stock QMV row band; use ordinary QMM')
    if rows == 1:
        return dict(rows=rows, helper='fp_qmv_fast_impl<float,32,4,false,false,4>',
                    vecs=1, output_rows_per_group=8, grid=(288*64,1,2), threadgroup=(64,1,1))
    vector_groups = (rows+4)//5
    vecs = (rows+vector_groups-1)//vector_groups
    return dict(rows=rows, helper=f'fp_qmv_wide_impl<float,32,4,{vecs},16>',
                vecs=vecs, output_rows_per_group=4,
                grid=(576*64,(rows+vecs-1)//vecs,2), threadgroup=(64,1,1))


def _source_header(rows):
    text, spans = installed_texts()
    lines = text['fp_quantized.h'].splitlines(keepends=True)
    # Pinned full-file hashes make these exact source spans unambiguous.
    pieces = ''.join(''.join(lines[a:b]) for a, b in (spans['shared'], spans['fast'], spans['wide']))
    assert 'fp_qmv_fast_impl' in pieces and 'fp_qmv_wide_impl' in pieces
    # The generated helper text is derived MLX code and retains its attribution.
    return '''// Derived helpers: Copyright © 2025-2026 Apple Inc. MIT license.
// License retained in provenance/source-mxfp4-installed-MLX-LICENSE.txt.
#include <metal_stdlib>
#include <metal_simdgroup>
namespace mlx2_source_stock_qmv {
using namespace metal;
typedef half float16_t;
typedef bfloat bfloat16_t;
#define MLX_MTL_CONST static constant constexpr const
MLX_MTL_CONST int SIMD_SIZE = 32;
MLX_MTL_CONST int QUAD_SIZE = 4;
''' + text['fp4.h'].replace('#pragma once','') + text['fp8.h'].replace('#pragma once','') + pieces + f'''
constant int K = 5120;
constant int N = 2304;
constant int M = {rows};
}} // namespace mlx2_source_stock_qmv
'''


def kernel_spec(rows, *, verified_m5_vector_band=False):
    """Generate metadata; locate installed MLX without compiling or launching GPU work."""
    d = descriptor(rows, verified_m5_vector_band=verified_m5_vector_band)
    args = ('w, s, nullptr, x, dst, mlx2_source_stock_qmv::K, mlx2_source_stock_qmv::N, '
            'virtual_tid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup')
    if rows > 1:
        args = ('w, s, x, dst, mlx2_source_stock_qmv::K, mlx2_source_stock_qmv::N, '
                'mlx2_source_stock_qmv::M, virtual_tid, '
                'simdgroup_index_in_threadgroup, thread_index_in_simdgroup')
    source = '''
const bool up = threadgroup_position_in_grid.z != 0;
const device uint32_t* w = up ? up_q : gate_q;
const device uint8_t* s = up ? up_s : gate_s;
device float* dst = up ? up_out : gate_out;
const uint3 virtual_tid = uint3(threadgroup_position_in_grid.y, threadgroup_position_in_grid.x, 0);
''' + f'mlx2_source_stock_qmv::{d["helper"]}({args});\n'
    return dict(d, header=_source_header(rows), source=source)


@lru_cache(maxsize=12)
def _kernel(rows, verified_m5_vector_band):
    import mlx.core as mx
    spec = kernel_spec(rows, verified_m5_vector_band=verified_m5_vector_band)
    return mx.fast.metal_kernel(
        name=f'mlx2_source_stock_mxfp4_pair_m{rows}',
        input_names=['x','gate_q','gate_s','up_q','up_s'],
        output_names=['gate_out','up_out'],
        header=spec['header'], source=spec['source'], ensure_row_contiguous=True,
    )


def paired_matmul(x, gate_q, gate_s, up_q, up_s, *, verified_m5_vector_band=False):
    """Pair original projections without moving the provider BF16 boundaries."""
    import mlx.core as mx
    if x.ndim != 2 or x.shape[1] != 5120 or x.dtype != mx.float32:
        raise ValueError('Source pair requires already-quantized F32 rows of width 5120')
    for q,s in ((gate_q,gate_s),(up_q,up_s)):
        if q.shape != (2304,640) or q.dtype != mx.uint32 or s.shape != (2304,160) or s.dtype != mx.uint8:
            raise ValueError('Source pair original MXFP4 geometry/dtype mismatch')
    spec = descriptor(x.shape[0], verified_m5_vector_band=verified_m5_vector_band)
    return tuple(_kernel(x.shape[0],verified_m5_vector_band)(
        inputs=[x,gate_q,gate_s,up_q,up_s], grid=spec['grid'], threadgroup=spec['threadgroup'],
        output_shapes=[(x.shape[0],2304)]*2, output_dtypes=[mx.float32]*2))


@lru_cache(maxsize=1)
def _device_supported():
    import mlx.core as mx
    return mx.device_info().get("device_name") == "Apple M5 Max"


def supports(x, gate_q, gate_s, up_q, up_s):
    import mlx.core as mx
    return (x.ndim == 2 and 1 <= x.shape[0] <= 12 and x.shape[1] == 5120
            and x.dtype == mx.float32
            and all(q.shape == (2304,640) and q.dtype == mx.uint32
                    and s.shape == (2304,160) and s.dtype == mx.uint8
                    for q,s in ((gate_q,gate_s),(up_q,up_s)))
            and _device_supported())


def supports_mapped(x, rows, gate_q, gate_s, up_q, up_s):
    import mlx.core as mx
    return (x.ndim==2 and x.shape[0]>0 and x.shape[1]==5120 and x.dtype==mx.float32
            and rows.ndim==1 and rows.dtype==mx.uint32 and 1<=rows.size<=12
            and all(q.shape==(2304,640) and q.dtype==mx.uint32
                    and s.shape==(2304,160) and s.dtype==mx.uint8
                    for q,s in ((gate_q,gate_s),(up_q,up_s))) and _device_supported())


def mapped_kernel_spec(rows):
    spec=kernel_spec(rows,verified_m5_vector_band=True)
    header,source=spec['header'],spec['source']
    if rows>1:
        begin=header.index('METAL_FUNC void fp_qmv_wide_impl(')
        tail=header[begin:]
        needle='    const device T* x,\n    device T* y,'
        assert tail.count(needle)==1
        tail=tail.replace(needle,'    const device T* x,\n    RowMap row_map,\n    device T* y,')
        needle='x + min(vec0 + v, M - 1) * in_vec_size'
        assert tail.count(needle)==1
        tail=tail.replace(needle,'x + row_map[min(vec0 + v, M - 1)] * in_vec_size')
        header=header[:begin]+tail
        needle='template <typename T, int group_size, int bits, int vecs_per_tg, int k_lanes>\nMETAL_FUNC void fp_qmv_wide_impl('
        assert header.count(needle)==1
        header=header.replace(needle,'template <typename T, int group_size, int bits, int vecs_per_tg, int k_lanes, typename RowMap>\nMETAL_FUNC void fp_qmv_wide_impl(')
        needle='w, s, x, dst,'
        assert source.count(needle)==1
        source=source.replace(needle,'w, s, x, rows, dst,')
    selected='token_x + rows[0]*5120' if rows==1 else 'token_x'
    return dict(spec,header=header,source=f'const device float* x={selected};\n'+source)


@lru_cache(maxsize=12)
def _mapped_kernel(rows):
    import mlx.core as mx
    spec=mapped_kernel_spec(rows)
    return mx.fast.metal_kernel(name=f'mlx2_source_stock_mxfp4_mapped_pair_m{rows}',
        input_names=['token_x','rows','gate_q','gate_s','up_q','up_s'],
        output_names=['gate_out','up_out'],header=spec['header'],source=spec['source'],
        ensure_row_contiguous=True)


def paired_matmul_mapped(x, rows, gate_q, gate_s, up_q, up_s):
    import mlx.core as mx
    if not supports_mapped(x,rows,gate_q,gate_s,up_q,up_s):
        raise ValueError('Source original mapped pair unsupported geometry/device')
    spec=descriptor(rows.size,verified_m5_vector_band=True)
    return tuple(_mapped_kernel(rows.size)(inputs=[x,rows,gate_q,gate_s,up_q,up_s],
        grid=spec['grid'],threadgroup=spec['threadgroup'],
        output_shapes=[(rows.size,2304)]*2,output_dtypes=[mx.float32]*2))
