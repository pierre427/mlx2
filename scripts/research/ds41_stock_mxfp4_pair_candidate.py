# SPDX-License-Identifier: MIT
"""Research-only paired Source MXFP4 launcher; NOT compiled or GPU validated.

Reads the pinned, installed MIT MLX 0.32.4 helpers; no upstream repository
access. Original wrapper preserves the stock virtual grid and calls unchanged
helper bodies. See docs/research/2026-10-25-ds41-original-pair-design.md.
Nothing runs on import. Parent owns any later Metal compilation/GPU oracle.
"""
from functools import lru_cache
import hashlib
from pathlib import Path

INSTALLED = Path('~/Desktop/mlx2/.venv/lib/python3.12/site-packages/mlx')
HASHES = {
    'fp_quantized.h': 'fb4bf0919642717c5e36cfa7be0628e651844586cae1b405c27b2057a17dd1c8',
    'fp4.h': 'e3159945d7e5ae24f7a1ca9c4875b2a5d15cfb6952405df24a0df1bb2622a93a',
    'fp8.h': '9a9d77a6295f66b4ada4311c3b855b6ecb7a2b38f28623109e3c61d2e17e81ae',
}


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
    root = INSTALLED/'include/mlx/backend/metal/kernels'
    text = {}
    for name, expected in HASHES.items():
        raw = (root/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise RuntimeError('installed stock helper changed; re-audit before compiling')
        text[name] = raw.decode()
    lines = text['fp_quantized.h'].splitlines(keepends=True)
    # Pinned full-file hashes make these exact source spans unambiguous.
    pieces = ''.join(lines[19:147]) + ''.join(lines[332:421]) + ''.join(lines[582:697])
    assert 'fp_qmv_fast_impl' in pieces and 'fp_qmv_wide_impl' in pieces
    # The generated helper text is derived MLX code and retains its attribution.
    return '''// Derived helpers: Copyright © 2025-2026 Apple Inc. MIT license.
// License retained in installed mlx-0.32.4.dist-info/licenses/LICENSE.
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
    """Generate source/launch metadata without importing MLX or using GPU."""
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
        name=f'mlx2_source_stock_mxfp4_pair_candidate_m{rows}',
        input_names=['x','gate_q','gate_s','up_q','up_s'],
        output_names=['gate_out','up_out'],
        header=spec['header'], source=spec['source'], ensure_row_contiguous=True,
    )


def paired_matmul(x, gate_q, gate_s, up_q, up_s, *, verified_m5_vector_band=False):
    """Parent GPU oracle entry point; no selection/default or parity claim."""
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
