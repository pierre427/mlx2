# SPDX-License-Identifier: MIT
"""Explicit width-one Source down/ordered-sum candidate.

Each expert retains its own packed weights and dot-product arithmetic. Only
the output sink changes: a small threadgroup tile holds F32 contributions,
then each is BF16-rounded and added in stable ascending expert-ID order.
No kernel or device operation runs on import. Provenance and qualification
limits: provenance/deepseek-v41-source-down-fusion.json.
"""
from functools import lru_cache
import hashlib
import operator

HIDDEN = 5120
INTERMEDIATE = 2304
MAX_EXPERTS = 6


def plan(ids, kinds, *, tile_rows=4):
    """CPU-only geometry/ordering plan; each route occurrence remains distinct."""
    if tile_rows not in (4, 8) or type(tile_rows) is not int:
        raise ValueError("Source fused down tile_rows must be4 or8")
    ids, kinds = tuple(ids), tuple(kinds)
    if not 1 <= len(ids) <= MAX_EXPERTS or len(kinds) != len(ids):
        raise ValueError("Source fused down requires1..6 expert occurrences")
    clean = []
    for value in ids:
        try:
            if isinstance(value, bool):
                raise TypeError
            number = operator.index(value)
        except TypeError as error:
            raise ValueError("Source fused down expert ID must be an integer") from error
        if not 0 <= number < 384:
            raise ValueError("Source fused down expert ID out of range")
        clean.append(number)
    if any(kind not in ("original", "q1") for kind in kinds):
        raise ValueError("Source fused down unknown expert format")
    order = tuple(sorted(range(len(clean)), key=lambda i: clean[i]))
    sorted_kinds = tuple(kinds[i] for i in order)
    threads = 32 * len(clean) * (tile_rows//4)
    return dict(ids=tuple(clean[i] for i in order), order=order,
                kinds=sorted_kinds, tile_rows=tile_rows, threads=threads,
                grid=((HIDDEN//tile_rows)*threads, 1, 1),
                threadgroup=(threads, 1, 1),
                logical_input_bindings=sum(3 if kind == "original" else 4 for kind in sorted_kinds),
                scratch_bytes=len(clean)*tile_rows*4)


@lru_cache(maxsize=1)
def _original_header():
    # Read only the exact installed source already reviewed for the paired
    # launcher. The library/header hashes protect dispatch and source spans.
    from .source_mxfp4_pair import HASHES, _verify_library, installed_root
    _verify_library()
    root = installed_root()/"include/mlx/backend/metal/kernels"
    texts = {}
    for name, expected in HASHES.items():
        raw = (root/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise RuntimeError("installed Source down helper changed; re-audit required")
        texts[name] = raw.decode()
    lines = texts["fp_quantized.h"].splitlines(keepends=True)
    # Helpers include the exact fp4 decode, scale, load and qdot laws.
    pieces = "".join(lines[19:147])
    helper = "".join(lines[421:578])
    if (helper.count("METAL_FUNC void fp_qmv_impl(") != 1
            or not helper.startswith("template <typename T, int group_size, int bits, bool has_global_scale = false>")):
        raise RuntimeError("pinned Source down helper span mismatch")
    changes = {
        "fp_qmv_impl(": "fp_qmv_to_shared_impl(",
        "    device T* y,": "    threadgroup T* y,",
        "    y += tid.x * out_vec_size + out_row;": "    y += 0;",
        "    y += tid.x * out_vec_size + used_out_row;": "    y += 0;",
    }
    for before, after in changes.items():
        if helper.count(before) != 1:
            raise RuntimeError("pinned Source down sink anchor mismatch: "+before)
        helper = helper.replace(before, after)
    return r"""
// Derived installed MLX helpers; MIT attribution and license retained in
// provenance/source-mxfp4-installed-MLX-LICENSE.txt.
#include <metal_stdlib>
#include <metal_simdgroup>
namespace mlx2_source_down_stock {
using namespace metal;
typedef half float16_t;
typedef bfloat bfloat16_t;
#ifndef MLX_MTL_CONST
#define MLX_MTL_CONST static constant constexpr const
#endif
MLX_MTL_CONST int SIMD_SIZE =32;
MLX_MTL_CONST int QUAD_SIZE =4;
""" + texts['fp4.h'].replace('#pragma once','') + texts['fp8.h'].replace('#pragma once','') + pieces + helper + r"""
constant int K =2304;
constant int N =5120;
} // namespace mlx2_source_down_stock
"""


def _q1_header():
    from .source_q1 import SOURCE
    source = SOURCE.replace("GROUP_SIZE", "64").replace("OUTPUTS_PER_THREADGROUP", "8").replace("OUTPUTS_PER_SIMDGROUP", "4").replace("ROW_VALID", "output_row < output_dims")
    changes = {
        "    uint lane = thread_index_in_simdgroup;": "",
        "    uint simd_group = simdgroup_index_in_threadgroup;": "",
        "threadgroup_position_in_grid.y": "0",
        "threadgroup_position_in_grid.x": "tile",
        "x_shape[1]": "2304",
        "weight_shape[0]": "5120",
        "weight_shape[1]": "72",
        "scales_shape[1]": "36",
        "out[input_row * output_dims + output_row]": "out[row]",
        "static_cast<T>": "static_cast<float>",
    }
    for before, after in changes.items():
        if source.count(before) != 1:
            raise RuntimeError("Source Q1 down source anchor changed: "+before)
        source = source.replace(before, after)
    return r"""
inline void mlx2_source_q1_down_shared(
    const device float* x, const device uint32_t* weight,
    const device half* scales, const device half* biases,
    threadgroup float* out, uint tile, uint simd_group, uint lane) {
""" + source + "\n}\n"


def kernel_spec(kinds, *, tile_rows=4, return_contributions=False):
    """Build source metadata only; CPU source/hash checks, no Metal compile."""
    kinds = tuple(kinds)
    if type(return_contributions) is not bool:
        raise ValueError('Source fused down contribution flag must be boolean')
    d = plan(range(len(kinds)), kinds, tile_rows=tile_rows)
    kinds = d['kinds']
    names, cases = [], []
    for slot, kind in enumerate(kinds):
        names += [f'x{slot}', f'q{slot}', f's{slot}']
        if kind == 'q1':
            names += [f'b{slot}']
            call = f'mlx2_source_q1_down_shared(x{slot},q{slot},s{slot},b{slot},sink,virtual_tile,virtual_sg,lane);'
        else:
            call = f'mlx2_source_down_stock::fp_qmv_to_shared_impl<float,32,4,false>(q{slot},s{slot},nullptr,x{slot},sink,mlx2_source_down_stock::K,mlx2_source_down_stock::N,uint3(0,virtual_tile,0),virtual_sg,lane);'
        cases.append(f'case {slot}: {call} break;')
    count = len(kinds)
    header = (_original_header() if 'original' in kinds else '') + (_q1_header() if 'q1' in kinds else '')
    source = f"""
const uint tile =threadgroup_position_in_grid.x;
const uint sg =simdgroup_index_in_threadgroup;
const uint lane =thread_index_in_simdgroup;
const uint slot =sg/{tile_rows//4};
const uint virtual_tile ={'tile/2' if tile_rows == 4 else 'tile'};
const uint virtual_sg ={'tile%2' if tile_rows == 4 else 'sg%2'};
threadgroup float partials[{count*tile_rows}];
threadgroup float* sink =partials+slot*{tile_rows}+{'0' if tile_rows == 4 else '(sg%2)*4'};
switch(slot) {{
""" + '\n'.join(cases) + f"""
}}
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0 && lane<{tile_rows}) {{
    volatile float total =0.0f;
    #pragma clang loop unroll(disable)
    for(uint item=0;item<{count};++item) {{
        const float raw =partials[item*{tile_rows}+lane];
""" + (f'        contributions[item*5120+tile*{tile_rows}+lane]=raw;\n' if return_contributions else '') + f"""
        total =total+float(bfloat(raw));
    }}
    output[tile*{tile_rows}+lane]=total;
}}
"""
    return dict(d, header=header, source=source, input_names=names,
                output_names=['output']+(['contributions'] if return_contributions else []))


@lru_cache(maxsize=128)
def _kernel(kinds, tile_rows, return_contributions):
    import mlx.core as mx
    spec = kernel_spec(kinds, tile_rows=tile_rows, return_contributions=return_contributions)
    suffix = ''.join('q' if k=='q1' else 'o' for k in kinds)
    return mx.fast.metal_kernel(
        name=f'mlx2_source_down_combine_{suffix}_r{tile_rows}_p{int(return_contributions)}',
        input_names=spec['input_names'], output_names=spec['output_names'],
        header=spec['header'], source=spec['source'], ensure_row_contiguous=True,
    )


def prepare_inputs(middles, experts, ids, *, tile_rows=4):
    """Validate without evaluating arrays; return sorted arguments and plan."""
    import mlx.core as mx
    middles, experts, ids = tuple(middles), tuple(experts), tuple(ids)
    if len(middles) != len(ids) or len(experts) != len(ids):
        raise ValueError('Source fused down input count mismatch')
    kinds = tuple('q1' if 'down.biases' in expert else 'original' for expert in experts)
    d = plan(ids, kinds, tile_rows=tile_rows)
    args = []
    for index in d['order']:
        x, expert, kind = middles[index], experts[index], kinds[index]
        if x.shape != (1, INTERMEDIATE) or x.dtype != mx.float32:
            raise ValueError('Source fused down requires F32[1,2304] middle activations')
        if not {'down.qweight','down.scales'}.issubset(expert):
            raise ValueError('Source fused down missing packed projection fields')
        q, s = expert['down.qweight'], expert['down.scales']
        if kind == 'original':
            if q.shape != (HIDDEN, INTERMEDIATE//8) or q.dtype != mx.uint32 or s.shape != (HIDDEN, INTERMEDIATE//32) or s.dtype != mx.uint8:
                raise ValueError('Source fused down original geometry/dtype mismatch')
            args += [x, q, s]
        else:
            b = expert['down.biases']
            if q.shape != (HIDDEN, INTERMEDIATE//32) or q.dtype != mx.uint32 or s.shape != (HIDDEN, INTERMEDIATE//64) or s.dtype != mx.float16 or b.shape != s.shape or b.dtype != mx.float16:
                raise ValueError('Source fused down Q1 geometry/dtype mismatch')
            args += [x, q, s, b]
    return d, args


def down_combine(middles, experts, ids, *, tile_rows=4, return_contributions=False):
    """Fuse width-one down dots, BF16 rounding and stable expert-ID F32 sum.

    The diagnostic option additionally returns raw F32 dot contributions in
    sorted-ID order. Normal execution allocates only the final[1,5120] output.
    Expert lifetime remains the caller's completion-window responsibility.
    """
    import mlx.core as mx
    if type(return_contributions) is not bool:
        raise ValueError('Source fused down contribution flag must be boolean')
    d, args = prepare_inputs(middles, experts, ids, tile_rows=tile_rows)
    if mx.default_device() != mx.gpu or not mx.metal.is_available() or mx.device_info().get('device_name') != 'Apple M5 Max':
        raise ValueError('Source fused down requires an owned M5 Max GPU window')
    count = len(d['ids'])
    result = _kernel(d['kinds'],tile_rows,return_contributions)(
        inputs=args, grid=d['grid'], threadgroup=d['threadgroup'],
        output_shapes=[(1,HIDDEN)]+([(count,HIDDEN)] if return_contributions else []),
        output_dtypes=[mx.float32]*(2 if return_contributions else 1))
    return tuple(result) if return_contributions else result[0]
