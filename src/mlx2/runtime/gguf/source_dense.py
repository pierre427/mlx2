# SPDX-License-Identifier: MIT
"""Explicit Source BF16-coefficient/F32-arithmetic projection candidates.

Only coefficient loads differ from the pinned installed MLX F32 kernels.
No expanded coefficient copy or policy is installed here. Parent integration
must compare raw products against the actual ordinary dispatch before use.
Provenance: provenance/deepseek-v41-source-dense.json.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
import re


SHAPES = {"wo_a": (8, 1024, 4096), "head": (1, 129280, 5120)}
HASHES = {
    'steel/utils/type_traits.h': '4766c5a3809d3cf7740e63acc5703ef76b6b4e85d8baf16602c9d517a2fd426c',
    'steel/utils/integral_constant.h': '886eccaac4f8f00eab5a10551fdb5315df7ab7462aba13d69bb95765c1a4735e',
    'steel/defines.h': 'b03cea6a7d5cfe814e6838d8536e2aacdcafd2744cb408e954fda336ce21759a',
    "gemv.h": "296ef622ea3c67d7485aa7cdf102bbfcb582e07bd3207793179a9d1e65ee5a65",
    "steel/gemm/nax.h": "fb10bcae44095e9fed5c81213d03527dd5fe5ac11be11ec7784bd357c6f317ec",
    "steel/gemm/gemm_nax.h": "8f8d11f3c29e30dbfcaa6799963edd5fb71b81f912c4f3494e294d3002e3d337",
}
LIBRARY_SHA256 = "d8c9393d584e781393c3effccfccd8ba85ad0c70264edaf9f897dca9875375c6"
DEFAULT_TILE = (64, 64, 128, 2, 2)


def plan(kind, rows, *, algorithm="auto", tile=DEFAULT_TILE):
    """CPU-only geometry. Selection here is structural, not a parity verdict."""
    if kind not in SHAPES or type(rows) is not int or not 1 <= rows <= 1024:
        raise ValueError("Source dense kind/rows outside admitted shape")
    if algorithm not in ("auto", "gemv", "nax"):
        raise ValueError("Source dense unknown algorithm")
    algorithm = ("gemv" if rows == 1 else "nax") if algorithm == "auto" else algorithm
    groups, n, k = SHAPES[kind]
    output = (1, rows, groups, n) if kind == "wo_a" else (1, rows, n)
    if algorithm == "gemv":
        if rows != 1:
            raise ValueError("Source dense GEMV admits one token row only")
        # Pinned stock non-transposed GEMV: same row-local law for both BM.
        bm = 4 if kind == "wo_a" else 8
        geometry = (bm, 1, 1, 32, 4, 4)
        threads, block_m = bm * 32, bm * 4
        grid = ((n + block_m - 1) // block_m * threads, rows, groups)
        params = {"gemv_geometry": geometry, "output_rows_per_group": block_m}
    else:
        if (type(tile) is not tuple or len(tile) != 5
                or any(type(v) is not int for v in tile)):
            raise ValueError("Source dense NAX tile must be five integer extents")
        bm, bn, bk, wm, wn = tile
        if (bm not in (32, 64, 128) or bn not in (32, 64, 128)
                or bk not in (32, 64, 128) or (wm, wn) not in ((1, 2), (2, 2), (2, 4))
                or bm % (wm * 16) or bn % (wn * 16) or k % bk or n % bn):
            raise ValueError("Source dense NAX tile outside static aligned domain")
        tm, tn = bm//wm//16, bn//wn//16
        if not (tn % 2 == 0 or (tn == 1 and tm % 2 == 0)):
            raise ValueError("Source dense NAX tile has unsupported fragment pairing")
        threads = wm * wn * 32
        grid = (n // bn * threads, (rows + bm - 1) // bm, groups)
        params = {"tile": tile, "k_step": 32, "split_k": 1}
    return {"kind": kind, "algorithm": algorithm, "rows": rows, "groups": groups,
            "n": n, "k": k, "grid": grid, "threadgroup": (threads, 1, 1),
            "output_shape": output, "weight_dtype": "bfloat16", "input_dtype": "float32",
            "output_dtype": "float32", "coefficient_storage_bytes": groups*n*k*2,
            "global_f32_coefficient_copy_bytes": 0, "profile": "candidate-unqualified", **params}


def _geometry(x, weight, kind, algorithm, tile):
    import mlx.core as mx
    if kind not in SHAPES:
        raise ValueError("Source dense unknown projection kind")
    groups, n, k = SHAPES[kind]
    if (x.dtype != mx.float32 or weight.dtype != mx.bfloat16
            or weight.shape != (groups*n, k)):
        raise ValueError("Source dense coefficient geometry/dtype mismatch")
    expected_ndim = 4 if kind == "wo_a" else 3
    if x.ndim != expected_ndim or x.shape[0] != 1 or x.shape[-1] != k:
        raise ValueError("Source dense activation geometry/dtype mismatch")
    if kind == "wo_a" and x.shape[2] != groups:
        raise ValueError("Source dense grouped activation shape mismatch")
    return plan(kind, x.shape[1], algorithm=algorithm, tile=tile)


@lru_cache(maxsize=1)
def _hardware_supported():
    import mlx.core as mx
    return mx.metal.is_available() and mx.device_info().get("device_name") == "Apple M5 Max"


def device_supported():
    import mlx.core as mx
    return mx.default_device() == mx.gpu and _hardware_supported()


def supports(x, weight, kind, *, algorithm="auto", tile=DEFAULT_TILE, check_device=True):
    """Structural/device gate only. False lets the caller retain ordinary math."""
    try:
        _geometry(x, weight, kind, algorithm, tile)
        return not check_device or device_supported()
    except (ValueError, AttributeError, TypeError):
        return False


@lru_cache(maxsize=1)
def _installed():
    import mlx.core as mx
    root = Path(mx.__file__).resolve().parent
    if hashlib.sha256((root/"lib/libmlx.dylib").read_bytes()).hexdigest() != LIBRARY_SHA256:
        raise RuntimeError("Source dense installed MLX library changed; re-audit required")
    headers = root/"include/mlx/backend/metal/kernels"
    for name, expected in HASHES.items():
        if hashlib.sha256((headers/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Source dense installed MLX helper changed; re-audit required")
    return root, headers


@lru_cache(maxsize=1)
def _gemv_header():
    _, headers = _installed()
    text = (headers/"gemv.h").read_text()
    start = text.index("template <\n    typename T,\n    const int BM")
    stop = text.index("/// Vector matrix multiplication", start)
    helper = text[start:stop]
    helper = helper[:helper.rfind("///////////////////////////////////////////////////////////////////////////////")]
    before = "template <typename U = T>\n  static METAL_FUNC void"
    assert helper.count(before) == 2
    helper = helper.replace(before, "template <typename U = T, typename Src = T>\n  static METAL_FUNC void")
    assert helper.count("const device T* src") == 2
    helper = helper.replace("const device T* src", "const device Src* src")
    assert helper.count("const device T* mat [[buffer(0)]]") == 1
    helper = helper.replace("const device T* mat [[buffer(0)]]", "const device bfloat* mat [[buffer(0)]]")
    # Inline wrapper constants are thread-private values, not argument-buffer
    # constant references. This changes argument passing, not arithmetic.
    helper = helper.replace("const constant int&", "const int").replace("const constant float&", "const float")
    helper = re.sub(r" \[\[buffer\(\d+\)\]\]", "", helper)
    return r"""// Derived installed MLX GEMV helper, Apple 2023-2024, MIT.
// License: provenance/source-mxfp4-installed-MLX-LICENSE.txt.
#include <metal_stdlib>
#include <metal_simdgroup>
#ifndef MLX_MTL_PRAGMA_UNROLL
#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#endif
namespace mlx2_source_dense {
using namespace metal;
#define MLX_MTL_CONST static constant constexpr const
template <typename T> struct DefaultAccT { using type = float; };
""" + helper + "\n} // namespace mlx2_source_dense\n"


@lru_cache(maxsize=1)
def _nax_header():
    _, headers = _installed()
    seen = set()
    def expand(name):
        if name in seen:
            return ""
        seen.add(name)
        result = []
        for line in (headers/name).read_text().splitlines():
            if line.startswith('#include "mlx/backend/metal/kernels/'):
                result.append(expand(line.split('kernels/', 1)[1].rstrip('"')))
            elif line != "#pragma once":
                result.append(line)
        return "\n".join(result)
    nax = expand("steel/gemm/nax.h")
    text = (headers/"steel/gemm/gemm_nax.h").read_text()
    helper = text[text.index("template <"):text.rindex("} // namespace")]
    assert helper.count("const device T* B,") == 1
    helper = helper.replace("const device T* B,", "const device bfloat* B,")
    helper = helper.replace("auto gemm_loop(", "auto source_dense_gemm_loop(")
    return nax + "\nnamespace mlx::steel {\n" + helper + "\n}\n"


def kernel_spec(kind, rows, *, algorithm="auto", tile=DEFAULT_TILE):
    """Generate pinned helper source without compiling or scheduling GPU work."""
    p = plan(kind, rows, algorithm=algorithm, tile=tile)
    g, n, k = p['groups'], p['n'], p['k']
    constants = f"constexpr int G={g}, N={n}, K={k}, M={rows};\n"
    if p['algorithm'] == "gemv":
        bm, bn, sm, sn, tm, tn = p['gemv_geometry']
        source = constants + f"""
using dense_kernel = mlx2_source_dense::GEMVKernel<float,{bm},{bn},{sm},{sn},{tm},{tn},false>;
threadgroup float scratch[dense_kernel::tgp_mem_size == 0 ? 1 : dense_kernel::tgp_mem_size];
const uint group=threadgroup_position_in_grid.z;
const uint row=threadgroup_position_in_grid.y;
const device float* input=x+row*G*K+group*K;
const device bfloat* coeff=weight+group*N*K;
device float* dst=out+row*G*N+group*N;
const int in_size=K,out_size=N,ld=K,bias_stride=1;
const float alpha=1.0f,beta=0.0f;
dense_kernel::run(coeff,input,nullptr,dst,in_size,out_size,ld,alpha,beta,bias_stride,scratch,
    uint3(threadgroup_position_in_grid.x,0,0),thread_position_in_threadgroup,
    simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
"""
        header = _gemv_header()
    else:
        bm, bn, bk, wm, wn = p['tile']
        sm, sn = bm//wm, bn//wn
        source = constants + f"""
using namespace mlx::steel;
constexpr short SM={sm},SN={sn},BK={bk},TM=SM/16,TN=SN/16;
const uint group=threadgroup_position_in_grid.z;
const int block_row=threadgroup_position_in_grid.y*{bm};
const int block_col=threadgroup_position_in_grid.x*{bn};
const int tm=SM*(simdgroup_index_in_threadgroup/{wn});
const int tn=SN*(simdgroup_index_in_threadgroup%{wn});
const short live_m=short(max(0,min(int(SM),M-block_row-tm)));
const short live_n=SN;
const device float* input=x+group*K+(block_row+tm)*G*K;
const device bfloat* coeff=weight+(group*N+block_col+tn)*K;
device float* dst=out+group*N+(block_row+tm)*G*N+block_col+tn;
NAXTile<float,TM,TN> result;
// This branch must be threadgroup-uniform: both loop specializations contain
// threadgroup barriers, including the masked SIMD groups of a ragged tile.
if(block_row+{bm}<=M) result=source_dense_gemm_loop<float,SM,SN,32,BK,false,true,true,true,true>(
        input,coeff,G*K,K,K,K/BK,live_m,live_n);
else result=source_dense_gemm_loop<float,SM,SN,32,BK,false,true,false,true,true>(
        input,coeff,G*K,K,K,K/BK,live_m,live_n);
if(live_m>0) result.store_safe(dst,G*N,short2(SN,live_m));
"""
        header = _nax_header()
    return {**p, "header": header, "source": source,
            "input_names": ["x", "weight"], "output_names": ["out"]}


@lru_cache(maxsize=16)
def _kernel(kind, rows, algorithm, tile):
    import mlx.core as mx
    spec = kernel_spec(kind, rows, algorithm=algorithm, tile=tile)
    suffix = "_".join(map(str, tile)) if spec['algorithm'] == 'nax' else 'stock'
    return mx.fast.metal_kernel(name=f"mlx2_source_dense_{kind}_{spec['algorithm']}_r{rows}_{suffix}",
        input_names=spec['input_names'], output_names=spec['output_names'],
        header=spec['header'], source=spec['source'], ensure_row_contiguous=True)


def project(x, weight, kind, *, algorithm="auto", tile=DEFAULT_TILE):
    """Schedule an explicit candidate; unsupported inputs raise before dispatch.

    WoA: x[1,T,8,4096], BF16 weight[8192,4096] -> F32[1,T,8,1024].
    Head: x[1,T,5120], BF16 weight[129280,5120] -> F32[1,T,129280].
    No activation quantization or BF16 output rounding is inserted.
    """
    import mlx.core as mx
    p = _geometry(x, weight, kind, algorithm, tile)
    if not device_supported():
        raise ValueError("Source dense candidate requires owned M5 Max Metal device")
    return _kernel(kind, p['rows'], p['algorithm'], tile)(inputs=[x, weight],
        grid=p['grid'], threadgroup=p['threadgroup'], output_shapes=[p['output_shape']],
        output_dtypes=[mx.float32])[0]


def maybe_project(x, weight, kind, *, algorithm="auto", tile=DEFAULT_TILE):
    """Return None on a structural/device refusal; ordinary fallback is caller-owned."""
    if not supports(x, weight, kind, algorithm=algorithm, tile=tile):
        return None
    return project(x, weight, kind, algorithm=algorithm, tile=tile)


def compilation_source(kind, rows, *, algorithm="auto", tile=DEFAULT_TILE):
    """Standalone Metal translation unit for CPU/offline compiler validation."""
    spec = kernel_spec(kind, rows, algorithm=algorithm, tile=tile)
    return ('#include "mlx/backend/metal/kernels/utils.h"\n' + spec['header']
        + '\nusing namespace metal;\nkernel void source_dense_offline(\n'
        + 'device const float* x [[buffer(0)]],device const bfloat* weight [[buffer(1)]],\n'
        + 'device float* out [[buffer(2)]],uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],\n'
        + 'uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],\n'
        + 'uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],\n'
        + 'uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {\n'
        + spec['source'] + '\n}\n')
