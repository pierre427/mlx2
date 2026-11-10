# SPDX-License-Identifier: MIT
"""Source width-one router logits from the resident BF16 coefficients.

The ordinary Source router projection is ``x.astype(f32) @ W.astype(f32).T``
with ``W`` the official BF16 ``ffn.gate.weight`` [384, 5120]. MLX does not
fuse the cast into the product, so every decode layer materializes and then
re-reads a fresh F32 copy of ``W`` (3.9 MB read + 7.9 MB written + 7.9 MB read
instead of 3.9 MB). This candidate runs the pinned installed MLX ``GEMVKernel``
F32 lane/loop/shuffle law (``source_dense._gemv_header``: BN1 SM1 SN32 TM4 TN4,
K blocks of 128, shuffle-down tree, no threadgroup reduction) with only the
coefficient pointer typed BF16 and converted to F32 per load. ``float(bf16)``
is exact, so each output row reproduces the stock product of the F32 copy
when the stock dispatch uses the same row-local law (SM1/SN32/TM4/TN4; BM only
changes how many rows a threadgroup owns). That dispatch match is the claim a
GPU run must check: ``first_use_check`` compares this kernel bitwise against
the stock expression once per process and refuses the candidate on mismatch.

Decode ``M=1`` only. Prefill rows and every other shape stay caller-owned.
Reuses ``source_dense`` helpers unchanged so existing head/WoA identities do
not move. Provenance: provenance/deepseek-v41-source-router-gemv.json.
"""
from functools import lru_cache
import hashlib
from pathlib import Path

POLICY = "bf16_coeff_gemv_v1"
N, K = 384, 5120
GEOMETRY = dict(bm=4, bn=1, sm=1, sn=32, tm=4, tn=4)
_CHECKED = {"state": None, "names": set()}  # state False = refused for the process; names = weights that passed


def plan(rows=1):
    """CPU-only geometry; the row partition is checked by the CPU tests."""
    if rows != 1:
        raise ValueError("Source router GEMV admits one decode row only")
    g = GEOMETRY
    threads, block_m = g["bm"] * 32, g["bm"] * g["tm"]
    if N % block_m or K % (g["sn"] * g["tn"]):
        raise ValueError("Source router GEMV geometry is not aligned")
    return {"rows": 1, "n": N, "k": K, "grid": (N // block_m * threads, 1, 1),
            "threadgroup": (threads, 1, 1), "output_shape": (1, 1, N),
            "gemv_geometry": (g["bm"], g["bn"], g["sm"], g["sn"], g["tm"], g["tn"]),
            "coefficient_storage_bytes": N * K * 2, "global_f32_coefficient_copy_bytes": 0,
            "avoided_f32_copy_bytes_per_call": N * K * 4,
            "input_dtype": "float32", "weight_dtype": "bfloat16", "output_dtype": "float32",
            "profile": "candidate-unqualified"}


def _geometry(x, weight):
    import mlx.core as mx
    if (getattr(x, "dtype", None) != mx.float32 or getattr(x, "ndim", None) != 3
            or tuple(x.shape) != (1, 1, K)):
        raise ValueError("Source router GEMV activation geometry/dtype mismatch")
    if getattr(weight, "dtype", None) != mx.bfloat16 or tuple(weight.shape) != (N, K):
        raise ValueError("Source router GEMV coefficient geometry/dtype mismatch")
    return plan(1)


def supports(x, weight, *, check_device=True):
    """Structural/device gate only; False keeps the ordinary F32-copy product."""
    try:
        _geometry(x, weight)
    except (ValueError, AttributeError, TypeError):
        return False
    if not check_device:
        return True
    from .source_dense import device_supported
    return device_supported()


def kernel_spec():
    """Generate pinned helper source without compiling or scheduling GPU work."""
    from .source_dense import _gemv_header
    p = plan(1)
    bm, bn, sm, sn, tm, tn = p["gemv_geometry"]
    source = f"constexpr int N={N}, K={K};\n" + f"""
using dense_kernel = mlx2_source_dense::GEMVKernel<float,{bm},{bn},{sm},{sn},{tm},{tn},false>;
threadgroup float scratch[dense_kernel::tgp_mem_size == 0 ? 1 : dense_kernel::tgp_mem_size];
const int in_size=K,out_size=N,ld=K,bias_stride=1;
const float alpha=1.0f,beta=0.0f;
dense_kernel::run(weight,x,nullptr,out,in_size,out_size,ld,alpha,beta,bias_stride,scratch,
    uint3(threadgroup_position_in_grid.x,0,0),thread_position_in_threadgroup,
    simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
"""
    return {**p, "header": _gemv_header(), "source": source,
            "input_names": ["x", "weight"], "output_names": ["out"]}


@lru_cache(maxsize=1)
def _kernel():
    import mlx.core as mx
    spec = kernel_spec()
    return mx.fast.metal_kernel(name="mlx2_source_router_gemv_bf16_v1",
        input_names=spec["input_names"], output_names=spec["output_names"],
        header=spec["header"], source=spec["source"], ensure_row_contiguous=True)


def project(x, weight):
    """Raw F32 logits [1,1,384]; unsupported inputs raise before dispatch."""
    import mlx.core as mx
    p = _geometry(x, weight)
    return _kernel()(inputs=[x, weight], grid=p["grid"], threadgroup=p["threadgroup"],
                     output_shapes=[p["output_shape"]], output_dtypes=[mx.float32])[0]


def first_use_check(x, weight, name=None):
    """One synchronous bitwise comparison against the stock expression, per weight.

    Runs on the first live decode call for each router weight (each layer), so
    every layer's real coefficients are checked under the dispatch the ordinary
    route uses. Any mismatch, or any error from the candidate, refuses the
    candidate for the whole process. Returns the stock product either way.
    """
    import mlx.core as mx
    import numpy as np
    stock = x.astype(mx.float32) @ weight.astype(mx.float32).T
    try:
        candidate = project(x, weight)
        mx.eval(stock, candidate)
        ok = bool(np.array_equal(np.array(stock).view(np.uint32), np.array(candidate).view(np.uint32)))
    except Exception:
        ok = False
    if ok and _CHECKED["state"] is not False:
        _CHECKED["state"] = True
        _CHECKED["names"].add(name)
    else:
        _CHECKED["state"] = False
    return stock


def checked(name=None):
    """None: this weight still needs its check; True: passed; False: refused process-wide."""
    if _CHECKED["state"] is False:
        return False
    return True if name in _CHECKED["names"] else None


def implementation_identity():
    from . import source_dense
    own = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    helper = hashlib.sha256(Path(source_dense.__file__).read_bytes()).hexdigest()
    return hashlib.sha256((POLICY + ":" + own + ":" + helper + ":" + source_dense.LIBRARY_SHA256
                           + ":" + repr(sorted(source_dense.HASHES.items()))).encode()).hexdigest()
