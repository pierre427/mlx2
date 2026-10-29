# SPDX-License-Identifier: MIT
"""Q8 projection/rounding fusion retaining the selected cooperative dot law."""

from functools import lru_cache

from .cooperative_metal import BLOCK_HEADER, DENSE, STAGE, PAIRED_Q8
from .v41_metal import BF16_HEADER
from .reader import GGUFError

ROUND_HEADER = BF16_HEADER[BF16_HEADER.index("inline float v41_bf16"):]


@lru_cache(maxsize=2)
def kernel(paired):
    import mlx.core as mx

    source = DENSE.replace(STAGE, "const uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;\n")
    source = source.replace("FORMAT==8 ? q8_dot32(p,chunk,a) : iq2_dot32(p,chunk,a,grids,signs)", "q8_dot32(p,chunk,a)")
    if paired:
        source = source.replace("const uint row0=(threadgroup_position_in_grid.x*4+sg)*ROWS;", r"""
const bool second=threadgroup_position_in_grid.x>=BLOCKS_A;
const uint block=second ? threadgroup_position_in_grid.x-BLOCKS_A : threadgroup_position_in_grid.x;
const uint row0=(block*4+sg)*ROWS, N=second ? NB : NA;
device const uchar *w=second ? wb : wa;
device float *y=second ? yb : ya;
const bool rounded=second ? ROUND_B : ROUND_A;
""")
        names, outs = ["x", "wa", "wb"], ["ya", "yb"]
    else:
        names, outs = ["x", "w"], ["y"]
        source = "const bool rounded=ROUND_A;\n"+source
    source = source.replace("y[size_t(token)*N+row0+r]=sum;", "y[size_t(token)*N+row0+r]=rounded ? v41_bf16(sum) : sum;")
    return mx.fast.metal_kernel(name=f"mlx2_v41_q8_epilogue_{int(paired)}_v1", input_names=names, output_names=outs,
                               header=BLOCK_HEADER+ROUND_HEADER, source=source, ensure_row_contiguous=True)


def project(x, a, b=None, *, rounded=True, rounded_b=True):
    import mlx.core as mx

    if a.ggml_type != 8 or len(a.shape) != 2 or a.shape[1] != x.shape[-1] or x.shape[-1] % 32:
        raise GGUFError("fused V4.1 dense requires cooperative Q8 geometry")
    n, k = a.shape
    flat = x.reshape(-1, k).astype(mx.float32)
    blocks = (n+15)//16
    template = [("K", k), ("ROWS", 4), ("FORMAT", 8), ("ROW_BYTES", a.row_bytes), ("ROUND_A", rounded)]
    if b is None:
        result = kernel(False)(inputs=[flat, a.data], template=template+[("N", n)],
                               grid=(blocks*128, 1, flat.shape[0]), threadgroup=(128,1,1),
                               output_shapes=[(flat.shape[0], n)], output_dtypes=[mx.float32])[0]
        return result.reshape(*x.shape[:-1], n)
    if b.ggml_type != 8 or len(b.shape) != 2 or b.shape[1] != k:
        raise GGUFError("paired V4.1 Q8 axes disagree")
    nb = b.shape[0]
    out = kernel(True)(inputs=[flat,a.data,b.data],
                       template=template+[("NA",n),("NB",nb),("BLOCKS_A",blocks),("ROUND_B",rounded_b)],
                       grid=((blocks+(nb+15)//16)*128,1,flat.shape[0]), threadgroup=(128,1,1),
                       output_shapes=[(flat.shape[0],n),(flat.shape[0],nb)], output_dtypes=[mx.float32,mx.float32])
    return tuple(v.reshape(*x.shape[:-1], dim) for v,dim in zip(out,(n,nb)))


@lru_cache(maxsize=1)
def shared_kernel():
    import mlx.core as mx

    source=PAIRED_Q8.replace("gate_out[size_t(token)*N+row0+r]=g;\n        up_out[size_t(token)*N+row0+r]=u;",r"""
        float clipped=min(v41_bf16(g),10.0f), up_value=clamp(v41_bf16(u),-10.0f,10.0f);
        float y=1.0f/(1.0f+metal::precise::exp(abs(clipped)));
        float sigmoid=clipped<0.0f ? y : 1.0f-y;
        volatile float silu=clipped*sigmoid;
        mid[size_t(token)*N+row0+r]=v41_bf16(silu*up_value);
""")
    return mx.fast.metal_kernel(name="mlx2_v41_q8_shared_swiglu_v1",input_names=["x","gate","up"],output_names=["mid"],
                               header=BLOCK_HEADER+ROUND_HEADER,source=source,ensure_row_contiguous=True)


def shared_middle(x, gate, up):
    import mlx.core as mx

    self_check()
    if gate.ggml_type != 8 or gate.shape != up.shape or up.ggml_type != 8 or gate.shape[1] != x.shape[-1]:
        raise GGUFError("shared Q8 axes disagree")
    n,k=gate.shape;flat=x.reshape(-1,k)
    out=shared_kernel()(inputs=[flat,gate.data,up.data],
                        template=[("N",n),("K",k),("ROWS",4),("ROW_BYTES",gate.row_bytes)],
                        grid=(((n+15)//16)*128,1,flat.shape[0]),threadgroup=(128,1,1),
                        output_shapes=[(flat.shape[0],n)],output_dtypes=[mx.float32])[0]
    return out.reshape(*x.shape[:-1],n)


@lru_cache(maxsize=1)
def self_check():
    import mlx.core as mx
    import numpy as np
    from .packed_metal import PackedMatrix
    from .cooperative_metal import cooperative_dense
    from .reader import GGUFTensor
    from ..models.deepseek_v4 import limited_swiglu

    rng=np.random.default_rng(4196)
    matrices=[]
    for n in (48,32):
        raw=rng.integers(0,256,(n*8,34),np.uint8)
        raw[:,:2]=np.frombuffer(np.float16(.0625).tobytes(),np.uint8)
        matrices.append(PackedMatrix(mx.array(raw.ravel()),GGUFTensor("q8",(256,n),8,0,raw.nbytes)))
    for rows in (1,6,10):
        x=mx.array(rng.normal(size=(rows,256)).astype(np.float32))
        for rounded in (False,True):
            expected=[cooperative_dense(x,w,rows=4) for w in matrices]
            if rounded:expected=[v.astype(mx.bfloat16).astype(mx.float32) for v in expected]
            actual=project(x,*matrices,rounded=rounded,rounded_b=rounded)
            for a,b in zip(actual,expected):
                mx.eval(a,b)
                if np.array(a).tobytes()!=np.array(b).tobytes():raise GGUFError("V4.1 paired Q8 BF16 gate failed")
            single=project(x,matrices[0],rounded=rounded);mx.eval(single)
            if np.array(single).tobytes()!=np.array(expected[0]).tobytes():raise GGUFError("V4.1 Q8 epilogue gate failed")
        # Invoke the raw kernel to avoid recursive first-use validation.
        gate,up=matrices[1],matrices[1]
        a=cooperative_dense(x,gate,rows=4).astype(mx.bfloat16).astype(mx.float32)
        expected=limited_swiglu(a,a).astype(mx.bfloat16).astype(mx.float32)
        actual=shared_kernel()(inputs=[x,gate.data,up.data],template=[("N",32),("K",256),("ROWS",4),("ROW_BYTES",gate.row_bytes)],
                               grid=(256,1,rows),threadgroup=(128,1,1),output_shapes=[(rows,32)],output_dtypes=[mx.float32])[0]
        mx.eval(expected,actual)
        if np.array(actual).tobytes()!=np.array(expected).tobytes():raise GGUFError("V4.1 shared SwiGLU gate failed")
