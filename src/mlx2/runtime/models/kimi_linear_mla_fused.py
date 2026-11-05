# SPDX-License-Identifier: MIT
"""Kimi B1 MLA fused decode, mined from GLM's split online-softmax kernel.

Adds Kimi's 64 shared NoPE key channels to the 512 latent channels used by
GLM. Values remain the 512 latent channels. At most 32 split partials keep
workspace bounded even at 1M. Prefill and wider batches keep tiled MLX ops.
This is a separate FP32 dot/reduction law, not exact expanded-SDPA parity.
"""
from functools import lru_cache

import mlx.core as mx

HEADS, LATENT, SHARED, THREADS = 32, 512, 64, 128


def kernel_specs():
    from .glm53_sparse_attention import kernel_spec
    donor = kernel_spec()
    partial = dict(donor["partial"])
    source = partial["source"]
    changes = [
        ("const int T0 = meta[0];\nconst int D = meta[1];\nconst int W = meta[2];\nconst int LAST_ID = T0 + D - 1;", "const int W = meta[0];"),
        ("threadgroup int sid[STAGE];", "threadgroup int sid[STAGE];\nthreadgroup float shared_stage[STAGE * 64];\nfloat shared_q[2];\nfor (uint i = 0; i < 2; i++) shared_q[i] = extra_q[head * 64 + lane + 32 * i];"),
        ("sid[tid] = k < last ? min(ids[size_t(row) * W + k], LAST_ID) : -1;", "sid[tid] = k < last ? k : -1;"),
        ("stage[e] = id < T0 ? base[size_t(id) * DIM + c] : tail[size_t(id - T0) * DIM + c];", "stage[e] = float(base[size_t(id) * DIM + c]);"),
        ("    for (uint r = 0; r < STAGE; r++) {", "    for (uint e = tid; e < STAGE * 64; e += TG) {\n        const int id = sid[e / 64];\n        if (id >= 0) shared_stage[e] = float(extra_k[size_t(id) * 64 + e % 64]);\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    for (uint r = 0; r < STAGE; r++) {"),
        ("        const float score = simd_sum(part) * SCALE;", "        for (uint i = 0; i < 2; i++) part += shared_q[i] * shared_stage[r * 64 + lane + 32 * i];\n        const float score = simd_sum(part) * SCALE;"),
    ]
    for old,new in changes:
        if source.count(old) != 1:
            raise RuntimeError("Kimi MLA fused donor changed: "+old[:60])
        source = source.replace(old,new)
    header = "constant constexpr float SCALE = "+float(192**-0.5).hex()+"f;\n"
    partial.update(name="mlx2_kimi_linear_mla_decode_v1",source=source,header=header,
                   input_names=["q","base","extra_q","extra_k","meta"])
    combine = dict(donor["combine"])
    combine.update(name="mlx2_kimi_linear_mla_combine_v1",header=header)
    return partial,combine


@lru_cache(maxsize=1)
def _kernels():
    return tuple(mx.fast.metal_kernel(**spec,ensure_row_contiguous=True) for spec in kernel_specs())


def admits(q, qs, latent, shared, mask):
    return (mx.default_device() == mx.gpu and mask is None
            and q.shape == (1,32,1,512) and qs.shape == (1,32,1,64)
            and latent.shape[:2] == shared.shape[:2] == (1,1)
            and latent.shape[-1] == 512 and shared.shape[-1] == 64
            and latent.shape[2] == shared.shape[2] and latent.shape[2] > 0)


def attend(q, qs, latent, shared):
    if not admits(q,qs,latent,shared,None):
        raise ValueError("Kimi fused MLA requires one unmasked B1 decode row")
    width = latent.shape[2]
    # More splits expose decode parallelism; their number never scales to 1M.
    split = max(128, ((width+31)//32+7)//8*8)
    count = (width+split-1)//split
    template = [("DIM",512),("HEADS",32),("SPLIT",split),("STAGE",8),
                ("TG",THREADS),("SPLITS",count)]
    partial,combine = _kernels()
    m,l,o = partial(inputs=[q,latent,qs,shared,mx.array([width],mx.int32)],
        template=template,grid=(THREADS*count,8,1),threadgroup=(THREADS,1,1),
        output_shapes=[(1,32,count),(1,32,count),(1,32,count,512)],output_dtypes=[mx.float32]*3)
    return combine(inputs=[m,l,o],template=template,grid=(512,32,1),threadgroup=(512,1,1),
        output_shapes=[(1,32,512)],output_dtypes=[mx.float32])[0][:,:,None,:]
