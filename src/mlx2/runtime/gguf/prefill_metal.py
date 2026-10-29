"""Opt-in packed GGML prefill tile candidate using Metal SIMDgroup MMA.

This is a distinct arithmetic route: packed weights and activations round to
F16 in bounded 8-token/32-row/32-K shared tiles, FP32 accumulation. It is not
an exact ordinary-GEMV replacement and is not selected by default. It must
pass artifact-bound probes before selection. This is SIMDgroup MMA, not a
claim that M5 TensorOps/NAX instructions were used.
"""

from functools import lru_cache

from .packed_metal import HEADER, PackedMatrix
from .reader import GGUFError

SOURCE = r"""
const uint tid=thread_index_in_threadgroup;
const uint sg=simdgroup_index_in_threadgroup;
const uint base_row=threadgroup_position_in_grid.x*32;
const uint base_token=threadgroup_position_in_grid.y*8;
threadgroup half a[8*32];
threadgroup half b[32*32];
threadgroup float c[8*32];
simdgroup_float8x8 acc=simdgroup_float8x8(0.0f);
for(uint kb=0;kb<K;kb+=32) {
    for(uint i=tid;i<8*32;i+=128) {
        uint token=base_token+i/32, k=kb+i%32;
        a[i]=(token<TOKENS && k<K) ? half(x[size_t(token)*K+k]) : half(0.0f);
    }
    for(uint i=tid;i<32*32;i+=128) {
        uint row=base_row+i/32, k=kb+i%32;
        b[i]=(row<N && k<K) ? half(gguf_value(w+size_t(row)*ROW_BYTES,k,FORMAT)) : half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint kk=0;kk<32;kk+=8) {
        simdgroup_half8x8 am,bm;
        simdgroup_load(am,a+kk,32);
        simdgroup_load(bm,b+sg*8*32+kk,32,ulong2(0),true);
        simdgroup_multiply_accumulate(acc,am,bm,acc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
simdgroup_store(acc,c+sg*8,32);
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint i=tid;i<8*32;i+=128) {
    uint token=base_token+i/32,row=base_row+i%32;
    if(token<TOKENS && row<N) y[size_t(token)*N+row]=c[i];
}
"""


@lru_cache(maxsize=1)
def _kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="mlx2_gguf_packed_prefill_simd_fp16_v1",
        input_names=["x", "w"],
        output_names=["y"],
        header=HEADER,
        source=SOURCE,
        ensure_row_contiguous=True,
    )


def packed_prefill(x, weight: PackedMatrix):
    """Explicit route only; rank2 activations/weights, packed bytes retained."""
    import mlx.core as mx

    if x.ndim != 2 or len(weight.shape) != 2 or x.shape[-1] != weight.shape[-1]:
        raise GGUFError("packed prefill requires matching rank2 input/weights")
    tokens, k = x.shape
    n = weight.shape[0]
    if tokens < 1:
        raise GGUFError("packed prefill requires at least one token")
    template = [
        ("K", k),
        ("N", n),
        ("TOKENS", tokens),
        ("ROW_BYTES", weight.row_bytes),
        ("FORMAT", weight.ggml_type),
    ]
    return _kernel()(
        inputs=[x.astype(mx.float32), weight.data],
        template=template,
        grid=(((n + 31) // 32) * 128, (tokens + 7) // 8, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(tokens, n)],
        output_dtypes=[mx.float32],
    )[0]
