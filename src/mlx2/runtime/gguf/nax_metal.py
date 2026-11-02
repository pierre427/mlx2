"""Opt-in M5 tensor-op packed GGUF tiles, with a distinct FP16-input law.

Fragment scheduling follows the existing TensorFold-derived lane matmul.
See provenance/deepseek-v4-gguf-nax.json and the retained MIT notices. Packed
GGUF values are decoded by their actual block format before tensor execution.
This candidate never registers itself as a selected serving route.
"""

from functools import lru_cache
from math import prod

from .k_blocks import SIZES as K_FORMATS
from .packed_metal import HEADER, PackedMatrix
from .reader import GGUFError

# Q8_0/Q2_K/IQ2_XXS were the original DeepSeek candidates. The K-block types
# (Q4_0, Q3_K..Q6_K) reuse the same tile through gguf_value's block decode;
# GLM-5.3 REAP50 is their first consumer (still an explicit FP16-operand law).
LEGACY_FORMATS = (8, 10, 16)
NAX_FORMATS = LEGACY_FORMATS + tuple(sorted(K_FORMATS))


def nax_admits(ggml_type, n, k):
    """Whether a packed [n, k] matrix (or each group of one) fits the NAX tile."""
    return ggml_type in NAX_FORMATS and n % 32 == 0 and k % 256 == 0

NAX_HEADER = (
    HEADER
    + "\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;\n"
)
NAX_SOURCE = r"""
const ushort lane=thread_index_in_simdgroup;
const ushort sg=simdgroup_index_in_threadgroup;
const ushort tid=thread_index_in_threadgroup;
const int rowbase=threadgroup_position_in_grid.y*16;
const int colbase=(threadgroup_position_in_grid.x*4+sg)*32;
const short qid=lane>>2;
const short fm=(qid&4)|((lane>>1)&3);
const short fn=((qid&2)|(lane&1))*4;
threadgroup half xa[16*64];
threadgroup half weights[4*32*64];
threadgroup half* wb=weights+sg*32*64;
constexpr auto desc=matmul2d_descriptor(16,32,64,false,true,false,
    matmul2d_descriptor::mode::multiply);
matmul2d<desc,execution_simdgroup> op;
float sums[16];
#pragma clang loop unroll(full)
for(int j=0;j<16;j++) sums[j]=0.0f;
for(int kb=0;kb<K;kb+=64) {
    for(int i=tid;i<16*64;i+=128) {
        const int row=rowbase+i/64,k=kb+i%64;
        xa[i]=row<TOKENS ? half(x[ulong(row)*K+k]) : half(0.0f);
    }
    for(int i=lane;i<32*64;i+=32) {
        const int row=colbase+i/64,k=kb+i%64;
        wb[i]=row<N ? half(gguf_value(w+ulong(row)*ROW_BYTES,k,FORMAT)) : half(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tensor<threadgroup half,dextents<int32_t,2>,tensor_inline> a(xa,dextents<int32_t,2>(64,16));
    tensor<threadgroup half,dextents<int32_t,2>,tensor_inline> b(wb,dextents<int32_t,2>(64,32));
    auto product=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    op.run(a,b,product);
#pragma clang loop unroll(full)
    for(int j=0;j<16;j++) sums[j]+=product[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
#pragma clang loop unroll(full)
for(int f=0;f<2;f++) {
#pragma clang loop unroll(full)
    for(int r=0;r<2;r++) {
        const int row=rowbase+fm+r*8,col=colbase+f*16+fn;
        if(row<TOKENS && col<N) {
#pragma clang loop unroll(full)
            for(int j=0;j<4;j++) y[ulong(row)*N+col+j]=sums[f*8+r*4+j];
        }
    }
}
"""


GROUPED_NAX_SOURCE = NAX_SOURCE.replace(
    "const int rowbase=threadgroup_position_in_grid.y*16;",
    "const int rowbase=threadgroup_position_in_grid.y*16;\n"
    "const int group=threadgroup_position_in_grid.z;",
).replace(
    "x[ulong(row)*K+k]", "x[(ulong(row)*GROUPS+group)*K+k]"
).replace(
    "w+ulong(row)*ROW_BYTES", "w+(ulong(group)*N+row)*ROW_BYTES"
).replace(
    "y[ulong(row)*N+col+j]", "y[(ulong(row)*GROUPS+group)*N+col+j]"
)


@lru_cache(maxsize=2)
def _kernel(grouped=False):
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="mlx2_gguf_nax_fp16_tile_v2" + ("_grouped" if grouped else ""),
        input_names=["x", "w"],
        output_names=["y"],
        header=NAX_HEADER,
        source=GROUPED_NAX_SOURCE if grouped else NAX_SOURCE,
        ensure_row_contiguous=True,
    )


def nax_prefill(x, weight: PackedMatrix):
    """Explicit M5-only candidate; no fallback and no ordinary route mutation."""
    import mlx.core as mx

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise GGUFError("GGUF NAX requires an owned Metal GPU execution window")
    if "Apple M5" not in mx.device_info().get("device_name", ""):
        raise GGUFError("GGUF NAX candidate is restricted to Apple M5")
    if (
        x.ndim != 2
        or len(weight.shape) != 2
        or x.shape[-1] != weight.shape[-1]
        or x.shape[0] < 1
        or weight.shape[0] % 32
        or x.shape[-1] % 256
        or weight.ggml_type not in NAX_FORMATS
    ):
        raise GGUFError("GGUF NAX requires block-aligned rank2 Q8_0/Q2_K/IQ2_XXS/K-block")
    m, k = x.shape
    n = weight.shape[0]
    return _kernel()(
        inputs=[x.astype(mx.float32), weight.data],
        template=[
            ("TOKENS", m),
            ("K", k),
            ("N", n),
            ("ROW_BYTES", weight.row_bytes),
            ("FORMAT", weight.ggml_type),
        ],
        grid=(((n + 127) // 128) * 128, (m + 15) // 16, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[mx.float32],
    )[0]


def nax_grouped_prefill(x, weight: PackedMatrix):
    """Explicit half-operand candidate, preserving independent group axes."""
    import mlx.core as mx

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise GGUFError("GGUF NAX requires an owned Metal GPU execution window")
    if "Apple M5" not in mx.device_info().get("device_name", ""):
        raise GGUFError("GGUF NAX candidate is restricted to Apple M5")
    if (len(weight.shape) != 3 or x.ndim < 2
            or x.shape[-2:] != (weight.shape[0], weight.shape[2])
            or weight.shape[1] % 32 or weight.shape[2] % 256
            or weight.ggml_type not in NAX_FORMATS):
        raise GGUFError("GGUF grouped NAX requires block-aligned [...,groups,in] input")
    groups, n, k = weight.shape
    leading = x.shape[:-2]
    m = prod(leading)
    if m < 1:
        raise GGUFError("GGUF grouped NAX requires positive row count")
    y = _kernel(True)(
        inputs=[x.reshape(m, groups, k).astype(mx.float32), weight.data],
        template=[("TOKENS", m), ("K", k), ("N", n), ("GROUPS", groups),
                  ("ROW_BYTES", weight.row_bytes), ("FORMAT", weight.ggml_type)],
        grid=(((n + 127) // 128) * 128, (m + 15) // 16, groups),
        threadgroup=(128, 1, 1), output_shapes=[(m, groups, n)],
        output_dtypes=[mx.float32],
    )[0]
    return y.reshape(*leading, groups, n)
