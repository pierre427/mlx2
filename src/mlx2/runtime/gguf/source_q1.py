# SPDX-License-Identifier: MIT
# Packed QMV adapted from mlx-vlm; provenance/deepseek-v41-source-q1-lfu.json.
from functools import lru_cache

GROUP_SIZE = 64
REVISION = "source-q1-affine-g64-lloyd4-f16-v1"
SOURCE = "\n    uint lane = thread_index_in_simdgroup;\n    uint simd_group = simdgroup_index_in_threadgroup;\n    uint input_row = threadgroup_position_in_grid.y;\n    uint input_dims = x_shape[1];\n    uint output_dims = weight_shape[0];\n    uint groups = scales_shape[1];\n    uint output_start = threadgroup_position_in_grid.x * OUTPUTS_PER_THREADGROUP +\n        simd_group * OUTPUTS_PER_SIMDGROUP;\n\n    float accumulators[OUTPUTS_PER_SIMDGROUP] = {0.0f};\n    constexpr uint VALUES_PER_THREAD = 16;\n    constexpr uint BLOCK_SIZE = VALUES_PER_THREAD * 32;\n\n    // Derived from PrismML's qmv_fast SIMD layout. Each lane owns one packed\n    // 16-bit half-word and 16 contiguous activations, lowering register\n    // pressure while four output rows reuse the activation block.\n    for (uint block_start = lane * VALUES_PER_THREAD;\n         block_start < input_dims;\n         block_start += BLOCK_SIZE) {\n        float x_thread[VALUES_PER_THREAD];\n        float total_sum = 0.0f;\n        #pragma clang loop unroll(full)\n        for (uint i = 0; i < VALUES_PER_THREAD; ++i) {\n            float value = static_cast<float>(\n                x[input_row * input_dims + block_start + i]);\n            x_thread[i] = value;\n            total_sum += value;\n        }\n\n        uint group = block_start / GROUP_SIZE;\n        uint packed_column = block_start >> 5;\n        uint packed_shift = block_start & 31;\n        for (uint row = 0; row < OUTPUTS_PER_SIMDGROUP; ++row) {\n            uint output_row = output_start + row;\n            if (ROW_VALID) {\n                ushort packed = ushort(\n                    weight[output_row * weight_shape[1] + packed_column] >>\n                    packed_shift);\n                float selected_sum = 0.0f;\n                #pragma clang loop unroll(full)\n                for (uint i = 0; i < VALUES_PER_THREAD; ++i) {\n                    selected_sum +=\n                        (packed & (ushort(1) << i)) ? x_thread[i] : 0.0f;\n                }\n\n                uint parameter_index = output_row * groups + group;\n                accumulators[row] += selected_sum *\n                    static_cast<float>(scales[parameter_index]);\n                accumulators[row] += total_sum *\n                    static_cast<float>(biases[parameter_index]);\n            }\n        }\n    }\n\n    for (uint row = 0; row < OUTPUTS_PER_SIMDGROUP; ++row) {\n        accumulators[row] = simd_sum(accumulators[row]);\n        uint output_row = output_start + row;\n        if (lane == 0 && output_row < output_dims) {\n            out[input_row * output_dims + output_row] =\n                static_cast<T>(accumulators[row]);\n        }\n    }\n"

@lru_cache(maxsize=1)
def kernel():
    import mlx.core as mx
    source=SOURCE.replace("GROUP_SIZE","64").replace("OUTPUTS_PER_THREADGROUP","8").replace("OUTPUTS_PER_SIMDGROUP","4").replace("ROW_VALID","output_row < output_dims")
    return mx.fast.metal_kernel(name="mlx2_source_q1_g64_f32_v1",input_names=["x","weight","scales","biases"],output_names=["out"],source=source)

def matmul(x, weight, scales, biases):
    import mlx.core as mx
    if x.ndim!=2 or weight.ndim!=2 or weight.dtype!=mx.uint32 or x.shape[1]!=weight.shape[1]*32 or scales.shape!=biases.shape or scales.shape!=(weight.shape[0],x.shape[1]//64):
        raise ValueError("Q1 packed expert shape mismatch")
    n=weight.shape[0]
    return kernel()(inputs=[x.astype(mx.float32),weight,scales,biases],template=[("T",mx.float32)],grid=(((n+7)//8)*64,x.shape[0],1),threadgroup=(64,1,1),output_shapes=[(x.shape[0]*n,)],output_dtypes=[mx.float32])[0].reshape(x.shape[0],n)


@lru_cache(maxsize=1)
def pair_kernel():
    """The identical Q1 projection body, dispatched over two separate planes."""
    import mlx.core as mx
    source = SOURCE.replace("GROUP_SIZE", "64").replace("OUTPUTS_PER_THREADGROUP", "8").replace("OUTPUTS_PER_SIMDGROUP", "4").replace("ROW_VALID", "output_row < output_dims")
    source = source.replace("weight_shape", "gate_weight_shape").replace("scales_shape", "gate_scales_shape")
    select = """
    const bool up = threadgroup_position_in_grid.z != 0;
    auto weight = up ? up_weight : gate_weight;
    auto scales = up ? up_scales : gate_scales;
    auto biases = up ? up_biases : gate_biases;
    auto out = up ? up_out : gate_out;
    """
    return mx.fast.metal_kernel(
        name="mlx2_source_q1_g64_paired_f32_v1",
        input_names=["x", "gate_weight", "gate_scales", "gate_biases", "up_weight", "up_scales", "up_biases"],
        output_names=["gate_out", "up_out"], source=select+source,
    )


def paired_matmul(x, gate_weight, gate_scales, gate_biases, up_weight, up_scales, up_biases):
    """Pair launches without concatenating weights or changing the K reduction."""
    import mlx.core as mx
    triples = ((gate_weight, gate_scales, gate_biases), (up_weight, up_scales, up_biases))
    for weight, scales, biases in triples:
        if (x.ndim != 2 or not x.shape[0] or weight.ndim != 2 or weight.dtype != mx.uint32
                or x.shape[1] != weight.shape[1]*32 or x.shape[1] % 64
                or scales.shape != biases.shape or scales.shape != (weight.shape[0], x.shape[1]//64)
                or scales.dtype != mx.float16 or biases.dtype != mx.float16):
            raise ValueError("Q1 paired expert shape/dtype mismatch")
    if gate_weight.shape != up_weight.shape or not gate_weight.shape[0]:
        raise ValueError("Q1 paired expert projection geometry mismatch")
    n = gate_weight.shape[0]
    return tuple(v.reshape(x.shape[0], n) for v in pair_kernel()(
        inputs=[x.astype(mx.float32), gate_weight, gate_scales, gate_biases, up_weight, up_scales, up_biases],
        template=[("T", mx.float32)], grid=(((n+7)//8)*64, x.shape[0], 2), threadgroup=(64, 1, 1),
        output_shapes=[(x.shape[0]*n,), (x.shape[0]*n,)], output_dtypes=[mx.float32, mx.float32],
    ))


@lru_cache(maxsize=1)
def mapped_pair_kernel():
    """Preserve each projection's K loop while selecting token rows at load."""
    import mlx.core as mx
    base=SOURCE.replace('GROUP_SIZE','64').replace('OUTPUTS_PER_THREADGROUP','8').replace('OUTPUTS_PER_SIMDGROUP','4').replace('ROW_VALID','output_row < output_dims')
    base=base.replace('weight_shape','gate_weight_shape').replace('scales_shape','gate_scales_shape')
    needle='x[input_row * input_dims + block_start + i]'
    assert base.count(needle)==1
    base=base.replace(needle,'x[rows[input_row] * input_dims + block_start + i]')
    select='''
const bool is_up=threadgroup_position_in_grid.z!=0;
auto weight=is_up ? up_weight : gate_weight;
auto scales=is_up ? up_scales : gate_scales;
auto biases=is_up ? up_biases : gate_biases;
auto out=is_up ? up_out : gate_out;
'''
    return mx.fast.metal_kernel(name='mlx2_source_q1_mapped_pair_v1',
        input_names=['x','rows','gate_weight','gate_scales','gate_biases','up_weight','up_scales','up_biases'],
        output_names=['gate_out','up_out'],source=select+base)


def paired_matmul_mapped(x, rows, gate_weight, gate_scales, gate_biases, up_weight, up_scales, up_biases):
    """Token rows selected by a provider-owned uint32 map; no input gather."""
    import mlx.core as mx
    if (x.ndim!=2 or x.dtype!=mx.float32 or not x.shape[0] or x.shape[1]%64
            or rows.ndim!=1 or rows.dtype!=mx.uint32 or not rows.size):
        raise ValueError('Q1 mapped pair input geometry/dtype mismatch')
    for q,s,b in ((gate_weight,gate_scales,gate_biases),(up_weight,up_scales,up_biases)):
        if (q.ndim!=2 or q.dtype!=mx.uint32 or q.shape[1]*32!=x.shape[1]
                or s.shape!=b.shape or s.shape!=(q.shape[0],x.shape[1]//64)
                or s.dtype!=mx.float16 or b.dtype!=mx.float16):
            raise ValueError('Q1 mapped pair weight geometry/dtype mismatch')
    if gate_weight.shape!=up_weight.shape or not gate_weight.shape[0]:
        raise ValueError('Q1 mapped pair projection mismatch')
    n=gate_weight.shape[0]
    return tuple(v.reshape(rows.size,n) for v in mapped_pair_kernel()(
        inputs=[x,rows,gate_weight,gate_scales,gate_biases,up_weight,up_scales,up_biases],
        template=[('T',mx.float32)],grid=(((n+7)//8)*64,rows.size,2),threadgroup=(64,1,1),
        output_shapes=[(rows.size*n,)]*2,output_dtypes=[mx.float32]*2))
