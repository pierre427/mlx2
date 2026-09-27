#include <metal_stdlib>
using namespace metal;

// Original research kernels. No tensor/matrix accelerator instructions.
inline float precise_op(float x, uint op) {
    // Mirrors the FP32 helper used for BF16-valued z gates in fused GDN.
    if (op == 0) {
        float e = precise::exp(abs(x));
        float y = 1.0f / (1.0f + e);
        return x < 0.0f ? y : 1.0f - y;
    }
    return max(x, 0.0f) + precise::log(1.0f + precise::exp(-abs(x)));
}
inline float fast_op(float x, uint op) {
    if (op == 0) {
        float e = fast::exp(abs(x));
        float y = 1.0f / (1.0f + e);
        return x < 0.0f ? y : 1.0f - y;
    }
    return max(x, 0.0f) + fast::log(1.0f + fast::exp(-abs(x)));
}
struct Params { uint n; uint op; uint mode; uint tableSize; };

kernel void make_exact_table(device float *table [[buffer(0)]],
                             constant uint &op [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) {
    table[i] = precise_op(as_type<float>(i << 16), op);
}

kernel void lookup_bench(device const ushort *input [[buffer(0)]],
                         device float *output [[buffer(1)]],
                         device const float *table [[buffer(2)]],
                         constant Params &p [[buffer(3)]],
                         texture1d<float, access::sample> lut [[texture(0)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    ushort bits = input[i];
    float x = as_type<float>(uint(bits) << 16);
    float y;
    if (p.mode == 0) y = precise_op(x, p.op);
    else if (p.mode == 1) y = fast_op(x, p.op);
    else if (p.mode == 2) y = table[bits];
    else {
        constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
        float coord = ((clamp(x, -12.0f, 12.0f) + 12.0f) *
                       (float(p.tableSize - 1) / 24.0f) + 0.5f) / float(p.tableSize);
        y = lut.sample(s, coord).x;
        // Preserve asymptotes beyond table domain, still approximate.
        if (x > 12.0f) y = p.op == 0 ? 1.0f : x;
        if (x < -12.0f) y = 0.0f;
    }
    output[i] = y;
}
