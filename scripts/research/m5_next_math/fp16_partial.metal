// Standalone mlx2 research probe, 2026-09-25.
// No tensor, simdgroup-matrix, NAX, or ANE operations.
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint rows;
    uint k;
};

#define DOT_ARGS \
    device const float* x [[buffer(0)]], \
    device const float* w [[buffer(1)]], \
    device float* out [[buffer(2)]], \
    constant Params& p [[buffer(3)]], \
    uint lane [[thread_index_in_simdgroup]], \
    uint row [[threadgroup_position_in_grid]]

inline float finish(float value) {
    return simd_sum(value);
}

kernel void fp32_fma(DOT_ARGS) {
    if (row >= p.rows) return;
    size_t base = (size_t)row * p.k;
    float sum = 0.0f;
    for (uint j = lane; j < p.k; j += 32)
        sum = fma(x[base + j], w[base + j], sum);
    sum = finish(sum);
    if (lane == 0) out[row] = sum;
}

// This arm rounds each product to FP16, but retains the per-lane accumulator
// in FP32. It separates product rounding from FP16 partial-sum rounding.
kernel void half_partial1_fp32_acc(DOT_ARGS) {
    if (row >= p.rows) return;
    size_t base = (size_t)row * p.k;
    float sum = 0.0f;
    for (uint j = lane; j < p.k; j += 32) {
        half product = half(x[base + j]) * half(w[base + j]);
        sum += float(product);
    }
    sum = finish(sum);
    if (lane == 0) out[row] = sum;
}

template <uint L>
inline float half_partial_impl(
    device const float* x,
    device const float* w,
    size_t base,
    uint k,
    uint lane
) {
    float total = 0.0f;
    for (uint block = lane; block < k; block += 32 * L) {
        half partial = half(0.0h);
#pragma clang loop unroll(full)
        for (uint t = 0; t < L; ++t) {
            uint j = block + 32 * t;
            if (j < k)
                partial = fma(half(x[base + j]), half(w[base + j]), partial);
        }
        total += float(partial);
    }
    return finish(total);
}

#define HALF_PARTIAL(NAME, LENGTH) \
kernel void NAME(DOT_ARGS) { \
    if (row >= p.rows) return; \
    float sum = half_partial_impl<LENGTH>(x, w, (size_t)row * p.k, p.k, lane); \
    if (lane == 0) out[row] = sum; \
}

HALF_PARTIAL(half_partial4, 4)
HALF_PARTIAL(half_partial8, 8)
HALF_PARTIAL(half_partial16, 16)
HALF_PARTIAL(half_partial32, 32)

// Select a conservative power-of-two downscale independently for each
// 16-product lane block. Scale selection, operand conversion, rescaling and
// accumulation are all inside the measured kernel. This protects range but
// does not attempt to protect small terms from cancellation.
inline float half_partial16_scaled_impl(
    device const float* x,
    device const float* w,
    size_t base,
    uint k,
    uint lane
) {
    constexpr uint L = 16;
    float total = 0.0f;
    for (uint block = lane; block < k; block += 32 * L) {
        float max_product = 0.0f;
#pragma clang loop unroll(full)
        for (uint t = 0; t < L; ++t) {
            uint j = block + 32 * t;
            if (j < k)
                max_product = max(max_product, abs(x[base + j] * w[base + j]));
        }

        // The exponent-only scale is exactly representable. Leaving headroom
        // for L equal-signed products prevents a finite FP16 partial overflow.
        float required = max_product * float(L) / 60000.0f;
        uint bits = as_type<uint>(required);
        int exponent = int((bits >> 23) & 0xffu) - 127;
        int downshift = required > 1.0f ? clamp(exponent + 1, 0, 30) : 0;
        float scale = as_type<float>(uint(127 - downshift) << 23);

        half partial = half(0.0h);
#pragma clang loop unroll(full)
        for (uint t = 0; t < L; ++t) {
            uint j = block + 32 * t;
            if (j < k)
                partial = fma(half(x[base + j] * scale),
                              half(w[base + j]), partial);
        }
        total += float(partial) / scale;
    }
    return finish(total);
}

kernel void half_partial16_scaled(DOT_ARGS) {
    if (row >= p.rows) return;
    float sum = half_partial16_scaled_impl(
        x, w, (size_t)row * p.k, p.k, lane);
    if (lane == 0) out[row] = sum;
}
