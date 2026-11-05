// mlx2 helper over the Sushi-derived header (header_mcg_w16): accumulate one
// 16-column output tile over input tiles [tk0, tk1) of prepared input held in
// threadgroup memory, with the lane mapping of the GEMV kernels (rows prow,
// prow+1, prow+8, prow+9; columns pcol, pcol+8).
template <uint N>
inline void exl3_acc_tiles(const device uint *words, size_t step, threadgroup const half *prep,
                           uint tk0, uint tk1, uint lane, thread float *acc) {
  const uint sh[8] = {exl3_lane_sh(N, 0u), exl3_lane_sh(N, 1u), exl3_lane_sh(N, 2u), exl3_lane_sh(N, 3u),
                      exl3_lane_sh(N, 4u), exl3_lane_sh(N, 5u), exl3_lane_sh(N, 6u), exl3_lane_sh(N, 7u)};
  const uint prow = (lane & 3u) * 2u;
  for (uint tk = tk0; tk < tk1; tk++) {
    const auto merged = exl3_lane<N>(words + (size_t)tk * step, lane);
    float2 w[4];
    for (uint p = 0u; p < 4u; p++) {
      const uint2 cw = uint2(exl3_lane_word(merged, sh[p * 2u]), exl3_lane_word(merged, sh[p * 2u + 1u])) & uint2(0xffffu);
      w[p] = exl3_decode2(cw);
    }
    const float in0 = float(prep[tk * 16u + prow]);
    const float in1 = float(prep[tk * 16u + prow + 1u]);
    const float in2 = float(prep[tk * 16u + prow + 8u]);
    const float in3 = float(prep[tk * 16u + prow + 9u]);
    acc[0] = fma(in0, w[0].x, acc[0]);
    acc[1] = fma(in1, w[0].y, acc[1]);
    acc[2] = fma(in2, w[1].x, acc[2]);
    acc[3] = fma(in3, w[1].y, acc[3]);
    acc[4] = fma(in0, w[2].x, acc[4]);
    acc[5] = fma(in1, w[2].y, acc[5]);
    acc[6] = fma(in2, w[3].x, acc[6]);
    acc[7] = fma(in3, w[3].y, acc[7]);
  }
}

// H128 of a 128-value block read through `v` (four values per lane, lane +
// 32k), PREPARE/FINISH arithmetic; returns the four outputs before scaling.
inline float4 exl3_h128_lane(float4 v, ushort lane) {
  for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    const float p0 = simd_shuffle_xor(v.x, bit);
    const float p1 = simd_shuffle_xor(v.y, bit);
    const float p2 = simd_shuffle_xor(v.z, bit);
    const float p3 = simd_shuffle_xor(v.w, bit);
    const bool lower = (lane & bit) == 0u;
    v.x = lower ? v.x + p0 : p0 - v.x;
    v.y = lower ? v.y + p1 : p1 - v.y;
    v.z = lower ? v.z + p2 : p2 - v.z;
    v.w = lower ? v.w + p3 : p3 - v.w;
  }
  const float s0 = v.x + v.y;
  const float s1 = v.x - v.y;
  const float s2 = v.z + v.w;
  const float s3 = v.z - v.w;
  return float4(s0 + s2, s1 + s3, s0 - s2, s1 - s3);
}

// SwiGLU of one finished gate/up pair as FUSED_GATE_UP rounds it: g and u to
// half as FINISH writes them, then to R, MLX's sigmoid form, each op in R.
template <typename R>
inline float exl3_swiglu(float gv, float uv) {
  const float g = float(R(half(gv)));
  const float u = float(R(half(uv)));
  const float y = float(R(1.0f / (1.0f + metal::exp(-metal::abs(g)))));
  const float sig = g < 0.0f ? float(R(1.0f - y)) : y;
  const float gs = float(R(g * sig));
  return float(R(gs * u));
}
