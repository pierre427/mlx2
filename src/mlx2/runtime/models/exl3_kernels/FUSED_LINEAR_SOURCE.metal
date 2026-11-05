// mlx2 fused EXL3 dense linear over the Sushi-derived header (header_mcg_w16):
// prepare, trellis inner product and finish in one launch. One threadgroup
// (eight simdgroups) computes one row's 128-column output block:
//   1. the row's prepared input, H128(x * suh) rounded to half, into
//      threadgroup memory (PREPARE_SOURCE's arithmetic);
//   2. simdgroup s accumulates output tile 8*block + s over every input
//      tile in order, two tiles' words loaded ahead;
//   3. H128 of the block's 128 inner sums times svh, rounded to half, then
//      to the output type (FINISH_SOURCE's arithmetic, then the cast).
// Every row runs the same threadgroup program: row-invariant at any width.
threadgroup half prepared[uint(IDIM)];
threadgroup float inner[128];
const uint block = uint(threadgroup_position_in_grid.x);
const uint row = uint(threadgroup_position_in_grid.y);
const uint sg = uint(simdgroup_index_in_threadgroup);
const ushort lane = ushort(thread_index_in_simdgroup);
constexpr uint TILE = 16u;
constexpr uint N = uint(NHW);
constexpr uint PACKED_W = N / 2u;
constexpr uint IT = uint(IDIM) / TILE;
constexpr uint OT = uint(ODIM) / TILE;
constexpr uint SGS = 8u;
constexpr float sc = 0.08838834764831845f;
{
  const size_t xr = (size_t)row * (size_t)(IDIM);
  for (uint b = sg; b < uint(IDIM) / 128u; b += SGS) {
    const uint base = b * 128u;
    float4 v = float4(
      float(x[xr + base + lane]) * float(suh[base + lane]),
      float(x[xr + base + lane + 32u]) * float(suh[base + lane + 32u]),
      float(x[xr + base + lane + 64u]) * float(suh[base + lane + 64u]),
      float(x[xr + base + lane + 96u]) * float(suh[base + lane + 96u]));
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
    prepared[base + lane] = half((s0 + s2) * sc);
    prepared[base + lane + 32u] = half((s1 + s3) * sc);
    prepared[base + lane + 64u] = half((s0 - s2) * sc);
    prepared[base + lane + 96u] = half((s1 - s3) * sc);
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
{
  const uint ot = block * 8u + sg;
  const uint prow = (uint(lane) & 3u) * 2u;
  const uint pcol = uint(lane) >> 2u;
  const uint sh[8] = {exl3_lane_sh(N, 0u), exl3_lane_sh(N, 1u), exl3_lane_sh(N, 2u), exl3_lane_sh(N, 3u),
                      exl3_lane_sh(N, 4u), exl3_lane_sh(N, 5u), exl3_lane_sh(N, 6u), exl3_lane_sh(N, 7u)};
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  const device uint *words = (const device uint *)trellis + (size_t)ot * PACKED_W;
  constexpr size_t STEP = (size_t)OT * PACKED_W;
  for (uint tk = 0u; tk < IT; tk += 2u) {
    const auto m0 = exl3_lane<N>(words + (size_t)tk * STEP, uint(lane));
    const auto m1 = exl3_lane<N>(words + (size_t)(tk + 1u) * STEP, uint(lane));
    for (uint u = 0u; u < 2u; u++) {
      const auto merged = u == 0u ? m0 : m1;
      const uint t = tk + u;
      float2 w[4];
      for (uint p = 0u; p < 4u; p++) {
        const uint2 cw = uint2(exl3_lane_word(merged, sh[p * 2u]), exl3_lane_word(merged, sh[p * 2u + 1u])) & uint2(0xffffu);
        w[p] = exl3_decode2(cw);
      }
      const float in0 = float(prepared[t * TILE + prow]);
      const float in1 = float(prepared[t * TILE + prow + 1u]);
      const float in2 = float(prepared[t * TILE + prow + 8u]);
      const float in3 = float(prepared[t * TILE + prow + 9u]);
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
  float clo = (acc[0] + acc[1]) + (acc[2] + acc[3]);
  float chi = (acc[4] + acc[5]) + (acc[6] + acc[7]);
  clo += simd_shuffle_xor(clo, 1u);
  chi += simd_shuffle_xor(chi, 1u);
  clo += simd_shuffle_xor(clo, 2u);
  chi += simd_shuffle_xor(chi, 2u);
  if ((uint(lane) & 3u) == 0u) {
    inner[sg * 16u + pcol] = clo;
    inner[sg * 16u + pcol + 8u] = chi;
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0u) {
  const uint base = block * 128u;
  float4 v = float4(inner[lane], inner[lane + 32u], inner[lane + 64u], inner[lane + 96u]);
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
  const size_t yb = (size_t)row * (size_t)(ODIM) + base;
  y[yb + lane] = OUT(float(half((s0 + s2) * sc * float(svh[base + lane]))));
  y[yb + lane + 32u] = OUT(float(half((s1 + s3) * sc * float(svh[base + lane + 32u]))));
  y[yb + lane + 64u] = OUT(float(half((s0 - s2) * sc * float(svh[base + lane + 64u]))));
  y[yb + lane + 96u] = OUT(float(half((s1 - s3) * sc * float(svh[base + lane + 96u]))));
}
