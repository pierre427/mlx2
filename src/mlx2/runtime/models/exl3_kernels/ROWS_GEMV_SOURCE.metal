// mlx2 row-blocked EXL3 GEMV over the Sushi-derived header (header_mcg_w16).
// Block b covers rows order[b*ROWS .. b*ROWS+ROWS) of the call, sorted by
// expert (DENSE: rows b*ROWS.. of one expert, no order/eids reads). Each
// 16-column output tile is decoded once per expert run inside the block and
// applied to every row of that run. A row's own accumulators,
// input distribution and reduction order never depend on the other rows,
// so one law serves every width. Rows are read and written in caller order.
threadgroup float partial[4u * 16u * uint(ROWS)];
const uint ot = uint(threadgroup_position_in_grid.x);
const uint blk = uint(threadgroup_position_in_grid.y);
const uint sg = uint(simdgroup_index_in_threadgroup);
const uint lane = uint(thread_index_in_simdgroup);
const uint lid = uint(thread_index_in_threadgroup);
constexpr uint TILE = 16u;
constexpr uint N = uint(NHW);
constexpr uint PACKED_W = N / 2u;
constexpr uint IT = uint(IDIM) / TILE;
constexpr uint OT = uint(ODIM) / TILE;
constexpr uint SGS = 4u;
constexpr uint R = uint(ROWS);
const uint total = uint(order_shape[0]);
const uint first = blk * R;
const uint count = min(R, total - first);
const uint prow = (lane & 3u) * 2u;
const uint pcol = lane >> 2u;
const uint sh[8] = {exl3_lane_sh(N, 0u), exl3_lane_sh(N, 1u), exl3_lane_sh(N, 2u), exl3_lane_sh(N, 3u),
                    exl3_lane_sh(N, 4u), exl3_lane_sh(N, 5u), exl3_lane_sh(N, 6u), exl3_lane_sh(N, 7u)};
uint row_of[R];
uint eid_of[R];
for (uint r = 0u; r < R; r++) {
  const uint row = DENSE ? first + min(r, count - 1u) : order[first + min(r, count - 1u)];
  row_of[r] = row;
  eid_of[r] = DENSE ? 0u : eids[row];
}
float acc[R][8];
for (uint r = 0u; r < R; r++) {
  for (uint s = 0u; s < 8u; s++) {
    acc[r][s] = 0.0f;
  }
}
constexpr size_t EXPERT = (size_t)IT * (size_t)OT * (size_t)N;
for (uint tk = sg; tk < IT; tk += SGS) {
  float2 w[4];
  uint decoded = 0xffffffffu;
  for (uint r = 0u; r < R; r++) {
    if (r < count) {
      if (eid_of[r] != decoded) {
        decoded = eid_of[r];
        const device uint *words = (const device uint *)(trellis + (size_t)decoded * EXPERT)
            + ((size_t)tk * (size_t)OT + ot) * PACKED_W;
        const auto merged = exl3_lane<N>(words, lane);
        for (uint p = 0u; p < 4u; p++) {
          const uint2 cw = uint2(exl3_lane_word(merged, sh[p * 2u]), exl3_lane_word(merged, sh[p * 2u + 1u])) & uint2(0xffffu);
          w[p] = exl3_decode2(cw);
        }
      }
      // Each lane reads its four tile inputs (rows prow, prow+1, prow+8,
      // prow+9); independent loads beat shuffles off one load here (v2).
      const size_t xb = (size_t)row_of[r] * (size_t)(IDIM) + (size_t)tk * TILE;
      const float in0 = float(x[xb + prow]);
      const float in1 = float(x[xb + prow + 1u]);
      const float in2 = float(x[xb + prow + 8u]);
      const float in3 = float(x[xb + prow + 9u]);
      acc[r][0] = fma(in0, w[0].x, acc[r][0]);
      acc[r][1] = fma(in1, w[0].y, acc[r][1]);
      acc[r][2] = fma(in2, w[1].x, acc[r][2]);
      acc[r][3] = fma(in3, w[1].y, acc[r][3]);
      acc[r][4] = fma(in0, w[2].x, acc[r][4]);
      acc[r][5] = fma(in1, w[2].y, acc[r][5]);
      acc[r][6] = fma(in2, w[3].x, acc[r][6]);
      acc[r][7] = fma(in3, w[3].y, acc[r][7]);
    }
  }
}
for (uint r = 0u; r < R; r++) {
  if (r < count) {
    float clo = (acc[r][0] + acc[r][1]) + (acc[r][2] + acc[r][3]);
    float chi = (acc[r][4] + acc[r][5]) + (acc[r][6] + acc[r][7]);
    clo += simd_shuffle_xor(clo, 1u);
    chi += simd_shuffle_xor(chi, 1u);
    clo += simd_shuffle_xor(clo, 2u);
    chi += simd_shuffle_xor(chi, 2u);
    if ((lane & 3u) == 0u) {
      partial[(r * SGS + sg) * 16u + pcol] = clo;
      partial[(r * SGS + sg) * 16u + pcol + 8u] = chi;
    }
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lid < 16u * R) {
  const uint r = lid >> 4u;
  if (r < count) {
    float sum = 0.0f;
    for (uint g = 0u; g < SGS; g++) {
      sum += partial[(r * SGS + g) * 16u + (lid & 15u)];
    }
    y[(size_t)row_of[r] * (size_t)(ODIM) + ot * TILE + (lid & 15u)] = sum;
  }
}
