// mlx2 EXL3 tile reconstruction over the Sushi-derived header (header_mcg_w16):
// decoded codebook values (exact halves) of output tiles [OT0, OT0 + OS/16) of
// the listed experts, as inner[u, i, o] with i the input and o the slice
// column (TRANS: inner[u, o, i]). One simdgroup per 16x16 tile; lane l holds the tile's weights at
// rows prow, prow+1, prow+8, prow+9 and columns pcol, pcol+8, the mapping the
// GEMV kernels and the host decoder use.
const uint t = uint(threadgroup_position_in_grid.x);
const uint tk = uint(threadgroup_position_in_grid.y);
const uint u = uint(threadgroup_position_in_grid.z);
const uint lane = uint(thread_index_in_simdgroup);
constexpr uint TILE = 16u;
constexpr uint N = uint(NHW);
constexpr uint PACKED_W = N / 2u;
constexpr uint OT = uint(ODIM) / TILE;
constexpr uint OS = uint(OSLICE);
constexpr size_t EXPERT = (size_t)(uint(IDIM) / TILE) * (size_t)OT * (size_t)N;
const uint ot = uint(OT0) + t;
const device uint *words = (const device uint *)(trellis + (size_t)experts[u] * EXPERT)
    + ((size_t)tk * (size_t)OT + ot) * PACKED_W;
const auto merged = exl3_lane<N>(words, lane);
const uint sh[8] = {exl3_lane_sh(N, 0u), exl3_lane_sh(N, 1u), exl3_lane_sh(N, 2u), exl3_lane_sh(N, 3u),
                    exl3_lane_sh(N, 4u), exl3_lane_sh(N, 5u), exl3_lane_sh(N, 6u), exl3_lane_sh(N, 7u)};
float2 w[4];
for (uint p = 0u; p < 4u; p++) {
  const uint2 cw = uint2(exl3_lane_word(merged, sh[p * 2u]), exl3_lane_word(merged, sh[p * 2u + 1u])) & uint2(0xffffu);
  w[p] = exl3_decode2(cw);
}
const uint prow = (lane & 3u) * 2u;
const uint pcol = lane >> 2u;
// Element (row r, column c) of this tile: input tk*16 + r, slice column t*16 + c.
const size_t i0 = (size_t)tk * TILE;
const size_t o0 = (size_t)t * TILE;
const size_t ub = (size_t)u * (size_t)(IDIM) * OS;
#define AT(r, c) (TRANS ? ub + (o0 + (c)) * (size_t)(IDIM) + i0 + (r) : ub + (i0 + (r)) * OS + o0 + (c))
inner[AT(prow, pcol)] = half(w[0].x);
inner[AT(prow + 1u, pcol)] = half(w[0].y);
inner[AT(prow + 8u, pcol)] = half(w[1].x);
inner[AT(prow + 9u, pcol)] = half(w[1].y);
inner[AT(prow, pcol + 8u)] = half(w[2].x);
inner[AT(prow + 1u, pcol + 8u)] = half(w[2].y);
inner[AT(prow + 8u, pcol + 8u)] = half(w[3].x);
inner[AT(prow + 9u, pcol + 8u)] = half(w[3].y);
#undef AT
