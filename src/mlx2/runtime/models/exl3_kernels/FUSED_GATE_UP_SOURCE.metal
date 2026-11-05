// mlx2 fused EXL3 gate/up/SwiGLU for one routed slot (or one dense row with
// TOPK=1 and expert 0) per threadgroup and 128-column output block:
//   1. the token row's two prepared inputs, H128(x * suh) rounded to half, for
//      the gate and the up projection of the slot's expert (PREPARE arithmetic);
//   2. 32 simdgroups: projection (sg >> 4), output tile ((sg >> 1) & 7) and
//      input half (sg & 1); the halves add in a fixed order;
//   3. FINISH arithmetic for both 128-column blocks (H128, svh, half);
//   4. SwiGLU in element type T (half for routed experts, the trunk dtype for
//      dense MLPs): g and u rounded to half as FINISH writes them, cast to T,
//      then MLX's sigmoid form (1 / (1 + exp(-|g|)), mirrored for g < 0),
//      g * sigmoid and the product with u, each rounded to T. Close to MLX's
//      nn.silu(g) * u, not claimed bit-exact.
// Each slot runs the same threadgroup program, so slots are row-invariant.
threadgroup half prep[2u * uint(IDIM)];
threadgroup float part[2u * 8u * 2u * 16u];
threadgroup float val[2u * 128u];
const uint block = uint(threadgroup_position_in_grid.x);
const uint slot = uint(threadgroup_position_in_grid.y);
const uint sg = uint(simdgroup_index_in_threadgroup);
const ushort lane = ushort(thread_index_in_simdgroup);
const uint lid = uint(thread_index_in_threadgroup);
constexpr uint TILE = 16u;
constexpr uint IT = uint(IDIM) / TILE;
constexpr uint OT = uint(ODIM) / TILE;
constexpr uint BLOCKS = uint(IDIM) / 128u;
constexpr float sc = 0.08838834764831845f;
const uint e = uint(slots[slot]);
const size_t xr = (size_t)(slot / uint(TOPK)) * (size_t)(IDIM);
for (uint b = sg; b < 2u * BLOCKS; b += 32u) {
  const uint proj = b / BLOCKS;
  const uint base = (b % BLOCKS) * 128u;
  const device half *suh = (proj == 0u ? gsuh : usuh) + (size_t)e * (size_t)(IDIM);
  float4 v = float4(
    float(x[xr + base + lane]) * float(suh[base + lane]),
    float(x[xr + base + lane + 32u]) * float(suh[base + lane + 32u]),
    float(x[xr + base + lane + 64u]) * float(suh[base + lane + 64u]),
    float(x[xr + base + lane + 96u]) * float(suh[base + lane + 96u]));
  const float4 s = exl3_h128_lane(v, lane);
  threadgroup half *dst = prep + proj * uint(IDIM) + base;
  dst[lane] = half(s.x * sc);
  dst[lane + 32u] = half(s.y * sc);
  dst[lane + 64u] = half(s.z * sc);
  dst[lane + 96u] = half(s.w * sc);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
{
  const uint proj = sg >> 4u;
  const uint tile = (sg >> 1u) & 7u;
  const uint half_k = sg & 1u;
  const uint ot = block * 8u + tile;
  const uint tk0 = half_k * (IT / 2u);
  const uint tk1 = tk0 + IT / 2u;
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  threadgroup const half *src = prep + proj * uint(IDIM);
  if (proj == 0u) {
    constexpr uint W = uint(NG) / 2u;
    const device uint *words = (const device uint *)(gt + (size_t)e * IT * OT * uint(NG)) + (size_t)ot * W;
    exl3_acc_tiles<uint(NG)>(words, (size_t)OT * W, src, tk0, tk1, uint(lane), acc);
  } else {
    constexpr uint W = uint(NU) / 2u;
    const device uint *words = (const device uint *)(ut + (size_t)e * IT * OT * uint(NU)) + (size_t)ot * W;
    exl3_acc_tiles<uint(NU)>(words, (size_t)OT * W, src, tk0, tk1, uint(lane), acc);
  }
  float clo = (acc[0] + acc[1]) + (acc[2] + acc[3]);
  float chi = (acc[4] + acc[5]) + (acc[6] + acc[7]);
  clo += simd_shuffle_xor(clo, 1u);
  chi += simd_shuffle_xor(chi, 1u);
  clo += simd_shuffle_xor(clo, 2u);
  chi += simd_shuffle_xor(chi, 2u);
  if ((uint(lane) & 3u) == 0u) {
    const uint pcol = uint(lane) >> 2u;
    part[((proj * 8u + tile) * 2u + half_k) * 16u + pcol] = clo;
    part[((proj * 8u + tile) * 2u + half_k) * 16u + pcol + 8u] = chi;
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lid < 256u) {
  const uint proj = lid >> 7u;
  const uint col = lid & 127u;
  const uint at = ((proj * 8u + (col >> 4u)) * 2u) * 16u + (col & 15u);
  val[lid] = part[at] + part[at + 16u];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg < 2u) {
  const uint base = block * 128u;
  const device half *svh = (sg == 0u ? gsvh : usvh) + (size_t)e * (size_t)(ODIM) + base;
  threadgroup float *row = val + sg * 128u;
  const float4 s = exl3_h128_lane(float4(row[lane], row[lane + 32u], row[lane + 64u], row[lane + 96u]), lane);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  row[lane] = float(half(s.x * sc * float(svh[lane])));
  row[lane + 32u] = float(half(s.y * sc * float(svh[lane + 32u])));
  row[lane + 64u] = float(half(s.z * sc * float(svh[lane + 64u])));
  row[lane + 96u] = float(half(s.w * sc * float(svh[lane + 96u])));
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lid < 128u) {
  // Float arithmetic, rounded to T after each op (MLX's element ops round
  // their float result the same way; bfloat has no native exp here).
  const float g = float(T(half(val[lid])));
  const float u = float(T(half(val[128u + lid])));
  const float y = float(T(1.0f / (1.0f + metal::exp(-metal::abs(g)))));
  const float sig = g < 0.0f ? float(T(1.0f - y)) : y;
  const float gs = float(T(g * sig));
  h[(size_t)slot * (size_t)(ODIM) + block * 128u + lid] = T(gs * u);
}
