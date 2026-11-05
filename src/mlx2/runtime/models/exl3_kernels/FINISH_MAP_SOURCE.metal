// mlx2 FINISH with an output row map and dtype: slot s finishes against
// expert slots[s] (FINISH_SOURCE's arithmetic, rounded to half) and writes
// row dst[s] as OUT, so an expert-sorted result is un-permuted and cast in
// the same launch.
uint block = uint(threadgroup_position_in_grid.x);
uint slot = uint(threadgroup_position_in_grid.y);
ushort lane = thread_index_in_simdgroup;
const uint eid = uint(slots[slot]);
const uint base = block * 128u;
const size_t xb = (size_t)slot * (size_t)(ODIM) + base;
const size_t sb = (size_t)eid * (size_t)(ODIM) + base;
const size_t yb = (size_t)dst[slot] * (size_t)(ODIM) + base;
float4 v = float4(
  float(inner[xb + lane]),
  float(inner[xb + lane + 32u]),
  float(inner[xb + lane + 64u]),
  float(inner[xb + lane + 96u]));
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
const float sc = 0.08838834764831845f;
y[yb + lane] = OUT(float(half((s0 + s2) * sc * float(svh[sb + lane]))));
y[yb + lane + 32u] = OUT(float(half((s1 + s3) * sc * float(svh[sb + lane + 32u]))));
y[yb + lane + 64u] = OUT(float(half((s0 - s2) * sc * float(svh[sb + lane + 64u]))));
y[yb + lane + 96u] = OUT(float(half((s1 - s3) * sc * float(svh[sb + lane + 96u]))));
