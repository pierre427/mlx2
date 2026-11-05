// mlx2 EXL3 routed-MoE decode tail, one thread per (token row, column):
//   y   = sum_k T(T(d[row, k]) * scores[row, k])     (k ascending, float sum, then T)
//   out = T(y + T(sigmoid_T(gl[row]) * shared[row]))
// d is the fused down output (half, one row per routed slot); T is the trunk
// dtype. Each op rounds to T as the stock tail's element ops do (astype,
// multiply, sum, sigmoid, multiply, add); the sum's float accumulation order
// is fixed, so this is close to the stock tail, not claimed bit-exact. Every
// row runs the same arithmetic: row-invariant at any width.
const uint col = uint(thread_position_in_grid.x);
const uint row = uint(thread_position_in_grid.y);
if (col >= uint(D)) {
  return;
}
float acc = 0.0f;
for (uint k = 0u; k < uint(K); k++) {
  const float yk = float(T(d[((size_t)row * uint(K) + k) * (size_t)(D) + col]));
  acc += float(T(yk * float(scores[row * uint(K) + k])));
}
const float y = float(T(acc));
const float g = float(gl[row]);
const float s = float(T(1.0f / (1.0f + metal::exp(-metal::abs(g)))));
const float sig = g < 0.0f ? float(T(1.0f - s)) : s;
const float p = float(T(sig * float(shared[(size_t)row * (size_t)(D) + col])));
out[(size_t)row * (size_t)(D) + col] = T(y + p);
