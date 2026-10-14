static inline half2 exl3_pairh(uint2 cw) {
  const uint2 r = ((cw * uint2(0xCBAC1FEDu)) & uint2(0x8FFF8FFFu)) ^ uint2(0x3B603B60u);
  const half4 h = as_type<half4>(r);
  return half2(h.x + h.y, h.z + h.w);
}
static inline float2 exl3_decode2(uint2 cw) { return float2(exl3_pairh(cw)); }
static inline float exl3_decode1(uint cw) { return exl3_decode2(uint2(cw, 0u)).x; }
// Eight weights span N/2 whole bits, so weight j of an eight-weight group
// ends exl3_end(N, j) bits past the group's first bit.
static inline constexpr uint exl3_end(uint N, uint j) { return ((j + 1u) * N) >> 4u; }
// The 64 stream bits ending at bit `end` of a tile of `nwords` words (the tile
// wraps). Two words carry only the last 32 + end % 32 of them; `full` reads a third.
static inline ulong exl3_window(const device uint *words, uint end, uint nwords, bool full) {
  const uint last = (end - 1u) >> 5u;
  const uint prev = last == 0u ? nwords - 1u : last - 1u;
  const uint s = (0u - end) & 31u;
  ulong bits = (((ulong)words[prev] << 32u) | (ulong)words[last]) >> s;
  if (full) bits |= ((ulong)words[prev == 0u ? nwords - 1u : prev - 1u] << 32u) << (32u - s);
  return bits;
}
// Lane l reads weights 8l..8l+7 from the bits ending at its group's last bit; weight
// j's codeword ends exl3_lane_sh(N, j) bits before that. The least nonzero end % 32 is
// W's lowest set bit, so a longer first codeword takes the third word.
static inline constexpr uint exl3_lane_sh(uint N, uint j) { return N / 2u - exl3_end(N, j); }
template<bool Wide> struct exl3_lane_type {
  using type = ulong;
  static inline type read(const device uint *words, uint end, uint W, bool full) {
    return exl3_window(words, end, W, full);
  }
};
template<> struct exl3_lane_type<true> {
  using type = ulong2;
  static inline type read(const device uint *words, uint end, uint W, bool full) {
    return ulong2(exl3_window(words, end, W, full), exl3_window(words, end - 32u, W, true));
  }
};
template<uint N> using exl3_lane_bits = typename exl3_lane_type<(exl3_lane_sh(N, 0u) + 16u > 64u)>::type;
static inline uint exl3_lane_word(ulong bits, uint sh) { return uint(bits >> sh); }
static inline uint exl3_lane_word(ulong2 bits, uint sh) {
  return sh <= 48u ? uint(bits.x >> sh) : uint(bits.y >> (sh - 32u));
}
template<uint N>
static inline exl3_lane_bits<N> exl3_lane(const device uint *words, uint lane) {
  constexpr uint W = N / 2u;
  constexpr uint low = W & (0u - W);
  const uint end = W * lane + W;
  return exl3_lane_type<(exl3_lane_sh(N, 0u) + 16u > 64u)>::read(words, end, W, exl3_lane_sh(N, 0u) + 16u > 32u + (low < 32u ? low : 32u));
}
// Weight t's 16-bit codeword is the window ending at floor((t+1)*K), with
// K = N/16; the pair (t0, t0+1) shares one 32-bit funnel read.
struct exl3_win { uint i0; uint i1; uint sh; uint fresh; };
static inline exl3_win exl3_pair_window(uint t0, uint N) {
  const uint e0 = ((t0 + 1u) * N) >> 4u;
  const uint e1 = ((t0 + 2u) * N) >> 4u;
  const uint b0 = e0 + 16u * N - 16u;
  const uint b2 = e1 + 16u * N;
  const uint j0 = b0 >> 5u;
  const uint j1 = (b2 - 1u) >> 5u;
  exl3_win w;
  w.i0 = j0 % (N >> 1u);
  w.i1 = j1 % (N >> 1u);
  w.sh = (j1 + 1u) * 32u - b2;
  w.fresh = e1 - e0;
  return w;
}

