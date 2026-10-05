#pragma once
#include "vendor/stock_nax_source.h"
namespace mlx2::paged_kv {
inline constexpr const char* kPrefillLongNAX = R"metal(
using namespace mlx::steel;
struct LongNAXParams { int qL,kL,NQ,NK,NQ_aligned,NK_aligned,qL_rem,kL_rem,qL_off; float scale; ulong Q_strides[3],O_strides[3]; };
struct LongMaxOp { template<typename T> METAL_FUNC static T apply(T x,T y){return max(x,y);} };
struct LongSumOp { template<typename T> METAL_FUNC static T apply(T x,T y){return x+y;} };
struct LongMulOp { template<typename T> METAL_FUNC static T apply(T x,T y){return x*y;} };
struct LongExpSubOp { template<typename T> METAL_FUNC static T apply(T x,T y){return fast::exp2(x-y);} };
// clang-format off
[[kernel,max_total_threads_per_threadgroup(256)]] void mlx2_prefill_long_nax(
    const device bfloat* Q [[buffer(0)]], const device bfloat* K [[buffer(1)]],
    const device bfloat* V [[buffer(2)]], const device uchar* dependency [[buffer(3)]],
    const device uint* row_span [[buffer(4)]], const device uint* row_begin [[buffer(5)]],
    const device uint* query_start [[buffer(6)]], const device uint* retained [[buffer(7)]],
    const device uint* first_block [[buffer(8)]], const device uint* table_begin [[buffer(9)]],
    const device uint* window [[buffer(10)]], const device uint* pages [[buffer(11)]],
    device bfloat* O [[buffer(12)]], constant float& scale [[buffer(13)]],
    constant uint& heads [[buffer(14)]], constant uint& kv_heads [[buffer(15)]],
    constant uint& dim [[buffer(16)]], constant ulong& qrs [[buffer(17)]],
    constant ulong& qhs [[buffer(18)]], constant ulong& qds [[buffer(19)]],
    constant uint& span_count [[buffer(20)]], constant uint& total_rows [[buffer(21)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  (void)lid; (void)dependency; (void)row_span; (void)retained;
  (void)window; (void)span_count; (void)dim; (void)qds;
  constexpr int BQ=64,BK=32,BD=256,WM=4,WN=2;
  const uint span=tid.z;
  const uint begin=row_begin[span];
  const uint count=(span+1<span_count?row_begin[span+1]:total_rows)-begin;
  if(tid.x*BQ>=count)return;
  if(dependency[0]!=1){
    const uint valid=min(uint(BQ),count-tid.x*BQ);
    const uint worker=simd_group_id*32+simd_lane_id;
    for(uint item=worker;item<valid*BD;item+=256)
      O[(ulong(begin+tid.x*BQ+item/BD)*heads+tid.y)*BD+item%BD]=bfloat(0);
    return;
  }
  const uint end=query_start[span]+count;
  const uint kv_head_idx=tid.y/(heads/kv_heads);
  const uint table=table_begin[span];
  LongNAXParams params;
  params.scale=scale;params.qL=count;params.kL=end;params.qL_off=query_start[span];
  params.NQ=(count+BQ-1)/BQ;params.NK=(end+BK-1)/BK;
  params.NQ_aligned=count/BQ;params.NK_aligned=end/BK;
  params.qL_rem=count%BQ;params.kL_rem=end%BK;
  params.Q_strides[2]=qrs;params.O_strides[2]=ulong(heads)*BD;
  const bool align_Q=(params.qL_rem==0),align_K=(params.kL_rem==0);
  Q+=(ulong(begin)+ulong(tid.x)*BQ)*qrs+ulong(tid.y)*qhs;
  O+=(ulong(begin)+ulong(tid.x)*BQ)*heads*BD+ulong(tid.y)*BD;
  const float scale2 = params.scale * 1.44269504089f;
  // Prepare MMA tiles
  constexpr short kU = 16;

  // The WM simdgroups along the first warp dimension split the Q sequence;
  // the WN simdgroups along the second split the head dim. The exchange
  // below reduces exactly one peer, so WN is fixed at 2.
  static_assert(WN == 2, "The head-dim split kernel needs WN == 2");
  constexpr int kNWarps = WM;
  static_assert(
      BQ >= (kNWarps * kU) && BQ % (kNWarps * kU) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * kU);
  // HeadDim frags over the full head dim
  constexpr int TD = BD / kU;
  // KV seq frags per warp
  constexpr short TK = BK / kU;

  static_assert(TQ == 1, "Check TQ");
  static_assert(TD % WN == 0, "The head dim must split evenly across WN");

  // HeadDim frags / columns owned by each of the WN simdgroups of a row group
  constexpr int TDh = TD / WN;
  constexpr int BDh = BD / WN;

  static_assert(TDh % 2 == 0, "P@V accumulates output fragments in pairs");
  static_assert(TK % 2 == 0, "S fragments are exchanged pair by pair");

  const short row_group = simd_group_id / WN;
  const short d_half = simd_group_id % WN;

  using otile_t = NAXTile<float, TQ, TDh>;
  otile_t Otile;
  Otile.clear();

  const short tm = kU * TQ * row_group;
  Q += tm * int(params.Q_strides[2]) + d_half * BDh;
  O += tm * int(params.O_strides[2]) + d_half * BDh;

  constexpr short kRowsPT = otile_t::kRowsPerThread;

  metal::vec<float, kRowsPT> max_score;
  metal::vec<float, kRowsPT> sum_score{0};

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = -3.402823466e38f;
  }



  int kb_lim = params.NK;
  int kb_min_causal = params.NK;

  {
    int q_max = (tid.x + 1) * BQ + params.qL_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params.NK, kb_lim);

    int q_min = tid.x * BQ + params.qL_off;
    q_min = max(0, q_min);
    kb_min_causal = (q_min / BK);
  }

  const bool is_last_q = int(tid.x) == (params.NQ_aligned);
  const short lim_rows_q = params.qL_rem - tm;

  using stile_t = NAXTile<float, TQ, TK>;
  constexpr short kEPF = stile_t::NAXFrag_t::kElemsPerFrag;

  // One slot per (row group, half): a fragment pair in per-lane-linear
  // layout. Both halves share the fragment-to-lane mapping, so the
  // exchange needs no coordinate math.
  threadgroup float s_xchg[WM][WN][2 * kEPF * 32];

  // Keep the simdgroup's Q half resident in registers for the whole KV
  // loop: TDh fragments of T are cheap next to the accumulators.
  NAXTile<bfloat, 1, 1> Qtiles[TDh];
  STEEL_PRAGMA_UNROLL
  for (short id = 0; id < TDh; id++) {
    const int Q_load_off = id * kU;
    if (!align_Q && is_last_q) {
      Qtiles[id].load_rows(
          Q + Q_load_off, int(params.Q_strides[2]), lim_rows_q);
    } else {
      Qtiles[id].load(Q + Q_load_off, int(params.Q_strides[2]));
    }
  }

  const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  // Loop over KV seq length
  for (int kb = 0; kb < kb_lim; kb++) {
    const int is_last_k = (kb == (params.NK_aligned));
    // Cold origin zero, page64 and BK32 put the entire KV tile in one page.
    // Bind a tile-local contiguous row base so the stock fragment loader can
    // issue vector loads rather than eight scalar page gathers per fragment.
    const uint page=pages[table+uint(kb)/2];

    stile_t Stile;
    Stile.clear();

    // S = Q @ K.T, this half of D only, exchanged pair by pair.
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik += 2) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TDh; id++) {
        NAXTile<bfloat, 2, 1> Ktile;
        const device bfloat* ktile = K +
            (ulong(page)*kv_heads*64 + ulong(kv_head_idx)*64 +
             ulong(kb%2)*BK)*BD + ulong(d_half)*BDh + ulong(id)*kU;
        if (!align_K && is_last_k)
          Ktile.load_rows(ktile, BD, short(params.kL_rem));
        else
          Ktile.load(ktile, BD);

        stile_t::NAXFrag_t::mma(
            Stile.frag_at(0, ik),
            Stile.frag_at(0, ik + 1),
            Qtiles[id].frag_at(0, 0),
            metal::false_type{},
            Ktile.frag_at(0, 0),
            Ktile.frag_at(1, 0),
            metal::true_type{});
      }

      // Exchange the partial pair and reduce.
      threadgroup float* slot = s_xchg[row_group][d_half];
      thread auto& s0 = Stile.frag_at(0, ik);
      thread auto& s1 = Stile.frag_at(0, ik + 1);
      const short base = short(simd_lane_id) * (2 * kEPF);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kEPF; i++) {
        slot[base + i] = s0[i];
        slot[base + kEPF + i] = s1[i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const threadgroup float* peer = s_xchg[row_group][1 - d_half];
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kEPF; i++) {
        s0[i] += peer[base + i];
        s1[i] += peer[base + kEPF + i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Scale S
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= float(scale2);
    }

    // Mask out length sequence
    if (!align_K && is_last_k) {
      constexpr auto neg_inf = -3.402823466e38f;

      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        const short col_pos = ik * kU + sn;
        thread auto& fg = Stile.frag_at(0, ik);

        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
            const auto loc = ii * stile_t::kFragThrCols + jj;
            fg[loc] = ((col_pos + jj) < params.kL_rem) ? fg[loc] : neg_inf;
          }
        }
      }
    }

    // Mask out if causal
    if (kb >= kb_min_causal) {
      constexpr auto neg_inf = -3.402823466e38f;

      const int base_row = tid.x * BQ + params.qL_off + tm;
      const int base_col = kb * BK;

      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        thread auto& fg = Stile.frag_at(0, ik);

        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
            const auto r = base_row + ii * stile_t::kFragRowsJump + sm;
            const auto c = base_col + ik * kU + jj + sn;
            const auto loc = ii * stile_t::kFragThrCols + jj;
            fg[loc] = (r < c) ? neg_inf : fg[loc];
          }
        }
      }
    }

    // Other masking as needed


    // Do softmax (redundantly per half; the row statistics are cheap)
    metal::vec<float, kRowsPT> new_max;
    metal::vec<float, kRowsPT> factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    Stile.template row_reduce<LongMaxOp>(new_max);
    Stile.template row_bin_op<LongExpSubOp>(new_max);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i];
    }

    Stile.template row_reduce<LongSumOp>(sum_score);

    Otile.template row_bin_op<LongMulOp>(factor);

    simdgroup_barrier(mem_flags::mem_none);

    // O = P @ V, this half of Dv only.
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TDh; id += 2) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        NAXTile<bfloat, 1, 2> Vtile;

        const device bfloat* vtile = V +
            (ulong(page)*kv_heads*64 + ulong(kv_head_idx)*64 +
             ulong(kb%2)*BK + ulong(ik)*kU)*BD +
            ulong(d_half)*BDh + ulong(id)*kU;
        const short valid_v = short(params.kL_rem - ik*kU);
        if (!align_K && is_last_k && valid_v < kU)
          Vtile.load_rows(vtile, BD, max(short(0), valid_v));
        else
          Vtile.load(vtile, BD);

        otile_t::NAXFrag_t::mma(
            Otile.frag_at(0, id),
            Otile.frag_at(0, id + 1),
            Stile.frag_at(0, ik),
            metal::false_type{},
            Vtile.frag_at(0, 0),
            Vtile.frag_at(0, 1),
            metal::false_type{});
      }
    }

    // Next block
  }

  // Normalize output
  threadgroup_barrier(mem_flags::mem_none);

  metal::vec<float, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = 1.f / sum_score[i];
  }

  Otile.template row_bin_op<LongMulOp>(rcp);

  if (!align_Q && is_last_q) {
    if (lim_rows_q <= 0)
      return;
    Otile.store_rows(O, int(params.O_strides[2]), lim_rows_q);
  } else {
    Otile.store(O, int(params.O_strides[2]));
  }
}

)metal";
inline std::string prefill_long_nax_source(){return std::string(kPrefillMMA)+kStockNAX+kPrefillLongNAX;}
}
