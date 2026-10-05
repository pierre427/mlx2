#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include "vendor/steel_attn_mma_source.h"
namespace mlx2::paged_kv {
inline bool prefill_matrix_requested(const char* flag) {
  if (!flag || std::string(flag) == "0") return false;
  if (std::string(flag) == "1") return true;
  throw std::invalid_argument("prefill matrix selector must be 0 or 1");
}
inline void validate_long_nax_q1_handoff(bool long_requested, bool multiquery,
    bool stock_long_b2, bool bounded_scalar_b1) {
  if (long_requested && !multiquery && !stock_long_b2 && !bounded_scalar_b1)
    throw std::invalid_argument("long fused NAX selector has no admitted Q1 successor");
}
inline bool bounded_long_scalar_b1(bool stock_long_requested, bool bf16,
    uint32_t dim, uint32_t heads, uint32_t kvheads, uint32_t rows,
    const std::vector<std::vector<uint32_t>>& spans, uint32_t split_partition) {
  if (!stock_long_requested || !bf16 || dim != 256 || heads != 24 ||
      kvheads != 4 || rows != 1 || spans.size() != 1 ||
      spans[0].size() != 8 || split_partition != 0)
    return false;
  const auto& s = spans[0];
  return s[0] == 1 && static_cast<uint64_t>(s[1]) + 1 == s[2] &&
      s[2] > 1024 && s[2] <= 8192 && s[3] == 0 && s[4] == 0 &&
      s[5] == 0 && s[6] == (s[2] + 63) / 64 && s[7] == 0;
}
struct PrefillMatrixPlan { uint32_t max_tiles=0, spans=0; bool exact_nax=false, long_nax=false, long_n20=false; uint64_t scratch_bytes=0; };
inline PrefillMatrixPlan prefill_long_nax_plan(bool bf16,uint32_t dim,uint32_t heads,uint32_t kvheads,
    const std::vector<std::vector<uint32_t>>& spans) {
  if (!bf16 || dim!=256 || heads!=24 || kvheads!=4 || spans.size()!=2)
    throw std::invalid_argument("long fused NAX requires BF16 D256 H24/KV4 two spans");
  PrefillMatrixPlan p{};p.spans=2;p.long_nax=true;
  for(const auto& s:spans){
    if(s.size()!=8 || s[0]<256 || s[0]>8192 || s[2]>8192 || s[1]>=s[2] ||
       static_cast<uint64_t>(s[1])+s[0]!=s[2] ||
       s[3]!=0 || s[4]!=0 || s[7]!=0 || s[6]!=(s[2]+63)/64)
      throw std::invalid_argument("long fused NAX requires full unwindowed spans and 256..8192 query rows");
    p.max_tiles=std::max(p.max_tiles,(s[0]+63)/64);
  }
  return p;
}
inline PrefillMatrixPlan prefill_long_n20_plan(bool bf16,uint32_t dim,uint32_t heads,uint32_t kvheads,
    const std::vector<std::vector<uint32_t>>& spans) {
  if (!bf16 || dim!=256 || heads!=24 || kvheads!=4 ||
      spans.empty() || spans.size()>20)
    throw std::invalid_argument("long N20 requires BF16 D256 H24/KV4 and1..20 spans");
  PrefillMatrixPlan p{};p.spans=static_cast<uint32_t>(spans.size());
  p.long_nax=true;p.long_n20=true;
  uint64_t rows=0;
  for(const auto& s:spans) {
    if(s.size()!=8 || s[0]<256 || s[0]>8192 || s[2]>8192 || s[1]>=s[2] ||
       uint64_t(s[1])+s[0]!=s[2] || s[3]!=0 || s[4]!=0 || s[7]!=0 ||
       s[6]!=(s[2]+63)/64)
      throw std::invalid_argument("long N20 requires full unwindowed origin0 spans and256..8192 rows");
    rows+=s[0];p.max_tiles=std::max(p.max_tiles,(s[0]+63)/64);
  }
  if(rows>20*8192)throw std::invalid_argument("long N20 total rows exceed bound");
  return p;
}
inline PrefillMatrixPlan prefill_matrix_plan(bool bf16, uint32_t dim, uint32_t heads,
    uint32_t kv_heads, const std::vector<std::vector<uint32_t>>& spans) {
  if (!bf16 || dim != 256 || heads == 0 || kv_heads == 0 || heads % kv_heads ||
      spans.empty() || spans.size() > 8)
    throw std::invalid_argument("prefill matrix requires BF16 D256 GQA and1..8 spans");
  PrefillMatrixPlan plan{0,static_cast<uint32_t>(spans.size())};
  for (const auto& s:spans) {
    if (s.size()!=8 || s[0]<9 || s[0]>=1024 || s[2]>8192 || s[1]>=s[2] ||
        static_cast<uint64_t>(s[1])+s[0]!=s[2] || s[3]>s[1])
      throw std::invalid_argument("prefill matrix stock-short requires query counts9..1023 and causal end<=8192");
    plan.max_tiles=std::max(plan.max_tiles,(s[0]+15)/16);
  }
  return plan;
}
inline PrefillMatrixPlan prefill_nax_plan(bool bf16,uint32_t dim,uint32_t heads,uint32_t kvheads,const std::vector<std::vector<uint32_t>>& spans){
 auto p=prefill_matrix_plan(bf16,dim,heads,kvheads,spans);
 if(heads!=24||kvheads!=4||spans.size()>2)throw std::invalid_argument("prefill exact NAX requires24/4 heads and1..2spans");
 uint64_t rows=0;for(const auto& s:spans){
  if(s[0]>129||s[2]>129||s[3]!=0||s[4]!=0||s[7]!=0)throw std::invalid_argument("prefill exact NAX requires count/end<=129 and unwindowed origin0");
  rows+=s[0];
 }
 p.max_tiles=0;for(const auto& s:spans)p.max_tiles=std::max(p.max_tiles,(s[0]+63)/64);
 p.exact_nax=true;p.scratch_bytes=rows*heads*129*2*2;return p;
}
inline constexpr const char* kPrefillMatrix = R"metal(
using namespace mlx::steel;
struct PFMax { template<typename T> METAL_FUNC static T apply(T x,T y){return max(x,y);} };
struct PFSum { template<typename T> METAL_FUNC static T apply(T x,T y){return x+y;} };
struct PFMul { template<typename T> METAL_FUNC static T apply(T x,T y){return x*y;} };
struct PFRoundProb { template<typename T> METAL_FUNC static T apply(T x,T inverse){return T(bfloat(x*inverse));} };
struct PFExp { template<typename T> METAL_FUNC static T apply(T x,T y){return fast::exp(x-y);} };
// Multi-query16x16 tiles with two-pass stock-short BF16 score/probability law.
// Score/value MMA order and row reductions are mined from pinned Steel attention.
[[kernel,max_total_threads_per_threadgroup(64)]] void mlx2_paged_prefill_matrix_bf16(
 device const bfloat* Q [[buffer(0)]], device const bfloat* K [[buffer(1)]],
 device const bfloat* V [[buffer(2)]], device const uchar* dependency [[buffer(3)]],
 device const uint* row_span [[buffer(4)]], device const uint* row_begin [[buffer(5)]],
 device const uint* query_start [[buffer(6)]], device const uint* retained [[buffer(7)]],
 device const uint* first_block [[buffer(8)]], device const uint* table_begin [[buffer(9)]],
 device const uint* window [[buffer(10)]], device const uint* pages [[buffer(11)]],
 device bfloat* O [[buffer(12)]], constant float& scale [[buffer(13)]],
 constant uint& heads [[buffer(14)]],constant uint& kv_heads [[buffer(15)]],
 constant uint& dim [[buffer(16)]],constant ulong& qrs [[buffer(17)]],
 constant ulong& qhs [[buffer(18)]],constant ulong& qds [[buffer(19)]],
 constant uint& span_count [[buffer(20)]],constant uint& total_rows [[buffer(21)]],
 uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]],
 uint3 tile [[threadgroup_position_in_grid]],uint flat_thread [[thread_index_in_threadgroup]]) {
 (void)dependency; (void)row_span; (void)dim;
 constexpr int BQ=16,BK=16,BD=256,LDQ=264,LDK=24,LDV=264;
 const uint span=tile.z, begin=row_begin[span];
 const uint count=(span+1<span_count?row_begin[span+1]:total_rows)-begin;
 const uint local_begin=tile.x*BQ;
 if(local_begin>=count)return; // Uniform threadgroup exit, including padded grid tiles.
 const uint valid_q=min(uint(BQ),count-local_begin);
 const uint kv_end=query_start[span]+count;
 const uint kv_length=kv_end-retained[span];
 const uint kh=tile.y/(heads/kv_heads);
 const uint q_absolute=query_start[span]+local_begin;
 const uint blocks=min((kv_length+BK-1)/BK,(q_absolute+valid_q-retained[span]+BK-1)/BK);
 threadgroup bfloat Qs[BQ*LDQ];
 threadgroup bfloat KVs[BD*LDK];
 for(uint i=flat_thread;i<BQ*BD;i+=64){uint r=i/BD,d=i%BD;
   Qs[r*LDQ+d]=r<valid_q?bfloat(float(Q[ulong(begin+local_begin+r)*qrs+ulong(tile.y)*qhs+ulong(d)*qds])*float(bfloat(scale))):bfloat(0);}
 using Frag=BaseMMAFrag<float,8,8>;
 MMATile<float,1,1,Frag> Qtile;
 MMATile<float,1,2,Frag> Ktile,Stile;
 MMATile<float,1,1,Frag> Vtile;
 MMATile<float,1,32,Frag> Otile; Otile.clear();
 const short2 coord=Frag::get_coord(lane);const short sm=coord.y,sn=coord.x,tm=8*simd;
 constexpr short R=decltype(Stile)::kRowsPerThread;
 float max_score[R],sum_score[R]={0};
 for(short i=0;i<R;i++)max_score[i]=-3.402823466e38f;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint phase=0;phase<2;phase++){
 for(uint kb=0;kb<blocks;kb++){
   threadgroup_barrier(mem_flags::mem_threadgroup);
   for(uint i=flat_thread;i<BK*BD;i+=64){uint r=i/BD,d=i%BD,pos=retained[span]+kb*BK+r;
     bfloat value=bfloat(0);
     if(pos<kv_end){uint page=pages[table_begin[span]+pos/64-first_block[span]];
       value=K[(ulong(page)*kv_heads*64+ulong(kh)*64+pos%64)*BD+d];}
     KVs[d*LDK+r]=value;}
   Stile.clear();threadgroup_barrier(mem_flags::mem_threadgroup);
   for(short dd=0;dd<32;dd++){
     simdgroup_barrier(mem_flags::mem_none);
     Qtile.load<bfloat,1,1,LDQ,1>(&Qs[(tm+sm)*LDQ+sn+dd*8]);
     Ktile.load<bfloat,1,1,LDK,1>(&KVs[sm*LDK+sn+dd*8*LDK]);
     simdgroup_barrier(mem_flags::mem_none);tile_matmad(Stile,Qtile,Ktile,Stile);}
   for(short i=0;i<decltype(Stile)::kElemsPerTile;i++)Stile.elems()[i]=float(bfloat(Stile.elems()[i]));
   const uint qpos=q_absolute+tm+sm;
   const uint lower=window[span]?max(retained[span],qpos+1>window[span]?qpos+1-window[span]:0u):retained[span];
   for(short j=0;j<2;j++)for(short jj=0;jj<Frag::kElemCols;jj++){
     uint pos=retained[span]+kb*BK+sn+j*8+jj;
     if(pos>=kv_end||pos>qpos||pos<lower)Stile.frag_at(0,j)[jj]=-3.3895313892515355e38f;}
   if(phase==0){
     float new_max[R],factor[R],sum_tmp[R]={0};
     for(short i=0;i<R;i++)new_max[i]=max_score[i];
     Stile.row_reduce<PFMax>(new_max);Stile.row_bin_op<PFExp>(new_max);
     for(short i=0;i<R;i++){factor[i]=fast::exp(max_score[i]-new_max[i]);max_score[i]=new_max[i];}
     Stile.row_reduce<PFSum>(sum_tmp);
     for(short i=0;i<R;i++)sum_score[i]=sum_score[i]*factor[i]+sum_tmp[i];
     continue;
   }
   // Recompute rounded scores, then normalize across the FULL causal row
   // before casting probabilities to BF16, as pinned short SDPA fallback.
   Stile.row_bin_op<PFExp>(max_score);
   float inverse[R];for(short i=0;i<R;i++)inverse[i]=1.0f/sum_score[i];
   Stile.row_bin_op<PFRoundProb>(inverse);
   threadgroup_barrier(mem_flags::mem_threadgroup);
   for(uint i=flat_thread;i<BK*BD;i+=64){uint r=i/BD,d=i%BD,pos=retained[span]+kb*BK+r;
     bfloat value=bfloat(0);
     if(pos<kv_end){uint page=pages[table_begin[span]+pos/64-first_block[span]];
       value=V[(ulong(page)*kv_heads*64+ulong(kh)*64+pos%64)*BD+d];}
     KVs[r*LDV+d]=value;}
   threadgroup_barrier(mem_flags::mem_threadgroup);
   for(short id=0;id<32;id++)for(short ik=0;ik<2;ik++){
     Vtile.load<bfloat,1,1,LDV,1>(&KVs[sm*LDV+sn+ik*8*LDV+id*8]);
     Frag::mma(Otile.frag_at(0,id),Stile.frag_at(0,ik),Vtile.frag_at(0,0),Otile.frag_at(0,id));}
 }
 } // two passes; normalized BF16 P has already supplied the denominator
 const short2 dims=short2(BD-sn,valid_q-(tm+sm));
 if(dims.y>0)Otile.store_safe<bfloat,1,1>(O+ulong(begin+local_begin+tm+sm)*heads*BD+ulong(tile.y)*BD+sn,heads*BD,dims);
}
)metal";
inline std::string prefill_matrix_source(){return std::string(kPrefillMMA)+kPrefillMatrix;}
} // namespace mlx2::paged_kv
