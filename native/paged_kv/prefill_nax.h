#pragma once
#include "vendor/stock_nax_source.h"
namespace mlx2::paged_kv {
inline constexpr const char* kPrefillNAX = R"metal(
using namespace mlx::steel;
// Exact installed stock regular GEMM shape: BM64 BN128 BK256 WM2 WN4.
// Only the register fragment loaders change to revision-checked paged addresses.
METAL_FUNC bfloat exact_page(device const bfloat* plane,device const uint* pages,
 device const uint* table,device const uint* first,uint span,uint pos,uint kh,uint kvheads,uint d){
 uint page=pages[table[span]+pos/64-first[span]];
 return plane[(ulong(page)*kvheads*64+ulong(kh)*64+pos%64)*256+d];
}
[[kernel,max_total_threads_per_threadgroup(256)]] void mlx2_prefill_nax_score(device const bfloat* Q [[buffer(0)]], device const bfloat* K [[buffer(1)]],
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
 device bfloat* scores [[buffer(22)]],device bfloat* probabilities [[buffer(23)]],
 uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]],
 uint3 tile [[threadgroup_position_in_grid]],uint flat_thread [[thread_index_in_threadgroup]]) {
 const uint span=tile.z,begin=row_begin[span];
 const uint count=(span+1<span_count?row_begin[span+1]:total_rows)-begin;
 const uint end=query_start[span]+count;
 const uint head=tile.y%heads,kh=head/(heads/kv_heads);
 const uint m0=tile.x*64+32*(simd/4),n0=128*(tile.y/heads)+32*(simd%4);
 const short2 coord=BaseNAXFrag::get_coord();
 NAXTile<float,2,2> accum;accum.clear();
const uint columns=end;
for(uint kk=0;kk<256;kk+=32){
 threadgroup_barrier(mem_flags::mem_none);
 NAXTile<bfloat,2,2> a,b;
 for(short fm=0;fm<2;fm++)for(short fn=0;fn<2;fn++)for(short i=0;i<8;i++){
 uint r=coord.y+(i/4)*8+fm*16,c=coord.x+i%4+fn*16;
 uint qr=m0+r,d=kk+c;
 a.frag_at(fm,fn)[i]=qr<count?bfloat(float(Q[ulong(begin+qr)*qrs+ulong(head)*qhs+ulong(d)*qds])*float(bfloat(scale))):bfloat(0);
 uint kr=n0+r,kd=kk+c;
 b.frag_at(fm,fn)[i]=kr<end?exact_page(K,pages,table_begin,first_block,span,kr,kh,kv_heads,kd):bfloat(0);
}
 tile_matmad_nax(accum,a,metal::bool_constant<false>{},b,metal::bool_constant<true>{});
}
for(short fm=0;fm<2;fm++)for(short fn=0;fn<2;fn++)for(short i=0;i<8;i++){
 uint r=m0+coord.y+(i/4)*8+fm*16,c=n0+coord.x+i%4+fn*16;
 if(r<count&&c<columns){
 uint qpos=query_start[span]+r;uint lower=window[span]?max(retained[span],qpos+1>window[span]?qpos+1-window[span]:0u):retained[span];
 scores[(ulong(begin+r)*heads+head)*129+c]=(c>qpos||c<lower)?bfloat(-3.3895313892515355e38f):bfloat(accum.frag_at(fm,fn)[i]);
}
}
}
[[kernel,max_total_threads_per_threadgroup(256)]] void mlx2_prefill_nax_value(device const bfloat* Q [[buffer(0)]], device const bfloat* K [[buffer(1)]],
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
 device bfloat* scores [[buffer(22)]],device bfloat* probabilities [[buffer(23)]],
 uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]],
 uint3 tile [[threadgroup_position_in_grid]],uint flat_thread [[thread_index_in_threadgroup]]) {
 const uint span=tile.z,begin=row_begin[span];
 const uint count=(span+1<span_count?row_begin[span+1]:total_rows)-begin;
 const uint end=query_start[span]+count;
 const uint head=tile.y%heads,kh=head/(heads/kv_heads);
 const uint m0=tile.x*64+32*(simd/4),n0=128*(tile.y/heads)+32*(simd%4);
 const short2 coord=BaseNAXFrag::get_coord();
 NAXTile<float,2,2> accum;accum.clear();
const uint columns=256;
for(uint kk=0;kk<end;kk+=32){
 threadgroup_barrier(mem_flags::mem_none);
 NAXTile<bfloat,2,2> a,b;
 for(short fm=0;fm<2;fm++)for(short fn=0;fn<2;fn++)for(short i=0;i<8;i++){
 uint r=coord.y+(i/4)*8+fm*16,c=coord.x+i%4+fn*16;
 uint qr=m0+r,ki=kk+c;
 a.frag_at(fm,fn)[i]=qr<count&&ki<end?probabilities[(ulong(begin+qr)*heads+head)*129+ki]:bfloat(0);
 uint kr=kk+r,d=n0+c;
 b.frag_at(fm,fn)[i]=kr<end?exact_page(V,pages,table_begin,first_block,span,kr,kh,kv_heads,d):bfloat(0);
}
 tile_matmad_nax(accum,a,metal::bool_constant<false>{},b,metal::bool_constant<false>{});
}
for(short fm=0;fm<2;fm++)for(short fn=0;fn<2;fn++)for(short i=0;i<8;i++){
 uint r=m0+coord.y+(i/4)*8+fm*16,c=n0+coord.x+i%4+fn*16;
 if(r<count&&c<columns){
O[(ulong(begin+r)*heads+head)*256+c]=bfloat(accum.frag_at(fm,fn)[i]);
}
}
}
[[kernel,max_total_threads_per_threadgroup(64)]] void mlx2_prefill_nax_softmax(device const bfloat* Q [[buffer(0)]], device const bfloat* K [[buffer(1)]],
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
 device bfloat* scores [[buffer(22)]],device bfloat* probabilities [[buffer(23)]],
 uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]],
 uint3 tile [[threadgroup_position_in_grid]],uint flat_thread [[thread_index_in_threadgroup]]) {
 uint row=tile.x,head=tile.y,span=row_span[row];
 uint count=(span+1<span_count?row_begin[span+1]:total_rows)-row_begin[span];
 uint axis=query_start[span]+count;
 ulong offset=(ulong(row)*heads+head)*129;
 threadgroup float local_max[32],local_normalizer[32];
 float ld[4];for(uint i=0;i<4;i++)ld[i]=flat_thread*4+i<axis?float(scores[offset+flat_thread*4+i]):-INFINITY;
 if(simd==0){local_max[lane]=-INFINITY;local_normalizer[lane]=0;}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float maxval=-3.402823466e38f;for(uint i=0;i<4;i++)maxval=max(maxval,ld[i]);
 maxval=simd_max(maxval);if(lane==0)local_max[simd]=maxval;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(simd==0){maxval=simd_max(local_max[lane]);if(lane==0)local_max[0]=maxval;}
 threadgroup_barrier(mem_flags::mem_threadgroup);maxval=local_max[0];
 float normalizer=0;for(uint i=0;i<4;i++){ld[i]=fast::exp(ld[i]-maxval);normalizer+=ld[i];}
 normalizer=simd_sum(normalizer);if(lane==0)local_normalizer[simd]=normalizer;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(simd==0){normalizer=simd_sum(local_normalizer[lane]);if(lane==0)local_normalizer[0]=normalizer;}
 threadgroup_barrier(mem_flags::mem_threadgroup);normalizer=1/local_normalizer[0];
 for(uint i=0;i<4;i++)if(flat_thread*4+i<axis)probabilities[offset+flat_thread*4+i]=bfloat(ld[i]*normalizer);
}

)metal";
inline std::string prefill_nax_source(){return std::string(kPrefillMMA)+kStockNAX+kPrefillNAX;}
}
