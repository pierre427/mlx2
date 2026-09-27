// Original mlx2 research microbenchmarks, 2026-09-25. No upstream code mined.
// No tensor, simdgroup-matrix, Neural Accelerator, or ANE operations.
#include <metal_stdlib>
using namespace metal;
struct Params { uint m, n, k, iterations; };
#define ARGS device const float* x [[buffer(0)]], device const float* w [[buffer(1)]], device const uchar* q [[buffer(2)]], device const float* scales [[buffer(3)]], device const float* biases [[buffer(4)]], device float* out [[buffer(5)]], constant Params& p [[buffer(6)]], device float* xsum [[buffer(7)]], uint tid [[thread_index_in_threadgroup]], uint lid [[thread_index_in_simdgroup]], uint sid [[simdgroup_index_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]]

inline float shuffle_sum(float v) {
  for (ushort d = 16; d > 0; d /= 2) v += simd_shuffle_down(v, d);
  return v;
}
template<int Mode> inline float reduce256(float v, uint tid, uint lid, uint sid, threadgroup float* scratch) {
  if (Mode == 2) {
    scratch[tid] = v; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = 128; d > 0; d /= 2) {
      if (tid < d) scratch[tid] += scratch[tid+d];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return scratch[0];
  } else {
    v = Mode == 0 ? simd_sum(v) : shuffle_sum(v);
    if (lid == 0) scratch[sid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = tid < 8 ? scratch[tid] : 0.0f;
    s = Mode == 0 ? simd_sum(s) : shuffle_sum(s);
    if (tid == 0) scratch[0] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
  }
}
#define REDUCE(NAME,MODE) kernel void NAME(ARGS) { \
  threadgroup float tmp[256]; float v = 0; \
  for (uint j=tid; j<p.k; j+=256) v += x[group.x*p.k+j]; \
  float s=reduce256<MODE>(v,tid,lid,sid,tmp); if(tid==0) out[group.x]=s; }
REDUCE(reduce_simd,0)
REDUCE(reduce_shuffle,1)
REDUCE(reduce_shared,2)
// Intentionally sequential per row; comparison demonstrates available parallelism,
// not an instruction-only speedup. 256 rows share one threadgroup.
kernel void reduce_serial(ARGS) {
  uint row=group.x*256+tid; if(row>=p.n) return;
  float s=0; for(uint j=0;j<p.k;++j) s+=x[row*p.k+j]; out[row]=s;
}

template<int Mode> inline float invroot(float z) {
  if(Mode==0) return 1.0f / precise::sqrt(z);
  if(Mode==1) return precise::rsqrt(z);
  return fast::rsqrt(z);
}
#define INVROOT(NAME,MODE) kernel void NAME(ARGS) { \
  uint i=group.x*256+tid; if(i>=p.n)return; float z=x[i]; \
  for(uint j=0;j<p.iterations;++j) z=invroot<MODE>(z)+0.125f; out[i]=z; }
INVROOT(sqrt_div,0)
INVROOT(rsqrt_precise,1)
INVROOT(rsqrt_fast,2)

template<int Mode> inline float efunc(float z) {
  if(Mode==0) return precise::exp(z);
  if(Mode==1) return fast::exp(z);
  return fast::exp2(z*1.4426950408889634f);
}
#define SOFTMAX(NAME,MODE) kernel void NAME(ARGS) { \
  threadgroup float tmp[256]; float mx=-INFINITY; uint base=group.x*p.k; \
  for(uint j=tid;j<p.k;j+=256) mx=max(mx,x[base+j]); \
  mx=simd_max(mx); if(lid==0) tmp[sid]=mx; threadgroup_barrier(mem_flags::mem_threadgroup); \
  mx=tid<8?tmp[tid]:-INFINITY; mx=simd_max(mx); if(tid==0)tmp[0]=mx; \
  threadgroup_barrier(mem_flags::mem_threadgroup); mx=tmp[0]; threadgroup_barrier(mem_flags::mem_threadgroup); \
  float s=0; for(uint j=tid;j<p.k;j+=256)s+=efunc<MODE>(x[base+j]-mx); \
  s=reduce256<0>(s,tid,lid,sid,tmp); \
  for(uint j=tid;j<p.k;j+=256)out[base+j]=efunc<MODE>(x[base+j]-mx)/s; }
SOFTMAX(softmax_precise,0)
SOFTMAX(softmax_fast,1)
SOFTMAX(softmax_exp2,2)

template<int Acc> inline float dot_acc(device const float* a, device const float* b, uint k, uint lid) {
  float acc[Acc]; for(uint t=0;t<Acc;++t)acc[t]=0;
  for(uint j=lid;j<k;j+=32*Acc) {
    #pragma clang loop unroll(full)
    for(uint t=0;t<Acc;++t) if(j+32*t<k)acc[t]=fma(a[j+32*t],b[j+32*t],acc[t]);
  }
  float s=0;for(uint t=0;t<Acc;++t)s+=acc[t];return simd_sum(s);
}
#define DOT(NAME,ACC) kernel void NAME(ARGS) { \
  uint row=group.x; uint m=row/p.n,n=row%p.n; \
  float s=dot_acc<ACC>(x+m*p.k,w+n*p.k,p.k,lid); if(lid==0)out[row]=s; }
DOT(dot_acc1,1)
DOT(dot_acc2,2)
DOT(dot_acc4,4)
DOT(dot_acc8,8)
DOT(dot_acc16,16)

// Identical per-lane group assignment in both arms: each lane owns a 64-value
// quantization group. Factorization changes the arithmetic and rounding only.
// q4_affine: sum x*(scale*q+bias); q4_factored: scale*sum(x*q)+bias*sum(x).
template<int Mode> inline float qdot(device const float* a, device const uchar* b, device const float* sc, device const float* bi, device const float* precomputed, uint k, uint lid) {
  float total=0; uint ng=k/64;
  for(uint g=lid;g<ng;g+=32) {
    float s=sc[g], bias=bi[g], qx=0, xs=0;
    for(uint j=0;j<32;++j) {
      uchar packed=b[g*32+j]; float q0=float(packed&15),q1=float(packed>>4);
      float x0=a[g*64+2*j],x1=a[g*64+2*j+1];
      if(Mode==1 || Mode==2) {qx=fma(x0,q0,qx);qx=fma(x1,q1,qx);if(Mode==1){xs+=x0;xs+=x1;}}
      else if(Mode==3) {total=fma(x0,float(half(fma(s,q0,bias))),total);total=fma(x1,float(half(fma(s,q1,bias))),total);}
      else {total=fma(x0,fma(s,q0,bias),total);total=fma(x1,fma(s,q1,bias),total);}
    }
    if(Mode==2)xs=precomputed[g];
    if(Mode==1 || Mode==2)total+=fma(s,qx,bias*xs);
  }
  return simd_sum(total);
}
#define QDOT(NAME,FACTORED) kernel void NAME(ARGS) { \
  uint row=group.x;uint m=row/p.n,n=row%p.n; \
  float s=qdot<FACTORED>(x+m*p.k,q+n*(p.k/2),scales+n*(p.k/64),biases+n*(p.k/64),xsum+m*(p.k/64),p.k,lid); \
  if(lid==0)out[row]=s; }
QDOT(q4_affine,0)
QDOT(q4_factored,1)
QDOT(q4_precomputed,2)
QDOT(q4_rounded_half,3)
kernel void activation_group_sums(ARGS) {
  uint i=group.x*256+tid; if(i>=p.m*(p.k/64))return;
  float s=0;for(uint j=0;j<64;++j)s+=x[i*64+j];xsum[i]=s;
}

// Register arithmetic throughput probe, identical FP16-representable inputs and
// coefficients. 8 independent vector accumulators, each 2 components. FP16
// accumulation is approximate; it is not a drop-in FP32 replacement.
template<typename V> inline float2 recurrent_fma(float v, uint iterations) {
  V a[8]; for(uint j=0;j<8;++j)a[j]=V(v+float(j)*0.03125f,v-float(j)*0.03125f);
  const V mul=V(0.9990234375f), add=V(0.0009765625f);
  for(uint i=0;i<iterations;++i) {
    #pragma clang loop unroll(full)
    for(uint j=0;j<8;++j)a[j]=fma(a[j],mul,add);
  }
  float2 s=0;for(uint j=0;j<8;++j)s+=float2(a[j]);return s;
}
#define PACKED(NAME,TYPE) kernel void NAME(ARGS) { \
  uint i=group.x*256+tid; if(i>=p.n)return;float2 s=recurrent_fma<TYPE>(x[i],p.iterations); \
  out[2*i]=s.x;out[2*i+1]=s.y; }
PACKED(fma_float2,float2)
PACKED(fma_half2,half2)
