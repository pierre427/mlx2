// Original mathematical mechanism experiment; no production code copied.
// A scalar-decayed delta-rule state S is represented as c*R in lazy variants.
#include <metal_stdlib>
using namespace metal;
struct Params { uint heads, steps, unused0, unused1; };
template<int Renorm>
inline void recurrence(device const float* initial,device const float* keys,device const float* queries,device const float* values,device const float* gates,device const float* betas,device float* finalState,device float* outputs,device float* snapshots,constant Params& p,uint head,uint sid,uint lid) {
  float state[16][4];
  #pragma clang loop unroll(full)
  for(uint r=0;r<16;++r) {
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)state[r][j]=initial[head*16384+(sid+8*r)*128+lid+32*j];
  }
  float scale=1.0f;
  for(uint t=0;t<p.steps;++t) {
    uint base=(t*p.heads+head)*128;float g=gates[t*p.heads+head],beta=betas[t*p.heads+head];
    float k[4],q[4];
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j){k[j]=keys[base+lid+32*j];q[j]=queries[base+lid+32*j];}
    if(Renorm>0)scale*=g;
    float inverseScale=Renorm>0 ? 1.0f/scale : 1.0f;
    #pragma clang loop unroll(full)
    for(uint r=0;r<16;++r) {
      float pred=0.0f;
      #pragma clang loop unroll(full)
      for(uint j=0;j<4;++j){if(Renorm==0)state[r][j]*=g;pred=fma(state[r][j],k[j],pred);}
      pred=simd_sum(pred);if(Renorm>0)pred*=scale;
      float delta=beta*(values[base+sid+8*r]-pred);
      if(Renorm>0)delta*=inverseScale;
      float o=0.0f;
      #pragma clang loop unroll(full)
      for(uint j=0;j<4;++j){state[r][j]=fma(delta,k[j],state[r][j]);o=fma(state[r][j],q[j],o);}
      o=simd_sum(o);if(Renorm>0)o*=scale;
      if(lid==0)outputs[base+sid+8*r]=o;
      if(p.unused0) {
        #pragma clang loop unroll(full)
        for(uint j=0;j<4;++j)snapshots[(t*p.heads+head)*16384+(sid+8*r)*128+lid+32*j]=state[r][j]*scale;
      }
    }
    if(Renorm>0 && (t+1)%Renorm==0) {
      #pragma clang loop unroll(full)
      for(uint r=0;r<16;++r) {
        #pragma clang loop unroll(full)
        for(uint j=0;j<4;++j)state[r][j]*=scale;
      }
      scale=1.0f;
    }
  }
  #pragma clang loop unroll(full)
  for(uint r=0;r<16;++r) {
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j)finalState[head*16384+(sid+8*r)*128+lid+32*j]=state[r][j]*scale;
  }
}
#define GDN(NAME,INTERVAL) kernel void NAME(device const float* initial [[buffer(0)]],device const float* keys [[buffer(1)]],device const float* queries [[buffer(2)]],device const float* values [[buffer(3)]],device const float* gates [[buffer(4)]],device const float* betas [[buffer(5)]],device float* finalState [[buffer(6)]],device float* outputs [[buffer(7)]],constant Params& p [[buffer(8)]],device float* snapshots [[buffer(9)]],uint3 tg [[threadgroup_position_in_grid]],uint sid [[simdgroup_index_in_threadgroup]],uint lid [[thread_index_in_simdgroup]]) {recurrence<INTERVAL>(initial,keys,queries,values,gates,betas,finalState,outputs,snapshots,p,tg.x,sid,lid);}
GDN(gdn_direct,0)
GDN(gdn_lazy8,8)
GDN(gdn_lazy32,32)
