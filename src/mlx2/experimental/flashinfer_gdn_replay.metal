// SPDX-License-Identifier: Apache-2.0
// Original mlx2 serial-FP32 candidate. Design provenance in adjacent JSON.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// HEADER_BEGIN
template <typename DimsPointer>
inline ulong gdn_state_index(ulong l, ulong s, ulong h, ulong v, ulong k, DimsPointer d) {
  const ulong base = ((l*d[1]+s)*d[2]+h)*d[3]*d[4];
  return base + (d[8] == 0 ? v*d[4]+k : k*d[3]+v);
}
// HEADER_END

kernel void mlx2_output_only_gdn_verify(
    device const float* checkpoint [[buffer(0)]], device const float* query [[buffer(1)]],
    device const float* keys [[buffer(2)]], device const float* values [[buffer(3)]],
    device const float* decay [[buffer(4)]], device const float* beta [[buffer(5)]],
    device const long* slots [[buffer(6)]], device const long* lengths [[buffer(7)]],
    constant ulong* dims [[buffer(8)]], device float* outputs [[buffer(9)]],
    device float* corrections [[buffer(10)]], uint3 thread_position_in_grid [[thread_position_in_grid]]) {
// VERIFY_BODY_BEGIN
  const ulong row = ulong(thread_position_in_grid.x);
  const ulong H=dims[2], V=dims[3], K=dims[4], B=dims[5], T=dims[6], HK=dims[7];
  if (row >= dims[0]*B*H*V) return;
  const ulong vi=row%V, h=(row/V)%H, b=(row/(V*H))%B, l=row/(V*H*B);
  const long slot=slots[b];
  float s[128];
  if (slot >= 0) for (ulong j=0;j<K;j++) s[j]=checkpoint[gdn_state_index(l,ulong(slot),h,vi,j,dims)];
  for (ulong t=0;t<T;t++) {
    const ulong oi=(((l*B+b)*T+t)*H+h)*V+vi;
    if (slot < 0 || t>=ulong(lengths[b])) {outputs[oi]=0.0f;corrections[oi]=0.0f;continue;}
    const ulong ki=(((l*B+b)*T+t)*HK+h/(H/HK))*K;
    const ulong gi=(((l*B+b)*T+t)*H+h);
    const float g=decay[gi];
    float prediction=0.0f;
    for (ulong j=0;j<K;j++) {s[j]=s[j]*g;prediction=prediction+s[j]*keys[ki+j];}
    const float delta=(values[oi]-prediction)*beta[gi];
    float output=0.0f;
    for (ulong j=0;j<K;j++) {s[j]=s[j]+keys[ki+j]*delta;output=output+s[j]*query[ki+j];}
    corrections[oi]=delta;outputs[oi]=output;
  }
// VERIFY_BODY_END
}

kernel void mlx2_output_only_gdn_fold(
    device const float* checkpoint [[buffer(0)]], device const float* keys [[buffer(1)]],
    device const float* corrections [[buffer(2)]], device const float* decay [[buffer(3)]],
    device const long* slots [[buffer(4)]], device const long* accepted [[buffer(5)]],
    constant ulong* dims [[buffer(6)]], device float* successor [[buffer(7)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]) {
// FOLD_BODY_BEGIN
  const ulong row=ulong(thread_position_in_grid.x);
  const ulong H=dims[2], V=dims[3], K=dims[4], B=dims[5], T=dims[6], HK=dims[7];
  if (row>=dims[0]*B*H*V) return;
  const ulong vi=row%V, h=(row/V)%H, b=(row/(V*H))%B, l=row/(V*H*B);
  const long slot=slots[b];
  if (slot<0 || accepted[b]==0) {for(ulong j=0;j<K;j++) successor[row*K+j]=0.0f;return;}
  float s[128];
  for (ulong j=0;j<K;j++) s[j]=checkpoint[gdn_state_index(l,ulong(slot),h,vi,j,dims)];
  for (long t=0;t<accepted[b];t++) {
    const ulong token=ulong(t);
    const ulong ki=(((l*B+b)*T+token)*HK+h/(H/HK))*K;
    const ulong gi=(((l*B+b)*T+token)*H+h);
    const ulong ci=gi*V+vi;
    for (ulong j=0;j<K;j++) {s[j]=s[j]*decay[gi];s[j]=s[j]+keys[ki+j]*corrections[ci];}
  }
  for (ulong j=0;j<K;j++) successor[row*K+j]=s[j];
// FOLD_BODY_END
}
