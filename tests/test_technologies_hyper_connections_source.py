"""CPU string tests: generated HC Metal sources are byte-identical to the old copies.

``deepseek_v4_hc`` and ``glm53_hc_mega`` used to carry the HC4 Sinkhorn split,
weighted sum and expand as two pasted copies. Both now generate them from
``technologies.hyper_connections``; the ``_LEGACY_*`` literals below are the
former sources moved here verbatim (d46e7a3ea), and the sha256 pins were taken
from those modules before the change. Kernel names, input/output names,
template parameters and grids are unchanged in the call sites. No GPU work.
"""
import hashlib

from mlx2.runtime.models import deepseek_v4_hc, glm53_hc_fused, glm53_hc_mega
from mlx2.runtime.technologies import hyper_connections as hc

_LEGACY_DS4_SPLIT_SOURCE = r'''
const uint row = thread_position_in_grid.x;
if(row >= ROWS) return;
const float e=epsilon[0];
float c[16];
for(uint i=0;i<4;i++) {
  const float pre=mixes[row*24+i]*scale[0]+base[i];
  const float post=mixes[row*24+4+i]*scale[1]+base[4+i];
  out[row*24+i]=1.0f/(1.0f+exp(-pre))+e;
  out[row*24+4+i]=2.0f/(1.0f+exp(-post));
}
for(uint r=0;r<4;r++) {
  float maximum=-INFINITY;
  for(uint s=0;s<4;s++) { c[r*4+s]=mixes[row*24+8+r*4+s]*scale[2]+base[8+r*4+s]; maximum=max(maximum,c[r*4+s]); }
  float sum=0.0f;
  for(uint s=0;s<4;s++) { c[r*4+s]=exp(c[r*4+s]-maximum); sum+=c[r*4+s]; }
  for(uint s=0;s<4;s++) c[r*4+s]=c[r*4+s]/sum+e;
}
for(uint s=0;s<4;s++) {
  float sum=0.0f;
  for(uint r=0;r<4;r++) sum+=c[r*4+s];
  for(uint r=0;r<4;r++) c[r*4+s]/=(sum+e);
}
for(uint it=1;it<ITERS;it++) {
  for(uint r=0;r<4;r++) {
    float sum=0.0f;
    for(uint s=0;s<4;s++) sum+=c[r*4+s];
    for(uint s=0;s<4;s++) c[r*4+s]/=(sum+e);
  }
  for(uint s=0;s<4;s++) {
    float sum=0.0f;
    for(uint r=0;r<4;r++) sum+=c[r*4+s];
    for(uint r=0;r<4;r++) c[r*4+s]/=(sum+e);
  }
}
for(uint i=0;i<16;i++) out[row*24+8+i]=c[i];
'''


_LEGACY_DS4_SUM_SOURCE = r'''
const uint index=thread_position_in_grid.x;
if(index>=ROWS*DIM) return;
const uint row=index/DIM, d=index%DIM;
float sum=0.0f;
for(uint h=0;h<4;h++) sum+=value[(row*4+h)*DIM+d]*weights[row*4+h];
out[index]=sum;
'''


_LEGACY_DS4_EXPAND_SOURCE = r'''
const uint index=thread_position_in_grid.x;
if(index>=ROWS*4*DIM) return;
const uint row=index/(4*DIM), h=(index/DIM)%4, d=index%DIM;
float sum=branch[row*DIM+d]*post[row*4+h];
for(uint s=0;s<4;s++) sum+=residual[(row*4+s)*DIM+d]*comb[row*16+s*4+h];
out[index]=sum;
'''


_LEGACY_DS4_HEAD_SOURCE = r'''
const uint index=thread_position_in_grid.x;
if(index>=ROWS*DIM) return;
const uint row=index/DIM,d=index%DIM;
float sum=0.0f;
for(uint h=0;h<4;h++) {
  const float z=mixes[row*4+h]*scale[0]+base[h];
  const float w=1.0f/(1.0f+exp(-z))+epsilon[0];
  sum+=value[(row*4+h)*DIM+d]*w;
}
out[index]=sum;
'''


_LEGACY_GLM53_HC_MEGA_HEADER = r"""
// Verbatim pieces (default FP options, as deepseek_v4_hc's kernels).
template<typename M, typename S, typename B>
inline void glm_hc_split(M mixes, S scale, B base, float e, uint ITERS, threadgroup float *out) {
    float c[16];
    for(uint i=0;i<4;i++) {
      const float pre=mixes[i]*scale[0]+base[i];
      const float post=mixes[4+i]*scale[1]+base[4+i];
      out[i]=1.0f/(1.0f+exp(-pre))+e;
      out[4+i]=2.0f/(1.0f+exp(-post));
    }
    for(uint r=0;r<4;r++) {
      float maximum=-INFINITY;
      for(uint s=0;s<4;s++) { c[r*4+s]=mixes[8+r*4+s]*scale[2]+base[8+r*4+s]; maximum=max(maximum,c[r*4+s]); }
      float sum=0.0f;
      for(uint s=0;s<4;s++) { c[r*4+s]=exp(c[r*4+s]-maximum); sum+=c[r*4+s]; }
      for(uint s=0;s<4;s++) c[r*4+s]=c[r*4+s]/sum+e;
    }
    for(uint s=0;s<4;s++) {
      float sum=0.0f;
      for(uint r=0;r<4;r++) sum+=c[r*4+s];
      for(uint r=0;r<4;r++) c[r*4+s]/=(sum+e);
    }
    for(uint it=1;it<ITERS;it++) {
      for(uint r=0;r<4;r++) {
        float sum=0.0f;
        for(uint s=0;s<4;s++) sum+=c[r*4+s];
        for(uint s=0;s<4;s++) c[r*4+s]/=(sum+e);
      }
      for(uint s=0;s<4;s++) {
        float sum=0.0f;
        for(uint r=0;r<4;r++) sum+=c[r*4+s];
        for(uint r=0;r<4;r++) c[r*4+s]/=(sum+e);
      }
    }
    for(uint i=0;i<16;i++) out[8+i]=c[i];
}
template<typename V>
inline float glm_hc_sum(V value, threadgroup const float *weights, uint d, uint DIMS) {
    float sum=0.0f;
    for(uint h=0;h<4;h++) sum+=value[h*DIMS+d]*weights[h];
    return sum;
}
template<typename Br, typename R, typename P, typename C>
inline float glm_hc_expand(Br branch, R residual, P post, C comb, uint h, uint d, uint DIMS) {
    float sum=branch[d]*post[h];
    for(uint s=0;s<4;s++) sum+=residual[s*DIMS+d]*comb[s*4+h];
    return sum;
}
// Lane-parallel form of glm_hc_split for simdgroup 0: lane e<16 owns comb
// element (r,s)=(e/4,e%4), lanes 16..19 pre, 20..23 post. Each four-term sum
// gathers its operands by simd_shuffle and adds them from 0 in the serial
// body's order, so every operation is the serial one on another lane.
template<typename M, typename S, typename B>
inline void glm_hc_split_lanes(M mixes, S scale, B base, float e, uint ITERS, uint lane, threadgroup float *out) {
    if(lane>=16 && lane<20) {
        const uint i=lane-16;
        const float pre=mixes[i]*scale[0]+base[i];
        out[i]=1.0f/(1.0f+exp(-pre))+e;
    } else if(lane>=20 && lane<24) {
        const uint i=lane-20;
        const float post=mixes[4+i]*scale[1]+base[4+i];
        out[4+i]=2.0f/(1.0f+exp(-post));
    }
    const uint el=lane&15u, r=el/4, s=el%4;
    float c=mixes[8+el]*scale[2]+base[8+el];
    float maximum=-INFINITY;
    for(uint k=0;k<4;k++) maximum=max(maximum,simd_shuffle(c,ushort(r*4+k)));
    c=exp(c-maximum);
    {
        float sum=0.0f;
        for(uint k=0;k<4;k++) sum+=simd_shuffle(c,ushort(r*4+k));
        c=c/sum+e;
    }
    {
        float sum=0.0f;
        for(uint k=0;k<4;k++) sum+=simd_shuffle(c,ushort(k*4+s));
        c/=(sum+e);
    }
    for(uint it=1;it<ITERS;it++) {
        {
            float sum=0.0f;
            for(uint k=0;k<4;k++) sum+=simd_shuffle(c,ushort(r*4+k));
            c/=(sum+e);
        }
        {
            float sum=0.0f;
            for(uint k=0;k<4;k++) sum+=simd_shuffle(c,ushort(k*4+s));
            c/=(sum+e);
        }
    }
    if(lane<16) out[8+lane]=c;
}
// Norm reductions in mx.fast.rms_norm's order, FP contraction off.
#pragma clang fp contract(off)
inline float glm_hc_sq_add(float acc, float v) { const float sq=v*v; return acc+sq; }
inline float glm_hc_sq_fma(float acc, float v) { return metal::fma(v, v, acc); }
inline float glm_hc_inv(float total, float n, float eps) { const float mean=total/n; return metal::precise::rsqrt(mean+eps); }
inline float glm_hc_scale(float w, float x, float inv) { const float y=x*inv; return w*y; }
"""


PINS = {
    "split": "1ce8e7cdc390857310917c81503f603f1bbfe4c5b4d43edcf439b74969f5ac1a",
    "sum": "7279ab41ac1c6ec2ff4aa3d5af0af7c4d3d3b517a2c971e17aaeb2a826773576",
    "expand": "6f04100e4ee76411b4c94c978e3e96fcab0fc84c526dbadbe064db911aa894f0",
    "head": "f6974dde55933e8e56ae956344a6cd06ae331af4ecfaf562a57a7c574d0650a6",
    "glm53_hc_mega_header": "20d65ad2b8d369fa9748eec13e0b86cb1dc361f8c5f74cdb7f884819fd970385",
    "glm53_hc_fused_header": "6bccfd0f6ea31076c5d93b81a5c158518932aad92a78002415c6bcec93275840",
    "ds4_compilation_source": "73b3f874f821641239acb1ea5ea0869396d1b60694ffcb4de5c5eedd541a3a5f",
}


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def test_ds4_hc_sources_byte_identical():
    assert deepseek_v4_hc.SPLIT_SOURCE == _LEGACY_DS4_SPLIT_SOURCE
    assert deepseek_v4_hc.SUM_SOURCE == _LEGACY_DS4_SUM_SOURCE
    assert deepseek_v4_hc.EXPAND_SOURCE == _LEGACY_DS4_EXPAND_SOURCE
    assert deepseek_v4_hc.HEAD_SOURCE == _LEGACY_DS4_HEAD_SOURCE
    sources = {k: v[0] for k, v in deepseek_v4_hc.SOURCES.items()}
    assert {k: _sha(v) for k, v in sources.items()} == {
        "split": PINS["split"], "sum": PINS["sum"], "expand": PINS["expand"], "head": PINS["head"]}
    assert [v[1] for v in deepseek_v4_hc.SOURCES.values()] == [
        ["mixes", "scale", "base", "epsilon"], ["value", "weights"],
        ["branch", "residual", "post", "comb"], ["value", "mixes", "scale", "base", "epsilon"]]
    assert _sha(deepseek_v4_hc.compilation_source()) == PINS["ds4_compilation_source"]


def test_glm53_hc_mega_header_byte_identical():
    assert glm53_hc_mega._HEADER == _LEGACY_GLM53_HC_MEGA_HEADER
    assert _sha(glm53_hc_mega._HEADER) == PINS["glm53_hc_mega_header"]
    # glm53_hc_fused compiles mega's header plus its own GEMV header.
    assert _sha(glm53_hc_fused._HEADER) == PINS["glm53_hc_fused_header"]


def test_one_generated_body_serves_both_kernels():
    body = hc.split_sinkhorn_body(row_offset="row*24+")
    assert deepseek_v4_hc.SPLIT_SOURCE.endswith(body)
    glm_body = hc.split_sinkhorn_body(row_offset="", indent="    ")
    assert glm53_hc_mega._HEADER.count(glm_body) == 1
    # Same statements: only the per-row offset and indentation differ.
    assert [line.strip() for line in glm_body.splitlines()] == [
        line.strip().replace("row*24+", "") for line in body.splitlines()]


def test_generator_refuses_other_stream_counts():
    # The kernels and their row offsets are HC4; any other n is refused.
    import pytest

    for n in (1, 2, 8):
        with pytest.raises(ValueError, match="HC4"):
            hc.split_sinkhorn_body(n=n, row_offset="row*24+")
