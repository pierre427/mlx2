"""Original segmented GQA7 attention; independent KV views, shared staging.

Candidates retain the old implicit split boundaries using virtual left padding,
but never allocate padded/joined historical KV. See provenance entry of same name.
"""
import hashlib
import math
from functools import lru_cache
from pathlib import Path

import mlx.core as mx

from .nanbeige43_attention import _COMBINE

LAW = "nanbeige43-segmented-gqa7-d128-shared-tile16-v1"


def source_identity():
    from .nanbeige43_attention import source_identity as base_identity
    return hashlib.sha256(Path(__file__).read_bytes()+base_identity().encode()).hexdigest()


_BODY = r"""
    uint tid=thread_index_in_threadgroup;
    uint lane=thread_index_in_simdgroup;
    uint sg=simdgroup_index_in_threadgroup;
    uint tile=threadgroup_position_in_grid.x;
    uint split=threadgroup_position_in_grid.y;
    const int W=int(Q_shape[2]);
    const int TILES=(W+ROWS-1)/ROWS;
    uint row0=(tile%TILES)*ROWS;
    uint kvhead=(tile/TILES)%8;
    uint batch=tile/(TILES*8);
    uint head=kvhead*7+sg;
    int maximum=0;
    for(int i=0;i<BATCH;++i) maximum=max(maximum,prefixes[i]);
    int left=maximum-prefixes[batch];
    int span=max(256,(maximum+W+127)/128);
    int splits=(maximum+W+span-1)/span;
    int begin=int(split)*span, end=min(begin+span,maximum+W);
    // SELECT_LANE
    threadgroup float kt[16*128];
    threadgroup float vt[16*128];
    float qq[ROWS][4], acc[ROWS][4], mm[ROWS], zz[ROWS];
    for(int r=0;r<ROWS;++r){
        mm[r]=-INFINITY; zz[r]=0.0f;
        for(int f=0;f<4;++f){
            qq[r][f]=(row0+r<W)?float(Q[batch*Q_strides[0]+head*Q_strides[1]+(row0+r)*Q_strides[2]+(lane+f*32)*Q_strides[3]]):0.0f;
            acc[r][f]=0.0f;
        }
    }
    for(int start=begin;start<end;start+=16){
        for(uint i=tid;i<16*128;i+=224){
            int pos=start+int(i/128), dim=i%128;
            int logical=pos-left;
            bool valid=pos<end && logical>=0 && logical<prefixes[batch]+W;
            kt[i]=valid?float(kp[kvhead*ks1+logical*ks2+dim*ks3]):0.0f;
            vt[i]=valid?float(vp[kvhead*vs1+logical*vs2+dim*vs3]):0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(int p=0;p<16 && start+p<end;++p){
            int pos=start+p;
            for(int r=0;r<ROWS;++r){
                bool visible=row0+r<W && pos>=left && (pos<maximum || ancestors[(batch*W+row0+r)*W+pos-maximum]);
                if(!visible) continue;
                float score=0.0f;
                for(int f=0;f<4;++f) score+=qq[r][f]*kt[p*128+lane+f*32];
                score=simd_sum(score)*SCALE;
                float next=max(mm[r],score), old=metal::exp(mm[r]-next), weight=metal::exp(score-next);
                for(int f=0;f<4;++f) acc[r][f]=acc[r][f]*old+weight*vt[p*128+lane+f*32];
                zz[r]=zz[r]*old+weight; mm[r]=next;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(int r=0;r<ROWS;++r){
        if(row0+r>=W) continue;
        uint slot=((batch*56+head)*W+row0+r)*splits+split;
        if(lane==0){stats[slot*2]=mm[r];stats[slot*2+1]=zz[r];}
        for(int f=0;f<4;++f) partial[slot*128+lane+f*32]=acc[r][f];
    }
"""


@lru_cache(maxsize=32)
def _kernels(batch, rows, scale):
    names=["Q"]+[name for i in range(batch) for name in (f"K{i}",f"V{i}")]+["prefixes","ancestors"]
    select="device const T* kp=K0; device const T* vp=V0; int ks1=K0_strides[1],ks2=K0_strides[2],ks3=K0_strides[3]; int vs1=V0_strides[1],vs2=V0_strides[2],vs3=V0_strides[3];\n"
    for i in range(1,batch):
        select+=f"if(batch=={i}){{kp=K{i};vp=V{i};ks1=K{i}_strides[1];ks2=K{i}_strides[2];ks3=K{i}_strides[3];vs1=V{i}_strides[1];vs2=V{i}_strides[2];vs3=V{i}_strides[3];}}\n"
    return (mx.fast.metal_kernel(name=f"nanbeige43_segmented_b{batch}_r{rows}",input_names=names,output_names=["partial","stats"],source=_BODY.replace("// SELECT_LANE",select),header=f"constant constexpr int BATCH={batch}, ROWS={rows}; constant constexpr float SCALE={float(scale):.17g}f;",ensure_row_contiguous=False),
            mx.fast.metal_kernel(name="nanbeige43_segmented_combine",input_names=["partial","stats"],output_names=["Y"],source=_COMBINE))


def attend_segments(q, keys, values, *, prefixes, ancestors, scale, shared_kv_rows=1, dispatch_trace=None):
    """Consume B independent [1,8,N,D] views; concatenate only small outputs."""
    if shared_kv_rows not in (1,4) or not math.isfinite(scale) or scale<=0:
        raise ValueError("segmented attention requires rows1/4 and positive finite scale")
    if q.ndim!=4 or not 1<=q.shape[0]<=4 or q.shape[1]!=56 or q.shape[3]!=128 or not 1<=q.shape[2]<=32:
        raise ValueError("segmented attention requires B1..4,Q56,W1..32,D128")
    b,h,w,d=q.shape
    if len(prefixes)!=b or len(keys)!=b or len(values)!=b or ancestors.shape!=(b,w,w) or ancestors.dtype!=mx.bool_:
        raise ValueError("segmented attention lane geometry mismatch")
    if q.dtype not in (mx.float16,mx.bfloat16,mx.float32):
        raise ValueError("segmented attention requires float KV")
    for p,k,v in zip(prefixes,keys,values,strict=True):
        if type(p) is not int or p<0 or k.shape!=(1,8,p+w,128) or v.shape!=k.shape or k.dtype!=q.dtype or v.dtype!=q.dtype:
            raise ValueError("segmented attention KV geometry mismatch")
    if mx.default_device()==mx.cpu:
        outputs=[]
        for i,p in enumerate(prefixes):
            mask=mx.concatenate([mx.ones((w,p),dtype=mx.bool_),ancestors[i]],axis=1)[None,None]
            outputs.append(mx.fast.scaled_dot_product_attention(q[i:i+1],keys[i],values[i],scale=scale,mask=mask))
        return outputs[0] if b==1 else mx.concatenate(outputs,axis=0)
    maximum=max(prefixes); span=max(256,(maximum+w+127)//128); splits=(maximum+w+span-1)//span
    count=b*h*w
    partial_kernel,combine=_kernels(b,shared_kv_rows,scale)
    template=[("T",q.dtype)]
    if dispatch_trace is not None:
        dispatch_trace.append({"law":LAW,"batch":b,"shared_kv_rows":shared_kv_rows,"prefixes":list(prefixes),"span":span,"splits":splits,"padded_kv_bytes":0,"scratch_bytes":count*splits*130*4,"partial_template":[(k,str(v)) for k,v in template],"factory_cache":_kernels.cache_info()._asdict(),"observation_scope":"dispatch templates and Python factory cache, not compiler events"})
    inputs=[q]+[a for pair in zip(keys,values,strict=True) for a in pair]+[mx.array(prefixes,dtype=mx.int32),ancestors]
    partial,stats=partial_kernel(inputs=inputs,template=template,grid=(b*8*((w+shared_kv_rows-1)//shared_kv_rows)*224,splits,1),threadgroup=(224,1,1),output_shapes=[(count*splits,d),(count*splits,2)],output_dtypes=[mx.float32,mx.float32])
    return combine(inputs=[partial,stats],template=[("T",q.dtype),("COUNT",count)],grid=(count*32,1,1),threadgroup=(32,1,1),output_shapes=[q.shape],output_dtypes=[q.dtype])[0]
