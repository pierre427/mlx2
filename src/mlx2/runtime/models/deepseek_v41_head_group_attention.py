# SPDX-License-Identifier: Apache-2.0
"""Unwired manual-F32 attention numerical profile with shared KV tiles.

Adapted internal visible-row online-softmax mechanism; retained notice:
provenance/deepseek-v4-visible-attention.NOTICE. See the new head-group
provenance for source hashes/modifications. This is not the stock Source
reduction law, model qualification, or an inference fallback.
"""
from functools import lru_cache

VARIANTS=((1,1),(4,1),(4,4))
HEADER='#include <metal_stdlib>\nusing namespace metal;\n#pragma clang fp contract(off)\n'
PRODUCER=r'''
const uint tid=thread_index_in_threadgroup,lane=tid%32,sg=tid/32;
const uint head=threadgroup_position_in_grid.x*HEAD_GROUP+sg;
const uint query=threadgroup_position_in_grid.y,split=threadgroup_position_in_grid.z;
const uint count=LOCAL_ROWS+SELECTED_ROWS;
const uint first=count*split/SPLITS,last=count*(split+1)/SPLITS;
threadgroup float tile[8*512];
threadgroup int kinds[8];
threadgroup uint offsets[8];
float qv[16],acc[16];
for(uint j=0;j<16;j++) {
 qv[j]=q[(head*LENGTH+query)*512+j*32+lane]*0.04419417382415922f;
 acc[j]=0.0f;
}
float maximum=-INFINITY,denominator=0.0f;
bool bad=false;
for(uint block=first;block<last;block+=8) {
 if(tid<8) {
  uint key=block+tid;kinds[tid]=0;offsets[tid]=0;
  if(key<last) {
   if(key<LOCAL_ROWS) {
    if(local_mask[query*LOCAL_ROWS+key]) {kinds[tid]=1;offsets[tid]=key;}
   } else {
    uint slot=key-LOCAL_ROWS;
    if(valid[query*SELECTED_ROWS+slot]) {
     long index=long(indices[query*SELECTED_ROWS+slot]);
     if(index<0 || index>=long(POOL_ROWS)) kinds[tid]=3;
     else {kinds[tid]=2;offsets[tid]=uint(index);}
    }
   }
  }
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint e=tid;e<8*512;e+=32*HEAD_GROUP) {
  uint row=e/512,d=e%512;
  tile[e]=kinds[row]==1 ? local[offsets[row]*512+d] :
          kinds[row]==2 ? pooled[offsets[row]*512+d] : 0.0f;
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint row=0;row<8;row++) {
  if(kinds[row]==0)continue;
  if(kinds[row]==3) {bad=true;continue;}
  float score=0.0f;
  for(uint j=0;j<16;j++)score+=qv[j]*tile[row*512+j*32+lane];
  score=simd_sum(score);
  if(!isfinite(score)) {bad=true;continue;}
  float next=max(maximum,score);
  float alpha=denominator==0.0f ? 0.0f : exp(maximum-next);
  float beta=exp(score-next);
  denominator=denominator*alpha+beta;
  for(uint j=0;j<16;j++)acc[j]=acc[j]*alpha+beta*tile[row*512+j*32+lane];
  maximum=next;
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
}
if(SPLITS==1) {
 float sink=sinks[head];
 if(!isfinite(sink))bad=true;
 float next=max(maximum,sink);
 float alpha=denominator==0.0f ? 0.0f : exp(maximum-next);
 float denom=denominator*alpha+exp(sink-next);
 float multiplier=alpha/denom;
 for(uint j=0;j<16;j++)out[(head*LENGTH+query)*512+j*32+lane]=bad ? NAN : acc[j]*multiplier;
} else {
 uint base=((split*64+head)*LENGTH+query)*514;
 if(lane==0) {out[base]=bad ? NAN : maximum;out[base+1]=bad ? NAN : denominator;}
 for(uint j=0;j<16;j++)out[base+2+j*32+lane]=bad ? NAN : acc[j];
}
'''

MERGE=r'''
const uint tid=thread_index_in_threadgroup,lane=tid%32,sg=tid/32;
const uint head=threadgroup_position_in_grid.x*HEAD_GROUP+sg,query=threadgroup_position_in_grid.y;
float maximum=sinks[head];bool bad=!isfinite(maximum);
for(uint split=0;split<SPLITS;split++) {
 uint base=((split*64+head)*LENGTH+query)*514;
 float m=partials[base],l=partials[base+1];
 if(!isfinite(l) || l<0.0f || (l>0.0f&&!isfinite(m)))bad=true;
 if(l>0.0f)maximum=max(maximum,m);
}
// The sink contributes denominator mass exactly once, never to partials/PV.
float denominator=exp(sinks[head]-maximum),acc[16];
for(uint j=0;j<16;j++)acc[j]=0.0f;
for(uint split=0;split<SPLITS;split++) {
 uint base=((split*64+head)*LENGTH+query)*514;
 float l=partials[base+1];
 if(l>0.0f) {
  float alpha=exp(partials[base]-maximum);
  denominator+=l*alpha;
  for(uint j=0;j<16;j++)acc[j]+=partials[base+2+j*32+lane]*alpha;
 }
}
for(uint j=0;j<16;j++)out[(head*LENGTH+query)*512+j*32+lane]=bad ? NAN : acc[j]/denominator;
'''


def plan(length,local_rows,pool_rows,selected_rows,*,head_group=4,kv_splits=1):
    values=(length,local_rows,pool_rows,selected_rows,head_group,kv_splits)
    if any(type(x)is not int for x in values)or length not in (1,99):raise ValueError('Head-group attention requires captured M1/M99')
    if not 1<=local_rows<=128 or not 1<=pool_rows<=512 or not 1<=selected_rows<=512:
        raise ValueError('Head-group attention local/pooled bounds')
    if (head_group,kv_splits)not in VARIANTS:raise ValueError('Head-group attention unreviewed grid')
    groups=(64//head_group,length,kv_splits);tg=(32*head_group,1,1)
    return dict(numerical_profile='source_manual_f32_online_attention_v1',head_group=head_group,kv_splits=kv_splits,
        query_heads=64,shared_KV_heads=1,length=length,local_rows=local_rows,pool_rows=pool_rows,selected_rows=selected_rows,
        threadgroups=groups,threadgroup=tg,grid=tuple(a*b for a,b in zip(groups,tg)),
        threadgroup_bytes=8*512*4+8*4*2,KV_tile_rows=8,head_channel_values_per_lane=16,
        partial_shape=None if kv_splits==1 else(kv_splits,64,length,514),
        partial_bytes=0 if kv_splits==1 else kv_splits*64*length*514*4,
        output_shape=(1,64,length,512),merge_threadgroups=None if kv_splits==1 else(64//head_group,length,1),
        logical_all_slot_KV_load_bytes=(64//head_group)*length*(local_rows+selected_rows)*512*4,
        logical_Q_load_bytes=64*length*512*4*kv_splits,
        traffic_scope='Logical instruction/dataflow upperbounds; invalidmaskrows skiploads; not measuredDRAM',
        sink_contributions_per_head_query=1,original_weight_quantization_unchanged=True,cache_layout_unchanged=True)


def geometry(q,local,local_mask,pooled,indices,valid,sinks,**options):
    import mlx.core as mx
    if q.ndim!=4 or q.shape[:2]!=(1,64)or q.shape[-1]!=512:raise ValueError('Head-group attention requires B1/H64/D512')
    if local.ndim!=4 or local.shape[:2]!=(1,1)or local.shape[-1]!=512:raise ValueError('Head-group attention single localKVlatent')
    if pooled.ndim!=3 or pooled.shape[0]!=1 or pooled.shape[-1]!=512:raise ValueError('Head-group attention single pooledKVlatent')
    if indices.ndim!=3 or indices.shape[:2]!=(1,q.shape[2]):raise ValueError('Head-group selectedID geometry')
    p=plan(q.shape[2],local.shape[2],pooled.shape[1],indices.shape[2],**options)
    if (local_mask.shape!=(1,1,q.shape[2],local.shape[2])or valid.shape!=indices.shape or sinks.shape!=(64,)
            or local_mask.dtype!=mx.bool_ or valid.dtype!=mx.bool_ or indices.dtype not in (mx.int32,mx.uint32)
            or any(x.dtype!=mx.float32 for x in (q,local,pooled,sinks))):
        raise ValueError('Head-group attention dtype/mask geometry')
    return p


@lru_cache(maxsize=2)
def _kernel(kind):
    import mlx.core as mx
    names=['q','local','local_mask','pooled','indices','valid','sinks']if kind=='producer'else['partials','sinks']
    return mx.fast.metal_kernel(name='mlx2_v41_headgroup_'+kind+'_v1',input_names=names,output_names=['out'],
        header=HEADER,source=PRODUCER if kind=='producer'else MERGE,ensure_row_contiguous=True)


def attention(q,local,local_mask,pooled,indices,valid,sinks,*,head_group=4,kv_splits=1,return_partials=False):
    import mlx.core as mx
    p=geometry(q,local,local_mask,pooled,indices,valid,sinks,head_group=head_group,kv_splits=kv_splits)
    if type(return_partials)is not bool:raise ValueError('Head-group diagnostic return flag must bebool')
    if mx.default_device()!=mx.gpu or not mx.metal.is_available():raise ValueError('Head-group attention requires caller-owned Metal GPU')
    template=[('HEAD_GROUP',head_group),('SPLITS',kv_splits),('LENGTH',p['length']),('LOCAL_ROWS',p['local_rows']),
              ('POOL_ROWS',p['pool_rows']),('SELECTED_ROWS',p['selected_rows'])]
    value=_kernel('producer')(inputs=[q,local,local_mask,pooled,indices,valid,sinks],template=template,
        grid=p['grid'],threadgroup=p['threadgroup'],output_shapes=[p['output_shape']if kv_splits==1 else p['partial_shape']],output_dtypes=[mx.float32])[0]
    partials=None
    if kv_splits>1:
        partials=value
        value=_kernel('merge')(inputs=[partials,sinks],template=template,
            grid=(p['grid'][0],p['length'],1),threadgroup=p['threadgroup'],output_shapes=[p['output_shape']],output_dtypes=[mx.float32])[0]
    return(value,partials)if return_partials else value


def reference_numpy(q,local,local_mask,pooled,indices,valid,sinks):
    """Independent F64 masked/sink oracle for small CPU/stability fixtures only."""
    import numpy as np
    output=np.zeros(q.shape,np.float32)
    for row in range(q.shape[2]):
        chosen=local[0,0,np.asarray(local_mask[0,0,row],bool)]
        ids=indices[0,row,np.asarray(valid[0,row],bool)].astype(np.int64)
        if np.any(ids<0)or np.any(ids>=pooled.shape[1]):raise ValueError('Oracle invalid valid-selectedID')
        values=np.concatenate((chosen,pooled[0,ids]),axis=0).astype(np.float64)
        # The profile prescales each component in F32 before the dot.
        query=(q[0,:,row]*np.float32(512**-0.5)).astype(np.float64)
        scores=query@values.T
        maximum=np.maximum(np.max(scores,axis=-1,initial=-np.inf),sinks.astype(np.float64))
        weights=np.exp(scores-maximum[:,None]);denominator=weights.sum(-1)+np.exp(sinks-maximum)
        output[0,:,row]=(weights@values/denominator[:,None]).astype(np.float32)
    return output


def compilation_source(head_group=4,kv_splits=1,*,kind='producer',index_type='int'):
    plan(1,101,50,50,head_group=head_group,kv_splits=kv_splits)
    if kind not in ('producer','merge')or index_type not in ('int','uint'):raise ValueError('Offlineattention source kind/type')
    constants=f'constant uint HEAD_GROUP={head_group},SPLITS={kv_splits},LENGTH=1,LOCAL_ROWS=101,POOL_ROWS=50,SELECTED_ROWS=50;\n'
    if kind=='producer':
        args=f'const device float* q [[buffer(0)]],const device float* local [[buffer(1)]],const device bool* local_mask [[buffer(2)]],const device float* pooled [[buffer(3)]],const device {index_type}* indices [[buffer(4)]],const device bool* valid [[buffer(5)]],const device float* sinks [[buffer(6)]],device float* out [[buffer(7)]],'
    else:args='const device float* partials [[buffer(0)]],const device float* sinks [[buffer(1)]],device float* out [[buffer(2)]],'
    return HEADER+constants+'kernel void attention_probe('+args+'uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]){\n'+(PRODUCER if kind=='producer'else MERGE)+'\n}\n'
