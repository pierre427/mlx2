# SPDX-License-Identifier: MIT
"""Wide original PP extension using pinned legacy F32 split-K helpers.

The partition law is derived from the exact installed qmm_splitk dispatch,
libmlx.dylib d8c9393d584e781393c3effccfccd8ba85ad0c70264edaf9f897dca9875375c6,
addresses0xd96948..0xd969bc. Parent-owned raw GPU parity is required before use.
No mode is selected by import. Source attribution remains in the mined helper.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
from . import source_resident_prefill as narrow
from .source_resident_bank_gather import layout_key,_constants,_bank_select


def split_partitions(rows,outputs,inputs):
    if any(type(v)is not int or v<=0 for v in (rows,outputs,inputs))or inputs%32:
        raise ValueError('Invalid Source stock split-K geometry')
    target=max(1,512//(((outputs+31)//32)*((rows+31)//32)))
    p=min(inputs//32,target)
    while p>1 and inputs%(p*32):p-=1
    return p


def implementation_identity():
    from .source_prefill_legacy_tiles import HASHES
    return hashlib.sha256((Path(__file__).read_text()+narrow.implementation_identity()+repr(HASHES)).encode()).hexdigest()


def route_spec(key,rows,layer,generation):
    s=narrow.route_spec(key,rows,layer,generation)
    source=s['source'].replace('count>12u','count>128u')
    return dict(s,source=source)


def task_spec(rows):
    r=narrow.plan(rows)['occurrences']
    partitions='''
inline uint pp_parts(uint m,uint n,uint k){
 uint target=max(1u,512u/(((n+31u)/32u)*((m+31u)/32u)));
 uint p=min(k/32u,target);while(p>1u&&k%(p*32u))--p;return p;
}
'''
    status=narrow.task_spec(rows)['source']
    status=status[status.index('for(uint row='):]
    return dict(input_names=['groups','ids','scores'],output_names=['tasks','wide_gu','wide_down','counts','status'],header=partitions,source=f'''
if(thread_position_in_grid.x!=0)return;
uint nc=0,gc=0,dc=0;
for(uint e=0;e<384;++e){{
 const uint count=groups[e*8],kind=groups[e*8+2],vecs=groups[e*8+6],bad=groups[e*8+5];
 if(bad||count==0)continue;
 if(kind==1u&&count>12u){{
  const uint gp=pp_parts(count,2304u,5120u),dp=pp_parts(count,5120u,2304u);
  for(uint tile=0;tile<(count+31u)/32u;++tile){{
   for(uint part=0;part<gp;++part)for(uint projection=0;projection<2;++projection){{
    wide_gu[gc*4]=e;wide_gu[gc*4+1]=tile;wide_gu[gc*4+2]=part;wide_gu[gc*4+3]=projection;++gc;
   }}
   for(uint part=0;part<dp;++part){{wide_down[dc*3]=e;wide_down[dc*3+1]=tile;wide_down[dc*3+2]=part;++dc;}}
  }}
 }}else for(uint group=0;group<(count+vecs-1u)/vecs;++group){{tasks[nc*2]=e;tasks[nc*2+1]=group;++nc;}}
}}
counts[0]=nc;counts[1]=gc;counts[2]=dc;
'''+status)


def gather_spec(rows):
    r=narrow.plan(rows)['occurrences']
    return dict(input_names=['x','scores','ordered'],output_names=['selected','selected_scores'],header='',source=f'''
const uint column=thread_position_in_grid.x,row=thread_position_in_grid.y;
if(column>=5120u||row>={r}u)return;
const uint location=ordered[row];selected[row*5120u+column]=x[(location/6u)*5120u+column];
if(column==0u)selected_scores[row]=scores[location];
''')


def projection_spec(key,rows,*,down=False):
    from .source_prefill_legacy_tiles import _header
    k,n,maxparts=(2304,5120,3)if down else(5120,2304,5)
    r=rows*6
    source=f'''
const uint job=threadgroup_position_in_grid.z;if(job>=counts[{2 if down else 1}])return;
const uint expert=jobs[job*{3 if down else 4}],row_tile=jobs[job*{3 if down else 4}+1],part=jobs[job*{3 if down else 4}+2];
const uint count=groups[expert*8],start=groups[expert*8+1],bank=groups[expert*8+3],offset=groups[expert*8+4];
uint partitions=max(1u,512u/((({n}u+31u)/32u)*((count+31u)/32u)));
partitions=min({k}u/32u,partitions);while(partitions>1u&&{k}u%(partitions*32u))--partitions;
const uint projection={0 if down else 'jobs[job*4+3]'};
'''+_bank_select(len(key))+f'''
const int input_width={k},output_width={n},local_rows=int(count),partition_width={k}/int(partitions);
const int k_start=int(part)*partition_width;
const device uint32_t* w=(const device uint32_t*)(base+SB_OFFSETS[bank][{6 if down else 'projection==0u?0:3'}]+k_start/2);
const device uchar* scales=base+SB_OFFSETS[bank][{7 if down else 'projection==0u?1:4'}]+k_start/32;
const device float* selected_x=x+start*{k}+k_start;
device float* selected_y={'partials'if down else '(projection==0u?gate_partials:up_partials)'}+part*{r}*{n}+start*{n};
threadgroup float Xs[32*36],Ws[32*36];
mlx2_source_legacy_tiles::fp_qmm_t_impl<float,32,4,true,float,false,32,32,32>(
 w,scales,nullptr,selected_x,selected_y,Xs,Ws,input_width,output_width,local_rows,partition_width,
 uint3(threadgroup_position_in_grid.x,row_tile,part),thread_index_in_threadgroup,
 simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
'''
    header=_header()
    if header.count('const constant int&')!=4:raise RuntimeError('Pinned wide PP dimension signature changed')
    header=header.replace('const constant int&','const int')
    return dict(input_names=['x','groups','jobs','counts']+[f'bank{i}'for i in range(len(key))],
        output_names=['partials']if down else['gate_partials','up_partials'],header=header+_constants(key),source=source,
        output_shape=(maxparts,r,n),n=n)


def middle_spec(rows,spelling):
    from .source_gate_up_fusion import _EPILOGUE,EXP_SPELLINGS
    if spelling not in EXP_SPELLINGS:raise ValueError('Unknown PP served sigmoid')
    r=rows*6
    source=f'''
const uint row=threadgroup_position_in_grid.y,lane=thread_index_in_simdgroup,tile=threadgroup_position_in_grid.x;
const uint location=ordered[row],expert=ids[location];
const uint index=row*2304u+tile*32u+lane;
if(expert>=384u){{output[index]=NAN;return;}}
const uint count=groups[expert*8],kind=groups[expert*8+2];
if(kind!=1u||count<=12u){{output[index]=narrow_middle[index];return;}}
uint partitions=max(1u,512u/(72u*((count+31u)/32u)));
partitions=min(160u,partitions);while(partitions>1u&&5120u%(partitions*32u))--partitions;
volatile float g=partitions==1u?gate_partials[index]:0.0f;
volatile float u=partitions==1u?up_partials[index]:0.0f;
#pragma clang loop unroll(disable)
for(uint part=partitions==1u?1u:0u;part<partitions;++part){{g=g+gate_partials[part*{r}u*2304u+index];u=u+up_partials[part*{r}u*2304u+index];}}
const float value=source_gu_weighted(g,u,scores[row]);
float maximum=abs(value);for(uint step=16;step;step/=2)maximum=max(maximum,simd_shuffle_xor(maximum,step));
output[index]=source_gu_qat(value,maximum,as_type<float>(0x3b124925u));
'''
    return dict(input_names=['narrow_middle','gate_partials','up_partials','groups','ordered','ids','scores'],
        output_names=['output'],header=_EPILOGUE.replace('__EXP__',spelling),source=source)


def down_reduce_spec(rows):
    r=rows*6
    return dict(input_names=['narrow_down','partials','groups','ordered','ids'],output_names=['output'],header='',source=f'''
const uint column=thread_position_in_grid.x,row=thread_position_in_grid.y;if(column>=5120u||row>={r}u)return;
const uint expert=ids[ordered[row]],index=row*5120u+column;
if(expert>=384u){{output[index]=NAN;return;}}
const uint count=groups[expert*8],kind=groups[expert*8+2];
if(kind!=1u||count<=12u){{output[index]=narrow_down[index];return;}}
uint partitions=max(1u,512u/(160u*((count+31u)/32u)));
partitions=min(72u,partitions);while(partitions>1u&&2304u%(partitions*32u))--partitions;
volatile float value=partitions==1u?partials[index]:0.0f;
#pragma clang loop unroll(disable)
for(uint part=partitions==1u?1u:0u;part<partitions;++part)value=value+partials[part*{r}u*5120u+index];
output[index]=float(bfloat(value));
''')


@lru_cache(maxsize=128)
def _compiled_kernel(which,key,rows,layer,generation,spelling):
    import mlx.core as mx
    s=(route_spec(key,rows,layer,generation)if which=='route'else task_spec(rows)if which=='tasks'
       else gather_spec(rows)if which=='gather'else projection_spec(key,rows)if which=='gu_wide'
       else projection_spec(key,rows,down=True)if which=='down_wide'else middle_spec(rows,spelling)if which=='middle'
       else down_reduce_spec(rows))
    s={k:s[k]for k in ('input_names','output_names','header','source')}
    digest=hashlib.sha256(repr((which,key,rows,layer,generation,spelling)).encode()).hexdigest()[:16]
    return mx.fast.metal_kernel(name='mlx2_source_pp_wide_'+which+'_'+digest,
        ensure_row_contiguous=which not in ('gu_wide','down_wide'),**s)


def _kernel(which,key,rows,layer,generation,spelling):
    if which!='route':layer,generation=0,0
    return _compiled_kernel(which,key,rows,layer,generation,spelling)


def bank_prefill_wide(x,ids,scores,lease,*,layer,expected_generation,check_live,exp_spelling=None,return_middle=False,task_policy='padded_v1'):
    """Candidate wide extension; complete native-stock raw oracle is required."""
    import mlx.core as mx
    from .source_resident_prefill_tasks import validate_policy,task_grid,kernel as persistent_kernel
    validate_policy(task_policy)
    rows=x.shape[0]if getattr(x,'ndim',0)==2 else 0
    d=narrow.plan(rows);r=d['occurrences']
    if not callable(check_live)or check_live()is not True or mx.default_device()!=mx.gpu:raise ValueError('Wide PP needs owned GPU')
    if (x.shape!=(rows,5120)or x.dtype!=mx.float32 or ids.shape not in ((rows,6),(r,))or ids.dtype!=mx.uint32
            or scores.shape!=ids.shape or scores.dtype!=mx.float32 or type(layer)is not int or not 0<=layer<40
            or type(expected_generation)is not int or not 0<expected_generation<1<<32):raise ValueError('Wide PP geometry mismatch')
    if exp_spelling is None:
        from .source_middle_fusion import exp_spelling as served
        exp_spelling=served()
    key=layout_key(lease)
    if len(key)>23:raise ValueError('Wide PP exceeds finite bank binding limit')
    ids=ids.reshape(-1);scores=scores.reshape(-1);args=(key,rows,layer,expected_generation,exp_spelling)
    groups,ordered,ranks=_kernel('route',*args)(inputs=[ids,lease.lut],grid=(384,1,1),threadgroup=(128,1,1),
        output_shapes=[(384,8),(r,),(r,)],output_dtypes=[mx.uint32]*3)
    tasks,wgu,wdown,counts,status=_kernel('tasks',*args)(inputs=[groups,ids,scores],grid=(1,1,1),threadgroup=(1,1,1),
        output_shapes=[(r,2),(r,4),(r,3),(3,),(rows,)],output_dtypes=[mx.uint32]*5)
    selected,selected_scores=_kernel('gather',*args)(inputs=[x,scores,ordered],grid=(5120,r,1),threadgroup=(256,1,1),
        output_shapes=[(r,5120),(r,)],output_dtypes=[mx.float32]*2)
    nk=lambda which:(narrow._kernel(which,*args)if task_policy=='padded_v1'else persistent_kernel(which,key,rows,exp_spelling))
    wk=lambda which:(_kernel(which,*args)if task_policy=='padded_v1'else persistent_kernel(which,key,rows,exp_spelling))
    narrow_middle=nk('gu')(inputs=[x,scores,groups,ordered,tasks,counts[:1],mx.array([1/448],mx.float32),*lease.anchors],
        grid=(72*256,1,task_grid(task_policy,'gu',r)),threadgroup=(256,1,1),output_shapes=[(r,2304)],output_dtypes=[mx.float32])[0]
    gp,up=wk('gu_wide')(inputs=[selected,groups,wgu,counts,*lease.anchors],grid=(72*128,1,task_grid(task_policy,'gu_wide',r)),threadgroup=(128,1,1),
        output_shapes=[(5,r,2304)]*2,output_dtypes=[mx.float32]*2)
    middle=_kernel('middle',*args)(inputs=[narrow_middle,gp,up,groups,ordered,ids,selected_scores],grid=(72*32,r,1),threadgroup=(32,1,1),
        output_shapes=[(r,2304)],output_dtypes=[mx.float32])[0]
    narrow_down=nk('down')(inputs=[middle,groups,tasks,counts[:1],*lease.anchors],grid=(1280*64,1,task_grid(task_policy,'down',r)),threadgroup=(64,1,1),
        output_shapes=[(r,5120)],output_dtypes=[mx.float32])[0]
    dp=wk('down_wide')(inputs=[middle,groups,wdown,counts,*lease.anchors],grid=(160*128,1,task_grid(task_policy,'down_wide',r)),threadgroup=(128,1,1),
        output_shapes=[(3,r,5120)],output_dtypes=[mx.float32])[0]
    contribution=_kernel('down_reduce',*args)(inputs=[narrow_down,dp,groups,ordered,ids],grid=(5120,r,1),threadgroup=(256,1,1),
        output_shapes=[(r,5120)],output_dtypes=[mx.float32])[0]
    output=narrow._kernel('combine',*args)(inputs=[contribution,ids,ranks,status],grid=(5120,rows,1),threadgroup=(256,1,1),
        output_shapes=[(rows,5120)],output_dtypes=[mx.float32])[0]
    roots=(groups,ordered,ranks,tasks,wgu,wdown,counts,status,selected,selected_scores,narrow_middle,gp,up,middle,narrow_down,dp,contribution,output)
    return narrow.PrefillBankResult(output,status,middle if return_middle else None,groups,roots)
