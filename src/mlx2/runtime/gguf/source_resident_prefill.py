# SPDX-License-Identifier: MIT
"""GPU expert-major Source PP routing with existing M1/M2–12 reduction laws.

Explicit component/adapter seam, default-off. All router IDs and row/score maps
stay on the GPU. Original groups reuse decoded weights across their existing
vector group. An original occupancy >12 returns status128, never silently
changes its QMM law. Provenance: deepseek-v41-source-resident-prefill.json.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path

from .source_resident_bank_gather import layout_key, _constants, _bank_select

MAX_ROWS=128
UNSUPPORTED_ORIGINAL_BAND=128


def plan(rows):
    if type(rows)is not int or not 2<=rows<=MAX_ROWS:raise ValueError('Resident PP admits rows2..128')
    r=rows*6
    return dict(rows=rows,occurrences=r,max_tasks=r,expert_rows=384,
        metadata_bytes=(384*8+r*4+r+1+384+r)*4,middle_bytes=r*2304*4,
        contribution_bytes=r*5120*4,original_occupancy_limit=12,
        grouped_weight_reuse=True,host_route_readbacks=0,qualified=False)


def implementation_identity():
    names=('source_resident_prefill.py','source_gate_up_fusion.py','source_down_fusion.py','source_down_prefill.py','source_resident_prefill_tasks.py')
    return hashlib.sha256(''.join(n+hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()for n in names).encode()).hexdigest()


def route_spec(key,rows,layer,generation):
    d=plan(rows);r=d['occurrences'];n=len(key)
    source=f'''
const uint e=thread_position_in_grid.x;if(e>=384)return;
uint count=0,start=0;
for(uint i=0;i<{r};++i){{const uint v=ids[i];start+=uint(v<e);count+=uint(v==e);}}
const uint p=({layer}u*384u+e)*4u;
uint kind=lut[p],bank=lut[p+1],offset=lut[p+2],status=0;
if(count){{
 if(kind==0)status|=1u;
 if(lut[p+3]!={generation}u)status|=8u;
 if(bank>={n}u)status|=16u;
 else if(kind!=SB_KIND[bank]||ulong(offset)>SB_WORDS[bank]||SB_PAYLOAD[bank]>SB_WORDS[bank]-ulong(offset))status|=16u;
 if(kind==1u&&count>12u)status|=128u;
}}
const uint vecs=kind==1u&&count>1u?(count+(count+4u)/5u-1u)/((count+4u)/5u):1u;
groups[e*8]=count;groups[e*8+1]=start;groups[e*8+2]=kind;groups[e*8+3]=bank;
groups[e*8+4]=offset;groups[e*8+5]=status;groups[e*8+6]=vecs;groups[e*8+7]={generation}u;
uint local=0;
for(uint i=0;i<{r};++i)if(ids[i]==e){{ordered[start+local]=i;ranks[i]=start+local;++local;}}
// Invalid-ID tail is initialized without racing any valid expert writer.
if(e==383u)for(uint i=start+count;i<{r};++i)ordered[i]=0u;
'''
    return dict(input_names=['ids','lut'],output_names=['groups','ordered','ranks'],header=_constants(key),source=source)


def task_spec(rows):
    r=plan(rows)['occurrences']
    return dict(input_names=['groups','ids','scores'],output_names=['tasks','task_count','status'],header='',source=f'''
if(thread_position_in_grid.x!=0)return;
uint cursor=0;
for(uint e=0;e<384;++e){{
 const uint count=groups[e*8],vecs=groups[e*8+6],bad=groups[e*8+5];
 if(bad||count==0)continue;
 for(uint group=0;group<(count+vecs-1u)/vecs;++group){{tasks[cursor*2]=e;tasks[cursor*2+1]=group;++cursor;}}
}}
task_count[0]=cursor;
for(uint row=0;row<{rows};++row){{
 uint bad=0;
 for(uint s=0;s<6;++s){{const uint e=ids[row*6+s];
  if(e>=384)bad|=2u;else bad|=groups[e*8+5];
  if(!isfinite(scores[row*6+s]))bad|=32u;
  for(uint t=0;t<s;++t)if(e==ids[row*6+t])bad|=4u;
 }}
 status[row]=bad;
}}
''')


def _gu_pointers():
    return '''
const device uint32_t* gate_q=(const device uint32_t*)(base+SB_OFFSETS[bank][0]);
const device uint32_t* up_q=(const device uint32_t*)(base+SB_OFFSETS[bank][3]);
'''


def gu_spec(key,exp_spelling='metal::exp'):
    from . import source_gate_up_fusion as gu
    from .source_mxfp4_pair import _source_header
    if exp_spelling not in gu.EXP_SPELLINGS:raise ValueError('Unknown served PP sigmoid spelling')
    prefix='''
const uint task=threadgroup_position_in_grid.z;if(task>=task_count[0])return;
const uint expert=tasks[task*2],vector_group=tasks[task*2+1];
const uint count=groups[expert*8],start=groups[expert*8+1],kind=groups[expert*8+2];
const uint bank=groups[expert*8+3],offset=groups[expert*8+4];
'''+_bank_select(len(key))+_gu_pointers()
    one_common=gu._COMMON.replace('threadgroup_position_in_grid.y','vector_group')
    one_common=one_common.replace('rows[selected_row]','ordered[start+selected_row]/6u')
    one_store=gu._STORE.replace('scores[selected_row]','scores[ordered[start+selected_row]]')
    one_store=one_store.replace('out[selected_row*2304','out[(start+selected_row)*2304')
    q1='''const device half* gate_s=(const device half*)(base+SB_OFFSETS[bank][1]);
const device half* gate_b=(const device half*)(base+SB_OFFSETS[bank][2]);
const device half* up_s=(const device half*)(base+SB_OFFSETS[bank][4]);
const device half* up_b=(const device half*)(base+SB_OFFSETS[bank][5]);
'''+one_common+gu._Q1+one_store
    original='''const device uint8_t* gate_s=base+SB_OFFSETS[bank][1];
const device uint8_t* up_s=base+SB_OFFSETS[bank][4];
switch(count){
case 1: {
'''+one_common+gu._ORIGINAL+one_store+'\nbreak;}\n'
    for count in range(2,13):
        vecs=gu.descriptor('original',count)['vecs']
        body=gu._WIDE_ORIGINAL.replace('__ROWS__',str(count)).replace('__VECS__',str(vecs))
        body=body.replace('threadgroup_position_in_grid.y','vector_group')
        body=body.replace('rows[min(vec0+v,GU_ROWS-1)]','ordered[start+min(vec0+v,GU_ROWS-1)]/6u')
        body=body.replace('scores[vec0+v]','scores[ordered[start+vec0+v]]')
        body=body.replace('(vec0+sg)*2304','(start+vec0+sg)*2304')
        original+=f'case {count}: {{\n'+body+'\nbreak;}\n'
    original+='default:return;\n}\n'
    return dict(input_names=['x','scores','groups','ordered','tasks','task_count','inverse']+[f'bank{i}'for i in range(len(key))],
        output_names=['out'],header=_source_header(1)+gu._EPILOGUE.replace('__EXP__',exp_spelling)+_constants(key),
        source=prefix+'if(kind==1u){\n'+original+'\n}else{\n'+q1+'\n}')


def _down_header():
    from .source_down_fusion import _original_header,_q1_header
    from .source_down_prefill import _wide_header
    wide=_wide_header(2)
    a=wide.index('template <typename T, int group_size, int bits, int vecs_per_tg, int k_lanes>')
    b=wide.index('\nconstant int K =',a)
    # Reuse already-defined stock scalar decode helpers, without reintroducing
    # their global headers/constants or changing a dot/reduction statement.
    header=_original_header()+_q1_header()+'\nnamespace mlx2_source_down_stock {\n'+wide[a:b]+'\n}\n'
    # Stock wrappers bind uniform dimensions as constant-address-space
    # references. Here M is a GPU worklist value. Copy the five INTEGER
    # dimension parameters by value; no coefficient/dot/reduction statement
    # changes. This avoids illegal constant/thread reference binding.
    if header.count('const constant int&')!=5:
        raise RuntimeError('Pinned PP down dimension signature changed')
    return header.replace('const constant int&','const int')


def down_spec(key):
    source='''
const uint task=threadgroup_position_in_grid.z;if(task>=task_count[0])return;
const uint expert=tasks[task*2],vector_group=tasks[task*2+1];
const uint count=groups[expert*8],start=groups[expert*8+1],kind=groups[expert*8+2];
const uint bank=groups[expert*8+3],offset=groups[expert*8+4],vecs=groups[expert*8+6];
const uint tile=threadgroup_position_in_grid.x,sg=simdgroup_index_in_threadgroup,lane=thread_index_in_simdgroup;
const int input_width=2304,output_width=5120,local_rows=int(count);
threadgroup float partials[20];
'''+_bank_select(len(key))+'''
const device uint32_t* q=(const device uint32_t*)(base+SB_OFFSETS[bank][6]);
if(kind==2u){
 if(sg==0){const device half* s=(const device half*)(base+SB_OFFSETS[bank][7]);
 const device half* b=(const device half*)(base+SB_OFFSETS[bank][8]);
 mlx2_source_q1_down_shared(middles+(start+vector_group)*2304,q,s,b,partials,tile/2u,tile%2u,lane);}
}else{
 const device uint8_t* s=base+SB_OFFSETS[bank][7];
 if(count==1u){if(sg==0)mlx2_source_down_stock::fp_qmv_to_shared_impl<float,32,4,false>(
 q,s,nullptr,middles+start*2304,partials,input_width,output_width,uint3(0,tile/2u,0),tile%2u,lane);}
 else switch(vecs){
'''
    for vecs in range(1,6):
        source+=f'''case {vecs}u: mlx2_source_down_stock::fp_qmv_wide_to_shared_impl<float,32,4,{vecs},16>(
 q,s,middles+start*2304,partials,input_width,output_width,local_rows,uint3(vector_group,tile,0),sg,lane);break;
'''
    source+='''default:return;
 }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0&&lane<4){for(uint v=0;v<vecs;++v){
 const uint local=vector_group*vecs+v;if(local>=count)continue;
 output[(start+local)*5120+tile*4+lane]=float(bfloat(partials[v*4+lane]));
}}
'''
    return dict(input_names=['middles','groups','tasks','task_count']+[f'bank{i}'for i in range(len(key))],
        output_names=['output'],header=_down_header()+_constants(key),source=source)


def combine_spec(rows):
    return dict(input_names=['contributions','ids','ranks','status'],output_names=['output'],header='',source=f'''
const uint col=thread_position_in_grid.x,row=thread_position_in_grid.y;
if(col>=5120||row>={rows})return;
if(status[row]){{output[row*5120+col]=NAN;return;}}
uint slots[6];for(uint s=0;s<6;++s)slots[s]=s;
for(uint i=1;i<6;++i){{uint a=slots[i],j=i;while(j&&ids[row*6+slots[j-1]]>ids[row*6+a]){{slots[j]=slots[j-1];--j;}}slots[j]=a;}}
volatile float total=0.0f;
#pragma clang loop unroll(disable)
for(uint s=0;s<6;++s)total=total+contributions[ranks[row*6+slots[s]]*5120+col];
output[row*5120+col]=total;
''')


@lru_cache(maxsize=128)
def _compiled_kernel(which,key,rows,layer,generation,spelling):
    import mlx.core as mx
    spec=(route_spec(key,rows,layer,generation)if which=='route'else task_spec(rows)if which=='tasks'
        else gu_spec(key,spelling)if which=='gu'else down_spec(key)if which=='down'else combine_spec(rows))
    digest=hashlib.sha256(repr((which,key,rows,layer,generation,spelling)).encode()).hexdigest()[:16]
    return mx.fast.metal_kernel(name='mlx2_source_resident_pp_'+which+'_'+digest,
        ensure_row_contiguous=which in ('route','tasks','combine'),**spec)


def _kernel(which,key,rows,layer,generation,spelling):
    # Only the route shader embeds layer/generation; all other stages consume
    # those decisions through GPU metadata. Do not compile40 identical copies.
    if which!='route':layer,generation=0,0
    return _compiled_kernel(which,key,rows,layer,generation,spelling)


@dataclass
class PrefillBankResult:
    output: object
    status: object
    middle: object
    route_meta: object
    roots: tuple


def bank_prefill(x,ids,scores,lease,*,layer,expected_generation,check_live,exp_spelling=None,return_middle=False,task_policy='padded_v1'):
    """Lazy expert-major PP. Caller owns a full-forward bank lease and verdict."""
    import mlx.core as mx
    from .source_resident_prefill_tasks import validate_policy,task_grid,kernel as persistent_kernel
    validate_policy(task_policy)
    if not callable(check_live)or check_live()is not True:raise ValueError('Resident PP requires live ownership')
    if mx.default_device()!=mx.gpu:raise ValueError('Resident PP requires owned GPU')
    rows=x.shape[0]if getattr(x,'ndim',0)==2 else 0;d=plan(rows);r=d['occurrences']
    if(x.shape!=(rows,5120)or x.dtype!=mx.float32 or ids.shape not in ((rows,6),(r,))or ids.dtype!=mx.uint32
       or scores.shape!=ids.shape or scores.dtype!=mx.float32 or type(layer)is not int or not 0<=layer<40
       or type(expected_generation)is not int or not 0<expected_generation<1<<32):raise ValueError('Resident PP geometry mismatch')
    if exp_spelling is None:
        from .source_middle_fusion import exp_spelling as served
        exp_spelling=served()
    key=layout_key(lease)
    # Seven GU inputs plus one output leave at most23 bank bindings in the
    # existing31-buffer Metal contract. The ~67GB PP fixture fits22 at3GiB.
    if len(key)>23:raise ValueError('Resident PP exceeds23 finite bank bindings')
    if any(b.dtype!=mx.uint32 or a.dtype!=mx.uint32 for b,a in zip(lease.banks,lease.anchors)):
        raise ValueError('Resident PP bank dtype mismatch')
    ids=ids.reshape(-1);scores=scores.reshape(-1)
    if lease.lut.shape!=(40,384,4)or lease.lut.dtype!=mx.uint32:raise ValueError('Resident PP LUT mismatch')
    args=(key,rows,layer,expected_generation,exp_spelling)
    groups,ordered,ranks=_kernel('route',*args)(inputs=[ids,lease.lut],grid=(384,1,1),threadgroup=(128,1,1),
        output_shapes=[(384,8),(r,),(r,)],output_dtypes=[mx.uint32]*3)
    tasks,count,status=_kernel('tasks',*args)(inputs=[groups,ids,scores],grid=(1,1,1),threadgroup=(1,1,1),
        output_shapes=[(r,2),(1,),(rows,)],output_dtypes=[mx.uint32]*3)
    gu_kernel=(_kernel('gu',*args)if task_policy=='padded_v1'else persistent_kernel('gu',key,rows,exp_spelling))
    down_kernel=(_kernel('down',*args)if task_policy=='padded_v1'else persistent_kernel('down',key,rows,exp_spelling))
    middle=gu_kernel(inputs=[x,scores,groups,ordered,tasks,count,mx.array([1/448],mx.float32),*lease.anchors],
        grid=(72*256,1,task_grid(task_policy,'gu',r)),threadgroup=(256,1,1),output_shapes=[(r,2304)],output_dtypes=[mx.float32])[0]
    contributions=down_kernel(inputs=[middle,groups,tasks,count,*lease.anchors],grid=(1280*64,1,task_grid(task_policy,'down',r)),
        threadgroup=(64,1,1),output_shapes=[(r,5120)],output_dtypes=[mx.float32])[0]
    output=_kernel('combine',*args)(inputs=[contributions,ids,ranks,status],grid=(5120,rows,1),threadgroup=(256,1,1),
        output_shapes=[(rows,5120)],output_dtypes=[mx.float32])[0]
    return PrefillBankResult(output,status,middle if return_middle else None,groups,
        (groups,ordered,ranks,tasks,count,middle,contributions,status,output))
