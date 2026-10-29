# SPDX-License-Identifier: MIT
"""GPU resident M16 routing; GU coefficient panels shared by two private M1 rows.

No import-time kernel creation. GPU parity remains required for this explicit
unqualified profile. DOWN and ordered combine retain the served B1 laws.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
from .source_resident_bank_gather import layout_key,_constants,_bank_select

ROWS=16
OCCURRENCES=96
GU_REUSE_ROWS=2
MAX_BANKS=22


def plan():
    return dict(rows=ROWS,occurrences=OCCURRENCES,gu_reuse_rows=GU_REUSE_ROWS,
        gu_private_dot_rows=1,down_private_dot_rows=1,max_banks=MAX_BANKS,
        host_expert_id_reads=0,contribution_bytes=96*5120*4,middle_bytes=96*2304*4,
        coefficient_reuse='two occurrences of one expert share packed codes/scales in thread registers',qualified=False)


def implementation_identity():
    names=('source_ast_batched_moe.py','source_resident_prefill.py','source_resident_bank_gather.py',
           'source_gate_up_fusion.py','source_down_fusion.py','source_mxfp4_pair.py')
    return hashlib.sha256(''.join(n+hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()for n in names).encode()).hexdigest()


def route_spec(key,layer,generation):
    from .source_resident_prefill import route_spec as ordinary
    spec=ordinary(key,16,layer,generation);source=spec['source']
    anchor=' if(kind==1u&&count>12u)status|=128u;'
    if source.count(anchor)!=1:raise RuntimeError('Pinned route occupancy guard changed')
    source=source.replace(anchor,'')
    anchor='const uint vecs=kind==1u&&count>1u?(count+(count+4u)/5u-1u)/((count+4u)/5u):1u;'
    if source.count(anchor)!=1:raise RuntimeError('Pinned route vector metadata changed')
    spec=dict(spec,source=source.replace(anchor,'const uint vecs=2u;'))
    return spec


def task_spec():
    from .source_resident_prefill import task_spec as ordinary
    return ordinary(16)


def _cached_qdot(header):
    start=header.index('template <typename U, int values_per_thread, int bits>\ninline U qdot(')
    stop=header.index('template <typename U, int values_per_thread, int bits>',start+20)
    body=header[start:stop]
    if body.count('const device')!=3:raise RuntimeError('Pinned qdot addressspace inventory changed')
    return '\nnamespace mlx2_source_stock_qmv {\n'+body.replace('qdot(', 'qdot_cached(').replace('const device','const thread')+'\n}\n'


def gu_spec(key,spelling='metal::exp'):
    from . import source_gate_up_fusion as gu
    from .source_mxfp4_pair import _source_header
    if spelling not in gu.EXP_SPELLINGS:raise ValueError('Unknown Source sigmoid law')
    header=_source_header(1)
    header+=_cached_qdot(header)+gu._EPILOGUE.replace('__EXP__',spelling)+_constants(key)
    source=r'''
const uint task=threadgroup_position_in_grid.z;if(task>=task_count[0])return;
const uint expert=tasks[task*2],first=tasks[task*2+1]*2u;
const uint count=groups[expert*8],start=groups[expert*8+1];
const uint active=min(2u,count-first),kind=groups[expert*8+2];
const uint bank=groups[expert*8+3],offset=groups[expert*8+4];
const uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
const uint tile=threadgroup_position_in_grid.x,out_row=tile*32+sg*4;
threadgroup float weighted[64];
float gates[2][4]={{0.0f}},ups[2][4]={{0.0f}};
'''+_bank_select(len(key))+r'''
const device uint32_t* gate_q=(const device uint32_t*)(base+SB_OFFSETS[bank][0]);
const device uint32_t* up_q=(const device uint32_t*)(base+SB_OFFSETS[bank][3]);
if(kind==1u){
 using namespace mlx2_source_stock_qmv;
 const device uint8_t* gate_s=base+SB_OFFSETS[bank][1];
 const device uint8_t* up_s=base+SB_OFFSETS[bank][4];
 for(uint k=0;k<5120;k+=512){
  // One coefficient panel load feeds both independent row reduction chains.
  ushort gp[4][4],up[4][4];float gs[4],us[4];
  for(uint r=0;r<4;++r){
   for(uint i=0;i<4;++i){
    gp[r][i]=((const device ushort*)gate_q)[(out_row+r)*1280+k/4+lane*4+i];
    up[r][i]=((const device ushort*)up_q)[(out_row+r)*1280+k/4+lane*4+i];
   }
   gs[r]=dequantize_scale<float,32>(gate_s[(out_row+r)*160+k/32+lane/2]);
   us[r]=dequantize_scale<float,32>(up_s[(out_row+r)*160+k/32+lane/2]);
  }
  for(uint v=0;v<active;++v){
   const uint input_row=ordered[start+first+v]/6u;float xt[16];
   load_vector<float,float,16>(x+input_row*5120+k+lane*16,xt);
   for(uint r=0;r<4;++r){
    gates[v][r]+=qdot_cached<float,16,4>((const thread uchar*)gp[r],xt,gs[r]);
    ups[v][r]+=qdot_cached<float,16,4>((const thread uchar*)up[r],xt,us[r]);
   }
  }
 }
}else{
 const device half* gate_s=(const device half*)(base+SB_OFFSETS[bank][1]);
 const device half* gate_b=(const device half*)(base+SB_OFFSETS[bank][2]);
 const device half* up_s=(const device half*)(base+SB_OFFSETS[bank][4]);
 const device half* up_b=(const device half*)(base+SB_OFFSETS[bank][5]);
 for(uint block_start=lane*16;block_start<5120;block_start+=512){
  ushort gp[4],up[4];float gs[4],gb[4],us[4],ub[4];
  for(uint r=0;r<4;++r){
   const uint row=out_row+r,pi=row*80+block_start/64;
   gp[r]=ushort(gate_q[row*160+(block_start>>5)]>>(block_start&31));
   up[r]=ushort(up_q[row*160+(block_start>>5)]>>(block_start&31));
   gs[r]=float(gate_s[pi]);gb[r]=float(gate_b[pi]);
   us[r]=float(up_s[pi]);ub[r]=float(up_b[pi]);
  }
  for(uint v=0;v<active;++v){
   const uint input_row=ordered[start+first+v]/6u;float xt[16],total_sum=0.0f;
   #pragma clang loop unroll(full)
   for(uint i=0;i<16;++i){const float value=x[input_row*5120+block_start+i];xt[i]=value;total_sum+=value;}
   for(uint r=0;r<4;++r){
    float selected_g=0.0f,selected_u=0.0f;
    #pragma clang loop unroll(full)
    for(uint i=0;i<16;++i){selected_g+=(gp[r]&(ushort(1)<<i))?xt[i]:0.0f;selected_u+=(up[r]&(ushort(1)<<i))?xt[i]:0.0f;}
    gates[v][r]+=selected_g*gs[r];gates[v][r]+=total_sum*gb[r];
    ups[v][r]+=selected_u*us[r];ups[v][r]+=total_sum*ub[r];
   }
  }
 }
}
for(uint v=0;v<active;++v)for(uint r=0;r<4;++r){
 const float g=simd_sum(gates[v][r]),u=simd_sum(ups[v][r]);
 if(lane==0)weighted[v*32+sg*4+r]=source_gu_weighted(g,u,scores[ordered[start+first+v]]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0)for(uint v=0;v<active;++v){
 const float value=weighted[v*32+lane];float maximum=abs(value);
 for(uint step=16;step>0;step/=2)maximum=max(maximum,simd_shuffle_xor(maximum,step));
 const float result=source_gu_qat(value,maximum,inverse[0]);
 out[(start+first+v)*2304+tile*32+lane]=result;
 const bool finite=simd_all(isfinite(result));if(lane==0)finite_tiles[(start+first+v)*72+tile]=finite?0u:64u;
}
'''
    return dict(input_names=['x','scores','groups','ordered','tasks','task_count','inverse']+[f'bank{i}'for i in range(len(key))],
        output_names=['out','finite_tiles'],header=header,source=source)


def status_spec():
    return dict(input_names=['route_status','ranks','finite_tiles'],output_names=['status'],header='',source=r'''
const uint row=thread_position_in_grid.x;if(row>=16)return;
uint flags=route_status[row];
if(flags==0)for(uint slot=0;slot<6;++slot){const uint occurrence=ranks[row*6+slot];for(uint t=0;t<72;++t)flags|=finite_tiles[occurrence*72+t];}
status[row]=flags;
''')


def down_spec(key):
    from . import source_down_fusion as down
    source=r'''
const uint occurrence=threadgroup_position_in_grid.y;
const uint tile=threadgroup_position_in_grid.x,lane=thread_index_in_simdgroup;
const uint valid_count=groups[383*8]+groups[383*8+1];
if(occurrence>=valid_count){if(lane<4)output[occurrence*5120+tile*4+lane]=NAN;return;}
const uint flat=ordered[occurrence];
if(status[flat/6u]!=0){if(lane<4)output[occurrence*5120+tile*4+lane]=NAN;return;}
const uint expert=ids[flat],bank=groups[expert*8+3],offset=groups[expert*8+4];
const uint virtual_tile=tile/2,virtual_sg=tile%2;
threadgroup float partials[4];
'''+_bank_select(len(key))+r'''
const device float* xp=middle+occurrence*2304;
const device uint32_t* q=(const device uint32_t*)(base+SB_OFFSETS[bank][6]);
if(groups[expert*8+2]==1u){
 const device uchar* s=base+SB_OFFSETS[bank][7];
 mlx2_source_down_stock::fp_qmv_to_shared_impl<float,32,4,false>(q,s,nullptr,xp,partials,mlx2_source_down_stock::K,mlx2_source_down_stock::N,uint3(0,virtual_tile,0),virtual_sg,lane);
}else{
 const device half* s=(const device half*)(base+SB_OFFSETS[bank][7]);
 const device half* b=(const device half*)(base+SB_OFFSETS[bank][8]);
 mlx2_source_q1_down_shared(xp,q,s,b,partials,virtual_tile,virtual_sg,lane);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if(lane<4)output[occurrence*5120+tile*4+lane]=float(bfloat(partials[lane]));
'''
    return dict(input_names=['middle','ids','groups','ordered','status']+[f'bank{i}'for i in range(len(key))],
        output_names=['output'],header=down._original_header()+down._q1_header()+_constants(key),source=source)


def combine_spec():
    from .source_resident_prefill import combine_spec as ordinary
    return ordinary(16)


@lru_cache(maxsize=128)
def _kernel(which,key,layer=0,generation=0,spelling='metal::exp'):
    import mlx.core as mx
    spec=(route_spec(key,layer,generation)if which=='route'else task_spec()if which=='tasks'else
          gu_spec(key,spelling)if which=='gu'else down_spec(key)if which=='down'else
          status_spec()if which=='status'else combine_spec())
    identity=hashlib.sha256(repr((which,key,layer,generation,spelling)).encode()).hexdigest()[:18]
    return mx.fast.metal_kernel(name='mlx2_source_ast_bank_'+which+'_'+identity,
        ensure_row_contiguous=which in('route','tasks','status','combine'),**spec)


@dataclass
class TileResult:
    output:object
    status:object
    middle:object
    route_meta:object
    roots:tuple


def decode_tile(x,ids,scores,lease,*,layer,expected_generation,check_live,exp_spelling=None):
    import mlx.core as mx
    if not callable(check_live)or check_live()is not True:raise ValueError('Resident AST MoE requires live whole-tile lease')
    if(x.shape!=(16,5120)or x.dtype!=mx.float32 or ids.shape!=(16,6)or ids.dtype!=mx.uint32
       or scores.shape!=ids.shape or scores.dtype!=mx.float32 or type(layer)is not int or not 0<=layer<40
       or type(expected_generation)is not int or not 0<expected_generation<1<<32):raise ValueError('Resident AST MoE geometry mismatch')
    key=layout_key(lease)
    if len(key)>MAX_BANKS:raise ValueError('Resident AST GU requires at most22 bank bindings')
    if lease.lut.shape!=(40,384,4)or lease.lut.dtype!=mx.uint32:raise ValueError('Resident AST LUT mismatch')
    if any(b.dtype!=mx.uint32 or a.dtype!=mx.uint32 for b,a in zip(lease.banks,lease.anchors)):raise ValueError('Resident AST bank dtype mismatch')
    if mx.default_device()!=mx.gpu:raise ValueError('Resident AST MoE requires parent-owned GPU')
    if exp_spelling is None:
        from .source_middle_fusion import exp_spelling as current
        exp_spelling=current()
    ids=ids.reshape(-1);scores=scores.reshape(-1)
    groups,ordered,ranks=_kernel('route',key,layer,expected_generation)(inputs=[ids,lease.lut],
        grid=(384,1,1),threadgroup=(128,1,1),output_shapes=[(384,8),(96,),(96,)],output_dtypes=[mx.uint32]*3)
    tasks,count,route_status=_kernel('tasks',())(inputs=[groups,ids,scores],grid=(1,1,1),threadgroup=(1,1,1),
        output_shapes=[(96,2),(1,),(16,)],output_dtypes=[mx.uint32]*3)
    middle,finite=_kernel('gu',key,spelling=exp_spelling)(inputs=[x,scores,groups,ordered,tasks,count,mx.array([1/448],mx.float32),*lease.anchors],
        grid=(72*256,1,96),threadgroup=(256,1,1),output_shapes=[(96,2304),(96,72)],output_dtypes=[mx.float32,mx.uint32])
    status=_kernel('status',())(inputs=[route_status,ranks,finite],grid=(16,1,1),threadgroup=(32,1,1),output_shapes=[(16,)],output_dtypes=[mx.uint32])[0]
    parts=_kernel('down',key)(inputs=[middle,ids,groups,ordered,status,*lease.anchors],grid=(1280*32,96,1),threadgroup=(32,1,1),
        output_shapes=[(96,5120)],output_dtypes=[mx.float32])[0]
    output=_kernel('combine',())(inputs=[parts,ids,ranks,status],grid=(5120,16,1),threadgroup=(256,1,1),output_shapes=[(16,5120)],output_dtypes=[mx.float32])[0]
    return TileResult(output,status,middle,groups,(x,ids,scores,groups,ordered,ranks,tasks,count,route_status,middle,finite,status,parts,output))
