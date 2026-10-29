# SPDX-License-Identifier: MIT
"""Default-off GPU-routed Source bank decode. No host router-ID readback.

Exact existing Source GU/middle/DOWN helpers; dynamic bank selection and route
ordering are new. Bank publication validates full payload spans; 16-word alias
bindings retain the same Metal allocation while avoiding size-based flushes.
Provenance: provenance/deepseek-v41-source-resident-bank-gather.json.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path

PROFILE='source_resident_bank_gather_decode_v1'
MAX_BANKS=24
FIELDS=('gate.qweight','gate.scales','gate.biases','up.qweight','up.scales','up.biases',
        'down.qweight','down.scales','down.biases')


def implementation_identity():
    paths=(Path(__file__),Path(__file__).with_name('source_gate_up_fusion.py'),
           Path(__file__).with_name('source_down_fusion.py'),Path(__file__).with_name('source_mxfp4_pair.py'))
    return hashlib.sha256(''.join(p.name+hashlib.sha256(p.read_bytes()).hexdigest()for p in paths).encode()).hexdigest()


@dataclass
class BankDecodeResult:
    output: object
    status: object
    middle: object
    route_meta: object


def layout_key(lease):
    """CPU descriptors only; never retain bank arrays in a kernel cache."""
    banks,anchors,layouts=tuple(lease.banks),tuple(lease.anchors),tuple(lease.bank_layouts)
    if not 1<=len(banks)<=MAX_BANKS or len(anchors)!=len(banks)or len(layouts)!=len(banks):
        raise ValueError('Source resident gather requires1..24 complete bank roots and anchors')
    key=[]
    for bank,anchor,layout,kind in zip(banks,anchors,layouts,lease.bank_kinds):
        if layout.kind!=kind or kind not in ('original','q1')or bank.ndim!=1 or anchor.shape!=(16,):
            raise ValueError('Source resident bank layout/alias geometry mismatch')
        offsets={item.name:item.offset for item in layout.tensors}
        key.append((1 if kind=='original'else 2,int(bank.shape[0]),layout.nbytes//4,
                    tuple(int(offsets.get(name,0))for name in FIELDS)))
    if len(key)!=len(banks):raise ValueError('Source resident bank kinds inventory mismatch')
    return tuple(key)


def _constants(key):
    n=len(key)
    kinds=','.join(str(row[0])+'u'for row in key)
    words=','.join(str(row[1])+'ul'for row in key)
    payload=','.join(str(row[2])+'ul'for row in key)
    offsets=','.join('{'+','.join(str(v)+'ul'for v in row[3])+'}'for row in key)
    return f'constant uint SB_KIND[{n}]={{{kinds}}};\nconstant ulong SB_WORDS[{n}]={{{words}}};\nconstant ulong SB_PAYLOAD[{n}]={{{payload}}};\nconstant ulong SB_OFFSETS[{n}][9]={{{offsets}}};\n'


def route_spec(key,layer):
    n=len(key)
    return dict(input_names=['ids','scores','lut','expected_generation'],output_names=['route_meta','route_status'],
        header=_constants(key),source=f'''
if(thread_position_in_grid.x!=0)return;
uint e[6],slots[6];
for(uint i=0;i<6;++i){{e[i]=ids[i];slots[i]=i;}}
for(uint i=1;i<6;++i){{uint a=e[i],s=slots[i],j=i;while(j>0&&e[j-1]>a){{e[j]=e[j-1];slots[j]=slots[j-1];--j;}}e[j]=a;slots[j]=s;}}
const uint generation=expected_generation[0];
bool duplicate=false;for(uint i=1;i<6;++i)duplicate=duplicate||(e[i]==e[i-1]);
for(uint i=0;i<6;++i){{
 uint status=duplicate?4u:0u;uint kind=0,bank=0,offset=0;
 if(e[i]>=384)status|=2u;
 else{{const uint p=({layer}u*384u+e[i])*4u;kind=lut[p];bank=lut[p+1];offset=lut[p+2];
  if(kind==0)status|=1u;
  if(lut[p+3]!=generation)status|=8u;
  if(bank>={n}u)status|=16u;
  else if(kind!=SB_KIND[bank]||ulong(offset)>SB_WORDS[bank]||SB_PAYLOAD[bank]>SB_WORDS[bank]-ulong(offset))status|=16u;
 }}
 if(!isfinite(scores[slots[i]]))status|=32u;
 route_meta[i*5]=e[i];route_meta[i*5+1]=slots[i];route_meta[i*5+2]=kind;route_meta[i*5+3]=bank;route_meta[i*5+4]=offset;
 route_status[i]=status;
}}
''')


def _bank_select(count):
    return 'const device uint32_t* backing=nullptr;\nswitch(bank){\n'+''.join(
        f'case {i}:backing=bank{i}+ulong(offset);break;\n'for i in range(count))+'}\nconst device uchar* base=(const device uchar*)backing;\n'


def gu_spec(key,exp_spelling='metal::exp'):
    from . import source_gate_up_fusion as gu
    from .source_mxfp4_pair import _source_header
    if exp_spelling not in gu.EXP_SPELLINGS:raise ValueError('Unsupported Source sigmoid spelling')
    header=_source_header(1)+gu._EPILOGUE.replace('__EXP__',exp_spelling)+_constants(key)
    common=gu._COMMON.replace('const uint selected_row=threadgroup_position_in_grid.y;','const uint selected_row=threadgroup_position_in_grid.z;')
    common=common.replace('const uint input_row=rows[selected_row];','const uint input_row=0;')
    pointers='''
const uint bank=route_meta[selected_row*5+3],offset=route_meta[selected_row*5+4];
'''+_bank_select(len(key))+'''
const device uint32_t* gate_q=(const device uint32_t*)(base+SB_OFFSETS[bank][0]);
const device uint32_t* up_q=(const device uint32_t*)(base+SB_OFFSETS[bank][3]);
'''
    original='''const device uint8_t* gate_s=base+SB_OFFSETS[bank][1];
const device uint8_t* up_s=base+SB_OFFSETS[bank][4];\n'''+gu._ORIGINAL
    q1='''const device half* gate_s=(const device half*)(base+SB_OFFSETS[bank][1]);
const device half* gate_b=(const device half*)(base+SB_OFFSETS[bank][2]);
const device half* up_s=(const device half*)(base+SB_OFFSETS[bank][4]);
const device half* up_b=(const device half*)(base+SB_OFFSETS[bank][5]);\n'''+gu._Q1
    store=gu._STORE.replace('scores[selected_row]','scores[route_meta[selected_row*5+1]]')
    # One tile owns each finite flag; no inter-thread status races.
    store=store.replace('out[selected_row*2304+tile*32+lane]=source_gu_qat(value,maximum,inverse[0]);',
        '''const float result=source_gu_qat(value,maximum,inverse[0]);
    out[selected_row*2304+tile*32+lane]=result;
    const bool valid=simd_all(isfinite(result));
    if(lane==0)finite_tiles[selected_row*72+tile]=valid?0u:64u;''')
    source=common+'''
if(route_status[selected_row]!=0){
 if(sg==0){out[selected_row*2304+tile*32+lane]=NAN;if(lane==0)finite_tiles[selected_row*72+tile]=0;}
 return;
}
'''+pointers+'if(route_meta[selected_row*5+2]==1u){\n'+original+'\n}else{\n'+q1+'\n}\n'+store
    return dict(input_names=['x','scores','route_meta','route_status','inverse']+[f'bank{i}'for i in range(len(key))],
        output_names=['out','finite_tiles'],header=header,source=source)


def status_spec():
    return dict(input_names=['route_status','finite_tiles'],output_names=['status'],header='',source='''
const uint i=thread_position_in_grid.x;if(i>=6)return;
uint result=route_status[i];for(uint t=0;t<72;++t)result|=finite_tiles[i*72+t];status[i]=result;
''')


def down_spec(key):
    from . import source_down_fusion as down
    header=down._original_header()+down._q1_header()+_constants(key)
    source='''
const uint tile=threadgroup_position_in_grid.x;
const uint sg=simdgroup_index_in_threadgroup,lane=thread_index_in_simdgroup;
const uint virtual_tile=tile/2,virtual_sg=tile%2;
threadgroup float partials[24];threadgroup float* sink=partials+sg*4;
const uint bank=route_meta[sg*5+3],offset=route_meta[sg*5+4];
if(status[sg]!=0){if(lane<4)sink[lane]=NAN;}
else{
'''+_bank_select(len(key))+'''
 const device float* xp=middle+sg*2304;
 const device uint32_t* q=(const device uint32_t*)(base+SB_OFFSETS[bank][6]);
 if(route_meta[sg*5+2]==1u){
  const device uint8_t* s=base+SB_OFFSETS[bank][7];
  mlx2_source_down_stock::fp_qmv_to_shared_impl<float,32,4,false>(q,s,nullptr,xp,sink,mlx2_source_down_stock::K,mlx2_source_down_stock::N,uint3(0,virtual_tile,0),virtual_sg,lane);
 }else{
  const device half* s=(const device half*)(base+SB_OFFSETS[bank][7]);
  const device half* b=(const device half*)(base+SB_OFFSETS[bank][8]);
  mlx2_source_q1_down_shared(xp,q,s,b,sink,virtual_tile,virtual_sg,lane);
 }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0&&lane<4){
 volatile float total=0.0f;
 #pragma clang loop unroll(disable)
 for(uint item=0;item<6;++item){const float raw=partials[item*4+lane];total=total+float(bfloat(raw));}
 output[tile*4+lane]=total;
}
'''
    return dict(input_names=['middle','route_meta','status']+[f'bank{i}'for i in range(len(key))],
        output_names=['output'],header=header,source=source)


@lru_cache(maxsize=128)
def _kernel(which,key,layer=0,exp_spelling='metal::exp'):
    import mlx.core as mx
    spec=route_spec(key,layer)if which=='route'else gu_spec(key,exp_spelling)if which=='gu'else down_spec(key)if which=='down'else status_spec()
    identity=hashlib.sha256(repr((which,key,layer,exp_spelling)).encode()).hexdigest()[:16]
    return mx.fast.metal_kernel(name='mlx2_source_bank_'+which+'_'+identity,
        ensure_row_contiguous=(which in ('route','status')),**spec)


def bank_decode(x,ids,scores,lease,*,layer,expected_generation,check_live,exp_spelling=None,return_middle=False):
    """Lazy GPU decode; caller retains all bank roots through the whole forward."""
    import mlx.core as mx
    if not callable(check_live)or check_live()is not True:raise ValueError('Source bank decode needs live ownership')
    if mx.default_device()!=mx.gpu:raise ValueError('Source bank decode requires owned GPU')
    if (x.shape!=(1,5120)or x.dtype!=mx.float32 or ids.shape!=(6,)or ids.dtype!=mx.uint32
            or scores.shape!=(6,)or scores.dtype!=mx.float32 or type(layer)is not int or not 0<=layer<40
            or type(expected_generation)is not int or not 0<expected_generation<1<<32):
        raise ValueError('Source resident bank decode geometry mismatch')
    if any(bank.dtype!=mx.uint32 or anchor.dtype!=mx.uint32 for bank,anchor in zip(lease.banks,lease.anchors)):
        raise ValueError('Source resident bank binding dtype mismatch')
    key=layout_key(lease)
    if lease.lut.shape!=(40,384,4)or lease.lut.dtype!=mx.uint32:raise ValueError('Source bank LUT geometry mismatch')
    if exp_spelling is None:
        from .source_middle_fusion import exp_spelling as current_exp
        exp_spelling=current_exp()
    generation=mx.array([expected_generation],mx.uint32)
    meta,route_status=_kernel('route',key,layer)(inputs=[ids,scores,lease.lut,generation],grid=(1,1,1),threadgroup=(1,1,1),
        output_shapes=[(6,5),(6,)],output_dtypes=[mx.uint32,mx.uint32])
    mid,finite=_kernel('gu',key,exp_spelling=exp_spelling)(inputs=[x,scores,meta,route_status,mx.array([1/448],mx.float32),*lease.anchors],
        grid=(72*256,1,6),threadgroup=(256,1,1),output_shapes=[(6,2304),(6,72)],output_dtypes=[mx.float32,mx.uint32])
    status=_kernel('status',())(inputs=[route_status,finite],grid=(6,1,1),threadgroup=(32,1,1),output_shapes=[(6,)],output_dtypes=[mx.uint32])[0]
    output=_kernel('down',key)(inputs=[mid,meta,status,*lease.anchors],grid=(1280*192,1,1),threadgroup=(192,1,1),
        output_shapes=[(1,5120)],output_dtypes=[mx.float32])[0]
    return BankDecodeResult(output,status,mid if return_middle else None,meta)
