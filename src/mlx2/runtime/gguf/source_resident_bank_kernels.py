# SPDX-License-Identifier: MIT
"""Default-off Source resident scheduling variants; baseline remains available.

Only dispatch geometry and metadata location change. Exact Source dot loops,
QAT, score-before-DOWN and ascending BF16 contribution sum are retained.
Provenance: deepseek-v41-source-resident-kernel-variants.json.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
import struct
from . import source_resident_bank_gather as base

VARIANTS=('reference','down8_v1','route_fused_v1','route_down8_v1')


def implementation_identity():
    return hashlib.sha256((Path(__file__).read_bytes()+base.implementation_identity().encode())).hexdigest()


def _once(text,before,after):
    if text.count(before)!=1:raise ValueError('Source kernel scheduling anchor changed: '+before[:90])
    return text.replace(before,after,1)


def down_spec(key,*,tile_rows=8,fused_status=False):
    if tile_rows not in (4,8):raise ValueError('DOWN tile rows must be4or8')
    spec=base.down_spec(key);source=spec['source'];per=tile_rows//4
    source=_once(source,'const uint virtual_tile=tile/2,virtual_sg=tile%2;',
        f'const uint slot=sg/{per};\nconst uint virtual_tile='+('tile/2,virtual_sg=tile%2;'if tile_rows==4 else'tile,virtual_sg=sg%2;'))
    source=_once(source,'threadgroup float partials[24];threadgroup float* sink=partials+sg*4;',
        f'threadgroup float partials[{6*tile_rows}];threadgroup float* sink=partials+slot*{tile_rows}'+(';'if tile_rows==4 else'+(sg%2)*4;'))
    source=source.replace('route_meta[sg*5+','route_meta[slot*5+')
    source=_once(source,'if(status[sg]!=0)', 'if(status[slot]!=0)')
    source=_once(source,'middle+sg*2304','middle+slot*2304')
    source=_once(source,'if(sg==0&&lane<4)',f'if(sg==0&&lane<{tile_rows})')
    source=_once(source,'partials[item*4+lane]',f'partials[item*{tile_rows}+lane]')
    source=_once(source,'output[tile*4+lane]',f'output[tile*{tile_rows}+lane]')
    names=['middle','route_meta','status']
    outputs=['output']
    if fused_status:
        source=source.replace('route_meta[slot*5+','route_meta[slot*6+')
        prefix='''uint local_status=route_meta[slot*6+5];
for(uint t=lane;t<72;t+=32)local_status|=finite_tiles[slot*72+t];
local_status=simd_or(local_status);
'''+f'if(tile==0&&lane==0&&sg%{per}==0)status[slot]=local_status;\n'
        source=_once(source,'if(status[slot]!=0)',prefix+'if(local_status!=0)')
        names=['middle','route_meta','finite_tiles'];outputs=['output','status']
    return dict(input_names=names+[f'bank{i}'for i in range(len(key))],output_names=outputs,header=spec['header'],source=source)


def gu_route_spec(key,layer,generation,exp_spelling='metal::exp'):
    spec=base.gu_spec(key,exp_spelling)
    routing=f'''
threadgroup uint local_meta[6];
if(thread_index_in_threadgroup==0){{
 uint e[6],slots[6];for(uint i=0;i<6;++i){{e[i]=ids[i];slots[i]=i;}}
 for(uint i=1;i<6;++i){{uint a=e[i],s=slots[i],j=i;while(j>0&&e[j-1]>a){{e[j]=e[j-1];slots[j]=slots[j-1];--j;}}e[j]=a;slots[j]=s;}}
 bool duplicate=false;for(uint i=1;i<6;++i)duplicate=duplicate||(e[i]==e[i-1]);
 const uint selected=threadgroup_position_in_grid.z;
 uint flags=duplicate?4u:0u,kind=0,bank=0,offset=0;
 if(e[selected]>=384)flags|=2u;
 else{{const uint p=({layer}u*384u+e[selected])*4u;kind=lut[p];bank=lut[p+1];offset=lut[p+2];
  if(kind==0)flags|=1u;
  if(lut[p+3]!={generation}u)flags|=8u;
  if(bank>={len(key)}u)flags|=16u;
  else if(kind!=SB_KIND[bank]||ulong(offset)>SB_WORDS[bank]||SB_PAYLOAD[bank]>SB_WORDS[bank]-ulong(offset))flags|=16u;
 }}
 if(!isfinite(scores[slots[selected]]))flags|=32u;
 local_meta[0]=e[selected];local_meta[1]=slots[selected];local_meta[2]=kind;
 local_meta[3]=bank;local_meta[4]=offset;local_meta[5]=flags;
}}
threadgroup_barrier(mem_flags::mem_threadgroup);
'''
    source=spec['source']
    for column in range(5):source=source.replace(f'route_meta[selected_row*5+{column}]',f'local_meta[{column}]')
    source=source.replace('route_status[selected_row]','local_meta[5]')
    inverse=struct.unpack('<I',struct.pack('<f',1/448))[0]
    source=source.replace('inverse[0]',f'as_type<float>(0x{inverse:08x}u)')
    publish='''if(tile==0&&sg==0&&lane==0){for(uint c=0;c<6;++c)route_meta[selected_row*6+c]=local_meta[c];}
'''
    source=_once(source,'if(local_meta[5]!=0){',publish+'if(local_meta[5]!=0){')
    return dict(input_names=['x','scores','ids','lut']+[f'bank{i}'for i in range(len(key))],
        output_names=['out','finite_tiles','route_meta'],header=spec['header'],source=routing+source)


@lru_cache(maxsize=256)
def _kernel(which,key,layer,generation,spelling,tile_rows):
    import mlx.core as mx
    spec=gu_route_spec(key,layer,generation,spelling)if which=='gu_route'else down_spec(key,tile_rows=tile_rows,fused_status=which=='down_status')
    identity=hashlib.sha256(repr((which,key,layer,generation,spelling,tile_rows)).encode()).hexdigest()[:18]
    return mx.fast.metal_kernel(name='mlx2_source_bank_variant_'+identity,ensure_row_contiguous=False,**spec)


def decode(x,ids,scores,lease,*,layer,expected_generation,check_live,variant='reference',exp_spelling=None,return_middle=False):
    if variant not in VARIANTS:raise ValueError('Unknown Source expert-kernel variant')
    if variant=='reference':return base.bank_decode(x,ids,scores,lease,layer=layer,expected_generation=expected_generation,
        check_live=check_live,exp_spelling=exp_spelling,return_middle=return_middle)
    import mlx.core as mx
    if not callable(check_live)or check_live()is not True:raise ValueError('Source kernel variant requires live ownership')
    if mx.default_device()!=mx.gpu:raise ValueError('Source kernel variant requires GPU')
    if (x.shape!=(1,5120)or x.dtype!=mx.float32 or ids.shape!=(6,)or ids.dtype!=mx.uint32 or scores.shape!=(6,)or scores.dtype!=mx.float32
            or type(layer)is not int or not 0<=layer<40 or type(expected_generation)is not int or not 0<expected_generation<1<<32):
        raise ValueError('Source kernel variant geometry mismatch')
    key=base.layout_key(lease)
    if lease.lut.shape!=(40,384,4)or lease.lut.dtype!=mx.uint32:raise ValueError('Source kernel variant LUT geometry mismatch')
    if exp_spelling is None:
        from .source_middle_fusion import exp_spelling as current_exp
        exp_spelling=current_exp()
    rows=8 if variant in ('down8_v1','route_down8_v1')else 4
    fused=variant in ('route_fused_v1','route_down8_v1')
    if fused:
        mid,finite,meta=_kernel('gu_route',key,layer,expected_generation,exp_spelling,rows)(
            inputs=[x,scores,ids,lease.lut,*lease.anchors],grid=(72*256,1,6),threadgroup=(256,1,1),
            output_shapes=[(6,2304),(6,72),(6,6)],output_dtypes=[mx.float32,mx.uint32,mx.uint32])
        output,status=_kernel('down_status',key,layer,expected_generation,exp_spelling,rows)(
            inputs=[mid,meta,finite,*lease.anchors],grid=((5120//rows)*(6*32*(rows//4)),1,1),threadgroup=(6*32*(rows//4),1,1),
            output_shapes=[(1,5120),(6,)],output_dtypes=[mx.float32,mx.uint32])
    else:
        meta,route_status=base._kernel('route',key,layer)(inputs=[ids,scores,lease.lut,mx.array([expected_generation],mx.uint32)],grid=(1,1,1),threadgroup=(1,1,1),
            output_shapes=[(6,5),(6,)],output_dtypes=[mx.uint32,mx.uint32])
        mid,finite=base._kernel('gu',key,exp_spelling=exp_spelling)(inputs=[x,scores,meta,route_status,mx.array([1/448],mx.float32),*lease.anchors],
            grid=(72*256,1,6),threadgroup=(256,1,1),output_shapes=[(6,2304),(6,72)],output_dtypes=[mx.float32,mx.uint32])
        status=base._kernel('status',())(inputs=[route_status,finite],grid=(6,1,1),threadgroup=(32,1,1),output_shapes=[(6,)],output_dtypes=[mx.uint32])[0]
        output=_kernel('down',key,layer,expected_generation,exp_spelling,rows)(inputs=[mid,meta,status,*lease.anchors],
            grid=(640*384,1,1),threadgroup=(384,1,1),output_shapes=[(1,5120)],output_dtypes=[mx.float32])[0]
    return base.BankDecodeResult(output,status,mid if return_middle else None,meta)
