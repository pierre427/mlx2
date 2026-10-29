# SPDX-License-Identifier: MIT
"""Unwired stock-law F32/MXFP8 projection tile and owner-store candidates.

Only captured Q_b/WoB M1/M99 shapes are admitted. F32 dot arithmetic remains
the pinned installed helper's; each geometry and fused store still needs its
own actual GPU bit gate. No serving/default selection or work on import.
Provenance: deepseek-v41-source-fp8-projection-tiles.json.
"""
from functools import lru_cache

SHAPES={'q_b':(1280,32768),'wo_b':(8192,5120)}
TILES=((32,32),(64,32),(32,64),(16,64))
_STORE=r'''
namespace mlx2_source_fp8_store {
inline float owner_bf16(float x) {
 uint bits=as_type<uint>(x);
 return as_type<float>((bits+0x7fffu+((bits>>16)&1u))&0xffff0000u);
}
struct OwnerBF16 {
 static METAL_FUNC float apply(float x) {return owner_bf16(x);}
 static METAL_FUNC float apply(float x,float) {return owner_bf16(x);}
};
}
'''


def plan(kind,rows,*,tile=(32,32),simdgroups=2,results_per_simd=4,fused_boundary=True):
    if kind not in SHAPES or type(rows)is not int or rows not in (1,99):
        raise ValueError('Source FP8 component requires captured Q_b/WoB M1/M99')
    if type(fused_boundary)is not bool:raise ValueError('Source FP8 store flag must beboolean')
    if (type(tile)is not tuple or len(tile)!=2 or any(type(v)is not int for v in tile)or tile not in TILES
            or type(simdgroups)is not int or simdgroups not in (2,4,8)
            or type(results_per_simd)is not int or results_per_simd not in (4,8)):
        raise ValueError('Source FP8 component unknown tile/SIMD geometry')
    k,n=SHAPES[kind]
    if rows==1:
        if tile!=(32,32):raise ValueError('Decode uses vector geometry, not matrix tile')
        groups=(1,n//(simdgroups*results_per_simd),1);tg=(32,simdgroups,1)
        more=dict(algorithm='qmv_fast',k_step=256,k_steps=k//256,threadgroup_bytes=0,
            virtual_stock_groups_per_group=simdgroups//2,results_per_simd=results_per_simd)
    else:
        if simdgroups!=2 or results_per_simd!=4:
            raise ValueError('PP uses fixed WM2/WN2 helper geometry')
        bm,bn=tile;groups=(n//bn,(rows+bm-1)//bm,1);tg=(32,2,2)
        more=dict(algorithm='qmm_t',k_step=8,k_steps=k//8,threadgroup_bytes=(bm+bn)*36*4,
            tile=tile,bk=32,wm=2,wn=2,m_tiles=(rows+bm-1)//bm,
            logical_weight_tile_read_bytes=((rows+bm-1)//bm)*n*k,
            accumulator_f32_values_per_thread=(bm//16)*(bn//16)*2)
    return dict(kind=kind,rows=rows,k=k,n=n,output_shape=(1,rows,n),threadgroups=groups,
        threadgroup=tg,grid=tuple(a*b for a,b in zip(groups,tg)),fused_boundary=fused_boundary,
        input_dtype='float32',weight_format='originalMXFP8E4M3/E8M0g32',output_dtype='float32',
        coefficient_repack_bytes=0,full_weight_dequantization_bytes=0,input_bindings=3,**more)


def _once(text,old,new):
    if text.count(old)!=1:raise RuntimeError('Source FP8 pinned helper anchor changed: '+old)
    return text.replace(old,new)


@lru_cache(maxsize=4)
def _header(algorithm,fused_boundary):
    from .source_prefill_legacy_tiles import _header as legacy_header,_installed
    _,headers=_installed();base=legacy_header()
    # _STORE is inserted after steel definitions but before the projection
    # helper namespace. Its independent integer bit operation never changes
    # dot-product contraction or operand storage.
    marker='namespace mlx2_source_legacy_tiles {'
    base=_once(base,marker,_STORE+'\n'+marker)
    if algorithm=='qmm_t':
        if fused_boundary:
            old='BlockMMA<T, U, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;'
            new='BlockMMA<T, U, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded, float, mlx2_source_fp8_store::OwnerBF16>;'
            base=_once(base,old,new)
        return base
    if algorithm!='qmv_fast':raise ValueError('Source FP8 unknown helper algorithm')
    text=(headers/'fp_quantized.h').read_text()
    begin=text.index('template <\n    typename T,\n    int group_size,\n    int bits,\n    bool partial_rows = false,')
    end=text.index('\ntemplate <typename T, int group_size, int bits, bool has_global_scale = false>\nMETAL_FUNC void fp_qmv_impl(',begin)
    helper=text[begin:end]
    if fused_boundary:helper=_once(helper,'y[row] = static_cast<T>(result[row]);',
        'y[row] = static_cast<T>(mlx2_source_fp8_store::owner_bf16(result[row]));')
    return base+'\nnamespace mlx2_source_fp8_vector {\nusing namespace mlx2_source_legacy_tiles;\n'+helper+'\n}\n'


def kernel_spec(kind,rows,*,tile=(32,32),simdgroups=2,results_per_simd=4,fused_boundary=True):
    p=plan(kind,rows,tile=tile,simdgroups=simdgroups,results_per_simd=results_per_simd,fused_boundary=fused_boundary)
    constants=f'\nnamespace mlx2_source_fp8_constants {{constant int K={p["k"]},N={p["n"]},M={rows};}}\n'
    source='using namespace mlx2_source_fp8_constants;\n'
    if rows==1:
        source+=f'''
const uint sg=simdgroup_index_in_threadgroup,lane=thread_index_in_simdgroup;
const uint stock_group=threadgroup_position_in_grid.y*{simdgroups//2}+sg/2;
mlx2_source_fp8_vector::fp_qmv_fast_impl<float,32,8,false,false,{results_per_simd}>(
 q,scales,nullptr,x,out,K,N,uint3(0,stock_group,0),sg%2,lane);
'''
    else:
        bm,bn=tile
        source+=f'''
threadgroup float Xs[{bm}*36],Ws[{bn}*36];
mlx2_source_legacy_tiles::fp_qmm_t_impl<float,32,8,true,float,false,{bm},32,{bn}>(
 q,scales,nullptr,x,out,Xs,Ws,K,N,M,K,threadgroup_position_in_grid,
 thread_index_in_threadgroup,simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
'''
    return dict(p,header=_header(p['algorithm'],fused_boundary)+constants,source=source,
        input_names=['x','q','scales'],output_names=['out'])


@lru_cache(maxsize=96)
def _kernel(kind,rows,tile,simdgroups,results_per_simd,fused_boundary):
    import mlx.core as mx
    s=kernel_spec(kind,rows,tile=tile,simdgroups=simdgroups,results_per_simd=results_per_simd,fused_boundary=fused_boundary)
    return mx.fast.metal_kernel(name=f'mlx2_source_fp8_{kind}_m{rows}_tile{tile[0]}x{tile[1]}_s{simdgroups}_r{results_per_simd}_f{int(fused_boundary)}',
        input_names=s['input_names'],output_names=s['output_names'],header=s['header'],source=s['source'],ensure_row_contiguous=True)


def geometry(x,q,scales,kind,**options):
    import mlx.core as mx
    if (kind not in SHAPES or x.ndim!=3 or x.shape[0]!=1 or x.dtype!=mx.float32):
        raise ValueError('Source FP8 component expects preparedF32 B1 activation')
    k,n=SHAPES[kind];p=plan(kind,x.shape[1],**options)
    if (x.shape[-1]!=k or q.shape!=(n,k//4)or q.dtype!=mx.uint32
            or scales.shape!=(n,k//32)or scales.dtype!=mx.uint8):
        raise ValueError('Source FP8 original coefficient/input geometry mismatch')
    return p


def project(x,q,scales,kind,*,tile=(32,32),simdgroups=2,results_per_simd=4,fused_boundary=True):
    """Prepared F32 → rawF32 or owner-BF16-rounded F32, without input QAT."""
    import mlx.core as mx
    p=geometry(x,q,scales,kind,tile=tile,simdgroups=simdgroups,results_per_simd=results_per_simd,fused_boundary=fused_boundary)
    from .source_dense_legacy import _device_supported
    if not _device_supported():raise ValueError('Source FP8 component needs caller-owned M5Max/TF32off GPU')
    return _kernel(kind,p['rows'],tile,simdgroups,results_per_simd,fused_boundary)(
        inputs=[x,q,scales],grid=p['grid'],threadgroup=p['threadgroup'],
        output_shapes=[p['output_shape']],output_dtypes=[mx.float32])[0]


def compilation_source(kind,rows,**options):
    s=kernel_spec(kind,rows,**options)
    return '#include "mlx/backend/metal/kernels/utils.h"\n'+s['header']+'''
kernel void source_fp8_probe(const device float* x [[buffer(0)]],
const device uint* q [[buffer(1)]],const device uchar* scales [[buffer(2)]],
device float* out [[buffer(3)]],uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
'''+s['source']+'\n}\n'
