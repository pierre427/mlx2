# SPDX-License-Identifier: MIT
"""Bounded immutable Source prefill down batches, not a serving default.

One packed expert backing per binding; input/output rows are consecutive
expert segments. The only moved boundary is the BF16 down-output store.
Original multirow dispatch is a candidate until its raw stock oracle passes.
No expert subtotal, atomics, repack, GPU work or source access on import.
"""
from functools import lru_cache
import operator
import numpy as np

from .source_down_fusion import HIDDEN, INTERMEDIATE
MAX_EXPERTS = 15
MAX_ROWS = 6144


def plan(kind, row_counts):
    if kind not in ('original','q1'):
        raise ValueError('Source PP down unknown expert kind')
    clean=[]
    for v in row_counts:
        try:
            if isinstance(v,bool): raise TypeError
            n=operator.index(v)
        except TypeError as error:
            raise ValueError('Source PP down row counts must be integers') from error
        if not 1 <= n <= 1024:
            raise ValueError('Source PP down local rows must be1..1024')
        clean.append(n)
    if not 1 <= len(clean) <= MAX_EXPERTS or sum(clean)>MAX_ROWS:
        raise ValueError('Source PP down batch/row extent exceeds bound')
    if kind=='original' and (len(set(clean))!=1 or clean[0]>12):
        raise ValueError('Source PP original down requires homogeneous R1..12 candidate')
    local=clean[0] if kind=='original' else 1
    vecs=1 if local==1 else (local+(local+4)//5-1)//((local+4)//5)
    tile=4 if kind=='original' and local>1 else 8
    cursor,jobs,offsets=0,[],[]
    for slot,count in enumerate(clean):
        offsets.append(cursor)
        for group in range((count+vecs-1)//vecs):
            jobs.append((slot,group,cursor))
        cursor+=count
    return dict(kind=kind,row_counts=tuple(clean),row_offsets=tuple(offsets),
        rows=cursor,experts=len(clean),local_rows=local,vecs=vecs,tile=tile,
        jobs=np.asarray(jobs,np.uint32),grid=(HIDDEN//tile*64,len(jobs),1),
        threadgroup=(64,1,1),input_bindings=len(clean)+3,
        output_bytes=cursor*HIDDEN*4,scratch_bytes=vecs*tile*4,
        raw_original_multirow_dispatch_proven=False)


@lru_cache(maxsize=12)
def _wide_header(rows):
    from .source_mxfp4_pair import _source_header
    text=_source_header(rows)
    first=text.index('template <typename T, int group_size, int bits, int vecs_per_tg, int k_lanes>')
    last=text.index('\nconstant int K =')
    helper=text[first:last]
    for before,after in (
        ('fp_qmv_wide_impl(', 'fp_qmv_wide_to_shared_impl('),
        ('    device T* y,', '    threadgroup T* y,'),
        ('y[(vec0 + v) * out_vec_size + out_row]', 'y[v * 4 + out_row - tid.y * 4]')):
        if helper.count(before)!=1:
            raise RuntimeError('Source PP down pinned wide helper anchor changed: '+before)
        helper=helper.replace(before,after)
    text=text[:first]+helper+text[last:]
    return text.replace('constant int K = 5120;','constant int K = 2304;').replace('constant int N = 2304;','constant int N = 5120;')


def kernel_spec(kind, local_rows, expert_count, *, return_raw=False):
    if type(return_raw) is not bool:
        raise ValueError('Source PP down raw flag must be boolean')
    d=plan(kind,[local_rows]*expert_count)
    if kind=='q1' and local_rows!=1:
        raise ValueError('Source PP Q1 uses one-row jobs, not a compiled row bucket')
    from .source_down_fusion import _original_header,_q1_header
    header=(_q1_header() if kind=='q1' else _original_header() if local_rows==1 else _wide_header(local_rows))
    names=['middles','jobs','layout_offsets']+[f'expert{i}' for i in range(expert_count)]
    selects='\n'.join(f'case {i}: backing=expert{i}; break;' for i in range(expert_count))
    declaration=r'''
const uint tile=threadgroup_position_in_grid.x, job=threadgroup_position_in_grid.y;
const uint sg=simdgroup_index_in_threadgroup, lane=thread_index_in_simdgroup;
const uint slot=jobs[job*3], vector_group=jobs[job*3+1], row_base=jobs[job*3+2];
const device uchar* backing=nullptr;
switch(slot) {
'''+selects+r'''
}
const device uint32_t* q=(const device uint32_t*)(backing+layout_offsets[slot*3]);
'''
    if kind=='q1':
        declaration+=r'''
const device half* scales=(const device half*)(backing+layout_offsets[slot*3+1]);
const device half* biases=(const device half*)(backing+layout_offsets[slot*3+2]);
'''
        call='mlx2_source_q1_down_shared(middles+(row_base+vector_group)*2304,q,scales,biases,partials+sg*4,tile,sg,lane);'
    elif local_rows==1:
        declaration+='const device uint8_t* scales=backing+layout_offsets[slot*3+1];\n'
        call='mlx2_source_down_stock::fp_qmv_to_shared_impl<float,32,4,false>(q,scales,nullptr,middles+row_base*2304,partials+sg*4,mlx2_source_down_stock::K,mlx2_source_down_stock::N,uint3(0,tile,0),sg,lane);'
    else:
        declaration+='const device uint8_t* scales=backing+layout_offsets[slot*3+1];\n'
        call=f'mlx2_source_stock_qmv::fp_qmv_wide_to_shared_impl<float,32,4,{d["vecs"]},16>(q,scales,middles+row_base*2304,partials,mlx2_source_stock_qmv::K,mlx2_source_stock_qmv::N,mlx2_source_stock_qmv::M,uint3(vector_group,tile,0),sg,lane);'
    source=declaration+f'threadgroup float partials[{d["vecs"]*d["tile"]}];\n'+call+r'''
threadgroup_barrier(mem_flags::mem_threadgroup);
'''+f'if(sg==0 && lane<{d["tile"]}) {{\nfor(uint v=0;v<{d["vecs"]};++v) {{\n'
    if kind=='original':
        source+=f'if(vector_group*{d["vecs"]}+v>={local_rows})continue;\n'
    source+=f'const uint output_row=row_base+vector_group*{d["vecs"]}+v;\nconst uint index=output_row*5120+tile*{d["tile"]}+lane;\nconst float value=partials[v*{d["tile"]}+lane];\n'
    if return_raw: source+='raw[index]=value;\n'
    source+='output[index]=float(bfloat(value));\n}\n}\n'
    return dict(d,header=header,source=source,input_names=names,
        output_names=['output']+(['raw'] if return_raw else []),
        total_buffer_bindings=expert_count+4+int(return_raw))


@lru_cache(maxsize=128)
def _kernel(kind,local_rows,expert_count,return_raw):
    import mlx.core as mx
    s=kernel_spec(kind,local_rows,expert_count,return_raw=return_raw)
    return mx.fast.metal_kernel(name=f'mlx2_source_pp_down_{kind}_r{local_rows}_e{expert_count}_p{int(return_raw)}',
        input_names=s['input_names'],output_names=s['output_names'],header=s['header'],
        source=s['source'],ensure_row_contiguous=True)


def prepare_inputs(middles, experts, row_counts):
    import mlx.core as mx
    from ...adapters.deepseek_v41_source_buffers import PackedExpert,ExpertLayout
    experts=tuple(experts)
    if not experts or not all(isinstance(e,PackedExpert) and isinstance(e.layout,ExpertLayout) for e in experts):
        raise ValueError('Source PP down requires immutable PackedExpert objects')
    kind=experts[0].layout.kind
    if any(e.layout.kind!=kind for e in experts):
        raise ValueError('Source PP down batch kinds must be homogeneous')
    d=plan(kind,row_counts)
    if len(experts)!=d['experts'] or middles.shape!=(d['rows'],INTERMEDIATE) or middles.dtype!=mx.float32:
        raise ValueError('Source PP down middle/expert segment geometry mismatch')
    offsets=[]
    for expert in experts:
        backing=expert.backing
        if backing.ndim!=1 or backing.dtype!=mx.uint8 or backing.shape!=(expert.layout.nbytes,):
            raise ValueError('Source PP down backing geometry mismatch')
        for name in ('down.qweight','down.scales')+ (('down.biases',) if kind=='q1' else ()):
            t=expert.layout.tensor(name)
            expected={'<u4':mx.uint32,'|u1':mx.uint8,'<f2':mx.float16}[t.dtype]
            if expert[name].shape!=t.shape or expert[name].dtype!=expected:
                raise ValueError('Source PP down packed typed-view mismatch')
        offsets.append(tuple(expert.layout.tensor('down.'+f).offset for f in ('qweight','scales'))+
                       ((expert.layout.tensor('down.biases').offset,) if kind=='q1' else (0,)))
    return d,np.asarray(offsets,np.uint32),experts


def pp_down(middles, experts, row_counts, *, return_raw=False):
    """F32 middle segments -> BF16-rounded F32 contribution segments.

    Normal execution emits no raw global scratch or expert subtotal. The
    caller keeps backing leases and producer middle arrays until completion.
    """
    import mlx.core as mx
    if type(return_raw) is not bool:
        raise ValueError('Source PP down raw flag must be boolean')
    d,offsets,experts=prepare_inputs(middles,experts,row_counts)
    from .source_mxfp4_pair import _device_supported
    if not _device_supported():
        raise ValueError('Source PP down requires a caller-owned M5 Max GPU window')
    result=_kernel(d['kind'],d['local_rows'],d['experts'],return_raw)(
        inputs=[middles,mx.array(d['jobs'],mx.uint32),mx.array(offsets,mx.uint32)]+[e.backing for e in experts],
        grid=d['grid'],threadgroup=d['threadgroup'],
        output_shapes=[(d['rows'],HIDDEN)]*(2 if return_raw else 1),
        output_dtypes=[mx.float32]*(2 if return_raw else 1))
    return tuple(result) if return_raw else result[0]
