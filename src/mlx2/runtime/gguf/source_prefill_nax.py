# SPDX-License-Identifier: Apache-2.0
"""Experimental one-expert, one-projection Source split-K NAX component.

Not wired to serving. Mapped F32 Source-QAT inputs load into BF16 fragments;
decoded MXFP4 tiles are BF16, accumulators and partition stores stay F32.
The legacy K8 and NAX K16 dot laws can differ despite matching partitions.
See provenance/deepseek-v41-source-prefill-nax.json and the oMLX NOTICE there.
"""
from functools import lru_cache
import hashlib
from pathlib import Path

MECHANISM_SHA256 = '62c5522a09bd178ac2218f6eafefdd4fc55382cd10c150f0d67ff73a14c80811'
K, N = 5120, 2304


def plan(rows):
    """CPU geometry only; matching the split is not a numerical proof."""
    if type(rows) is not int or not 13 <= rows <= 96:
        raise ValueError('Source NAX component admits M13..96 only')
    partitions = 5 if rows <= 32 else 2
    return dict(rows=rows, partitions=partitions, k_partition=K//partitions,
                partition_starts=tuple(i*(K//partitions) for i in range(partitions)),
                partial_shape=(partitions, rows, N), partial_bytes=partitions*rows*N*4,
                grid=(N//64*64, (rows+31)//32, partitions), threadgroup=(64,1,1),
                threadgroup_bytes=64*(64+8)*2, input_bindings=4,
                full_weight_repack_bytes=0, partial_dtype='float32',
                profile='experimental-unselected-new-dot-law')


def _replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('Source NAX pinned mechanism anchor changed: '+old)
    return text.replace(old,new)


@lru_cache(maxsize=1)
def _header():
    # Source retrieval and hashing is CPU-only; no kernel is compiled here.
    from .source_dense import _installed
    from .source_mxfp4_pair import _source_header
    from ..models import moe_nax_gather as mined
    path=Path(mined.__file__)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=MECHANISM_SHA256:
        raise RuntimeError('Source NAX local mechanism changed; re-audit required')
    _installed()
    _source_header(1)  # Pins fp4/fp8, legacy dispatch and installed library.
    dependencies=mined._read_mlx_headers(mined._MLX_MM_HEADERS)
    if dependencies is None:
        raise RuntimeError('Source NAX installed headers unavailable')
    text=mined._MM_HEADER.split('// db:',1)[0]
    begin=text.index('template <typename T, short R>')
    end=text.index('// Element offsets',begin)
    mapped=text[begin:end].replace('const device T*','const device float*')
    mapped=mapped.replace('vec<T, 4>', 'vec<float, 4>')
    mapped=_replace_once(mapped,'A.frag_at(r, kk)[h * 4 + c] = v[c];',
                          'A.frag_at(r, kk)[h * 4 + c] = T(v[c]);')
    text=text[:begin]+mapped+text[end:]
    begin=text.index('// seg:')
    helper=text[begin:]
    for old,new in (
        ('void gather_seg(', 'void source_prefill_partition_seg('),
        ('const device T* x,', 'const device float* x,'),
        ('device T* y,', 'device float* y,'),
        ('const int K,', 'const int K,\n    const int k_begin, const int k_count,'),
        ('const int K_it = K / G::kBK;', 'const int K_it = k_count / G::kBK;'),
        ('const int k_remain = K - K_it * G::kBK;', 'const int k_remain = k_count - K_it * G::kBK;'),
        ('q.advance(w_row * K_g);','q.advance(w_row * K_g + k_begin / Q::kGroup);'),
        ('w + w_row * K_w, w_up, K,', 'w + w_row * K_w + k_begin * Q::kBits / 8, w_up, K,'),
        ('if constexpr (!MAP) {', 'x += k_begin;\n  if constexpr (!MAP) {'),
        ('const device T* xn =', 'const device float* xn =')):
        helper=_replace_once(helper,old,new)
    # The component instantiates only EPI0/MAPtrue. Its F32 y pointer must
    # never reach the BF16 activation-store branch of the general helper.
    old='''    if constexpr (kPair) {
      if (sg_active) {
        store_act<T, EPI>(
            Dtile, y + tm * ldy + tn / 2, ldy, int(sgp_sm), limit);
      }
    } else if (kAlignedM.value && sgp_sn == kSN) {'''
    helper=_replace_once(helper,old,'    if (kAlignedM.value && sgp_sn == kSN) {')
    return ('// Adapted oMLX NAX gather mechanism, Apache-2.0; see provenance NOTICE.\n'
            +dependencies+'\n'+text[:begin]+helper+'\n} // namespace omlx_gqmm\n')


def kernel_spec(rows):
    d=plan(rows)
    source=f'''
const uint column=threadgroup_position_in_grid.x*64;
const uint first=threadgroup_position_in_grid.y*32;
const uint part=threadgroup_position_in_grid.z;
const uint sg=simdgroup_index_in_threadgroup, lane=thread_index_in_simdgroup;
const uint4 desc=uint4(first,0,min(uint(32),uint({rows})-first),0);
const device uchar* w=backing+layout_offsets[0];
omlx_gqmm::Mxfp4Q<32> q{{backing+layout_offsets[1]}};
threadgroup bfloat Ws[64*(64+8)];
omlx_gqmm::source_prefill_partition_seg<bfloat,omlx_gqmm::Mxfp4Q<32>,
    omlx_gqmm::Geo<32,64>,true,true,0,true,false>(
    x,rows_map,w,q,desc,column,partials+part*{rows}*2304,
    2304,5120,part*{d['k_partition']},{d['k_partition']},Ws,sg,lane);
'''
    return dict(d, header=_header(), source=source,
                input_names=['x','rows_map','backing','layout_offsets'],output_names=['partials'])


@lru_cache(maxsize=84)
def _kernel(rows):
    import mlx.core as mx
    s=kernel_spec(rows)
    return mx.fast.metal_kernel(name=f'mlx2_source_pp_one_projection_nax_m{rows}',
        input_names=s['input_names'],output_names=s['output_names'],
        header=s['header'],source=s['source'],ensure_row_contiguous=True)


def prepare_inputs(x, expert, rows, projection='gate'):
    import mlx.core as mx
    from ...adapters.deepseek_v41_source_buffers import PackedExpert,ExpertLayout
    if projection not in ('gate','up'):
        raise ValueError('Source NAX component admits gate or up projection only')
    if (not isinstance(expert,PackedExpert) or not isinstance(expert.layout,ExpertLayout)
            or expert.layout.kind!='original'):
        raise ValueError('Source NAX component needs one original PackedExpert')
    if (x.ndim!=2 or x.shape[0]<1 or x.shape[-1]!=K or x.dtype!=mx.float32
            or rows.ndim!=1 or rows.dtype!=mx.uint32 or x.shape[0]*K>=2**32):
        raise ValueError('Source NAX mapped F32 activation/row geometry mismatch')
    d=plan(rows.shape[0])
    if (expert.backing.ndim!=1 or expert.backing.dtype!=mx.uint8
            or expert.backing.shape!=(expert.layout.nbytes,)):
        raise ValueError('Source NAX immutable backing geometry mismatch')
    q=expert.layout.tensor(projection+'.qweight')
    s=expert.layout.tensor(projection+'.scales')
    if (q.shape!=(N,K//8) or q.dtype!='<u4' or s.shape!=(N,K//32) or s.dtype!='|u1'
            or expert[projection+'.qweight'].shape!=q.shape
            or expert[projection+'.qweight'].dtype!=mx.uint32
            or expert[projection+'.scales'].shape!=s.shape
            or expert[projection+'.scales'].dtype!=mx.uint8):
        raise ValueError('Source NAX packed projection geometry mismatch')
    return d,(q.offset,s.offset)


def project(x, expert, rows, *, projection='gate', return_partials=False):
    """One projection; caller supplies valid rows, BF16-grid F32 QAT and lease.

    Returns BF16-rounded F32. Diagnostic tuple is (rounded, raw, F32partials).
    BF16-grid and row contents are trusted caller contracts, not CPU readbacks.
    The returned projection can differ from stock even after BF16 rounding.
    """
    import mlx.core as mx
    if type(return_partials) is not bool:
        raise ValueError('Source NAX diagnostic flag must be boolean')
    d,offsets=prepare_inputs(x,expert,rows,projection)
    from .source_mxfp4_pair import _device_supported
    from ...process_env import tf32_enabled
    if mx.default_device()!=mx.gpu or not _device_supported() or tf32_enabled():
        raise ValueError('Source NAX component needs caller-owned M5 Max GPU window')
    partials=_kernel(d['rows'])(inputs=[x,rows,expert.backing,mx.array(offsets,mx.uint32)],
        grid=d['grid'],threadgroup=d['threadgroup'],output_shapes=[d['partial_shape']],
        output_dtypes=[mx.float32])[0]
    # Stock split-K has contiguous [P,M*N] partition planes and reduces axis0.
    raw=mx.sum(partials.reshape(d['partitions'],-1),axis=0).reshape(d['rows'],N)
    rounded=raw.astype(mx.bfloat16).astype(mx.float32)
    return (rounded,raw,partials) if return_partials else rounded
