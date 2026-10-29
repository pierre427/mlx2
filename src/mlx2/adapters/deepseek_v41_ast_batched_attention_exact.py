# SPDX-License-Identifier: MIT
"""Experimental batched M1-law attention; exactness is an actual-GPU gate.

No native attention policy is changed. A query keeps its ordinary local and
selected chronological reduction lengths. Unequal lengths form bounded GPU
batch classes rather than masked extra columns. Prefix KV is never copied16x.
"""
import hashlib
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar
import mlx.core as mx
from ..runtime.models.deepseek_v41 import candidate_blocks
from ..runtime.models.deepseek_v4 import stable_pool_topk
from .deepseek_v41_ast_batched_attention import (
    LayerAncestrySegments as _Segments, PARENTS, PATHS, DEPTHS, _logical_map, _prior_candidates,
)
PROFILE='source_ast_m1_batch_axis_attention_candidate_v1'
_SPLITK_OWNERSHIP=ContextVar('source_ast_stock_splitk_owner',default=None)


@contextmanager
def owned_stock_splitk(check_live):
    """Compatibility scope name for caller-owned row-exact attention.

    This does not acquire GPU ownership. A successful caller check may return
    None; explicit False or an exception refuses work. The shader boundary
    receives True only after that actual supplied check has succeeded.
    """
    if not callable(check_live):raise ValueError('Source AST exact attention needs a caller-owned live callback')
    if _SPLITK_OWNERSHIP.get()is not None:raise ValueError('Source AST exact attention ownership scope already active')
    def checked():
        if check_live()is False:raise RuntimeError('Source AST exact attention caller ownership lost')
        return True
    checked();token=_SPLITK_OWNERSHIP.set(checked)
    try:yield
    finally:_SPLITK_OWNERSHIP.reset(token)


def implementation_identity():
    from ..runtime.gguf.source_ast_batched_splitk import implementation_identity as selected_identity
    return hashlib.sha256(Path(__file__).read_bytes()+selected_identity().encode()).hexdigest()


def plan(prefix_tokens,*,ratio,window=128,selected_limit=512,head_dim=512):
    """CPU inventory of ordinary per-query lengths; no GPU or admission."""
    if (any(type(v)is not int for v in(prefix_tokens,ratio,window,selected_limit,head_dim))
            or prefix_tokens<0 or ratio not in(0,1,2)or window<1 or selected_limit<1 or head_dim<1):
        raise ValueError('Source AST exactattention count-plan geometry mismatch')
    groups={}
    for node,depth in enumerate(DEPTHS):
        local=min(window,prefix_tokens+depth+1)
        selected=0 if not ratio else min(selected_limit,(prefix_tokens+depth+1)//ratio)
        groups.setdefault((local,selected),[]).append(node)
    return dict(profile=PROFILE,physical_query_rows=16,
        count_classes=[dict(local_rows=local,selected_rows=selected,nodes=nodes)
                       for(local,selected),nodes in sorted(groups.items())],
        local_gather_bound_bytes=16*window*head_dim*4,
        selected_gather_bound_bytes=16*selected_limit*head_dim*4,
        full_prefix_KV_copies=0,exact_M1_claim=False,qualified=False,default_selected=False,
        unresolved='Common index-table output width and installed stock batch dispatch require actual full rawbit gate')


class LayerAncestrySegments(_Segments):
    def receipt(self):
        result=super().receipt()
        result.update(profile=PROFILE,arithmetic='M1 batch-axis candidate; actual rawbit gate pending',
                      local_reduction_lengths='ordinary chronological count, no extra maskedslots',
                      exact_M1_claim=False,local_copy_bound='16xwindowxhead_dim',
                      selected_copy_bound='16xindex_topkxhead_dim')
        result['attention_count_classes']=[dict(local_rows=local,selected_rows=selected,nodes=list(nodes))
            for local,selected,nodes in getattr(self,'_exact_attention_count_classes',())]
        result['selected_kv_preparation'] = dict(
            policy=getattr(self, '_kv_preparation_policy', 'reference'),
            builds=getattr(self, '_selected_prepare_builds', 0),
            hits=getattr(self, '_selected_prepare_hits', 0),
            built_payload_bytes=getattr(self, '_selected_prepare_bytes', 0),
            accounting='Logical prepared payload only; not measured device traffic')
        return result


def _private_matmul(left,right,*,transpose_b=False,role='local'):
    """Keep each B's ordinary MxK @KxN; do not flatten B into matrix M."""
    if role not in('local','selected','index'):
        raise ValueError('Source AST exactattention private matrix role unknown')
    if left.ndim!=3 or right.ndim!=3 or left.shape[-1]!=right.shape[-2]:
        raise ValueError('Source AST exactattention private matrix geometry mismatch')
    batches,rows,k=left.shape
    if right.shape[0]not in(1,batches):
        raise ValueError('Source AST exactattention coefficient batch mismatch')
    if mx.default_device()==mx.gpu:
        if rows==64 and 1<=batches<=16 and 1<=k<=512 and 1<=right.shape[-1]<=512:
            owner=_SPLITK_OWNERSHIP.get()
            if owner is None:
                raise ValueError('Source AST exact attention requires owned_stock_splitk caller scope')
            if role=='index':
                raise ValueError('Source AST exact index-score GPU role is not admitted')
            if batches==1:
                owner()
                query=mx.contiguous(left).reshape(1,rows,1,k)
                if role=='selected':
                    if transpose_b:
                        rows_kv=mx.contiguous(mx.swapaxes(right,-1,-2)).reshape(1,1,right.shape[-1],k)
                        return mx.einsum('bhld,blkd->bhlk',query,rows_kv)[:,:,0]
                    rows_kv=mx.contiguous(right).reshape(1,1,k,right.shape[-1])
                    return mx.einsum('bhlk,blkd->bhld',query,rows_kv)[:,:,0]
                # Ordinary local matmul depends on BOTH singleton strides:
                # reshape preserves M-stride K and coefficient head stride KN.
                if transpose_b:
                    kv=mx.contiguous(mx.swapaxes(right,-1,-2)).reshape(1,1,right.shape[-1],k)
                    coefficient=mx.swapaxes(kv,-1,-2)
                else:coefficient=mx.contiguous(right).reshape(1,1,k,right.shape[-1])
                return mx.matmul(query,coefficient).squeeze(2)
            # Actual-stride v54 proved all32 local+selected matrices and every
            # stage of the complete local chain with this ordinary M64 law.
            from ..runtime.gguf.source_ast_batched_splitk import private_matmul
            return private_matmul(left,mx.broadcast_to(right,(batches,*right.shape[1:])),
                                  transpose_b=transpose_b,check_live=owner)
        raise ValueError('Source AST exact M64 GPU shape is not admitted; no stock-batch fallback')
    storage=mx.concatenate((left.astype(mx.float32).reshape(batches,rows*k),
                            mx.zeros((batches,1),mx.float32)),axis=1)
    private=mx.as_strided(storage,(batches,rows,k),(rows*k+1,k,1))
    coefficient=mx.broadcast_to(right.astype(mx.float32),(batches,*right.shape[1:]))
    return mx.matmul(private,coefficient)


def count_classes(segments,owner=None,shared=None):
    """CPU metadata only; every node occurs exactly once, no shape padding."""
    groups={}
    for node in range(16):
        selected=0 if owner is None or not len(owner.logical_rows[node]) else shared[node]['indices'].shape[-1]
        key=(len(segments.local_rows[node]),selected)
        groups.setdefault(key,[]).append(node)
    return tuple((local,selected,tuple(nodes))for(local,selected),nodes in sorted(groups.items()))


def attention_core(q,segments,sinks,*,owner=None,shared=None):
    """Batched same-length stock score/PV classes; rows stay private."""
    if q.ndim!=4 or q.shape[0]!=1 or q.shape[2]!=16:
        raise ValueError('Source AST exactattention requires one16-query tile')
    outputs=[];order=[]
    preparation = getattr(segments, '_kv_preparation_policy', 'reference')
    if preparation not in ('reference', 'descendant_reuse_v1'):
        raise ValueError('Source AST selected KV preparation policy unknown')
    reuse = preparation == 'descendant_reuse_v1'
    mapping=None if owner is None or reuse else _logical_map(owner)[0]
    groups=count_classes(segments,owner,shared)
    for local_count,selected_count,nodes in groups:
        ids=mx.array(nodes,mx.int32)
        query=mx.take(q,ids,axis=2)[0].transpose(1,0,2).astype(mx.float32)*q.shape[-1]**-.5
        rows=mx.array([segments.local_rows[node]for node in nodes],mx.int32)
        local=segments.local_table[0,0,rows].astype(mx.float32)
        ls=_private_matmul(query,mx.swapaxes(local,-1,-2),transpose_b=True)[:,:,None]
        norm=mx.logsumexp(ls,axis=-1,keepdims=True)
        selected=None
        if selected_count:
            if reuse:
                from .deepseek_v41_ast_kv_preparation import prepare_selected_class
                selected, valid, prepared = prepare_selected_class(
                    mx, owner, shared, nodes, check_live=_SPLITK_OWNERSHIP.get(), profile=PROFILE)
                field = '_selected_prepare_hits' if prepared['reused'] else '_selected_prepare_builds'
                setattr(segments, field, getattr(segments, field, 0) + 1)
                segments._selected_prepare_bytes = getattr(segments, '_selected_prepare_bytes', 0) + prepared['built_payload_bytes']
            else:
                logical=mx.concatenate([shared[node]['indices']for node in nodes],axis=0)[:,0]
                valid=mx.concatenate([shared[node]['selected_mask']for node in nodes],axis=0)[:,0]
                physical=mx.take_along_axis(mx.take(mapping,ids,axis=0),logical,axis=1)
                selected=owner.pool_table[0,physical].astype(mx.float32)
            ps=_private_matmul(query,mx.swapaxes(selected,-1,-2),transpose_b=True,role='selected')[:,:,None]
            ps=mx.where(valid[:,None,None],ps,-mx.inf)
            norm=mx.logaddexp(norm,mx.logsumexp(ps,axis=-1,keepdims=True))
        norm=mx.logaddexp(norm,sinks[None,:,None,None].astype(mx.float32))
        output=_private_matmul(mx.exp(ls-norm).squeeze(2),local,transpose_b=False)
        if selected is not None:
            output+=_private_matmul(mx.exp(ps-norm).squeeze(2),selected,transpose_b=False,role='selected')
        outputs.append(output);order.extend(nodes)
    inverse=mx.array([order.index(node)for node in range(16)],mx.int32)
    output=mx.take(mx.concatenate(outputs,axis=0),inverse,axis=0).transpose(1,0,2)[None]
    segments._exact_attention_count_classes=groups
    return output.astype(q.dtype)


def _publish_source(attn, x, positions, base, kv):
    from .deepseek_v41_ast_batched_dense_exact import linear_tile, paired_linear_tile
    m, c, ratio = attn.model, attn.c, attn.ratio
    if ratio == 2:
        projected, gate = paired_linear_tile(m, x, attn.prefix+'attn_compressor_kv.weight',
                                            attn.prefix+'attn_compressor_gate.weight', rounded=False)
    else:
        projected = linear_tile(m, x, attn.prefix+'attn_compressor_kv.weight')
        gate = None
    raw, pair_gates, buffers, gates, completed, pool_positions = [], [], [], [], [], []
    for node, parent in enumerate(PARENTS):
        value = projected[:,node:node+1]
        if ratio == 2:
            prior = base.buf_kv if parent < 0 else buffers[parent]
            prior_gate = base.buf_gate if parent < 0 else gates[parent]
            weights = gate[:,node:node+1]
            if prior is not None:
                value = mx.concatenate((prior,value),1);weights = mx.concatenate((prior_gate,weights),1)
            complete = value.shape[1]//2
            buffers.append(mx.asarray(value[:,complete*2:],copy=True))
            gates.append(mx.asarray(weights[:,complete*2:],copy=True))
            if not complete:continue
            # Stage pair descriptors in ancestry order; the softmax, weighted
            # reduction and BF16 boundary execute once over all complete rows.
            value=value[:,:2];pair_gates.append(weights[:,:2])
        else:
            buffers.append(mx.zeros((1,0,c.head_dim),mx.float32))
            gates.append(mx.zeros((1,0,c.head_dim),mx.float32))
        raw.append(value);completed.append(node);pool_positions.append((positions[node]//ratio)*ratio)
    if raw:
        values=(m.round(mx.sum(mx.stack(raw,axis=1)*mx.softmax(mx.stack(pair_gates,axis=1),axis=2),axis=2))
                if ratio==2 else mx.concatenate(raw,1))
        latent = m.norm(values,attn.w.array(attn.prefix+'attn_compressor_norm.weight'))
        key = m.norm(linear_tile(m,latent,attn.prefix+'indexer.attn_k.weight'),
                     attn.w.array(attn.prefix+'indexer.k_norm.weight'))
        key = attn.rotate_quantized(key[:,None], 'index_fp4',positions=tuple(pool_positions)).squeeze(1)
        pooled = attn.rotate_quantized(latent[:,None], 'compressed_fp4',positions=tuple(pool_positions)).squeeze(1)
    else:
        pooled=mx.zeros((1,0,c.head_dim),mx.float32);key=mx.zeros((1,0,c.index_head_dim),mx.float32)
    return LayerAncestrySegments(base,kv,pooled,key,tuple(completed),tuple(buffers),tuple(gates))

def _select(attn, x, qr, owner, positions, shared):
    """Chronological index count classes; no speculative physical padding."""
    from .deepseek_v41_ast_batched_index_exact import select_tile
    return select_tile(attn, x, qr, owner, positions, shared,
                       check_live=_SPLITK_OWNERSHIP.get())

def attention_half_tile(block,h,pre,base_cache,layer_segments,shared):
    """Batch the whole HC/attention half, returning one lazy layer descriptor."""
    from .deepseek_v41_ast_batched_dense_exact import linear_tile,paired_linear_tile,grouped_tile,mixes_tile
    m,c,a=block.model,block.c,block.attn
    if h.shape!=(1,16,c.hc_mult,c.hidden_size)or pre.shape!=(1,16,c.hc_mult):
        raise ValueError('Source AST batched attention requires exactly16 HC rows')
    if len(base_cache)!=c.num_hidden_layers or len(layer_segments)!=c.num_hidden_layers or len(shared)!=16:
        raise ValueError('Source AST layer/source/shared ownership mismatch')
    if len({id(row)for row in shared})!=16 or layer_segments[block.layer]is not None:
        raise ValueError('Source AST layer already published or sibling dictionaries alias')
    base=base_cache[block.layer]
    if base.pending or base.ratio!=a.ratio or base.source!=a.source:
        raise ValueError('Source AST prefix is incomplete or has different source topology')
    positions=tuple(base.offset+d for d in DEPTHS)
    attn_pre,post,comb=mixes_tile(block,h,'attn')
    x=m.norm(m.collapse(h,pre),block.attn_norm)
    qa,kva=paired_linear_tile(m,x,a.prefix+'attn_q_a.weight',a.prefix+'attn_kv.weight')
    qr=m.norm(qa,a.q_norm)
    q=linear_tile(m,qr,a.prefix+'attn_q_b.weight').reshape(1,16,c.num_attention_heads,c.head_dim).transpose(0,2,1,3)
    q=a.rotate(q,positions=positions)
    kv=a.rotate_quantized(m.norm(kva,a.kv_norm)[:,None],'fp8',positions=positions)
    descriptor=_publish_source(a,x,positions,base,kv)if a.source else LayerAncestrySegments(base,kv)
    owner=None
    if a.ratio:
        source_layer=c.kv_source_for(block.layer)
        owner=descriptor if a.source else layer_segments[source_layer]
        if owner is None or not owner.base.source or owner.base.offset!=base.offset:
            raise ValueError('Source AST compressed owner segment failed to publish')
        owner.assert_current()
        if a.index_source:_select(a,x,qr,owner,positions,shared)
        if max(map(len,owner.logical_rows)):
            if any('indices'not in row or 'selected_mask'not in row for row in shared):
                raise ValueError('Source AST index source failed to publish')
    descriptor._kv_preparation_policy = getattr(m, 'source_ast_kv_preparation_policy', 'reference')
    out=m.round(attention_core(q,descriptor,a.sinks,owner=owner,shared=shared))
    fused=getattr(m,'attention_epilogue_policy','reference')=='fused'
    out=(a.rotate_quantized(out,'fp8',positions=positions,inverse=True)if fused else
         a.rotate(out,positions=positions,inverse=True))
    out=out.reshape(1,c.o_groups,c.num_attention_heads//c.o_groups,16,c.head_dim).transpose(0,3,1,2,4).reshape(1,16,c.o_groups,-1)
    if not fused:out=m.qat(out,'fp8')
    out=m.round(grouped_tile(m,out,a.prefix+'attn_output_a.weight',groups=c.o_groups))
    out=linear_tile(m,out.reshape(1,16,-1),a.prefix+'attn_output_b.weight')
    h=m.expand(out,h,post,comb)
    ffn_pre,post,comb=mixes_tile(block,h,'ffn')
    x=m.norm(m.collapse(h,attn_pre),block.ffn_norm)
    counters=m.scheduled_kv_calls
    counters['source_ast_row_exact_attention_candidate_tiles']=counters.get('source_ast_row_exact_attention_candidate_tiles',0)+1
    return h,x,ffn_pre,post,comb,descriptor
