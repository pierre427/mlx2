# SPDX-License-Identifier: MIT
"""Default-off16-query Source AST attention over shared prefix segments.

Native M16 dense/reduction arithmetic has its own profile: this module does
not inherit the B1 target's numerical admission. See provenance entry.
"""
from dataclasses import dataclass, field
import hashlib
from pathlib import Path

import mlx.core as mx

from ..runtime.models.deepseek_v41 import candidate_blocks
from ..runtime.models.deepseek_v4 import stable_pool_topk

PROFILE = "source_ast_descendant_batched_attention_v1"
PARENTS = (-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14)
PATHS = []
for _node, _parent in enumerate(PARENTS):
    PATHS.append((_node,) if _parent < 0 else PATHS[_parent] + (_node,))
PATHS = tuple(PATHS)
DEPTHS = tuple(len(path)-1 for path in PATHS)
DESCENDANTS = tuple(sum(1 << row for row, path in enumerate(PATHS) if node in path)
                    for node in range(16))


def implementation_identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _copy(value):
    return None if value is None else mx.asarray(value, copy=True)


@dataclass
class LayerAncestrySegments:
    """Pinned prefix owners and private small deltas, never sibling frontiers.

    The caller owns the epoch/generation check and retains this descriptor until
    its last GPU consumer finishes. ``materialize`` constructs a private cache;
    publication still requires the ordinary cache-operation contract/fence.
    """
    base: object
    kv: object
    pooled_delta: object = None
    index_delta: object = None
    completed_nodes: tuple = ()
    buf_kv: tuple = ()
    buf_gate: tuple = ()
    node_hash_tails: object = None
    _base_stamp: tuple = field(init=False, repr=False)

    def __post_init__(self):
        self._base_stamp = self._stamp()
        self.prefix_count = self.base._pool_count if self.base.source else 0
        self.prefix_local_count = 0 if self.base.keys is None else self.base.keys.shape[2]
        self.local_table = (self.kv if self.base.keys is None else
                            mx.concatenate((self.base.keys, self.kv), 2))
        self.local_positions = tuple(range(self.base.offset-self.prefix_local_count,
                                           self.base.offset)) + tuple(self.base.offset+d for d in DEPTHS)
        self.local_rows = tuple(tuple(i for i in range(self.prefix_local_count)
                                     if self.local_positions[i] > self.offset(node)-1-self.base.window_size)
                                + tuple(self.prefix_local_count+i for i in path
                                        if self.base.offset+DEPTHS[i] > self.offset(node)-1-self.base.window_size)
                                for node, path in enumerate(PATHS))
        self.local_visibility = mx.array([[column in rows for column in range(self.local_table.shape[2])]
                                         for rows in self.local_rows], mx.bool_)
        self.pool_table = self.index_table = None
        self.logical_rows = tuple(() for _ in PATHS)
        if self.base.source:
            prefix_pool = self.base.pooled
            prefix_index = self.base.index_keys
            if prefix_pool is None:
                prefix_pool = mx.zeros((1, 0, self.base.head_dim), mx.float32)
                prefix_index = mx.zeros((1, 0, self.base.index_head_dim), mx.float32)
            self.pool_table = mx.concatenate((prefix_pool, self.pooled_delta), 1)
            self.index_table = mx.concatenate((prefix_index, self.index_delta), 1)
            delta_index = {node: self.prefix_count+i for i, node in enumerate(self.completed_nodes)}
            self.logical_rows = tuple(tuple(range(self.prefix_count)) +
                                      tuple(delta_index[i] for i in path if i in delta_index)
                                      for path in PATHS)

    def _stamp(self):
        return (self.base.meta_state, self.base._capacity, self.base._pool_count,
                tuple(id(getattr(self.base, name)) for name in
                      ('keys', 'buf_kv', 'buf_gate', '_pooled', '_index_keys', 'hash_tail')))

    def assert_current(self):
        if self._stamp() != self._base_stamp:
            raise ValueError("Source AST prefix owner changed while segments were live")

    def offset(self, node):
        if type(node) is not int or not 0 <= node < 16:
            raise ValueError("Source AST segment node outside16-row tile")
        return self.base.offset+DEPTHS[node]+1

    def evaluation_roots(self):
        return (self.base.keys, self.base.buf_kv, self.base.buf_gate,
                self.base._pooled, self.base._index_keys, self.base.hash_tail,
                self.kv, self.pooled_delta, self.index_delta, self.buf_kv,
                self.buf_gate, self.node_hash_tails, self.local_table,
                self.pool_table, self.index_table, self.local_visibility)

    def materialize(self, node):
        """Materialize one accepted frontier; no evaluation/publication here."""
        from ..runtime.models.deepseek_v41_cache import DeepSeekV41LayerCache
        self.assert_current();self.offset(node)
        if self.base.history and self.node_hash_tails is None:
            raise ValueError("Source AST history owner needs all node hash tails")
        b = self.base
        state = DeepSeekV41LayerCache(b.ratio, xp=mx, identity=b.identity,
            head_dim=b.head_dim, index_head_dim=b.index_head_dim,
            window_size=b.window_size, source=b.source, history=b.history,
            compressed_vocab=b.compressed_vocab)
        # Preserve full capacity and unused physical prefix slots. A selected
        # branch owns new array wrappers before any append mutates a wrapper.
        for name in ('buf_kv','buf_gate','_pooled','_index_keys','hash_tail'):
            setattr(state, name, _copy(getattr(b, name)))
        state._capacity, state._pool_count = b._capacity, b._pool_count
        state.offset = self.offset(node)
        state.keys = mx.take(self.local_table, mx.array(self.local_rows[node], mx.int32), axis=2)
        if b.source:
            state.buf_kv = _copy(self.buf_kv[node]);state.buf_gate = _copy(self.buf_gate[node])
            additions = [i for i, n in enumerate(self.completed_nodes) if n in PATHS[node]]
            indices = mx.array(additions, mx.int32)
            state.append_pools(mx.take(self.pooled_delta, indices, axis=1),
                               mx.take(self.index_delta, indices, axis=1))
        if b.history:
            state.hash_tail = _copy(self.node_hash_tails[node])
        state.validate_state()
        return state

    def receipt(self):
        return dict(profile=PROFILE, physical_query_rows=16,
                    shared_prefix=True, full_prefix_materializations_during_attention=0,
                    descendant_masks=list(DESCENDANTS), local_visibility_shape=list(self.local_visibility.shape),
                    source_ratio=self.base.ratio, compressed_delta_nodes=list(self.completed_nodes),
                    qualified=False, arithmetic='nativeM16; exact B1 parity pending artifact gate')


def _logical_map(owner):
    width = max(map(len, owner.logical_rows))
    mapping = mx.array([list(row)+[0]*(width-len(row)) for row in owner.logical_rows], mx.int32)
    valid = mx.arange(width)[None] < mx.array([len(row) for row in owner.logical_rows], mx.int32)[:,None]
    return mapping, valid


def _publish_source(attn, x, positions, base, kv):
    from .deepseek_v41_ast_batched_dense import linear_tile, paired_linear_tile
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
    """Shared index-key projection table; scalar chronological row gather."""
    from .deepseek_v41_ast_batched_dense import linear_tile
    m,c=attn.model,attn.c
    mapping,reachable=_logical_map(owner);width=mapping.shape[1]
    for row in shared:
        row['_index_generation']=row.get('_index_generation',0)+1
        for key in ('indices','selected_mask','_decode_selected_kv','_all_retained_chronological'):
            row.pop(key,None)
    if not width:
        indices=mx.zeros((16,0),mx.int32);valid=mx.zeros((16,0),mx.bool_)
        candidates=valid
    else:
        finite=m.index_policy=='finite_all_retained'
        all_blocks=(width+c.candidate_block_size-1)//c.candidate_block_size<=c.candidate_topk_blocks
        lenses=mx.array([(p+1)//attn.ratio for p in positions],mx.int32)[:,None,None]
        bypass=finite and width<=c.index_topk and(attn.layer!=c.candidate_source_layer_id or all_blocks)
        if bypass:
            indices=mx.broadcast_to(mx.arange(width,dtype=mx.uint32)[None],(16,width))
            valid=reachable
            candidates=(mx.arange(width)[None,None]//c.candidate_block_size*c.candidate_block_size<lenses)[:,0]
        else:
            query=linear_tile(m,qr,attn.prefix+'indexer.attn_q_b.weight').reshape(1,16,c.index_n_heads,c.index_head_dim).transpose(0,2,1,3)
            query=attn.rotate_quantized(query,'index_fp4',positions=positions)
            weight=linear_tile(m,x,attn.prefix+'indexer.proj.weight')*(c.index_head_dim**-.5*c.index_n_heads**-.5)
            # One common physical key table. Only scalar scores are expanded
            # into query-private chronological order; prefix keys stay shared.
            score=mx.einsum('bhld,btd->blht',query,owner.index_table)
            score=mx.sum(mx.maximum(score,0)*weight[...,None],2)[0]
            score=mx.take_along_axis(score,mapping,axis=1)
            score=mx.where(reachable,score,-mx.inf)
            candidates=candidate_blocks(score[:,None],lenses,c.candidate_topk_blocks,
                                        c.candidate_block_size,finite_all_retained=finite)[:,0]
            if attn.layer>c.candidate_source_layer_id>=0:
                prior=_prior_candidates(shared,owner,width)
                score=mx.where(prior,score,-mx.inf)
            indices=stable_pool_topk(score[:,None],min(c.index_topk,width))[:,0]
            valid=mx.take_along_axis(score,indices,axis=1)>-mx.inf
        if bypass and attn.layer>c.candidate_source_layer_id>=0:
            valid=valid&_prior_candidates(shared,owner,width)
    for node,row in enumerate(shared):
        count=min(c.index_topk,len(owner.logical_rows[node]))
        row['indices']=indices[node:node+1,None,:count]
        row['selected_mask']=valid[node:node+1,None,:count]
        if attn.layer==c.candidate_source_layer_id:
            row['candidates']=candidates[node:node+1,None,:len(owner.logical_rows[node])]


def _prior_candidates(shared,owner,width):
    rows=[]
    for node,row in enumerate(shared):
        prior=row.get('candidates');length=len(owner.logical_rows[node])
        if prior is None or prior.shape!=(1,1,length):
            raise ValueError('Source AST candidate source failed to publish ancestry-local mask')
        rows.append(mx.concatenate((prior.reshape(length),mx.zeros((width-length,),mx.bool_))))
    return mx.stack(rows)


def _gather_selected(owner,shared):
    width=max(row['indices'].shape[-1]for row in shared)
    if not width:return None,None
    marker=(id(owner),tuple(id(row['indices'])for row in shared),tuple(id(row['selected_mask'])for row in shared))
    cached=shared[0].get('_ast_segment_selected')
    if cached is not None and cached[0]==marker:return cached[1:]
    mapping,_=_logical_map(owner);indices=[];valid=[]
    for row in shared:
        count=row['indices'].shape[-1]
        indices.append(mx.concatenate((row['indices'].reshape(count),mx.zeros((width-count,),mx.int32))))
        valid.append(mx.concatenate((row['selected_mask'].reshape(count),mx.zeros((width-count,),mx.bool_))))
    indices=mx.stack(indices);valid=mx.stack(valid)
    physical=mx.take_along_axis(mapping,indices,axis=1)
    values=owner.pool_table[0,physical][None]
    # Unreachable tails never poison a valid branch through0*NaN.
    values=mx.where(valid[None,:,:,None],values,0)
    shared[0]['_ast_segment_selected']=(marker,values,valid[None])
    return values,valid[None]


def attention_core(q,segments,sinks,selected=None,valid=None):
    """One grouped16-query graph; local table is prefix+private delta once."""
    query=q.astype(mx.float32)*q.shape[-1]**-.5
    local=segments.local_table.astype(mx.float32)
    ls=query@mx.swapaxes(local,-1,-2)
    ls=mx.where(segments.local_visibility[None,None],ls,-mx.inf)
    norm=mx.logsumexp(ls,-1,keepdims=True)
    if selected is not None:
        ps=mx.einsum('bhld,blkd->bhlk',query,selected)
        ps=mx.where(valid[:,None],ps,-mx.inf)
        norm=mx.logaddexp(norm,mx.logsumexp(ps,-1,keepdims=True))
    norm=mx.logaddexp(norm,sinks[None,:,None,None].astype(mx.float32))
    out=mx.exp(ls-norm)@local
    if selected is not None:out+=mx.einsum('bhlk,blkd->bhld',mx.exp(ps-norm),selected)
    return out.astype(q.dtype)


def attention_half_tile(block,h,pre,base_cache,layer_segments,shared):
    """Batch the whole HC/attention half, returning one lazy layer descriptor."""
    from .deepseek_v41_ast_batched_dense import linear_tile,paired_linear_tile,grouped_tile
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
    attn_pre,post,comb=block.mixes(h,'attn')
    x=m.norm(m.collapse(h,pre),block.attn_norm)
    qa,kva=paired_linear_tile(m,x,a.prefix+'attn_q_a.weight',a.prefix+'attn_kv.weight')
    qr=m.norm(qa,a.q_norm)
    q=linear_tile(m,qr,a.prefix+'attn_q_b.weight').reshape(1,16,c.num_attention_heads,c.head_dim).transpose(0,2,1,3)
    q=a.rotate(q,positions=positions)
    kv=a.rotate_quantized(m.norm(kva,a.kv_norm)[:,None],'fp8',positions=positions)
    descriptor=_publish_source(a,x,positions,base,kv)if a.source else LayerAncestrySegments(base,kv)
    selected=valid=None
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
            selected,valid=_gather_selected(owner,shared)
    out=m.round(attention_core(q,descriptor,a.sinks,selected,valid))
    fused=getattr(m,'attention_epilogue_policy','reference')=='fused'
    out=(a.rotate_quantized(out,'fp8',positions=positions,inverse=True)if fused else
         a.rotate(out,positions=positions,inverse=True))
    out=out.reshape(1,c.o_groups,c.num_attention_heads//c.o_groups,16,c.head_dim).transpose(0,3,1,2,4).reshape(1,16,c.o_groups,-1)
    if not fused:out=m.qat(out,'fp8')
    out=m.round(grouped_tile(m,out,a.prefix+'attn_output_a.weight',groups=c.o_groups))
    out=linear_tile(m,out.reshape(1,16,-1),a.prefix+'attn_output_b.weight')
    h=m.expand(out,h,post,comb)
    ffn_pre,post,comb=block.mixes(h,'ffn')
    x=m.norm(m.collapse(h,attn_pre),block.ffn_norm)
    counters=m.scheduled_kv_calls
    counters['source_ast_grouped_attention_tiles']=counters.get('source_ast_grouped_attention_tiles',0)+1
    return h,x,ffn_pre,post,comb,descriptor
