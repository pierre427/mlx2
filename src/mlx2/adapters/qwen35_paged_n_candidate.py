# SPDX-License-Identifier: MIT
"""Explicit N1..20 hybrid Q1 over private native owners; original B2 unchanged."""
from __future__ import annotations
import os
from .qwen35_paged_candidate import Qwen35PagedCandidate,HybridPackedLane,_pack_attention_q1


def configure_native_ragged_prompt_lookup(candidate,environ=None):
    """Install the explicit, default-off deterministic N20 proposal source."""
    environ=os.environ if environ is None else environ
    selected=environ.get('MLX2_PAGED_N20_RAGGED_PROMPT_LOOKUP','0')
    if selected not in ('0','1'):
        raise ValueError('N20 ragged prompt-lookup selector must be 0 or 1')
    if selected=='0':
        candidate._native_ragged_prompt_lookup=None
        return None
    raw=environ.get('MLX2_PAGED_N20_RAGGED_DEPTH','4')
    try:depth=int(raw)
    except (TypeError,ValueError) as error:
        raise ValueError('N20 ragged depth must be an integer') from error
    if str(depth)!=str(raw) or not 1<=depth<=15:
        raise ValueError('N20 ragged depth must be in 1..15')
    ngrams=[]
    for name,default in (('MIN','3'),('MAX','6')):
        value=environ.get(f'MLX2_PAGED_N20_RAGGED_NGRAM_{name}',default)
        try:parsed=int(value)
        except (TypeError,ValueError) as error:
            raise ValueError('N20 ragged ngram bounds must be integers') from error
        if str(parsed)!=str(value) or not 1<=parsed<=16:
            raise ValueError('N20 ragged ngram bounds must be in 1..16')
        ngrams.append(parsed)
    if ngrams[0]>ngrams[1]:
        raise ValueError('N20 ragged ngram range is invalid')
    from ..runtime.proposal_providers import ContinuationPoolPolicy
    policy=ContinuationPoolPolicy.from_value({
        'sources':['prompt_lookup'],'limit':1,
        'ngram_min':ngrams[0],'ngram_max':ngrams[1]})
    candidate._native_ragged_prompt_lookup=(policy,depth)
    return candidate._native_ragged_prompt_lookup


def validate_n_lanes(lanes,branches,candidate,proposal_steps=None):
    if type(lanes) is not tuple or type(branches) is not tuple or not 1<=len(lanes)<=20 or len(branches)!=len(lanes) or len({id(b) for b in branches})!=len(branches):
        raise ValueError('one..twenty distinct private N branches required')
    if proposal_steps is None:
        steps=(0,)*len(lanes);ordinary=True
    elif (type(proposal_steps) is not tuple or len(proposal_steps)!=len(lanes) or
            any(type(step) is not int or step<0 for step in proposal_steps)):
        raise ValueError('one nonnegative proposal step is required per N lane')
    else:
        steps=proposal_steps;ordinary=False
    offsets=[]
    for lane,branch,step in zip(lanes,branches,steps):
        if type(lane) is not HybridPackedLane or lane.layers is not branch.layers or lane.recurrent_caches is not branch.recurrent_caches or branch._closed or (ordinary and branch._request.proposed_rows!=1) or not step<branch._request.proposed_rows or branch._request.planes!=('kv','gdn') or len(lane.token_ids)!=1 or type(lane.token_ids[0]) is not int or not 0<=lane.token_ids[0]<candidate.args.vocab_size or len(lane.recurrent_caches)!=len(candidate.layer_map.recurrent):
            raise ValueError('N lane does not match its private branch')
        offset=branch._origin.layers[0].offset+step
        if type(offset) is not int or not 1025<=offset+1<=8192:raise ValueError('N20 Q1 requires long bounded context')
        offsets.append(offset)
    if len({id(cache) for lane in lanes for cache in lane.recurrent_caches})!=len(lanes)*len(candidate.layer_map.recurrent):
        raise ValueError('private recurrent containers alias across N lanes')
    return tuple(offsets)


def validate_finite_roots(mx,logits,recurrent,projection_roots):
    roots=(logits,*recurrent,*(v for root in projection_roots for v in root[:3]))
    checks=mx.stack(tuple(mx.all(mx.isfinite(value)) for value in roots));mx.eval(checks)
    if not all(checks.tolist()):raise ValueError('N20 logits/recurrent/native KV nonfinite')


class Qwen35PagedNCandidate(Qwen35PagedCandidate):
    serving_route='native_hybrid_packed_n20_research'
    def __init__(self,model,backend,*,runtime_factory=None):
        kwargs={} if runtime_factory is None else {'_runtime_factory':runtime_factory}
        super().__init__(model,backend,q1_stripes=16,stock_long=True,**kwargs)
        self._prefill_packed_n20=True;self._prefill_layer_lifetime=True;self._serving_n20=True
        self.owns_physical_dispatch_proof=True
    def forward_scratch_bytes(self,offsets):
        if type(offsets) is not tuple or not 1<=len(offsets)<=20 or any(type(o) is not int or not 1025<=o+1<=8192 for o in offsets):raise ValueError('bounded long N offsets required')
        return self.native_layer_count*len(offsets)*24*128*258*4
    def forward_staged(self,lanes,branches,*,permit_candidate=False,reserve_scratch=None,
                       _proposal_steps=None):
        if permit_candidate is not True or os.environ.get('MLX2_PAGED_PACKED_N20')!='1':raise ValueError('explicit source-bound N20 selector required')
        offsets=validate_n_lanes(lanes,branches,self,_proposal_steps);runtime=self._runtime_factory();mx=runtime.mx
        from ..runtime.hybrid_packed_prefill_n import require_n_capabilities
        arena=self.backend.writer.backend
        require_n_capabilities(arena._native,type(self.backend),tuple(o+1 for o in offsets),q1=True)
        prepare=getattr(runtime,'prepare_read_n20',None)
        if prepare is None:
            from ..runtime.paged_attention_pack import prepare_staged_token_read_n20
            prepare=prepare_staged_token_read_n20
        if self.native_dtype_preflight()!='bfloat16' or self.args.num_attention_heads!=24 or self.args.num_key_value_heads!=4:
            raise ValueError('N20 requires same BF16 dense H24/KV4 model')
        writer=self.backend.writer
        if writer.poisoned or writer.pending_epochs or writer.ledger.pending_count or self.backend._orphaned_reads:raise RuntimeError('N20 writer/read ledger not idle')
        for lane,offset in zip(lanes,offsets):self._check_owners(lane.layers,offset,mx)
        for ordinal,index in enumerate(self.layer_map.recurrent):
            module=self.trunk.layers[index].linear_attn;expected=(mx.bfloat16,getattr(module,'_gdn_state_dtype',None) or mx.float32)
            reference=lanes[0].recurrent_caches[ordinal]
            for lane in lanes:
                cache=lane.recurrent_caches[ordinal]
                if len(cache.cache)!=2 or cache.speculating or cache.lengths is not None or cache.left_padding is not None:raise ValueError('plain initialized private GDN caches required')
                for slot,dtype in enumerate(expected):
                    value=cache.cache[slot]
                    if value is None or value.shape[0]!=1 or value.shape[1:]!=reference.cache[slot].shape[1:] or value.dtype!=dtype:raise ValueError('every N recurrent slot must match model shape/dtype')
        scratch=self.forward_scratch_bytes(offsets)
        if not callable(reserve_scratch):raise ValueError('charged N20 decode scratch required')
        reservation=reserve_scratch(scratch)
        uses=[];roots=[];views=[];hidden=logits=None;failed=True
        names=('grouped_n20_write_count','grouped_n20_row_count','q1_stock_long_n20_partial_dispatch_count','q1_stock_long_n20_reduce_dispatch_count','q1_scalar_dispatch_count','q1_stock_long_n20_singleton_partial_dispatch_count','q1_stock_long_n20_singleton_reduce_dispatch_count')
        layers=tuple(owner for lane in lanes for owner in lane.layers)
        try:
            before={name:int(getattr(arena._native,name)(arena._arena)) for name in names}
            if getattr(reservation,'bytes',0)<scratch or not callable(getattr(reservation,'release',None)) or not callable(getattr(reservation,'retain_failure_roots',None)):raise ValueError('N scratch reservation insufficient')
            with mx.stream(arena.stream):
                hidden=self.trunk.embed_tokens(mx.array([list(lane.token_ids) for lane in lanes]))
                fa=gdn=0
                for layer in self.trunk.layers:
                    residual=hidden;mixed=layer.input_layernorm(hidden)
                    if layer.is_linear:
                        view=runtime.batch_cache(tuple(lane.recurrent_caches[gdn] for lane in lanes));view.speculating=False;view.prepare(lengths=[1]*len(lanes));views.append(view)
                        attended=layer.linear_attn(mixed,None,view);view.finalize();gdn+=1
                    else:
                        attention=layer.self_attn;q,k,v,gate=_pack_attention_q1(attention,mixed,offsets,runtime)
                        if any(value.dtype!=mx.bfloat16 for value in (q,k,v)):raise ValueError('N20 native boundary dtype changed')
                        owners=tuple(lane.layers[fa] for lane in lanes)
                        tickets=self.backend.append_staged_grouped_q1_n20(owners,k,v,permit_candidate=True)
                        roots.append((q,k,v,tickets,owners))
                        use=prepare(owners,(1,)*len(lanes),query_heads=24,permit_candidate=True,profile_host=self.backend.profiling_enabled);uses.append(use)
                        native=self.backend.read_staged(use,q,tickets,scale=attention.scale)
                        attended=attention.o_proj(native.reshape(len(lanes),1,-1)*runtime.gate_sigmoid(gate));fa+=1
                    hidden=residual+attended;hidden=hidden+layer.mlp(layer.post_attention_layernorm(hidden))
                logits=self.language_model.logits(self.trunk.norm(hidden))[:,0,:]
                recurrent=tuple(value for lane in lanes for cache in lane.recurrent_caches for value in cache.cache)
                mx.eval(logits,*recurrent)
                validate_finite_roots(mx,logits,recurrent,roots)
            proofs=self.backend.drain_staged(layers,tuple(uses));depth=self.native_layer_count
            after={name:int(getattr(arena._native,name)(arena._arena)) for name in names};delta={name:after[name]-before[name] for name in names}
            if len(proofs)!=depth or delta!={names[0]:depth,names[1]:depth*len(lanes),names[2]:depth,names[3]:depth,names[4]:0,names[5]:depth if len(lanes)==1 else 0,names[6]:depth if len(lanes)==1 else 0}:raise RuntimeError('N20 exact physical Q1/read terminal proof differs')
            for index,proof in enumerate(proofs):
                for lane_index,branch in enumerate(branches):branch.prove_staged_layer_read(index,lane_index,proof)
            if _proposal_steps is None:
                for lane,branch,offset in zip(lanes,branches,offsets):branch.stage_recurrent_boundary(lane.recurrent_caches,offset=offset+1)
            reservation.release();failed=False
        finally:
            if failed:
                for use in uses:
                    if use.state=='prepared':
                        try:use.abort_before_submit()
                        except BaseException:pass
                    elif use.state=='submitted':self.backend._orphaned_reads[use.lease.epoch]=use
                self.backend._failed=True;writer.poisoned=True
                retained=(lanes,branches,tuple(uses),tuple(roots),tuple(views),hidden,logits,reservation)
                self._failure_roots.append(retained)
                try:reservation.retain_failure_roots(retained)
                except BaseException:pass
        return logits,{'route':self.serving_route,'packed_lanes':len(lanes),'native_span_counts':[len(lanes)]*depth,
            'native_n20_selected':True,'physical_counters':delta,'full_attention_layers':depth,'native_recurrent_boundaries':len(lanes),
            'native_split_scratch_reserved_bytes':scratch,'proposal_steps':list(_proposal_steps or (0,)*len(lanes)),
            'qualified':False,'price_usable':False}

    def forward_staged_ragged(self,layout,lanes,branches,*,decide_next,
                              permit_candidate=False,reserve_scratch=None,
                              collect_logits=False):
        """Execute one online ragged verify with a shrinking active cohort.

        ``decide_next`` receives ``(lane_indices, step, logits)`` and returns
        one boolean per active lane. True consumes that lane's next proposed
        token; false seals its current prefix. Only the selected recurrent
        boundary is cloned, so transition peak remains one successor state per
        lane rather than K full recurrent snapshots.
        """
        from ..runtime.ragged_verify_layout import (
            RaggedVerifyCapabilities, RaggedVerifyLayout)
        if (type(layout) is not RaggedVerifyLayout or type(lanes) is not tuple or
                type(branches) is not tuple or len(lanes)!=layout.lane_count or
                len(branches)!=layout.lane_count or not callable(decide_next) or
                type(collect_logits) is not bool or
                tuple(branch._request.lane_id for branch in branches)!=layout.lane_uids or
                any(len(lane.token_ids)!=length or branch._request.proposed_rows!=length
                    for lane,branch,length in zip(lanes,branches,layout.query_lengths))):
            raise ValueError('ragged staged lanes must exactly match their immutable layout')
        lane_logits=([[] for _ in lanes] if collect_logits else None)
        executed=[0 for _ in lanes];active=list(range(layout.lane_count))
        round_receipts=[]
        for step in range(layout.max_query_len):
            current=tuple(index for index in active if step<layout.query_lengths[index])
            if not current:break
            step_lanes=tuple(HybridPackedLane(
                (lanes[index].token_ids[step],),branches[index].layers,
                branches[index].recurrent_caches) for index in current)
            step_branches=tuple(branches[index] for index in current)
            logits,receipt=self.forward_staged(
                step_lanes,step_branches,permit_candidate=permit_candidate,
                reserve_scratch=reserve_scratch,
                _proposal_steps=(step,)*len(current))
            round_receipts.append(receipt)
            for row,index in enumerate(current):
                executed[index]+=1
                if lane_logits is not None:lane_logits[index].append(logits[row:row+1])
            decisions=tuple(decide_next(current,step,logits))
            if len(decisions)!=len(current) or any(type(value) is not bool for value in decisions):
                raise ValueError('ragged verifier decision must cover every active lane')
            following=[]
            for row,index in enumerate(current):
                can_continue=step+1<layout.query_lengths[index]
                if decisions[row] and can_continue:
                    following.append(index);continue
                branch=branches[index]
                branch.stage_recurrent_prefix(
                    branch.recurrent_caches,accepted_rows=step+1,
                    offset=branch._origin.layers[0].offset+step+1)
            active=following
        executed=tuple(executed)
        if any(rows<1 for rows in executed):
            raise RuntimeError('ragged verification left a live lane unexecuted')
        for branch,rows in zip(branches,executed):branch.seal_executed_rows(rows)
        physical={}
        for receipt in round_receipts:
            for name,value in receipt['physical_counters'].items():
                physical[name]=physical.get(name,0)+value
        outputs=()
        if lane_logits is not None:
            mx=self._runtime_factory().mx
            outputs=tuple(mx.concatenate(rows,axis=0) for rows in lane_logits)
        return outputs,{
            'route':self.serving_route,'native_n20_ragged_selected':True,
            'layout':layout.receipt(layout.select_backend(RaggedVerifyCapabilities(
                flattened=True,max_lanes=20,max_query_len=layout.max_query_len,
                max_total_rows=layout.logical_rows))),
            'declared_query_lengths':list(layout.query_lengths),
            'executed_query_lengths':list(executed),
            'physical_counters':physical,'round_widths':[r['packed_lanes'] for r in round_receipts],
            'verification_logits_retained':collect_logits,
            'recurrent_transition_peak':'one_selected_successor_per_lane',
            'qualified':False,'price_usable':False}
