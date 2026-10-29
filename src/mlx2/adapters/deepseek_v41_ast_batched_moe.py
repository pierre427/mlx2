# SPDX-License-Identifier: MIT
"""Explicit M16 resident AST MoE and one whole-tile bank lease.

The AST transaction still owns private cache state and publication. This scope
owns bank resources and the aggregate status gate; missing experts fail closed.
It never streams or replays a failed bank tile. Default routes are unchanged.
"""
import hashlib
from pathlib import Path

LAW='source_ast_resident_rows_v1'
_QUARANTINE=[]


class ResidentCohortMiss(ValueError):
    """A fully drained private tile missed only absent expert entries."""
    def __init__(self):
        super().__init__('Resident AST private cohort miss; drained before publication')
        self.drained=True


def _hybrid_status_summary(xp,statuses,outputs,metadata):
    # Missing LUT entries have kind0 and historically carry1|8|16. Inspect
    # each used group as well: a stale EXISTING expert must never be hidden
    # behind another absent expert's aggregate row flags.
    missing=[];invalid=[];finite=[];row_missing=[]
    if len(metadata)!=len(statuses):raise ValueError('Resident AST status metadata inventory differs')
    for status,out,groups in zip(statuses,outputs,metadata):
        used=groups[:,0]!=0;absent=used&(groups[:,2]==0)
        missing.append(xp.any(absent));row_missing.append(xp.any((status&1)!=0))
        invalid.append(xp.any(used&(groups[:,2]!=0)&(groups[:,5]!=0)))
        invalid.append(xp.any(absent&((groups[:,5]&1)==0)))
        invalid.append(xp.any((status&0xffffffe6)!=0))
        finite.append(xp.all(xp.where(status.reshape(-1,1)==0,xp.isfinite(out),True)))
    miss=xp.any(xp.stack(missing));reported=xp.any(xp.stack(row_missing))
    bad=xp.any(xp.stack(invalid))|(miss!=reported)
    return miss,bad,xp.all(xp.stack(finite))



def implementation_identity():
    from ..runtime.gguf.source_ast_batched_moe import implementation_identity as kernels
    return hashlib.sha256((hashlib.sha256(Path(__file__).read_bytes()).hexdigest()+kernels()).encode()).hexdigest()


class ResidentASTBankScope:
    def __init__(self,weights,ownercheck,*,bank=None,cancelled=None,backend=None,consumer=None,recover_misses=False):
        if not callable(ownercheck)or(cancelled is not None and not callable(cancelled)):
            raise ValueError('Resident AST scope requires ownership/cancellation callbacks')
        session=getattr(weights,'source_resident_bank_session',None)
        if session is not None and any(getattr(session,k,False)for k in('active','replaying','poisoned')):
            raise ValueError('Resident AST cannot borrow an active or poisoned B1 bank session')
        bank=bank if bank is not None else getattr(session,'bank',None)
        if bank is None or getattr(weights.expert_stream,'_resident_bank',None)is not bank:
            raise ValueError('Resident AST requires an attached published bank for this expert store')
        if getattr(weights,'source_resident_gather_policy','reference')!='resident_bank_v1':
            raise ValueError('Resident AST requires explicit resident_bank_v1 profile')
        if type(recover_misses)is not bool:raise ValueError('Resident AST miss recovery must be explicit')
        self.recover_misses=recover_misses
        self.weights,self.bank,self.ownercheck=weights,bank,ownercheck
        self.cancelled=cancelled or(lambda:False);self.backend,self.consumer=backend,consumer
        self.active=self.poisoned=self.completed=False;self.drained=True
        self.lease=self.context=None;self.roots=[];self.statuses=[];self.outputs=[];self.external_roots=();self.metadata=[]
        self.calls=0;self.metadata=[]

    def __enter__(self):
        if self.active or self.poisoned or getattr(self.weights,'_source_ast_bank_scope',None)is not None:
            raise RuntimeError('Resident AST scope is already active or poisoned')
        if getattr(self.weights,'_source_current_frame',None)is not None:
            raise ValueError('Resident AST cannot enter a contiguous Source prefill frame')
        if self.cancelled():raise InterruptedError('Resident AST cancelled before lease')
        if self.ownercheck()is False:raise RuntimeError('Resident AST ownership lost')
        self.context=self.bank.lease(ownercheck=self.ownercheck);self.lease=self.context.__enter__()
        self.active=True;self.weights._source_ast_bank_scope=self
        return self

    def check_live(self):
        if not self.active or self.poisoned or self.weights._source_ast_bank_scope is not self or self.lease is not self.bank:
            raise RuntimeError('Resident AST lease is absent or stale')
        if self.cancelled():raise InterruptedError('Resident AST cancelled')
        self.bank.assert_current(contents=False)
        return True

    def _backend(self):
        if self.backend is None:
            import mlx.core as mx
            return mx
        return self.backend

    def retain(self,result):
        self.check_live();self.calls+=1;self.drained=False
        self.statuses.append(result.status);self.outputs.append(result.output);self.roots.extend(result.roots)
        if self.recover_misses:self.metadata.append(result.route_meta)

    def complete(self,*private_roots):
        """Evaluate once and validate before the caller publishes any cache path."""
        self.check_live()
        if self.completed:raise RuntimeError('Resident AST scope already completed')
        if not self.statuses:raise RuntimeError('Resident AST tile produced no MoE status')
        self.external_roots=private_roots;mx=self._backend();self.drained=False
        miss=None
        if self.recover_misses:
            miss,bad,finite=_hybrid_status_summary(mx,self.statuses,self.outputs,self.metadata)
        else:
            bad=mx.any(mx.stack([mx.any(s!=0)for s in self.statuses]))
            finite=mx.all(mx.stack([mx.all(mx.isfinite(o))for o in self.outputs]))
        completion=(bad,finite)if miss is None else(bad,finite,miss)
        self.roots.extend(completion);mx.eval(self.roots,self.external_roots,*completion)
        self.drained=True
        if self.ownercheck()is False:raise RuntimeError('Resident AST ownership lost before publication')
        if self.cancelled():raise InterruptedError('Resident AST cancelled before publication')
        if bool(bad.item())or not bool(finite.item()):
            raise ValueError('Resident AST bank cohort/status/nonfinite failure; no streaming fallback')
        if miss is not None and bool(miss.item()):raise ResidentCohortMiss()
        self.completed=True
        return self.receipt()

    def receipt(self):
        return dict(law=LAW,active=self.active,completed=self.completed,poisoned=self.poisoned,
            bank_generation=self.bank.generation,moe_tiles=self.calls,rows=16,
            host_expert_id_reads=0,ordinary_fallbacks=0,gu_coefficient_reuse_rows=2,
            recover_misses=self.recover_misses,qualified=False,defaulted=False)

    def __exit__(self,kind,error,tb):
        stream=self.weights.expert_stream
        if (getattr(stream,'_active',False)or getattr(stream,'_leased',{})
                or getattr(stream,'_pending_uploads',{})or getattr(stream,'_pending_host',{})
                or getattr(stream,'_completion_collector',None)is not None):
            # A provider final fence failed or its ownership window is still
            # open. Completed output roots alone do not release bank anchors.
            self.poisoned=True;_QUARANTINE.append(self)
            raise RuntimeError('Resident AST provider window did not complete; bank lease quarantined')from error
        missing_gate=kind is None and not self.completed
        if not self.drained:
            try:self._backend().eval(self.roots,self.external_roots);self.drained=True
            except BaseException:
                self.poisoned=True;_QUARANTINE.append(self)
                # Keep the context and its live lease, every encoded resource,
                # and the caller-supplied private cache roots. Never release.
                raise
        self.context.__exit__(kind,error,tb);self.active=False
        self.weights._source_ast_bank_scope=None
        self.lease=self.context=None;self.roots=[];self.statuses=[];self.outputs=[];self.external_roots=()
        if missing_gate:raise RuntimeError('Resident AST scope.complete is required before publication')
        return False


def resident_scope(weights,ownercheck=None,**options):
    if ownercheck is None:
        session=getattr(weights,'source_resident_bank_session',None)
        bank=options.get('bank',getattr(session,'bank',None))
        ownercheck=getattr(session,'ownercheck',None)or getattr(bank,'_ownercheck',None)
    return ResidentASTBankScope(weights,ownercheck,**options)


def moe_tile(weights,x,ids,scores,prefix):
    """F32[16,5120] +GPU U32/F32[16,6] under an active whole-tile scope."""
    import mlx.core as mx
    from ..runtime.models.deepseek_v41_quant import fused_quantize
    weights._ready()
    if(x.shape!=(16,5120)or x.dtype!=mx.float32 or ids.shape!=(16,6)or ids.dtype!=mx.uint32
       or scores.shape!=ids.shape or scores.dtype!=mx.float32):
        raise ValueError('Resident AST requires F32[16,5120],U32/F32[16,6]')
    if getattr(weights,'expert_math_policy',None)!='routed_fused_fp32':
        raise ValueError('Resident AST requires served routed_fused_fp32 boundaries')
    if getattr(weights,'source_ffn_input_policy','reference')!='reference':
        raise ValueError('Resident AST requires reference per-row FFN input')
    scope=getattr(weights,'_source_ast_bank_scope',None)
    if not isinstance(scope,ResidentASTBankScope):raise ValueError('Resident AST MoE needs an explicit whole-tile bank scope')
    scope.check_live()
    fields=prefix.split('.')
    if len(fields)<2 or fields[0]!='blk'or not fields[1].isdigit()or not 0<=int(fields[1])<40:
        raise ValueError('Resident AST layer prefix differs')
    # Same Source per-row block32 FP8 QAT and BF16 boundary as the ordinary
    # M1-law AST adapter. x already has physical F32 storage.
    prepared=fused_quantize(x.astype(mx.float32),'fp8').astype(x.dtype).astype(mx.float32)
    consumer=scope.consumer
    if consumer is None:
        from ..runtime.gguf.source_ast_batched_moe import decode_tile
        consumer=decode_tile
    result=consumer(prepared,ids,scores,scope.lease,layer=int(fields[1]),
        expected_generation=scope.bank.generation,check_live=scope.check_live)
    scope.retain(result)
    counts=weights.scheduled_expert_calls
    for key,amount in(('source_ast_resident_m16_moe',1),('source_ast_resident_gpu_occurrences',96),
                     ('source_ast_resident_host_expert_ids',0),('source_ast_resident_streaming_fallbacks',0)):
        counts[key]=counts.get(key,0)+amount
    if scope.recover_misses:
        # Failed rows stay private and are never admitted. Mask their sentinel
        # NaNs only to prevent a missing coefficient from causing spurious
        # downstream ID/nonfinite errors before the aggregate status fence.
        return mx.where(result.status.reshape(16,1)!=0,mx.zeros_like(result.output),result.output)
    return result.output
