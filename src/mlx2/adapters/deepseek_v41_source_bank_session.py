# SPDX-License-Identifier: MIT
"""Explicit prepared-bank decode epochs with exact private cache publication.

Bank resources stay leased through one complete forward. A failed drain keeps
all owners quarantined. Miss replay uses another private exact frontier and is
bounded to one ordinary attempt. This route is default-off and unqualified.
"""
from contextlib import nullcontext
import hashlib
from pathlib import Path

POLICY='resident_bank_v1'
CACHE_FORK_POLICIES=('physical_copy_v1','descriptor_cow_v1')
EXPERT_KERNEL_VARIANTS=('reference','down8_v1','route_fused_v1','route_down8_v1')
SUBMISSION_POLICIES=('per_layer_async_v1','whole_forward_v1','layer_group_2_v1','layer_group_4_v1')
OPERAND_POLICIES=('full_operands_v1','coefficient_views_v1')
RAW_ARRAYS=('keys','buf_kv','buf_gate','_pooled','_index_keys','hash_tail')


def implementation_identity():
    from ..runtime.gguf.source_resident_bank_gather import implementation_identity as consumer_identity
    from ..runtime.gguf.source_resident_bank_kernels import implementation_identity as variant_identity
    from ..runtime import cow_cache
    return hashlib.sha256((hashlib.sha256(Path(__file__).read_bytes()).hexdigest()+consumer_identity()
        +variant_identity()+hashlib.sha256(Path(cow_cache.__file__).read_bytes()).hexdigest()).encode()).hexdigest()


def _revision(cache):
    return tuple((tuple(s.meta_state),tuple(id(getattr(s,k,None))for k in RAW_ARRAYS))for s in cache)


def _fork(cache,policy='physical_copy_v1'):
    if policy not in CACHE_FORK_POLICIES:raise ValueError('Invalid Source bank cache fork policy')
    if policy=='descriptor_cow_v1' and cache and any(s.xp.__name__=='mlx.core'for s in cache):
        if not all(s.xp.__name__=='mlx.core'for s in cache):
            raise ValueError('Source cache descriptor fork cannot mix backends')
        from ..runtime.cow_cache import snapshot_prompt_cache_descriptors
        # Existing maintained COW clones the descriptor graph and stop_gradient
        # wrappers, including full backing capacity, without from_state's
        # trimmed serialization. Array updates rebind only the private wrappers.
        clone,_,_=snapshot_prompt_cache_descriptors(cache,require_committed=True)
        return clone
    # from_state owns physical copies of every array; shallow aliases are unsafe
    # because V4.1 appends pooled rows through mutable MLX array wrappers.
    return [type(s).from_state(s.state,s.meta_state,xp=s.xp)for s in cache]


def _isolated(left,right):
    if left is right or len(left)!=len(right):return False
    for a,b in zip(left,right):
        if a is b or a.meta_state[:-1]!=b.meta_state[:-1]:return False
        for name in RAW_ARRAYS:
            av,bv=getattr(a,name,None),getattr(b,name,None)
            if av is not None and av is bv:return False
    return True


def _discard(cache):
    for state in cache:state.close()


def _commit(owner,candidate,expected):
    if _revision(owner)!=expected:raise ValueError('Published cache changed before bank frontier commit')
    # Prepare all dictionaries before touching any owner. Candidate array wrappers are
    # independent of private math: physical copies or maintained COW aliases.
    dictionaries=[dict(s.__dict__)for s in candidate]
    for state in candidate:state.validate_state()
    for state,values in zip(owner,dictionaries):state.__dict__=values
    return 'committed'


class PreparedBankSession:
    def __init__(self,weights,bank,*,ownercheck,cancelled=None,consumer=None,backend=None,
                 cache_fork_policy='physical_copy_v1'):
        if not callable(ownercheck)or cancelled is not None and not callable(cancelled):
            raise ValueError('Prepared bank session requires ownership/cancellation callbacks')
        if getattr(weights,'source_resident_gather_policy','reference')!=POLICY:
            raise ValueError('Prepared bank session requires explicit resident_bank_v1 policy')
        self.weights,self.bank=weights,bank
        self.ownercheck,self.cancelled=ownercheck,cancelled or(lambda:False)
        self.consumer,self.backend=consumer,backend
        self.active=self.replaying=self.poisoned=False
        if cache_fork_policy not in CACHE_FORK_POLICIES:raise ValueError('Invalid Source bank cache fork policy')
        self.cache_fork_policy=cache_fork_policy
        self.expert_kernel_variant='reference'
        self.submission_policy='per_layer_async_v1'
        self.operand_policy='full_operands_v1'
        self.roots=[];self.statuses=[];self.lease=None;self._quarantined=[]
        self.stats=dict(forwards=0,commits=0,misses=0,ordinary_replays=0,host_route_materializations=0,
                        whole_forward_evaluations=0,publication_copy_evaluations=0,
                        publication_descriptor_evaluations=0,layer_async_submitted=0,layer_async_deferred=0)
        self.last_receipt=None

    def selected(self,inputs,cache):
        return (not self.active and not self.replaying and getattr(inputs,'shape',())[:2]==(1,1)
                and cache is not None and len(cache)>0 and cache[0].offset>0)

    def retain(self,*values):
        if not self.active:raise RuntimeError('No active bank epoch for lazy dense roots')
        self.roots.extend(values)

    def moe(self,x,ids,scores,prefix):
        if not self.active or self.lease is None:raise RuntimeError('Bank MoE requires whole-forward ownership')
        consumer=self.consumer
        if consumer is None:
            if self.expert_kernel_variant=='reference':
                from ..runtime.gguf.source_resident_bank_gather import bank_decode
                consumer=bank_decode
            else:
                from ..runtime.gguf.source_resident_bank_kernels import decode
                consumer=decode
        options={}if self.expert_kernel_variant=='reference'else{'variant':self.expert_kernel_variant}
        result=consumer(x,ids,scores,self.lease,layer=int(prefix.split('.')[1]),
                        expected_generation=self.lease.generation,check_live=self.check_live,**options)
        self.statuses.append(result.status)
        self.roots.extend((result.output,result.status,result.middle,result.route_meta))
        return result.output

    def check_live(self):
        if not self.active or self.poisoned or self.lease is not self.bank:
            raise RuntimeError('Source bank epoch is absent or stale')
        self.bank.assert_current(contents=False)
        return True

    def _backend(self):
        if self.backend is None:
            import mlx.core as mx
            return mx
        return self.backend

    def set_cache_fork_policy(self,policy):
        if self.active or self.replaying or self.poisoned:
            raise RuntimeError('Source cache fork policy requires an idle healthy bank session')
        if policy not in CACHE_FORK_POLICIES:raise ValueError('Invalid Source bank cache fork policy')
        self.cache_fork_policy=policy
        return self.receipt()

    def set_expert_kernel_variant(self,variant):
        if self.active or self.replaying or self.poisoned:
            raise RuntimeError('Expert kernel variant requires an idle healthy bank session')
        if variant not in EXPERT_KERNEL_VARIANTS:raise ValueError('Invalid Source bank expert kernel variant')
        self.expert_kernel_variant=variant
        return self.receipt()

    def set_submission_policy(self,policy):
        if self.active or self.replaying or self.poisoned:
            raise RuntimeError('Source submission policy requires an idle healthy bank session')
        if policy not in SUBMISSION_POLICIES:raise ValueError('Invalid Source bank submission policy')
        self.submission_policy=policy
        return self.receipt()

    def set_operand_policy(self,policy):
        if self.active or self.replaying or self.poisoned:
            raise RuntimeError('Source operand policy requires an idle healthy bank session')
        if policy not in OPERAND_POLICIES:raise ValueError('Invalid Source bank operand policy')
        if policy=='coefficient_views_v1':
            self.weights.prepare_resident_coefficient_views(ownercheck=self.ownercheck)
        self.operand_policy=policy
        return self.receipt()

    def forward(self,model,inputs,cache,ordinary):
        if (getattr(model, 'kv_storage_policy', 'reference') != 'reference'
                or any(hasattr(row, '_packed_local') for row in cache)):
            raise ValueError('Source bank requires reference FP32 cache state')
        if self.poisoned:raise RuntimeError('Bank session has quarantined incomplete GPU resources')
        if self.active or self.replaying:raise RuntimeError('Nested bank transaction')
        if self.cancelled():raise InterruptedError('Bank decode cancelled before execution')
        from .deepseek_v41_source_resident import ForwardTransaction
        mx=self._backend();self.ownercheck()
        fork_policy=self.cache_fork_policy
        tx=ForwardTransaction(cache,fork=lambda rows:_fork(rows,fork_policy),validate_fork=_isolated,revision=_revision,
                              commit=_commit,discard=_discard,max_replays=1)
        context=self.bank.lease(ownercheck=self.ownercheck);lease=context.__enter__()
        self.lease=lease;self.active=True;self.roots=[];self.statuses=[]
        completed=set();result=None;status=None;drained=False;released=False
        def wait(frontier):
            if id(frontier)in completed:return
            mx.eval([s.state for s in frontier])
            completed.add(id(frontier))
            descriptor=(fork_policy=='descriptor_cow_v1'and all(s.xp.__name__=='mlx.core'for s in frontier))
            key='publication_descriptor_evaluations'if descriptor else'publication_copy_evaluations'
            self.stats[key]+=1
        try:
            working=tx.begin();result=ordinary(working)
            if not self.statuses:raise RuntimeError('Bank forward produced no GPU route status')
            status=mx.any(mx.stack([mx.any(s!=0)for s in self.statuses]))
            finite=mx.all(mx.isfinite(result[0]))
            mx.eval(result,[s.state for s in working],self.roots,status,finite)
            completed.add(id(working));drained=True;self.stats['whole_forward_evaluations']+=1
            self.ownercheck();miss=bool(status.item());valid=bool(finite.item())
            verdict='cancelled'if self.cancelled()else'miss'if miss else'ok'if valid else'error'
            outcome=tx.finish(status=verdict,wait=wait)
            self.stats['forwards']+=1
            self.active=False;self.lease=None;context.__exit__(None,None,None);released=True
            if verdict=='ok':self.stats['commits']+=1
            elif verdict=='miss'and outcome=='ordinary_replay_required':
                self.stats['misses']+=1;self.stats['ordinary_replays']+=1
                self.roots=[];self.statuses=[];self.replaying=True
                try:
                    replay=tx.begin();result=ordinary(replay)
                    mx.eval(result,[s.state for s in replay]);completed.add(id(replay))
                    if self.cancelled():
                        tx.finish(status='cancelled',wait=wait);raise InterruptedError('Bank replay cancelled before publication')
                    if not bool(mx.all(mx.isfinite(result[0])).item()):
                        tx.finish(status='error',wait=wait);raise RuntimeError('Ordinary bank-miss replay was nonfinite')
                    self.ownercheck();tx.finish(status='ok',wait=wait);self.stats['commits']+=1
                finally:self.replaying=False
            elif verdict=='cancelled':raise InterruptedError('Bank decode cancelled without publication')
            else:raise RuntimeError('Bank decode rejected without publication: '+verdict)
            self.last_receipt=dict(status=verdict,transaction=tx.receipt(),bank_generation=self.bank.generation,
                cache_fork_policy=fork_policy,
                expert_kernel_variant=self.expert_kernel_variant,
                submission_policy=self.submission_policy,
                operand_policy=self.operand_policy,
                ordinary_replay=verdict=='miss',qualified=False,selected=True,observed_used=True,
                cache_integration='exact private fork and independent candidate publication',stats=dict(self.stats))
            return result
        except BaseException:
            if not drained:
                try:
                    mx.eval(self.roots,[s.state for s in tx.working]if tx.working is not None else [])
                    drained=True
                except BaseException:
                    self.poisoned=True
                    self._quarantined.append((context,lease,tx,result,status,tuple(self.roots)))
            if drained and not tx.closed and tx.working is not None:
                try:tx.finish(status='error',wait=wait)
                except BaseException:
                    self.poisoned=True;self._quarantined.append((context,lease,tx,result,status,tuple(self.roots)))
            raise
        finally:
            self.active=False;self.lease=None
            if not released and not self.poisoned:context.__exit__(None,None,None)
            if not self.poisoned:self.roots=[];self.statuses=[]

    def receipt(self):
        return dict(policy=POLICY,prepared=True,active=self.active,poisoned=self.poisoned,
            cache_fork_policy=self.cache_fork_policy,
            expert_kernel_variant=self.expert_kernel_variant,
            submission_policy=self.submission_policy,
            operand_policy=self.operand_policy,
            quarantined_epochs=len(self._quarantined),qualification='unqualified',stats=dict(self.stats),last=self.last_receipt)

    def close(self):
        if self.active or self.replaying or self.poisoned:
            raise RuntimeError('Cannot close a live or quarantined Source bank session')
        self.ownercheck();self.bank.close()
        if getattr(self.weights,'source_resident_bank_session',None)is self:
            self.weights.source_resident_bank_session=None
