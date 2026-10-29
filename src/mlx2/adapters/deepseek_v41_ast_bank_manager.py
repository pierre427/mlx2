# SPDX-License-Identifier: MIT
"""Owned hybrid AST banks with one exact, private whole-tree miss recovery.

Bank maxima are soft performance choices, never model-load reservations. Banks
retain whole backing cohorts until a fence proves replacement or retirement.
Mined contracts: provenance/deepseek-v41-ast-hybrid-bank-manager.json.
"""
from collections import Counter
from contextlib import contextmanager
import hashlib
from pathlib import Path
import threading

LAW='source_ast_hybrid_rows_v1'
M1='source_ast_m1_rows_v1'
RESIDENT='source_ast_resident_rows_v1'
_QUARANTINE=[]


def implementation_identity():
    names=('deepseek_v41_ast_bank_manager.py','deepseek_v41_source_bank.py',
           'deepseek_v41_source_prefill_bank.py','deepseek_v41_source_bank_session.py')
    return hashlib.sha256(''.join(n+hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()for n in names).encode()).hexdigest()


class SourceASTBankManager:
    """One model owner; no background thread, speculative publication or IO."""
    def __init__(self,model,*,ownercheck,cancelled=None,available_bytes=None,backend=None,
                 builder=None,cooldown_trees=2):
        if not callable(ownercheck)or(cancelled is not None and not callable(cancelled)):
            raise ValueError('Hybrid AST manager requires live owner/cancellation callbacks')
        if type(cooldown_trees)is not int or cooldown_trees<1:raise ValueError('Invalid hybrid bank rebuild cooldown')
        self.model,self.weights,self.ownercheck=model,model.weights,ownercheck
        self.cancelled=cancelled or(lambda:False);self.available_bytes=available_bytes
        self.backend,self.builder=backend,builder;self.cooldown_trees=cooldown_trees
        self.bank=None;self.active=self.poisoned=self.closed=False;self._recording=False;self._retiring=False
        self._lock=threading.RLock();self.observed=Counter();self.hint_predictions=Counter();self._committed_hint=None;self.current=set();self.last_keys=frozenset()
        self.misses_since_build=0;self.failed_coverage=False;self.last=None;self.last_retirement=None;self.cooldown_remaining=0
        self.stats=dict(resident_trees=0,m1_trees=0,cohort_misses=0,bank_builds=0,bank_reuses=0,
                        bank_replacements=0,pressure_retirements=0,construction_allocation_failures=0)

    def check_live(self):
        if self.closed or self.poisoned:raise RuntimeError('Hybrid AST manager is closed or quarantined')
        if self.cancelled()and not self._retiring:raise InterruptedError('Hybrid AST cancelled')
        if self.ownercheck()is False:raise RuntimeError('Hybrid AST ownership lost')
        if self.model.weights is not self.weights:raise ValueError('Hybrid AST weights owner changed')
        return True

    def _idle(self):
        self.check_live();w=self.weights
        if (getattr(self.model,'_mlx2_ds41_ast_active',None)is not None
                or getattr(self.model,'_mlx2_ds41_ast_poisoned',False)
                or getattr(self.model,'_weight_executions',0)
                or getattr(w,'_source_ast_bank_scope',None)is not None):
            raise RuntimeError('Hybrid bank operation requires a completed private tree')
        if getattr(w,'source_resident_bank_session',None)is not None:
            raise ValueError('Hybrid manager cannot replace another bank session owner')
        if getattr(w.expert_stream,'_resident_bank',None)is not self.bank:
            raise ValueError('Hybrid manager bank/store ownership differs')
        if self.bank is not None:
            self.bank.assert_current()
            if self.bank.receipt()['active_leases']:raise ValueError('Hybrid bank still has live leases')

    def _available(self):
        if self.available_bytes is None:
            from ..memory import host_available_bytes
            value=host_available_bytes(host_signals=True)
        else:value=self.available_bytes()
        if value is not None and(type(value)is not int or value<0):raise ValueError('Invalid measured hybrid availability')
        return value

    def _working_floor(self):
        # Actual declared maximum96 M1 occurrences per layer, not a complete
        #384-expert layer/model. Nothing is allocated or reserved for this floor.
        store=self.weights.expert_stream
        return 96*store.minimum_cache_bytes+getattr(store,'_adaptive_staging',0)

    def _retire(self,reason):
        self._idle()
        if self.bank is not None:
            try:self.bank.close()
            except BaseException:
                self.poisoned=True;_QUARANTINE.append((self,self.bank));raise
            self.bank=None;self.stats['pressure_retirements']+=1
        self.failed_coverage=False;self.misses_since_build=0
        self.cooldown_remaining=self.cooldown_trees
        self.last=dict(action='whole_bank_retired',reason=reason,whole_backing_cohorts=True)
        self.last_retirement=dict(self.last)

    def before_ordinary(self):
        with self._lock:
            if self.active:return
            self._idle()
            if self.bank is None:return
            available=self._available();store=self.weights.expert_stream
            target=getattr(store,'_adaptive_payload_ceiling',store.cache_budget_bytes)
            payload=self.bank.plan.payload_bytes
            if available==0 or payload+self._working_floor()>min(store.cache_budget_bytes,target):
                self._retire('actual_pressure_or_imminent_M1_working_floor')

    def record_routes(self,prefix,values):
        if not self._recording:return
        fields=prefix.split('.')
        if len(fields)!=2 or fields[0]!='blk' or not fields[1].isdecimal():raise ValueError('Hybrid route prefix differs')
        layer=int(fields[1])
        for row in values:
            for expert in row:
                key=(layer,int(expert));self.current.add(key);self.observed[key]+=1

    def _plan(self,available,replacing):
        store=self.weights.expert_stream;meta=40*384*4*4
        if available is None:return None,'unknown_actual_availability'
        ceiling=store.cache_budget_bytes;cached=store._cache_bytes;floor=self._working_floor()
        room=min(available,max(0,ceiling-cached))-3*meta
        if room<=0:return None,'no_actual_construction_room'
        max_bank=min(4<<30,room//2)//4*4
        candidates=[]
        for key,value in store._cache.items():
            layout=getattr(value,'layout',None)
            if layout is None or layout.kind not in ('original','q1'):continue
            if layout.nbytes<=max_bank:candidates.append((key,layout.nbytes))
        old_keys=self.bank.plan.keys if replacing else frozenset()
        candidates.sort(key=lambda pair:(pair[0]not in old_keys if replacing else pair[0]not in self.last_keys,
            pair[0]not in self.last_keys,-self.observed[pair[0]],-self.hint_predictions[pair[0]],pair[0]))
        keys=[];payload=0;limit=max(0,ceiling-floor)
        for key,size in candidates:
            if payload+size<=limit:keys.append(key);payload+=size
        if not keys:return None,'no_cohort_with_M1_acquisition_room'
        if replacing:
            old=self.bank.plan.keys
            if not old.issubset(keys):return None,'replacement_would_drop_old_cohort'
            if set(keys)==set(old):return None,'no_strict_cohort_growth'
            # Old shared backing remains live throughout atomic replacement;
            # charge ALL new payload plus scratch, never incremental free credit.
            if payload+max_bank+3*meta>available:return None,'replacement_copy_not_available'
        from .deepseek_v41_source_bank import plan_bank
        # Match finite Metal binding capacity before any allocation. Drop only
        # optional tail keys; never drop an old shared-backing replacement key.
        while keys:
            try:
                plan=plan_bank(store,tuple(keys),generation=getattr(store,'_resident_bank_generation',0)+1,
                    budget_bytes=ceiling,max_bank_bytes=max_bank,staging_bytes=0,
                    construction='atomic'if replacing else'incremental')
            except ValueError as error:
                if not any(reason in str(error)for reason in ('finite Metal input binding limit','peak exceeds expert ceiling')):raise
                if replacing and keys[-1]in self.bank.plan.keys:return None,'finite_bank_bindings'
                keys.pop();continue
            if len(plan.bank_nbytes)>22:
                if replacing and keys[-1]in self.bank.plan.keys:return None,'finite_AST_bank_bindings'
                keys.pop();continue
            return (tuple(keys),max_bank,plan),'candidate'
        return None,'finite_bank_bindings'

    def _prepare(self):
        self._idle();available=self._available()
        if self.bank is not None and not self.failed_coverage:
            self.stats['bank_reuses']+=1;return True,'existing_bank'
        if self.cooldown_remaining:return False,'whole_bank_retirement_cooldown'
        replacing=self.bank is not None
        if replacing and self.misses_since_build<self.cooldown_trees:return False,'cohort_miss_cooldown'
        if replacing and self.last_keys and self.last_keys.issubset(self.bank.plan.keys):
            self.failed_coverage=False;self.stats['bank_reuses']+=1
            return True,'actual_M1_cohort_coverage_regained'
        planned,reason=self._plan(available,replacing)
        if planned is None:return False,reason
        keys,max_bank,plan=planned;store=self.weights.expert_stream
        if self.backend is None:
            import mlx.core as mx
        else:mx=self.backend
        builder=self.builder
        if builder is None:
            if replacing:
                from .deepseek_v41_source_bank import build_bank as builder
            else:
                from .deepseek_v41_source_prefill_bank import build_prefill_bank as builder
        old=self.bank
        try:
            if replacing:
                with old.replacement(keys):
                    fresh=builder(mx,store,keys,ownercheck=self.check_live,generation=plan.generation,
                        budget_bytes=store.cache_budget_bytes,max_bank_bytes=max_bank,staging_bytes=0)
            else:
                fresh=builder(mx,store,keys,ownercheck=self.check_live,generation=plan.generation,
                    budget_bytes=store.cache_budget_bytes,max_bank_bytes=max_bank,staging_bytes=0)
        except MemoryError:
            if getattr(store,'_error',None)or getattr(store,'_resident_bank_preparing',None):
                self.poisoned=True;_QUARANTINE.append((self,old));raise
            self.stats['construction_allocation_failures']+=1
            self.bank=old;return False,'actual_bank_allocation_failed'
        except BaseException:
            if getattr(store,'_error',None)or getattr(store,'_resident_bank_preparing',None):
                self.poisoned=True;_QUARANTINE.append((self,old))
            raise
        self.bank=fresh;self.failed_coverage=False;self.misses_since_build=0
        self.stats['bank_builds']+=1;self.stats['bank_replacements']+=int(replacing)
        return True,'prepared_from_actual_cached_cohort'

    @contextmanager
    def _consumer(self,resident):
        w=self.weights;saved=getattr(w,'source_resident_gather_policy','reference')
        if saved!='reference':raise ValueError('Hybrid manager requires ordinary gather outside private execution')
        if resident:w.source_resident_gather_policy='resident_bank_v1'
        try:yield
        finally:w.source_resident_gather_policy=saved

    def _m1(self,attempt):
        self.current=set();self._recording=True
        try:
            with self._consumer(False):result=attempt(M1)
        finally:self._recording=False
        self.last_keys=frozenset(self.current);self.stats['m1_trees']+=1
        if self.failed_coverage:self.misses_since_build+=1
        if self.cooldown_remaining:self.cooldown_remaining-=1
        return result

    def execute(self,model,attempt):
        from .deepseek_v41_ast_batched_moe import ResidentCohortMiss
        if model is not self.model:raise ValueError('Foreign hybrid target model')
        with self._lock:
            if self.active:raise RuntimeError('Hybrid AST target execution is already active')
            self.before_ordinary();ready,reason=self._prepare();self.active=True
            fallback=False
            try:
                if ready:
                    try:
                        with self._consumer(True):result=attempt(RESIDENT)
                        self.stats['resident_trees']+=1;used='resident'
                    except ResidentCohortMiss as error:
                        # The attempted private transaction and bank lease must
                        # already be aborted/drained before any recovery call.
                        if not error.drained:raise
                        self._idle();self.check_live();self.stats['cohort_misses']+=1
                        self.failed_coverage=True;self.misses_since_build=0
                        available=self._available();store=self.weights.expert_stream
                        if (available is None or available<store.minimum_cache_bytes+getattr(store,'_adaptive_staging',0)
                                or self.bank.plan.payload_bytes+self._working_floor()>store.cache_budget_bytes):
                            self._retire('cohort_miss_pressure_or_full_pinned_acquisition_limit')
                            reason='cohort_miss_whole_bank_retired_for_progress'
                        else:reason='cohort_miss_existing_backing_retained'
                        result=self._m1(attempt);used='m1';fallback=True
                else:result=self._m1(attempt);used='m1'
                self.last=dict(law=LAW,actual_expert_route=used,exact_M1_recovery=fallback,reason=reason,
                    bank_generation=None if self.bank is None else self.bank.generation,
                    bank_receipt=None if self.bank is None else self.bank.receipt(),
                    source_implementation=implementation_identity(),stats=dict(self.stats),
                    cohort_miss_cooldown_trees=self.cooldown_trees,last_retirement=self.last_retirement,qualified=False,defaulted=False,
                    original_precision_unchanged=True,cache_identity_rewritten=False)
                result.receipt['logical_target_invocations']=1
                result.receipt['successful_target_invocations']=1
                result.receipt['physical_target_invocations']=2 if fallback else 1
                result.receipt['failed_private_target_invocations']=int(fallback)
                if fallback:
                    for name in ('layer_traversals','grouped_moe_calls','attention_calls','shared_expert_calls','head_calls'):
                        if name in result.receipt:result.receipt[name]*=2
                result.receipt['hybrid_bank']=self.last
                return result
            except BaseException:
                scope=getattr(self.weights,'_source_ast_bank_scope',None)
                if (getattr(model,'_mlx2_ds41_ast_poisoned',False)or getattr(scope,'poisoned',False)
                        or(self.bank is not None and getattr(self.bank,'_leases',0))):
                    self.poisoned=True;_QUARANTINE.append((self,model,scope))
                raise
            finally:self.active=False

    def hint_binding(self):
        from .deepseek_v41_ast_target import source_revision_token
        from .deepseek_v41_source_bank import _binding
        import json
        store=self.weights.expert_stream
        digest=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()
        return dict(checkpoint=self.weights.artifact['identity']['fingerprint'],
            source_revision=digest((self.model.identity,source_revision_token(LAW,'fully_batched_m1_v1'))),
            quantization=digest((self.weights.expert_math_policy,getattr(store,'q1_binding',None),getattr(store,'hot_binding',None))),
            layout=digest(_binding(store)),resource_abi='mlx2.source-original-q1-packed-bank.v1')

    def record_committed_experts(self,experts,*,covered_tokens,snapshot_revision,token_digest):
        """Adapter post-commit evidence only; physical-tree unions are excluded.

        The calling source adapter must supply actual selected/ordinary route
        observations and the exact checkpoint publication identity. The bank
        manager never derives committed IDs from its speculative cohort.
        """
        from ..runtime.expert_residency_hints import ExpertResidencyHints
        self.check_live()
        hint=ExpertResidencyHints(**self.hint_binding(),snapshot_revision=snapshot_revision,
            token_digest=token_digest,covered_tokens=covered_tokens,layer_expert_counts=(384,)*40,
            experts=tuple(sorted(set(experts))),phase='committed',coverage='exact')
        self._committed_hint=hint
        return hint

    def snapshot_hints(self,*,covered_tokens,snapshot_revision,token_digest):
        self.check_live();hint=self._committed_hint
        if (hint is None or hint.covered_tokens!=covered_tokens or hint.snapshot_revision!=snapshot_revision
                or hint.token_digest!=token_digest):return None
        return hint.for_restore(covered_tokens,self.hint_binding())

    def prefetch(self,hints,*,prefix_tokens,snapshot_revision=None,token_digest=None):
        """Queue optional non-owning predictions; never authorize KV or reserve.

        Actual lazy expert acquisition remains in the exact target. Existing
        cached candidates can be prioritized at the next owned bank boundary.
        This method claims no weight upload, IO avoidance or resource lease.
        """
        from ..runtime.expert_residency_hints import ExpertResidencyHints
        self.check_live()
        if type(hints)is not ExpertResidencyHints:return dict(accepted=False,reason='absent_or_foreign_metadata')
        restored=hints.for_restore(prefix_tokens,self.hint_binding())
        if (restored is None or(snapshot_revision is not None and hints.snapshot_revision!=snapshot_revision)
                or(token_digest is not None and hints.token_digest!=token_digest)):
            return dict(accepted=False,reason='stale_speculative_or_prefix_mismatch')
        for key in restored.experts:self.hint_predictions[key]+=1
        return dict(accepted=True,experts=len(restored.experts),coverage=restored.coverage,
            metadata_only=True,expert_IO=False,resource_lease=False,reservation_bytes=0,
            numerical_authority=False,scope='Prioritize already cached candidates; target lazily acquires real routes')

    def receipt(self):
        return dict(law=LAW,active=self.active,poisoned=self.poisoned,closed=self.closed,
            bank_generation=None if self.bank is None else self.bank.generation,stats=dict(self.stats),last=self.last,
            fallback='Only drained missing-cohort status; all other failures refuse',qualified=False,defaulted=False)

    def close(self):
        with self._lock:
            if self.active or self.poisoned:raise RuntimeError('Cannot close active or quarantined hybrid AST manager')
            if not self.closed:
                # Request cancellation cannot strand a completed bank; owner
                # and poison guards remain mandatory during its final fence.
                self._retiring=True
                try:self._retire('adapter_close');self.closed=True
                finally:self._retiring=False
