# SPDX-License-Identifier: MIT
"""Explicit PP extension of the exact resident-bank forward transaction.

No serving default is changed. Attach only through an identity-bound adapter
profile after preloading the requested closed set and completing bank prepare.
Original occupancies above the proved row band cause whole-forward reference
replay from untouched authoritative state, never partial cache publication.
"""
import hashlib
from pathlib import Path
from .deepseek_v41_source_bank_session import PreparedBankSession


def implementation_identity():
    from .deepseek_v41_source_bank_session import implementation_identity as session_identity
    from ..runtime.gguf.source_resident_prefill import implementation_identity as consumer_identity
    from ..runtime.gguf.source_resident_prefill_wide import implementation_identity as wide_identity
    names=('deepseek_v41_source_prefill_session.py','deepseek_v41_source_prefill_bank.py',
           'deepseek_v41_source_bank.py')
    hashes=''.join(n+hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()for n in names)
    return hashlib.sha256((hashes+session_identity()+consumer_identity()+wide_identity()).encode()).hexdigest()


class PreparedPrefillBankSession(PreparedBankSession):
    max_token_rows=128

    def __init__(self,weights,*args,**kwargs):
        self.prefill_policy=getattr(weights,'source_resident_prefill_policy','reference')
        if self.prefill_policy not in ('expert_major_v1','expert_major_wide_v1'):
            raise ValueError('PP session requires explicit expert-major policy')
        self.prefill_task_policy='padded_v1'
        super().__init__(weights,*args,**kwargs)

    def set_prefill_task_policy(self,policy):
        from ..runtime.gguf.source_resident_prefill_tasks import validate_policy
        if self.active or self.replaying or self.poisoned:
            raise RuntimeError('PP task policy requires an idle healthy bank session')
        self.prefill_task_policy=validate_policy(policy)
        return self.receipt()

    def selected(self,inputs,cache):
        shape=getattr(inputs,'shape',())
        return (not self.active and not self.replaying and len(shape)==2
                and shape[0]==1 and 1<=shape[1]<=self.max_token_rows
                and cache is not None and len(cache)>0)

    def moe(self,x,ids,scores,prefix):
        if x.shape[0]==1:return super().moe(x,ids,scores,prefix)
        if not self.active or self.lease is None:raise RuntimeError('PP MoE needs whole-forward ownership')
        if self.prefill_policy=='expert_major_wide_v1':
            from ..runtime.gguf.source_resident_prefill_wide import bank_prefill_wide as bank_prefill
        else:
            from ..runtime.gguf.source_resident_prefill import bank_prefill
        result=bank_prefill(x,ids,scores,self.lease,layer=int(prefix.split('.')[1]),
            expected_generation=self.lease.generation,check_live=self.check_live,task_policy=self.prefill_task_policy)
        self.statuses.append(result.status);self.roots.extend(result.roots)
        return result.output

    def receipt(self):
        return dict(super().receipt(),prefill=True,max_token_rows=self.max_token_rows,
            prefill_implementation_sha256=implementation_identity(),prefill_policy=self.prefill_policy,prefill_task_policy=self.prefill_task_policy,original_occupancy_limit=128 if self.prefill_policy=='expert_major_wide_v1'else 12,
            unsupported_occupancy='whole_forward_reference_replay',qualified=False)


def prepare_prefill_session(weights,keys,*,ownercheck,cancelled=None,generation=1,
                            budget_bytes=75_000_000_000,max_bank_bytes=3<<30):
    """Explicit owned-idle preparation; adapter must bind this implementation."""
    import mlx.core as mx
    from .deepseek_v41_source_prefill_bank import build_prefill_bank
    if getattr(weights,'source_resident_gather_policy','reference')!='resident_bank_v1':
        raise ValueError('PP bank session requires explicit resident_bank_v1 adapter profile')
    if getattr(weights,'source_resident_prefill_policy','reference')not in ('expert_major_v1','expert_major_wide_v1'):
        raise ValueError('PP bank session requires explicit expert_major_v1 adapter profile')
    if getattr(weights,'source_resident_bank_session',None)is not None:
        raise ValueError('Close the prior resident bank session before PP prepare')
    bank=build_prefill_bank(mx,weights.expert_stream,keys,ownercheck=ownercheck,generation=generation,
        budget_bytes=budget_bytes,max_bank_bytes=max_bank_bytes)
    session=PreparedPrefillBankSession(weights,bank,ownercheck=ownercheck,cancelled=cancelled)
    weights.source_resident_bank_session=session
    return session
