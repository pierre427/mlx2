# SPDX-License-Identifier: MIT
"""Opt-in host-only Source policy call observer for caller-owned diagnostics.

Import is CPU-only. install() checks paired CPG ownership before importing model
modules. It wraps existing helper calls, performs no tensor/logit reads or extra
MLX synchronization, excludes synthetic first-use checks, and flushes once per
completed/failed outer model forward rather than once per helper call.
"""
from __future__ import annotations
import atexit
from collections import Counter
from functools import wraps
import importlib
from pathlib import Path
import threading
import time

_INSTALLED=None


class Observer:
    def __init__(self,unit,*,writer=None):
        self.unit=Path(unit);self.path=self.unit/'policy-observer.json'
        self.local=threading.local();self.lock=threading.RLock()
        self.calls=Counter();self.completed_forwards=self.failed_forwards=0
        self.check_calls_excluded=0;self.sequence=0;self.patches=[]
        self.last_model=None;self.writer=writer
        self.last_native_vision=None;self.verification_decisions=[];self.verification_dropped=0;self.verification_metadata_declined=0;self.finalized=False

    def _count(self,key):
        if getattr(self.local,'forward_depth',0)and not getattr(self.local,'checking',0):
            with self.lock:self.calls[key]+=1

    def patch(self,target,name,label,*,check=False):
        original=getattr(target,name)
        if check:
            @wraps(original)
            def wrapped(*args,**kwargs):
                self.local.checking=getattr(self.local,'checking',0)+1
                try:return original(*args,**kwargs)
                finally:
                    self.local.checking-=1
                    with self.lock:self.check_calls_excluded+=1
        else:
            @wraps(original)
            def wrapped(*args,**kwargs):
                result=original(*args,**kwargs)
                self._count(label)
                return result
        setattr(target,name,wrapped);self.patches.append((target,name,original,wrapped))

    def patch_qat(self,model_class):
        original=model_class.qat
        @wraps(original)
        def wrapped(model,*args,**kwargs):
            result=original(model,*args,**kwargs)
            if getattr(model.args,'simulate_quantization',False):
                self._count('model.qat.'+getattr(model,'qat_policy','reference'))
            return result
        model_class.qat=wrapped;self.patches.append((model_class,'qat',original,wrapped))

    def patch_forward(self,model_class):
        original=model_class._forward_ordinary
        @wraps(original)
        def wrapped(model,*args,**kwargs):
            depth=getattr(self.local,'forward_depth',0);self.local.forward_depth=depth+1
            if depth==0:
                self.last_model={'identity':str(getattr(model,'identity','unknown')),
                    'hc_policy':getattr(model,'hc_policy','reference'),
                    'qat_policy':getattr(model,'qat_policy','reference'),
                    'pointwise_policy':getattr(model,'pointwise_policy','reference'),
                    'attention_epilogue_policy':getattr(model,'attention_epilogue_policy','reference'),
                    'hc_epilogue_policy':getattr(model,'hc_epilogue_policy','reference')}
            try:
                result=original(model,*args,**kwargs)
                if depth==0:
                    with self.lock:self.completed_forwards+=1
                return result
            except BaseException:
                if depth==0:
                    with self.lock:self.failed_forwards+=1
                raise
            finally:
                self.local.forward_depth=depth
                if depth==0:self.snapshot(write=True)
        model_class._forward_ordinary=wrapped
        self.patches.append((model_class,'_forward_ordinary',original,wrapped))

    def patch_native_vision(self,vision_class,model_class):
        original=vision_class.__call__
        @wraps(original)
        def encode(owner,tokens,cache,images,*args,**kwargs):
            result=original(owner,tokens,cache,images,*args,**kwargs)
            # SourceVision owns its existing mx.eval(h) before successful return.
            # Only inspect already-CPU artifact/grid metadata, never pixel or
            # tensor contents and never add a synchronization/readback.
            with self.lock:
                self.calls['vision.encoder_calls_completed']+=1
                self.calls['vision.images_encoded']+=len(images)
                self.last_native_vision={'source_fingerprint':owner.artifact['identity']['fingerprint'],
                    'target_identity':str(owner.target.identity),'image_count':len(images),
                    'token_rows':int(tokens.shape[1]),
                    'encoder':'mlx2.adapters.deepseek_v41_vision.SourceVision.__call__',
                    'existing_completion':'SourceVision merged embedding mx.eval completed inside original method'}
            return result
        vision_class.__call__=encode;self.patches.append((vision_class,'__call__',original,encode))
        prefill=model_class.prefill_forward
        @wraps(prefill)
        def forward(model,*args,**kwargs):
            images=kwargs.get('image_inputs')
            result=prefill(model,*args,**kwargs)
            if images is not None:
                with self.lock:
                    self.calls['vision.native_image_prefill_returns']+=1
                    if self.last_native_vision is not None:
                        self.last_native_vision['prefill_returned']=True
                        self.last_native_vision['prefill_target_identity']=str(model.identity)
                # This is one completed media model forward, not per helper I/O.
                self.snapshot(write=True)
            return result
        model_class.prefill_forward=forward
        self.patches.append((model_class,'prefill_forward',prefill,forward))

    def patch_verify(self,generator_class):
        original=generator_class._verify
        @wraps(original)
        def wrapped(batch,cohort,blocks,logits,*args,**kwargs):
            decisions=original(batch,cohort,blocks,logits,*args,**kwargs)
            # _verify returns CPU RoundDecision.accepted BEFORE delivery stop/
            # length truncation. Never inspect logits, target laws or tensors.
            for index,decision in enumerate(decisions):
                lengths=getattr(blocks[index],'lengths',None) if blocks[index] is not None else [0]
                module=type(lengths).__module__
                if module not in ('builtins','numpy'):
                    with self.lock:self.verification_metadata_declined+=1
                    continue
                try:
                    value=lengths[0]
                    if type(value) is not int and type(value).__module__!='numpy':raise TypeError('non-host length')
                    proposed=int(value);accepted=decision.accepted
                    if type(accepted)is not int or not 0<=accepted<=proposed:raise ValueError('invalid host decision')
                except (TypeError,ValueError,IndexError):
                    with self.lock:self.verification_metadata_declined+=1
                    continue
                kind=str(getattr(batch,'receipt_kind','unknown'))
                record={'kind':kind,'proposed':proposed,'raw_accepted':accepted,
                    'emitted_after_stop_count':len(decision.emitted),
                    'rejection_observed':proposed>0 and accepted<proposed,
                    'full_acceptance_observed':proposed>0 and accepted==proposed,
                    'source':'ExternalDraftBatchGenerator._verify returned CPU RoundDecision.accepted before delivery truncation'}
                with self.lock:
                    self.calls['external.raw_proposed']+=proposed
                    self.calls['external.raw_accepted']+=accepted
                    self.calls['external.chain_verified_rounds']+=int(proposed>0)
                    self.calls['external.chain_rejection_rounds']+=int(record['rejection_observed'])
                    self.calls['external.chain_full_accept_rounds']+=int(record['full_acceptance_observed'])
                    self.calls['external.chain_positive_accept_rounds']+=int(accepted>0)
                    if len(self.verification_decisions)<512:self.verification_decisions.append(record)
                    else:self.verification_dropped+=1
            return decisions
        generator_class._verify=wrapped
        self.patches.append((generator_class,'_verify',original,wrapped))

    def finalize(self):
        self.finalized=True
        return self.snapshot(write=True)

    def snapshot(self,*,write=False):
        with self.lock:
            if write:self.sequence+=1
            result={'schema':'mlx2.ds41-policy-host-observer.v1','at_unix':time.time(),
                'sequence':self.sequence,'shutdown_snapshot':self.finalized,'calls_after_successful_return':dict(self.calls),
                'completed_outer_forwards':self.completed_forwards,'failed_outer_forwards':self.failed_forwards,
                'synthetic_check_calls_excluded':self.check_calls_excluded,'last_model':self.last_model,
                'native_vision':self.last_native_vision,
                'chain_verification_decisions':list(self.verification_decisions),
                'chain_verification_records_dropped':self.verification_dropped,
                'chain_verification_metadata_declined':self.verification_metadata_declined,
                'observation':'host graph/helper scheduling only; no added synchronization or GPU completion proof',
                'partial_forward_notice':'Successful helper returns from a later failed forward remain scheduling observations; failed_outer_forwards retains that limit',
                'qualification':False}
        if write:
            if self.writer is not None:self.writer(self.path,result)
            else:
                from mlx2.desktop_campaign_worker import atomic
                atomic(self.path,result)
        return result

    def uninstall(self):
        for target,name,original,wrapped in reversed(self.patches):
            if getattr(target,name)is not wrapped:raise RuntimeError('policy observer patch was replaced: '+name)
            setattr(target,name,original)
        self.patches=[]


def patch_modules(observer,model_module,hc,fused,quant):
    # Every helper below unconditionally schedules its custom branch on success;
    # no selected-policy flag or reference fallback increments these counters.
    for name in ('self_check','epilogue_check'):observer.patch(fused,name,'check.'+name,check=True)
    for name in ('split_sinkhorn','weighted_sum','expand','head_reduce'):
        observer.patch(hc,name,'hc.'+name)
    for name in ('rounded','rotate','router','rotate_qat','expand'):
        observer.patch(fused,name,'fused.'+name)
    observer.patch(quant,'fused_quantize','qat.fused_quantize')
    observer.patch_qat(model_module.DeepSeekV41Model)
    observer.patch_forward(model_module.DeepSeekV41Model)
    return observer


def install(unit):
    global _INSTALLED
    unit=Path(unit).resolve()
    if _INSTALLED is not None:
        if _INSTALLED.unit!=unit:raise ValueError('policy observer already belongs to another window')
        return _INSTALLED
    from probe_deepseek_v4_packed import verify_ownership
    verify_ownership(unit,[Path('/Users/Shared/mlxuag/gpu.lock/owner.json'),Path('/tmp/gpu.lock/owner.json')])
    models='mlx2.runtime.models.'
    observer=Observer(unit)
    try:
        patch_modules(observer,*(importlib.import_module(models+name)for name in
            ('deepseek_v41','deepseek_v4_hc','deepseek_v41_fused','deepseek_v41_quant')))
        external=importlib.import_module('mlx2.runtime.external_speculative')
        observer.patch_verify(external.ExternalDraftBatchGenerator)
        vision=importlib.import_module('mlx2.adapters.deepseek_v41_vision')
        model=importlib.import_module(models+'deepseek_v41')
        observer.patch_native_vision(vision.SourceVision,model.DeepSeekV41Model)
    except BaseException:
        observer.uninstall();raise
    _INSTALLED=observer
    atexit.register(observer.finalize)
    observer.snapshot(write=True)
    return observer


def snapshot(*,write=False):
    if _INSTALLED is None:return None
    return _INSTALLED.snapshot(write=write)
