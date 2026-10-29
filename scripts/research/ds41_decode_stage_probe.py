# SPDX-License-Identifier: MIT
"""Private fixed-frame decode timing cuts; not a serving/state operation.

Caller supplies a private cache clone and owns token-end completion/timing.
Capture copies are evaluated OUTSIDE measured baseline/bypass repetitions.
Skipping all attention freezes every layer's KV frontier. Never publish these
counterfactual caches through APCv2. No tensor math changes or GPU work on import.
Provenance: deepseek-v41-decode-stage-probe.json.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import threading
import time
import weakref

MODES=('capture','baseline','bypass_attention','bypass_moe','bypass_attention_moe','stage_timed',
       'bypass_hyperconnections','bypass_dense')
_LOCK=threading.Lock()


def _leaves(value,mx):
    if isinstance(value,mx.array):return[value]
    if isinstance(value,(list,tuple)):return[a for v in value for a in _leaves(v,mx)]
    if isinstance(value,dict):return[a for v in value.values()for a in _leaves(v,mx)]
    return[]


def _copy(value,mx):
    if isinstance(value,mx.array):return mx.asarray(value,copy=True)
    if value is None or type(value)in(bool,int,float,complex,str,bytes):return value
    if isinstance(value,tuple):return tuple(_copy(v,mx)for v in value)
    if isinstance(value,list):return[_copy(v,mx)for v in value]
    if isinstance(value,dict):return{k:_copy(v,mx)for k,v in value.items()}
    raise ValueError('Decode probe bank accepts tensor/tuple outputs only')


class ProbeBank:
    def __init__(self,max_bytes=128<<20):
        if type(max_bytes)is not int or not 0<max_bytes<=512<<20:raise ValueError('Decode bank byte bound')
        self.max_bytes=max_bytes;self.bytes=0;self.records={};self._model=None;self.identity=None
    def bind(self,model,frontier):
        identity=(getattr(model,'identity',None),int(frontier))
        if self._model is None:self._model=weakref.ref(model);self.identity=identity
        elif self._model()is not model or self.identity!=identity:raise ValueError('Decode bank model/frontier changed')
    def store(self,key,value,mx,eval_fn):
        if key in self.records:raise ValueError('Duplicate decode bank output')
        values=_leaves(value,mx);size=sum(int(v.nbytes)for v in values)
        if not values or self.bytes+size>self.max_bytes:raise MemoryError('Decode output bank ceiling')
        copied=_copy(value,mx);eval_fn(*_leaves(copied,mx))
        self.records[key]=copied;self.bytes+=size
    def get(self,key):
        if key not in self.records:raise ValueError('Missing captured decode module output: '+str(key))
        return self.records[key]
    def clear(self):self.records.clear();self.bytes=0;self._model=None;self.identity=None


class _Profiler:
    def __init__(self,mx,eval_fn,clock=time.perf_counter):
        self.mx,self.evaluate,self.clock=mx,eval_fn,clock;self.stack=[];self.events=[]
    def call(self,category,name,fn,args,kwargs,layer):
        start=self.clock();inputs=_leaves((args,kwargs),self.mx)
        if inputs:self.evaluate(*inputs)
        ready=self.clock()
        frame={'children':0.};self.stack.append(frame)
        try:
            out=fn(*args,**kwargs);outputs=_leaves(out,self.mx)
            if outputs:self.evaluate(*outputs)
            end=self.clock()
        finally:self.stack.pop()
        total=end-start
        if self.stack:self.stack[-1]['children']+=total
        self.events.append(dict(category=category,name=name,layer=layer,inclusive_seconds=total,
            input_barrier_seconds=ready-start,exclusive_seconds=max(0.,total-frame['children'])))
        return out


@contextmanager
def instrument(model,unit,label,index,*,input_rows,frontier,mode='capture',bank=None):
    """Capture/bypass fixed actual M1 outputs; caller handles private frame reset.

    baseline/bypass modes perform NO per-stage eval/readbacks. stage_timed is a
    separate heavily synchronized intervention, not ordinary token latency.
    Root's runner owns fixed routing, cache clones, completion and no publication.
    """
    if type(input_rows)is not int or input_rows!=1 or type(frontier)is not int or frontier<1 or mode not in MODES:
        raise ValueError('Decode probe requires explicit fixed width-one frame/mode')
    if bank is None and mode!='stage_timed':raise ValueError('Decode probe requires explicit output bank')
    if not _LOCK.acquire(blocking=False):raise RuntimeError('Nested/concurrent decode stage probe')
    patches=[];layer=[None];counts={};owner=threading.get_ident()
    try:
        import mlx.core as mx
        from mlx2.runtime.models import deepseek_v41 as model_module
        if bank is not None:bank.bind(model,frontier)
        evaluate=mx.eval;p=_Profiler(mx,evaluate)
        report=dict(schema='mlx2.ds41-decode-stage-probe.v1',label=label,index=index,frontier=frontier,
            input_rows=1,mode=mode,approximate_diagnostic=mode.startswith('bypass'),rate_eligible=False,
            APCv2_publication_allowed=False,quality_assessed=False,per_stage_synchronization=mode=='stage_timed',
            capture_copy_seconds=0.,bypassed={},events=p.events,bank_bytes=0,completed=False)
        def patch(obj,name,fn):
            local=name in vars(obj);before=getattr(obj,name);patches.append((obj,name,local,before));setattr(obj,name,fn);return before
        def observe(kind,name,fn,args,kwargs):
            if threading.get_ident()!=owner:raise RuntimeError('Concurrent decode probe call')
            key0=(kind,layer[0],name);ordinal=counts.get(key0,0);counts[key0]=ordinal+1
            key=(*key0,ordinal)
            cut=(kind=='attention'and mode in('bypass_attention','bypass_attention_moe')
                or kind=='moe'and mode in('bypass_moe','bypass_attention_moe')
                or kind=='hyperconnections'and mode=='bypass_hyperconnections'
                or kind=='dense'and mode=='bypass_dense')
            if cut:
                report['bypassed'][kind]=report['bypassed'].get(kind,0)+1
                return bank.get(key)
            out=p.call(kind,name,fn,args,kwargs,layer[0])if mode=='stage_timed'else fn(*args,**kwargs)
            if mode=='capture'and kind in('attention','moe','hyperconnections','dense'):
                start=time.perf_counter();bank.store(key,out,mx,evaluate);report['capture_copy_seconds']+=time.perf_counter()-start
            return out
        # A block scope supplies stable layer keys even when nested modules cut.
        block_type=type(model.layers[0]);before_block=block_type.__call__
        def block(obj,*args,**kwargs):
            if getattr(obj,'model',None)is not model:return before_block(obj,*args,**kwargs)
            previous=layer[0];layer[0]=obj.layer
            try:return before_block(obj,*args,**kwargs)
            finally:layer[0]=previous
        patch(block_type,'__call__',block)
        attention_type=type(model.layers[0].attn);before_attention=attention_type.__call__
        def attention(obj,*args,**kwargs):
            if getattr(obj,'model',None)is not model:return before_attention(obj,*args,**kwargs)
            return observe('attention','Attention',lambda *a,**k:before_attention(obj,*a,**k),args,kwargs)
        patch(attention_type,'__call__',attention)
        weights=model.weights;before_moe=weights.moe
        patch(weights,'moe',lambda *a,**k:observe('moe','routed_moe',before_moe,a,k))
        for name,kind in [('mixes','hyperconnections'),('finish','hyperconnections'),('shared_expert','shared_expert'),('router','routing')]:
            before=getattr(block_type,name)
            def method(obj,*a,_fn=before,_name=name,_kind=kind,**k):
                if getattr(obj,'model',None)is not model:return _fn(obj,*a,**k)
                return observe(_kind,_name,lambda *args,**kwargs:_fn(obj,*args,**kwargs),a,k)
            patch(block_type,name,method)
        for name,kind in [('norm','normalization'),('qat','QAT'),('collapse','hyperconnections'),('expand','hyperconnections')]:
            before=getattr(model,name)
            patch(model,name,lambda *a,_fn=before,_kind=kind,_name=name,**k:observe(_kind,_name,_fn,a,k))
        before_linear=weights.linear
        def linear(name):
            fn=before_linear(name)
            return lambda *a,**k:observe('dense',name,fn,a,k)
        patch(weights,'linear',linear)
        for name in ('grouped','paired_project'):
            if hasattr(weights,name):
                before=getattr(weights,name)
                patch(weights,name,lambda *a,_fn=before,_name=name,**k:observe('dense',_name,_fn,a,k))
        # Extra stage labels are timing-only and never affect bypass arms.
        if mode=='stage_timed':
            for name,kind in [('selected_attention','attention_core'),('decode_selected_kv','selected_KV_gather')]:
                before=getattr(model_module,name)
                patch(model_module,name,lambda *a,_fn=before,_kind=kind,_name=name,**k:observe(_kind,_name,_fn,a,k))
            from mlx2.runtime.models.deepseek_v41_cache import DeepSeekV41LayerCache
            for name in ('update_local','append_pools'):
                before=getattr(DeepSeekV41LayerCache,name)
                patch(DeepSeekV41LayerCache,name,lambda obj,*a,_fn=before,_name=name,**k:observe('KV_update',_name,lambda *x,**kw:_fn(obj,*x,**kw),a,k))
            for name in ('publish','select'):
                before=getattr(attention_type,name)
                patch(attention_type,name,lambda obj,*a,_fn=before,_name=name,**k:observe('KV_compression'if _name=='publish'else'index_selection',_name,lambda *x,**kw:_fn(obj,*x,**kw),a,k))
            import importlib
            for module_name,names,kind in (
                ('mlx2.runtime.models.deepseek_v41_quant',('quantize','fused_quantize'),'QAT'),
                ('mlx2.runtime.gguf.source_gate_up_fusion',('middle','batch_middle'),'routed_GU_middle_fused'),
                ('mlx2.runtime.gguf.source_middle_fusion',('middle',),'routed_middle'),
                ('mlx2.runtime.gguf.source_down_fusion',('down_combine',),'routed_DOWN_assembly_fused'),
                ('mlx2.adapters.deepseek_v41_source_math',('ordered_combine',),'routed_assembly')):
                module=importlib.import_module(module_name)
                for name in names:
                    if hasattr(module,name):
                        before=getattr(module,name)
                        patch(module,name,lambda *a,_fn=before,_name=name,_kind=kind,**k:observe(_kind,_name,_fn,a,k))
            for name in ('eval','synchronize'):
                if hasattr(mx,name):
                    before=getattr(mx,name)
                    patch(mx,name,lambda *a,_fn=before,_name=name,**k:observe('explicit_completion',_name,_fn,a,k))
        yield report
        report.update(completed=True,bank_bytes=0 if bank is None else bank.bytes,
            call_counts={str(k):v for k,v in counts.items()})
    finally:
        try:
            for obj,name,local,before in reversed(patches):
                if local:setattr(obj,name,before)
                else:delattr(obj,name)
        finally:_LOCK.release()
