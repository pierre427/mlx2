"""Owned ten-token attribution: evaluated wall boundaries and calling-thread CPU.

The difference between wall and calling-thread CPU is not a GPU timer: it also
includes I/O, driver/worker work and descheduling. Reference throughput controls
are kept separate from the intrusive boundary instrumentation.
"""

from contextlib import contextmanager
from concurrent.futures import Future
import threading
import time

import numpy as np

import profile_deepseek_v41 as core
from mlx2.desktop_campaign_worker import atomic

REQUEST = {"messages": [{"role":"user","content":"Count from one to ten."}], "enable_thinking":False}


class CPUWallProfiler(core.base.BoundaryProfiler):
    def __init__(self,mx):
        super().__init__(mx)
        self.cpu_stack=[]
        self.category_stack=[]
        self.copy_bytes=0

    def call(self,category,name,fn,args,kwargs,*,layer=None):
        started=time.thread_time()
        frame={"children":0.0}
        self.cpu_stack.append(frame)
        self.category_stack.append(category)
        try:
            result=super().call(category,name,fn,args,kwargs,layer=layer)
        finally:
            self.cpu_stack.pop()
            self.category_stack.pop()
        cpu=time.thread_time()-started
        if self.cpu_stack:
            self.cpu_stack[-1]["children"]+=cpu
        event=self.events[-1]
        event["exclusive_calling_thread_cpu_seconds"]=max(0.0,cpu-frame["children"])
        return result

    def summary(self):
        value=super().summary()
        for category,group in value["categories"].items():
            group["exclusive_calling_thread_cpu_seconds"]=sum(e["exclusive_calling_thread_cpu_seconds"] for e in self.events if e["category"]==category)
            group["other_wall_seconds"]=max(0.0,group["exclusive_seconds"]-group["exclusive_calling_thread_cpu_seconds"])
        return value


@contextmanager
def instrument(model,profiler):
    from mlx2.adapters.deepseek_v41_expert_stream import PackedExpertStream

    main_thread=threading.get_ident()
    with core.instrument(model,profiler) as reads:
        result_fn,eval_fn,verify_fn=Future.result,profiler.mx.eval,PackedExpertStream.verify
        in_eval=[False]

        def result(future,*args,**kwargs):
            if threading.get_ident()!=main_thread or not profiler.category_stack:
                return result_fn(future,*args,**kwargs)
            if "dense_page_read" in profiler.category_stack:
                return result_fn(future,*args,**kwargs)
            return profiler.call("read_future_wait","expert_or_engram",lambda:result_fn(future,*args,**kwargs),(),{})

        def evaluate(*args):
            values=core.base.leaves(args,profiler.mx)
            if (not in_eval[0] and profiler.category_stack and profiler.category_stack[-1]=="expert_gather_other"
                    and values and all(v.dtype==profiler.mx.uint8 for v in values)):
                in_eval[0]=True
                try:
                    return profiler.call("expert_upload_eval","bounded_batch",eval_fn,args,{})
                finally:
                    in_eval[0]=False
            return eval_fn(*args)

        def verify(stream):
            return profiler.call("expert_source_verify","fstat/stat",lambda:verify_fn(stream),(),{})

        Future.result,profiler.mx.eval,PackedExpertStream.verify=result,evaluate,verify
        try:
            yield reads
        finally:
            Future.result,profiler.mx.eval,PackedExpertStream.verify=result_fn,eval_fn,verify_fn


def direct(adapter,mx,label):
    model=adapter.model
    pages=getattr(model.weights,"dense_pages",None)
    def dense_counters():
        return dict(pages.stats) if pages is not None else {}
    def dense_delta(before):
        return {key:value-before[key] for key,value in dense_counters().items()}
    begin=time.perf_counter()
    ids=list(adapter.prompt_tokens(REQUEST))
    tokenize=time.perf_counter()-begin
    if len(ids)!=10:
        raise ValueError("ten-token diagnostic prompt changed")
    # Flush only the private expert LRU, retaining resident non-expert weights.
    stream=model.weights.expert_stream
    for cache in stream.caches.values():
        cache.clear()
    mx.clear_cache()
    profiler=CPUWallProfiler(mx)
    before=stream.counters()
    dense_before=dense_counters()
    cache=model.make_cache()
    first,last=mx.array([ids[:9]],mx.uint32),mx.array([ids[9:]],mx.uint32)
    with instrument(model,profiler) as reads:
        begin=time.perf_counter()
        profiler.call("forward_other","prefill9",model.prefill_forward,(first,cache),{})
        boundary=time.perf_counter()
        logits=profiler.call("forward_other","last_prompt_token",model,(last,cache),{})
        end=time.perf_counter()
    cold={"prefill9_seconds":boundary-begin,"last_prompt_token_seconds":end-boundary,
          "wall_seconds":end-begin,"read_tasks":reads,"selected_stack_copy_bytes":profiler.copy_bytes,
          "dense_page_counter_delta":dense_delta(dense_before),
          "counter_delta":{k:v-before[k] for k,v in stream.counters().items()},**profiler.summary()}
    mx.eval(logits)
    if not bool(mx.all(mx.isfinite(logits)).item()):
        raise RuntimeError("nonfinite ten-token logits")
    controls=[]
    for _ in range(3):
        state=model.make_cache()
        start=time.perf_counter();pre=model.prefill_forward(first,state);mx.eval(pre,[s.state for s in state])
        middle=time.perf_counter();value=model(last,state);mx.eval(value,[s.state for s in state])
        controls.append({"prefill9_seconds":middle-start,"last_prompt_token_seconds":time.perf_counter()-middle})
    # Warm the exact full10 shape, then collect a separate boundary profile.
    state=model.make_cache();whole=mx.array([ids],mx.uint32)
    value=model.prefill_forward(whole,state);mx.eval(value,[s.state for s in state])
    reference_value=np.array(value)
    reference_states=[core.arrays_numpy(s.state) for s in state]
    whole_controls=[]
    for _ in range(3):
        state=model.make_cache()
        start=time.perf_counter();value=model.prefill_forward(whole,state);mx.eval(value,[s.state for s in state])
        whole_controls.append(time.perf_counter()-start)
        if not core.arrays_equal((reference_value,reference_states), (np.array(value),[core.arrays_numpy(s.state) for s in state])):
            raise RuntimeError("warm ten-token repetition changed logits or state bytes")
    p=CPUWallProfiler(mx);state=model.make_cache();before=stream.counters()
    dense_before=dense_counters()
    with instrument(model,p) as reads:
        value=p.call("forward_other","whole_prefill10",model.prefill_forward,(whole,state),{})
    warm_prefill={"logits_bytes_equal":core.arrays_equal(reference_value,np.array(value)),
                  "state_bytes_equal":core.arrays_equal(reference_states,[core.arrays_numpy(s.state) for s in state]),
                  "selected_stack_copy_bytes":p.copy_bytes,"read_tasks":reads,
                  "dense_page_counter_delta":dense_delta(dense_before),
                  "counter_delta":{k:v-before[k] for k,v in stream.counters().items()},**p.summary()}
    # Decode a bounded continuation and record every individual target step.
    tokens,steps=[],[]
    for index in range(10):
        start=time.perf_counter();token=int(mx.argmax(value[0,-1]).item());selection=time.perf_counter()-start
        tokens.append(token)
        if token in adapter.tokenizer.eos_token_ids:
            break
        start=time.perf_counter();value=model(mx.array([[token]],mx.uint32),state);mx.eval(value,[s.state for s in state])
        steps.append({"index":index,"selection_seconds":selection,"forward_seconds":time.perf_counter()-start})
    saved=[s.snapshot() for s in state];mx.eval([s.arrays for s in saved])
    selected=int(mx.argmax(value[0,-1]).item())
    p=CPUWallProfiler(mx);before=stream.counters()
    dense_before=dense_counters()
    with instrument(model,p) as reads:
        actual=p.call("forward_other","decode1",model,(mx.array([[selected]],mx.uint32),state),{})
    mx.eval(actual)
    decode={"selected_stack_copy_bytes":p.copy_bytes,"read_tasks":reads,
            "dense_page_counter_delta":dense_delta(dense_before),
            "counter_delta":{k:v-before[k] for k,v in stream.counters().items()},**p.summary()}
    actual_value=np.array(actual)
    actual_states=[core.arrays_numpy(s.state) for s in state]
    for s,snapshot in zip(state,saved):s.restore(snapshot)
    control=model(mx.array([[selected]],mx.uint32),state);mx.eval(control,[s.state for s in state])
    decode["logits_bytes_equal"]=core.arrays_equal(actual_value,np.array(control))
    decode["state_bytes_equal"]=core.arrays_equal(actual_states,[core.arrays_numpy(s.state) for s in state])
    start=time.perf_counter();snapshots=[s.snapshot() for s in state];mx.eval([s.arrays for s in snapshots]);snapshot_seconds=time.perf_counter()-start
    start=time.perf_counter()
    for s,snapshot in zip(state,snapshots):s.restore(snapshot)
    mx.eval([s.state for s in state]);restore_seconds=time.perf_counter()-start
    passed=all(section[k] for section in (warm_prefill,decode) for k in ("logits_bytes_equal","state_bytes_equal"))
    if not passed:raise RuntimeError("ten-token instrumentation changed logits or state bytes")
    return {"passed":passed,"label":label,"prompt":REQUEST,"prompt_ids":ids,"prompt_tokens":len(ids),
            "policy":adapter.kernel_policy,"tokenization_seconds":tokenize,
            "cold_expert_instrumented_split9_plus1":cold,"warm_uninstrumented_split9_plus1":controls,
            "warm_uninstrumented_prefill10":whole_controls,
            "warm_instrumented_prefill10":warm_prefill,"warm_instrumented_decode1":decode,
            "decode_steps":steps,"output_tokens":tokens,"text":adapter.tokenizer.decode(tokens,skip_special_tokens=True),
            "snapshot_seconds":snapshot_seconds,"restore_seconds":restore_seconds,
            "scope":"forced boundaries perturb scheduling; other wall includes GPU/driver/I/O/descheduling, not pure GPU time"}


def http(base,model_id,out,guard):
    from deepseek_v41_context_ladder import http_json,stream

    body={**REQUEST,"model":model_id,"max_tokens":10,"temperature":0,"stream":True,"return_token_trace":True}
    arms=[]
    for i in range(4):
        guard();before=http_json(base,"/v1/status")
        value=stream(base,body,out/f"ten-token-http-{i}.sse")
        after=http_json(base,"/v1/status")
        value.update(before=before,after=after,arm="cold" if i==0 else "warm")
        arms.append(value)
    gates={"prompt10":all(a["receipt"]["prompt_tokens"]==10 for a in arms),
           "cold_hit_zero":arms[0]["receipt"]["cached_tokens"]==0,
           "warm_hit9":all(a["receipt"]["cached_tokens"]==9 for a in arms[1:]),
           "token_parity":all(a["token_ids"]==arms[0]["token_ids"] for a in arms[1:])}
    result={"prompt":REQUEST,"arms":arms,"gates":gates,"passed":all(gates.values())}
    atomic(out/"ten-token-http.json",result)
    return result
