# SPDX-License-Identifier: MIT
"""Synthetic real-shape attention functions; no model/weights or GPU on import.

Parent owns GPU/admission. Provenance: deepseek-v41-synthetic-attention.json.
"""
import statistics
import time
from types import SimpleNamespace

CONTEXTS=(100,128,1024,4096,16384)
RATIOS=(1,2)
HEADS,DIM,WINDOW,TOPK=64,512,128,512
REPEATS=7


def geometry(length,ratio):
    if type(length)is not int or length not in CONTEXTS or type(ratio)is not int or ratio not in RATIOS:
        raise ValueError('Synthetic attention requires actual bounded geometry')
    pool=length//ratio;selected=min(TOPK,pool);local=min(WINDOW,length)
    size=(HEADS*DIM+local*DIM+pool*DIM+selected*DIM+HEADS)*4+selected*5+local
    return dict(frontier_post_token=length,ratio=ratio,query_shape=(1,HEADS,1,DIM),
        local_shape=(1,1,local,DIM),pool_shape=(1,pool,DIM),indices_shape=(1,1,selected),
        local_tokens=local,pool_tokens=pool,selected_tokens=selected,logical_input_bytes=size)


def _timed(mx,functions,check_live,clock=time.perf_counter,reset=None):
    # Compile and all input creation/evaluation happen outside these samples.
    for fn in functions.values():
        if reset is not None:reset()
        mx.eval(fn())
    names=list(functions);samples={name:[]for name in names}
    for repetition in range(REPEATS):
        for name in names[repetition%len(names):]+names[:repetition%len(names)]:
            if check_live is not None:check_live()
            if reset is not None:reset()
            mx.synchronize()
            start=clock();value=functions[name]();submitted=clock();mx.eval(value);done=clock()
            samples[name].append(dict(repetition=repetition,submit_seconds=submitted-start,
                completion_seconds=done-submitted,total_seconds=done-start))
            del value
    return {name:dict(samples=values,median_total_seconds=statistics.median(x['total_seconds']for x in values),
        minimum_total_seconds=min(x['total_seconds']for x in values),maximum_total_seconds=max(x['total_seconds']for x in values),
        median_submit_seconds=statistics.median(x['submit_seconds']for x in values),
        median_completion_seconds=statistics.median(x['completion_seconds']for x in values))for name,values in samples.items()}


def run(mx,check_live=None):
    from mlx2.runtime.models.deepseek_v41 import selected_attention,decode_selected_kv
    from mlx2.runtime.models.deepseek_v41_cache import DeepSeekV41LayerCache
    mx.random.seed(41)
    report=dict(schema='mlx2.synthetic-ds41-attention.v1',seed=41,repeats=REPEATS,
        source_functions=['deepseek_v41.selected_attention','deepseek_v41.decode_selected_kv'],
        shape=dict(query_heads=HEADS,KV_heads=1,head_dim=DIM,SWA=WINDOW,selected_topk=TOPK),
        dtype='float32',contexts_are='Post-one-token frontier',
        scope='Synthetic finite functions only; no model construction/weights/checkpoint/accuracy or quality gates',
        timing='Warm JIT excluded; interleaved variants; graph submission vs eval completion; fence outside each variant',cases=[])
    for length in CONTEXTS:
        for ratio in RATIOS:
            if check_live is not None:check_live()
            g=geometry(length,ratio)
            q=mx.random.normal(g['query_shape'],dtype=mx.float32)*.1
            local=mx.random.normal(g['local_shape'],dtype=mx.float32)*.1
            pool=mx.random.normal(g['pool_shape'],dtype=mx.float32)*.1
            indices=mx.arange(g['pool_tokens']-g['selected_tokens'],g['pool_tokens'],dtype=mx.uint32)[None,None,:]
            valid=mx.ones(g['indices_shape'],dtype=mx.bool_)
            mask=mx.ones((1,1,1,g['local_tokens']),dtype=mx.bool_)
            sinks=mx.zeros((HEADS,),dtype=mx.float32)
            mx.eval(q,local,pool,indices,valid,mask,sinks)
            source=SimpleNamespace(pooled=pool,_pooled=pool,offset=length,_pool_count=g['pool_tokens'])
            gathered=decode_selected_kv(source,indices,valid,{})
            mx.eval(gathered)
            def gather():return decode_selected_kv(source,indices,valid,{})
            def full():return selected_attention(q,local,mask,pool,indices,valid,sinks,gathered=gather())
            def math_only():return selected_attention(q,local,mask,pool,indices,valid,sinks,gathered=gathered)
            def local_only():return selected_attention(q,local,mask,None,None,None,sinks)
            functions=dict(full_selected_fresh_gather=full,KV_data_frozen=full,
                gather_only_fresh_shared=gather,math_only_pre_gathered=math_only,local_only=local_only)
            # Literal with/without local KV storage update, in one completion
            # graph. Pool/index publication and compressor math are excluded.
            state=DeepSeekV41LayerCache(ratio,xp=mx,source=True,identity='synthetic-attention-only')
            old_local=local[:,:,:-1,:]
            new_token=local[:,:,-1:,:]
            def reset():state.keys=old_local;state.offset=length-1
            def with_local_update():
                band=state.update_local(new_token)
                value=selected_attention(q,band,mask,pool,indices,valid,sinks,gathered=gather())
                return value,state.keys
            def without_local_update():return full(),local
            functions.update(local_update_then_attention=with_local_update,
                attention_without_local_update=without_local_update)
            timing=_timed(mx,functions,check_live,reset=reset)
            # Lightweight execution check only, outside timing. No parity test.
            finite={}
            for name,fn in functions.items():
                reset();value=fn();output=value[0]if isinstance(value,tuple)else value
                flag=mx.all(mx.isfinite(output));mx.eval(flag);finite[name]=bool(flag.item());del value,output
            if not all(finite.values()):raise ValueError('Synthetic attention produced nonfinite output')
            report['cases'].append(dict(geometry=g,timing=timing,finite=finite,
                KV_data_frozen_scope='Same immutable synthetic KV/pools as full baseline; no KV writes exist inside either measured attention function',
                local_update_scope='Actual update_local plus attention; retained owned KV copy also evaluated; prebuilt post-update band in no-update arm; cache reset outside timers; excludes pool publication/compression',
                pre_gathered_bytes=int(gathered.nbytes)))
            q=None;local=None;pool=None;indices=None;valid=None;mask=None;sinks=None;source=None;gathered=None;state=None;old_local=None;new_token=None;del functions
            mx.synchronize();mx.clear_cache()
    report.update(complete=True,cases_completed=len(report['cases']),timed_function_calls=len(report['cases'])*7*REPEATS,
        maximum_logical_inputs_bytes=max(c['geometry']['logical_input_bytes']for c in report['cases']))
    return report
