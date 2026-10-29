#!/usr/bin/env python3
"""One owned exact-artifact ordinary diagnostic; not model qualification.

The external caller owns CPG/paired locks, source freezing, thermal envelope
and final process cleanup. This harness refuses absent ownership/source and
partial/unsealed artifacts before constructing the native serving adapter.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from probe_deepseek_v4_packed import atomic, compare, source_receipt, verify_ownership

REQUEST={"messages":[{"role":"user","content":"Reply with exactly Hello."}],"enable_thinking":False}


def validate_args(args):
    if not args.run_gpu:
        raise ValueError('ordinary model execution requires explicit --run-gpu plus live ownership')
    if args.output.exists():
        raise ValueError('refusing existing evidence directory')
    if args.ownership_window is None:
        raise ValueError('ordinary GPU model probe requires ownership window')
    maximum=128 if args.request_json is not None else 32
    if not 1<=args.max_tokens<=maximum or not 1<=args.prefill_step<=2048:
        raise ValueError(f'bounded max_tokens1..{maximum} and prefill_step1..2048 required; custom request required above32')
    if not args.expected_text or len(args.expected_text)>8192:
        raise ValueError('expected-text must be nonempty and at most8192characters')
    if args.functional_law=='nonempty_finite' and args.request_json is None:
        raise ValueError('nonempty_finite oracle requires an explicit custom request')
    if not 80<=args.memory_limit_gib<=112:
        raise ValueError('memory limit must be80..112GiB; host/driver reserve stays external')
    if not isinstance(args.execution_policy,dict) or any(type(k) is not str or type(v) is not str for k,v in args.execution_policy.items()):
        raise ValueError('execution-policy must be a strict JSON dictionary of string choices')
    if type(args.snapshot_token) is not int or args.snapshot_token<0:
        raise ValueError('snapshot token must be nonnegative integer')
    if not 0<=args.capture_output_position<args.max_tokens:
        raise ValueError('capture-output-position must be inside the bounded output budget')


def phase(receipt,output,name,**values):
    receipt['phase']=name
    receipt.setdefault('phases',[]).append({'phase':name,'at_unix':time.time(),**values})
    atomic(output/'results.json',receipt)
    print(json.dumps({'phase':name,**values}),flush=True)


def memory(mx):
    return {'active_bytes':mx.get_active_memory(),'allocator_peak_bytes':mx.get_peak_memory(),
            'cache_bytes':mx.get_cache_memory()}


def evaluated_route_diagnostics(adapter,cache):
    """Capture selected/scheduled paths after the caller evaluated outputs."""
    import importlib
    result={'adapter':adapter.diagnostics(),
            'scope':'captured after forced generation-logit/cache evaluation; scheduled counters are not hardware dispatch traces',
            'cache_storage':{'encoded_state_bytes':sum(int(state.nbytes) for state in cache),
                'layers':[{'index':i,'class':type(state).__name__,'offset':state.offset,
                           'encoded_state_bytes':int(state.nbytes)} for i,state in enumerate(cache)],
                'scope':'materialized persistent cache state component sizes, including packed codes/scales/RoPE tails; excludes snapshots, allocator slack and transient unpack views'}}
    for key,module_name in [('visible_attention','deepseek_v4_visible_attention'),('hyper_connections','deepseek_v4_hc')]:
        module=importlib.import_module('mlx2.runtime.models.'+module_name)
        status=getattr(module,'route_status',None)
        result[key]=status() if status is not None else {'scheduled_counters_available':False}
    return result


def load_request(path):
    """Bind a bounded text-only request; media/tools never enter this harness."""
    if path is None:
        return REQUEST,None
    before=path.stat()
    if before.st_size>64*1024:
        raise ValueError('request JSON exceeds64KiB')
    raw=path.read_bytes(); value=json.loads(raw)
    if (not isinstance(value,dict) or set(value)-{'messages','enable_thinking'}
            or value.get('enable_thinking',False) is not False):
        raise ValueError('custom request permits messages and enable_thinking=false only')
    messages=value.get('messages')
    if not isinstance(messages,list) or not 1<=len(messages)<=64:
        raise ValueError('custom request requires1..64text messages')
    for message in messages:
        if (not isinstance(message,dict) or set(message)!={'role','content'}
                or message['role'] not in ('system','user','assistant')
                or type(message['content']) is not str or not message['content']):
            raise ValueError('custom request messages require role and nonempty text content')
    after=path.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
        raise ValueError('request changed while being bound')
    return {**value,'enable_thinking':False},{'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}


def functional_result(text, *, law, expected, finite):
    passed=bool(finite and (text.strip()==expected if law=='exact' else text.strip()))
    return {'law':law,'expected_text':expected if law=='exact' else None,
            'finite_logits':bool(finite),'passed':passed,
            'scope':'targeted functional diagnostic only; nonempty text is not answer correctness or qualification'}


def first_logit_comparison(actual,reference,position=0):
    actual=np.asarray(actual,np.float32).reshape(-1)
    reference=np.asarray(reference,np.float32).reshape(-1)
    if actual.shape!=reference.shape or not actual.size or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError('reference logits must be one finite matching vocabulary vector')
    result=compare(actual,reference,atol=.005,rtol=.001)
    ai=int(np.argmax(actual)); ri=int(np.argmax(reference))
    at=np.argsort(-actual)[:2]; rt=np.argsort(-reference)[:2]
    result.update(actual_argmax=ai,reference_argmax=ri,argmax_equal=ai==ri,
        actual_top2=[int(x) for x in at],reference_top2=[int(x) for x in rt],
        actual_top2_margin=float(actual[at[0]]-actual[at[1]]),
        reference_top2_margin=float(reference[rt[0]]-reference[rt[1]]),
        actual_selected_vs_reference_selected_margin=float(actual[ai]-actual[ri]),
        reference_selected_vs_actual_selected_margin=float(reference[ri]-reference[ai]),
        output_position=position,
        interpretation='one already-selected generation position only; no token-parity or near-tie-equivalence verdict')
    return result


def capture_selected_position(receipt,output,logits,position,reference,reference_binding):
    """Save exactly one caller-selected row; never collect every token's logits."""
    value=np.asarray(logits,dtype=np.float32)
    if value.ndim!=1 or not value.size or not np.isfinite(value).all():
        raise ValueError('selected-position capture requires one finite vocabulary vector')
    name='first-position-logits.npy' if position==0 else f'output-position-{position}-logits.npy'
    path=output/name;np.save(path,value,allow_pickle=False)
    receipt['captured_logits']={'output_position':position,'path':name,
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'selected_token':int(np.argmax(value)),
        'scope':'one selected position only; later-position diagnostic capture perturbs that replay timing'}
    if reference is not None:
        receipt['reference_logits']={**reference_binding,'output_position':position,
            'comparison':first_logit_comparison(value,reference,position)}


def load_reference(path, vocab):
    """Bound a one-position NPY before model upload; never unpickle/expand a tensor dump."""
    before=path.stat()
    if before.st_size>vocab*8+4096 or before.st_size<64:
        raise ValueError("reference file must contain only one bounded vocabulary vector")
    data=np.load(path,allow_pickle=False,mmap_mode='r')
    if not isinstance(data,np.ndarray) or data.shape not in ((vocab,),(1,vocab)) or data.dtype.kind!='f' or data.dtype.itemsize>8:
        raise ValueError("reference NPY must be one floating vocabulary vector")
    value=np.array(data,dtype=np.float32).reshape(-1)
    if not np.isfinite(value).all():
        raise ValueError("reference logits must be finite")
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    after=path.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
        raise ValueError("reference logits changed while being bound")
    return value,{'path':str(path),'sha256':digest,'shape':[vocab]}


def arrays_numpy(value):
    if isinstance(value,(tuple,list)):
        return tuple(arrays_numpy(x) for x in value)
    return None if value is None else np.array(value)


def arrays_equal(a,b):
    if isinstance(a,(tuple,list)):
        return isinstance(b,(tuple,list)) and len(a)==len(b) and all(arrays_equal(x,y) for x,y in zip(a,b))
    if a is None or b is None:
        return a is None and b is None
    return np.array_equal(a,b)


def snapshot_check(model,cache,fixed_token,mx):
    snapshots=[state.snapshot() for state in cache]
    mx.eval([s.arrays for s in snapshots])
    offsets=[state.offset for state in cache]
    token=mx.array([[fixed_token]],dtype=mx.uint32)
    before=time.perf_counter()
    a=model(token,cache=cache,logits_to_keep=0)
    mx.eval(a,[state.state for state in cache])
    av=np.array(a); astates=[arrays_numpy(state.state) for state in cache]
    ameta=[state.meta_state for state in cache]
    for state,snapshot in zip(cache,snapshots): state.restore(snapshot)
    mx.eval([state.state for state in cache])
    b=model(token,cache=cache,logits_to_keep=0)
    mx.eval(b,[state.state for state in cache])
    bv=np.array(b); bstates=[arrays_numpy(state.state) for state in cache]
    bmeta=[state.meta_state for state in cache]
    # Leave the original prompt boundary intact for the functional decode.
    for state,snapshot in zip(cache,snapshots): state.restore(snapshot)
    mx.eval([state.state for state in cache])
    return {'fixed_next_token':fixed_token,'prompt_frontier_offsets':offsets,
            'logits_bitwise_equal':bool(np.array_equal(av,bv)),
            'state_arrays_bitwise_equal':arrays_equal(astates,bstates),
            'state_metadata_equal':ameta==bmeta,
            'restored_original_frontier':all(state.offset==offset for state,offset in zip(cache,offsets)),
            'numerical':compare(av,bv,atol=0,rtol=0),
            'seconds':time.perf_counter()-before,
            'scope':'direct native cache snapshot/restore branch diagnostic; no APC publication/qualification claim'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run-gpu',action='store_true')
    parser.add_argument('--ownership-window',type=Path)
    parser.add_argument('--owner-path',type=Path,action='append')
    parser.add_argument('--max-tokens',type=int,default=16)
    parser.add_argument('--prefill-step',type=int,default=128)
    parser.add_argument('--memory-limit-gib',type=float,default=108)
    parser.add_argument('--snapshot-token',type=int,default=42)
    parser.add_argument('--reference-logits',type=Path)
    parser.add_argument('--execution-policy',type=json.loads,default={})
    parser.add_argument('--request-json',type=Path)
    parser.add_argument('--expected-text',default='Hello')
    parser.add_argument('--functional-law',choices=('exact','nonempty_finite'),default='exact')
    parser.add_argument('--capture-output-position',type=int,default=0)
    args=parser.parse_args(argv)
    try:
        validate_args(args)
        request,request_binding=load_request(args.request_json)
    except ValueError as error: parser.error(str(error))
    owner_paths=args.owner_path or [Path('/Users/Shared/mlxuag/gpu.lock/owner.json'),Path('/tmp/gpu.lock/owner.json')]
    if len(owner_paths)!=2: parser.error('exactly two paired owner receipt paths required')
    def guard(): return verify_ownership(args.ownership_window,owner_paths)
    ownership=guard()
    source=source_receipt()
    # CPU artifact seal/header validation occurs before any weight upload.
    from mlx2.adapters.deepseek_v4_gguf_artifact import inspect_artifact
    artifact=inspect_artifact(args.model_path)
    reference=reference_binding=None
    if args.reference_logits:
        reference,reference_binding=load_reference(args.reference_logits,int(artifact['metadata']['deepseek4.vocab_size']))
    args.output.mkdir(parents=True)
    checkpoint=args.model_path.name+'@'+artifact['identity']['fingerprint']
    receipt={'schema':'mlx2.deepseek-v4-ordinary-targeted-probe.v1','status':'running',
        'qualification':False,'performance_claim':False,'source':source,'checkpoint':checkpoint,
        'artifact':artifact['identity'],'ownership':ownership,'request':request,'request_binding':request_binding,
        'controls':{'physical_width':1,'thinking':False,'sampling':'greedy','max_output_tokens':args.max_tokens,
                    'prefill_step':args.prefill_step,'requested_execution_policy':args.execution_policy,
                    'memory_limit_gib':args.memory_limit_gib},
        'timing_scope':'observational single ordinary functional diagnostic; not thermally replicated performance',
        'phases':[]}
    phase(receipt,args.output,'artifact_verified_before_load')
    adapter=model=cache=logits=None
    try:
        import mlx.core as mx
        from mlx2.process_env import require_process_numerics
        from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows
        require_process_numerics('DeepSeek V4 ordinary targeted probe')
        mx.set_default_device(mx.gpu); mx.set_memory_limit(int(args.memory_limit_gib*(1<<30))); mx.set_cache_limit(256<<20)
        receipt['device_info']=mx.metal.device_info()
        guard(); start=time.perf_counter()
        from mlx2.adapters.deepseek_v4_gguf import DeepSeekV4GGUFAdapter
        adapter=DeepSeekV4GGUFAdapter(str(args.model_path),execution_policy=args.execution_policy); model=adapter.model
        if adapter.identity!=artifact['identity']:
            raise RuntimeError('artifact identity changed during adapter load')
        receipt['load_seconds']=time.perf_counter()-start
        receipt['route']=adapter.diagnostics(); receipt['tokenizer']=adapter.tokenizer_receipt
        receipt['controls'].update(execution_policy=dict(adapter.kernel_policy),
            cache_layout=adapter.layout,attention_policy=model.attention_policy,
            kv_storage=adapter.kernel_policy['deepseek_kv'],hc_policy=model.hc_policy)
        from deepseek_v4_probe_residency import wire_preloaded_weights
        wire_preloaded_weights(mx, receipt)
        phase(receipt,args.output,'loaded',seconds=receipt['load_seconds'],memory=memory(mx))
        raw=adapter.render_prompt(request); tokens=list(adapter.prompt_tokens(request))
        if not tokens or len(tokens)>2048 or not 0<=args.snapshot_token<model.args.vocab_size:
            raise ValueError('bounded prompt/snapshot-token contract failed')
        (args.output/'prompt.txt').write_text(raw,encoding='utf-8')
        atomic(args.output/'prompt-tokens.json',{'tokens':tokens,'request':request,
            'rendered_utf8_sha256':hashlib.sha256(raw.encode()).hexdigest(),
            'token_ids_sha256':hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()})
        receipt['prompt']={'tokens':len(tokens),'rendered_utf8_sha256':hashlib.sha256(raw.encode()).hexdigest(),
                           'raw_path':'prompt.txt','tokens_path':'prompt-tokens.json',
                           'prefill_law':'scoped chunks over all-but-last prompt token, then ordinary single-token tail'}
        cache=model.make_cache(); chunks=[]; start=time.perf_counter()
        for offset in range(0,len(tokens)-1,args.prefill_step):
            guard(); part=tokens[offset:min(offset+args.prefill_step,len(tokens)-1)]
            before=time.perf_counter()
            with prefill_checkpoint_rows([len(part)],width=len(part)) as selected:
                checkpoint_logits=model.prefill_forward(mx.array([part],dtype=mx.uint32),cache=cache)
            mx.eval(checkpoint_logits,[state.state for state in cache])
            for state in cache: state.finalize()
            if selected.applied!=1 or not bool(mx.all(mx.isfinite(checkpoint_logits)).item()):
                raise RuntimeError('prefill checkpoint projection/finite-state contract failed')
            chunks.append({'start':offset,'length':len(part),'seconds':time.perf_counter()-before,
                           'checkpoint_rows_applied':selected.applied,'memory':memory(mx)})
            phase(receipt,args.output,'prefill_chunk_complete',chunk=chunks[-1])
        tail_start=time.perf_counter()
        logits=model(mx.array([[tokens[-1]]],dtype=mx.uint32),cache=cache,logits_to_keep=0)
        mx.eval(logits,[state.state for state in cache])
        tail_seconds=time.perf_counter()-tail_start
        prefill_seconds=time.perf_counter()-start
        first=np.array(logits[0,-1],dtype=np.float32)
        if first.shape!=(model.args.vocab_size,) or not np.isfinite(first).all():
            raise RuntimeError('first-position vocabulary logits are nonfinite or malformed')
        if args.capture_output_position==0:
            capture_selected_position(receipt,args.output,first,0,reference,reference_binding)
        forward_seconds=sum(chunk['seconds'] for chunk in chunks)+tail_seconds
        receipt['prefill']={'prompt_tokens':len(tokens),'seconds':prefill_seconds,
                            'observed_tokens_per_second':len(tokens)/prefill_seconds,'chunks':chunks,
                            'single_token_tail_seconds':tail_seconds,
                            'summed_forward_seconds':forward_seconds,
                            'summed_forward_tokens_per_second':len(tokens)/forward_seconds,
                            'summed_forward_scope':'prompt forwards/evaluation/finalize; excludes ownership checks and atomic receipt writes; observational, not controlled performance'}
        phase(receipt,args.output,'prefill_complete_first_position_saved' if args.capture_output_position==0 else 'prefill_complete_capture_position_pending',prefill=receipt['prefill'],memory=memory(mx))
        guard(); receipt['snapshot_restore']=snapshot_check(model,cache,args.snapshot_token,mx)
        phase(receipt,args.output,'snapshot_restore_checked',result=receipt['snapshot_restore'],memory=memory(mx))
        generated=[int(np.argmax(first))]; step_times=[]; eos=set(adapter.tokenizer.eos_token_ids)
        start=time.perf_counter()
        for _ in range(1,args.max_tokens):
            if generated[-1] in eos: break
            guard(); before=time.perf_counter()
            logits=model(mx.array([[generated[-1]]],dtype=mx.uint32),cache=cache,logits_to_keep=0)
            mx.eval(logits,[state.state for state in cache])
            if not bool(mx.all(mx.isfinite(logits)).item()): raise RuntimeError('decode logits nonfinite')
            if len(generated)==args.capture_output_position:
                capture_selected_position(receipt,args.output,np.array(logits[0,-1]),len(generated),reference,reference_binding)
            generated.append(int(mx.argmax(logits[0,-1]).item()))
            step_times.append(time.perf_counter()-before)
        seconds=time.perf_counter()-start
        text=adapter.tokenizer.decode(generated,skip_special_tokens=True)
        if 'captured_logits' not in receipt:
            raise RuntimeError('selected capture position was not reached before EOS; original outcome preserved')
        receipt['generation']={'token_ids':generated,'text':text,
            'finish_reason':'eos' if generated[-1] in eos else 'length',
            'output_tokens':len(generated),'timed_decode_forwards':len(step_times),
            'first_token_source':'prefill first-position logits; excluded from timed decode rate',
            'seconds':seconds,'forward_seconds':step_times,
            'median_forward_seconds':statistics.median(step_times) if step_times else None,
            'observed_tokens_per_second':len(step_times)/seconds if step_times else None,
            'functional_exact_Hello':text.strip()=='Hello'}
        receipt['generation']['functional']=functional_result(text,law=args.functional_law,
            expected=args.expected_text,finite=True)
        receipt['route_after_evaluated_generation']=evaluated_route_diagnostics(adapter,cache)
        phase(receipt,args.output,'generation_complete',generation=receipt['generation'],memory=memory(mx))
        numeric=receipt['snapshot_restore']
        receipt['passed']=all(numeric[k] for k in ('logits_bitwise_equal','state_arrays_bitwise_equal','state_metadata_equal','restored_original_frontier')) and receipt['generation']['functional']['passed']
        receipt['status']='completed'
        guard()
    except BaseException as error:
        receipt.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        model=cache=logits=None
        if adapter is not None:
            try:
                adapter.close()
            except Exception as cleanup_error:
                receipt.update(status="failed",passed=False,
                    cleanup_error={"type":type(cleanup_error).__name__,"message":str(cleanup_error)})
            adapter=None
        gc.collect()
        try:
            mx.clear_cache(); receipt['cleanup_memory']=memory(mx)
        except (NameError,RuntimeError): pass
        try:
            from deepseek_v4_probe_residency import restore_probe_wired_limit
            restore_probe_wired_limit(mx, receipt)
        except Exception as cleanup_error:
            receipt.update(status='failed',passed=False,
                residency_cleanup_error={'type':type(cleanup_error).__name__,'message':str(cleanup_error)})
        atomic(args.output/'results.json',receipt)
    return 0 if receipt['passed'] else 1


if __name__=='__main__': raise SystemExit(main())
