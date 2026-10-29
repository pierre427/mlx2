# SPDX-License-Identifier: MIT
"""Owned callable PP99/128 functional gate; import does not use the GPU.

The parent GPU unit calls run(adapter, unit_path, live_guard, prompt_ids). This
uses one adapter, an ordinary reference and fresh private caches. Expert-bank
construction and all cold reads stay outside the resident PP timer. It never
publishes APCv2 state and never changes serving defaults.
"""
from pathlib import Path
import hashlib
import json
import os
import time

CHECKPOINT='DeepSeek-V4.1-Flash-1017-Source-Hot60GB-Q1g64@927325be8884e00a7bf60b9aa7d540b167a5efec569b9c7c30b1d34c2793d000'
PLANES=('keys','buf_kv','buf_gate','_pooled','_index_keys','hash_tail')


def plan(rows=99):
    if type(rows)is not int or rows not in (99,128):raise ValueError('PP model gate supports rows99 or128')
    return dict(schema='mlx2.ds41-resident-prefill-model-gate.plan.v1',checkpoint=CHECKPOINT,
        rows=rows,expert_budget_bytes=75_000_000_000,max_bank_bytes=3<<30,
        bank_construction='incremental',candidate='expert_major_wide_v1',
        exact_last_logits=True,exact_every_cache_plane=True,exact_cache_meta=True,
        continuation='one ordinary continuation after bank close, outside PP timing',
        APCv2_publication=False,qualified=False,controlled_performance_claim=False)


def _delta(after,before):
    return {k:v-before.get(k,0)for k,v in after.items()if type(v)in(int,float)}


def _checkpoint(adapter):
    """Compare base/provider and effective derivative identities in their roles."""
    from mlx2.adapters.deepseek_v41_identity import effective_source_identity
    weights=adapter.weights;store=weights.expert_stream
    base=weights.artifact['identity']
    if store.binding!=base['fingerprint']:
        raise ValueError('PP gate source provider differs from actual base artifact')
    if weights.verify_source()is not True:
        raise ValueError('PP gate actual source/stat verification did not pass')
    expected=effective_source_identity(base,q1_bank=store.q1_binding,hot_map=store.hot_binding)
    for name in ('fingerprint','base_fingerprint','derivation'):
        if adapter.identity.get(name)!=expected.get(name):
            raise ValueError('PP gate effective derivative identity differs: '+name)
    observed=CHECKPOINT.split('@')[0]+'@'+expected['fingerprint']
    if observed!=CHECKPOINT:raise ValueError('PP gate exact hybrid checkpoint differs')
    return observed


def _counters(adapter):
    w=adapter.weights
    result={'expert_bytes_read':w.expert_stream.stats.get('bytes_read',0),
            'expert_uploads':w.expert_stream.stats.get('uploaded_experts',0),
            'host_route_materializations':getattr(w,'source_route_host_materializations',0),
            'engram_bytes_read':sum(t.stats.get('bytes_read',0)for t in adapter.engram_tables.values())}
    session=getattr(w,'source_resident_bank_session',None)
    if session is not None:result['ordinary_replays']=session.stats['ordinary_replays']
    return result


def _array_compare(left,right,mx):
    import numpy as np
    if left is None or right is None:return dict(equal=left is right,none=True)
    shape_equal=tuple(left.shape)==tuple(right.shape);dtype_equal=left.dtype==right.dtype
    if not shape_equal or not dtype_equal:
        return dict(equal=False,left_shape=list(left.shape),right_shape=list(right.shape),
                    left_dtype=str(left.dtype),right_dtype=str(right.dtype))
    # Host transfer occurs after the end barrier and outside every timer. Viewing
    # raw bytes also handles bfloat16 caches without a NumPy dtype conversion.
    a=np.asarray(left.view(mx.uint8)).tobytes();b=np.asarray(right.view(mx.uint8)).tobytes()
    return dict(equal=a==b,shape=list(left.shape),dtype=str(left.dtype),bytes=len(a),
        left_sha256=hashlib.sha256(a).hexdigest(),right_sha256=hashlib.sha256(b).hexdigest())


def _cache_compare(left,right,mx):
    if len(left)!=len(right):return dict(equal=False,layers=[])
    layers=[]
    for index,(a,b)in enumerate(zip(left,right)):
        planes={name:_array_compare(getattr(a,name,None),getattr(b,name,None),mx)for name in PLANES}
        metadata_equal=tuple(a.meta_state)==tuple(b.meta_state)
        capacity_equal=all(getattr(a,name,None)==getattr(b,name,None)for name in ('_capacity','_pool_count'))
        layers.append(dict(layer=index,metadata_equal=metadata_equal,capacity_equal=capacity_equal,
            metadata=list(a.meta_state),planes=planes,
            equal=metadata_equal and capacity_equal and all(v['equal']for v in planes.values())))
    return dict(equal=all(row['equal']for row in layers),layers=layers)


def _write(path,value):
    temporary=path.with_suffix('.json.tmp')
    with temporary.open('x')as output:
        json.dump(value,output,indent=2);output.write('\n');output.flush();os.fsync(output.fileno())
    os.replace(temporary,path)


def _park_experts(store,live,*,idlecheck=None):
    if idlecheck is None:
        from mlx2.adapters.deepseek_v41_source_bank import _idle
        idlecheck=_idle
    idlecheck(store);store.park();live()
    if store._cache:raise RuntimeError('Expert-only park left old coefficient keys')
    # NativeExpertStore deliberately refuses acquisitions while parked. Resume
    # only this store; dense weights and Engram tables were never parked.
    store.activate();live()


def _swap_probe():
    """Outside-timer counters; parent owns thermal/process sampling and lease."""
    import subprocess
    import re
    try:
        vm=subprocess.check_output(['vm_stat'],text=True)
        swap=subprocess.check_output(['sysctl','vm.swapusage'],text=True)
        counts={name:int(re.search(r'^'+name+r':\s*(\d+)',vm,re.M).group(1))for name in ('Swapins','Swapouts')}
        counts['raw_swapusage']=swap.strip()
        return counts
    except (OSError,ValueError,AttributeError,subprocess.SubprocessError)as error:
        return {'unavailable':str(error)}


def _warm_order(rounds):
    if type(rounds)is not int or not 0<=rounds<=3:raise ValueError('Warm PP rounds must be0..3')
    return [(variant,arm)for _ in range(rounds)for variant in ('padded_v1','persistent_8_2_v1')
            for arm in ('ordinary',variant,variant,'ordinary')]


def _warm_gates(delta,before,after,*,ordinary):
    io=all(delta.get(k,0)==0 for k in ('expert_bytes_read','expert_uploads','engram_bytes_read','ordinary_replays'))
    route=ordinary or delta.get('host_route_materializations',0)==0
    swap=(all(type(v.get(k))is int for v in (before,after)for k in ('Swapins','Swapouts'))
          and all(before[k]==after[k]for k in ('Swapins','Swapouts'))
          and before.get('raw_swapusage')==after.get('raw_swapusage'))
    thermal=all(v.get('thermal_ok')is True for v in (before,after))
    foreign=all(v.get('foreign_model_processes')==[]for v in (before,after))
    return dict(zero_expert_and_engram_io=io,required_host_route_contract=route,swap_stable=swap,
        parent_thermal_ok=thermal,parent_no_foreign_models=foreign,rate_eligible=False,
        performance_ineligible_reasons=['exploratory_functional_comparison']+
            ([]if swap else['swap_unknown_or_changed'])+
            ([]if thermal else['thermal_evidence_deferred_to_parent_unit'])+
            ([]if foreign else['process_evidence_deferred_to_parent_unit']))


def _warm_resident(adapter,session,inputs,reference,reference_cache,live,record,*,rounds,environment_probe=None):
    """ABBA comparisons over one frozen packed coefficient set, never disk."""
    import statistics
    import mlx.core as mx
    w=adapter.weights;store=w.expert_stream;model=adapter.model;bank=session.bank
    policies=(w.source_resident_gather_policy,w.source_resident_prefill_policy)
    sample=environment_probe or _swap_probe;acquire=store.acquire;keys=bank.keys
    originals={key:id(store._cache[key])for key in keys}
    def closed_acquire(prefix,expert):
        key=(int(prefix.split('.')[1]),int(expert))
        if key not in keys or id(store._cache.get(key))!=originals[key]:
            raise RuntimeError('Warm PP reference left its frozen resident set before acquisition')
        return acquire(prefix,expert)
    store.acquire=closed_acquire
    record.update(warmups=[],samples=[],rounds=rounds,order=[list(v)for v in _warm_order(rounds)],
        coefficient_set_frozen=True,coefficient_precision='same original MXFP4g32 and Q1g64 bytes',
        APCv2_publication=False,controlled_performance_claim=False,rate_eligible=False)
    def one(arm):
        live();bank.assert_current()
        ordinary=arm=='ordinary'
        if ordinary:
            w.source_resident_bank_session=None
            w.source_resident_gather_policy=w.source_resident_prefill_policy='reference'
        else:
            w.source_resident_bank_session=session
            w.source_resident_gather_policy,w.source_resident_prefill_policy=policies
            session.set_prefill_task_policy(arm)
        cache=model.make_cache();drained=False;out=None
        try:
            # Cache creation, environment sampling, control selection and all
            # raw comparisons are outside the end-barrier forward timer.
            env_before=sample();before=_counters(adapter);live();start=time.perf_counter()
            out=model(inputs,cache,logits_to_keep=1);mx.eval(out,[s.state for s in cache])
            seconds=time.perf_counter()-start;drained=True;live()
            delta=_delta(_counters(adapter),before);env_after=sample();bank.assert_current()
            same_set=(set(store._cache)==set(keys)and all(id(store._cache[k])==v for k,v in originals.items()))
            gates=_warm_gates(delta,env_before,env_after,ordinary=ordinary)
            row=dict(arm=arm,seconds=seconds,counter_delta=delta,environment_before=env_before,environment_after=env_after,
                gates=gates,resident_set_unchanged=same_set,raw_last_logits=_array_compare(reference,out,mx),
                cache=_cache_compare(reference_cache,cache,mx),peak_memory_bytes=mx.get_peak_memory(),
                active_memory_bytes=mx.get_active_memory())
            row['passed']=bool(same_set and gates['zero_expert_and_engram_io']and gates['required_host_route_contract']
                and row['raw_last_logits']['equal']and row['cache']['equal'])
            if not row['passed']:
                error=RuntimeError('Warm resident PP exact/noIO gate failed');error.sample_result=row;raise error
            return row
        finally:
            w.source_resident_bank_session=session
            w.source_resident_gather_policy,w.source_resident_prefill_policy=policies
            if not drained:
                try:mx.synchronize();drained=True
                except BaseException:
                    session.poisoned=True;session._quarantined.append(('warm_model_gate',cache,out))
            if drained:
                for state in cache:state.close()
    try:
        for arm in ('ordinary','padded_v1','persistent_8_2_v1'):record['warmups'].append(one(arm))
        for variant,arm in _warm_order(rounds):
            row=one(arm);row['paired_variant']=variant;record['samples'].append(row)
        record['summary']={}
        for variant in ('padded_v1','persistent_8_2_v1'):
            values=[v for v in record['samples']if v['paired_variant']==variant]
            times={arm:[v['seconds']for v in values if v['arm']==arm]for arm in ('ordinary',variant)}
            medians={arm:statistics.median(v)for arm,v in times.items()}
            record['summary'][variant]=dict(samples_seconds=times,medians_seconds=medians,
                diagnostic_ordinary_to_candidate_ratio=medians['ordinary']/medians[variant],
                all_functional_gates_pass=all(v['passed']for v in values),
                all_swap_gates_pass=all(v['gates']['swap_stable']for v in values),
                rate_eligible=False,controlled_performance_claim=False)
        record['passed']=True
    except BaseException as error:
        if hasattr(error,'sample_result'):record['failed_sample']=error.sample_result
        record['passed']=False
        raise
    finally:
        store.acquire=acquire
        w.source_resident_bank_session=session
        w.source_resident_gather_policy,w.source_resident_prefill_policy=policies
        if not session.poisoned:session.set_prefill_task_policy('padded_v1')


def run(adapter,unit,guard,ids,*,rows=99,warm_rounds=1,environment_probe=None):
    """Run only inside the parent's bounded GPU ownership unit, after Hello."""
    import mlx.core as mx
    from mlx2.adapters.deepseek_v41_source_prefill_session import prepare_prefill_session,implementation_identity
    spec=plan(rows);_warm_order(warm_rounds)
    if environment_probe is not None and not callable(environment_probe):raise ValueError('Environment probe must be callable')
    spec['warm_rounds']=warm_rounds
    unit=Path(unit);path=unit/'resident-prefill-model-gate.json'
    if path.exists()or path.with_suffix('.json.tmp').exists():raise FileExistsError('PP evidence already exists')
    if not callable(guard):raise ValueError('PP gate requires a live owner guard')
    def live():
        if guard()is False:raise RuntimeError('PP gate ownership lost')
        return True
    live()
    if mx.default_device()!=mx.gpu:raise ValueError('PP gate requires owned GPU')
    w=adapter.weights;store=w.expert_stream;model=adapter.model
    _checkpoint(adapter)
    if getattr(w,'source_resident_bank_session',None)is not None:raise ValueError('PP gate requires idle unattached bank')
    if getattr(w,'_source_current_frame',None)is not None:raise ValueError('PP gate cannot enter an active model frame')
    tokens=[int(v)for v in ids]
    if len(tokens)<rows:raise ValueError('PP gate prompt is shorter than selected fixed row count')
    old=(w.source_resident_gather_policy,w.source_resident_prefill_policy)
    if old!=('resident_bank_v1','expert_major_wide_v1'):
        raise ValueError('Load adapter with explicit resident_bank_v1/expert_major_wide_v1 profile first')
    result=dict(schema='mlx2.ds41-resident-prefill-model-gate.v1',plan=spec,checkpoint=CHECKPOINT,
        source_revision=__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),
        implementation_sha256=implementation_identity(),prompt_ids_sha256=hashlib.sha256(json.dumps(tokens[:rows]).encode()).hexdigest(),
        status='starting',qualified=False,controlled_performance_claim=False,APCv2_publication=False)
    sealed=unit/'spec.json'
    if sealed.exists():
        bound=json.loads(sealed.read_text())
        result['parent_source_snapshot_sha256']=bound.get('source_snapshot_sha256')
    reference_cache=candidate_cache=None;session=None
    try:
        # This drops only old expert coefficients from Hello. Dense weights and
        # Engram row caches remain live; their counters are measured separately.
        _park_experts(store,live)
        w.source_resident_gather_policy='reference';w.source_resident_prefill_policy='reference'
        reference_cache=model.make_cache();inputs=mx.array([tokens[:rows]],mx.int32)
        seen=set();acquire=store.acquire
        def observed(prefix,expert):
            seen.add((int(prefix.split('.')[1]),int(expert)))
            return acquire(prefix,expert)
        store.acquire=observed
        before=_counters(adapter);start=time.perf_counter()
        try:
            reference=model(inputs,reference_cache,logits_to_keep=1)
            mx.eval(reference,[s.state for s in reference_cache]);live()
        finally:store.acquire=acquire
        result['reference']=dict(seconds_including_cold_reads=time.perf_counter()-start,
            counter_delta=_delta(_counters(adapter),before),expert_keys=len(seen),cache_bytes=store._cache_bytes)
        if seen!=set(store._cache):raise RuntimeError('Ordinary PP expert set did not remain completely resident')
        if any(s.offset!=rows for s in reference_cache):raise RuntimeError('Ordinary PP cache frontier differs')
        result['reference']['expert_keys_sha256']=hashlib.sha256(json.dumps(sorted(seen)).encode()).hexdigest()
        w.source_resident_gather_policy,w.source_resident_prefill_policy=old
        live();construction_start=time.perf_counter()
        session=prepare_prefill_session(w,tuple(sorted(seen)),ownercheck=live,
            generation=getattr(store,'_resident_bank_generation',0)+1,
            budget_bytes=75_000_000_000,max_bank_bytes=3<<30)
        result['construction']=dict(seconds=time.perf_counter()-construction_start,bank=session.bank.receipt(),
            active_memory_bytes=mx.get_active_memory(),peak_memory_bytes=mx.get_peak_memory())
        session.set_cache_fork_policy('descriptor_cow_v1')
        candidate_cache=model.make_cache();before=_counters(adapter);live();start=time.perf_counter()
        candidate=model(inputs,candidate_cache,logits_to_keep=1)
        mx.eval(candidate,[s.state for s in candidate_cache]);seconds=time.perf_counter()-start;live()
        delta=_delta(_counters(adapter),before)
        no_io=all(delta.get(k,0)==0 for k in ('expert_bytes_read','expert_uploads','engram_bytes_read','host_route_materializations','ordinary_replays'))
        result['candidate']=dict(seconds=seconds,counter_delta=delta,zero_io_and_no_host_routes_or_replay=no_io,
            session=session.receipt(),active_memory_bytes=mx.get_active_memory(),peak_memory_bytes=mx.get_peak_memory(),
            raw_last_logits=_array_compare(reference,candidate,mx),cache=_cache_compare(reference_cache,candidate_cache,mx))
        if not(no_io and result['candidate']['raw_last_logits']['equal']and result['candidate']['cache']['equal']):
            raise RuntimeError('Resident PP model exact/no-IO functional gate failed')
        if any(s.offset!=rows for s in candidate_cache):raise RuntimeError('Candidate PP frontier differs')
        if warm_rounds:
            result['warm_resident']={}
            _warm_resident(adapter,session,inputs,reference,reference_cache,live,result['warm_resident'],
                rounds=warm_rounds,environment_probe=environment_probe)
        # Continuation correctness is a state/output gate. It is ordinary in both
        # arms and outside timing, because the PP-only bank is not decode-closed.
        session.close();session=None
        w.source_resident_gather_policy='reference';w.source_resident_prefill_policy='reference'
        continuation=int(mx.argmax(reference[0,-1]).item())
        one=mx.array([[continuation]],mx.int32);live()
        a=model(one,reference_cache,logits_to_keep=1);mx.eval(a,[s.state for s in reference_cache])
        b=model(one,candidate_cache,logits_to_keep=1);mx.eval(b,[s.state for s in candidate_cache]);live()
        result['continuation']=dict(token=continuation,ordinary_both_arms=True,
            logits=_array_compare(a,b,mx),cache=_cache_compare(reference_cache,candidate_cache,mx),frontier=rows+1)
        if not(result['continuation']['logits']['equal']and result['continuation']['cache']['equal']):
            raise RuntimeError('PP continuation exact gate failed')
        result.update(status='passed',passed=True)
        return result
    except BaseException as error:
        result.update(status='failed',passed=False,error=dict(type=type(error).__name__,message=str(error)))
        error.partial_result=result
        raise
    finally:
        cleanup_error=None
        if session is not None and not session.poisoned:
            try:session.close()
            except BaseException as error:
                cleanup_error=error
                result.update(status='cleanup_failed',passed=False,
                    cleanup_error=dict(type=type(error).__name__,message=str(error)))
                session._quarantined.append(('model_gate_close',reference_cache,candidate_cache))
        w.source_resident_gather_policy,w.source_resident_prefill_policy=old
        if session is not None and session.poisoned:
            session._quarantined.append(('model_gate',reference_cache,candidate_cache))
        elif cleanup_error is None:
            for cache in (reference_cache,candidate_cache):
                if cache is not None:
                    for state in cache:state.close()
        _write(path,result)
        if cleanup_error is not None:raise cleanup_error
