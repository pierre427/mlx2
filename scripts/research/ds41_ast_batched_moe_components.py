# SPDX-License-Identifier: MIT
"""Actual-format M16 resident MoE parity, callable only by an owned GPU unit.

Import is CPU-only. run(adapter,unit,check_live) compares unchanged Source M1
GU/DOWN and stable combine with the new GPU-grouped resident tile. Parent owns
source/artifact/CPG/paired locks/thermal admission and process-group cleanup.
"""
from pathlib import Path
import gc
import hashlib
import json
import os
import time

_QUARANTINE=[]


def _write(path,value):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('x')as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def _selection(store):
    # A96-object component bank remains about1.22GB. All route cases share it;
    # every coefficient keeps the original immutable hot/Q1 format selection.
    for layer in range(40):
        original=[(layer,e)for e in range(384)if store.kind((layer,e))=='original']
        q1=[(layer,e)for e in range(384)if store.kind((layer,e))=='q1']
        if len(original)>=48 and len(q1)>=48:
            return layer,tuple(original[:48]),tuple(q1[:48])
    raise ValueError('Actual-format component needs one layer with48 original and48Q1 experts')


def _routes(original,q1):
    import numpy as np
    a=[k[1]for k in original];b=[k[1]for k in q1]
    shared=lambda v:np.broadcast_to(np.array(v,np.uint32),(16,6)).copy()
    mixed=[a[2],b[1],a[0],b[2],a[1],b[0]]
    distinct=np.array(a+b,np.uint32).reshape(16,6)
    return [('original_shared16',shared(a[:6])),('q1_shared16',shared(b[:6])),
            ('mixed_shared16',shared(mixed)),('globally_distinct96',distinct)]


def _injection(consumer, consumer_identity, evidence_name, timing_repeats, down_consumer):
    if (consumer is None) != (consumer_identity is None):
        raise ValueError('Explicit consumer and identity must be supplied together')
    if consumer is not None and (not callable(consumer) or not isinstance(consumer_identity,str)
            or len(consumer_identity)!=64 or any(c not in '0123456789abcdef' for c in consumer_identity)):
        raise ValueError('Explicit consumer requires callable and SHA256 identity')
    if (not isinstance(evidence_name,str) or Path(evidence_name).name!=evidence_name
            or not evidence_name.endswith('.json') or len(evidence_name)>96 or '..' in evidence_name):
        raise ValueError('Component evidence must be a bounded JSON basename')
    if type(timing_repeats)is not int or not 0<=timing_repeats<=8:
        raise ValueError('Component timing repeats must be0..8')
    if down_consumer is not None and (consumer is None or not callable(down_consumer)):
        raise ValueError('Standalone DOWN requires explicit callable consumer')


def run(adapter,unit,check_live,*,consumer=None,consumer_identity=None,
        evidence_name='ast-batched-moe-components.json',timing_repeats=0,down_consumer=None):
    import numpy as np
    import mlx.core as mx
    if not callable(check_live):raise ValueError('Actual MoE component requires live ownership callback')
    if mx.default_device()!=mx.gpu:raise ValueError('Actual MoE component requires parent-owned GPU')
    _injection(consumer,consumer_identity,evidence_name,timing_repeats,down_consumer)
    from mlx2.adapters.deepseek_v41_source_prefill_bank import build_prefill_bank
    from mlx2.runtime.models.deepseek_v41_quant import fused_quantize
    from mlx2.runtime.gguf.source_ast_m1_expert import middle
    from mlx2.runtime.gguf.source_ast_down_rows import down
    from mlx2.runtime.gguf.source_middle_fusion import exp_spelling
    from mlx2.runtime.gguf import source_ast_batched_moe as control
    from mlx2.runtime.gguf.source_resident_bank_gather import layout_key
    decode_tile=control.decode_tile if consumer is None else consumer
    implementation_identity=control.implementation_identity;plan=control.plan
    from mlx2.adapters.deepseek_v41_source_math import ordered_combine
    from ds41_resident_prefill_model_gate import _checkpoint,_array_compare,_park_experts
    def live():
        if check_live()is False:raise RuntimeError('Actual MoE component ownership lost')
        return True
    live()
    unit=Path(unit);path=unit/evidence_name
    if path.exists()or path.with_name(path.name+'.tmp').exists():raise FileExistsError('Component evidence already exists')
    w=adapter.weights;store=w.expert_stream
    if getattr(w,'source_resident_bank_session',None)is not None:raise ValueError('Component requires idle unattached expert bank')
    checkpoint=_checkpoint(adapter);layer,original,q1=_selection(store);keys=frozenset(original+q1)
    result=dict(schema='mlx2.ds41-ast-batched-moe-components.v1',checkpoint=checkpoint,
        implementation_sha256=implementation_identity()if consumer_identity is None else consumer_identity,
        control_implementation_sha256=implementation_identity(),explicit_consumer=consumer is not None,helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        profile='source_ast_resident_rows_v1',component_plan=plan(),layer=layer,
        coefficient_precision='unchanged actual Source original MXFP4g32 and Q1g64',
        expert_keys=[list(k)for k in sorted(keys)],status='preparing',cases=[],controls=[],
        qualified=False,rate_eligible=False,APCv2_publication=False)
    sealed=unit/'spec.json'
    if sealed.exists():
        spec=json.loads(sealed.read_text());result.update(source_revision=spec.get('source_revision'),
            source_snapshot_sha256=spec.get('source_snapshot_sha256'))
    _write(path,result)
    bank=context=lease=None;drained=True;owners=[];prior_pins=getattr(store,'_resident_bank_pinned_keys',frozenset())
    try:
        _park_experts(store,live);gc.collect();mx.clear_cache();live()
        store._resident_bank_pinned_keys=prior_pins|keys
        by_key={key:(prefix,index)for prefix,table in store.tables.items()for index,key in enumerate(table)if key in keys}
        if set(by_key)!=keys:raise ValueError('Component selected expert absent from actual provider table')
        for key in sorted(keys):
            live()
            with store.window():store.acquire(*by_key[key])
        if set(store._cache)!=keys:raise ValueError('Component tiny coefficient cohort was not retained')
        bank=build_prefill_bank(mx,store,tuple(sorted(keys)),ownercheck=live,
            generation=getattr(store,'_resident_bank_generation',0)+1,budget_bytes=75_000_000_000,max_bank_bytes=3<<30)
        context=bank.lease(ownercheck=live);lease=context.__enter__()
        result['bank']=bank.receipt();result['status']='parity';_write(path,result)
        rng=np.random.default_rng(16096);raw=rng.normal(0,.2,(16,5120)).astype(np.float32)
        prepared=fused_quantize(mx.array(raw),'fp8').astype(mx.float32);mx.eval(prepared)
        result['activation_sha256']=hashlib.sha256(raw.tobytes()).hexdigest()
        baseline_mixed=None;scores_mixed=None;routes_mixed=None
        for label,values in _routes(original,q1):
            live();scores=rng.uniform(.1,1.,(16,6)).astype(np.float32);scores/=scores.sum(axis=1,keepdims=True)
            gpu_ids=mx.array(values);gpu_scores=mx.array(scores);flat=values.reshape(-1)
            order=np.argsort(flat,kind='stable');sorted_ids=flat[order];middles=[];parts=[];owners=[];drained=False
            for expert in np.unique(sorted_ids):
                locations=order[sorted_ids==expert];rows=mx.array((locations//6).astype(np.uint32))
                local_scores=gpu_scores.reshape(-1)[mx.array(locations.astype(np.uint32))]
                data=store._cache[(layer,int(expert))]
                m=middle(prepared,rows,local_scores,data,exp_spelling=exp_spelling());p=down(m,data)
                middles.append(m);parts.append(p);owners.extend((m,p))
            ref_middle=mx.concatenate(middles);ref_down=mx.concatenate(parts)
            ref_output=ordered_combine(parts,values,order)
            owners.extend((prepared,gpu_ids,gpu_scores,ref_middle,ref_down,ref_output))
            candidate=decode_tile(prepared,gpu_ids,gpu_scores,lease,layer=layer,
                expected_generation=lease.generation,check_live=live)
            owners.extend(candidate.roots);mx.eval(owners);drained=True;live()
            contributions=candidate.roots[-2]
            if contributions.shape!=(96,5120):raise RuntimeError('Pinned component contribution inventory changed')
            status=np.asarray(candidate.status).tolist()
            comparisons=dict(middle=_array_compare(ref_middle,candidate.middle,mx),
                down=_array_compare(ref_down,contributions,mx),output=_array_compare(ref_output,candidate.output,mx))
            finite=all(bool(mx.all(mx.isfinite(v)).item())for v in(ref_middle,ref_down,ref_output,candidate.middle,contributions,candidate.output))
            counts={int(v):int((flat==v).sum())for v in np.unique(flat)}
            row=dict(label=label,routes=values.tolist(),scores=scores.tolist(),status=status,
                raw= comparisons,finite=finite,expert_occurrences=counts,
                repeated_expert_groups=sum(v>1 for v in counts.values()),
                actual_gu_shared_panels=sum(v//2 for v in counts.values()),
                passed=finite and all(v==0 for v in status)and all(v['equal']for v in comparisons.values()))
            if row['passed'] and timing_repeats:
                # Compile/warm outside timing. Complete scope includes route, tasks,
                # GU, status, DOWN, combine and host submission; no model-rate claim.
                def complete(fn):
                    return fn(prepared,gpu_ids,gpu_scores,lease,layer=layer,
                        expected_generation=lease.generation,check_live=live)
                def ordinary_down():
                    key=layout_key(lease)
                    return control._kernel('down',key)(inputs=[candidate.middle,candidate.roots[1],candidate.route_meta,
                        candidate.roots[4],candidate.status,*lease.anchors],grid=(1280*32,96,1),threadgroup=(32,1,1),
                        output_shapes=[(96,5120)],output_dtypes=[mx.float32])[0]
                def selected_down():
                    return down_consumer(candidate.middle,candidate.roots[1],candidate.route_meta,
                        candidate.roots[4],candidate.status,lease,check_live=live)
                scopes=[('complete',lambda:complete(control.decode_tile),lambda:complete(decode_tile))]
                if down_consumer is not None:scopes.append(('standalone_DOWN',ordinary_down,selected_down))
                timings=[]
                for scope,base,selected in scopes:
                    for fn in(base,selected):
                        live();drained=False;v=fn();owners.extend(v.roots if scope=='complete' else(v,));mx.eval(owners);drained=True
                        if scope!='complete' and not _array_compare(ref_down,v,mx)['equal']:
                            raise RuntimeError('Standalone DOWN warm raw parity failed')
                    for repeat in range(timing_repeats):
                        for arm,fn in ([('control',base),('candidate',selected)]if repeat%2==0 else[('candidate',selected),('control',base)]):
                            live();drained=False;start=time.perf_counter_ns();v=fn()
                            roots=v.roots if scope=='complete' else(v,);owners.extend(roots);mx.eval(roots)
                            elapsed=time.perf_counter_ns()-start;drained=True;live()
                            timings.append(dict(scope=scope,arm=arm,repeat=repeat,elapsed_ns=elapsed,
                                warm_excluded=True,includes_host_submission=True,rate_eligible=False))
                row['timing_samples']=timings
            result['cases'].append(row);_write(path,result)
            if not row['passed']:raise RuntimeError('Actual M16 resident MoE raw M1 parity failed: '+label)
            if label=='mixed_shared16':
                baseline_mixed=ref_output;scores_mixed=gpu_scores;routes_mixed=values.copy()
            owners=[]
        # Controls use the same actual banks and valid input/scores, changing
        # only route admission metadata. Invalid rows must never publish zeros.
        absent=next(e for e in range(384)if(layer,e)not in keys)
        controls=[('missing_cohort',absent,1),('invalid_id',384,2),('duplicate_id',None,4)]
        for label,replacement,bit in controls:
            values=routes_mixed.copy();values[0,0]=values[0,1]if replacement is None else replacement
            live();drained=False;candidate=decode_tile(prepared,mx.array(values),scores_mixed,lease,
                layer=layer,expected_generation=lease.generation,check_live=live)
            owners=list(candidate.roots);mx.eval(owners);drained=True;live()
            status=np.asarray(candidate.status).tolist();row_bad=not bool(mx.all(mx.isfinite(candidate.output[0])).item())
            good=_array_compare(baseline_mixed[1:],candidate.output[1:],mx)
            row=dict(label=label,status=status,required_status_bit=bit,bad_row_nonfinite=row_bad,
                unaffected_rows_raw=good,passed=bool(status[0]&bit)and all(v==0 for v in status[1:])and row_bad and good['equal'])
            result['controls'].append(row);_write(path,result)
            if not row['passed']:raise RuntimeError('Actual M16 resident MoE refusal failed: '+label)
            owners=[]
        result.update(status='passed',passed=True)
        return result
    except BaseException as error:
        result.update(status='failed',passed=False,error=dict(type=type(error).__name__,message=str(error)))
        error.partial_result=result;raise
    finally:
        cleanup_error=None
        if not drained:
            try:mx.synchronize();drained=True
            except BaseException as error:cleanup_error=error
        if drained:
            if context is not None:context.__exit__(None,None,None);context=lease=None
            if bank is not None:
                try:bank.close();bank=None
                except BaseException as error:cleanup_error=error
            if not getattr(store,'_resident_bank_preparing',None)and getattr(store,'_resident_bank',None)is None:
                store._resident_bank_pinned_keys=prior_pins
        else:_QUARANTINE.append((adapter,context,lease,bank,tuple(owners)))
        if cleanup_error is not None:result.update(status='cleanup_failed',passed=False,cleanup_error=str(cleanup_error))
        _write(path,result)
        if cleanup_error is not None:raise cleanup_error
