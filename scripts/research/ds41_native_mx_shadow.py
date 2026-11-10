# SPDX-License-Identifier: MIT
"""Bounded actual PP native-MX shadow; every serving object remains reference.

Provenance: deepseek-v41-native-mx-shadow.json. No GPU work on import. Native
HALF consumption is an explicit numerical profile; differences are observations,
not an automatic quality failure. The caller owns GPU and existing leases.
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
from contextlib import contextmanager
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import threading
import time

import numpy as np

SCHEMA='mlx2.ds41-native-mx-shadow.v1'
MAX_BYTES=32<<20
_LOCK=threading.Lock()


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _digest(a):return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()
def _eligible(job):return job.kind=='original'and 16<=job.count<=99


def _atomic(path,value):
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if len(data)>MAX_BYTES:raise MemoryError('Native MX shadow JSON ceiling')
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('xb')as f:f.write(data)
    temporary.replace(path)


def _compare(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.dtype!=np.float32 or b.dtype!=np.float32 or a.shape!=b.shape:
        raise ValueError('Native MX shadow F32 comparison geometry')
    aa,bb=a.view(np.uint32).ravel(),b.view(np.uint32).ravel();mask=aa!=bb
    count=int(np.count_nonzero(mask));finite=bool(np.isfinite(a).all()and np.isfinite(b).all())
    maximum=0.
    if finite:
        for first in range(0,a.size,16384):
            maximum=max(maximum,float(np.max(np.abs(a.ravel()[first:first+16384].astype(np.float64)-b.ravel()[first:first+16384].astype(np.float64)))))
    result=dict(exact=count==0,changed_words=count,finite=finite,shape=list(a.shape),
        reference_sha256=_digest(a),candidate_sha256=_digest(b),max_abs_error=maximum if finite else None)
    if count:
        i=int(np.argmax(mask))
        result['first_mismatch']=dict(flat_index=i,coordinate=[int(x)for x in np.unravel_index(i,a.shape)],
            reference_bits=f'{int(aa[i]):08x}',candidate_bits=f'{int(bb[i]):08x}',
            reference_value=float(a.ravel()[i])if np.isfinite(a.ravel()[i])else None,
            candidate_value=float(b.ravel()[i])if np.isfinite(b.ravel()[i])else None)
    return result


def _half_domain(a):
    a=np.asarray(a)
    if a.dtype!=np.float32 or a.ndim!=2:raise ValueError('Native MX shadow activation geometry')
    with np.errstate(over='ignore',invalid='ignore'):
        restored=a.astype(np.float16).astype(np.float32)
    result=_compare(a,restored)
    result['source_finite']=bool(np.isfinite(a).all())
    result['restored_finite']=bool(np.isfinite(restored).all())
    result['samples']=int(a.size)
    return result


def _validate_spec(spec,helper_sha,input_rows,frontier):
    if type(input_rows)is not int or type(frontier)is not int or(input_rows,frontier)!=(99,0):
        raise ValueError('Native MX shadow is PP99/frontier0 only')
    if (spec.get('native_mx_shadow')is not True or spec.get('resident_discovery')is not True
            or spec.get('rate_eligible')is not False or spec.get('context_cap')!=16384
            or spec.get('prompt_tokens')!=100 or spec.get('output_tokens')!=32
            or spec.get('harness_sha256',{}).get('ds41_native_mx_shadow.py')!=helper_sha
            or spec.get('expert_cache_bytes')!=75_000_000_000
            or spec.get('expert_policy',{}).get('hot_bytes')!=60_000_000_000
            or spec.get('expert_policy',{}).get('mode')!='hot_original_q1'):
        raise ValueError('Native MX shadow requires sealed isolated untimed75GB60/15 query')
    if any(spec.get(k)is True for k in ('small_m_shadow','woa_shadow','down_actual_capture','q1_down_shadow','native_ffn_capture','closed_resident_boundary_diagnostic')):
        raise ValueError('Native MX shadow diagnostic overlap')
    for k in ('checkpoint','source_revision','source_snapshot_sha256'):
        if not isinstance(spec.get(k),str)or not spec[k]:raise ValueError('Native MX shadow missing identity')
    if (not re.fullmatch('[0-9a-f]{40}',spec['source_revision'])
            or not re.fullmatch('[0-9a-f]{64}',spec['source_snapshot_sha256'])
            or not re.fullmatch('.+@[0-9a-f]{64}',spec['checkpoint'])
            or spec.get('source_expert_math')!='routed_fused_fp32'):
        raise ValueError('Native MX shadow source/checkpoint/expert-math identity')
    policy=spec.get('extra_source_policy',{})
    if (policy.get('deepseek_source_prefill_batch')!='gate_up'
            or policy.get('deepseek_source_native_mx','reference')!='reference'
            or policy.get('deepseek_source_prefill_wide_math','reference')!='reference'
            or policy.get('deepseek_source_prefill_wide_dtype','reference')!='reference'):
        raise ValueError('Native MX shadow requires ordinary reference expert profile')


def _coverage(layers):
    ids=[c['layer']for c in layers]
    if ids!=list(range(40)):raise ValueError('Native MX shadow incomplete/duplicate layer order')
    eligible=sum(c['eligible_jobs']for c in layers)
    for c in layers:
        count=c['eligible_jobs']
        if (c['acquired_jobs']!=c['planned_jobs']or c['independent_projection_calls']!=3*count
                or c['middle_checks']!=count or c['composed_down_calls']!=count or c['assembly_checks']!=count):
            raise ValueError('Native MX shadow incomplete actual job census')
    return eligible


def _replacement(parts,offsets,rows,ranks,chosen):
    """Bounded CPU ordered+0 fold; no extra GPU assembly graph/copy."""
    total=np.empty_like(chosen)
    for first in range(0,chosen.shape[1],128):
        last=min(first+128,chosen.shape[1]);acc=np.zeros((len(rows),last-first),np.float32)
        for slot in range(6):
            value=parts[offsets[rows,slot],first:last]
            mask=ranks==slot
            value[mask]=chosen[mask,first:last]
            np.add(acc,value,out=acc)
        total[:,first:last]=acc
    return total


@contextmanager
def instrument(model,unit,label,index,*,input_rows,frontier):
    if not _LOCK.acquire(blocking=False):raise RuntimeError('Nested/concurrent native MX shadow refused')
    patches=[];report=None;output=None;active=None;candidate_depth=0
    owner=threading.get_ident();started=time.perf_counter()
    try:
        if type(index)is not int or index<0 or not isinstance(label,str)or not re.fullmatch(r'[A-Za-z0-9_.-]+',label):
            raise ValueError('Native MX shadow forward identity')
        unit=Path(unit);output=unit/f'native-mx-shadow-{index:04d}.json'
        if output.exists():raise FileExistsError(output)
        spec=json.loads((unit/'spec.json').read_text());helper_sha=_sha(__file__)
        _validate_spec(spec,helper_sha,input_rows,frontier)
        import mlx.core as mx
        from mlx2.adapters import deepseek_v41_source_prefill as planner,deepseek_v41_source_math as source_math
        from mlx2.runtime.gguf import source_native_mx_runtime as native,source_middle_fusion as middle
        weights=model.weights;stream=weights.expert_stream
        if (getattr(weights,'source_native_mx_policy','reference')!='reference'
                or getattr(weights,'source_prefill_batch_policy',None)!='gate_up'
                or getattr(weights,'source_prefill_wide_math','reference')!='reference'
                or getattr(weights,'source_prefill_wide_dtype','reference')!='reference'
                or getattr(weights,'expert_math_policy',None)!='routed_fused_fp32'
                or not getattr(stream,'packed_buffers',False)
                or not callable(getattr(weights,'_source_native_mx_bound',None))):
            raise ValueError('Native MX shadow needs owning ordinary provider and enabled host attestations')
        report=dict(schema=SCHEMA,status='running',complete=False,**{k:spec[k]for k in ('checkpoint','source_revision','source_snapshot_sha256')},
            helper_sha256=helper_sha,label=label,forward_index=index,input_rows=input_rows,frontier=frontier,
            reference_returning=True,source_result_always_returned=True,candidate_used_by_model=False,rate_eligible=False,
            qualified=False,model_quality_assessed=False,expert_budget_bytes=75_000_000_000,original_hot_bytes=60_000_000_000,
            original_artifact_fingerprint=stream.artifact['identity']['fingerprint'],q1_bank_binding=stream.q1_binding,hot_map_binding=stream.hot_binding,
            runtime_module_sha256=_sha(native.__file__),scope='Actual original PP M16..99 independent projections plus composed GU/middle/DOWN/partial assembly shadow; no all-prompt or quality proof',
            layers=[],cases=[],first_differences={},missing_attestations=[],maximum_diagnostic_bytes=0,
            maximum_diagnostic_bytes_limit=MAX_BYTES,total_processed_array_bytes=0,CPU_assembly_packed_bytes=0,
            extra_expert_acquisitions=0,blocked_extra_acquire_attempts=0,coefficient_payload_exported=False,
            extra_coefficient_upload_bytes=0,extra_coefficient_repack_bytes=0,all_candidate_finite=True,
            half_domain_samples=0,half_domain_changed_words=0,half_domain_lossy_calls=0)

        def thread_guard():
            if threading.get_ident()!=owner:raise RuntimeError('Concurrent native MX shadow provider refused')
        def patch(obj,name,value):
            local=name in vars(obj);before=getattr(obj,name);patches.append((obj,name,local,before));setattr(obj,name,value);return before
        def peak(extras=(),cpu_bytes=0):
            arrays={}
            if active:
                for stage in active['stages'].values():
                    for a in stage['held'].values():arrays[id(a)]=a
            for a in extras:arrays[id(a)]=a
            size=sum(int(a.nbytes)for a in arrays.values())+int(cpu_bytes)
            report['maximum_diagnostic_bytes']=max(report['maximum_diagnostic_bytes'],size)
            if size>MAX_BYTES:raise MemoryError('Native MX shadow peak diagnostic ceiling')
            return size
        def inspect_domain(x):
            peak((x,),cpu_bytes=3*x.nbytes+x.size+min(x.size,16384)*8*4)
            value=np.asarray(x);result=_half_domain(value)
            report['half_domain_samples']+=result['samples'];report['half_domain_changed_words']+=result['changed_words']
            report['half_domain_lossy_calls']+=int(not result['exact'])
            report['total_processed_array_bytes']+=int(value.nbytes)
            return result
        def checks(name,reference,candidate,row):
            ref_owner=reference.astype(mx.bfloat16).astype(mx.float32)
            cand_owner=candidate.astype(mx.bfloat16).astype(mx.float32)
            size=peak((reference,candidate,ref_owner,cand_owner),
                cpu_bytes=2*reference.nbytes+reference.size+min(reference.size,16384)*8*4)
            mx.eval(reference,candidate,ref_owner,cand_owner)
            raw=_compare(np.asarray(reference),np.asarray(candidate));own=_compare(np.asarray(ref_owner),np.asarray(cand_owner))
            report['total_processed_array_bytes']+=sum(int(v.nbytes)for v in (reference,candidate,ref_owner,cand_owner))
            report['all_candidate_finite']&=raw['finite']and own['finite']
            result=dict(raw=raw,BF16_owner=own,diagnostic_bytes=size)
            for boundary,c in [('raw',raw),('BF16_owner',own)]:
                key=name+'.'+boundary
                if not c['exact']and key not in report['first_differences']:
                    report['first_differences'][key]=dict(layer=active['layer'],expert=row['expert'],rows=row['rows'],**c['first_mismatch'])
            return result
        before_plan=planner.plan_jobs
        def plan_jobs(values,kinds,*args,**kwargs):
            thread_guard()
            if candidate_depth:raise RuntimeError('Native MX candidate attempted route planning')
            result=before_plan(values,kinds,*args,**kwargs)
            if active is not None:
                if values is not active['values']or active['plan']is not None:raise ValueError('Native MX actual plan identity')
                active['plan']=result;active['jobs']={j.expert_id:j for j in result.jobs}
                active['eligible']={j.expert_id for j in result.jobs if _eligible(j)}
            return result
        patch(planner,'plan_jobs',plan_jobs)
        before_acquire=stream.acquire
        def acquire(prefix,eid):
            thread_guard()
            if candidate_depth:
                report['blocked_extra_acquire_attempts']+=1;raise RuntimeError('Native MX candidate extra acquire refused')
            data=before_acquire(prefix,eid)
            if active is not None:
                key=(active['layer'],eid)
                if prefix!=active['prefix']or eid not in active['jobs']or eid in active['acquired']or not stream._active or stream._leased.get(key)is not data:
                    raise ValueError('Native MX acquired job/lease identity')
                active['acquired'][eid]=data
                if eid in active['eligible']:
                    bound=weights._source_native_mx_bound(prefix,eid,data)
                    if bound is None:
                        report['missing_attestations'].append(dict(layer=key[0],expert=eid));raise ValueError('Native MX missing CPU-loaded attestation; no readback fallback')
                    proof=bound.attestation.receipt()
                    if (proof.get('eligible')is not True or proof.get('source_fingerprint')!=stream.artifact['identity']['fingerprint']
                            or(proof.get('layer'),proof.get('expert'))!=key):
                        raise ValueError('Native MX CPU attestation domain/source/job refused')
                    job=active['jobs'][eid]
                    stage=dict(bound=bound,held={},seen=set(),middle_seen=False,assembled=False,
                        row=dict(layer=key[0],expert=eid,rows=job.count,route_rows=job.rows.tolist(),route_locations=job.locations.tolist(),
                            borrowed_existing_lease=True,coefficient_payload_exported=False,coefficient_attestation=proof,
                            independent={},half_domains={}))
                    active['stages'][eid]=stage
                    for projection in ('gate','up','down'):
                        q=data[projection+'.qweight'];tag=id(q)
                        if tag in active['tags']:raise ValueError('Native MX ambiguous coefficient object')
                        active['tags'][tag]=(eid,projection)
            return data
        patch(stream,'acquire',acquire)
        before_qmm=mx.quantized_matmul
        def qmm(x,q,*args,**kwargs):
            nonlocal candidate_depth
            thread_guard()
            if candidate_depth:raise RuntimeError('Native MX candidate reentered stock QMM')
            reference=before_qmm(x,q,*args,**kwargs)
            tag=active['tags'].get(id(q))if active is not None else None
            if tag is None:return reference
            eid,projection=tag;stage=active['stages'][eid];job=active['jobs'][eid];data=active['acquired'][eid]
            width=2304 if projection=='down'else 5120
            if (args or kwargs.get('mode')!='mxfp4'or kwargs.get('transpose',True)is not True or kwargs.get('biases')is not None
                    or kwargs.get('scales')is not data[projection+'.scales']or q is not data[projection+'.qweight']
                    or not stream._active or stream._leased.get((active['layer'],eid))is not data or x.dtype!=mx.float32 or x.shape!=(job.count,width)
                    or projection in stage['seen']):raise ValueError('Native MX actual QMM operand/projection identity')
            stage['seen'].add(projection);row=stage['row']
            candidate_depth+=1
            try:
                if projection=='gate':
                    stage['GU_input_id']=id(x);stage['held']['GU_half']=native.prepare_half(x)
                    row['half_domains']['actual_GU']=inspect_domain(x)
                elif projection=='up'and id(x)!=stage.get('GU_input_id'):raise ValueError('Native MX G/U do not share actual input')
                if projection=='down':
                    if not stage['middle_seen']or id(x)!=stage.get('actual_middle_id'):raise ValueError('Native MX DOWN did not consume actual source middle')
                    row['half_domains']['actual_DOWN']=inspect_domain(x)
                    half=native.prepare_half(x)
                else:half=stage['held']['GU_half']
                candidate=native.project_half(half,data,stage['bound'],projection=projection)
                stage['held']['candidate_'+projection]=candidate
                row['independent'][projection]=checks('independent_'+projection,reference,candidate,row)
                active['independent_calls']+=1
                if projection in ('gate','up'):
                    active['raw_tags'][id(reference)]=(eid,projection)
                else:
                    composed_middle=stage['held']['native_middle'];row['half_domains']['native_composed_DOWN']=inspect_domain(composed_middle)
                    composed=native.project_half(native.prepare_half(composed_middle),data,stage['bound'],projection='down')
                    row['composed_DOWN']=checks('composed_down',reference,composed,row);active['composed_calls']+=1
                    stage['held']['composed_owner']=composed.astype(mx.bfloat16).astype(mx.float32);mx.eval(stage['held']['composed_owner'])
                    stage['held'].pop('native_middle');stage['held'].pop('candidate_down')
            finally:candidate_depth-=1
            return reference
        patch(mx,'quantized_matmul',qmm)
        before_middle=middle.middle
        def scored_middle(g,u,scores,slots):
            nonlocal candidate_depth
            thread_guard()
            reference=before_middle(g,u,scores,slots)
            tag=active['raw_tags'].get(id(g))if active is not None else None
            if tag is None:return reference
            eid,projection=tag;stage=active['stages'][eid]
            if projection!='gate'or active['raw_tags'].get(id(u))!=(eid,'up')or stage['middle_seen']:
                raise ValueError('Native MX scored middle producer identity')
            stage['middle_seen']=True;stage['actual_middle_id']=id(reference);candidate_depth+=1
            try:
                candidate=before_middle(stage['held']['candidate_gate'],stage['held']['candidate_up'],scores,slots)
                stage['held']['native_middle']=candidate;peak((reference,candidate));mx.eval(reference,candidate)
                check=_compare(np.asarray(reference),np.asarray(candidate));stage['row']['scored_FP8_middle']=check
                report['total_processed_array_bytes']+=int(reference.nbytes+candidate.nbytes)
                report['all_candidate_finite']&=check['finite'];active['middle_checks']+=1
                if not check['exact']and 'scored_FP8_middle'not in report['first_differences']:
                    report['first_differences']['scored_FP8_middle']=dict(layer=active['layer'],expert=eid,rows=active['jobs'][eid].count,**check['first_mismatch'])
                for name in ('candidate_gate','candidate_up','GU_half'):stage['held'].pop(name)
                active['raw_tags'].pop(id(g));active['raw_tags'].pop(id(u))
            finally:candidate_depth-=1
            return reference
        patch(middle,'middle',scored_middle)
        before_combine=source_math.ordered_combine
        def combine(parts,values,order):
            thread_guard();reference=before_combine(parts,values,order)
            if active is None:return reference
            if values is not active['values']or order is not active['order']or len(parts)!=len(active['plan'].jobs):
                raise ValueError('Native MX assembly actual parts identity')
            if not active['eligible']:return reference
            for stage in active['stages'].values():
                if stage['seen']!={'gate','up','down'}or not stage['middle_seen']or 'composed_owner'not in stage['held']:
                    raise ValueError('Native MX incomplete composed stage before assembly')
            mx.eval(reference,*parts,*[s['held']['composed_owner']for s in active['stages'].values()])
            packed=np.empty((values.size,5120),np.float32);baseline=np.asarray(reference);cursor=0
            for part,job in zip(parts,active['plan'].jobs):
                if part.shape!=(job.count,5120)or part.dtype!=mx.float32:raise ValueError('Native MX assembly source part geometry')
                value=np.asarray(part);peak(cpu_bytes=packed.nbytes+baseline.nbytes+value.nbytes)
                packed[cursor:cursor+len(value)]=value;cursor+=len(value)
            del value
            report['CPU_assembly_packed_bytes']+=int(packed.nbytes)
            offsets=source_math.ordered_offsets(values,order)
            for eid in sorted(active['eligible']):
                stage=active['stages'][eid];job=active['jobs'][eid];chosen=np.asarray(stage['held']['composed_owner'])
                ranks=np.sum(values[job.rows]<eid,axis=1).astype(np.uint32)
                cpu_bytes=packed.nbytes+baseline.nbytes+3*chosen.nbytes+chosen.size+min(chosen.size,16384)*8*4+16384
                peak(cpu_bytes=cpu_bytes)
                assembled=_replacement(packed,offsets,job.rows,ranks,chosen)
                check=_compare(baseline[job.rows],assembled);stage['row']['composed_partial_assembly']=check
                stage['row']['assembly_scope']='Only selected expert replaced; other five actual contributions fixed; ordered F32+0; not model logits'
                report['total_processed_array_bytes']+=int(chosen.nbytes+assembled.nbytes)
                if not check['exact']and 'composed_partial_assembly'not in report['first_differences']:
                    report['first_differences']['composed_partial_assembly']=dict(layer=active['layer'],expert=eid,rows=job.count,**check['first_mismatch'])
                stage['assembled']=True;stage['held'].clear();active['assembly_checks']+=1
                report['cases'].append(stage['row'])
            return reference
        patch(source_math,'ordered_combine',combine)
        before_provider=weights._prefill_batched_gate_up
        def provider(x,values,scores,prefix,order):
            nonlocal active
            thread_guard()
            if candidate_depth or active is not None:raise RuntimeError('Reentrant native MX provider')
            layer=int(prefix.split('.')[1])
            if layer!=len(report['layers'])or values.shape!=(99,6):raise ValueError('Native MX provider layer/frame order')
            active=dict(layer=layer,prefix=prefix,values=values,order=order,plan=None,jobs={},eligible=set(),acquired={},tags={},raw_tags={},stages={},
                independent_calls=0,middle_checks=0,composed_calls=0,assembly_checks=0)
            try:
                result=before_provider(x,values,scores,prefix,order)
                if active['plan']is None:raise ValueError('Native MX actual provider plan not observed')
                layer_receipt=dict(layer=layer,planned_jobs=len(active['jobs']),acquired_jobs=len(active['acquired']),eligible_jobs=len(active['eligible']),
                    eligible_M_histogram=dict(sorted(Counter(str(active['jobs'][e].count)for e in active['eligible']).items())),
                    independent_projection_calls=active['independent_calls'],middle_checks=active['middle_checks'],composed_down_calls=active['composed_calls'],assembly_checks=active['assembly_checks'])
                if set(active['acquired'])!=set(active['jobs'])or any(not s['assembled']for s in active['stages'].values()):
                    raise ValueError('Native MX provider acquire/stage coverage failed')
                report['layers'].append(layer_receipt);return result
            finally:active=None
        patch(weights,'_prefill_batched_gate_up',provider)
        yield report
        eligible=_coverage(report['layers'])
        if len(report['cases'])!=eligible:raise ValueError('Native MX serialized job census differs')
        report.update(status='completed',complete=True,model_forward_completed=True,eligible_jobs=eligible,
            coverage_complete=True,candidate_dispatched=eligible>0,
            independent_projection_calls=3*eligible,composed_down_calls=eligible,reference_objects_always_returned=True,
            actual_half_domain_exact=report['half_domain_changed_words']==0,
            numerical_scope='Lossy half input and raw/owner/middle/partial-assembly differences are exposed; no automatic quality verdict')
    except BaseException as error:
        if report is not None:report.update(status='failed',complete=False,error_type=type(error).__name__,error=str(error))
        raise
    finally:
        errors=[]
        try:
            for obj,name,local,before in reversed(patches):
                try:
                    if local:setattr(obj,name,before)
                    else:delattr(obj,name)
                except BaseException as error:errors.append(dict(field=name,type=type(error).__name__,message=str(error)))
            if report is not None and output is not None:
                if errors:report.update(status='failed',complete=False,cleanup_errors=errors)
                report['wall_seconds']=time.perf_counter()-started
                _atomic(output,report)
        finally:_LOCK.release()
        if errors:raise RuntimeError('Native MX shadow hook restoration failed')
