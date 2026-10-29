# SPDX-License-Identifier: MIT
"""Actual Source attention replay and bounded FP64 error oracle; no GPU on import."""
from pathlib import Path
from contextlib import nullcontext
import hashlib
import time
import statistics
import os

LIMIT=6<<30
KEY='omlx_nax_attention'
PROFILE='source_csa2_omlx_nax_bf16_online_v1'


def fp64_reference(np,arrays):
    """One query at a time; no H*L*K*D materialization, one BLAS CPU thread.

    Uses original captured F32 values promoted to double. Explicit mask/index
    selection, duplicates and per-head sink are retained; no ratio inference.
    """
    try:
        from threadpoolctl import threadpool_limits
        threading_scope=threadpool_limits(limits=1)
    except ImportError:
        if any(os.environ.get(k)!='1'for k in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS')):
            raise ValueError('FP64 oracle requires controller BLAS thread limits before NumPy import')
        threading_scope=nullcontext()
    q=arrays['q'];local=arrays['local'][0,0].astype(np.float64);pool=arrays['pooled'][0].astype(np.float64)
    ids=arrays['indices'];valid=arrays['valid'];mask=arrays['local_mask'];sinks=arrays['sinks'].astype(np.float64)
    if (q.ndim!=4 or q.shape[0]!=1 or q.shape[-1]!=local.shape[-1]or len(local)<1 or len(pool)<1
            or len(local)>255 or len(pool)>4096 or q.shape[2]>128 or q.shape[1]>64 or q.shape[-1]>512 or ids.shape[-1]>512
            or np.any(ids<0)or np.any(ids>=len(pool))):raise ValueError('Bounded FP64 attention geometry/indices')
    out=np.empty(q.shape,np.float64)
    with threading_scope:
        for i in range(q.shape[2]):
            query=q[0,:,i].astype(np.float64)*q.shape[-1]**-.5;selected=pool[ids[0,i].astype(np.int64)]
            ls=query@local.T;ps=query@selected.T
            ls=np.where(mask[0,0,i][None],ls,-np.inf);ps=np.where(valid[0,i][None],ps,-np.inf)
            maximum=np.maximum(np.maximum(np.max(ls,axis=-1),np.max(ps,axis=-1)),sinks)
            lp=np.exp(ls-maximum[:,None]);pp=np.exp(ps-maximum[:,None]);sink=np.exp(sinks-maximum)
            denom=lp.sum(-1)+pp.sum(-1)+sink
            out[0,:,i]=(lp@local+pp@selected)/denom[:,None]
    return out


def fp64_error(np,truth,actual):
    if truth.shape!=actual.shape:raise ValueError('FP64 error shape')
    if not np.isfinite(actual).all():return dict(finite=False)
    difference=actual.astype(np.float64)-truth
    return dict(finite=True,RMS_absolute_error=float(np.sqrt(np.mean(difference*difference))),
                max_absolute_error=float(np.max(np.abs(difference),initial=0.)),
                relative_L2_error=float(np.linalg.norm(difference.ravel())/max(np.linalg.norm(truth.ravel()),1e-300)))


def time_different_profile(mx,np,functions,*,repeats=3):
    """Interleave finite, engineering-admitted variants; no exactness admission."""
    if type(repeats) is not int or repeats not in (3,7):raise ValueError('Bounded peer timing requires three or seven repeats')
    result={k:dict(samples=[],timing_numerical_scope='distinct profile; does not imply exactness or quality') for k in functions}
    def invoke(fn):
        mx.synchronize();start=time.perf_counter();value=fn();submitted=time.perf_counter();mx.eval(value);end=time.perf_counter()
        return value,dict(wall_ms=(end-start)*1000,host_ms=(submitted-start)*1000,completion_ms=(end-submitted)*1000)
    for fn in functions.values():
        for _ in range(2):v,_=invoke(fn);del v
    rng=np.random.default_rng(4102601)
    for _ in range(repeats):
        for name in rng.permutation(list(functions)):
            value,sample=invoke(functions[name]);values=np.asarray(value)
            if not np.isfinite(values).all():raise ValueError('NAX timing became nonfinite')
            h=hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
            old=result[name].setdefault('output_sha256',h)
            if old!=h:raise ValueError('NAX profile timing output changed between repeats')
            result[name]['samples'].append(sample);del value,values
    for row in result.values():row['median']={k:statistics.median(v[k]for v in row['samples'])for k in ('wall_ms','host_ms','completion_ms')}
    return result


def preparation_timing(mx,prepare_fn,observe,*,repeats=3):
    samples=[]
    for _ in range(repeats):
        mx.synchronize();start=time.perf_counter();prepared=prepare_fn();submitted=time.perf_counter()
        mx.eval(*prepared.inputs);end=time.perf_counter();observe()
        samples.append(dict(wall_ms=(end-start)*1000,host_ms=(submitted-start)*1000,completion_ms=(end-submitted)*1000));del prepared
    return dict(samples=samples,median={k:statistics.median(r[k]for r in samples)for k in samples[0]},scope='BF16 operand and mask/index contiguous staging',repeats=repeats)


def run_components(mx,np,capture_dir,out,report,compare,atomic,*,candidate=None,timer=None):
    from ds41_attention_components import capture_module,restored_inputs,raw_details
    from mlx2.runtime.models.deepseek_v41 import selected_attention
    if candidate is None:
        from mlx2.runtime.models import deepseek_v41_omlx_nax_attention as candidate
    repeats=report.get('peer_options',{}).get('repeats',3)
    if type(repeats) is not int or repeats not in (3,7):raise ValueError('Bounded peer repeats must be3 or7')
    if KEY in report:raise ValueError('Refusing to overwrite oMLX NAX attention evidence')
    if mx.default_device()!=mx.gpu:raise ValueError('Caller-owned GPU required')
    parser,path=capture_module(capture_dir);manifest,cases=parser.load_capture(path)
    if len(cases)!=6 or {(c['layer'],c['phase'])for c,_ in cases}!={(l,p)for l in (2,20,21)for p in ('pp99','decode1')}:
        raise ValueError('oMLX attention requires six sealed actual capture cases')
    if any(report.get(k)!=manifest[k]for k in ('checkpoint','source_revision')):raise ValueError('oMLX attention capture identity mismatch')
    section=report[KEY]=dict(schema='mlx2.omlx-nax-attention-components.v1',checkpoint=manifest['checkpoint'],
        capture_source_snapshot_sha256=manifest['source_snapshot_sha256'],component_source_snapshot_sha256=report.get('source_snapshot_sha256'),
        capture_path=str(path),capture_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),numerical_profile=PROFILE,
        complete=False,harness_completed=False,required_reference_controls_exact=False,cases=[],maximum_allocator_bytes=LIMIT,
        candidate_used_by_model=False,model_quality_assessed=False,qualified=False,performance_qualified=False,
        scope='Sealed actual Source PP99/width1 operands; candidate PP only, explicit different numerical law; no state/cache mutation',
        timing_scope='Raw selected-attention core and prepare+core; owning BF16 boundary evaluated for error but excluded from both timed raw-core paths')
    out=Path(out)
    def write():atomic(out/'results.json',report)
    def observe():
        peak=int(mx.get_peak_memory());section['peak_mlx_bytes']=max(section.get('peak_mlx_bytes',0),peak)
        report['component_peak_mlx_bytes']=max(report.get('component_peak_mlx_bytes',0),peak)
        if peak>LIMIT:write();raise MemoryError('oMLX attention exceeded6GiB')
    write();observe()
    for case,arrays in cases:
        args,gathered,layout=restored_inputs(mx,np,case,arrays)
        def reference():return selected_attention(*args,gathered=gathered)
        ref=reference();rb=ref.astype(mx.bfloat16).astype(mx.float32);mx.eval(ref,rb);observe()
        controls=dict(raw=raw_details(np,arrays['reference_raw'],np.asarray(ref)),boundary=raw_details(np,arrays['reference_bf16'],np.asarray(rb)))
        row=dict(layer=case['layer'],phase=case['phase'],length=case['length'],frontier=case['frontier'],controls=controls,restored_layout=layout)
        section['cases'].append(row);write()
        if not all(v['exact']for v in controls.values()):raise ValueError('Required Source attention replay changed')
        if case['phase']=='decode1':
            row.update(candidate_dispatched=False,reason='PP-only minimum16 rows; width-one remains reference',timing=None);write();continue
        def broadcast():return selected_attention(*args,trusted_broadcast=True)
        broadcast_raw=broadcast();broadcast_boundary=broadcast_raw.astype(mx.bfloat16).astype(mx.float32);mx.eval(broadcast_raw,broadcast_boundary)
        row['broadcast_controls']=dict(raw=raw_details(np,np.asarray(ref),np.asarray(broadcast_raw)),boundary=raw_details(np,np.asarray(rb),np.asarray(broadcast_boundary)))
        if not all(v['exact']for v in row['broadcast_controls'].values()):raise ValueError('Required optimized broadcast replay changed')
        prepared=candidate.prepare(*args,numerical_profile=PROFILE);mx.eval(*prepared.inputs)
        row['BF16_operand_roundtrip']={k:raw_details(np,arrays[k][0,0]if k=='local'else arrays[k][0],np.asarray(prepared.inputs[j].astype(mx.float32)))for j,k in enumerate(('q','local','pooled'))}
        actual=candidate.prepared_attention(prepared,numerical_profile=PROFILE);rounded=actual.astype(mx.bfloat16).astype(mx.float32);mx.eval(actual,rounded);observe()
        ar=np.asarray(actual);ab=np.asarray(rounded);row.update(candidate_dispatched=True,raw=raw_details(np,np.asarray(ref),ar),boundary=raw_details(np,np.asarray(rb),ab),finite=bool(np.isfinite(ar).all()and np.isfinite(ab).all()),geometry=prepared.geometry)
        started=time.perf_counter();truth=fp64_reference(np,arrays);elapsed=time.perf_counter()-started
        row['fp64_reference']=dict(scope='Original F32 inputs promoted to double; all query rows/heads, one query at a time, BLAS threads1',compute_seconds=elapsed,shape=list(truth.shape),sha256=hashlib.sha256(truth.tobytes()).hexdigest())
        row['error_vs_fp64']={k:fp64_error(np,truth,v)for k,v in [('source_raw',np.asarray(ref)),('native_raw',ar),('source_BF16',np.asarray(rb)),('native_BF16',ab)]}
        row['upstream_error_bar_observation']={k:row['error_vs_fp64']['native_'+k]['RMS_absolute_error']<=row['error_vs_fp64']['source_'+k]['RMS_absolute_error']for k in ('raw','BF16')}if row['finite']else None
        if row['finite']:
            make=lambda:candidate.prepare(*args,numerical_profile=PROFILE)
            row['preparation_timing']=preparation_timing(mx,make,observe,repeats=repeats)
            row['timing']=(timer or time_different_profile)(mx,np,{'source_reference':reference,'source_broadcast':broadcast,
                'prepared_nax_core':lambda:candidate.prepared_attention(prepared,numerical_profile=PROFILE),
                'prepare_plus_nax_core':lambda:candidate.attention(*args,numerical_profile=PROFILE)},repeats=repeats)
        else:row['timing']=None
        write();del args,gathered,ref,rb,prepared,actual,rounded,ar,ab,truth;observe()
    selected=[r for r in section['cases']if r['candidate_dispatched']]
    section.update(complete=len(section['cases'])==6 and len(selected)==3,harness_completed=True,
                   required_reference_controls_exact=True,all_candidate_finite=all(r['finite']for r in selected),
                   all_candidate_raw_exact=all(r['raw']['exact']for r in selected),all_candidate_BF16_exact=all(r['boundary']['exact']for r in selected))
    section['verdicts']=dict(all3_PP_operator_finite=section['all_candidate_finite'],
        all3_PP_raw_exact=section['all_candidate_raw_exact'],all3_PP_BF16_exact=section['all_candidate_BF16_exact'],
        all3_PP_BF16_operand_roundtrips_exact=all(all(v['exact']for v in r['BF16_operand_roundtrip'].values())for r in selected),
        all3_optimized_broadcast_controls_exact=all(all(v['exact']for v in r['broadcast_controls'].values())for r in selected),
        decode_candidate_not_run=True,quality_evaluated=False)
    write();return section
