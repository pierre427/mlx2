"""Actual attention operands, distinct numerical profiles; caller owns GPU.

No checkpoint weights/state modifications and no GPU work on import. Stock
capture controls must be raw-exact; finite alternate arithmetic is diagnosed
and timed without misrepresenting bit equality or model-quality acceptance.
"""
import hashlib
from pathlib import Path

LIMIT=6<<30
TIMING_VARIANTS=('h1_s1','h4_s1','h4_s4')


def component_options(report):
    value=report.get('head_group_options',{})
    if not isinstance(value,dict)or set(value)-{'repeats','timing_variants'}:raise ValueError('Unknown head-group options')
    repeats=value.get('repeats',3);names=value.get('timing_variants',list(TIMING_VARIANTS))
    if type(repeats)is not int or not 3<=repeats<=7:raise ValueError('Head-group repetitions must be integer3..7')
    if (not isinstance(names,list)or not names or any(type(n)is not str for n in names)
            or len(set(names))!=len(names)or any(n not in TIMING_VARIANTS for n in names)):
        raise ValueError('Invalid head-group timing subset')
    return repeats,names


def numerical_metrics(np,reference,actual):
    from ds41_attention_components import raw_details
    result=raw_details(np,reference,actual)
    result['all_finite']=bool(np.isfinite(actual).all())
    if result['all_finite']and np.isfinite(reference).all():
        delta=actual.astype(np.float64)-reference.astype(np.float64)
        result['rmse']=float(np.sqrt(np.mean(delta*delta)))
        result['relative_L2']=float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(reference.astype(np.float64).ravel()),1e-30))
        result['mean_abs']=float(np.mean(np.abs(delta)))
    return result


def synthetic_stability(np):
    rng=np.random.default_rng(410126)
    q=np.zeros((1,64,1,512),np.float32);local=rng.normal(0,.3,(1,1,4,512)).astype(np.float32)
    pool=rng.normal(0,.3,(1,4,512)).astype(np.float32);ids=np.arange(4,dtype=np.int32)[None,None]
    lm=np.zeros((1,1,1,4),bool);valid=np.zeros((1,1,4),bool);sinks=np.linspace(-1000,1000,64,dtype=np.float32)
    cases=[('all_masked_empty_chunks_large_sinks',[q.copy(),local.copy(),lm.copy(),pool.copy(),ids.copy(),valid.copy(),sinks.copy()])]
    one=lm.copy();one[...,0]=True
    cases.append(('one_key_competing_sink',[q.copy(),local.copy(),one,pool.copy(),ids.copy(),valid.copy(),sinks.copy()]))
    huge=q.copy();huge[0,:,0,0]=np.where(np.arange(64)%2,2e4,-2e4).astype(np.float32)
    extremes=local.copy();extremes[0,0,0,0]=2e4;extremes[0,0,1,0]=-2e4
    two=lm.copy();two[...,:2]=True
    cases.append(('large_opposite_scores',[huge,extremes,two,pool.copy(),ids.copy(),valid.copy(),np.zeros(64,np.float32)]))
    return cases


def run_components(mx,np,captureDir,out,report,compare,atomic):
    if mx.default_device()!=mx.gpu:raise RuntimeError('Head-group components require caller-owned GPU')
    if 'head_group_attention'in report:raise ValueError('Refusing to overwrite head-group evidence')
    repeats,timing_variants=component_options(report)
    from ds41_attention_components import capture_module,restored_inputs,raw_details,timed_interleaved
    from mlx2.runtime.models.deepseek_v41 import selected_attention
    from mlx2.runtime.models.deepseek_v41_fused import rounded
    from mlx2.runtime.models import deepseek_v41_head_group_attention as candidate
    parser,path=capture_module(captureDir);manifest,cases=parser.load_capture(path)
    if len(cases)!=6 or {(c['layer'],c['phase'])for c,a in cases}!={(l,p)for l in (2,20,21)for p in ('pp99','decode1')}:
        raise ValueError('Head-group replay requires six actual cases')
    for key in ('checkpoint','source_revision'):
        if report.get(key)!=manifest[key]:raise ValueError('Head-group capture identity mismatch')
    section=report['head_group_attention']=dict(schema='mlx2.ds41-head-group-attention.v1',
        checkpoint=manifest['checkpoint'],source_revision=manifest['source_revision'],
        capture_source_snapshot_sha256=manifest['source_snapshot_sha256'],component_source_snapshot_sha256=report.get('source_snapshot_sha256'),
        capture_path=str(path),capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        numerical_profile='source_manual_f32_online_attention_v1',arithmetic='F32componentprescale/manualdot/SIMDsum/chronologicalonline-softmax; split4stablepartialmerge',
        stock_reduction_law=False,quality_verdict='not evaluated',qualified=False,performance_qualified=False,candidate_used_by_model=False,
        original_weight_quantization_unchanged=True,cache_layout_unchanged=True,maximum_allocator_bytes=LIMIT,
        scope='Actual6case component operands plus separatelylabelled syntheticstability; no modelquality/inference-rate claim',
        repetitions=repeats,timing_variants=timing_variants,
        timing_scope=f'{repeats}interleavedoperatorreps afterexactstockcontrols and finite/repeatstability; all3numericalchecks retained; candidate bitparity nottiminggate',
        cases=[],synthetic_stability=[],complete=False,harness_completed=False,required_reference_controls_exact=False,verdicts={})
    out=Path(out)
    def write():atomic(out/'results.json',report)
    def observe():
        peak=int(mx.get_peak_memory());section['peak_mlx_bytes']=max(section.get('peak_mlx_bytes',0),peak)
        report['component_peak_mlx_bytes']=max(report.get('component_peak_mlx_bytes',0),peak)
        if peak>LIMIT:write();raise MemoryError('Head-group component exceeds6GiB')
    write();observe()
    # Establish every captured stock control before timing any profile.
    for case,arrays in cases:
        if (case['length']!=(99 if case['phase']=='pp99'else 1)
                or case['frontier']!=(0 if case['phase']=='pp99'else 100)):
            raise ValueError('Head-group capture phase/frontier mismatch')
        args,gathered,layout=restored_inputs(mx,np,case,arrays)
        raw=selected_attention(*args,gathered=gathered);boundary=rounded(raw);mx.eval(raw,boundary)
        row=dict(capture=case,input_layout=layout,reference_controls=dict(raw=raw_details(np,arrays['reference_raw'],np.array(raw)),
            BF16=raw_details(np,arrays['reference_bf16'],np.array(boundary))),variants={})
        section['cases'].append(row);write()
        if not all(v['exact']for v in row['reference_controls'].values()):raise ValueError('Head-group stockcapture replay must beexact')
        mx.synchronize();del raw,boundary,args;observe();mx.clear_cache()
    section['required_reference_controls_exact']=True;write()
    # Independent stability cases gate operator validity, not model quality.
    stable={f'h{h}_s{s}':True for h,s in candidate.VARIANTS}
    for name,host in synthetic_stability(np):
        expected=candidate.reference_numpy(*host);inputs=[mx.array(x)for x in host];mx.eval(*inputs)
        row=dict(name=name,input_scope='deterministic synthetic stress, not modeltrajectory',variants={})
        section['synthetic_stability'].append(row)
        for h,s in candidate.VARIANTS:
            value=candidate.attention(*inputs,head_group=h,kv_splits=s);mx.eval(value);actual=np.array(value)
            check=numerical_metrics(np,expected,actual);accepted=check['all_finite']and bool(np.allclose(expected,actual,rtol=1e-4,atol=1e-5))
            row['variants'][f'h{h}_s{s}']=dict(check=check,operator_stability_passed=accepted,tolerance_scope='syntheticF64oracle sanityonly; notactualqualitythreshold')
            stable[f'h{h}_s{s}']&=accepted;del value;observe();write()
        mx.synchronize();del inputs;mx.clear_cache()
    for index,(case,arrays)in enumerate(cases):
        args,gathered,layout=restored_inputs(mx,np,case,arrays)
        def reference():return selected_attention(*args,gathered=gathered)
        row=section['cases'][index]
        functions={'reference':reference};host_outputs={}
        for h,s in candidate.VARIANTS:
            name=f'h{h}_s{s}';value,partials=candidate.attention(*args,head_group=h,kv_splits=s,return_partials=True)
            second=candidate.attention(*args,head_group=h,kv_splits=s);owner=rounded(value);mx.eval(value,second,owner)
            actual=np.array(value);owner_host=np.array(owner);repeat=raw_details(np,actual,np.array(second))
            checks=dict(raw=numerical_metrics(np,arrays['reference_raw'],actual),BF16=numerical_metrics(np,arrays['reference_bf16'],owner_host),repeat_raw=repeat)
            p=candidate.plan(case['length'],arrays['local'].shape[2],arrays['pooled'].shape[1],arrays['indices'].shape[-1],head_group=h,kv_splits=s)
            visible=int(np.count_nonzero(arrays['local_mask'])+np.count_nonzero(arrays['valid']))
            p['logical_visible_KV_load_bytes']=visible*(64//h)*512*4
            admitted=checks['raw']['all_finite']and checks['BF16']['all_finite']and repeat['exact']and stable[name]
            item=dict(checks=checks,plan=p,synthetic_stability_passed=stable[name],timing_admitted=admitted,
                observed_scope='bounded numericalprofile component only',quality_verdict='not evaluated')
            row['variants'][name]=item
            if partials is not None:
                mx.eval(partials);host=np.array(partials);m,l=host[...,0],host[...,1]
                valid_stats=np.isfinite(l)&(l>=0)&np.where(l>0,np.isfinite(m),np.isneginf(m))
                item['partial_stats']=dict(shape=list(host.shape),empty_partial_count=int(np.count_nonzero(l==0)),
                    all_stats_valid=bool(valid_stats.all()),PV_finite=bool(np.isfinite(host[...,2:]).all()),sink_in_partials=False)
                admitted&=item['partial_stats']['all_stats_valid']and item['partial_stats']['PV_finite'];item['timing_admitted']=admitted
                del host
            item['timing_selected']=name in timing_variants
            if admitted and name in timing_variants:functions[name]=lambda h=h,s=s:candidate.attention(*args,head_group=h,kv_splits=s)
            host_outputs[name]=actual;del value,second,owner,partials;observe();write()
        row['head_group_only_raw_consistency']=raw_details(np,host_outputs['h1_s1'],host_outputs['h4_s1'])
        row['split4_raw_consistency']=raw_details(np,host_outputs['h4_s1'],host_outputs['h4_s4'])
        del host_outputs
        row['timings']=timed_interleaved(mx,np,functions,repeats=repeats);observe();write()
        mx.synchronize();del args,functions;mx.clear_cache()
    section['required_reference_controls_exact']=all(all(v['exact']for v in row['reference_controls'].values())for row in section['cases'])
    section['verdicts']={name+'_all6_operator_valid':all(row['variants'][name]['timing_admitted']for row in section['cases'])for name in stable}
    section['complete']=section['harness_completed']=True;write();return section
