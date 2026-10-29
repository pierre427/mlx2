"""Unwired unchanged-FP8/F32 kernel tuning on sealed actual operands.

Caller owns GPU admission. Optional raw/store failures remain negative evidence;
only fully gated paths are timed. No work, weight load or MLX import on import.
"""
import hashlib
from pathlib import Path

LIMIT=6<<30


def configurations(rows):
    if rows==1:
        return[(f's{sg}_r{r}',dict(simdgroups=sg,results_per_simd=r))for sg in (2,4,8)for r in (4,8)]
    if rows==99:
        return[(f'm{bm}_n{bn}',dict(tile=(bm,bn)))for bm,bn in ((32,32),(64,32),(32,64),(16,64))]
    raise ValueError('FP8 kernel components require capturedM1/M99')


def component_options(report):
    """Keep r1 default inventory; allow a separately sealed bounded followup."""
    value=report.get('fp8_projection_options',{})
    if not isinstance(value,dict)or set(value)-{'repeats','pp_variants','decode_variants'}:
        raise ValueError('Unknown FP8 component options')
    repeats=value.get('repeats',3)
    if type(repeats)is not int or not 1<=repeats<=15:
        raise ValueError('FP8 component repeats must be integer1..15')
    selected={}
    for rows,key in ((1,'decode_variants'),(99,'pp_variants')):
        available=dict(configurations(rows));names=value.get(key,list(available))
        if (not isinstance(names,list)or not names or any(type(n)is not str for n in names)
                or len(set(names))!=len(names)or any(n not in available for n in names)):
            raise ValueError('Invalid FP8 component configuration subset')
        selected[rows]=[(n,available[n])for n in names]
    return repeats,selected


def run_components(mx,np,captureDir,out,report,compare,atomic):
    if mx.default_device()!=mx.gpu:raise RuntimeError('FP8 kernel components require caller-owned GPU')
    if 'fp8_projection_kernels'in report:raise ValueError('Refusing to overwrite FP8 kernel evidence')
    repeats,selected=component_options(report)
    from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact,read_tensors
    from mlx2.runtime.gguf import source_fp8_projection_tiles as kernels
    from ds41_attention_projection_components import capture_module,coefficient_payload,verify_coefficients,upload_coefficients,_boundary
    from ds41_attention_components import raw_details
    parser,path,capture_spec=capture_module(captureDir);manifest,cases=parser.load_capture(path)
    for key in ('checkpoint','source_revision'):
        if report.get(key)!=manifest[key]:raise ValueError('FP8 kernel capture/component identity mismatch')
    expected={(l,p,k)for l in (2,20,21)for p in ('pp99','decode1')for k in ('q_b','wo_b')}
    if len(cases)!=12 or {(c['layer'],c['phase'],c['projection'])for c,a in cases}!=expected:
        raise ValueError('FP8 kernel components require12actualcases')
    model_path=capture_spec.get('model')or report.get('model')
    if not model_path:raise ValueError('FP8 capturedartifactpathmissing')
    if report.get('model')and Path(report['model']).expanduser().resolve()!=Path(model_path).expanduser().resolve():
        raise ValueError('FP8 Source artifact path mismatch')
    artifact=inspect_artifact(model_path)
    for identity in (report,capture_spec):
        if identity.get('original_fingerprint',artifact['identity']['fingerprint'])!=artifact['identity']['fingerprint']:
            raise ValueError('FP8 Source fingerprint mismatch')
    boundary,boundary_scope=_boundary(mx,capture_spec);out=Path(out)
    section=report['fp8_projection_kernels']=dict(schema='mlx2.ds41-fp8-projection-kernels.v1',
        checkpoint=manifest['checkpoint'],source_revision=manifest['source_revision'],
        capture_source_snapshot_sha256=manifest['source_snapshot_sha256'],component_source_snapshot_sha256=report.get('source_snapshot_sha256'),
        capture_path=str(path),capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        original_artifact_fingerprint=artifact['identity']['fingerprint'],boundary_scope=boundary_scope,
        maximum_allocator_bytes=LIMIT,maximum_threadgroup_scratch_bytes=13824,shared_memory_limit_assumed_bytes=32768,
        qualified=False,performance_qualified=False,candidate_used_by_model=False,all_dense_provider_constructed=False,
        arithmetic='preparedF32+unchangedMXFP8codes;stockhelperdotlaws;ownerBF16onlyaftercompletedF32dot',
        dispatch_scope='Hostbinaryinferredstockfamilies only; profilerlabel evidence mustcome from parentMetaltrace',
        repetitions=repeats,configuration_subset={str(rows):[n for n,o in choices]for rows,choices in selected.items()},
        timing_scope=f'{repeats}interleavedreps onlygatedpaths;compare rawkernel+owninground vsfusedstore toseparategeometry/fusion',
        cases=[],loaded_coefficients=[],complete=False,harness_completed=False,required_reference_controls_exact=False,verdicts={})
    def write():atomic(out/'results.json',report)
    def observe():
        peak=int(mx.get_peak_memory());section['peak_mlx_bytes']=max(section.get('peak_mlx_bytes',0),peak)
        report['component_peak_mlx_bytes']=max(report.get('component_peak_mlx_bytes',0),peak)
        if peak>LIMIT:write();raise MemoryError('FP8 kernel components exceeded6GiB')
    write();observe();bank={}
    for case,arrays in cases:
        if not case['input_bf16_audit']['exact']or arrays['prepared_input'].tobytes()!=arrays['input_bf16_roundtrip'].tobytes():
            write();raise ValueError('FP8 prepared Source capture lost its BF16-grid input contract')
        alias=case['linear_name']
        if alias not in bank:
            q,s,load=coefficient_payload(np,artifact,alias,read_tensors);verify_coefficients(load,case)
            q_gpu,s_gpu=upload_coefficients(mx,q,s);bank[alias]=(q_gpu,s_gpu,load)
            section['loaded_coefficients'].append(load);del q,s
        q,s,load=bank[alias];verify_coefficients(load,case)
        x=mx.array(arrays['prepared_input'],mx.float32);mx.eval(x)
        def stock_raw():return mx.quantized_matmul(x,q,scales=s,mode='mxfp8')
        def stock_boundary():return boundary(stock_raw())
        raw=stock_raw();rounded=boundary(raw);mx.eval(raw,rounded)
        row=dict(capture=case,stock_controls=dict(raw=raw_details(np,arrays['reference_raw'],np.array(raw)),
            BF16=raw_details(np,arrays['reference_bf16'],np.array(rounded))),variants={})
        section['cases'].append(row);write()
        if not all(v['exact']for v in row['stock_controls'].values()):
            raise ValueError('FP8 stock replay didnotmatchsavedraw/BF16')
        functions={'reference':stock_boundary}
        for name,options in selected[case['length']]:
            def candidate(fused,options=options):
                return kernels.project(x,q,s,case['projection'],fused_boundary=fused,**options)
            candidate_raw=candidate(False);raw_plus_owner=boundary(candidate_raw);fused=candidate(True)
            mx.eval(candidate_raw,raw_plus_owner,fused)
            checks=dict(raw=raw_details(np,arrays['reference_raw'],np.array(candidate_raw)),
                raw_then_owner=raw_details(np,arrays['reference_bf16'],np.array(raw_plus_owner)),
                fused_owner=raw_details(np,arrays['reference_bf16'],np.array(fused)))
            raw_exact=checks['raw']['exact']and checks['raw_then_owner']['exact']
            fused_exact=raw_exact and checks['fused_owner']['exact']
            row['variants'][name]=dict(options=options,checks=checks,raw_pipeline_admitted=raw_exact,fused_pipeline_admitted=fused_exact,
                plan=kernels.plan(case['projection'],case['length'],**options),observed_scope='boundedoperatoronly')
            if raw_exact:functions[name+'_raw_owner']=lambda options=options:boundary(kernels.project(x,q,s,case['projection'],fused_boundary=False,**options))
            if fused_exact:functions[name+'_fused']=lambda options=options:kernels.project(x,q,s,case['projection'],fused_boundary=True,**options)
            del candidate_raw,raw_plus_owner,fused;observe();write()
        del raw,rounded
        row['timings']=compare(mx,np,functions,repeats=repeats)
        observe();write();mx.synchronize();del x,functions;mx.clear_cache()
    section['required_reference_controls_exact']=all(all(v['exact']for v in row['stock_controls'].values())for row in section['cases'])
    names={name for row in section['cases']for name in row['variants']}
    section['verdicts']={name+'_all_applicable_raw_and_fused_exact':all(row['variants'][name]['fused_pipeline_admitted']for row in section['cases']if name in row['variants'])for name in sorted(names)}
    section['loaded_alias_count']=len(bank);section['uploaded_coefficient_bytes']=sum(r['uploaded_coefficient_bytes']for r in section['loaded_coefficients'])
    section['complete']=section['harness_completed']=True
    write();return section
