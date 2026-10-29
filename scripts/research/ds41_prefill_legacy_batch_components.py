"""Derived actual-activation grouped legacy tile oracle; no GPU on import."""
import hashlib
import json
from pathlib import Path

LIMIT=6<<30
BANDS=((5,1,(13,16,32)),(2,2,(33,34,64)),(2,3,(65,96)))

def case_plans():
    result=[]
    for partitions,tiles,widths in BANDS:
        counts=[(m,)for m in widths]+[tuple(widths[i%len(widths)]for i in range(e))for e in (3,15)]
        for count in counts:
            for projection in ('gate','up'):
                offsets=[0]
                for m in count:offsets.append(offsets[-1]+m)
                result.append(dict(partitions=partitions,row_tiles=tiles,row_counts=count,row_offsets=offsets,
                    projection=projection,expert_count=len(count),input_shape=(sum(count),5120),
                    output_shape=(sum(count),2304),partial_shape=(partitions,sum(count),2304),
                    logical_shape_rule='Each expert keeps its whole local M; sumM is only arena storage stride'))
    return result

def derived_rows(np,activation,rows,offset=0):
    if activation.dtype!=np.float32 or activation.ndim!=2 or activation.shape[1]!=5120 or len(activation)==0:
        raise ValueError('Derived activation shape')
    return np.ascontiguousarray(activation[(np.arange(rows)+offset)%len(activation)])

def original_ids(capture_path,cases):
    unit=Path(capture_path).resolve().parent
    policy=json.loads((unit/'policy.json').read_text())['deepseek_source_experts']
    result=json.loads((unit/'results.json').read_text())
    if result.get('passed') is not True or result.get('cleanup_errors') != [] or result.get('state_discarded') is not True:
        raise ValueError('Capture unit not completed with cleanup')
    if any(c['layer']!=0 for c,_ in cases):raise ValueError('Batch fixture requires captured layer0 coefficients')
    raw=Path(policy['hotset_path']).read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if digest!=result['expert_initial']['hot_map_binding']:raise ValueError('Capture hotmap changed')
    hot=json.loads(raw)
    if hot['source']!=result['expert_initial']['source_binding'] or hot['hot_budget_bytes']!=60_000_000_000:
        raise ValueError('Capture hotmap source/pool mismatch')
    available=sorted({expert for layer,expert in hot['experts']if layer==0})
    captured=sorted({c['expert']for c,_ in cases if c['layer']==0})
    if len(captured)!=2 or any(e not in available for e in captured):raise ValueError('Expected two captured layer0 original experts')
    ids=(captured+[e for e in available if e not in captured])[:15]
    if len(ids)!=15:raise ValueError('Need15 distinct actual original experts')
    return ids,digest

def run_legacy_batch(mx,np,capture_path,load_expert,out,report,compare,atomic,*,single=None,batch=None):
    from ds41_prefill_legacy_gu_components import capture_module,check_coefficients
    from ds41_prefill_wide_components import raw_details
    if 'prefill_legacy_batch'in report:raise ValueError('Refusing to overwrite legacy batch evidence')
    if mx.default_device()!=mx.gpu:raise RuntimeError('Caller-owned GPU required')
    if single is None or batch is None:
        from mlx2.runtime.gguf.source_prefill_legacy_tiles import project,batch_project
        single=single or project;batch=batch or batch_project
    parser=capture_module(capture_path);manifest,captured=parser.load_capture(capture_path)
    if any(report.get(k)!=manifest[k]for k in ('checkpoint','source_revision')):raise ValueError('Batch/capture identity mismatch')
    ids,hotsha=original_ids(capture_path,captured)
    out=Path(out);plans=case_plans()
    section=report['prefill_legacy_batch']=dict(schema='mlx2.source-legacy-batch-components.v1',
        checkpoint=manifest['checkpoint'],capture_path=str(Path(capture_path).resolve()),
        capture_sha256=hashlib.sha256(Path(capture_path).read_bytes()).hexdigest(),
        capture_source_snapshot_sha256=manifest['source_snapshot_sha256'],component_source_snapshot_sha256=report.get('source_snapshot_sha256'),
        scope='Actual-activation-derived operator cases: cropped/repeated captured rows, independently selected actual original layer0 coefficients; modified rows/other experts are NOT a whole-model trajectory',
        hotmap_sha256=hotsha,layer=0,expert_ids=ids,maximum_allocator_bytes=LIMIT,
        qualification=False,performance_qualified=False,candidate_used_by_model=False,cases=[],planned_geometry=plans,
        coefficient_geometry={'qweight':[2304,640],'qweight_dtype':'uint32','scales':[2304,160],'scales_dtype':'uint8'},
        timing_scope='Prepared activation/coefficients;3 interleaved repeats only after all raw/boundary/partial exact gates',
        stock_internal_partial_equality_claimed=False)
    def write():atomic(out/'results.json',report)
    def observe():
        peak=int(mx.get_peak_memory());section['peak_mlx_bytes']=peak
        report['component_peak_mlx_bytes']=max(report.get('component_peak_mlx_bytes',0),peak)
        if peak>LIMIT:write();raise MemoryError('Legacy batch exceeded6GiB allocator ceiling')
    observe();write()
    bank={}
    for expert in ids:
        bank[expert]=load_expert(0,expert);observe()
    # Preserve all four original capture coefficient bindings before deriving rows.
    for c,arrays in captured:check_coefficients(np,c,bank[c['expert']])
    section['four_capture_coefficient_checks']=True
    section['coefficient_sha256']={}
    for expert,data in bank.items():
        if 'gate.biases'in data:raise ValueError('Batch bank contains Q1 expert')
        record={}
        for projection in ('gate','up'):
            q,s=np.array(data[projection+'.qweight']),np.array(data[projection+'.scales'])
            if q.dtype!=np.uint32 or q.shape!=(2304,640) or s.dtype!=np.uint8 or s.shape!=(2304,160):raise ValueError('Actual original coefficient shape')
            record[projection]={'qweight':hashlib.sha256(q.tobytes()).hexdigest(),'scales':hashlib.sha256(s.tobytes()).hexdigest()}
        section['coefficient_sha256'][str(expert)]=record
    bases={(c['partitions'],c['projection']):(c,a['activation'])for c,a in captured}
    write()
    for index,plan in enumerate(plans):
        projection=plan['projection'];counts=plan['row_counts'];experts=[bank[e]for e in ids[:len(counts)]]
        origin,base=bases[(plan['partitions'],projection)]
        rows=[derived_rows(np,base,m,j)for j,m in enumerate(counts)]
        values=np.concatenate(rows,axis=0);x=mx.array(values,mx.float32)
        slices=[x[a:b]for a,b in zip(plan['row_offsets'][:-1],plan['row_offsets'][1:])]
        def stock_raw():
            return mx.concatenate([mx.quantized_matmul(v,e[projection+'.qweight'],scales=e[projection+'.scales'],mode='mxfp4')for v,e in zip(slices,experts)],axis=0)
        def stock_boundary():return stock_raw().astype(mx.bfloat16).astype(mx.float32)
        def singles_boundary():return mx.concatenate([single(v,e,projection=projection)for v,e in zip(slices,experts)],axis=0)
        expected_raw=stock_raw();expected_boundary=expected_raw.astype(mx.bfloat16).astype(mx.float32)
        singles=[single(v,e,projection=projection,return_partials=True)for v,e in zip(slices,experts)]
        single_boundary=mx.concatenate([t[0]for t in singles],axis=0)
        single_raw=mx.concatenate([t[1]for t in singles],axis=0)
        single_parts=mx.concatenate([t[2]for t in singles],axis=1)
        batched_boundary,batched_raw,batched_parts=batch(x,experts,counts,projection=projection,return_partials=True)
        outputs=[expected_raw,expected_boundary,single_boundary,single_raw,single_parts,batched_boundary,batched_raw,batched_parts]
        mx.eval(*outputs);observe()
        er,eb,sb,sr,sp,bb,br,bp=[np.array(v)for v in outputs]
        if bp.shape!=tuple(plan['partial_shape'])or sp.shape!=bp.shape:raise ValueError('Batch partial arena shape')
        checks=dict(single_raw_vs_stock=raw_details(np,er,sr),single_boundary_vs_stock=raw_details(np,eb,sb),
                    batch_raw_vs_stock=raw_details(np,er,br),batch_boundary_vs_stock=raw_details(np,eb,bb),
                    batch_raw_vs_single=raw_details(np,sr,br),batch_partials_vs_single=raw_details(np,sp,bp))
        row={**plan,'expert_ids':ids[:len(counts)],'activation_origin':origin,'activation_derivation':'row[(arange(M)+job_index)%captured_M], then concatenate expert segments',
             'gathered_activation_sha256':hashlib.sha256(values.tobytes()).hexdigest(),'checks':checks,'exact':all(c['exact']for c in checks.values())}
        section['cases'].append(row);write()
        if not row['exact']:
            path=out/f'legacy-batch-mismatch-{index:02d}.npz'
            with path.open('xb')as f:np.savez(f,activation=values,stock_raw=er,stock_boundary=eb,single_raw=sr,single_partials=sp,batch_raw=br,batch_boundary=bb,batch_partials=bp)
            row['mismatch_arrays']={'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        row['timing']=compare(mx,np,{'reference':stock_boundary,'single_tiles':singles_boundary,'batch_tiles':lambda:batch(x,experts,counts,projection=projection)},repeats=3)if row['exact']else None
        mx.synchronize()
        x=None;slices=None;del rows,values,outputs,singles,expected_raw,expected_boundary,single_boundary,single_raw,single_parts,batched_boundary,batched_raw,batched_parts,er,eb,sb,sr,sp,bb,br,bp
        observe();write()
    section['all_raw_boundary_partial_exact']=len(section['cases'])==28 and all(c['exact']for c in section['cases'])
    section['complete']=True;write()
    return section
