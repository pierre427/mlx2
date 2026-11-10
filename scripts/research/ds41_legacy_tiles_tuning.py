"""Bounded optional tile/consumer diagnostics; parent owns GPU, no GPU on import."""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
import hashlib
from pathlib import Path

LIMIT=6<<30
TILES={'default32':(32,32),'alt16':(16,64)}

def case_inventory():
    bands=((13,16,32),(33,34,64),(65,96,65));result=[]
    for counts in bands:
        for experts in (3,15):
            result.append(dict(kind='derived_projection',row_counts=tuple(counts[i%3]for i in range(experts)),projection='gate'))
    for counts in ((13,16,32),tuple((13,16,32)[i%3]for i in range(15)),(33,34,64),(65,96,65)):
        result.append(dict(kind='derived_middle',row_counts=counts))
    return result

def score_inputs(np,rows):
    size=2*rows+17
    if size>6144:raise ValueError('Synthetic score vector exceeds consumer bound')
    scores=(np.float32(.125)+(np.arange(size)%31).astype(np.float32)*np.float32(.03125)).astype(np.float32)
    slots=((np.arange(rows)[::-1]*3+5)%size).astype(np.uint32)
    return scores,slots

def admitted_timings(mx,np,reference,functions,checks,compare):
    admitted={'reference':reference}
    admitted.update({k:fn for k,fn in functions.items()if checks[k]['exact']})
    timings=compare(mx,np,admitted,repeats=3)
    return {k:timings.get(k)for k in functions},timings['reference']

def run_tuning(mx,np,capture_path,load_expert,out,report,compare,atomic,*,single=None,batch=None,consumer=None,middle=None,spelling=None):
    from ds41_prefill_legacy_gu_components import capture_module,check_coefficients
    from ds41_prefill_legacy_batch_components import original_ids,derived_rows
    from ds41_prefill_wide_components import raw_details
    if 'legacy_tile_tuning'in report:raise ValueError('Refusing to overwrite tuning evidence')
    if mx.default_device()!=mx.gpu:raise RuntimeError('Caller-owned GPU required')
    if single is None or batch is None or consumer is None:
        from mlx2.runtime.gguf.source_prefill_legacy_tiles import project,batch_project,batch_middle
        single=single or project;batch=batch or batch_project;consumer=consumer or batch_middle
    if middle is None or spelling is None:
        from mlx2.runtime.gguf.source_middle_fusion import middle as owning_middle,exp_spelling
        middle=middle or owning_middle;spelling=spelling or exp_spelling()
    parser=capture_module(capture_path);manifest,captured=parser.load_capture(capture_path)
    if any(report.get(k)!=manifest[k]for k in ('checkpoint','source_revision')):raise ValueError('Tuning capture identity mismatch')
    ids,hotsha=original_ids(capture_path,captured);out=Path(out);plans=case_inventory()
    section=report['legacy_tile_tuning']=dict(schema='mlx2.source-legacy-tile-tuning.v1',checkpoint=manifest['checkpoint'],
        capture_path=str(Path(capture_path).resolve()),capture_sha256=hashlib.sha256(Path(capture_path).read_bytes()).hexdigest(),
        capture_source_snapshot_sha256=manifest['source_snapshot_sha256'],component_source_snapshot_sha256=report.get('source_snapshot_sha256'),
        maximum_allocator_bytes=LIMIT,hotmap_sha256=hotsha,expert_ids=ids,tiles=TILES,cases=[],expected_case_count=4+len(plans),
        exp_spelling=spelling,qualification=False,performance_qualified=False,candidate_used_by_model=False,
        scope='Four actual captured projections; six modified-row/other-weight operator projections; four middle operators with actual-derived activations and synthetic router scores/slots. No full-model trajectory claim for derived cases.',
        stock_internal_partial_equality_claimed=False,planned_derived_cases=plans,
        timing_scope='Three interleaved repetitions; each variant admitted only after its full raw/boundary/partial or middle gate; optional numerical failures do not imply harness failure')
    def write():atomic(out/'results.json',report)
    def observe():
        peak=int(mx.get_peak_memory());section['peak_mlx_bytes']=peak
        report['component_peak_mlx_bytes']=max(report.get('component_peak_mlx_bytes',0),peak)
        if peak>LIMIT:write();raise MemoryError('Tuning exceeds6GiB allocator cap')
    observe();write();bank={}
    for expert in ids:bank[expert]=load_expert(0,expert);observe()
    for case,arrays in captured:check_coefficients(np,case,bank[case['expert']])
    section['four_capture_coefficient_checks']=True
    section['coefficient_sha256']={}
    section['coefficient_geometry']={'qweight':[2304,640],'scales':[2304,160],'dtypes':['uint32','uint8']}
    for expert,data in bank.items():
        if 'gate.biases'in data:raise ValueError('Tuning requires original coefficients')
        record={}
        for projection in ('gate','up'):
            q,scale=np.array(data[projection+'.qweight']),np.array(data[projection+'.scales'])
            if q.dtype!=np.uint32 or q.shape!=(2304,640) or scale.dtype!=np.uint8 or scale.shape!=(2304,160):raise ValueError('Tuning coefficient geometry')
            record[projection]={'qweight':hashlib.sha256(q.tobytes()).hexdigest(),'scales':hashlib.sha256(scale.tobytes()).hexdigest()}
        section['coefficient_sha256'][str(expert)]=record
    del q,scale
    bases={(c['partitions'],c['projection']):(c,a['activation'])for c,a in captured}
    jobs=[dict(kind='actual_single',capture=c,arrays=a,row_counts=(c['rows'],),projection=c['projection'],expert_ids=[c['expert']])for c,a in captured]+plans
    for index,job in enumerate(jobs):
        counts=job['row_counts'];part=5 if counts[0]<=32 else 2
        experts_ids=job.get('expert_ids',ids[:len(counts)]);experts=[bank[e]for e in experts_ids]
        if job['kind']=='actual_single':values=job['arrays']['activation'];origin=job['capture']
        else:
            origin,base=bases[(part,job.get('projection','gate'))]
            values=np.concatenate([derived_rows(np,base,m,j)for j,m in enumerate(counts)],axis=0)
        offsets=[0]
        for m in counts:offsets.append(offsets[-1]+m)
        x=mx.array(values,mx.float32);slices=[x[a:b]for a,b in zip(offsets[:-1],offsets[1:])]
        row=dict(kind=job['kind'],row_counts=counts,partitions=part,expert_ids=experts_ids,activation_origin=origin,
                 activation_sha256=hashlib.sha256(values.tobytes()).hexdigest(),variants={})
        section['cases'].append(row)
        def stock_raw(projection):
            return mx.concatenate([mx.quantized_matmul(v,e[projection+'.qweight'],scales=e[projection+'.scales'],mode='mxfp4')for v,e in zip(slices,experts)],axis=0)
        if job['kind']!='derived_middle':
            projection=job['projection'];row['projection']=projection
            expected=stock_raw(projection);boundary=expected.astype(mx.bfloat16).astype(mx.float32);mx.eval(expected,boundary)
            expected_np,boundary_np=np.array(expected),np.array(boundary)
            if job['kind']=='actual_single':
                row['capture_stock_replay']={'raw':raw_details(np,job['arrays']['reference_preround'],expected_np),'boundary':raw_details(np,job['arrays']['reference_boundary'],boundary_np)}
                if not all(v['exact']for v in row['capture_stock_replay'].values()):write();raise ValueError('Stock capture replay mismatch')
            results={}
            for name,tile in TILES.items():
                result=(single(x,experts[0],projection=projection,return_partials=True,tile=tile)if job['kind']=='actual_single'
                        else batch(x,experts,counts,projection=projection,return_partials=True,tile=tile))
                mx.eval(*result);results[name]=tuple(np.array(v)for v in result)
                if results[name][2].shape!=(part,sum(counts),2304):raise ValueError('Tuning partial shape')
            functions={}
            for name,tile in TILES.items():
                rounded,raw,partials=results[name]
                checks={'raw_vs_stock':raw_details(np,expected_np,raw),'boundary_vs_stock':raw_details(np,boundary_np,rounded),
                        'partials_vs_default32':raw_details(np,results['default32'][2],partials)}
                if job['kind']!='actual_single':
                    parts=[single(v,e,projection=projection,return_partials=True,tile=tile)[2]for v,e in zip(slices,experts)]
                    joined=mx.concatenate(parts,axis=1);mx.eval(joined)
                    checks['partials_vs_same_tile_singles']=raw_details(np,np.array(joined),partials)
                    del parts,joined
                row['variants'][name]={'checks':checks,'exact':all(c['exact']for c in checks.values())}
                functions[name]=(lambda tile=tile:single(x,experts[0],projection=projection,tile=tile))if job['kind']=='actual_single'else(lambda tile=tile:batch(x,experts,counts,projection=projection,tile=tile))
            reference=lambda:stock_raw(projection).astype(mx.bfloat16).astype(mx.float32)
            del expected,boundary,expected_np,boundary_np,results,rounded,raw,partials,result
        else:
            scores_np,slots_np=score_inputs(np,sum(counts));scores=mx.array(scores_np,mx.float32);slots=mx.array(slots_np,mx.uint32)
            row['router_inputs']={'scope':'Synthetic deterministic F32 router weights and nonidentity U32 slots; not captured model scores',
                'score_shape':list(scores_np.shape),'slots_shape':list(slots_np.shape),'score_sha256':hashlib.sha256(scores_np.tobytes()).hexdigest(),'slots_sha256':hashlib.sha256(slots_np.tobytes()).hexdigest(),
                'minimum_score':float(scores_np.min()),'maximum_score':float(scores_np.max()),'minimum_slot':int(slots_np.min()),'maximum_slot':int(slots_np.max())}
            reference=lambda:middle(stock_raw('gate'),stock_raw('up'),scores,slots)
            expected=reference();mx.eval(expected);expected_np=np.array(expected)
            functions={}
            for name,tile in TILES.items():
                fn=lambda tile=tile:consumer(x,experts,counts,scores,slots,exp_spelling=spelling,tile=tile)
                value=fn();mx.eval(value);check=raw_details(np,expected_np,np.array(value));del value
                row['variants'][name]={'checks':{'middle_vs_stock_GU_owning_epilogue':check},'exact':check['exact']};functions[name]=fn
            del expected,expected_np
        observe();write()
        timings,reference_timing=admitted_timings(mx,np,reference,functions,row['variants'],compare)
        row['reference_timing']=reference_timing
        for name in functions:row['variants'][name]['timing']=timings[name]
        mx.synchronize();x=None;slices=None;del values,functions,reference
        if job['kind']=='derived_middle':del scores,slots,scores_np,slots_np
        observe();write()
    section['verdicts']={f'{name}_{kind}_exact':all(c['variants'][name]['exact']for c in section['cases']if(c['kind']=='derived_middle')==(kind=='middle'))for name in TILES for kind in ('projection','middle')}
    section['required_reference_controls_exact']=section['verdicts']['default32_projection_exact']
    section['harness_completed']=len(section['cases'])==section['expected_case_count'];section['complete']=True
    write();return section
