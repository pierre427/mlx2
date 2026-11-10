# SPDX-License-Identifier: MIT
"""Actual/derived dynamic-M runtime operands and OOB canaries, no work on import.

Parent-only GPU execution. No throughput timing. Native HALF profile remains
separate from stock; all model source/file-seal/ownership controls precede it.
Provenance: provenance/deepseek-v41-native-mx-runtime-smoke.json.
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
import hashlib
import json
from pathlib import Path

LIMIT=6<<30
PADDING=17
SENTINEL=-12345.625
VERDICTS=frozenset(('stock_capture_controls','fresh_stock_controls','coefficient_identity',
    'runtime_eligibility','finite_stable','actual_fp64_bounds','zero_second_aba','tail_guards','all_cases_dispatched'))


def case_matrix(np,ffn_cases,down_cases):
    """Real rows only; derived prefixes and counterfactual membership explicit."""
    full={case['rows']:(case,arrays)for case,arrays in ffn_cases}
    if set(full)!={32,34}:raise ValueError('Runtime smoke needs actual FFN M32/M34')
    result=[]
    def append(case,arrays,projection,rows,origin):
        actual=origin=='actual_full_routed_job'
        x=arrays['middle'] if projection=='down' else arrays['gu_input']
        if origin=='actual_shared_input_counterfactual_expert_membership':x=arrays['shared_gu_input']
        x=np.ascontiguousarray(x[:rows])
        k=2304 if projection=='down'else 5120
        if x.shape!=(rows,k)or x.dtype!=np.float32:raise ValueError('Runtime smoke actual input extent')
        name=('down'if projection=='down'else projection)
        result.append(dict(layer=case['layer'],expert=case['expert'],rows=rows,projection=projection,
            origin=origin,captured_routed_job=actual,input=x,capture_case=case,
            expected_raw=arrays[name+'_raw']if actual else None,
            expected_owner=arrays[name+'_owned']if actual else None))
    for rows in (34,32):
        case,arrays=full[rows]
        for p in ('gate','up','down'):append(case,arrays,p,rows,'actual_full_routed_job')
    case,arrays=full[34]
    for rows in (16,17,31):
        for p in ('gate','up','down'):append(case,arrays,p,rows,'derived_prefix_of_real_routed_job')
    for p in ('gate','up'):append(case,arrays,p,99,'actual_shared_input_counterfactual_expert_membership')
    selected=[(c,a)for c,a in down_cases if c['kind']=='original'and c['rows']==98]
    if len(selected)!=1:raise ValueError('Runtime smoke needs one actual original M98 DOWN')
    case,arrays=selected[0]
    result.append(dict(layer=case['layer'],expert=case['expert'],rows=98,projection='down',
        origin='actual_full_routed_job',captured_routed_job=True,input=arrays['activation'],capture_case=case,
        expected_raw=arrays['reference_raw'],expected_owner=arrays['reference_boundary']))
    if len(result)!=18:raise ValueError('Runtime smoke case inventory')
    return sorted(result,key=lambda c:(c['layer'],c['expert'],c['rows'],c['projection']))


def tail_canary(mx,np,runtime,half,expert,projection,normal):
    """Same cached kernel, actual C extent, larger sentinel-filled allocation."""
    p=runtime.plan(projection,half.shape[0]);q=expert.layout.tensor(projection+'.qweight');s=expert.layout.tensor(projection+'.scales')
    allocation=(p['rows']+PADDING,p['n'])
    out=runtime._kernel(p['k'],p['n'])(inputs=[half,expert.backing,mx.array([q.offset,s.offset,p['rows']],mx.uint32)],
        grid=p['grid'],threadgroup=p['threadgroup'],output_shapes=[allocation],output_dtypes=[mx.float32],init_value=SENTINEL)[0]
    mx.eval(out,normal)
    a,b=np.array(out),np.array(normal);sentinel=np.array(SENTINEL,np.float32).view(np.uint32).item()
    valid_exact=a[:p['rows']].tobytes()==b.tobytes()
    pad_ok=bool(np.all(a[p['rows']:].view(np.uint32)==sentinel))
    return dict(valid_rows_exact=valid_exact,padding_unchanged=pad_ok,passed=valid_exact and pad_ok,
        actual_rows=p['rows'],allocated_rows=allocation[0],padding_rows=PADDING,sentinel=SENTINEL,
        padding_bytes=int(PADDING*p['n']*4),same_cached_kernel=True,
        c_extent_from_actual_row_metadata=True,scope='Output allocation guard only; not an input-allocation OOB proof')


def fp64(np,x,output,views,projection):
    from ds41_native_mx_operands import oracle
    from ds41_native_mx_components import dot_bounds,_error
    rows=tuple(sorted({0,x.shape[0]//2,x.shape[0]-1}));n=output.shape[1]
    columns=(0,1,31,32,n-1);q,s=views[projection+'.qweight'],views[projection+'.scales']
    target=oracle(np,x,q,s,input_rows=rows,output_rows=columns)
    bounds,meta=dot_bounds(np,x,q,s,input_rows=rows,output_rows=columns)
    return dict(input_rows=list(rows),output_rows=list(columns),bound_metadata=meta,
        result=_error(np,target,output[np.ix_(rows,columns)],bounds))


def run_components(mx,np,capture_dir,out,report,compare,atomic):
    from ds41_native_ffn_components import capture_module as ffn_parser,validate_frame_inventory,source_coefficients
    from ds41_down_actual_components import capture_module as down_parser,_materialize,check_coefficients
    from ds41_prefill_wide_components import raw_details
    from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact
    from mlx2.adapters.deepseek_v41_source_buffers import read_original_packed
    from mlx2.adapters.deepseek_v41_source_weights import expert_names
    from mlx2.runtime.gguf import source_native_mx_runtime as r
    from probe_ds41_native_mx_runtime import OPTIONS
    if report.get('native_runtime_options')!=OPTIONS or 'native_runtime_smoke'in report:raise ValueError('Runtime smoke options/overwrite')
    dp,dpath,ds=down_parser(capture_dir);dm,down=dp.load_capture(dpath)
    fp,fpath,fs=ffn_parser(report['native_ffn_capture_path']);fm,full=fp.load_capture(fpath)
    validate_frame_inventory(full)
    for key in ('checkpoint','source_revision','original_artifact_fingerprint','hot_map_binding','q1_bank_binding'):
        if dm[key]!=fm[key]:raise ValueError('Runtime smoke captures source/checkpoint/precision mismatch')
    if report['checkpoint']!=fm['checkpoint']or report['source_revision']!=fm['source_revision']:raise ValueError('Runtime smoke current base/checkpoint mismatch')
    artifact=inspect_artifact(fs['model'])
    if artifact['identity']['fingerprint']!=fm['original_artifact_fingerprint']:raise ValueError('Runtime smoke original source changed')
    hot=Path(fs['expert_policy']['hotset_path'])
    if hot.is_symlink()or hashlib.sha256(hot.read_bytes()).hexdigest()!=fm['hot_map_binding']:raise ValueError('Runtime smoke precision map changed')
    hot_keys=frozenset(map(tuple,json.loads(hot.read_text())['experts']))
    cases=case_matrix(np,full,down);out=Path(out)
    section=report['native_runtime_smoke']=dict(schema='mlx2.ds41-native-mx-runtime-smoke.v1',
        checkpoint=report['checkpoint'],source_revision=report['source_revision'],source_snapshot_sha256=report['source_snapshot_sha256'],
        capture_source_snapshots=[dm['source_snapshot_sha256'],fm['source_snapshot_sha256']],
        original_artifact_fingerprint=fm['original_artifact_fingerprint'],hot_map_binding=fm['hot_map_binding'],
        profile=r.PROFILE,case_count=len(cases),loaded_experts_concurrently=1,extra_acquires=0,
        rate_eligible=False,qualification=False,native_used_by_model=False,cases=[],complete=False,harness_completed=False)
    def write():
        peak=int(mx.get_peak_memory());report['component_peak_mlx_bytes']=max(report.get('component_peak_mlx_bytes',0),peak)
        if peak>LIMIT:raise MemoryError('Runtime smoke6GiB allocator ceiling')
        atomic(out/'results.json',report)
    write();active=None
    try:
        for case in cases:
            key=(case['layer'],case['expert'])
            if key!=active:
                if active is not None:
                    mx.synchronize();expert=None;host=None;views=None;bound=None;active=None;mx.clear_cache()
                if key not in hot_keys:raise ValueError('Runtime smoke original expert outside frozen precision map')
                host=read_original_packed(artifact,*key);views=host.views()
                files=sorted({artifact['tensors'][name]['file']for name in expert_names(*key)})
                seals={name:artifact['seals'][name]for name in files}
                seal_sha=hashlib.sha256(json.dumps(seals,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                att=r.attest_host(host,source_fingerprint=artifact['identity']['fingerprint'],source_seals_sha256=seal_sha,layer=key[0],expert=key[1])
                if not att.eligible:raise ValueError('Runtime smoke coefficient eligibility refused')
                expert=_materialize(mx,host);bound=r.bind_expert(expert,att);active=key
            cc=case['capture_case']
            if 'source_binding'in cc:source_coefficients(artifact,cc,host,hot_keys)
            else:check_coefficients(np,cc,views)
            p=case['projection'];x=mx.array(case['input'],mx.float32)
            stock=lambda v:mx.quantized_matmul(v,expert[p+'.qweight'],scales=expert[p+'.scales'],mode='mxfp4')
            own=lambda v:v.astype(mx.bfloat16).astype(mx.float32)
            raw=stock(x);owner=own(raw);control=stock(x)
            half=r.prepare_half(x);native=r.project_half(half,expert,bound,projection=p);native_owner=own(native)
            repeated=[r.project_half(half,expert,bound,projection=p)for _ in range(OPTIONS['repeats'])]
            mx.eval(raw,owner,control,half,native,native_owner,*repeated)
            rawcpu,owncpu,nativecpu,nowncpu=np.array(raw),np.array(owner),np.array(native),np.array(native_owner)
            actual_half=np.array(half).astype(np.float32)
            row={k:case[k]for k in ('layer','expert','rows','projection','origin','captured_routed_job')}
            row.update(attestation=att.receipt(),coefficient_identity=True,native_dispatched=True,
                input_sha256=hashlib.sha256(case['input'].tobytes()).hexdigest(),
                half_roundtrip_bits_exact=case['input'].tobytes()==actual_half.tobytes(),
                fresh_stock_exact=raw_details(np,rawcpu,np.array(control))['exact'],
                captured_control=None,native_raw=raw_details(np,rawcpu,nativecpu),native_owner=raw_details(np,owncpu,nowncpu),
                native_finite=bool(np.isfinite(nativecpu).all()and np.isfinite(nowncpu).all()),
                stable=all(raw_details(np,nativecpu,np.array(v))['exact']for v in repeated),
                fp64_reference=fp64(np,case['input'],rawcpu,views,p),fp64_native=fp64(np,actual_half,nativecpu,views,p))
            if case['captured_routed_job']:
                row['captured_control']=dict(raw=raw_details(np,case['expected_raw'],rawcpu),owner=raw_details(np,case['expected_owner'],owncpu))
            row['tail_guard']=tail_canary(mx,np,r,half,expert,p,native)
            second=np.ascontiguousarray(case['input'][::-1])
            if second.tobytes()==case['input'].tobytes():second=np.ascontiguousarray(-case['input'])
            if second.tobytes()==case['input'].tobytes():raise ValueError('Runtime smoke distinct-input canary is not distinct')
            second_x=mx.array(second);second_half=r.prepare_half(second_x)
            second_native=r.project_half(second_half,expert,bound,projection=p)
            second_stock=stock(second_x);zero_half=r.prepare_half(mx.zeros_like(x))
            zero=r.project_half(zero_half,expert,bound,projection=p);again=r.project_half(half,expert,bound,projection=p)
            mx.eval(second_native,second_stock,zero,again,second_half)
            second_cpu=np.array(second_native)
            row['second_input']=dict(scope='Distinct reversed real row order; if symmetric, explicit negative-input canary',
                sha256=hashlib.sha256(second.tobytes()).hexdigest(),
                finite=bool(np.isfinite(second_cpu).all()),
                fp64_reference=fp64(np,second,np.array(second_stock),views,p),
                fp64_native=fp64(np,np.array(second_half).astype(np.float32),second_cpu,views,p))
            row['zero_aba_passed']=bool(np.all(np.array(zero)==0)and raw_details(np,nativecpu,np.array(again))['exact']and np.any(nativecpu!=0))
            row['fp64_passed']=all(item['result']['within_bound']is True for item in
                (row['fp64_reference'],row['fp64_native'],row['second_input']['fp64_reference'],row['second_input']['fp64_native']))
            section['cases'].append(row);write()
            if not(row['fresh_stock_exact']and row['native_finite']and row['stable']and row['fp64_passed']and row['zero_aba_passed']and row['tail_guard']['passed']and row['second_input']['finite']):
                raise ValueError('Runtime smoke finite/stock/stability/FP64/canary/tail gate failed')
            if row['captured_control']and not all(v['exact']for v in row['captured_control'].values()):raise ValueError('Runtime smoke full captured stock parity failed')
            del x,raw,owner,control,half,native,native_owner,repeated,second_x,second_half,second_native,second_stock,zero_half,zero,again,stock,own
        verdicts=engineering_verdicts(section['cases']);section.update(verdicts=verdicts,
            runtime_kernel_cache_geometries=r._kernel.cache_info().currsize,
            complete=True,harness_completed=True,required_reference_controls_exact=verdicts['stock_capture_controls']and verdicts['fresh_stock_controls'])
        if section['runtime_kernel_cache_geometries']!=2:raise ValueError('Runtime smoke unexpectedly specialized by M')
        write();return section
    finally:
        mx.synchronize()
        if active is not None:expert=None;host=None;views=None;bound=None
        mx.clear_cache()


def engineering_verdicts(rows):
    return dict(stock_capture_controls=bool(rows)and all(c['captured_control']is None or all(v['exact']is True for v in c['captured_control'].values())for c in rows),
        fresh_stock_controls=bool(rows)and all(c['fresh_stock_exact']is True for c in rows),
        coefficient_identity=bool(rows)and all(c['coefficient_identity']is True for c in rows),
        runtime_eligibility=bool(rows)and all(c['attestation']['eligible']is True for c in rows),
        finite_stable=bool(rows)and all(c['native_finite']is True and c['stable']is True and c['second_input']['finite']is True for c in rows),
        actual_fp64_bounds=bool(rows)and all(c['fp64_passed']is True for c in rows),
        zero_second_aba=bool(rows)and all(c['zero_aba_passed']is True for c in rows),
        tail_guards=bool(rows)and all(c['tail_guard']['passed']is True for c in rows),
        all_cases_dispatched=len(rows)==18 and all(c['native_dispatched']is True for c in rows))
