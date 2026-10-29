"""Sealed actual-operand attention probes; caller owns GPU and paired leases.

No MLX import/work on import. Optional numerical failures are retained and
excluded from timing. A failed stock replay is a required-control failure.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

LIMIT=6<<30


def capture_module(capture_path):
    """Seal the capture-owned CPU parser before executing its local code."""
    capture_path=Path(capture_path)
    if capture_path.is_symlink():raise ValueError('Attention capture path symlink')
    capture_path=capture_path.resolve()
    if capture_path.is_dir():capture_path=capture_path/'attention-capture.json'
    unit=capture_path.parent
    if capture_path.is_symlink()or (unit/'spec.json').is_symlink():
        raise ValueError('Attention capture manifest/spec symlink')
    manifest=json.loads(capture_path.read_text());spec=json.loads((unit/'spec.json').read_text())
    path=unit/'harness/attention_probe.py'
    expected=spec.get('harness_sha256',{}).get('attention_probe.py')
    if path.is_symlink()or not expected or manifest.get('helper_sha256')!=expected:
        raise ValueError('Attention capture parser seal missing/inconsistent')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError('Attention capture parser seal mismatch')
    for key in ('checkpoint','source_revision','source_snapshot_sha256'):
        if not manifest.get(key)or manifest[key]!=spec.get(key):
            raise ValueError('Attention capture spec identity mismatch')
    module_spec=importlib.util.spec_from_file_location('_ds41_sealed_attention_capture',path)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    return module,capture_path


def raw_details(np,expected,actual):
    """F32 bits, signed zero and bounded first-mismatch diagnostics."""
    expected=np.ascontiguousarray(expected);actual=np.ascontiguousarray(actual)
    if expected.shape!=actual.shape or expected.dtype!=np.float32 or actual.dtype!=np.float32:
        raise ValueError('Attention oracle requires same-shape F32 arrays')
    a,b=expected.view(np.uint32).ravel(),actual.view(np.uint32).ravel();changed=np.flatnonzero(a!=b)
    finite=np.isfinite(expected)&np.isfinite(actual)
    delta=np.abs(actual[finite].astype(np.float64)-expected[finite].astype(np.float64))
    result=dict(exact=not len(changed),changed_words=int(len(changed)),shape=list(actual.shape),
        reference_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
        candidate_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
        max_abs_finite_pairs=float(np.max(delta,initial=0.)),
        changed_nonfinite_words=int(np.count_nonzero((a!=b)&~finite.ravel())))
    if actual.ndim==4:
        counts=np.count_nonzero((a!=b).reshape(actual.shape),axis=(0,1,3))
        result['query_row_changed_counts']=list(map(int,counts[:128]))
        result['query_rows_truncated']=len(counts)>128
        result['query_row_axis']=2
    if len(changed):
        i=int(changed[0]);first=max(0,i-3);last=min(len(a),i+5)
        result['first_mismatch']=dict(flat_index=i,coordinate=list(map(int,np.unravel_index(i,actual.shape))),
            reference_bits=f'{int(a[i]):08x}',candidate_bits=f'{int(b[i]):08x}',
            word_window_start=first,reference_word_window=[f'{int(x):08x}'for x in a[first:last]],
            candidate_word_window=[f'{int(x):08x}'for x in b[first:last]])
    return result


def restored_inputs(mx,np,case,arrays):
    """Restore known all-retained ID broadcasting, preserving source values.

    Generic arbitrary strided views are not fabricated. Captured pool backing
    capacity is restored when the capture records it; owning stock replay is
    required before any interpretation of changed physical input layout.
    """
    if (case.get('layer'),case.get('phase'))not in {(l,p)for l in (2,20,21)for p in ('pp99','decode1')}:
        raise ValueError('Attention component unexpected capture case')
    required=('q','local','local_mask','pooled','indices','valid','sinks','reference_raw','reference_bf16')
    if any(k not in arrays for k in required):raise ValueError('Attention component capture inventory')
    ids=arrays['indices'];count=arrays['pooled'].shape[1]
    if not np.array_equal(ids,np.broadcast_to(np.arange(count),ids.shape)):
        raise ValueError('Attention component requires chronological all-retained IDs')
    if any(not np.isfinite(arrays[k]).all()for k in ('q','local','pooled','sinks','reference_raw','reference_bf16')):
        raise ValueError('Attention component finite capture contract')
    values={k:mx.array(arrays[k])for k in ('q','local','local_mask','pooled','valid','sinks')}
    values['indices']=mx.broadcast_to(mx.arange(count,dtype=mx.uint32 if ids.dtype==np.uint32 else mx.int32)[None,None],ids.shape)
    layout=dict(indices='restored chronological arange broadcast',other_inputs='captured values uploaded; stock replay required',
        arbitrary_stride_reconstruction=False)
    if case.get('input_layout')is not None:
        layout['capture_export_layout']=case['input_layout']
        layout['device_stride_limit']='Capture NumPy export is not an MLX device-stride observation; no arbitrary layout claim'
    capacity=case.get('pool_capacity')
    if capacity is not None:
        if type(capacity)is not int or not count<=capacity<=65536:
            raise ValueError('Attention component captured pool capacity invalid')
        host=np.zeros((1,capacity,arrays['pooled'].shape[-1]),np.float32)
        host[:,:count]=arrays['pooled']
        values['pooled']=mx.array(host)[:,:count]
        layout['pool_capacity']=capacity
    if 'gathered'in arrays:values['gathered']=mx.array(arrays['gathered'])
    mx.eval(*values.values())
    args=[values[k]for k in ('q','local','local_mask','pooled','indices','valid','sinks')]
    return args,values.get('gathered'),layout


def timed_interleaved(mx,np,functions,*,repeats=3,clock=time.perf_counter):
    """Isolated operator timing, with build/dispatch and completion separated."""
    if type(repeats)is not int or not 3<=repeats<=7:
        raise ValueError('Attention component timing repeats must be3..7')
    names=list(functions);samples={name:[]for name in names}
    for fn in functions.values():mx.eval(fn())
    for rep in range(repeats):
        order=names[rep%len(names):]+names[:rep%len(names)]
        for name in order:
            mx.synchronize()  # Outside timed interval; isolate prior work.
            start=clock();value=functions[name]();built=clock();mx.eval(value);done=clock()
            samples[name].append(dict(graph_build_and_submission_seconds=built-start,
                completion_wait_seconds=done-built,total_seconds=done-start,repetition=rep))
            del value
    return {name:dict(samples=values,median_graph_build_and_submission_seconds=float(np.median([v['graph_build_and_submission_seconds']for v in values])),
        median_completion_wait_seconds=float(np.median([v['completion_wait_seconds']for v in values])),
        median_total_seconds=float(np.median([v['total_seconds']for v in values])))for name,values in samples.items()}


def run_components(mx,np,capture_path,out,report,compare,atomic):
    """Replay six actual cases; only exact-stage variants receive timings.

    compare is accepted for the common controller API; this helper uses its
    own split clock to expose build versus completion without duplicate runs.
    """
    if mx.default_device()!=mx.gpu:raise RuntimeError('Attention components require caller-owned GPU')
    if 'attention_components'in report:raise ValueError('Refusing to overwrite attention component evidence')
    from mlx2.runtime.models import deepseek_v41_attention_probe as probe
    parser,path=capture_module(capture_path);manifest,cases=parser.load_capture(path)
    for key in ('checkpoint','source_revision'):
        if report.get(key)!=manifest[key]:raise ValueError('Attention component/capture identity mismatch')
    expected={(layer,phase)for layer in (2,20,21)for phase in ('pp99','decode1')}
    if len(cases)!=6 or {(c['layer'],c['phase'])for c,a in cases}!=expected:
        raise ValueError('Attention component requires six actual PP99/decode1 cases')
    out=Path(out)
    section=report['attention_components']=dict(schema='mlx2.ds41-attention-components.v1',
        checkpoint=manifest['checkpoint'],source_revision=manifest['source_revision'],
        capture_path=str(path),capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        capture_source_snapshot_sha256=manifest['source_snapshot_sha256'],
        component_source_snapshot_sha256=report.get('source_snapshot_sha256'),
        scope='Actual finite all-retained Source attention operands, PP99frontier0/decode1frontier100 layers2/20/21; no model-path/capability/quality claim',
        maximum_allocator_bytes=LIMIT,qualified=False,performance_qualified=False,candidate_used_by_model=False,
        timing_scope='Prepared-input isolated operators;3 rotated interleaved reps after all raw/BF16/stage gates; graphbuild/submission and completionwall separate; not modelcriticalpath',
        common_compare_callback_used=False,cases=[],complete=False,harness_completed=False,
        required_reference_controls_exact=False,verdicts={})
    def write():atomic(out/'results.json',report)
    def observe():
        peak=int(mx.get_peak_memory());section['peak_mlx_bytes']=max(section.get('peak_mlx_bytes',0),peak)
        report['component_peak_mlx_bytes']=max(report.get('component_peak_mlx_bytes',0),peak)
        if peak>LIMIT:write();raise MemoryError('Attention components exceeded6GiB MLX allocator cap')
    write();observe()
    for case,arrays in cases:
        args,gathered,layout=restored_inputs(mx,np,case,arrays)
        row=dict(layer=case['layer'],phase=case['phase'],length=case['length'],frontier=case['frontier'],
            capture=case,prepared_layout=layout,reference_controls={},variants={})
        row['query_tile_plan']=probe.geometry(*args,variant='reference',all_retained=True,gathered=gathered)
        section['cases'].append(row)
        def fn(variant,stages=False):
            return probe.attention(*args,variant=variant,all_retained=True,gathered=gathered,return_stages=stages)
        actual=fn('reference');rounded=actual.astype(mx.bfloat16).astype(mx.float32);mx.eval(actual,rounded)
        row['reference_controls']={
            'raw_vs_capture':raw_details(np,arrays['reference_raw'],np.array(actual)),
            'BF16_vs_capture':raw_details(np,arrays['reference_bf16'],np.array(rounded))}
        if not all(c['exact']for c in row['reference_controls'].values()):
            section['required_reference_controls_exact']=False;write();raise ValueError('Attention owning stock replay differs from captured output')
        diagnostic,stages=fn('reference',True);mx.eval(diagnostic,*stages.values())
        row['reference_controls']['diagnostic_tree_vs_owning']=raw_details(np,np.array(actual),np.array(diagnostic))
        if not row['reference_controls']['diagnostic_tree_vs_owning']['exact']:
            write();raise ValueError('Attention diagnostic tree changed owning reference output')
        reference_stages={k:np.array(v)for k,v in stages.items()}
        del diagnostic,stages,actual,rounded
        admitted={'reference':lambda:fn('reference')}
        for variant in probe.VARIANTS[1:]:
            value,stages=fn(variant,True);boundary=value.astype(mx.bfloat16).astype(mx.float32)
            mx.eval(value,boundary,*stages.values())
            checks=dict(raw=raw_details(np,arrays['reference_raw'],np.array(value)),
                BF16=raw_details(np,arrays['reference_bf16'],np.array(boundary)))
            stage_checks={k:raw_details(np,reference_stages[k],np.array(v))for k,v in stages.items()}
            exact=all(c['exact']for c in checks.values())and all(c['exact']for c in stage_checks.values())
            row['variants'][variant]=dict(checks=checks,stage_checks=stage_checks,all_stage_and_output_exact=exact,
                timing_admitted=exact,numerical_profile_changed=not exact,
                decode_gather_reuse_preserved=(variant=='broadcast' or variant.endswith('_tail_safe'))and gathered is not None)
            plan=row['query_tile_plan']
            methods=[dict(first=first,last=last,method=probe.tile_variant(variant,last-first,plan['tile']))for first,last in plan['tiles']]
            row['variants'][variant]['query_tile_methods']=methods
            if variant in ('flat_pool','flat_pool_tail_safe'):
                contiguous_rows=sum(t['last']-t['first']for t in methods if t['method']=='flat_pool')
                row['variants'][variant]['explicit_contiguous_logical_bytes']={
                    'scaled_query':int(args[0].shape[1]*contiguous_rows*args[0].shape[-1]*4),
                    'pooled_probabilities':int(args[0].shape[1]*contiguous_rows*args[4].shape[-1]*4),
                    'scope':'logical bytes offered to mx.contiguous, not proof of physical copy'}
            if exact:admitted[variant]=lambda variant=variant:fn(variant)
            del value,boundary,stages
            observe();write()
        row['timings']=timed_interleaved(mx,np,admitted,repeats=3)
        mx.synchronize();del args,gathered,reference_stages,admitted
        mx.clear_cache();observe();write()
    section['required_reference_controls_exact']=all(all(c['exact']for c in r['reference_controls'].values())for r in section['cases'])
    section['verdicts']={variant+'_all_six_exact':all(r['variants'][variant]['all_stage_and_output_exact']for r in section['cases'])for variant in probe.VARIANTS[1:]}
    section['complete']=section['harness_completed']=True
    write();return section
