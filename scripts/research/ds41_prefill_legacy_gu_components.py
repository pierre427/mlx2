"""Actual-activation legacy GU tile oracle. Parent owns GPU and expert loading.

No MLX import or GPU execution on import. Does not admit a serving policy.
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
import hashlib
import importlib.util
import json
from pathlib import Path

LIMIT=6<<30

def capture_module(capture_path):
    """Import only the capture unit's sealed private parser, before any GPU work."""
    capture_path=Path(capture_path).resolve();unit=capture_path.parent
    manifest=json.loads(capture_path.read_text());spec=json.loads((unit/'spec.json').read_text())
    path=unit/'harness/legacy_gu_capture.py'
    expected=spec.get('harness_sha256',{}).get('legacy_gu_capture.py')
    if path.is_symlink() or not expected or manifest.get('hook_sha256')!=expected:
        raise ValueError('Capture parser seal missing or inconsistent')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError('Capture parser seal mismatch')
    for key in ('checkpoint','source_revision','source_snapshot_sha256'):
        if not manifest.get(key) or manifest[key]!=spec.get(key):raise ValueError('Capture spec identity mismatch')
    module_spec=importlib.util.spec_from_file_location('_ds41_sealed_legacy_capture',path)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    return module


def check_coefficients(np,case,data):
    if 'gate.biases'in data:raise ValueError('Legacy GU oracle requires original MXFP4 expert')
    p=case['projection'];q=np.asarray(data[p+'.qweight']);s=np.asarray(data[p+'.scales'])
    if q.dtype!=np.uint32 or q.shape!=(2304,640) or s.dtype!=np.uint8 or s.shape!=(2304,160):raise ValueError('Replay coefficient geometry')
    for x,key in [(q,'weight_sha256'),(s,'scales_sha256')]:
        if hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()!=case[key]:raise ValueError('Replay expert coefficient hash mismatch')

def run_legacy_gu(mx,np,capture_path,load_expert,out,report,compare,atomic,*,candidate=None):
    """Replay four actual cases, with raw and BF16-boundary oracles.

    load_expert(layer,expert) returns the same actual original packed mapping;
    caller owns its admission/lifetime. No full model or bank is loaded here.
    Candidate API returns (BF16-rounded F32, raw F32, split-K partials F32).
    Timings use three interleaved repeats only after both raw/boundary gates.
    """
    from ds41_prefill_wide_components import raw_details
    if 'prefill_legacy_gu' in report:raise ValueError('Refusing to overwrite legacy GU evidence')
    if mx.default_device()!=mx.gpu:raise RuntimeError('Caller-owned GPU required')
    if candidate is None:
        from mlx2.runtime.gguf.source_prefill_legacy_tiles import project
        candidate=project
    cap=capture_module(capture_path);manifest,cases=cap.load_capture(capture_path)
    for key in ('checkpoint','source_revision'):
        if report.get(key)!=manifest[key]:raise ValueError('Component/capture '+key+' mismatch')
    section=report['prefill_legacy_gu']=dict(schema='mlx2.source-legacy-gu-components.v1',
        capture_path=str(Path(capture_path).resolve()),capture_sha256=hashlib.sha256(Path(capture_path).read_bytes()).hexdigest(),
        checkpoint=manifest['checkpoint'],capture_source_snapshot_sha256=manifest['source_snapshot_sha256'],
        component_source_snapshot_sha256=report.get('source_snapshot_sha256'),
        scope='Actual reference-model selected FP8-QAT activations and same original expert coefficient hashes; four P5/P2 GU projections',
        maximum_allocator_bytes=LIMIT,cases=[],candidate_used_by_model=False,qualified=False,
        raw_reference='Captured original stock F32 QMM pre-round output, revalidated against replay stock QMM',
        partial_scope='Candidate partials retained for diagnosis; stock internal partials unavailable, no partial-equality claim',
        timing_scope='Three interleaved repeats only after exact raw F32 and owning BF16-boundary checks')
    out=Path(out)
    def write():atomic(out/'results.json',report)
    def observe():
        peak=int(mx.get_peak_memory());section['peak_mlx_bytes']=peak
        report['component_peak_mlx_bytes']=max(report.get('component_peak_mlx_bytes',0),peak)
        if peak>LIMIT:write();raise MemoryError('Legacy GU component exceeded6GiB')
    observe();write()
    for index,(case,arrays) in enumerate(cases):
        data=load_expert(case['layer'],case['expert'])
        check_coefficients(np,case,data)
        x=mx.array(arrays['activation'],dtype=mx.float32)
        projection=case['projection'];q=data[projection+'.qweight'];s=data[projection+'.scales']
        def stock_raw():return mx.quantized_matmul(x,q,scales=s,mode='mxfp4')
        def stock_boundary():return stock_raw().astype(mx.bfloat16).astype(mx.float32)
        raw=stock_raw();rounded=raw.astype(mx.bfloat16).astype(mx.float32);mx.eval(raw,rounded)
        stock_checks=dict(raw=raw_details(np,arrays['reference_preround'],np.array(raw)),boundary=raw_details(np,arrays['reference_boundary'],np.array(rounded)))
        row=dict(capture=case,replayed_stock=stock_checks)
        section['cases'].append(row);observe();write()
        if not all(v['exact']for v in stock_checks.values()):
            section['capture_replay_failed']=True;write();raise ValueError('Stock replay differs from captured reference')
        candidate_boundary,candidate_raw,partials=candidate(x,data,projection=projection,return_partials=True)
        mx.eval(candidate_boundary,candidate_raw,partials)
        a,b,c=np.array(candidate_boundary),np.array(candidate_raw),np.array(partials)
        if c.dtype!=np.float32 or c.shape!=(case['partitions'],case['rows'],2304):raise ValueError('Candidate partial geometry')
        row['candidate_raw']=raw_details(np,arrays['reference_preround'],b)
        row['candidate_boundary']=raw_details(np,arrays['reference_boundary'],a)
        row['candidate_partials_sha256']=hashlib.sha256(np.ascontiguousarray(c).tobytes()).hexdigest()
        row['exact']=row['candidate_raw']['exact'] and row['candidate_boundary']['exact']
        filename=f'legacy-gu-component-{index:02d}-p{case["partitions"]}-{projection}.npz'
        path=out/filename
        with path.open('xb')as f:np.savez(f,candidate_boundary=a,candidate_raw=b,candidate_partials=c)
        row['candidate_arrays']=dict(path=filename,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        observe();write()
        if row['exact']:
            row['timing']=compare(mx,np,{'reference':stock_boundary,'legacy_tiles':lambda:candidate(x,data,projection=projection)},repeats=3)
        else:row['timing']=None
        mx.synchronize();x=None;data=None;q=None;s=None;del raw,rounded,candidate_boundary,candidate_raw,partials,a,b,c
        observe();write()
    section['all_raw_and_boundaries_exact']=len(section['cases'])==4 and all(c['exact']for c in section['cases'])
    section['complete']=True;write()
    return section
