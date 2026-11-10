"""Actual Source dense FP8 projection probes; caller owns GPU and leases.

No serving policy, all-dense load, expert requantization or GPU work on import.
Native BF16 output parity never establishes its unexposed pre-cast F32 parity.
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
import time
from pathlib import Path

LIMIT=6<<30
_UPLOAD_QUARANTINE=[]
NATIVE_PROFILE='native_bf16_pp_v1'


def projection_options(report):
    options=report.get('projection_options',{})
    if not isinstance(options,dict) or set(options)-{'numerical_profile','repeats'}:
        raise ValueError('Unknown projection options')
    profile=options.get('numerical_profile','reference_exact_only')
    if profile not in ('reference_exact_only',NATIVE_PROFILE):raise ValueError('Unknown projection numerical profile')
    repeats=options.get('repeats',7 if profile==NATIVE_PROFILE else 3)
    if type(repeats)is not int or not 3<=repeats<=7:raise ValueError('Projection repeats must be3..7')
    return profile,repeats


def decode_effective_f32(np,codes,scales):
    """Original E4M3 code times E8M0 scale, rounded to effective F32.

    Decodes only caller-selected output rows. This is not a BF16 coefficient
    substitution and is independent of stock QMM's reduction program.
    """
    if codes.dtype!=np.uint8 or scales.dtype!=np.uint8 or codes.ndim!=2 or codes.shape[1]%32 or scales.shape!=(codes.shape[0],codes.shape[1]//32):
        raise ValueError('FP64 oracle coefficient geometry/dtype')
    exponent=((codes>>3)&15).astype(np.int32);mantissa=(codes&7).astype(np.float32)
    value=np.where(exponent==0,np.ldexp(mantissa,-9),np.ldexp(1+mantissa/8,exponent-7)).astype(np.float32)
    value=np.where((codes&128)!=0,-value,value).astype(np.float32)
    if np.any((exponent==15)&((codes&7)==7)) or np.any(scales==255):raise ValueError('Nonfinite original FP8 coefficient')
    with np.errstate(over='ignore',invalid='ignore'):
        value=np.ldexp(value,np.repeat(scales.astype(np.int32)-127,32,axis=1)).astype(np.float32)
    if not np.isfinite(value).all():raise ValueError('Nonfinite effective F32 coefficient')
    return value


def fp64_reference(np,x,q,scales,*,block_rows=128):
    """Common actual-input mathematical oracle; no full dense weight expansion."""
    from ds41_dwarfstar_attention_components import single_thread_blas
    if type(block_rows)is not int or not 1<=block_rows<=128:raise ValueError('FP64 coefficient block must be1..128')
    if x.dtype!=np.float32 or q.dtype!=np.uint32 or not np.isfinite(x).all():raise ValueError('FP64 input/packed coefficient dtype')
    codes=np.ascontiguousarray(q).view(np.uint8)
    if x.shape[-1]!=codes.shape[1]:raise ValueError('FP64 input width mismatch')
    x64=x.reshape(-1,x.shape[-1]).astype(np.float64)
    result=np.empty((x64.shape[0],q.shape[0]),np.float64);changed=0;total=0
    with single_thread_blas(np):
        for first in range(0,q.shape[0],block_rows):
            last=min(first+block_rows,q.shape[0]);w=decode_effective_f32(np,codes[first:last],scales[first:last])
            bits=w.view(np.uint32);rounded=(bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)
            changed+=int(np.count_nonzero(bits!=rounded));total+=int(bits.size)
            result[:,first:last]=x64@w.astype(np.float64).T
    return result.reshape(*x.shape[:-1],q.shape[0]),dict(effective_F32_coefficient_BF16_roundtrip_exact=changed==0,
        changed_words=changed,total_words=total,maximum_decoded_weight_rows=block_rows,BLAS_threads=1)


def fp64_error(np,truth,value):
    if value.shape!=truth.shape or not np.isfinite(value).all():raise ValueError('FP64 comparison nonfinite/shape')
    delta=value.astype(np.float64)-truth
    return dict(rmse=float(np.sqrt(np.mean(delta*delta))),max_abs=float(np.max(np.abs(delta))),
        relative_L2=float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(truth.ravel()),1e-300)))


def profile_timings(mx,np,functions,*,repeats):
    """Different numerical laws may time; each must be finite and self-stable."""
    from ds41_attention_components import timed_interleaved
    for name,fn in functions.items():
        a=fn();mx.eval(a);first=np.array(a);del a
        b=fn();mx.eval(b);second=np.array(b);del b
        if not np.isfinite(first).all() or first.tobytes()!=second.tobytes():
            raise ValueError('Projection timing requires finite repeat-stable output: '+name)
    return timed_interleaved(mx,np,functions,repeats=repeats)


def capture_module(capture_dir):
    path=Path(capture_dir)
    if path.is_symlink():raise ValueError('Projection capture path symlink')
    path=path.resolve()
    if path.is_dir():path=path/'attention-projection-capture.json'
    if path.is_symlink():raise ValueError('Projection capture manifest symlink')
    unit=path.parent;manifest=json.loads(path.read_text());spec_path=unit/'spec.json'
    if spec_path.is_symlink():raise ValueError('Projection capture spec symlink')
    spec=json.loads(spec_path.read_text());parser=unit/'harness/attention_projection_probe.py'
    expected=spec.get('harness_sha256',{}).get('attention_projection_probe.py')
    if (parser.is_symlink()or not expected or expected!=manifest.get('helper_sha256')
            or hashlib.sha256(parser.read_bytes()).hexdigest()!=expected):
        raise ValueError('Projection capture parser seal mismatch')
    for key in ('checkpoint','source_revision','source_snapshot_sha256'):
        if not manifest.get(key)or manifest[key]!=spec.get(key):raise ValueError('Projection capture source identity mismatch')
    loader=importlib.util.spec_from_file_location('_ds41_sealed_projection_capture',parser)
    module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    return module,path,spec


def coefficient_payload(np,artifact,alias,read_tensors):
    """Read exactly one captured weight and original block scale extent."""
    name=artifact['aliases'][alias];scale=name.removesuffix('weight')+'scale'
    row,srow=artifact['tensors'][name],artifact['tensors'][scale]
    if row['dtype']!='F8_E4M3'or srow['dtype']!='F8_E8M0'or len(row['shape'])!=2:
        raise ValueError('Projection components require original MXFP8 codes/scales')
    output,width=row['shape']
    if width%32 or tuple(srow['shape'])!=((output+31)//32,width//32):
        raise ValueError('Projection original FP8 block scale geometry')
    data=read_tensors(artifact,(name,scale))
    if data[name].dtype!=np.uint8 or data[scale].dtype!=np.uint8:
        raise ValueError('Projection original FP8 byte dtype')
    q=np.ascontiguousarray(data[name]).view('<u4')
    scales=np.ascontiguousarray(np.repeat(data[scale],32,axis=0)[:output])
    record={key:dict(shape=list(value.shape),dtype=str(value.dtype),
        sha256=hashlib.sha256(value.tobytes()).hexdigest())for key,value in [('qweight',q),('scales',scales)]}
    return q,scales,dict(alias=alias,source_names=[name,scale],source_read_bytes=int(data[name].nbytes+data[scale].nbytes),
        uploaded_coefficient_bytes=int(q.nbytes+scales.nbytes),coefficients=record)


def verify_coefficients(record,case):
    if record['coefficients']!=case['coefficients']:
        raise ValueError('Actual captured projection coefficient hash/geometry mismatch')


def upload_coefficients(mx,q,s):
    """Keep host slabs until upload completion, quarantine if fencing fails."""
    values=[]
    try:
        values.append(mx.array(q,mx.uint32));values.append(mx.array(s,mx.uint8))
        mx.eval(*values)
        return tuple(values)
    except BaseException:
        try:mx.synchronize()
        except BaseException:_UPLOAD_QUARANTINE.append((q,s,*values))
        raise


def _boundary(mx,capture_spec):
    if capture_spec.get('extra_source_policy',{}).get('deepseek_pointwise')=='fused':
        from mlx2.runtime.models.deepseek_v41_fused import rounded
        return rounded,'owning Source fused BF16-round F32 kernel'
    return lambda x:x.astype(mx.bfloat16).astype(mx.float32),'owning BF16→F32 cast boundary'


def run_components(mx,np,captureDir,out,report,compare,atomic):
    if mx.default_device()!=mx.gpu:raise RuntimeError('Projection components require caller-owned GPU')
    if 'attention_projection_components'in report:raise ValueError('Refusing to overwrite projection component evidence')
    profile,repeats=projection_options(report)
    distinct=profile==NATIVE_PROFILE
    from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact,read_tensors
    from ds41_prefill_wide_components import raw_details
    parser,path,capture_spec=capture_module(captureDir);manifest,cases=parser.load_capture(path)
    for key in ('checkpoint','source_revision'):
        if report.get(key)!=manifest[key]:raise ValueError('Projection component/capture identity mismatch')
    expected={(l,p,k)for l in (2,20,21)for p in ('pp99','decode1')for k in ('q_b','wo_b')}
    if len(cases)!=12 or {(c['layer'],c['phase'],c['projection'])for c,a in cases}!=expected:
        raise ValueError('Projection components require twelve actual captured cases')
    model_path=capture_spec.get('model')or report.get('model')
    if not model_path:raise ValueError('Projection captured Source artifact path missing')
    if report.get('model')and Path(report['model']).expanduser().resolve()!=Path(model_path).expanduser().resolve():
        raise ValueError('Projection Source artifact path mismatch')
    artifact=inspect_artifact(model_path)
    for source in (capture_spec,report):
        expected_fp=source.get('original_fingerprint')
        if expected_fp is not None and expected_fp!=artifact['identity']['fingerprint']:
            raise ValueError('Projection original Source fingerprint mismatch')
    boundary,boundary_scope=_boundary(mx,capture_spec);out=Path(out)
    section=report['attention_projection_components']=dict(schema='mlx2.ds41-attention-projection-components.v1',
        checkpoint=manifest['checkpoint'],source_revision=manifest['source_revision'],
        capture_source_snapshot_sha256=manifest['source_snapshot_sha256'],component_source_snapshot_sha256=report.get('source_snapshot_sha256'),
        capture_path=str(path),capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        original_artifact_fingerprint=artifact['identity']['fingerprint'],original_artifact_path=str(artifact['root']),
        scope='12actual preparedQMM operand cases; unchanged originaldenseFP8codes; no modelquality/default claim',
        boundary_scope=boundary_scope,qualified=False,performance_qualified=False,candidate_used_by_model=False,
        maximum_allocator_bytes=LIMIT,loaded_coefficients=[],cases=[],numerical_profile=profile,repetitions=repeats,
        timing_scope='Interleaved resident operators; native includesinputBF16cast+QMM+outputF32cast; explicit native_bf16_pp_v1 admits finite self-stable differing PP outputs, not quality',
        all_dense_provider_constructed=False,
        native_dispatch_scope='Stock library dtype-dependent dispatch; no specific NAX shader engagement claim without trace',
        native_raw_F32_precast_parity='not exposed bynativeBF16QMM; never claimed',complete=False,harness_completed=False,
        required_reference_controls_exact=False,verdicts={})
    def write():atomic(out/'results.json',report)
    def observe():
        peak=int(mx.get_peak_memory());section['peak_mlx_bytes']=max(section.get('peak_mlx_bytes',0),peak)
        report['component_peak_mlx_bytes']=max(report.get('component_peak_mlx_bytes',0),peak)
        if peak>LIMIT:write();raise MemoryError('Projection components exceeded6GiB MLX ceiling')
    write();observe();bank={};host_bank={}
    if distinct:cases=sorted(cases,key=lambda row:(row[0]['linear_name'],row[0]['phase']))
    for case,arrays in cases:
        row=dict(capture=case,reference_controls={},variants={});section['cases'].append(row)
        audit=case['input_bf16_audit'];row['input_BF16_roundtrip_audit']=audit
        if not audit['exact']or arrays['prepared_input'].tobytes()!=arrays['input_bf16_roundtrip'].tobytes():
            write();raise ValueError('Captured prepared Source input is not losslessly BF16-representable')
        alias=case['linear_name']
        if alias not in bank:
            if distinct:
                mx.synchronize();bank.clear();host_bank.clear();mx.clear_cache()
            q,s,load=coefficient_payload(np,artifact,alias,read_tensors);verify_coefficients(load,case)
            q_gpu,s_gpu=upload_coefficients(mx,q,s)
            if distinct:host_bank[alias]=(q,s)
            bank[alias]=(q_gpu,s_gpu,load);section['loaded_coefficients'].append(load);q=None;s=None;del q_gpu,s_gpu
        q,s,load=bank[alias];verify_coefficients(load,case);observe()
        x=mx.array(arrays['prepared_input'],mx.float32);mx.eval(x)
        def stock_raw():return mx.quantized_matmul(x,q,scales=s,mode='mxfp8')
        def stock_boundary():return boundary(stock_raw())
        raw=stock_raw();rounded=boundary(raw);mx.eval(raw,rounded)
        row['reference_controls']=dict(raw=raw_details(np,arrays['reference_raw'],np.array(raw)),
            BF16=raw_details(np,arrays['reference_bf16'],np.array(rounded)))
        if not all(c['exact']for c in row['reference_controls'].values()):
            write();raise ValueError('Stock F32 projection replay differs from actual capture')
        def native():
            value=mx.quantized_matmul(x.astype(mx.bfloat16),q,scales=s,mode='mxfp8')
            if value.dtype!=mx.bfloat16:raise ValueError('NativeBF16QMM did not expose expected BF16 output')
            return value.astype(mx.float32)
        output_shape=tuple(arrays['reference_raw'].shape)
        def flat_raw():return mx.quantized_matmul(x.reshape(-1,x.shape[-1]),q,scales=s,mode='mxfp8').reshape(output_shape)
        native_value=native();flat=flat_raw();flat_boundary=boundary(flat);mx.eval(native_value,flat,flat_boundary)
        native_check=raw_details(np,arrays['reference_bf16'],np.array(native_value))
        native_finite=bool(np.isfinite(np.array(native_value)).all())
        native_admitted=native_finite and (native_check['exact'] or distinct and case['phase']=='pp99')
        flat_checks=dict(raw=raw_details(np,arrays['reference_raw'],np.array(flat)),BF16=raw_details(np,arrays['reference_bf16'],np.array(flat_boundary)))
        row['variants']['native_bf16']=dict(boundary_check=native_check,boundary_exact=native_check['exact'],
            raw_F32_precast_parity='notexposed/notclaimed',input_dtype='bfloat16',output_dtype='bfloat16→float32',
            unchanged_weight_codes=True,input_cast_in_timing=True,observed_operator_execution=True,
            observed_scope='bounded component oracle only, not model use',timing_admitted=native_admitted,
            all_finite=native_finite,numerical_profile=NATIVE_PROFILE if distinct else 'historical exact-only probe',
            quality_evaluated=False)
        row['variants']['flat_f32']=dict(checks=flat_checks,raw_and_boundary_exact=all(c['exact']for c in flat_checks.values()),
            input_dtype='float32',unchanged_weight_codes=True,observed_operator_execution=True,
            observed_scope='bounded component oracle only, not model use',timing_admitted=all(c['exact']for c in flat_checks.values()))
        functions={'reference':stock_boundary}
        if native_admitted:functions['native_bf16']=native
        if row['variants']['flat_f32']['raw_and_boundary_exact']:functions['flat_f32']=lambda:boundary(flat_raw())
        if distinct:
            started=time.perf_counter();truth,coefficient_audit=fp64_reference(np,arrays['prepared_input'],*host_bank[alias])
            row['common_FP64_oracle']=dict(scope='Actual original F32 activations and effective F32 decoded original coefficients; mathematical error, not model quality',
                seconds=time.perf_counter()-started,sha256=hashlib.sha256(truth.tobytes()).hexdigest(),coefficient_audit=coefficient_audit,
                source_raw=fp64_error(np,truth,np.array(raw)),source_boundary=fp64_error(np,truth,np.array(rounded)),
                native_boundary=fp64_error(np,truth,np.array(native_value)))
            del truth
        del raw,rounded,native_value,flat,flat_boundary
        if distinct:
            row['timings']=profile_timings(mx,np,functions,repeats=repeats)
            row['timing_output_self_stability_verified']=True
        else:row['timings']=compare(mx,np,functions,repeats=repeats)
        observe();write();mx.synchronize();x=None;q=None;s=None;del functions;mx.clear_cache()
    section['required_reference_controls_exact']=all(all(c['exact']for c in r['reference_controls'].values())for r in section['cases'])
    section['verdicts']={
        'native_bf16_all12_boundary_exact':all(r['variants']['native_bf16']['boundary_exact']for r in section['cases']),
        'flat_f32_all12_raw_and_boundary_exact':all(r['variants']['flat_f32']['raw_and_boundary_exact']for r in section['cases'])}
    section['complete']=section['harness_completed']=True
    section['loaded_alias_count']=len(section['loaded_coefficients'])
    section['uploaded_coefficient_bytes']=sum(row['uploaded_coefficient_bytes']for row in section['loaded_coefficients'])
    write();return section
