# SPDX-License-Identifier: MIT
"""Bounded actual-capture NAX numerical-profile experiment, never serving math.

Caller owns paired GPU lease, original expert loading, source freeze and cleanup.
Nonexact finite candidate timing is explicit different-profile exploration, not
an exactness or model-quality verdict. Mandatory stock replay stays bitwise exact.
"""
import hashlib
from pathlib import Path
import statistics
import time

PROFILE='mxfp4_nax_k16_ordered_f32_v1'
LIMIT=6<<30


def ordered_partials(np,parts):
    if parts.dtype!=np.float32 or parts.ndim!=3 or parts.shape[0]not in (2,5):
        raise ValueError('NAX profile partial geometry')
    out=np.zeros(parts.shape[1:],np.float32)
    for p in parts:np.add(out,p,out=out)
    return out


def details(np,reference,candidate):
    from ds41_prefill_wide_components import raw_details
    finite=bool(np.isfinite(reference).all() and np.isfinite(candidate).all())
    if not finite:
        return dict(exact=False,finite=False,reference_nonfinite=int((~np.isfinite(reference)).sum()),
                    candidate_nonfinite=int((~np.isfinite(candidate)).sum()))
    d=raw_details(np,reference,candidate);diff=candidate.astype(np.float64)-reference.astype(np.float64)
    norm=float(np.linalg.norm(reference.astype(np.float64)))
    d.update(finite=True,RMS_absolute_error=float(np.sqrt(np.mean(diff*diff))),
             relative_L2_error=float(np.linalg.norm(diff))/norm if norm else None,
             changed_word_fraction=d['mismatched_f32_words']/candidate.size)
    return d


def time_different_profile(mx,np,functions,*,repeats=3):
    """Interleave finite, engineering-admitted variants; no exactness admission."""
    if repeats!=3:raise ValueError('Bounded profile timing requires three repeats')
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


def run_nax_profile(mx,np,capture_path,load_expert,out,report,compare,atomic,*,candidate=None,profile_timing=None):
    """Same controller signature as run_legacy_gu; compare intentionally unused.

    Existing compare admits only exact candidates. This separate profile instead
    records errors and admits finite/ordered-reduction-correct cases to timing.
    """
    from ds41_prefill_legacy_gu_components import capture_module,check_coefficients
    if 'prefill_nax_profile'in report:raise ValueError('Refusing to overwrite NAX profile evidence')
    if mx.default_device()!=mx.gpu:raise RuntimeError('Caller-owned GPU required')
    if candidate is None:
        from mlx2.runtime.gguf.source_prefill_nax_profile import project
        candidate=project
    timing=profile_timing or time_different_profile
    manifest,cases=capture_module(capture_path).load_capture(capture_path)
    if len(cases)!=4:raise ValueError('NAX profile requires four sealed actual GU cases')
    if {(c['partitions'],c['projection'])for c,_ in cases}!={(p,k)for p in (2,5)for k in ('gate','up')}:
        raise ValueError('NAX actual capture P5/P2 gate/up inventory')
    for key in ('checkpoint','source_revision'):
        if report.get(key)!=manifest[key]:raise ValueError('Component/capture '+key+' mismatch')
    section=report['prefill_nax_profile']=dict(schema='mlx2.source-nax-numerical-profile-components.v1',
        numerical_profile=PROFILE,capture_path=str(Path(capture_path).resolve()),capture_sha256=hashlib.sha256(Path(capture_path).read_bytes()).hexdigest(),
        checkpoint=manifest['checkpoint'],capture_source_snapshot_sha256=manifest['source_snapshot_sha256'],
        component_source_snapshot_sha256=report.get('source_snapshot_sha256'),
        scope='Four sealed actual selected Source FP8-QAT activations and identical original MXFP4 coefficients; no modified rows or synthetic routing',
        maximum_allocator_bytes=LIMIT,cases=[],candidate_used_by_model=False,qualified=False,model_quality_assessed=False,
        numerical_law='Native BF16 NAX K16 inner dot, F32 partials and ascending +0 F32 partition sum; different from stock legacy K8',
        timing_scope='Three interleaved repeats after exact stock control and finite/ordered-reducer candidate gates; numerical mismatch does not imply model-quality failure or pass')
    out=Path(out)
    def write():atomic(out/'results.json',report)
    def observe():
        peak=int(mx.get_peak_memory());section['peak_mlx_bytes']=max(section.get('peak_mlx_bytes',0),peak)
        report['component_peak_mlx_bytes']=max(report.get('component_peak_mlx_bytes',0),peak)
        if peak>LIMIT:write();raise MemoryError('NAX numerical profile exceeded6GiB')
    observe();write()
    for index,(case,arrays) in enumerate(cases):
        data=load_expert(case['layer'],case['expert']);check_coefficients(np,case,data)
        activation=np.ascontiguousarray(arrays['activation'])
        if activation.dtype!=np.float32 or not np.isfinite(activation).all() or np.any(activation.view(np.uint32)&0xffff):
            raise ValueError('Actual Source QAT input must be finite and lossless BF16-grid F32')
        x=mx.array(activation,dtype=mx.float32);projection=case['projection'];q=data[projection+'.qweight'];s=data[projection+'.scales']
        def stock_raw():return mx.quantized_matmul(x,q,scales=s,mode='mxfp4')
        def stock_boundary():return stock_raw().astype(mx.bfloat16).astype(mx.float32)
        def native(full=False):return candidate(x,data,projection=projection,return_partials=full,numerical_profile=PROFILE)
        raw=stock_raw();rounded=raw.astype(mx.bfloat16).astype(mx.float32);mx.eval(raw,rounded)
        controls={k:details(np,arrays[field],np.array(v))for k,field,v in [('raw','reference_preround',raw),('boundary','reference_boundary',rounded)]}
        row=dict(capture=case,replayed_stock=controls,input_BF16_roundtrip_exact=True);section['cases'].append(row);observe();write()
        if not all(v['finite']and v['exact']for v in controls.values()):
            section['required_reference_controls_exact']=False;write();raise ValueError('Stock replay differs from actual captured reference')
        cb,cr,parts=native(True);mx.eval(cb,cr,parts);a,b,c=np.array(cb),np.array(cr),np.array(parts)
        if c.dtype!=np.float32 or c.shape!=(case['partitions'],case['rows'],2304):raise ValueError('Candidate partial geometry')
        row['candidate_raw']=details(np,arrays['reference_preround'],b)
        row['candidate_boundary']=details(np,arrays['reference_boundary'],a)
        row['ordered_partial_reduction']=details(np,ordered_partials(np,c),b)
        row['candidate_partials_finite']=bool(np.isfinite(c).all())
        row['candidate_finite']=all(v.get('finite',False)for v in [row['candidate_raw'],row['candidate_boundary'],row['ordered_partial_reduction']]) and row['candidate_partials_finite']
        row['raw_and_boundary_exact']=row['candidate_raw']['exact'] and row['candidate_boundary']['exact']
        name=f'nax-profile-{index:02d}-p{case["partitions"]}-{projection}.npz';path=out/name
        with path.open('xb')as f:np.savez(f,candidate_boundary=a,candidate_raw=b,candidate_partials=c)
        row['candidate_arrays']=dict(path=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest());observe();write()
        row['timing']=timing(mx,np,{'reference':stock_boundary,PROFILE:lambda:native()},repeats=3) if row['candidate_finite'] and row['ordered_partial_reduction']['exact'] else None
        mx.synchronize();x=None;data=None;q=None;s=None;del raw,rounded,cb,cr,parts,a,b,c;observe();write()
    section['required_reference_controls_exact']=all(all(v['exact']and v['finite']for v in row['replayed_stock'].values())for row in section['cases'])
    section['candidate_finite']=all(row['candidate_finite']for row in section['cases'])
    section['ordered_reduction_exact']=all(row['ordered_partial_reduction']['exact']for row in section['cases'])
    section['candidate_raw_and_boundary_exact']=all(row['raw_and_boundary_exact']for row in section['cases'])
    section['harness_completed']=True;section['complete']=True;write();return section
