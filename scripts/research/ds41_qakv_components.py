"""Original-weight/synthetic-input Qa/KV components; caller owns GPU lease.

No serving execution or GPU work on import. Only one layer's coefficients
remain resident. Captured Q_b/Wo_b evidence supplies identity, not Qa/KV inputs.
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
import gc
import hashlib
from pathlib import Path

LIMIT = 6 << 30
ORIGINAL_FP = 'c30fd25e3cc79ba7044686fa3f9a7a1c0f514f77ee6293d2f5f62bc409e02ddb'
PHASES = (('pp99',99),('prompt_boundary1',1),('decode1',1))
_QUARANTINE = []


def synthetic_input(np, layer, phase, rows, width=5120):
    """Finite F32 values around power-of-two and FP8 rounding boundaries."""
    if phase not in dict(PHASES) or dict(PHASES)[phase]!=rows or not 0<=layer<40 or width%32:
        raise ValueError('Qa/KV synthetic geometry')
    seed=4102600+layer*7+list(dict(PHASES)).index(phase)
    rng=np.random.default_rng(seed)
    x=rng.normal(0,.35,(1,rows,width)).astype(np.float32)
    groups=x.reshape(-1,32)
    pattern=np.array([0.,-0.,1.,-1.,1.0625,-1.0625,1.1875,-1.1875,
                      .001953125,-.001953125,448.,-448.,224.,-224.,.5,-.5],np.float32)
    groups[:,:16]=pattern
    groups[:,16]=np.nextafter(np.float32(1.0625),np.float32(0))
    groups[:,17]=np.nextafter(np.float32(1.0625),np.float32(2))
    scale=np.exp2((np.arange(len(groups))%9)-4).astype(np.float32)
    groups*=scale[:,None]
    assert np.isfinite(x).all()
    return x,dict(kind='deterministic synthetic F32 with adversarial FP8-binade/midpoint patterns',
                  seed=seed,shape=list(x.shape),sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                  captured_trajectory=False)


def provider(cls, bank):
    """Use production math methods without constructing dense/expert stores."""
    w=cls.__new__(cls);w._quant=bank;w._ensure_dense=w._ready=lambda:None
    w.scheduled_dense_calls={};w.source_dense_qat_policy='fused_fp32'
    w.source_qakv_policy='qat_reuse';w.source_bf16_loads_policy='reference'
    return w


def capture_preparations(w, function):
    had='_dense_input' in w.__dict__;prior=w.__dict__.get('_dense_input');original=w._dense_input
    values=[]
    def prepare(x):
        result=original(x);values.append(result);return result
    w._dense_input=prepare
    try:return function(),values
    finally:
        if had:w._dense_input=prior
        else:del w._dense_input


def counter_delta(before, after):
    return {k:after.get(k,0)-before.get(k,0)for k in set(before)|set(after)if after.get(k,0)!=before.get(k,0)}


def raw_check(np, a, b):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or a.dtype!=np.float32 or b.dtype!=np.float32:
        raise ValueError('Qa/KV F32 shape/dtype mismatch')
    a,b=np.ascontiguousarray(a),np.ascontiguousarray(b)
    av,bv=a.view(np.uint32).ravel(),b.view(np.uint32).ravel();bad=np.flatnonzero(av!=bv)
    result=dict(exact=not len(bad),changed_words=int(len(bad)),shape=list(a.shape),reference_finite=bool(np.isfinite(a).all()),candidate_finite=bool(np.isfinite(b).all()),
                reference_sha256=hashlib.sha256(a.tobytes()).hexdigest(),candidate_sha256=hashlib.sha256(b.tobytes()).hexdigest())
    if len(bad):
        i=int(bad[0]);result['first_mismatch']=dict(coordinate=[int(v)for v in np.unravel_index(i,a.shape)],
            reference_bits=f'{int(av[i]):08x}',candidate_bits=f'{int(bv[i]):08x}')
    return result


def run_case(mx,np,w,x,a,b,boundary,compare):
    before=dict(w.scheduled_dense_calls)
    reference,independent=capture_preparations(w,lambda:(w.linear(a)(x),w.linear(b)(x)))
    after=dict(w.scheduled_dense_calls)
    candidate,shared=capture_preparations(w,lambda:w.paired_project(x,a,b,rounded=False))
    last=dict(w.scheduled_dense_calls)
    if len(independent)!=2 or len(shared)!=1 or len(reference)!=2 or len(candidate)!=2:
        raise ValueError('Qa/KV independent2/shared1 preparation inventory')
    rb=tuple(boundary(v)for v in reference);cb=tuple(v.astype(mx.bfloat16).astype(mx.float32)for v in candidate)
    mx.eval(*independent,*shared,*reference,*candidate,*rb,*cb)
    checks={name:raw_check(np,left,right)for name,left,right in [
        ('independent_qat',independent[0],independent[1]),('shared_qat_qa',independent[0],shared[0]),
        ('shared_qat_kv',independent[1],shared[0]),('qa_raw',reference[0],candidate[0]),
        ('kv_raw',reference[1],candidate[1]),('qa_BF16',rb[0],cb[0]),('kv_BF16',rb[1],cb[1])]}
    reference_counts=counter_delta(before,after);candidate_counts=counter_delta(after,last)
    if (reference_counts.get('source_dense_qat_fused')!=2 or reference_counts.get('official_mxfp8_act32')!=2
            or candidate_counts.get('source_dense_qat_fused')!=1 or candidate_counts.get('official_mxfp8_act32')!=2
            or candidate_counts.get('source_qakv_qat_reuse_pairs')!=1 or candidate_counts.get('source_qakv_input_qat_calls_avoided')!=1):
        raise ValueError('Qa/KV executed QAT/QMM counter contract')
    if not all(c['reference_finite']for c in checks.values()):raise ValueError('Qa/KV nonfinite synthetic reference')
    exact=all(c['exact']and c['candidate_finite']for c in checks.values())
    row=dict(checks=checks,all_exact=exact,reference_counter_delta=counter_delta(before,after),
             candidate_counter_delta=counter_delta(after,last),timing_admitted=exact)
    # Same concatenation on both paths solely gives the common timing helper
    # one result array; production serves two arrays and does not concatenate.
    functions={'reference':lambda:mx.concatenate(tuple(boundary(v)for v in (w.linear(a)(x),w.linear(b)(x))),axis=-1)}
    if exact:functions['qat_reuse']=lambda:mx.concatenate(w.paired_project(x,a,b,rounded=True),axis=-1)
    row['timings']=compare(mx,np,functions,repeats=3)
    if exact and not row['timings']['qat_reuse']['exact']:
        raise ValueError('Qa/KV rounded production API differs during timing precheck')
    return row


def release_layer(mx,w):
    """Fence before dropping owners; preserve them if completion fails."""
    try:mx.synchronize()
    except BaseException:
        _QUARANTINE.append(w);raise
    w._quant.clear();gc.collect();mx.clear_cache()


def run_components(mx,np,captureDir,out,report,compare,atomic):
    if mx.default_device()!=mx.gpu:raise RuntimeError('Qa/KV components require caller-owned GPU')
    if 'qakv_components'in report:raise ValueError('Refusing to overwrite Qa/KV evidence')
    from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact,read_tensors
    from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
    from ds41_attention_projection_components import capture_module,coefficient_payload,upload_coefficients,_boundary
    _,capture_path,capture_spec=capture_module(captureDir)
    for key in ('checkpoint','source_revision'):
        if not report.get(key)or report[key]!=capture_spec.get(key):raise ValueError('Qa/KV component reference identity mismatch')
    if not report.get('source_snapshot_sha256'):raise ValueError('Qa/KV component source snapshot missing')
    model_path=report.get('model')
    if not model_path:raise ValueError('Qa/KV original model path missing')
    artifact=inspect_artifact(model_path)
    if artifact['identity']['fingerprint']!=ORIGINAL_FP:raise ValueError('Qa/KV original artifact fingerprint mismatch')
    boundary,boundary_scope=_boundary(mx,capture_spec);out=Path(out)
    section=report['qakv_components']=dict(schema='mlx2.ds41-qakv-components.v1',checkpoint=report['checkpoint'],
        source_revision=report['source_revision'],component_source_snapshot_sha256=report['source_snapshot_sha256'],
        identity_reference_capture=str(capture_path),identity_reference_capture_sha256=hashlib.sha256(capture_path.read_bytes()).hexdigest(),
        original_artifact_fingerprint=ORIGINAL_FP,activation_source='synthetic deterministic/adversarial; NOT captured Qa/KV trajectory',
        scope='40 original layers × PP99/prompt-boundary1/decode1 synthetic operator cases; no model execution',
        qualified=False,performance_qualified=False,candidate_used_by_model=False,maximum_allocator_bytes=LIMIT,
        all_dense_provider_constructed=False,maximum_resident_coefficient_layers=1,loaded_coefficients=[],cases=[],
        boundary_scope=boundary_scope,timing_scope='3 interleaved repeats only exact pairs; includes QAT+twoQMM+separateBF16+common diagnostic concat; no model rate',
        complete=False,harness_completed=False,required_reference_controls_exact=False,verdicts={})
    def write():atomic(out/'results.json',report)
    def observe():
        peak=int(mx.get_peak_memory());section['peak_mlx_bytes']=max(section.get('peak_mlx_bytes',0),peak)
        report['component_peak_mlx_bytes']=max(report.get('component_peak_mlx_bytes',0),peak)
        if peak>LIMIT:write();raise MemoryError('Qa/KV components exceeded6GiB')
    write();observe()
    for layer in range(40):
        aliases=(f'blk.{layer}.attn_q_a.weight',f'blk.{layer}.attn_kv.weight')
        bank={};w=provider(SourceV41Weights,bank)
        try:
            for alias in aliases:
                q,s,record=coefficient_payload(np,artifact,alias,read_tensors)
                bank[alias]=upload_coefficients(mx,q,s);section['loaded_coefficients'].append(record);del q,s
            observe()
            expected_bytes=sum(q.nbytes+s.nbytes for q,s in bank.values())
            if expected_bytes!=9461760:raise ValueError('Qa/KV actual coefficient geometry/bytes')
            for phase,rows in PHASES:
                host,metadata=synthetic_input(np,layer,phase,rows);x=mx.array(host,mx.float32);mx.eval(x);del host
                try:
                    row=run_case(mx,np,w,x,*aliases,boundary,compare)
                    row.update(layer=layer,phase=phase,rows=rows,input=metadata,coefficient_bytes=expected_bytes)
                    section['cases'].append(row);observe();write()
                finally:
                    mx.synchronize();del x
        finally:release_layer(mx,w)
        del w,bank
    section['required_reference_controls_exact']=all(c['checks']['independent_qat']['exact']for c in section['cases'])
    section['verdicts']={'all120_raw_QAT_and_BF16_exact':len(section['cases'])==120 and all(c['all_exact']for c in section['cases'])}
    section['complete']=section['harness_completed']=True
    section['loaded_alias_count']=len(section['loaded_coefficients'])
    section['uploaded_coefficient_bytes_total']=sum(c['uploaded_coefficient_bytes']for c in section['loaded_coefficients'])
    observe();write();return section
