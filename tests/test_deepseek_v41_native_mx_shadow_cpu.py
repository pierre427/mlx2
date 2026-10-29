"""Reference-returning native MX shadow guards with fake MLX; no GPU use."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'scripts/research/ds41_native_mx_shadow.py'
s=importlib.util.spec_from_file_location('_native_mx_shadow_cpu',PATH)
h=importlib.util.module_from_spec(s);s.loader.exec_module(h)


def spec():
    return dict(native_mx_shadow=True,resident_discovery=True,rate_eligible=False,context_cap=16384,
        prompt_tokens=100,output_tokens=32,harness_sha256={PATH.name:h._sha(PATH)},
        expert_cache_bytes=75_000_000_000,expert_policy=dict(hot_bytes=60_000_000_000,mode='hot_original_q1'),
        checkpoint='DS41-SOURCE-CPU-FIXTURE@'+'f'*64,source_revision='89e6938ea58a0060026815c94e0ed4d33729d775',
        source_snapshot_sha256='a'*64,source_expert_math='routed_fused_fp32',extra_source_policy=dict(deepseek_source_prefill_batch='gate_up'))


@pytest.mark.parametrize('field,value',[('native_mx_shadow',False),('native_mx_shadow',1),('resident_discovery',False),
    ('rate_eligible',True),('context_cap',8192),('prompt_tokens',99),('output_tokens',16),
    ('expert_cache_bytes',74_000_000_000),('small_m_shadow',True),('native_ffn_capture',True),
    ('checkpoint',''),('harness_sha256',{})])
def test_sealed_shape_source_overlap_refusals(field,value):
    v=spec();v[field]=value
    with pytest.raises(ValueError):h._validate_spec(v,h._sha(PATH),99,0)


@pytest.mark.parametrize('rows,frontier',[(1,99),(99,1),(100,0)])
def test_non_PP_capture_refused(rows,frontier):
    with pytest.raises(ValueError):h._validate_spec(spec(),h._sha(PATH),rows,frontier)


def test_exact_half_domain_and_loss_are_observations():
    r=h._half_domain(np.asarray([[0.,-0.,.5]],np.float32));assert r['exact']and r['samples']==3
    r=h._half_domain(np.asarray([[.1,1e10]],np.float32));assert r['changed_words']==2
    assert r['first_mismatch']['coordinate']==[0,0]and not r['restored_finite']


def test_bit_compare_nonfinite_does_not_json_nan_or_tolerance_admit():
    a=np.asarray([[0.,-0.,np.nan]],np.float32);b=a.copy();b[0,1]=0.
    r=h._compare(a,b);assert r['changed_words']==1 and not r['finite']and r['max_abs_error']is None
    json.dumps(r,allow_nan=False)


def census():
    return [dict(layer=i,eligible_jobs=2,planned_jobs=3,acquired_jobs=3,independent_projection_calls=6,
        middle_checks=2,composed_down_calls=2,assembly_checks=2)for i in range(40)]


@pytest.mark.parametrize('field',['acquired_jobs','independent_projection_calls','middle_checks','composed_down_calls','assembly_checks'])
def test_census_cannot_skip_eligible_jobs_or_layers(field):
    c=census();assert h._coverage(c)==80;c[0][field]-=1
    with pytest.raises(ValueError):h._coverage(c)
    with pytest.raises(ValueError):h._coverage(census()[:-1])


def test_partial_assembly_preserves_order_and_rank_not_delta_addition():
    # Cancellation shows why adding (new-old) after a total is incorrect.
    p=np.zeros((6,5120),np.float32);p[0]=2**25;p[1]=1;p[2]=-(2**25);p[3]=1
    offset=np.arange(6,dtype=np.uint32)[None,:];chosen=np.full((1,5120),3,np.float32)
    got=h._replacement(p,offset,np.array([0],np.uint32),np.array([1],np.uint32),chosen)
    expected=np.zeros((1,5120),np.float32)
    for i in range(6):np.add(expected,chosen if i==1 else p[i:i+1],out=expected)
    assert got.tobytes()==expected.tobytes()


BF16=object()
class Array:
    def __init__(self,value,dtype=None):self.a=np.asarray(value,dtype=dtype)
    @property
    def dtype(self):return self.a.dtype
    @property
    def shape(self):return self.a.shape
    @property
    def size(self):return self.a.size
    @property
    def nbytes(self):return self.a.nbytes
    def __array__(self,dtype=None,copy=None):return np.asarray(self.a,dtype=dtype)
    def astype(self,dtype):
        if dtype is BF16:
            bits=self.a.astype(np.float32).view(np.uint32)
            return Array(((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32))
        return Array(self.a.astype(dtype))
    def __getitem__(self,key):return Array(self.a[key.a if isinstance(key,Array)else key])


@pytest.fixture
def fake_model(tmp_path,monkeypatch):
    from mlx2.adapters import deepseek_v41_source_prefill as planner,deepseek_v41_source_math as math
    from mlx2.runtime.gguf import source_middle_fusion as middle
    import mlx2.runtime.gguf as gguf
    calls=dict(native=[],half=[],reference=[],provider_received=[])
    mx=ModuleType('mlx.core');mx.__file__=str(PATH);mx.float32=np.dtype('float32');mx.bfloat16=BF16
    mx.array=Array;mx.eval=lambda *args:None
    mx.quantized_matmul=lambda x,q,**kw:backend(x,q[0])
    def backend(x,projection):
        width=5120 if projection=='down'else 2304
        result=Array(np.broadcast_to(x.a[:,0:1],(len(x.a),width)).copy())
        calls['reference'].append(result);return result
    monkeypatch.setitem(sys.modules,'mlx.core',mx)
    mlx=ModuleType('mlx');mlx.core=mx;monkeypatch.setitem(sys.modules,'mlx',mlx)
    native=ModuleType('mlx2.runtime.gguf.source_native_mx_runtime');native.__file__=str(PATH)
    def prepare(x):calls['half'].append(x);return x.astype(np.float16)
    def project(x,data,bound,*,projection):
        calls['native'].append((projection,x,data,bound))
        width=5120 if projection=='down'else 2304
        return Array(np.broadcast_to(x.a[:,0:1].astype(np.float32),(len(x.a),width)).copy())
    native.prepare_half=prepare;native.project_half=project
    monkeypatch.setitem(sys.modules,native.__name__,native);monkeypatch.setattr(gguf,'source_native_mx_runtime',native,raising=False)
    monkeypatch.setattr(middle,'middle',lambda g,u,scores,slots:g)
    original_offsets=math.ordered_offsets
    def combine(parts,values,order):
        packed=np.concatenate([p.a for p in parts],axis=0);offset=original_offsets(values,order)
        total=np.zeros((99,5120),np.float32)
        for slot in range(6):np.add(total,packed[offset[:,slot]],out=total)
        calls['backend_output']=Array(total);return calls['backend_output']
    monkeypatch.setattr(math,'ordered_combine',combine)
    class Stream:
        packed_buffers=True;_active=True;q1_binding='b'*64;hot_binding='d'*64
        artifact={'identity':{'fingerprint':'c'*64}}
        def __init__(self):self._leased={};self.count=0
        def acquire(self,prefix,eid):
            self.count+=1;data={p+'.'+f:(p,f,eid)for p in ('gate','up','down')for f in ('qweight','scales')}
            self._leased[(int(prefix.split('.')[1]),eid)]=data;return data
    stream=Stream()
    class Weights:
        expert_stream=stream;source_native_mx_policy='reference';source_prefill_batch_policy='gate_up';expert_math_policy='routed_fused_fp32'
        missing=False
        def _source_native_mx_bound(self,prefix,eid,data):
            if self.missing:return None
            proof=dict(eligible=True,source_fingerprint='c'*64,layer=0,expert=eid)
            return SimpleNamespace(attestation=SimpleNamespace(receipt=lambda:dict(proof)))
        def _prefill_batched_gate_up(self,x,values,scores,prefix,order):
            kinds={int(e):'original'if e==0 else'q1'for e in np.unique(values)}
            plan=planner.plan_jobs(values,kinds);experts=[stream.acquire(prefix,j.expert_id)for j in plan.jobs];parts=[]
            for job,data in zip(plan.jobs,experts):
                if job.kind=='original':
                    selected=x[Array(job.rows)]
                    g=mx.quantized_matmul(selected,data['gate.qweight'],scales=data['gate.scales'],mode='mxfp4')
                    u=mx.quantized_matmul(selected,data['up.qweight'],scales=data['up.scales'],mode='mxfp4')
                    mid=middle.middle(g,u,scores.reshape(-1),Array(job.locations))
                    d=mx.quantized_matmul(mid,data['down.qweight'],scales=data['down.scales'],mode='mxfp4')
                    calls['provider_received']=[g,u,d]
                    parts.append(d.astype(BF16).astype(np.float32))
                else:parts.append(Array(np.zeros((job.count,5120),np.float32)))
            return math.ordered_combine(parts,values,order)
    weights=Weights();model=SimpleNamespace(weights=weights)
    values=np.tile(np.arange(1,7,dtype=np.uint32),(99,1));values[:16]=np.arange(6,dtype=np.uint32)
    x=Array(np.zeros((99,5120),np.float32));scores=np.ones((99,6),np.float32);order=np.argsort(values.ravel(),kind='stable').astype(np.uint32)
    (tmp_path/'spec.json').write_text(json.dumps(spec()))
    # Isolate one real provider flow; independent census tests enforce40layers.
    original_coverage=h._coverage
    def one_layer(c):
        assert len(c)==1 and c[0]['eligible_jobs']==1 and c[0]['independent_projection_calls']==3
        assert c[0]['middle_checks']==c[0]['composed_down_calls']==c[0]['assembly_checks']==1
        return 1
    monkeypatch.setattr(h,'_coverage',one_layer)
    return SimpleNamespace(model=model,weights=weights,stream=stream,mx=mx,native=native,planner=planner,math=math,
        middle=middle,calls=calls,x=x,values=values,scores=scores,order=order,unit=tmp_path)


def run_fixture(f):
    with h.instrument(f.model,f.unit,'cpu_fixture',0,input_rows=99,frontier=0):
        result=f.weights._prefill_batched_gate_up(f.x,f.values,f.scores,'blk.0',f.order)
    return result,json.loads((f.unit/'native-mx-shadow-0000.json').read_text())


def test_reference_objects_original_once_prepareGU_composed_census(fake_model):
    f=fake_model;before=(f.mx.quantized_matmul,f.middle.middle,f.math.ordered_combine)
    result,r=run_fixture(f)
    assert result is f.calls['backend_output']
    assert all(a is b for a,b in zip(f.calls['reference'],f.calls['provider_received']))
    assert len(f.calls['native'])==4 and len(f.calls['half'])==3
    assert f.calls['native'][0][1]is f.calls['native'][1][1]
    assert f.stream.count==len(np.unique(f.values))
    assert r['complete']and r['reference_objects_always_returned']and r['candidate_used_by_model']is False
    assert r['cases'][0]['composed_partial_assembly']['exact']and r['maximum_diagnostic_bytes']<=h.MAX_BYTES
    assert before==(f.mx.quantized_matmul,f.middle.middle,f.math.ordered_combine)
    assert '_prefill_batched_gate_up'not in vars(f.weights)and'acquire'not in vars(f.stream)


def test_lossy_half_is_recorded_not_reference_result_replacement(fake_model):
    f=fake_model;f.x.a[0,0]=np.float32(.1)
    result,r=run_fixture(f)
    assert result is f.calls['backend_output']and r['half_domain_changed_words']>0
    assert not r['actual_half_domain_exact']and r['cases'][0]['half_domains']['actual_GU']['first_mismatch']['coordinate']==[0,0]


def test_missing_attestation_fails_coverage_without_candidate_or_extra_acquire(fake_model):
    f=fake_model;f.weights.missing=True;before=f.mx.quantized_matmul
    with pytest.raises(ValueError,match='missing CPU-loaded attestation'):run_fixture(f)
    r=json.loads((f.unit/'native-mx-shadow-0000.json').read_text())
    assert r['status']=='failed'and len(r['missing_attestations'])==1 and not f.calls['native']
    assert f.mx.quantized_matmul is before and not h._LOCK.locked()


def test_candidate_extra_acquire_refused_and_cleanup_exact(fake_model):
    f=fake_model
    def illegal(*args,**kw):return f.stream.acquire('blk.0',0)
    f.native.project_half=illegal
    with pytest.raises(RuntimeError,match='extra acquire'):run_fixture(f)
    assert json.loads((f.unit/'native-mx-shadow-0000.json').read_text())['blocked_extra_acquire_attempts']==1
    assert f.stream.count==len(np.unique(f.values))and'acquire'not in vars(f.stream)and not h._LOCK.locked()


def test_memory_or_publish_failure_restores_patches_and_lock(fake_model,monkeypatch):
    f=fake_model;before=f.mx.quantized_matmul
    monkeypatch.setattr(h,'MAX_BYTES',1024)
    with pytest.raises(MemoryError):run_fixture(f)
    assert f.mx.quantized_matmul is before and not h._LOCK.locked()


def test_cancellation_before_provider_restores_every_hook(fake_model):
    f=fake_model;original=f.mx.quantized_matmul
    with pytest.raises(KeyboardInterrupt):
        with h.instrument(f.model,f.unit,'cancel',0,input_rows=99,frontier=0):raise KeyboardInterrupt()
    assert f.mx.quantized_matmul is original and'acquire'not in vars(f.stream)and not h._LOCK.locked()
    assert json.loads((f.unit/'native-mx-shadow-0000.json').read_text())['error_type']=='KeyboardInterrupt'


def test_concurrent_calls_refused_without_ordinary_kernel_work(fake_model):
    import threading
    f=fake_model;errors=[]
    def concurrent():
        try:f.mx.quantized_matmul(None,None)
        except BaseException as error:errors.append(error)
    with pytest.raises(RuntimeError,match='CPU concurrent fixture complete'):
        with h.instrument(f.model,f.unit,'thread',0,input_rows=99,frontier=0):
            t=threading.Thread(target=concurrent);t.start();t.join()
            raise RuntimeError('CPU concurrent fixture complete')
    assert len(errors)==1 and isinstance(errors[0],RuntimeError)and not f.calls['reference']and not h._LOCK.locked()


def test_publisher_failure_does_not_leave_process_global_lock(fake_model,monkeypatch):
    f=fake_model;before=f.mx.quantized_matmul
    def refused(*args):raise OSError('publisher refused')
    monkeypatch.setattr(h,'_atomic',refused)
    with pytest.raises(OSError,match='publisher refused'):run_fixture(f)
    assert f.mx.quantized_matmul is before and'acquire'not in vars(f.stream)and not h._LOCK.locked()
