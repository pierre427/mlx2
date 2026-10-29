"""CPU actual-provider orchestration guards; no GPU/dot parity claim."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'scripts/research/ds41_synthetic_q1_provider.py'
spec=importlib.util.spec_from_file_location('synthetic_q1_provider_cpu',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_plan_has_four_realizable_profiles_nativeQ1_and_subGiB():
    p=m.plan();assert p['case_count']==8 and p['repeats']==7 and p['checkpoint']is None
    assert p['conservative_operand_scratch_bound_bytes']<p['array_budget_bytes']==1024**3
    assert p['physical_layouts']['q1']=='native_q1_v1'
    assert[p['expected_tiny_jobs']for p in p['profiles']]==[15,8,0,6]
    assert[p['expected_tiny_rows']for p in p['profiles']]==[30,18,0,12]
    util=m._utilities()
    for _,_,counts in m.PROFILES:
        values,_=util._routes(np,np.random.default_rng(m.SEED),counts)
        assert all(len(set(row))==6 for row in values)


def test_module_import_and_provider_construction_never_loads_model():
    tree=ast.parse(PATH.read_text())
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom))and any('mlx.core'in a.name for a in n.names)for n in tree.body)
    calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)]
    assert not any(isinstance(n.func,ast.Name)and n.func.id in('inspect_artifact','read_tensors','read_original_packed','SourceV41Weights')for n in calls)
    assert any(isinstance(n.func,ast.Attribute)and n.func.attr=='__new__'for n in calls)


def test_live_device_guard_refusals_precede_operand_creation():
    with pytest.raises(ValueError,match='callable'):m.run(object(),True)
    class NoDevice:
        def default_device(self):pytest.fail('live-refused device access')
    with pytest.raises(RuntimeError,match='live guard'):m.run(NoDevice(),lambda:False)
    with pytest.raises(ValueError,match='GPU'):m.run(SimpleNamespace(default_device=lambda:'cpu',gpu='gpu'))


def test_stream_window_cleanup_and_acquire_require_ownership():
    e=SimpleNamespace(layout=SimpleNamespace(kind='q1'));s=m.PreownedStream({3:e})
    with pytest.raises(RuntimeError,match='outside'):s.acquire('blk.0',3)
    with pytest.raises(LookupError):
        with s.window():
            assert s.acquire('blk.0',3)is e
            with pytest.raises(RuntimeError,match='Nested'):
                with s.window():pass
            raise LookupError
    assert not s.active;s.reset();assert s.acquired==[]and s.stats=={'lookups':0}


def cpu_case(monkeypatch,profile):
    mx=ModuleType('mlx.core');mx.array=np.array;mx.float32=np.float32;mx.bfloat16=np.float32;mx.uint32=np.uint32
    mx.eval=lambda *a:None
    mx.quantized_matmul=lambda x,q,**kw:np.broadcast_to((x[:,0]*q)[:,None],(len(x),5120)).copy()
    mlx=ModuleType('mlx');mlx.core=mx
    monkeypatch.setitem(sys.modules,'mlx',mlx);monkeypatch.setitem(sys.modules,'mlx.core',mx)
    from mlx2.adapters import deepseek_v41_source_math as math,deepseek_v41_source_prefill_down as down
    from mlx2.adapters.deepseek_v41_source_prefill import plan_jobs
    from mlx2.runtime.gguf import source_mxfp4_pair,source_middle_fusion,source_q1
    monkeypatch.setattr(source_mxfp4_pair,'_device_supported',lambda:True)
    monkeypatch.setattr(source_middle_fusion,'exp_spelling',lambda:'metal::precise::exp')
    monkeypatch.setattr(source_q1,'matmul',lambda x,q,s,b:mx.quantized_matmul(x,q))
    def tiny(x,experts,counts,**kw):
        out=[];start=0
        for e,n in zip(experts,counts):out.append(mx.quantized_matmul(x[start:start+n],e['down.qweight']));start+=n
        for suffix,n in (('batches',1),('jobs',len(experts)),('rows',sum(counts)),('output_views',len(experts))):
            key=down.TINY_COUNTER+suffix;kw['counters'][key]=kw['counters'].get(key,0)+n
        return np.concatenate(out)
    monkeypatch.setattr(down,'maybe_tiny_project',tiny)
    def combine(parts,values,order,**kw):
        packed=np.concatenate(parts);offsets=math.ordered_offsets(values,order)
        out=np.zeros((len(values),5120),np.float32)
        for slot in range(6):out=np.add(out,packed[offsets[:,slot]])
        return out
    monkeypatch.setattr(math,'ordered_combine',combine)
    name,kinds,counts=profile;util=m._utilities();values,ids=util._routes(np,np.random.default_rng(m.SEED),counts)
    class E(dict):pass
    bank={}
    for e,k in zip(ids,kinds):
        data=E({'down.qweight':np.float32(e+1),'down.scales':object()})
        if k=='q1':data['down.biases']=object()
        data.layout=SimpleNamespace(kind=k);bank[e]=data
    p=plan_jobs(values,dict(zip(ids,kinds)));middle=np.broadcast_to(np.arange(sum(counts),dtype=np.float32)[:,None],(sum(counts),2304))
    segments={};start=0
    for job in p.jobs:segments[job.expert_id]=middle[start:start+job.count];start+=job.count
    owners={}
    for b in p.batches:
        owners[(tuple(id(bank[p.jobs[i].expert_id])for i in b.job_indices),b.row_counts)]=np.concatenate([segments[p.jobs[i].expert_id]for i in b.job_indices])
    fixture=dict(values=values,order=np.argsort(values.ravel(),kind='stable'),x=np.zeros((len(values),5120),np.float32),scores=np.ones(values.shape,np.float32)/6,owners=owners,plan=p)
    return mx,util,bank,fixture


@pytest.mark.parametrize('profile',m.PROFILES)
def test_real_provider_method_reference_vs_tiny_counts_outputs_and_restoration(monkeypatch,profile):
    mx,util,bank,f=cpu_case(monkeypatch,profile)
    from mlx2.runtime.gguf import source_gate_up_prefill as gu
    before=gu.pp_middle;outputs=[]
    for policy in m.POLICIES:
        w=m._provider(bank,policy)
        result,t,control=m._execute(mx,util,w,f);outputs.append(result)
        assert gu.pp_middle is before and not w.expert_stream.active
        assert control['exact_acquire_census']and t['total_seconds']>=0
        assert control['expected']==m._expected(f['plan'],policy!='reference')
        if policy!='reference':
            assert control['expected']['jobs']==next(p['expected_tiny_jobs']for p in m.plan()['profiles']if p['name']==profile[0])
        # Repeated calls reset counters rather than accumulating across samples.
        _,_,again=m._execute(mx,util,w,f);assert again['counters']==control['counters']
    assert np.array_equal(outputs[0],outputs[1])


def test_gu_patch_restored_on_provider_or_completion_error(monkeypatch):
    mx,util,bank,f=cpu_case(monkeypatch,m.PROFILES[0])
    from mlx2.runtime.gguf import source_gate_up_prefill as gu
    before=gu.pp_middle;w=m._provider(bank,'reference')
    mx.eval=lambda *a:(_ for _ in ()).throw(RuntimeError('fence refused'))
    with pytest.raises(RuntimeError,match='fence refused'):m._execute(mx,util,w,f)
    assert gu.pp_middle is before and not w.expert_stream.active


def test_gpu_run_never_patches_identity_or_device_eligibility():
    node=ast.parse(PATH.read_text())
    attrs=[n.attr for n in ast.walk(node)if isinstance(n,ast.Attribute)and isinstance(n.ctx,ast.Store)]
    assert attrs.count('pp_middle')==2
    assert not set(attrs)&{'tiny_implementation_identity','_device_supported','maybe_tiny_project','eligible'}
