"""CPU guards only; synthetic wrapper contracts, no GPU/operator parity claim."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace,ModuleType
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'scripts/research/ds41_synthetic_down_stage_components.py'
spec=importlib.util.spec_from_file_location('synthetic_down_stage_cpu',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_plan_is_bounded_synthetic_full_stage_and_old_negative_control_present():
    p=m.plan()
    assert p['case_count']==50 and p['repeats']==7 and p['batch_size']==15
    assert p['resident_expert_payload_bytes']==381_542_400
    assert p['conservative_operand_and_scratch_bound_bytes']<p['array_budget_bytes']==3*1024**3
    assert p['checkpoint']is None and m.VARIANTS==('per_expert_stock_concat','old_worklist_mapped','direct_worklist_mapped')
    assert all(p['maximum_expert_bindings']<=15 for p in p['profiles'])
    assert max(p['middle_bytes']for p in p['profiles'])==90*2304*4


@pytest.mark.parametrize('profile',m.PROFILES)
def test_routes_are_realizable_six_distinct_ids_random_slots_exact_counts(profile):
    _,kinds,counts=profile
    values,ids=m._routes(np,np.random.default_rng(m.SEED),counts)
    assert values.dtype==np.uint32 and values.shape==(sum(counts)//6,6)
    assert all(len(set(map(int,row)))==6 for row in values)
    assert [int(np.count_nonzero(values==e))for e in ids]==list(counts)
    assert not np.array_equal(values,np.sort(values,axis=1))
    from mlx2.adapters.deepseek_v41_source_prefill import plan_jobs
    p=plan_jobs(values,dict(zip(ids,kinds)),15)
    assert not p.fallback_job_indices and len(p.jobs)==15
    assert sum(sum(b.row_counts)for b in p.batches)==values.size
    assert max(len(b.job_indices)for b in p.batches)<=15
    other,_=m._routes(np,np.random.default_rng(m.SEED),counts)
    assert np.array_equal(values,other)


@pytest.mark.parametrize('counts',[(1,)*6,(4,2),(2,)*7,(0,)*6])
def test_unrealizable_or_malformed_routes_fail(counts):
    # A one-token fixture could be valid as a graph, but is outside PP planner;
    # exercise malformed/infeasible cases here only.
    if counts==(1,)*6:
        v,_=m._routes(np,np.random.default_rng(m.SEED),counts);assert v.shape==(1,6)
    else:
        with pytest.raises(ValueError):m._routes(np,np.random.default_rng(m.SEED),counts)


def test_import_and_timed_closure_have_no_device_startup_or_internal_eval():
    tree=ast.parse(PATH.read_text())
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom))and any('mlx.core'in a.name for a in n.names)for n in tree.body)
    for fn in ('_stage','_prepare','_per_expert_parts','_assembly'):
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==fn)
        assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr in('eval','item','synchronize')for n in ast.walk(node))
    assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id in('inspect_artifact','read_tensors','read_original_packed','SourceV41Weights')for n in ast.walk(tree))


def test_live_and_device_fail_before_runtime_imports():
    class NoDevice:
        def default_device(self):pytest.fail('live refusal touched device')
    with pytest.raises(RuntimeError,match='live guard'):m.run(NoDevice(),lambda:False)
    with pytest.raises(ValueError,match='callable'):m.run(object(),False)
    with pytest.raises(ValueError,match='GPU'):m.run(SimpleNamespace(default_device=lambda:'cpu',gpu='gpu'))


def test_timer_one_final_fence_and_separate_submit_completion(monkeypatch):
    times=iter((10,30,80));monkeypatch.setattr(m.time,'perf_counter_ns',lambda:next(times))
    calls=[];value=object()
    result,t=m._measure(SimpleNamespace(eval=lambda x:calls.append(x)),lambda:value)
    assert result is value and calls==[value]
    assert t==dict(host_graph_submit_seconds=20e-9,completion_wait_seconds=50e-9,total_seconds=70e-9)


def test_output_identity_distinguishes_signedzero_and_reports_actual_index():
    a=np.array([[0.,1.],[-1.,2.]],np.float32);b=a.copy();b[0,0]=-0.
    d=m._identify(np,a,b)
    assert not d['raw_bits_exact']and d['different_words']==1 and d['first_mismatch']['index']==[0,0]
    assert d['sha256']!=d['reference_sha256']
    assert m._identify(np,a,a)['raw_bits_exact']


@pytest.mark.parametrize('kind',['original','q1'])
def test_synthetic_weights_independent_canonical_no_file_reads(monkeypatch,kind):
    from mlx2.adapters import deepseek_v41_source_buffers as buffers
    monkeypatch.setattr(buffers,'materialize',lambda h:buffers.PackedExpert(h.data,h.layout,h.views()))
    mx=SimpleNamespace(eval=lambda *x:None)
    e=m._expert(mx,np,np.random.default_rng(m.SEED),kind)
    f=m._expert(mx,np,np.random.default_rng(m.SEED+1),kind)
    assert not np.shares_memory(e.backing,f.backing)
    if kind=='original':assert e.layout==buffers.canonical_layout(kind)
    else:
        from mlx2.runtime.gguf.source_down_worklist import native_q1_layout
        assert e.layout==native_q1_layout() and e.layout!=buffers.canonical_layout('q1')
        assert {t.name:t.offset for t in e.layout.tensors}==m.plan()['q1_payload_offsets']
    assert np.any(e['down.qweight'])
    if kind=='original':assert np.all(e['down.scales']==119)
    else:assert np.all(e['down.scales']==np.float16(1/64))and np.all(e['down.biases']==np.float16(-1/128))


@pytest.mark.parametrize('profile',m.PROFILES)
def test_fresh_wrapper_routes_parts_and_bucketed_assembly_same_order(monkeypatch,profile):
    """A scalar CPU stand-in tests row mapping, not the kernel arithmetic."""
    from mlx2.adapters import deepseek_v41_source_math as math
    from mlx2.runtime.gguf import source_q1,source_down_prefill,source_prefill_combine
    calls=[]
    def raw(x,q,**kwargs):
        calls.append(('raw',x.shape[0],q,kwargs))
        return np.broadcast_to((x[:,0]*q)[:,None],(len(x),5120)).copy()
    def ordered(parts,values,order,**kwargs):
        flat=np.concatenate(parts);offsets=math.ordered_offsets(values,order)
        out=np.zeros((len(values),5120),np.float32)
        for slot in range(6):out=np.add(out,flat[offsets[:,slot]])
        return out
    def mapped(parts,values,order,descriptors):
        byid={};count=0
        for part,groups in zip(parts,descriptors):
            start=0
            for e,n in groups:byid[e]=part[start:start+n];start+=n;count+=n
            assert start==len(part)
        assert count==values.size
        return ordered([byid[e]for e in sorted(byid)],values,order)
    def pp(x,experts,counts,**kwargs):
        calls.append(('pp',tuple(counts),tuple(e['down.qweight']for e in experts),id(x),kwargs))
        cursor=0;parts=[]
        for e,n in zip(experts,counts):parts.append(raw(x[cursor:cursor+n],e['down.qweight']));cursor+=n
        return np.concatenate(parts)
    monkeypatch.setattr(math,'ordered_combine',ordered)
    monkeypatch.setattr(source_q1,'matmul',lambda x,q,s,b:raw(x,q))
    monkeypatch.setattr(source_down_prefill,'pp_down',pp)
    monkeypatch.setattr(source_prefill_combine,'combine_mapped',mapped)
    def prepare(parts,values,order,descriptors):
        calls.append(('prepare_packet',len(parts)))
        return(parts,values,order,descriptors)
    def consume(parts,packet):
        assert parts is packet[0]
        return mapped(*packet)
    monkeypatch.setattr(source_prefill_combine,'prepare_mapped',prepare,raising=False)
    monkeypatch.setattr(source_prefill_combine,'combine_mapped_prepared',consume,raising=False)
    module=ModuleType('mlx2.runtime.gguf.source_down_worklist');module.pp_down=pp
    monkeypatch.setitem(sys.modules,module.__name__,module)
    mx=SimpleNamespace(quantized_matmul=raw,bfloat16=np.float32,float32=np.float32,concatenate=np.concatenate)
    _,kinds,counts=profile;values,ids=m._routes(np,np.random.default_rng(m.SEED),counts)
    bank={e:{'down.qweight':np.float32(e+1),'down.scales':object(),'down.biases':object()}for e in ids}
    middle=np.broadcast_to(np.arange(sum(counts),dtype=np.float32)[:,None],(sum(counts),2304))
    mapping=dict(zip(ids,kinds))
    outputs=[m._stage(mx,np,values,mapping,middle,bank,v)for v in m.VARIANTS]
    assert all(np.array_equal(outputs[0],v)for v in outputs[1:])
    assert any(c[0]=='pp'for c in calls)
    p,parts,order=m._prepare(np,values,mapping,middle,bank)
    owned=m._per_expert_parts(mx,p,parts,bank);descriptors=tuple(((j.expert_id,j.count),)for j in p.jobs)
    for variant in m.ASSEMBLY_VARIANTS:assert np.array_equal(m._assembly(mx,owned,values,order,descriptors,variant),outputs[0])
    producer=[]
    for b in p.batches:
        owner=np.concatenate([parts[i]for i in b.job_indices])
        producer.append((tuple((p.jobs[i].expert_id,p.jobs[i].count)for i in b.job_indices),owner))
    owner_ids={id(o)for _,o in producer};begin=len(calls)
    mx.concatenate=lambda *a,**k:pytest.fail('borrowed scope repacked producer input')
    for variant in m.VARIANTS:
        assert np.array_equal(m._stage(mx,np,values,mapping,middle,bank,variant,producer),outputs[0])
    assert all(c[3]in owner_ids for c in calls[begin:]if c[0]=='pp')
    if profile[0]in ('q1_15_ragged','mixed15_ragged'):
        def no_mapped(*a,**k):pytest.fail('Q1 hybrid used mapped combine')
        monkeypatch.setattr(source_prefill_combine,'combine_mapped',no_mapped)
        begin=len(calls)
        for variant in m.HYBRID_VARIANTS:
            assert np.array_equal(m._stage(mx,np,values,mapping,middle,bank,variant,producer),outputs[0])
        pp_calls=[c for c in calls[begin:]if c[0]=='pp']
        assert len(pp_calls)==2 and all(c[3]in owner_ids for c in pp_calls)
    if profile[0]in ('original15_R2','original15_R6'):
        out=m._stage(mx,np,values,mapping,middle,bank,'direct_worklist_mapped',producer,traversal='stock')
        assert np.array_equal(out,outputs[0])
        assert any(c[4]=={'traversal':'stock'}for c in calls if c[0]=='pp')


@pytest.mark.parametrize('profile',m.ASSEMBLY_PROFILES)
def test_large_assembly_profiles_cover_real_routes_and_multiple_carries(profile):
    name,counts=profile
    values,ids=m._routes(np,np.random.default_rng(m.SEED),counts)
    assert values.shape==(sum(counts)//6,6)and max(ids)<384
    assert all(len(set(map(int,row)))==6 for row in values)
    assert [np.count_nonzero(values==e)for e in ids]==list(counts)
    from mlx2.runtime.gguf.source_prefill_combine import mapped_plan
    order=np.argsort(values.reshape(-1),kind='stable')
    descriptors=tuple(((e,c),)for e,c in zip(ids,counts))
    chunks=mapped_plan(values,order,descriptors)
    assert len(chunks)==(len(counts)+27)//28>1
    assert all(len(c['parts'])<=28 for c in chunks)
    assert chunks[0]['first']and not any(c['first']for c in chunks[1:])


def test_borrowed_producer_validates_fresh_jobs_and_retains_owner_identity():
    from mlx2.adapters.deepseek_v41_source_prefill import plan_jobs
    _,kinds,counts=m.PROFILES[-1];values,ids=m._routes(np,np.random.default_rng(m.SEED),counts)
    p=plan_jobs(values,dict(zip(ids,kinds)));producer=[]
    for b in p.batches:
        owner=np.zeros((sum(b.row_counts),2304),np.float32)
        producer.append((tuple((p.jobs[i].expert_id,p.jobs[i].count)for i in b.job_indices),owner))
    segments,owners=m._borrow_producer(p,producer)
    assert all(a is b[1]for a,b in zip(owners,producer))
    assert all(any(np.shares_memory(s,o)for o in owners)for s in segments)
    bad=list(producer);bad[0]=(((999,1),),bad[0][1])
    with pytest.raises(ValueError,match='fresh route plan'):m._borrow_producer(p,bad)


@pytest.mark.parametrize('width',m.WIDTH_SWEEP)
def test_width_sweep_splits_same_fifteen_expert_routes_without_repacking(width):
    _,kinds,counts=m.PROFILES[0];values,ids=m._routes(np,np.random.default_rng(m.SEED),counts)
    middle=np.zeros((sum(counts),2304),np.float32)
    p,segments,_=m._prepare(np,values,dict(zip(ids,kinds)),middle,{},batch_size=width)
    assert len(p.jobs)==15 and len(p.batches)==(15+width-1)//width
    assert sum(sum(b.row_counts)for b in p.batches)==sum(counts)
    assert max(len(b.job_indices)for b in p.batches)<=width
    assert all(np.shares_memory(x,middle)for x in segments)


def test_direct_stage_prepares_packet_inside_fresh_timed_closure():
    node=next(n for n in ast.parse(PATH.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='_stage')
    names=[n.func.id for n in ast.walk(node)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)]
    assert 'prepare_mapped'in names and 'combine_mapped_prepared'in names
