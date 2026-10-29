"""Small CPU guards for synthetic cost functions; no model/GPU startup."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'scripts/research/ds41_synthetic_math_components.py'
spec=importlib.util.spec_from_file_location('ds41_synthetic_math_components_cpu',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_plan_uses_actual_config_shapes_bounded_seeded_inputs_no_head_crop():
    p=m.plan()
    assert p['repeats']==7 and p['seed']==410126 and p['input_rows']==1
    assert[(x['name'],x['k'],x['n'])for x in p['dense']]==list(m.DENSE)
    assert p['experts']==6 and(p['hidden'],p['intermediate'])==(5120,2304)
    assert p['resident_operand_upper_bound_bytes']<p['array_budget_bytes']==2*1024**3
    assert'cropped head'in p['head_excluded']


def test_import_does_not_construct_device_model_or_read_weights():
    tree=ast.parse(PATH.read_text())
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom))and any('mlx.core'in a.name for a in n.names)for n in tree.body)
    calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)]
    assert not any(isinstance(n.func,ast.Name)and n.func.id in('inspect_artifact','read_tensors','read_original_packed','SourceV41Weights','DeepSeekV41Model')for n in calls)
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run')
    assert [a.arg for a in run.args.args]==['mx','check_live']


def test_live_refusal_precedes_device_or_array_work():
    class NoDevice:
        def default_device(self):pytest.fail('refused live guard touched device')
    with pytest.raises(RuntimeError,match='live guard'):m.run(NoDevice(),lambda:False)


def test_noncallable_guard_refused():
    with pytest.raises(ValueError,match='callable'):m.run(object(),True)


def test_cpu_device_refused_before_math_imports():
    with pytest.raises(ValueError,match='GPU'):m.run(SimpleNamespace(default_device=lambda:'cpu',gpu='gpu'))


def test_submit_completion_intervals_are_separate_and_eval_covers_tuple(monkeypatch):
    ticks=iter((100,120,180));monkeypatch.setattr(m.time,'perf_counter_ns',lambda:next(ticks))
    calls=[];a,b=object(),object();mx=SimpleNamespace(eval=lambda *x:calls.append(x))
    value,timing=m._measure(mx,lambda:(a,[b]))
    assert calls==[(a,b)]and value==(a,[b])
    assert timing=={'host_graph_submit_seconds':20e-9,'completion_wait_seconds':60e-9,'total_seconds':80e-9}


@pytest.mark.parametrize('finite',[True,False])
def test_finite_check_is_only_execution_gate(finite):
    mx=SimpleNamespace(all=np.all,isfinite=np.isfinite)
    assert m._finite(mx,[np.array([1.]),np.array([2. if finite else np.inf])])is finite


@pytest.mark.parametrize('kind',['original','q1'])
def test_synthetic_canonical_packed_payload_no_real_weight_reader(monkeypatch,kind):
    from mlx2.adapters import deepseek_v41_source_buffers as buffers
    seen=[]
    def materialize(host):
        value=buffers.PackedExpert(host.data,host.layout,host.views());seen.append(value);return value
    monkeypatch.setattr(buffers,'materialize',materialize)
    mx=SimpleNamespace(eval=lambda *a:None)
    e=m._expert(mx,np.random.default_rng(m.SEED),kind)
    assert e.layout==buffers.canonical_layout(kind)and e is seen[0]
    for name,value in e.items():
        if name.endswith('qweight'):assert value.dtype==np.uint32 and np.any(value)
        elif kind=='original':assert np.all(value==119)
        elif name.endswith('scales'):assert np.all(value==np.float16(1/64))
        else:assert np.all(value==np.float16(-1/128))
    other=m._expert(mx,np.random.default_rng(m.SEED+1),kind)
    assert not np.shares_memory(e.backing,other.backing)


def test_MXFP8_plan_adds_actual_packed_shapes_without_dense_replacement():
    p=m.plan()
    assert p['added_MXFP8_operand_bytes']==95_969_280
    assert p['resident_operand_upper_bound_bytes']==1_075_388_416
    for entry in p['quantized_dense']:
        assert entry['qweight_shape']==[entry['n'],entry['k']//4]
        assert entry['scales_shape']==[entry['n'],entry['k']//32]
        assert(entry['qweight_dtype'],entry['scales_dtype'],entry['mode'])==('uint32','uint8','mxfp8')
        assert entry['bf16_optional']is True


def test_MXFP8_payload_is_finite_byte_domain_with_source_shared_scale_tiles():
    q,s=m._fp8_payload(np.random.default_rng(m.SEED),64,35)
    assert q.shape==(35,16)and q.dtype==np.uint32
    assert s.shape==(35,2)and s.dtype==np.uint8
    codes=q.view(np.uint8)
    assert np.all((codes&0x7f)<=0x40)and np.any(codes&0x80)
    assert np.all((119<=s)&(s<=122))
    assert np.array_equal(s[:32],np.broadcast_to(s[0],(32,2)))
    assert np.array_equal(s[32:],np.broadcast_to(s[32],(3,2)))
    other=m._fp8_payload(np.random.default_rng(m.SEED),64,35)
    assert np.array_equal(q,other[0])and np.array_equal(s,other[1])


@pytest.mark.parametrize('k,n',[(0,32),(63,32),(True,32),(64,0)])
def test_MXFP8_invalid_geometry_refused_before_payload(k,n):
    with pytest.raises(ValueError):m._fp8_payload(np.random.default_rng(m.SEED),k,n)


def test_actual_Source_QMM_call_keeps_activation_and_shared_weight_identity():
    x,q,s=object(),object(),object();out=object();calls=[]
    def qmm(a,b,**kwargs):calls.append((a,b,kwargs));return out
    assert m._qmm(SimpleNamespace(quantized_matmul=qmm),x,q,s)is out
    assert calls==[(x,q,{'scales':s,'mode':'mxfp8'})]
