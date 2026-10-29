"""Runtime smoke recipe/tailed allocation controls; CPU only, no driver."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import ds41_native_mx_runtime_components as w
import probe_ds41_native_mx_runtime as c
from mlx2.runtime.gguf import source_native_mx_runtime as runtime
from mlx2.adapters.deepseek_v41_source_buffers import canonical_layout


def captured_frames():
    full=[]
    shared=np.arange(99*5120,dtype=np.float32).reshape(99,5120)
    for m,e in ((34,38),(32,50)):
        gu=np.arange(m*5120,dtype=np.float32).reshape(m,5120)
        mid=np.arange(m*2304,dtype=np.float32).reshape(m,2304)
        a=dict(shared_gu_input=shared,gu_input=gu,middle=mid)
        for p in ('gate','up','down'):
            a[p+'_raw']=np.zeros((m,5120 if p=='down'else 2304),np.float32)
            a[p+'_owned']=a[p+'_raw'].copy()
        full.append((dict(rows=m,layer=0,expert=e,kind='original'),a))
    case=dict(rows=98,layer=18,expert=147,kind='original')
    a=dict(activation=np.arange(98*2304,dtype=np.float32).reshape(98,2304),
        reference_raw=np.zeros((98,5120),np.float32),reference_boundary=np.zeros((98,5120),np.float32))
    return full,[(case,a)]


def test_recipe_actual_derived_and_counterfactual_scopes_no_repeat_rows():
    full,down=captured_frames();cases=w.case_matrix(np,full,down)
    assert len(cases)==18
    actual=[v for v in cases if v['captured_routed_job']]
    assert len(actual)==7
    assert all(v['expected_raw']is not None and v['expected_owner']is not None for v in actual)
    derived=[v for v in cases if v['origin']=='derived_prefix_of_real_routed_job']
    assert len(derived)==9
    for v in derived:
        source=full[0][1]['middle'if v['projection']=='down'else'gu_input']
        assert v['input'].tobytes()==source[:v['rows']].tobytes()
        assert v['expected_raw']is v['expected_owner']is None
        assert len(np.unique(v['input'],axis=0))==v['rows']
    counter=[v for v in cases if v['rows']==99]
    assert len(counter)==2 and {v['projection']for v in counter}=={'gate','up'}
    assert all(not v['captured_routed_job']and v['expected_raw']is None for v in counter)
    assert all(v['input'].tobytes()==full[0][1]['shared_gu_input'].tobytes()for v in counter)


@pytest.mark.parametrize('change',['missing_full','missing98','duplicate98'])
def test_recipe_missing_or_ambiguous_actual_shapes_refuse(change):
    full,down=captured_frames()
    if change=='missing_full':full=full[:1]
    if change=='missing98':down=[]
    if change=='duplicate98':down*=2
    with pytest.raises(ValueError):w.case_matrix(np,full,down)


@pytest.mark.parametrize('m',[16,17,31,32,34,98,99])
@pytest.mark.parametrize('projection',['gate','down'])
def test_padded_tail_dispatch_keeps_actual_cextent_and_normal_rows(m,projection):
    k,n=runtime.SHAPES[projection];normal=np.arange(m*n,dtype=np.float32).reshape(m,n)
    calls=[]
    def kernel(k_arg,n_arg):
        assert(k_arg,n_arg)==(k,n)
        def invoke(**kw):
            calls.append(kw);out=np.full(kw['output_shapes'][0],kw['init_value'],np.float32)
            out[:m]=normal
            return[out]
        return invoke
    mx=SimpleNamespace(array=np.array,uint32=np.uint32,float32=np.float32,eval=lambda *v:None)
    r=SimpleNamespace(plan=runtime.plan,_kernel=kernel)
    expert=SimpleNamespace(layout=canonical_layout('original'),backing=object())
    result=w.tail_canary(mx,np,r,np.zeros((m,k),np.float16),expert,projection,normal)
    assert result['passed']and result['padding_unchanged']and result['valid_rows_exact']
    assert calls[0]['inputs'][2].tolist()[-1]==m
    assert calls[0]['output_shapes']==[(m+17,n)]
    assert calls[0]['init_value']==w.SENTINEL
    assert calls[0]['grid']==runtime.plan(projection,m)['grid']


@pytest.mark.parametrize('corruption',['valid','padding'])
def test_tail_verdict_refuses_normal_or_padding_corruption(corruption):
    m,k,n=17,5120,2304;normal=np.ones((m,n),np.float32)
    def kernel(*args):
        def invoke(**kw):
            out=np.full(kw['output_shapes'][0],kw['init_value'],np.float32);out[:m]=normal
            out[0 if corruption=='valid'else m,0]=123.
            return[out]
        return invoke
    mx=SimpleNamespace(array=np.array,uint32=np.uint32,float32=np.float32,eval=lambda *v:None)
    result=w.tail_canary(mx,np,SimpleNamespace(plan=runtime.plan,_kernel=kernel),np.zeros((m,k),np.float16),
        SimpleNamespace(layout=canonical_layout('original'),backing=object()),'gate',normal)
    assert result['passed']is False


def row_pass():
    return dict(captured_control={'raw':{'exact':True},'owner':{'exact':True}},fresh_stock_exact=True,
        coefficient_identity=True,attestation={'eligible':True},native_finite=True,stable=True,
        second_input={'finite':True},fp64_passed=True,zero_aba_passed=True,tail_guard={'passed':True},native_dispatched=True)


def test_engineering_verdicts_exact_closed_inventory_and_tamper_failure():
    rows=[row_pass()for _ in range(18)]
    assert set(w.engineering_verdicts(rows))==w.VERDICTS and all(w.engineering_verdicts(rows).values())
    assert w.engineering_verdicts(rows[:-1])['all_cases_dispatched']is False
    for key in ('fresh_stock_exact','coefficient_identity','native_finite','stable','fp64_passed','zero_aba_passed','native_dispatched'):
        bad=copy.deepcopy(rows);bad[0][key]=False
        assert not all(w.engineering_verdicts(bad).values())
    bad=copy.deepcopy(rows);bad[0]['tail_guard']['passed']=False
    assert w.engineering_verdicts(bad)['tail_guards']is False
    bad=copy.deepcopy(rows);bad[0]['attestation']['eligible']=False
    assert w.engineering_verdicts(bad)['runtime_eligibility']is False


def valid_spec(out):
    return dict(schema=c.SCHEMA,out=str(out),job_id=out.name,native_runtime_options=dict(c.OPTIONS),
        memory_limit_bytes=c.base.MEMORY_LIMIT,host_live_reserve_bytes=c.base.HOST_RESERVE,
        host_initial_floor_bytes=c.base.HOST_INITIAL,rate_eligible=False,numerical_environment={'MLX_ENABLE_TF32':'0'})


def test_controller_spec_same_budgets_and_no_speed_claim(monkeypatch,tmp_path):
    monkeypatch.setenv('MLX_ENABLE_TF32','0');spec=valid_spec(tmp_path)
    c.validate_spec(spec,tmp_path)
    for key in ('memory_limit_bytes','host_live_reserve_bytes','host_initial_floor_bytes'):
        bad=copy.deepcopy(spec);bad[key]-=1
        with pytest.raises(ValueError):c.validate_spec(bad,tmp_path)
    bad=copy.deepcopy(spec);bad['native_runtime_options']['repeats']=4
    with pytest.raises(ValueError):c.validate_spec(bad,tmp_path)
    bad=copy.deepcopy(spec);bad['rate_eligible']=True
    with pytest.raises(ValueError):c.validate_spec(bad,tmp_path)


def test_unit_mode_refuses_unsealed_capture_inputs(tmp_path):
    with pytest.raises(SystemExit):c.main(['unit','--output',str(tmp_path),'--capture-unit',str(tmp_path)])
