"""HC component controller fixtures with tiny mocked operands, no GPU/math proof."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('hc_boundary_component_cpu',ROOT/'scripts/research/ds41_hc_boundary_components.py')
h=importlib.util.module_from_spec(s);s.loader.exec_module(h)


def test_seal_checked_before_parser_execution(tmp_path):
    harness=tmp_path/'harness';harness.mkdir();(harness/'hc_boundary_probe.py').write_text("raise AssertionError('unsealed parser executed')")
    spec=dict(checkpoint='synthetic@fixture',source_revision='fixture',source_snapshot_sha256='capture',harness_sha256={'hc_boundary_probe.py':'bad'})
    (tmp_path/'spec.json').write_text(json.dumps(spec));(tmp_path/'hc-boundary-capture.json').write_text(json.dumps({**spec,'helper_sha256':'bad'}))
    with pytest.raises(ValueError,match='seal'):h.capture_module(tmp_path)


def test_sealed_parser_load_is_cpu_only(tmp_path):
    code="def load_capture(path): return {'checkpoint':'synthetic@fixture'}, []\n";harness=tmp_path/'harness';harness.mkdir();(harness/'hc_boundary_probe.py').write_text(code)
    seal=hashlib.sha256(code.encode()).hexdigest();spec=dict(checkpoint='synthetic@fixture',source_revision='fixture',source_snapshot_sha256='capture',harness_sha256={'hc_boundary_probe.py':seal})
    (tmp_path/'spec.json').write_text(json.dumps(spec));(tmp_path/'hc-boundary-capture.json').write_text(json.dumps({**spec,'helper_sha256':seal}))
    module,path=h.capture_module(tmp_path);assert module.load_capture(path)[0]['checkpoint']=='synthetic@fixture'


def cases():
    values=[]
    for phase in ('pp99','decode1'):
        for layer in (2,20,21):
            rows=99 if phase=='pp99'else 1;branch=np.ones((1,rows,2),np.float32);res=np.ones((1,rows,4,2),np.float32)
            arrays=dict(branch=branch,residual=res,post=np.ones((1,rows,4),np.float32),comb=np.broadcast_to(np.eye(4,dtype=np.float32),(1,rows,4,4)).copy(),
                        pre=np.ones((1,rows,4),np.float32),norm_weight=np.ones(2,np.float32),updated_h=res.copy(),collapsed=branch.copy(),normalized=branch.copy())
            values.append((dict(layer=layer,phase=phase,frontier=0 if phase=='pp99'else 100,norm_eps=1e-6),arrays))
    return values


class Backend:
    gpu='mock_gpu'
    array=staticmethod(np.array)
    @staticmethod
    def default_device():return Backend.gpu
    @staticmethod
    def eval(*args):pass
    @staticmethod
    def synchronize():pass
    @staticmethod
    def get_peak_memory():return 1024


def setup(monkeypatch,tmp_path,*,negative=False,bad_saved=False):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/research'))
    packet=cases()
    if bad_saved:packet[0][1]['updated_h']*=2
    manifest=dict(checkpoint='synthetic@fixture',source_revision='fixture',source_snapshot_sha256='capture')
    capture=tmp_path/'capture.json';capture.write_text('{}')
    monkeypatch.setattr(h,'capture_module',lambda path:(SimpleNamespace(load_capture=lambda p:(manifest,packet)),capture))
    policies=('stock_frontier','inline_sequential','inline_pairwise');calls=[]
    def reference(branch,residual,*a,return_frontiers=False,**kw):
        outputs=(residual.copy(),branch.copy(),branch.copy())
        front=dict(mixed_residual=residual.copy(),expanded_raw=residual.copy(),collapse_products=residual.copy(),
                   collapsed_raw=branch.copy(),collapsed_bf16=branch.copy(),norm_raw=branch.copy())
        return(outputs,front)if return_frontiers else outputs
    def candidate(*args,collapse_policy='stock_frontier',return_frontiers=False,**kw):
        outputs,front=reference(*args,return_frontiers=True)
        if negative and collapse_policy=='inline_pairwise':outputs[1].flat[0]+=.5
        return(outputs,front)if return_frontiers else outputs
    fake=SimpleNamespace(COLLAPSE_POLICIES=policies,reference=reference,boundary=candidate,plan=lambda *a,**kw:{'synthetic_fixture':True})
    import mlx2.runtime.models as models
    monkeypatch.setattr(models,'deepseek_v41_source_hc_boundary',fake,raising=False)
    report=dict(checkpoint='synthetic@fixture',source_revision='fixture',source_snapshot_sha256='component')
    saved=[]
    return packet,report,saved,lambda p,r:saved.append(json.loads(json.dumps(r)))


def test_six_case_component_inventory_and_exact_only_timing(monkeypatch,tmp_path):
    packet,report,saved,atomic=setup(monkeypatch,tmp_path,negative=True)
    result=h.run_components(Backend,np,tmp_path,tmp_path,report,None,atomic)
    assert result['complete']and result['harness_completed']and result['required_reference_controls_exact']
    assert result['verdicts']==dict(stock_frontier_all_six_exact=True,inline_sequential_all_six_exact=True,inline_pairwise_all_six_exact=False)
    assert len(result['cases'])==6 and saved
    for case in result['cases']:
        assert set(case['timings'])=={'reference','stock_frontier','inline_sequential'}
        assert not case['variants']['inline_pairwise']['timing_admitted']
        assert len(case['timings']['stock_frontier']['samples'])==3


def test_saved_boundary_failure_retains_negative_and_refuses_interpretation(monkeypatch,tmp_path):
    packet,report,saved,atomic=setup(monkeypatch,tmp_path,bad_saved=True)
    with pytest.raises(ValueError,match='owning reference replay'):h.run_components(Backend,np,tmp_path,tmp_path,report,None,atomic)
    row=report['hc_boundary_components']['cases'][0]
    assert not row['reference_controls']['ordinary_upload_saved']['updated_h']['exact']
    assert not row['reference_controls']['inferred_packet_saved']['updated_h']['exact']
    assert not report['hc_boundary_components']['required_reference_controls_exact']


def test_capture_identity_and_inventory_refused(monkeypatch,tmp_path):
    packet,report,saved,atomic=setup(monkeypatch,tmp_path);report['checkpoint']='another'
    with pytest.raises(ValueError,match='identity'):h.run_components(Backend,np,tmp_path,tmp_path,report,None,atomic)
    report['checkpoint']='synthetic@fixture';packet.pop()
    with pytest.raises(ValueError,match='six actual'):h.run_components(Backend,np,tmp_path,tmp_path,report,None,atomic)


def test_allocator_guard_prevents_components(monkeypatch,tmp_path):
    packet,report,saved,atomic=setup(monkeypatch,tmp_path)
    class Huge(Backend):
        get_peak_memory=staticmethod(lambda:h.LIMIT+1)
    with pytest.raises(MemoryError,match='6GiB'):h.run_components(Huge,np,tmp_path,tmp_path,report,None,atomic)
    assert not report['hc_boundary_components']['cases']and saved


def test_replication_options_bound_work_and_preserve_selected_oracles(monkeypatch,tmp_path):
    packet,report,saved,atomic=setup(monkeypatch,tmp_path)
    report['hc_boundary_options']={'repeats':7,'policies':['stock_frontier','inline_sequential']}
    result=h.run_components(Backend,np,tmp_path,tmp_path,report,None,atomic)
    assert result['repeats']==7 and result['policies']==['stock_frontier','inline_sequential']
    assert set(result['verdicts'])=={'stock_frontier_all_six_exact','inline_sequential_all_six_exact'}
    for case in result['cases']:
        assert set(case['variants'])=={'stock_frontier','inline_sequential'}
        assert set(case['timings'])=={'reference','stock_frontier','inline_sequential'}
        assert all(len(v['samples'])==7 for v in case['timings'].values())


@pytest.mark.parametrize('value',[None,[],{'other':1},{'repeats':True},{'repeats':2},{'repeats':8},
    {'policies':[]},{'policies':['unknown']},{'policies':['stock_frontier','stock_frontier']},
    {'policies':'stock_frontier'}])
def test_invalid_replication_options_refuse(value):
    with pytest.raises(ValueError):h.options({'hc_boundary_options':value},('stock_frontier','inline_sequential','inline_pairwise'))
