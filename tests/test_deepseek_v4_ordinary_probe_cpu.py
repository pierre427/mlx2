"""Ordinary harness guards and direct cache-branch CPU diagnostic; no GPU load."""
import importlib.util
from pathlib import Path
import sys
import json
import numpy as np
import pytest
import mlx.core as mx
mx.set_default_device(mx.cpu)

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/research'))
spec=importlib.util.spec_from_file_location('ordinary_probe',ROOT/'scripts/research/probe_deepseek_v4_ordinary.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def test_model_probe_refuses_without_explicit_gpu_and_live_window(tmp_path,monkeypatch):
    monkeypatch.setattr(probe,'source_receipt',lambda:pytest.fail('source before ownership'))
    with pytest.raises(SystemExit): probe.main(['--output',str(tmp_path/'new'),'--model-path',str(tmp_path/'model')])
    with pytest.raises(SystemExit): probe.main(['--output',str(tmp_path/'new'),'--model-path',str(tmp_path/'model'),'--run-gpu'])
    assert not (tmp_path/'new').exists()


def test_ownership_failure_happens_before_adapter_import_or_output(tmp_path,monkeypatch):
    def refuse(*a,**k): raise RuntimeError('missing live owned window')
    monkeypatch.setattr(probe,'verify_ownership',refuse)
    monkeypatch.setattr(probe,'source_receipt',lambda:pytest.fail('source before ownership'))
    with pytest.raises(RuntimeError,match='owned window'):
        probe.main(['--output',str(tmp_path/'new'),'--model-path',str(tmp_path/'model'),
                    '--run-gpu','--ownership-window',str(tmp_path/'lease')])
    assert not (tmp_path/'new').exists()


def test_existing_evidence_refused_before_ownership(tmp_path,monkeypatch):
    monkeypatch.setattr(probe,'verify_ownership',lambda *a:pytest.fail('owner check unnecessary'))
    with pytest.raises(SystemExit):
        probe.main(['--output',str(tmp_path),'--model-path',str(tmp_path/'model'),
                    '--run-gpu','--ownership-window',str(tmp_path/'lease')])


def test_first_position_reports_cross_selected_margins_without_equivalence_claim():
    result=probe.first_logit_comparison(np.array([2.,1.,0]),np.array([1.,2.,0]))
    assert not result['argmax_equal']
    assert result['actual_selected_vs_reference_selected_margin']==1
    assert result['reference_selected_vs_actual_selected_margin']==1
    assert 'no token-parity' in result['interpretation']
    with pytest.raises(ValueError): probe.first_logit_comparison(np.array([1.,np.nan]),np.array([1.,2.]))


def test_snapshot_check_restores_original_frontier_and_exact_arrays_cpu():
    from test_deepseek_v4_forward_cpu import config,SmallWeights
    from mlx2.runtime.models.deepseek_v4 import DeepSeekV4Model
    c=config(); model=DeepSeekV4Model(c,SmallWeights(c),identity='probe-cpu')
    cache=model.make_cache(); out=model(mx.array([[1,2,3,4,5]],dtype=mx.int32),cache)
    mx.eval(out,[x.state for x in cache])
    result=probe.snapshot_check(model,cache,3,mx)
    assert result['logits_bitwise_equal']
    assert result['state_arrays_bitwise_equal'] and result['state_metadata_equal']
    assert result['restored_original_frontier'] and all(x.offset==5 for x in cache)


@pytest.mark.parametrize('value',['[]','null','{"deepseek_attention":true}'])
def test_execution_policy_must_be_json_string_choice_dictionary(value,tmp_path,monkeypatch):
    monkeypatch.setattr(probe,'verify_ownership',lambda *a:pytest.fail('policy validation after owner'))
    with pytest.raises(SystemExit):
        probe.main(['--output',str(tmp_path/'new'),'--model-path',str(tmp_path/'model'),
                    '--run-gpu','--ownership-window',str(tmp_path/'lease'),'--execution-policy',value])


def test_reference_file_is_one_bounded_finite_vector_before_model_upload(tmp_path):
    path=tmp_path/'logits.npy';np.save(path,np.arange(17,dtype=np.float32))
    values,binding=probe.load_reference(path,17)
    np.testing.assert_array_equal(values,np.arange(17,dtype=np.float32))
    assert len(binding['sha256'])==64
    with pytest.raises(ValueError):probe.load_reference(path,16)
    np.save(path,np.zeros((17,17),dtype=np.float32))
    with pytest.raises(ValueError):probe.load_reference(path,17)


def test_custom_text_request_is_bound_and_rejects_media_tools_or_thinking(tmp_path):
    path=tmp_path/'request.json'
    request={'messages':[{'role':'user','content':'Count from one to sixty four.'}]}
    path.write_text(json.dumps(request))
    loaded,binding=probe.load_request(path)
    assert loaded=={**request,'enable_thinking':False}
    assert len(binding['sha256'])==64
    assert probe.load_request(None)==(probe.REQUEST,None)
    for bad in [[],{'messages':[]},{**request,'tools':[]},{**request,'enable_thinking':True},
                {'messages':[{'role':'user','content':[{'type':'image'}]}]}]:
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):probe.load_request(path)


def test_custom_generation_bound_and_weak_oracle_are_explicit(tmp_path,monkeypatch):
    monkeypatch.setattr(probe,'verify_ownership',lambda *a:pytest.fail('owner reached'))
    common=['--output',str(tmp_path/'new'),'--model-path',str(tmp_path/'model'),
            '--run-gpu','--ownership-window',str(tmp_path/'lease')]
    with pytest.raises(SystemExit):probe.main(common+['--max-tokens','64'])
    with pytest.raises(SystemExit):probe.main(common+['--functional-law','nonempty_finite'])
    assert probe.functional_result(' Hello ',law='exact',expected='Hello',finite=True)['passed']
    assert not probe.functional_result('Other',law='exact',expected='Hello',finite=True)['passed']
    result=probe.functional_result('Some answer',law='nonempty_finite',expected='Hello',finite=True)
    assert result['passed'] and result['expected_text'] is None
    assert not probe.functional_result(' ',law='nonempty_finite',expected='Hello',finite=True)['passed']
    assert not probe.functional_result('Answer',law='nonempty_finite',expected='Hello',finite=False)['passed']


def test_post_evaluation_diagnostics_report_encoded_state_not_allocator_or_qualification():
    from types import SimpleNamespace
    adapter=SimpleNamespace(diagnostics=lambda:{'scheduled_expert_calls':{'packed_moe':2}})
    result=probe.evaluated_route_diagnostics(adapter,[SimpleNamespace(nbytes=732,offset=7),SimpleNamespace(nbytes=2048,offset=7)])
    assert result['cache_storage']['encoded_state_bytes']==2780
    assert result['adapter']['scheduled_expert_calls']['packed_moe']==2
    assert 'not hardware dispatch' in result['scope']
    assert not result['visible_attention']['qualified']
