"""Bounded NAX comparison planning/diagnostics; no model upload or GPU dispatch."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import mlx.core as mx
mx.set_default_device(mx.cpu)
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/research'))
spec=importlib.util.spec_from_file_location('prefill_compare',ROOT/'scripts/research/probe_deepseek_v4_prefill_compare.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def test_independent_candidate_defaults_and_sidecar_rejected():
    a,b=probe.policies({})
    assert a['deepseek_dense_prefill']==a['deepseek_expert_prefill']=='reference'
    assert b['deepseek_dense_prefill']=='nax_fp16'
    assert b['deepseek_expert_prefill']=='segmented_nax_fp16'
    for bad in [[],{'draft_model':'x'},{'deepseek_dense_prefill':'nax_fp16'}]:
        with pytest.raises(ValueError):probe.policies(bad)


def test_usage_is_required_after_evaluation_without_hardware_claim():
    assert probe.usage_check({},False)['passed']
    assert not probe.usage_check({},True)['passed']
    row={'route_after_evaluated_generation':{'adapter':{
        'scheduled_dense_calls':{'nax_prefill':3},'scheduled_grouped_calls':{'nax_grouped_prefill':2},
        'scheduled_expert_calls':{'segmented_moe_prefill':1}}}}
    assert probe.usage_check(row,True)['passed']
    assert not probe.usage_check(row,False)['passed']
    assert 'no hardware' in probe.usage_check(row,True)['scope']


def test_same_request_and_immutable_policy_command(tmp_path):
    args=SimpleNamespace(ownership_window=tmp_path/'owned',expected_text='Hello',functional_law='exact',
                         max_tokens=16,memory_limit_gib=108,owner_path=[])
    cmd=probe.arm_command(tmp_path/'model',tmp_path/'arm',tmp_path/'request',{},512,args,
                          position=7,reference=tmp_path/'reference.npy')
    assert cmd[cmd.index('--capture-output-position')+1]=='7'
    assert cmd[cmd.index('--prefill-step')+1]=='512'
    assert cmd[cmd.index('--request-json')+1]==str(tmp_path/'request')
    assert '--run-gpu' in cmd and '--reference-logits' in cmd


def test_pair_retains_first_divergence_and_distinct_arithmetic(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    np.save(a/'first.npy',np.array([1.,2.,0.],np.float32))
    np.save(b/'first.npy',np.array([1.01,2.,0.],np.float32))
    def row(tokens):return {'prompt':{'rendered_utf8_sha256':'same','tokens':640},
        'generation':{'token_ids':tokens,'text':'Hello'},'captured_logits':{'path':'first.npy'}}
    r=probe.pair_summary(row([1,0,2]),row([1,2,2]),a,b,3)
    assert r['same_prompt'] and r['first_selected_divergence']==1
    assert not r['first_position_logits_bitwise_exact'] and not r['selected_tokens_exact']
    assert r['equivalence_verdict'].startswith('not_assessed')
    assert probe.first_difference([1],[1,2])==1
    assert probe.first_difference([1],[1]) is None


def test_gpu_guard_precedes_source_and_artifact(tmp_path,monkeypatch):
    monkeypatch.setattr(probe,'verify_ownership',lambda *a:(_ for _ in ()).throw(RuntimeError('not owned')))
    monkeypatch.setattr(probe,'source_receipt',lambda:pytest.fail('source before ownership'))
    with pytest.raises(RuntimeError,match='not owned'):
        probe.main(['--model-path',str(tmp_path/'model'),'--output',str(tmp_path/'new'),
                    '--run-gpu','--ownership-window',str(tmp_path/'owned')])
    assert not (tmp_path/'new').exists()


def test_single_selected_diagnostic_row_only(tmp_path):
    receipt={};v=np.array([2.,1.,0.],np.float32)
    from probe_deepseek_v4_ordinary import capture_selected_position
    capture_selected_position(receipt,tmp_path,v,7,v,{'sha256':'reference'})
    assert [x.name for x in tmp_path.iterdir()]==['output-position-7-logits.npy']
    assert receipt['captured_logits']['output_position']==7
    assert receipt['reference_logits']['comparison']['output_position']==7


@pytest.mark.parametrize('selection,dense,experts',[('both',True,True),('dense',True,False),('experts',False,True)])
def test_selectable_policy_and_exact_scheduled_expectations(selection,dense,experts):
    ref,cand=probe.policies({},selection)
    assert (cand['deepseek_dense_prefill']=='nax_fp16')==dense
    assert (cand['deepseek_expert_prefill']=='segmented_nax_fp16')==experts
    assert ref['deepseek_dense_prefill']==ref['deepseek_expert_prefill']=='reference'
    row={'route_after_evaluated_generation':{'adapter':{
        'scheduled_dense_calls':{'nax_prefill':int(dense)},
        'scheduled_grouped_calls':{'nax_grouped_prefill':int(dense)},
        'scheduled_expert_calls':{'segmented_moe_prefill':int(experts)}}}}
    r=probe.usage_check(row,True,selection)
    assert r['passed']
    assert r['expected_scheduled_nonzero']=={'nax_prefill':dense,'nax_grouped_prefill':dense,'segmented_moe_prefill':experts}
    # A supposedly reference component firing is a failure, not harmless extra use.
    group='scheduled_expert_calls' if not experts else 'scheduled_dense_calls'
    key='segmented_moe_prefill' if not experts else 'nax_prefill'
    row['route_after_evaluated_generation']['adapter'][group][key]=2 if (not experts or not dense) else 0
    assert not probe.usage_check(row,True,selection)['passed']


def test_candidate_selection_rejects_unknown_and_reference_stays_zero():
    with pytest.raises(ValueError):probe.policies({},'unknown')
    with pytest.raises(ValueError):probe.usage_check({},True,'unknown')
    assert all(probe.usage_check({},False,s)['passed'] for s in ('both','dense','experts'))


def test_completed_functional_fail_is_comparable_but_crash_is_not(tmp_path):
    import json
    row={'status':'completed','passed':False,'generation':{'token_ids':[19923,16,1]},
         'captured_logits':{'path':'first.npy'}}
    (tmp_path/'results.json').write_text(json.dumps(row))
    result=probe.completed_arm(tmp_path,1)
    assert not result['passed'] and result['process_exit_code']==1
    assert 'process_exit_code' not in json.loads((tmp_path/'results.json').read_text())
    row['status']='failed';(tmp_path/'results.json').write_text(json.dumps(row))
    with pytest.raises(RuntimeError,match='incomplete/crashed'):probe.completed_arm(tmp_path,1)


def test_functional_mismatch_records_divergence_without_automatic_gpu_replay(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    for root in (a,b):np.save(root/'first.npy',np.array([0.,1.,2.],np.float32))
    def row(tokens,text,passed):return {'status':'completed','passed':passed,
        'prompt':{'rendered_utf8_sha256':'fixed','tokens':645},
        'generation':{'token_ids':tokens,'text':text},'captured_logits':{'path':'first.npy'}}
    first=row([19923,1],'Hello',True);second=row([19923,16,1],'Hello.',False)
    pair=probe.pair_summary(first,second,a,b,3)
    assert pair['first_selected_divergence']==1 and not pair['visible_text_exact']
    assert pair['first_position_logits_bitwise_exact']
    assert probe.replay_disposition(1,arms_passed=False,skip=False,replay_failed=False)=='pending_explicit_opt_in_functional_failure'
    assert probe.replay_disposition(1,arms_passed=False,skip=False,replay_failed=True)=='pending'
    assert probe.replay_disposition(1,arms_passed=True,skip=False,replay_failed=False)=='pending'
    assert probe.replay_disposition(1,arms_passed=False,skip=True,replay_failed=True)=='skipped_by_operator'


def test_child_exit_one_with_completed_false_returns_evidence(tmp_path):
    import json,time
    arm=tmp_path/'arm';arm.mkdir()
    script=tmp_path/'child.py'
    script.write_text('import pathlib,json,sys\np=pathlib.Path(sys.argv[2]);(p/"results.json").write_text(json.dumps('+repr({'status':'completed','passed':False,'generation':{'token_ids':[7,1]},'captured_logits':{'path':'first.npy'}})+'))\nsys.exit(1)\n')
    result=probe.execute([sys.executable,str(script),'--output',str(arm)],tmp_path/'child.log',lambda:None,time.monotonic()+5)
    assert result['process_exit_code']==1 and result['passed'] is False
