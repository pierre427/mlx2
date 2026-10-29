"""synthetic@cpu-validator; CPU harness contracts, no model/GPU qualification."""
import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('ds41_modality_options',ROOT/'scripts/research/ds41_modality_options.py')
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)


def status(case):
    from mlx2.adapters.deepseek_v41_source import source_policy
    return {'artifact':'a'*64,'base_artifact':'b'*64,'build_runtime':{'source_sha256':'c'*64},
            'runtime':{'source_sha256':'c'*64},'runtime_source_scope':{'mode':'full_source','reason':'test fixture'},
            'settings':{'route':case['route'],'max_lanes':case['width'],'max_inflight':case['width']},
            'execution':{'execution_policy':source_policy(case['policy']),'physical_width':case['width'],
                         'vision_admission':{'tensor_count':10},'native_dspark':'selected; embedded'}}


def row(label,text,route='ordinary',width=1,oracle=None,**receipt):
    return {'label':label,'oracle':oracle or{'kind':'exact','expected':text},'response':{
        'text':text,'receipt':{'route':route,'route_receipt':{'execution':'observed'},'cache':'apcv2','cached_tokens':0,**receipt},
        'token_trace':{'token_ids':[1,2],'execution_widths':[width,width],'ordinary_execution_widths':[width,width]}}}


def test_all_cases_have_valid_policy_and_distinct_identifiers():
    from mlx2.adapters.deepseek_v41_source import source_policy
    cases=m.cases();assert len(cases)==23 and len({c['id']for c in cases})==23
    for case in cases:
        source_policy(case['policy'])
        assert m.request_plan(case)


def test_draft_ab_plan_primes_then_times_the_same_prompts():
    for name in ('draft_ab_ordinary','draft_ab_chain_d2'):
        plan=m.request_plan(m.case_by_id(name));labels=[r['label']for r in plan]
        assert labels==['count-prime','count','copy-correction-prime','copy-correction','heldout-prime','heldout']
        assert plan[0]['body']==plan[1]['body'] and plan[0]['body'] is not plan[1]['body']
    chain=m.case_by_id('draft_ab_chain_d3')
    assert chain['route']=='external_draft' and chain['policy']['deepseek_verify_rows']=='lockstep_multirow_v1'
    baseline=m.assess(m.case_by_id('draft_ab_ordinary'),[row('heldout','9Qm2Pk7A')],{})
    assert baseline['passed'] and 'heldout_draft_kind' not in baseline['gates']


def test_new_deterministic_raster_decodes_independently():
    from PIL import Image
    raw=m._png();assert raw==m._png()
    image=Image.open(io.BytesIO(raw));image.load()
    assert image.size==(96,96)and image.getextrema()==((255,255),(0,0),(0,0))
    request=m.request_plan(m.case_by_id('vision_red'))[0]
    assert request['image']['sha256']and request['body']['messages'][0]['content'][1]['type']=='image_url'


@pytest.mark.parametrize('name',['physical_b2','physical_b4','vision_red','embedded_dspark_d3','prompt_lookup_replay_d1','prompt_lookup_local_frontier_d5'])
def test_server_identity_shape_and_normalized_policy_admission(name):
    case=m.case_by_id(name);value=status(case)
    m.validate_server(case,value,fingerprint='a'*64,source_snapshot='d'*64,runtime_build={'source_sha256':'c'*64})
    value['settings']['max_inflight']=0
    with pytest.raises(ValueError,match='lane/inflight'):
        m.validate_server(case,value,fingerprint='a'*64,source_snapshot='d'*64,runtime_build={'source_sha256':'c'*64})


def test_batch_cannot_pass_from_selected_width_or_scheduler_presence_only():
    case=m.case_by_id('physical_b2');rows=[row('a','CODEA',width=2),row('b','CODEB',width=2)]
    st={'execution':{'scheduled_batch_calls':{'joint_ffn_calls':40,'maximum_joint_ffn_requests':2,'maximum_live_requests':2}}}
    assert m.assess(case,rows,st)['passed']
    rows[1]['response']['token_trace']['ordinary_execution_widths']=[1,1]
    assert not m.assess(case,rows,st)['passed']
    rows[1]['response']['token_trace']['ordinary_execution_widths']=[2,2]
    st['execution']['scheduled_batch_calls']['maximum_joint_ffn_requests']=1
    assert not m.assess(case,rows,st)['passed']


def test_vision_needs_consumed_image_input_and_color_oracle():
    case=m.case_by_id('vision_red')
    rows=[row('image','red',oracle={'kind':'word','expected':'red'},media_decode={'media_prefill':'isolated_prefill_forward','media_prefill_inputs':['image_inputs']})]
    assert m.assess(case,rows,{})['passed']
    rows[0]['response']['receipt']['media_decode']['media_prefill_inputs']=[]
    assert not m.assess(case,rows,{})['passed']


def test_draft_selection_without_proposals_is_not_observed_execution():
    case=m.case_by_id('embedded_dspark_d2');rows=[row('draft','OK',route='external_draft',speculation={'kind':'external_deepseek_v41_embedded_dspark_chain','accepted':0,'proposed':0,'external_rounds':0})]
    assert not m.assess(case,rows,{})['passed']
    s=rows[0]['response']['receipt']['speculation'];s.update(accepted=3,proposed=5,external_rounds=2,round_proposed=2,round_accepted=1)
    verdict=m.assess(case,rows,{})
    assert verdict['passed']and verdict['draft_observations']['acceptance_observed']
    assert verdict['draft_observations']['unaccepted']==2
    assert verdict['draft_observations']['rejection_coverage'].startswith('not_proven')
    assert 'rejection_vs_output_truncation_unresolved'in verdict['draft_observations']['final_round_evidence'][0]['classification']


def test_request_probe_fails_before_network_on_lost_caller_ownership(tmp_path):
    window=tmp_path/'window';window.mkdir();out=tmp_path/'result'
    spec={'out':str(window),'advanced_case':'physical_b2','checkpoint':'synthetic@'+'a'*64,
          'source_revision':'b'*40,'source_snapshot_sha256':'c'*64}
    def lost():raise RuntimeError('caller lease lost')
    with pytest.raises(RuntimeError,match='caller lease lost'):
        m.run_requests(spec,out,guard=lost)
    saved=json.loads((out/'results.json').read_text())
    assert not saved['passed']and saved['error']['message']=='caller lease lost'
    with pytest.raises(ValueError,match='nonempty'):
        m.run_requests(spec,out,guard=lost)


def test_desktop_and_runtime_hashes_are_verified_in_their_own_namespaces():
    case=m.case_by_id('physical_b2');value=status(case)
    result=m.validate_server(case,value,fingerprint='a'*64,source_snapshot='d'*64,
                             runtime_build={'source_sha256':'c'*64})
    assert result['desktop_source_manifest_sha256']=='d'*64
    assert result['runtime_build']['source_sha256']=='c'*64
    value['build_runtime']['source_sha256']='e'*64
    with pytest.raises(ValueError,match='runtime build identity'):
        m.validate_server(case,value,fingerprint='a'*64,source_snapshot='d'*64,
                          runtime_build={'source_sha256':'c'*64})


def test_served_runtime_hash_is_not_trusted_merely_because_build_matches():
    case=m.case_by_id('physical_b2');value=status(case)
    value['runtime']['source_sha256']='e'*64
    with pytest.raises(ValueError,match='served runtime identity'):
        m.validate_server(case,value,fingerprint='a'*64,source_snapshot='d'*64,
                          runtime_build={'source_sha256':'c'*64})
    value['runtime']=dict(value['build_runtime']);value['runtime_source_scope']={'mode':'unknown'}
    with pytest.raises(ValueError,match='unknown runtime source scope'):
        m.validate_server(case,value,fingerprint='a'*64,source_snapshot='d'*64,
                          runtime_build={'source_sha256':'c'*64})


def test_native_ordinary_vision_has_its_own_use_proof_not_plain_decode_receipt():
    case=m.case_by_id('vision_red');source='a'*64
    before={'calls_after_successful_return':{'vision.encoder_calls_completed':4,'vision.images_encoded':4,'vision.native_image_prefill_returns':4},'sequence':9}
    after={'calls_after_successful_return':{'vision.encoder_calls_completed':5,'vision.images_encoded':5,'vision.native_image_prefill_returns':5},'sequence':10,
           'native_vision':{'source_fingerprint':source,'target_identity':'target','prefill_target_identity':'target','prefill_returned':True}}
    proof=m.native_media_observation(before,after,source_fingerprint=source)
    rows=[row('image','red',oracle={'kind':'word','expected':'red'})]
    assert not m.assess(case,rows,{})['passed']
    rows[0]['native_media_observation']=proof
    assert m.assess(case,rows,{})['passed']
    # Prior use alone is not this request's use proof.
    rows[0]['native_media_observation']=m.native_media_observation(after,after,source_fingerprint=source)
    assert not m.assess(case,rows,{})['passed']
    rows[0]['native_media_observation']=m.native_media_observation(before,after,source_fingerprint='b'*64)
    assert not m.assess(case,rows,{})['passed']
