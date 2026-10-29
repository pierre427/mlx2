"""Ladder arithmetic and cold/replay contracts without HTTP, MLX or models."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/research'))
import ds41_options_ladder as ladder


def request(kind='identifier'):
    return dict(context_budget=128,prompt_tokens=64,output_reserve=64,kind=kind,
        output_tokens=32 if kind=='counting'else 64,expected_code='D41-ABCDEF01',
        request={'messages':[{'role':'user','content':'test'}],'enable_thinking':False})


def response(kind='identifier',cached=0):
    return dict(text='D41-ABCDEF01'if kind=='identifier'else'\n'.join(map(str,range(1,17))),
        token_ids=list(range(32)),token_trace={'execution_widths':[1]*32},HTTP_seconds=5.2,
        receipt={'prompt_tokens':64,'completion_tokens':32,'cached_tokens':cached,'cache':'apcv2',
            'route':'ordinary','route_receipt':{'route':'ordinary'},'ttft_seconds':1.,'elapsed_seconds':5.})


def test_decode_uses_n_minus_one_and_ttft_is_separate():
    row=ladder.timing(response(),{'mlx2_request_prefill_time_seconds_sum':.8,'mlx2_request_prefill_time_seconds_count':1})
    assert row['decode_tokens_per_second']==31/4
    assert row['prompt_tokens_per_TTFT_second']==64
    assert row['uncached_prompt_tokens_per_prefill_metric_second']==80
    assert row['HTTP_wall_seconds']==5.2
    warm=ladder.timing(response(cached=63),{})
    assert warm['uncached_prompt_tokens']==1
    assert warm['prefill_metric_seconds']is None


def test_invalid_or_zero_timing_is_not_divided():
    value=response();value['receipt'].update(ttft_seconds=0,elapsed_seconds=0,completion_tokens=1)
    row=ladder.timing(value,{})
    assert row['decode_tokens_per_second']is None
    assert row['prompt_tokens_per_TTFT_second']is None


@pytest.mark.parametrize('kind',['identifier','counting'])
def test_ordinary_contract_and_exact_cache_expectation(kind):
    assert ladder.assess(response(kind),request(kind),expected_cached_tokens=0)['passed']
    assert not ladder.assess(response(kind,cached=1),request(kind),expected_cached_tokens=0)['contract_passed']
    external=response(kind);external['receipt']['route']='external_draft'
    assert not ladder.assess(external,request(kind),expected_cached_tokens=0)['contract_passed']


def test_identifier_dash_normalization_preserves_every_other_character():
    value=response();value['text']='D41–ABCDEF01'
    assert ladder.assess(value,request(),expected_cached_tokens=0)['passed']
    value['text']='d41-ABCDEF01'
    assert not ladder.assess(value,request(),expected_cached_tokens=0)['functional_passed']


def test_expert_counters_exclude_ple_and_apc_diagnostic_scopes():
    before={'execution':{'expert_stream':{'counters':{'v41_expert_cache_misses':3,'v41_expert_bytes_read':100},'preheat':{'counter_delta':{'v41_expert_cache_misses':2}}},'engram':{'cache_misses':7}},'apcv2':{'bytes_read':8}}
    after={'execution':{'expert_stream':{'counters':{'v41_expert_cache_misses':5,'v41_expert_bytes_read':120},'preheat':{'counter_delta':{'v41_expert_cache_misses':2}}},'engram':{'cache_misses':10}},'apcv2':{'bytes_read':18}}
    assert ladder.expert_delta(before,after)=={'v41_expert_cache_misses':2,'v41_expert_bytes_read':20}


def setup_run(tmp_path,monkeypatch,*,apc=False,diverge=False,fail=False):
    info=request();info['path']='request.json'
    (tmp_path/'requests.json').write_text(json.dumps([info]));(tmp_path/'request.json').write_text(json.dumps(info))
    calls=[]
    def http(base,path,*args,**kwargs):
        calls.append(path);return {'settings':{'route':'ordinary'}}
    monkeypatch.setattr(ladder,'http_json',http)
    monkeypatch.setattr(ladder,'metrics',lambda base:('metric 1\n',{}))
    streams=[]
    def stream(base,body,path,**kwargs):
        path.write_text('raw SSE fixture')
        if fail:raise RuntimeError('stream failed')
        value=response(cached=63 if apc and streams else 0)
        if diverge and streams:value['token_ids'][3]=99
        streams.append(body);return value
    monkeypatch.setattr(ladder,'stream',stream)
    spec={'out':str(tmp_path),'checkpoint':'test@'+'a'*64,'source_revision':'b'*40,
          'source_snapshot_sha256':'c'*64,'ladder_apc_enabled':apc,'execution_policy':{}}
    return spec,streams,calls


def test_single_ordinary_warm_pair_does_not_claim_apc_hit(tmp_path,monkeypatch):
    spec,streams,calls=setup_run(tmp_path,monkeypatch)
    phases=[]
    report=ladder.run(spec,'model',guard=lambda:None,environment=lambda:{'swap':17,'temperature':83},
        phase=lambda *a,**k:phases.append((a,k)),admin_token='secret')
    assert report['passed']and report['repetitions']==1 and report['thermal_controlled']is False
    assert len(streams)==2 and calls.count('/v1/admin/apc/clear')==1
    cell=json.loads((tmp_path/'context-128-identifier/result.json').read_text())
    assert cell['arms']['warm']['cache_arm']=='ordinary_expert_cache_warm'
    with pytest.raises(ValueError,match='overwrite'):
        ladder.run(spec,'model',guard=lambda:None,environment=lambda:{},phase=lambda *a,**k:None,admin_token='secret')


def test_canonical_pair_requires_exact_replay_tokens(tmp_path,monkeypatch):
    spec,streams,calls=setup_run(tmp_path,monkeypatch,apc=True,diverge=True)
    report=ladder.run(spec,'model',guard=lambda:None,environment=lambda:{},phase=lambda *a,**k:None,admin_token='secret')
    assert not report['passed']and report['cells'][0]['first_divergence']['zero_based_position']==3
    assert report['cells'][0]['first_divergence']['shared_prefix']==[0,1,2]
    assert 'exact APCv2 replay divergence'in report['stopped_reason']


def test_failure_keeps_atomic_report_and_raw_sse(tmp_path,monkeypatch):
    spec,streams,calls=setup_run(tmp_path,monkeypatch,fail=True)
    with pytest.raises(RuntimeError,match='stream failed'):
        ladder.run(spec,'model',guard=lambda:None,environment=lambda:{},phase=lambda *a,**k:None,admin_token='secret')
    report=json.loads((tmp_path/'ladder.json').read_text())
    assert report['status']=='failed'and report['error']['type']=='RuntimeError'
    assert (tmp_path/'context-128-identifier/cold.sse').read_text()=='raw SSE fixture'
