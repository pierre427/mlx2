"""Host-only evidence-contract checks for the bounded model probe."""
import importlib.util
from pathlib import Path

PATH=Path(__file__).resolve().parents[1]/'scripts/research/probe_microscaling_model.py'
spec=importlib.util.spec_from_file_location('sc_model_probe',PATH)
probe=importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)


def response(tokens,*,cached=0,text='ZX-3187',passed=True):
    return {'token_ids':tokens,'text':text,'oracle_passed':passed,'ordinary_execution_widths':[1]*len(tokens),
            'receipt':{'cache':'apcv2','cached_tokens':cached,'route':'ordinary','route_receipt':'candidate_validation'}}


def test_identifier_normalizes_dash_punctuation_only():
    assert probe.oracle('ZX–3187','identifier','ZX-3187')
    assert not probe.oracle('zx-3187','identifier','ZX-3187')
    assert not probe.oracle('ZX−3187','identifier','ZX-3187')
    assert not probe.oracle('Code: ZX-3187','identifier','ZX-3187')


def test_model_code_is_parsed_without_execution():
    assert probe.oracle('def add(a, b):\n    return a + b','python_add',None)
    assert not probe.oracle('def add(a, b):\n    return a - b','python_add',None)
    assert probe.oracle('{"ok":true,"count":3}','json',{'ok':True,'count':3})


def test_width_one_token_failure_is_numerical_not_missing_receipt():
    pair=probe.pair_assessment(response([1,2]),response([1,3],cached=512))
    assert pair['methodological_passed']
    assert not pair['numerical_passed']
    assert pair['first_divergence']['output_position']==1
    assert pair['numerical_classification']=='width_one_exact_parity_failed'


def test_cache_contract_failure_does_not_fabricate_token_mismatch():
    pair=probe.pair_assessment(response([1,2]),response([1,2],cached=0))
    assert not pair['methodological_passed'] and pair['numerical_passed']
    assert pair['first_divergence'] is None


def test_approximate_window_must_reproduce_selected_prefix():
    left,right=response([1,2]),response([1,3])
    windows=[response([9,2]),response([1,3])]
    for row in windows:
        row['logprobs']={'content':[{'id':2,'logprob':-.2,'top_logprobs':[{'id':3,'logprob':-.3}]}]}
    got=probe.compare_approximate(left,right,windows)
    assert got['classification']=='approximate_token_change_window_not_reproduced'


def test_cross_selected_near_tie_uses_one_shared_position():
    left,right=response([1,2]),response([1,3])
    windows=[response([1,2]),response([1,3])]
    windows[0]['logprobs']={'content':[{'id':2,'logprob':-.2,'top_logprobs':[{'id':3,'logprob':-.3}]}]}
    windows[1]['logprobs']={'content':[{'id':3,'logprob':-.2,'top_logprobs':[{'id':2,'logprob':-.4}]}]}
    got=probe.compare_approximate(left,right,windows)
    assert got['classification']=='near_tie_equivalent_approximate_arms'
    assert max(got['cross_selected_token_gaps_nats'])<=.5


def test_prompt_length_and_receipt_use_token_ids_from_batchencoding():
    from collections import UserDict
    import json
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            assert kwargs['return_dict'] is False
            count=20+len(messages[0]['content'].split())
            return UserDict({'input_ids':list(range(count))})
    prompt=probe.make_prompt(Tokenizer(),512)
    assert prompt['rendered_prompt_tokens']>=512
    assert prompt['rendered_prompt_tokens']<540
    assert all(type(t) is int for t in prompt['rendered_prompt_ids'])
    json.dumps(prompt)


def test_chat_uses_the_loaded_model_identity(monkeypatch,tmp_path):
    import mlx2.generation_trace as trace
    monkeypatch.setattr(trace,'reconcile_completion_tokens',lambda *a,**kw:{})
    seen=[]
    raw={'mlx2':{'token_trace':{'token_ids':[1],'ordinary_execution_widths':[1]}},
         'usage':{},'choices':[{'message':{'content':'ZX-3187'},'finish_reason':'stop'}]}
    monkeypatch.setattr(probe,'request_json',lambda base,path,body:seen.append(body) or raw)
    p={'model_id':'actual-sc-checkpoint','messages':[],'kind':'identifier','expected':'ZX-3187'}
    result=probe.chat('http://127.0.0.1:1',p,tmp_path/'response.json')
    assert seen[0]['model']=='actual-sc-checkpoint' and result['oracle_passed']
