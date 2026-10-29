"""Controlled comparison logic and actual tiny B1 generator paths, CPU only."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts/research'))
    spec=importlib.util.spec_from_file_location('ds4_compare_test',ROOT/'scripts/research/probe_deepseek_v4_decode_compare.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def sample():
    return {'thermal_ok':True,'foreign_model_processes':[],'numerics_environment':{'X':'1'},
            'Swapins':0,'Swapouts':0,'swap_used_bytes':4096}


def test_controlled_warmup_order_and_eligibility(probe):
    calls=[];writes=[]
    def run(arm):
        calls.append(arm)
        return {'arm':arm,'tokens':list(range(64)),'decode_seconds':2 if arm=='ordinary' else 1}
    result=probe.run_comparison(run,sample,lambda:None,lambda *args:writes.append(args),budget=64,sleep=lambda _:None)
    assert calls==list(probe.ARMS)+[arm for order in probe.ORDERS for arm in order]
    assert all({order[i] for order in probe.ORDERS}==set(probe.ARMS) for i in range(3))
    assert result['comparison_eligible'] and result['performance_claim'] is False
    assert result['median_speedup']=={'sequential':2,'causal_slab':2}
    assert len(writes)==13


def test_bad_admission_stops_before_measured_arm(probe):
    calls=[]
    def bad():return {**sample(),'thermal_ok':False}
    with pytest.raises(RuntimeError,match='admission'):
        probe.run_comparison(lambda arm:calls.append(arm),bad,lambda:None,lambda *args:None,budget=64,sleep=lambda _:None,settle_seconds=0)
    assert calls==list(probe.ARMS)


def test_divergence_short_output_and_swap_disable_ratios(probe):
    records=[{'repetition':i,'arm':arm,'tokens':list(range(64)),'decode_seconds':1.,'gates':{'thermal':True}}
             for i in range(3) for arm in probe.ARMS]
    records[1]['tokens'][5]=99
    records[2]['tokens']=records[2]['tokens'][:-1]
    records[-1]['gates']['swap_stable']=False
    result=probe.comparison_summary(records,64)
    assert not result['comparison_eligible'] and result['paired_ratios'] is None
    assert records[1]['first_divergence']=={'index':5,'ordinary_token':5,'arm_token':99,'shared_prefix_tokens':5}
    assert len(result['gate_failures'])==4
    assert not probe.gate_samples([sample()],[{**sample(),'Swapins':1}])['swap_stable']


def test_bounds_refuse_warm_cache_policy_and_small_output(probe,tmp_path):
    args=SimpleNamespace(run_gpu=True,ownership_window=tmp_path/'owner',output=tmp_path/'new',max_tokens=64,
                         prefill_step=128,memory_limit_gib=108,timeout_seconds=1800,execution_policy={},seed=42)
    probe.validate_args(args)
    for key,value in [('max_tokens',8),('execution_policy',{'draft_model':'foreign'}),('run_gpu',False)]:
        changed=copy.copy(args);setattr(changed,key,value)
        with pytest.raises(ValueError):probe.validate_args(changed)


def test_actual_three_arm_cpu_paths_have_fresh_exact_tokens(probe,monkeypatch):
    import mlx.core as mx
    from test_deepseek_v4_external_cpu import Backbone

    from mlx2.adapters import deepseek_v4_external as external
    from mlx2.runtime.models.deepseek_v4_cache import DeepSeekV4LayerCache
    old=mx.default_device();mx.set_default_device(mx.cpu)
    try:
        class ToyBackbone(Backbone):
            config=SimpleNamespace(target_layers=(0,),hidden_size=2,vocab_size=16,block_size=5,
                                   stages=3,window=128,head_dim=2,expert_hidden=2)
            def record_features(self,captures,position,*,history):
                x=captures[0];history.append(x,(x,x,x),position)
            def propose(self,seed,**kwargs):
                n=kwargs['max_proposals'];return tuple((seed+i+1)%16 for i in range(n)),(0.,)*n
        class Model:
            identity='synthetic-comparison';args=Backbone.target_config;weights=Backbone.target_weights
            def make_cache(self):return [DeepSeekV4LayerCache(xp=mx,head_dim=2,index_head_dim=2,identity=self.identity)]
            def _forward(self,inputs,cache):
                cache[0].update_local(mx.broadcast_to(inputs[:,None,:,None],(1,1,inputs.shape[1],2)).astype(mx.float32))
                logits=mx.full((1,inputs.shape[1],16),-10.)
                logits=mx.put_along_axis(logits,((inputs+1)%16)[:,:,None],mx.full((1,inputs.shape[1],1),10.),axis=-1)
                return logits,mx.broadcast_to(inputs[:,:,None],(1,inputs.shape[1],2)).astype(mx.float32)
            def forward_with_taps(self,inputs,cache,layers):return self._forward(inputs,cache)
            def verify_rows(self,inputs,cache,layers):
                rows=[self._forward(inputs[:,i:i+1],cache) for i in range(inputs.shape[1])]
                return tuple(mx.concatenate([row[j] for row in rows],axis=1) for j in (0,1))
            def prefill_body(self,inputs,cache,layers):return self._forward(inputs,cache)[1]
            def __call__(self,inputs,cache,**kwargs):return self._forward(inputs,cache)[0]
        model=Model();draft=external.DeepSeekV4DSparkDraftModel(ToyBackbone()).bind(model)
        adapter=SimpleNamespace(model=model,draft_model=draft,tokenizer=SimpleNamespace(eos_token_ids=[]),target_identity={'fingerprint':'synthetic'})
        real=external.DeepSeekV4ExternalBatchGenerator
        monkeypatch.setattr(external,'DeepSeekV4ExternalBatchGenerator',lambda *a,**kw:real(*a,memory_headroom=lambda:1<<30,**kw))
        runner=probe.LoadedArms(adapter,[1,2,3,4],8,2,42,lambda:None)
        rows=[runner(arm) for arm in probe.ARMS]
        assert all(row['tokens']==list(range(5,13)) for row in rows)
        assert all(row['prefill_evaluated_tokens']==3 and row['prefill_evaluated_seconds']>0 for row in rows)
        assert all(row['paired_cache_resumes']==0 for row in rows)
        assert rows[1]['proposed_tokens']>0 and rows[1]['accepted_proposals']>0
        assert rows[0]['scheduled_target_work']['target_forward_calls']==8
        assert runner('ordinary')['tokens']==rows[0]['tokens']
    finally:mx.set_default_device(old)


def test_missing_sensor_telemetry_published_before_refusal_no_wait(probe):
    events=[];calls=[]
    def missing():return {**sample(),'thermal_ok':False,'thermal':{'thermal_state':0},
        'thermal_failures':probe.thermal_failures({'thermal_state':0})}
    with pytest.raises(RuntimeError,match='admission'):
        probe.run_comparison(lambda arm:calls.append(arm),missing,lambda:None,lambda *args:None,
            budget=64,sleep=lambda _:None,publish_admission=lambda x:events.append(copy.deepcopy(x)))
    assert len(calls)==3
    assert events[-1]['status']=='refused'
    assert len(events[-1]['attempts'][0]['samples'])==3
    assert 'battery_temperature_c' in events[-1]['attempts'][0]['samples'][0]['thermal_failures']['missing_sensors_or_fields']
    assert all(e['status']!='settling' for e in events)


def test_one_bounded_settle_then_eligible_without_changing_limits(probe):
    clock=[0.];samples=[0];events=[]
    def sleep(n):clock[0]+=n
    def transient():
        samples[0]+=1
        return {**sample(),'thermal_ok':samples[0]>3}
    def arm(name):return {'arm':name,'tokens':list(range(64)),'decode_seconds':1.}
    result=probe.run_comparison(arm,transient,lambda:None,lambda *args:None,budget=64,
        sleep=sleep,clock=lambda:clock[0],publish_admission=lambda e:events.append(copy.deepcopy(e)))
    assert result['comparison_eligible']
    assert any(e['status']=='admitted_after_settling' for e in events)
    assert probe.THERMAL_POLICY['require_temperatures'] is True


def test_settling_yields_to_foreign_waiter_before_sleep(probe):
    events=[];wait=[]
    with pytest.raises(RuntimeError,match='foreign GPU waiter'):
        probe.run_comparison(lambda arm:None,lambda:{**sample(),'thermal_ok':False},lambda:None,
            lambda *args:None,budget=64,sleep=lambda n:wait.append(n),
            waiters=lambda:[{'pid':42,'name':'foreign'}],publish_admission=lambda e:events.append(copy.deepcopy(e)))
    assert wait==[1,1]
    assert events[-1]['status']=='yield_foreign_waiter'


def test_thermal_diagnosis_distinguishes_missing_from_heat(probe):
    state={'thermal_state':0,'pmset_no_thermal_warning':True,'pmset_no_performance_warning':True,'pmset_no_cpu_power_warning':True}
    missing=probe.thermal_failures(state)
    assert missing['missing_sensors_or_fields']==['battery_temperature_c','virtual_temperature_c']
    assert missing['above_limits']==[] and not missing['non_nominal_state']
    hot=probe.thermal_failures({**state,'battery_temperature_c':41,'virtual_temperature_c':46})
    assert not hot['missing_sensors_or_fields'] and len(hot['above_limits'])==2


def test_persistent_heat_stops_at_single_settling_budget(probe):
    now=[0.];events=[]
    def sleep(n):now[0]+=n
    with pytest.raises(RuntimeError,match='admission'):
        probe.run_comparison(lambda arm:None,lambda:{**sample(),'thermal_ok':False},lambda:None,
            lambda *args:None,budget=64,sleep=sleep,clock=lambda:now[0],
            publish_admission=lambda e:events.append(copy.deepcopy(e)))
    assert events[-1]['status']=='refused'
    assert events[-1]['settling_elapsed_seconds']<=60
    assert now[0]<=62  # initial three-sample bracket plus one settling budget


def test_explicit_hid_profile_limits_stability_and_stale_data(probe):
    state={'thermal_state':0,'pmset_no_thermal_warning':True,'pmset_no_performance_warning':True,
           'pmset_no_cpu_power_warning':True,'die_max_c':59.9,'hid_battery_temperature_c':39.9,
           'hid_raw_sensors_c':{'PMU tdie1':59.9,'gas gauge battery':39.9}}
    assert not any(probe.thermal_failures(state,probe.HID_PROFILE).values())
    assert probe.thermal_failures(state,probe.LEGACY_PROFILE)['missing_sensors_or_fields']
    rows=[{**sample(),'thermal':dict(state),'thermal_profile':probe.HID_PROFILE,'time_unix':i} for i in (98.,99.,100.)]
    assert all(probe.gate_samples(rows,rows,now=100.).values())
    assert not probe.gate_samples(rows,rows,now=111.)['hid_sample_recency']
    changed=copy.deepcopy(rows);changed[0]['thermal']['die_max_c']=59.
    assert not probe.gate_samples(changed,rows,now=100.)['hid_three_sample_stability']
    for key,value in [('die_max_c',60.1),('hid_battery_temperature_c',40.1),('die_max_c',float('nan'))]:
        assert any(probe.thermal_failures({**state,key:value},probe.HID_PROFILE).values())
    del state['hid_battery_temperature_c']
    assert 'hid_battery_temperature_c' in probe.thermal_failures(state,probe.HID_PROFILE)['missing_sensors_or_fields']
    assert probe.THERMAL_POLICY['max_virtual_temperature_c']==45.
