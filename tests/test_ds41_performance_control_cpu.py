"""Synthetic receipts and source files only: no model or GPU execution."""
from copy import deepcopy
from dataclasses import asdict,replace
import json
import math

import pytest

from mlx2.experimental.ds41_performance import (
    Binding,Shape,REQUIRED_GATES,campaign,decode_intervals,paired_analysis as bound_analysis,policy_identity,
    source_snapshot,validate_receipt,write_new_json,runtime_source_paths)


def binding():
    return Binding('synthetic-ds41-control@'+'a'*64,'b'*40,'c'*64,'d'*64,
                   'e'*64,'synthetic-cpu-hardware','synthetic-runtime','synthetic-profile')


def arms():
    return {arm:policy_identity({'candidate':'v1'} if arm=='candidate' else {},
                numerical_profile='synthetic-'+arm,implementation_sha256='f'*64)
            for arm in ('reference','candidate')}


def paired_analysis(pairs,**kwargs):
    return bound_analysis(pairs,arm_bindings=arms(),**kwargs)


def receipt(pair,arm,*,shape=None,step=.08):
    shape=shape or Shape('decode')
    return {'binding':asdict(binding()),'shape':asdict(shape),'pair':pair,'arm':arm,
            'arm_binding':arms()[arm],
            'unit':f'{pair}-{arm}','completed':True,
            'gates':dict.fromkeys(REQUIRED_GATES,True),
            'requests':[{'actual_prompt_tokens':shape.actual_prompt_tokens,
                'cached_tokens':shape.cached_tokens,'output_tokens':shape.output_tokens,
                'generated_timestamps':[17+i*step for i in range(shape.output_tokens)],
                'prefill_seconds':2.0} for _ in range(3)]}


def test_interval_rate_counts_n_minus_one_and_excludes_ttft():
    r=decode_intervals([100+i/12.5 for i in range(32)])
    assert r['interval_count']==31
    assert r['elapsed_seconds']==pytest.approx(31/12.5)
    assert r['tokens_per_second']==pytest.approx(12.5)
    assert r['first_token_included'] is False


@pytest.mark.parametrize('values',[[],[1],[1,1],[2,1],[0,float('inf')],[0,float('nan')],[False,1]])
def test_bad_timestamps_refused(values):
    with pytest.raises(ValueError):decode_intervals(values)


def test_preregistered_balanced_pair_order_is_reproducible_and_never_executes():
    a=campaign(binding(),{'deepseek_source_submission':'shared_first_v1'},seed=20261026)
    assert a==campaign(binding(),{'deepseek_source_submission':'shared_first_v1'},seed=20261026)
    assert len(a['windows'])==10 and len(a['aa_controls'])==2
    for i in range(5):
        rows=[w for w in a['windows'] if w['pair']==i]
        assert {w['arm'] for w in rows}=={'reference','candidate'}
        assert all(w['release_after_window'] for w in rows)
    assert a['executed'] is False and a['gpu_authorization']=='pending'
    assert a['windows'][0]['cells'][1]['actual_prompt_tokens']==100
    assert a['windows'][0]['cells'][1]['context_capacity']==16384


@pytest.mark.parametrize('mutation',[
    {'checkpoint':'model'}, {'source_revision':'x'*40},{'source_snapshot':'g'*64},
    {'hardware':''},{'hotmap_sha256':'d'*63}])
def test_incomplete_binding_refused(mutation):
    with pytest.raises(ValueError):replace(binding(),**mutation)


@pytest.mark.parametrize('mutation',[
    {'context_capacity':120},{'cached_tokens':101},{'physical_width':False},
    {'speculation':True},{'live_host_reserve_bytes':0},{'original_bytes':75_000_000_000}])
def test_invalid_serving_shape_refused(mutation):
    with pytest.raises(ValueError):Shape('decode',**mutation)


def test_pp_and_long_interval_shapes_are_explicit():
    with pytest.raises(ValueError):Shape('pp')
    with pytest.raises(ValueError):Shape('long_decode')
    Shape('pp',output_tokens=1,cached_tokens=0)
    Shape('long_decode',output_tokens=513)


@pytest.mark.parametrize('gate',REQUIRED_GATES)
def test_each_gate_is_required_and_missing_cannot_mean_passed(gate):
    value=receipt(0,'reference');value['gates'].pop(gate)
    with pytest.raises(ValueError,match='gates'):validate_receipt(value,binding=binding(),shape=Shape('decode'))


def test_partial_bookend_and_changed_shape_never_form_a_pair():
    pairs=[(receipt(i,'reference'),receipt(i,'candidate',step=.04)) for i in range(5)]
    pairs[-1][1]['completed']=False
    with pytest.raises(ValueError,match='partial'):paired_analysis(pairs,binding=binding(),shape=Shape('decode'),seed=1)
    pairs[-1][1]['completed']=True;pairs[-1][1]['shape']['apc_bytes']*=2
    with pytest.raises(ValueError,match='shape mismatch'):paired_analysis(pairs,binding=binding(),shape=Shape('decode'),seed=1)


def test_paired_analysis_bootstraps_independent_windows_and_keeps_raw_pairs():
    pairs=[(receipt(i,'reference'),receipt(i,'candidate',step=.04)) for i in range(5)]
    r=paired_analysis(pairs,binding=binding(),shape=Shape('decode'),seed=2)
    assert r['median_paired_speed_ratio']==pytest.approx(2)
    assert r['window_bootstrap_95_interval']==pytest.approx([2,2])
    assert len(r['pairs'])==5 and r['verdict']=='measured_win'
    assert r['performance_qualified'] is False and r['tail_latency_claim'] is False
    assert r==paired_analysis(pairs,binding=binding(),shape=Shape('decode'),seed=2)


def test_duplicate_pair_is_not_an_independent_replica():
    pair=(receipt(0,'reference'),receipt(0,'candidate'))
    with pytest.raises(ValueError,match='repeated'):paired_analysis([pair]*5,binding=binding(),shape=Shape('decode'),seed=1)


def test_no_noise_win_from_identical_control():
    pairs=[(receipt(i,'reference'),receipt(i,'candidate')) for i in range(5)]
    r=paired_analysis(pairs,binding=binding(),shape=Shape('decode'),seed=1)
    assert r['verdict']=='inconclusive_or_no_win'


def test_receipt_rejects_bad_counts_and_durations():
    for field,bad in (('cached_tokens',0),('actual_prompt_tokens',16384),('output_tokens',31)):
        r=receipt(0,'reference');r['requests'][0][field]=bad
        with pytest.raises(ValueError):validate_receipt(r,binding=binding(),shape=Shape('decode'))
    s=Shape('pp',output_tokens=1,cached_tokens=0);r=receipt(0,'reference',shape=s)
    r['requests'][0]['prefill_seconds']=float('nan')
    with pytest.raises(ValueError):validate_receipt(r,binding=binding(),shape=s)


def test_atomic_no_overwrite_and_invalid_json_leaves_no_partial_receipt(tmp_path):
    p=tmp_path/'receipt.json';write_new_json(p,{'value':1})
    with pytest.raises(FileExistsError):write_new_json(p,{'value':2})
    assert json.loads(p.read_text())=={'value':1}
    with pytest.raises(ValueError):write_new_json(tmp_path/'bad.json',{'value':math.nan})
    assert not (tmp_path/'bad.json').exists()
    assert list(tmp_path.glob('.*'))==[]


def test_source_snapshot_is_order_independent_detects_drift_and_refuses_escape(tmp_path):
    (tmp_path/'a.py').write_text('a');(tmp_path/'b.py').write_text('b')
    a=source_snapshot(tmp_path,['a.py','b.py'])
    assert a==source_snapshot(tmp_path,['b.py','a.py','b.py'])
    (tmp_path/'b.py').write_text('c')
    assert a['sha256']!=source_snapshot(tmp_path,['a.py','b.py'])['sha256']
    with pytest.raises(ValueError):source_snapshot(tmp_path,['../outside'])
    with pytest.raises(ValueError):source_snapshot(tmp_path,[])


def test_runtime_freeze_includes_cpu_thermal_probe_and_shell_owners(tmp_path):
    scripts=tmp_path/'scripts';scripts.mkdir()
    for name in ('thermal_probe.swift','run_with_gpu_locks.py','gpu_owner.sh'):
        (scripts/name).write_text('source')
    names=runtime_source_paths(tmp_path)
    assert set(names)=={'scripts/thermal_probe.swift','scripts/run_with_gpu_locks.py','scripts/gpu_owner.sh'}
