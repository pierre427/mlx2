"""Mocked wiring receipt/cleanup; no Metal dispatch or model load."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ds4_probe_residency', ROOT/'scripts/research/deepseek_v4_probe_residency.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def test_wiring_receipt_restores_previous_limit_once(monkeypatch):
    from mlx2.runtime import weight_residency
    calls=[]
    monkeypatch.setattr(weight_residency,'wire_serving_weights',lambda:{'wired':True,'previous_wired_limit_bytes':123,'wired_limit_bytes':900,'active_bytes_at_wire':850})
    mx=SimpleNamespace(device_info=lambda:{'max_recommended_working_set_size':900},set_wired_limit=lambda n:(calls.append(n) or 900))
    receipt={}
    result=probe.wire_preloaded_weights(mx,receipt)
    assert result['active_bytes_at_wire']==850
    assert result['device_working_set_cap_bytes']==900
    assert 'not independent' in result['observation']
    probe.restore_probe_wired_limit(mx,receipt)
    probe.restore_probe_wired_limit(mx,receipt)
    assert calls==[123] and result['prior_limit_restored']


def test_receipt_failure_keeps_restore_authority(monkeypatch):
    from mlx2.runtime import weight_residency
    monkeypatch.setattr(weight_residency,'wire_serving_weights',lambda:{'wired':True,'previous_wired_limit_bytes':0})
    def fail():raise RuntimeError('device receipt failure')
    calls=[];mx=SimpleNamespace(device_info=fail,set_wired_limit=lambda n:(calls.append(n) or 900))
    receipt={}
    with pytest.raises(RuntimeError):probe.wire_preloaded_weights(mx,receipt)
    probe.restore_probe_wired_limit(mx,receipt)
    assert calls==[0]


def test_no_wiring_never_mutates_limit():
    mx=SimpleNamespace(set_wired_limit=lambda n:pytest.fail('unexpected mutation'))
    probe.restore_probe_wired_limit(mx,{})
    probe.restore_probe_wired_limit(mx,{'weight_residency':{'wired':False}})


def test_harness_wiring_after_preload_before_cache_and_restore_in_finally():
    # Guard the actual integration ordering, not just helper behavior.
    for name in ('ordinary','dspark'):
        source=(ROOT/f'scripts/research/probe_deepseek_v4_{name}.py').read_text()
        wired=source.index('wire_preloaded_weights(mx, receipt)')
        cache=source.index('cache=model.make_cache()') if name=='ordinary' else source.index('cache = model.make_cache()')
        assert source.index('adapter=DeepSeekV4GGUFAdapter' if name=='ordinary' else 'adapter = DeepSeekV4GGUFAdapter')<wired<cache
        assert source.rindex('finally:')<source.index('restore_probe_wired_limit(mx, receipt)')
        if name=='dspark':assert source.index('mx.eval(list(weights._dense.values()))')<wired
