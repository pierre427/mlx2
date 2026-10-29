"""CPU regression for long-output Laguna admission; no model loading."""
import pytest
from mlx2.adapters.laguna_memory import LagunaCacheBudget
from mlx2.adapters.laguna_s21 import LagunaS21Adapter
from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController as Controller
from mlx2.serving import lane_admission_required_gib


def config(layers=48):
    return dict(num_hidden_layers=layers, num_key_value_heads=8, head_dim=128,
                sliding_window=512, hidden_size=3072, vocab_size=100352,
                layer_types=['full_attention' if i % 4 == 0 else 'sliding_attention' for i in range(layers)])


def test_adapter_supplies_geometry_without_loading():
    layers,global_layers=48,12
    a=object.__new__(LagunaS21Adapter);a.config=config(layers)
    b=a.cache_budget(mtp=False)
    assert b.global_layers==global_layers and b.sliding_layers==layers-global_layers
    assert b.project(65536) < 7 * 2**30
    assert b.project(256000) < 25 * 2**30
    with pytest.raises(ValueError,match='native MTP'):a.cache_budget(mtp=True)


def test_same_64k_request_fits_observed_headroom_without_weakening_reserves():
    b=LagunaCacheBudget.from_config(config(),mtp=False)
    old=Controller(host_memory_gib=128,advisory_gib=112,saturation_lane_cap=2)
    new=Controller(host_memory_gib=128,advisory_gib=112,saturation_lane_cap=2,cache_estimator=b.project)
    kwargs=dict(context_tokens=65536+1024,draft_depth=0,cache_gib=0,prefill_gib=b.prefill_transient_bytes(1024,1024)/2**30)
    # The resident campaign's idle headroom was 50,778,783,744 bytes.
    headroom=50778783744/2**30
    assert lane_admission_required_gib(old,**kwargs)>headroom
    assert lane_admission_required_gib(new,**kwargs)<headroom
    assert new.hard_reserve_gib==old.hard_reserve_gib==20
    # Reserving two cold lanes also fits this conservative live headroom.
    per_lane=lane_admission_required_gib(new,**kwargs)-new.hard_reserve_gib
    assert new.hard_reserve_gib+2*per_lane<headroom


def test_window_restore_count_and_kv_heads_control_bound(monkeypatch):
    monkeypatch.setenv('MLX_LM_STATE_CHECKPOINT_MAX','0');a=LagunaCacheBudget.from_config(config(),mtp=False)
    monkeypatch.setenv('MLX_LM_STATE_CHECKPOINT_MAX','9');b=LagunaCacheBudget.from_config(config(),mtp=False)
    assert b.project(65536)-a.project(65536)==36*512*2*8*128*4*9
    c=config();c['num_key_value_heads']=16
    doubled=LagunaCacheBudget.from_config(c,mtp=False)
    assert doubled.project(65536)>b.project(65536)
    points=[b.project(n) for n in [0,1,255,256,257,511,512,513,65536,256000]]
    assert points==sorted(points)
    assert b.prefill_transient_bytes(65536,1024)>b.prefill_transient_bytes(1024,1024)


@pytest.mark.parametrize('value',[-1,True,1.5])
def test_invalid_lengths_refused(value):
    b=LagunaCacheBudget.from_config(config(),mtp=False)
    with pytest.raises(ValueError):b.project(value)
    with pytest.raises(ValueError):b.prefill_transient_bytes(1,value)


@pytest.mark.parametrize('field,value',[('head_dim',0),('num_key_value_heads',True),('layer_types',['unknown'])])
def test_unknown_geometry_refused(field,value):
    c=config();c[field]=value
    with pytest.raises(ValueError):LagunaCacheBudget.from_config(c,mtp=False)
