"""CPU geometry and real-profile admission regression for ordinary Ling."""
import pytest
from mlx2.adapters.ling_memory import LingCacheBudget
from mlx2.adapters.ling3_flash import Ling3FlashAdapter
from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController as Controller
from mlx2.serving import lane_admission_required_gib


def config(count=42):
    return dict(num_hidden_layers=count, layer_group_size=6, kv_lora_rank=512,
                qk_rope_head_dim=64, num_attention_heads=32, head_dim=128,
                qk_nope_head_dim=128, v_head_dim=128, short_conv_kernel_size=4,
                hidden_size=2560, vocab_size=157184)


def test_ordinary_hook_and_no_external_underestimate():
    a=object.__new__(Ling3FlashAdapter);a.config=config();a.draft_model=None
    b=a.cache_budget(mtp=False)
    assert (b.global_layers,b.recurrent_layers)==(7,35)
    assert b.recurrent_live_bytes==35*(32*128*128+3*3*32*128)*4
    assert b.project(65536)<2*2**30
    with pytest.raises(ValueError,match='speculative'):a.cache_budget(mtp=True)
    a.draft_model=object()
    assert a.cache_budget(mtp=True) is None


def test_same_64k_budget_fits_without_lowering_reserves_or_allowance():
    b=LingCacheBudget.from_config(config(),mtp=False)
    old=Controller(host_memory_gib=128,advisory_gib=112,saturation_lane_cap=2)
    new=Controller(host_memory_gib=128,advisory_gib=112,saturation_lane_cap=2,cache_estimator=b.project)
    kwargs=dict(context_tokens=2050+65536,draft_depth=0,cache_gib=0)
    headroom=45791756288/2**30
    assert lane_admission_required_gib(old,prefill_gib=0,**kwargs)>headroom
    required=lane_admission_required_gib(new,prefill_gib=b.prefill_transient_bytes(2050,512)/2**30,**kwargs)
    assert required<headroom
    assert new.hard_reserve_gib==old.hard_reserve_gib==20
    assert new.hard_reserve_gib+2*(required-new.hard_reserve_gib)<headroom


def test_compressed_cache_slope_not_expanded_attention_and_fixed_state(monkeypatch):
    monkeypatch.setenv('MLX_LM_STATE_CHECKPOINT_MAX','4')
    b=LingCacheBudget.from_config(config(),mtp=False)
    assert b.project(65792)-b.project(65536)==256*(7*(512+64)*4+16)
    monkeypatch.setenv('MLX_LM_STATE_CHECKPOINT_MAX','0')
    no_snap=LingCacheBudget.from_config(config(),mtp=False)
    assert b.project(65536)-no_snap.project(65536)==4*b.recurrent_live_bytes
    assert b.prefill_transient_bytes(65536,512)>b.prefill_transient_bytes(65536,128)
    assert b.prefill_transient_bytes(65536,512)>b.prefill_transient_bytes(2050,512)


def test_partial_final_group_uses_runtime_mla_rule():
    b=LingCacheBudget.from_config(config(43),mtp=False)
    assert (b.global_layers,b.recurrent_layers)==(8,35)


@pytest.mark.parametrize('value',[-1,True,0.5])
def test_bad_lengths_refused(value):
    b=LingCacheBudget.from_config(config(),mtp=False)
    with pytest.raises(ValueError):b.project(value)
    with pytest.raises(ValueError):b.prefill_transient_bytes(2,value)


@pytest.mark.parametrize('field,value',[('head_dim',0),('layer_group_size',True),('kv_lora_rank',-1)])
def test_unknown_geometry_refused(field,value):
    c=config();c[field]=value
    with pytest.raises(ValueError):LingCacheBudget.from_config(c,mtp=False)
