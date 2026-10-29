"""Architecture-derived bounds: physical FP32, not simulated 8-bit storage."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
from mlx2.adapters.deepseek_v4_memory import DeepSeekV4CacheBudget


def config(ratios=(0,4,128)):
    return SimpleNamespace(num_hidden_layers=len(ratios),compress_ratios=ratios)


def test_cache_counts_one_shared_tensor_and_ratio_specific_buffers():
    b=DeepSeekV4CacheBudget.from_config(config())
    assert b.local_live_bytes==3*128*512*4
    # ratio4: compressor512 and indexer128, twice-wide partial projections;
    # plus previous completed first-halfKV andgate. ratio128 no previouscarry.
    expected=2*3*2*(512+128)*4 + 2*4*(512+128)*4 + 2*127*512*4
    assert b.compression_fixed_bytes==expected
    assert b.resident_fixed_bytes==b.local_live_bytes+expected+3*4096


def test_pool_projection_uses_context_over_ratio_and_single_shared_kv():
    b=DeepSeekV4CacheBudget.from_config(config())
    context=65536
    r4=(16384+256)*(512+128)*4
    r128=(512+256)*512*4
    assert b.pooled_bytes(context)==r4+r128
    assert b.project_resident(context)==b.resident_fixed_bytes+r4+r128+context*16
    assert b.project(context)==b.project_resident(context)+b.snapshot_bytes(context)


def test_snapshot_charged_only_to_running_lanes_and_grows_with_pools():
    b=DeepSeekV4CacheBudget.from_config(config())
    assert b.project_pool(65536,resident_lanes=4,running_lanes=0)==4*b.project_resident(65536)
    assert b.project_pool(65536,resident_lanes=4,running_lanes=2)==4*b.project_resident(65536)+2*b.snapshot_bytes(65536)
    assert b.snapshot_bytes(131072)>b.snapshot_bytes(65536)
    assert b.project_pool(0,resident_lanes=0,running_lanes=0)==0


def test_prefill_transient_separate_and_reserves_unchanged():
    b=DeepSeekV4CacheBudget.from_config(config())
    assert b.prefill_transient_bytes(65536,0)==0
    one=b.prefill_transient_bytes(65536,512)
    two=b.prefill_transient_bytes(65536,1024)
    assert two>one>b.snapshot_bytes(65536)
    assert b.transient_gib_per_lane==1.76
    assert b.project(65536)==b.project_resident(65536)+b.snapshot_bytes(65536)
    receipt=b.as_dict()
    assert receipt['physical_storage']=='float32-shared-kv-and-pools'
    assert 'does-not-pack-storage' in receipt['activation_simulation']


@pytest.mark.parametrize('method,args,kwargs',[
    ('project',(-1,),{}),('project',(True,),{}),('project',(3.5,),{}),
    ('project_pool',(5,),{'resident_lanes':1,'running_lanes':2}),
    ('project_pool',(5,),{'resident_lanes':True,'running_lanes':0}),
    ('prefill_transient_bytes',(5,-1),{}),('prefill_transient_bytes',(5,2.5),{}),
])
def test_invalid_projection_inputs_fail_closed(method,args,kwargs):
    b=DeepSeekV4CacheBudget.from_config(config())
    with pytest.raises(ValueError): getattr(b,method)(*args,**kwargs)


def test_invalid_geometry_not_guessed():
    with pytest.raises(ValueError): DeepSeekV4CacheBudget.from_config(config((0,3)))
    c=config();c.num_hidden_layers=43
    with pytest.raises(ValueError): DeepSeekV4CacheBudget.from_config(c)
    b=DeepSeekV4CacheBudget.from_config(config())
    with pytest.raises(ValueError): replace(b,item_bytes=1)
