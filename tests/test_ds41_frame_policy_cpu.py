"""Source-only opt-in and identity; no model/GPU."""
from types import SimpleNamespace
import pytest
from mlx2.adapters.deepseek_v41_source import source_policy,DeepSeekV41SourceAdapter
from mlx2.adapters.deepseek_v41_mlx import execution_policy
KEY='deepseek_source_frame_reuse'
GOOD={KEY:'prefill_v1','deepseek_source_expert_math':'routed_fused_fp32','deepseek_source_prefill_batch':'gate_up'}
def test_explicit_source_only_default_and_requirements():
 assert KEY not in source_policy()
 assert KEY in DeepSeekV41SourceAdapter.adapter_owned_default_policy_keys
 assert source_policy(GOOD)[KEY]=='prefill_v1'
 assert source_policy({KEY:'reference'})[KEY]=='reference'
 with pytest.raises(ValueError):execution_policy(GOOD)
 for missing in ('deepseek_source_expert_math','deepseek_source_prefill_batch'):
  opts=dict(GOOD);opts.pop(missing)
  with pytest.raises(ValueError,match='requires'):source_policy(opts)
@pytest.mark.parametrize('bad',[None,1,True,{},[],'bad'])
def test_invalid_profile_refuses(bad):
 with pytest.raises(ValueError):source_policy({**GOOD,KEY:bad})
def test_identity_distinguishes_prefill_profile(monkeypatch):
 from mlx2.adapters import deepseek_v41_source_frame_reuse as h
 monkeypatch.setattr(h,'implementation_identity',lambda:'synthetic-hash')
 x=SimpleNamespace(kernel_policy=GOOD,numerics_revision='ordinary',profile='ordinary')
 DeepSeekV41SourceAdapter._bind_frame_reuse_identity(x)
 assert 'source_frame_reuse=prefill_v1:implementation=synthetic-hash'in x.numerics_revision
 assert ':framereuse-'in x.profile
