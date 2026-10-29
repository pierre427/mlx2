"""Opt-in routing/identity CPU contracts; GPU kernel replaced by a spy."""
from types import SimpleNamespace
import hashlib
import mlx.core as mx
import pytest
from mlx2.runtime.models import deepseek_v41_source_attention as route
from mlx2.runtime.models import deepseek_v41_omlx_nax_attention as native
from mlx2.runtime.gguf import source_dense_legacy
from mlx2.adapters.deepseek_v41_source import source_policy
from test_deepseek_v41_source_attention_broadcast_cpu import _model

@pytest.fixture(autouse=True)
def cpu(monkeypatch):
 old=mx.default_device();mx.set_default_device(mx.cpu)
 monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**k:pytest.fail('GPU forbidden'))
 monkeypatch.setattr(source_dense_legacy,'_device_supported',lambda:True)
 yield
 mx.set_default_device(old)


def operands(n=16,h=64,d=512,p=3):
 return (mx.zeros((1,h,n,d)),mx.zeros((1,1,n,d)),mx.ones((1,1,n,n),mx.bool_),mx.zeros((1,p,d)),mx.zeros((1,n,p),mx.int32),mx.ones((1,n,p),mx.bool_),mx.zeros((h,)))


def test_identity_is_kernel_and_dispatch_bound_and_apc_separated():
 from pathlib import Path
 assert hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()==route.KERNEL_SHA256
 expected=hashlib.sha256((route.POLICY+':'+route.KERNEL_SHA256+':'+hashlib.sha256(Path(route.__file__).read_bytes()).hexdigest()).encode()).hexdigest()
 assert route.implementation_identity()==expected
 a,b=_model('all_retained_broadcast'),_model(route.POLICY)
 assert expected in b.identity and a.identity!=b.identity
 with pytest.raises(ValueError,match='identity'):b(mx.array([[1]],mx.int32),a.make_cache())
 assert source_policy({'deepseek_index_policy':'finite_all_retained','deepseek_source_attention_math':route.POLICY})['deepseek_source_attention_math']==route.POLICY
 with pytest.raises(ValueError):source_policy({'deepseek_source_attention_math':route.POLICY})


def test_real_dispatch_engages_pinned_candidate_and_counts(monkeypatch):
 calls=[];sentinel=object();a=operands();c={}
 monkeypatch.setattr(native,'attention',lambda *args,**kw:(calls.append((args,kw))or sentinel))
 assert route.dispatch(*a,trusted_retained=True,reference=lambda:pytest.fail('fallback'),counters=c)is sentinel
 assert calls[0][1]['numerical_profile']==native.PROFILE
 assert c['source_omlx_nax_attention']==1
 assert c['source_omlx_nax_operand_staging_logical_bytes']==2*(a[0].size+a[1].size+a[3].size)

@pytest.mark.parametrize('kind',['decode','small','wide','heads','dim','marker','device','empty'])
def test_reference_fallback(monkeypatch,kind):
 kw={};trusted=True
 if kind in ('decode','small','wide'):kw['n']={'decode':1,'small':15,'wide':129}[kind]
 if kind=='heads':kw['h']=32
 if kind=='dim':kw['d']=256
 if kind=='empty':kw['p']=0
 if kind=='marker':trusted=False
 if kind=='device':monkeypatch.setattr(source_dense_legacy,'_device_supported',lambda:False)
 monkeypatch.setattr(native,'attention',lambda *a,**kw:pytest.fail('candidate forbidden'))
 c={};sentinel=object()
 assert route.dispatch(*operands(**kw),trusted_retained=trusted,reference=lambda:sentinel,counters=c)is sentinel
 assert c=={'source_omlx_nax_attention_reference_fallback':1}


def test_actual_model_fallback_uses_broadcast_then_decode_reference(monkeypatch):
 from mlx2.runtime.models import deepseek_v41 as model
 m=_model(route.POLICY);cache=m.make_cache();calls=[];old=model.selected_attention
 def selected(*a,**kw):calls.append((a[0].shape[2],kw.get('trusted_broadcast',False)));return old(*a,**kw)
 monkeypatch.setattr(model,'selected_attention',selected)
 mx.eval(m(mx.array([[1,2,3,4,5,6,7]],mx.int32),cache))
 assert [v for _,v in calls]==[False,True,True,True,True]
 assert m.scheduled_kv_calls['source_omlx_nax_attention_reference_fallback']==5
 calls.clear();mx.eval(m(mx.array([[8]],mx.int32),cache))
 assert [v for _,v in calls]==[False]*5
 assert m.scheduled_kv_calls.get('source_omlx_nax_attention',0)==0


def test_pin_drift_fails_closed(monkeypatch):
 route.implementation_identity.cache_clear()
 monkeypatch.setattr(route,'KERNEL_SHA256','0'*64)
 with pytest.raises(RuntimeError,match='re-audit'):route.implementation_identity()
 route.implementation_identity.cache_clear()


def test_attention_call_actual_geometry_engages_and_preserves_owner(monkeypatch):
 from mlx2.runtime.models import deepseek_v41 as v41
 from test_deepseek_v41_cpu import config,Weights
 c=config(num_attention_heads=64,head_dim=512,index_topk=512,candidate_topk_blocks=2048)
 m=v41.DeepSeekV41Model(c,Weights(c),identity='synthetic@omlx-route',index_policy='finite_all_retained',source_attention_math_policy=route.POLICY)
 calls=[]
 def kernel(*a,**kw):
  calls.append(a[0].shape)
  return v41.selected_attention(*a,trusted_broadcast=True)
 monkeypatch.setattr(native,'attention',kernel)
 cache=m.make_cache();mx.eval(m(mx.array([[1]*16],mx.int32),cache))
 assert calls==[(1,64,16,512)]*4
 assert m.scheduled_kv_calls['source_omlx_nax_attention']==4
 assert m.scheduled_kv_calls['source_attention_math_reference']==1
 assert m.scheduled_kv_calls['source_omlx_nax_attention_reference_fallback']==1
 mx.eval(m(mx.array([[1]],mx.int32),cache))
 assert len(calls)==4 and m.scheduled_kv_calls['source_omlx_nax_attention_reference_fallback']==6

@pytest.mark.parametrize('mixed',[False,True])
def test_diagnostics_describe_native_numerical_law_for_both_storage_profiles(monkeypatch,mixed):
 from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter,DeepSeekV41MLXAdapter
 monkeypatch.setattr(DeepSeekV41MLXAdapter,'diagnostics',lambda self:{})
 a=DeepSeekV41SourceAdapter.__new__(DeepSeekV41SourceAdapter)
 a.identity={'fingerprint':'synthetic@source-attention-diagnostics'};a.native_admission={}
 a.draft_model=None
 a.kernel_policy={'deepseek_source_experts':{'mode':'hot_original_q1'}}if mixed else{}
 a.weights=SimpleNamespace(expert_math_policy='reference',dense_receipt=lambda:{},source_qakv_policy='reference',source_q_b_prefill_policy='reference')
 a.model=SimpleNamespace(source_attention_math_policy=route.POLICY,source_hc_boundary_policy='reference',index_policy='finite_all_retained',selected_kv_policy='decode_reuse',scheduled_hc_calls={},hc_coeff_policy='reference')
 d=a.diagnostics()
 assert d['source_attention_numerical_profile']==native.PROFILE
 assert d['source_attention_implementation_sha256']==route.implementation_identity()
 assert 'MLX reference reductions'not in d['numerical_scope']and 'online F32 softmax'in d['numerical_scope']
 assert ('Q1 expert derivative'in d['numerical_scope'])==mixed
