"""CPU-only Source policy/dispatch oracle; actual QMM replaced with spy."""
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace
import mlx.core as mx
import pytest
from mlx2.adapters import deepseek_v41_source_fp8_pp as native
from mlx2.adapters import deepseek_v41_source_weights as source
from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter,source_policy
from mlx2.adapters.deepseek_v41_mlx import execution_policy,DeepSeekV41MLXAdapter
from mlx2.runtime.gguf import source_dense_legacy

KEY='deepseek_source_fp8_pp'

@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**k:pytest.fail('GPU forbidden'))
    monkeypatch.setattr(source_dense_legacy,'_device_supported',lambda:True)
    yield
    mx.set_default_device(old)


def weights(kind='attn_q_b',policy=native.POLICY):
    k,n=native.SHAPES[kind];name='blk.2.'+kind+'.weight'
    w=source.SourceV41Weights.__new__(source.SourceV41Weights)
    w.source_fp8_pp_policy=policy;w.source_q_b_prefill_policy='reference'
    w._quant={name:(mx.broadcast_to(mx.zeros((1,1),mx.uint32),(n,k//4)),mx.broadcast_to(mx.zeros((1,1),mx.uint8),(n,k//32)))}
    w._ensure_dense=w._ready=lambda:None;w.scheduled_dense_calls={};w._dense_input=lambda x:x.astype(mx.float32)
    return w,name,k,n


def install(monkeypatch):
    calls=[]
    def qmm(x,q,*,scales,mode):
        calls.append((x,q,scales,mode))
        return mx.broadcast_to(mx.array(1.0,x.dtype),(*x.shape[:-1],q.shape[0]))
    monkeypatch.setattr(mx,'quantized_matmul',qmm)
    monkeypatch.setattr(native,'implementation_identity',lambda:'pinned-implementation')
    return calls


def test_policy_default_incompatible_foreign_and_invalid():
    assert KEY not in source_policy()
    assert KEY in DeepSeekV41SourceAdapter.adapter_owned_default_policy_keys
    assert source_policy({KEY:native.POLICY})[KEY]==native.POLICY
    with pytest.raises(ValueError):execution_policy({KEY:native.POLICY})
    with pytest.raises(ValueError,match='cannot combine'):source_policy({KEY:native.POLICY,'deepseek_source_q_b_prefill':'tile64x32_raw'})
    for bad in (False,None,[],{},'auto'):
        with pytest.raises(ValueError):source_policy({KEY:bad})


def test_constructor_refuses_before_loading_and_wires(monkeypatch,tmp_path):
    with pytest.raises(ValueError,match='FP8 PP'):source.SourceV41Weights({},expert_cache_bytes=1,fp8_pp_policy='bad')
    with pytest.raises(ValueError,match='cannot combine'):source.SourceV41Weights({},expert_cache_bytes=1,fp8_pp_policy=native.POLICY,q_b_prefill_policy='tile64x32_raw')
    monkeypatch.setattr(source,'weight_plan',lambda *a:{})
    monkeypatch.setattr(source,'ReadPool',lambda *a,**k:SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(source,'SourceExpertStore',lambda *a,**k:SimpleNamespace())
    w=source.SourceV41Weights(dict(root=tmp_path,aliases={},tensors={}),expert_cache_bytes=1,fp8_pp_policy=native.POLICY)
    assert w.source_fp8_pp_policy==native.POLICY


@pytest.mark.parametrize('kind',native.SHAPES)
def test_real_source_linear_dispatch_preserves_weight_objects_and_counts(monkeypatch,kind):
    w,name,k,n=weights(kind);calls=install(monkeypatch);x=mx.zeros((1,99,k),mx.float32)
    result=w.linear(name)(x)
    assert len(calls)==1 and calls[0][0].dtype==mx.bfloat16
    assert calls[0][1]is w._quant[name][0]and calls[0][2]is w._quant[name][1]
    assert calls[0][3]=='mxfp8'and result.dtype==mx.float32 and result.shape==(1,99,n)
    assert w.scheduled_dense_calls=={'source_fp8_native_pp':1,'source_fp8_native_pp_'+kind:1}


@pytest.mark.parametrize('shape',[(1,1,1280),(1,98,1280),(1,100,1280),(2,99,1280),(99,1280)])
def test_decode_other_PP_shapes_and_widths_fall_back(monkeypatch,shape):
    w,name,k,n=weights();calls=install(monkeypatch);w.linear(name)(mx.zeros(shape,mx.float32))
    assert calls[0][0].dtype==mx.float32
    assert w.scheduled_dense_calls=={'source_fp8_native_pp_reference_fallback':1,'official_mxfp8_act32':1}


@pytest.mark.parametrize('kind',['device','input_dtype','q_dtype','q_shape','scale_dtype','scale_shape'])
def test_bad_device_or_physical_contract_never_selects(monkeypatch,kind):
    w,name,k,n=weights();calls=install(monkeypatch);x=mx.zeros((1,99,k),mx.float32);q,s=w._quant[name]
    if kind=='device':monkeypatch.setattr(source_dense_legacy,'_device_supported',lambda:False)
    elif kind=='input_dtype':x=x.astype(mx.bfloat16)
    elif kind=='q_dtype':q=q.astype(mx.int32)
    elif kind=='q_shape':q=q[:-1]
    elif kind=='scale_dtype':s=s.astype(mx.uint32)
    else:s=s[:,:-1]
    w._quant[name]=(q,s);w.linear(name)(x)
    assert calls[0][0].dtype==mx.float32 and 'source_fp8_native_pp'not in w.scheduled_dense_calls


@pytest.mark.parametrize('name',['blk.2.attn_q_a.weight','blk.2.attn_kv.weight','blk.2.attn_output_a.weight','blk.40.attn_q_b.weight','blk.02.attn_q_b.weight'])
def test_foreign_projections_unchanged(monkeypatch,name):
    w,old,k,n=weights();w._quant[name]=w._quant[old];calls=install(monkeypatch)
    w.linear(name)(mx.zeros((1,99,k),mx.float32))
    assert calls[0][0].dtype==mx.float32 and w.scheduled_dense_calls=={'official_mxfp8_act32':1}


def test_reference_never_probes_native_device_or_identity(monkeypatch):
    w,name,k,n=weights(policy='reference');calls=install(monkeypatch)
    monkeypatch.setattr(native,'maybe_project',lambda *a,**k:pytest.fail('reference must not dispatch'))
    w.linear(name)(mx.zeros((1,99,k)));assert calls[0][0].dtype==mx.float32


def test_pin_failure_not_silent_fallback(monkeypatch):
    w,name,k,n=weights();calls=install(monkeypatch)
    monkeypatch.setattr(native,'implementation_identity',lambda:(_ for _ in ()).throw(RuntimeError('pin drift')))
    with pytest.raises(RuntimeError,match='pin drift'):w.linear(name)(mx.zeros((1,99,k)))
    assert not calls


def test_actual_installed_build_helper_identity_and_drift(monkeypatch):
    native.implementation_identity.cache_clear()
    value=native.implementation_identity()
    own=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
    expected=hashlib.sha256((native.POLICY+':'+native.LIBRARY_SHA256+':'+own+':'+repr(sorted(native.HEADER_PINS.items()))).encode()).hexdigest()
    assert value==expected
    native.implementation_identity.cache_clear();monkeypatch.setattr(native,'LIBRARY_SHA256','0'*64)
    with pytest.raises(RuntimeError,match='build changed'):native.implementation_identity()
    native.implementation_identity.cache_clear()


def test_profile_and_model_APC_binding_precedes_construction(monkeypatch):
    monkeypatch.setattr(native,'implementation_identity',lambda:'implementation-A')
    a=DeepSeekV41SourceAdapter.__new__(DeepSeekV41SourceAdapter);a.kernel_policy={KEY:native.POLICY}
    old_revision=a.numerics_revision;old_profile=a.profile;a._bind_fp8_pp_identity()
    assert 'implementation-A'in a.numerics_revision and a.numerics_revision!=old_revision and a.profile!=old_profile
    code=inspect.getsource(DeepSeekV41SourceAdapter._load)
    assert code.index('self._bind_fp8_pp_identity()')<code.index('binding =')<code.index('self.model =')
    b=DeepSeekV41SourceAdapter.__new__(DeepSeekV41SourceAdapter);b.kernel_policy={};b._bind_fp8_pp_identity()
    assert b.numerics_revision==old_revision and b.profile==old_profile


def test_diagnostics_explicit_distinct_law(monkeypatch):
    monkeypatch.setattr(DeepSeekV41MLXAdapter,'diagnostics',lambda self:{})
    monkeypatch.setattr(native,'implementation_identity',lambda:'implementation-A')
    a=DeepSeekV41SourceAdapter.__new__(DeepSeekV41SourceAdapter);a.kernel_policy={}
    a.weights=SimpleNamespace(expert_math_policy='reference',dense_receipt=lambda:{},source_qakv_policy='reference',source_q_b_prefill_policy='reference',source_fp8_pp_policy=native.POLICY)
    a.model=SimpleNamespace(source_attention_math_policy='all_retained_broadcast',source_hc_boundary_policy='reference',index_policy='finite_all_retained',selected_kv_policy='decode_reuse',scheduled_hc_calls={},hc_coeff_policy='reference')
    d=a.diagnostics()
    assert d['source_fp8_pp_numerical_profile']==native.POLICY and d['source_fp8_pp_implementation_sha256']=='implementation-A'
    assert 'distinct numerical profile'in d['numerical_scope'] and 'MLX reference reductions'not in d['numerical_scope']
