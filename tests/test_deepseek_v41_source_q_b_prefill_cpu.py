"""Source PP Q_b integration; synthetic CPU stubs, no Metal execution."""
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import mlx.core as mx
import pytest
from mlx2.adapters import deepseek_v41_source_weights as source
from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter,source_policy
from mlx2.adapters.deepseek_v41_mlx import execution_policy
from mlx2.runtime.gguf import source_fp8_projection_tiles as tiles,source_dense_legacy as legacy

KEY='deepseek_source_q_b_prefill';NAME='blk.2.attn_q_b.weight'

@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Metal forbidden')))
    yield
    mx.set_default_device(old)

@pytest.fixture
def w():
    w=source.SourceV41Weights.__new__(source.SourceV41Weights)
    w.source_q_b_prefill_policy='tile64x32_raw'
    q=mx.broadcast_to(mx.zeros((1,1),mx.uint32),(32768,320))
    s=mx.broadcast_to(mx.zeros((1,1),mx.uint8),(32768,40))
    w._quant={NAME:(q,s)};w._ensure_dense=w._ready=lambda:None;w.scheduled_dense_calls={}
    def prepare(x):
        w.scheduled_dense_calls['test_qat']=w.scheduled_dense_calls.get('test_qat',0)+1
        return x.astype(mx.float32)+1
    w._dense_input=prepare
    return w


def install_math(monkeypatch,*,device=True):
    calls=[];stock=[];pins=[]
    monkeypatch.setattr(legacy,'_device_supported',lambda:device)
    monkeypatch.setattr(source,'_verify_q_b_prefill_source',lambda:pins.append(True))
    def project(x,q,s,kind,**options):
        out=mx.broadcast_to(mx.array(1.00390625,mx.float32),(1,99,32768))
        calls.append((x,q,s,kind,options,out));return out
    def qmm(x,q,*,scales,mode):
        stock.append((x,q,scales,mode));return mx.broadcast_to(mx.array(1.01171875,mx.float32),(*x.shape[:-1],q.shape[0]))
    monkeypatch.setattr(tiles,'project',project);monkeypatch.setattr(mx,'quantized_matmul',qmm)
    return calls,stock,pins


def test_policy_default_source_only_and_hash_binding():
    assert KEY not in source_policy()
    assert KEY in DeepSeekV41SourceAdapter.adapter_owned_default_policy_keys
    off=source_policy({KEY:'reference'});on=source_policy({KEY:'tile64x32_raw'})
    assert off[KEY]=='reference'and on[KEY]=='tile64x32_raw'
    digest=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
    assert digest(off)!=digest(on)
    for p in ({KEY:'bad'},{KEY:True},{KEY:None},{KEY:[]}):
        with pytest.raises(ValueError):source_policy(p)
    with pytest.raises(ValueError):execution_policy({KEY:'tile64x32_raw'})


def test_constructor_rejects_unknown_before_artifact():
    with pytest.raises(ValueError,match='Q_b prefill'):source.SourceV41Weights({},expert_cache_bytes=1,q_b_prefill_policy='bad')

@pytest.mark.parametrize('policy',['reference','tile64x32_raw'])
def test_constructor_wires_instance_policy(monkeypatch,tmp_path,policy):
    monkeypatch.setattr(source,'weight_plan',lambda *a:{})
    monkeypatch.setattr(source,'ReadPool',lambda *a,**kw:SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(source,'SourceExpertStore',lambda *a,**kw:SimpleNamespace())
    w=source.SourceV41Weights(dict(root=tmp_path,aliases={},tensors={}),expert_cache_bytes=1,q_b_prefill_policy=policy)
    assert w.source_q_b_prefill_policy==policy


def test_raw_qb_tile_uses_same_prepared_input_original_objects_no_owner_round(monkeypatch,w):
    calls,stock,pins=install_math(monkeypatch);x=mx.zeros((1,99,1280),mx.float32)
    result=w.linear(NAME)(x)
    assert result is calls[0][-1]and not stock and pins==[True]
    assert calls[0][1]is w._quant[NAME][0]and calls[0][2]is w._quant[NAME][1]
    assert calls[0][3:] == ('q_b',{'tile':(64,32),'fused_boundary':False},result)
    assert float(np.asarray(calls[0][0][0,0,0]))==1
    assert float(np.asarray(result[0,0,0]))==1.00390625
    assert w.scheduled_dense_calls=={'test_qat':1,'source_q_b_prefill_tile64x32_raw':1}


def test_model_linear_retains_raw_and_owning_BF16_modes(monkeypatch,w):
    from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
    calls,stock,_=install_math(monkeypatch);m=DeepSeekV41Model.__new__(DeepSeekV41Model)
    m.weights=w;m.args=SimpleNamespace(activation_dtype='bfloat16');x=mx.zeros((1,99,1280),mx.float32)
    raw=m.linear(x,NAME,rounded=False);rounded=m.linear(x,NAME,rounded=True)
    assert float(np.asarray(raw[0,0,0]))==1.00390625
    assert float(np.asarray(rounded[0,0,0]))==1.
    assert len(calls)==2 and not stock

@pytest.mark.parametrize('shape',[(1,1,1280),(1,98,1280),(1,100,1280),(2,99,1280),(99,1280),(1,99,1279)])
def test_other_rows_widths_batch_fall_back_stock(monkeypatch,w,shape):
    calls,stock,pins=install_math(monkeypatch);w.linear(NAME)(mx.zeros(shape,mx.float32))
    assert not calls and len(stock)==1 and not pins
    assert w.scheduled_dense_calls['source_q_b_prefill_fallback_shape']==1
    assert w.scheduled_dense_calls['official_mxfp8_act32']==1

@pytest.mark.parametrize('dtype',[mx.bfloat16,mx.float16])
def test_original_input_dtype_gates_before_prepared_F32(monkeypatch,w,dtype):
    calls,stock,_=install_math(monkeypatch);w.linear(NAME)(mx.zeros((1,99,1280),dtype))
    assert not calls and len(stock)==1 and stock[0][0].dtype==mx.float32

@pytest.mark.parametrize('name',['blk.2.attn_output_b.weight','blk.2.attn_q_a.weight','blk.40.attn_q_b.weight','blk.02.attn_q_b.weight'])
def test_foreign_aliases_never_select(monkeypatch,w,name):
    calls,stock,pins=install_math(monkeypatch);w._quant[name]=w._quant[NAME]
    w.linear(name)(mx.zeros((1,99,1280),mx.float32))
    assert not calls and not pins and len(stock)==1
    assert not any('q_b_prefill' in k for k in w.scheduled_dense_calls)

@pytest.mark.parametrize('field',['q_dtype','q_shape','scale_dtype','scale_shape'])
def test_coefficient_gate_preserves_ordinary_path(monkeypatch,w,field):
    calls,stock,pins=install_math(monkeypatch);q,s=w._quant[NAME]
    if field=='q_dtype':q=q.astype(mx.int32)
    if field=='q_shape':q=q[:32767]
    if field=='scale_dtype':s=s.astype(mx.uint32)
    if field=='scale_shape':s=s[:,:39]
    w._quant[NAME]=(q,s);w.linear(NAME)(mx.zeros((1,99,1280)))
    assert not calls and not pins and len(stock)==1


def test_reference_policy_no_device_or_source_probe(monkeypatch,w):
    calls,stock,pins=install_math(monkeypatch);w.source_q_b_prefill_policy='reference'
    monkeypatch.setattr(legacy,'_device_supported',lambda:(_ for _ in ()).throw(AssertionError('default must not probe')))
    w.linear(NAME)(mx.zeros((1,99,1280)))
    assert not calls and not pins and len(stock)==1


def test_unsupported_device_falls_back(monkeypatch,w):
    calls,stock,pins=install_math(monkeypatch,device=False);w.linear(NAME)(mx.zeros((1,99,1280)))
    assert not calls and not pins and len(stock)==1
    assert w.scheduled_dense_calls['source_q_b_prefill_fallback_device']==1


def test_pin_failure_is_not_silent_stock_fallback(monkeypatch,w):
    calls,stock,_=install_math(monkeypatch)
    monkeypatch.setattr(source,'_verify_q_b_prefill_source',lambda:(_ for _ in ()).throw(RuntimeError('pin drift')))
    with pytest.raises(RuntimeError,match='pin drift'):w.linear(NAME)(mx.zeros((1,99,1280)))
    assert not calls and not stock and 'source_q_b_prefill_tile64x32_raw'not in w.scheduled_dense_calls


def test_kernel_failure_not_retried_with_altered_graph(monkeypatch,w):
    _,stock,_=install_math(monkeypatch)
    monkeypatch.setattr(tiles,'project',lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('kernel failure')))
    with pytest.raises(RuntimeError,match='kernel failure'):w.linear(NAME)(mx.zeros((1,99,1280)))
    assert not stock and 'source_q_b_prefill_tile64x32_raw'not in w.scheduled_dense_calls


def test_source_pin_verifier_hashes_all_dependencies_and_installed(monkeypatch,tmp_path):
    from mlx2.runtime.gguf import source_prefill_legacy_tiles as installed
    root=Path(tiles.__file__).parent;calls=[]
    for name in source.Q_B_PREFILL_SOURCE_PINS:(tmp_path/name).write_bytes((root/name).read_bytes())
    monkeypatch.setattr(tiles,'__file__',str(tmp_path/'source_fp8_projection_tiles.py'))
    monkeypatch.setattr(installed,'_installed',lambda:calls.append(True))
    source._verify_q_b_prefill_source.cache_clear()
    try:
        source._verify_q_b_prefill_source();assert calls==[True]
        (tmp_path/'source_prefill_legacy_tiles.py').write_text('changed')
        source._verify_q_b_prefill_source.cache_clear()
        with pytest.raises(RuntimeError,match='changed'):source._verify_q_b_prefill_source()
    finally:source._verify_q_b_prefill_source.cache_clear()


def test_dense_receipt_and_adapter_wiring(monkeypatch,w):
    from mlx2.adapters.deepseek_v41_mlx_weights import NativeV41Weights
    monkeypatch.setattr(NativeV41Weights,'dense_receipt',lambda self:{})
    receipt=w.dense_receipt();assert receipt['source_q_b_prefill']=='tile64x32_raw'
    text=inspect.getsource(DeepSeekV41SourceAdapter._load)
    assert "q_b_prefill_policy=self.kernel_policy.get('deepseek_source_q_b_prefill','reference')" in text
    assert 'json.dumps(self.kernel_policy, sort_keys=True)'in text
