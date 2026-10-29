"""Exact preparation contract/source/offline tests, no GPU execution."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import subprocess
import shutil
import numpy as np
import mlx.core as mx
import pytest
from mlx2.runtime.models import deepseek_v41_dwarfstar_staging as s
from mlx2.runtime.models import deepseek_v41_dwarfstar_attention as d


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('GPU execution attempted')))
    yield
    mx.set_default_device(old)


@pytest.fixture
def helper(monkeypatch):
    directory=Path(__file__).resolve().parents[1]/'scripts/research';monkeypatch.syspath_prepend(str(directory))
    spec=importlib.util.spec_from_file_location('dwarfstar_stage_components_cpu',directory/'ds41_dwarfstar_attention_components.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


@pytest.mark.parametrize('local,pool',[(1,1),(1,30),(1,31),(1,32),(101,50),(101,101),(128,512)])
def test_CPU_oracle_matches_old_MLX_concat_cast_mask_BYTES(local,pool):
    rng=np.random.default_rng(124);raw=rng.normal(0,.1,(1,1,local,512)).astype(np.float32)
    comp=rng.normal(0,.1,(1,pool,512)).astype(np.float32)
    # Halfway/tie values and signed zero make approximate equality inadequate.
    raw.ravel()[:8]=np.array([0.,-0.,1+2**-11,1+3*2**-11,-1-2**-11,65504.,2**-24,2**-25],np.float32)
    lm=rng.random((1,1,1,local))>.4;valid=rng.random((1,1,pool))>.4
    actual=s.reference_numpy(np,raw,comp,lm,valid);p=d.plan(1,local,pool);pad=p['padded_keys']-p['key_count']
    bank=mx.concatenate([mx.array(raw)[0,0],mx.array(comp)[0],mx.zeros((pad,512),mx.float32)],0).astype(mx.float16)
    visible=mx.concatenate([mx.array(lm)[0,0],mx.array(valid)[0],mx.zeros((1,pad),mx.bool_)],1)
    mask=mx.where(visible,mx.array(0,mx.float16),mx.array(-65504,mx.float16));blocks=mx.zeros((1,),mx.int8)
    expected=[np.array(x)for x in (bank,mask,blocks)]
    for a,b in zip(actual,expected):assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    assert actual[0][0,1].view(np.uint16)==np.uint16(0x8000)


@pytest.mark.parametrize('local,pool',[(101,50),(101,101),(128,512)])
def test_vector_grid_covers_all_outputs_without_sharedscratch(local,pool):
    p=s.layout(local,pool);assert p['grid']==(p['padded_rows']*128,1,1)
    assert p['grid'][0]%p['threadgroup'][0]==0
    assert p['threadgroup_memory_bytes']==0 and p['input_bindings']+p['output_bindings']==7
    assert p['output_shapes']==[(p['padded_rows'],512),(1,p['padded_rows']),(1,)]
    assert p['logical_F32_concat_intermediate_eliminated_bytes']==p['padded_rows']*512*4


def fixture():
    return [SimpleNamespace(ndim=4,shape=(1,1,101,512),dtype=mx.float32),
        SimpleNamespace(ndim=3,shape=(1,50,512),dtype=mx.float32),
        SimpleNamespace(shape=(1,1,1,101),dtype=mx.bool_),SimpleNamespace(shape=(1,1,50),dtype=mx.bool_)]


@pytest.mark.parametrize('bad',['multipleKV','querywidth','maskdtype','valueprecision','globaldim'])
def test_malformed_geometry_refuses_before_dispatch(bad):
    args=fixture()
    if bad=='multipleKV':args[0].shape=(1,4,101,512)
    elif bad=='querywidth':args[2].shape=(1,1,99,101)
    elif bad=='maskdtype':args[3].dtype=mx.float32
    elif bad=='valueprecision':args[1].dtype=mx.float16
    else:args[1].shape=(1,50,256)
    with pytest.raises(ValueError):s.stage(*args)


def test_GPU_guard_then_no_backing_mutation_with_mock_kernel(monkeypatch):
    args=fixture()
    with pytest.raises(ValueError,match='caller-owned'):s.stage(*args)
    seen={};outputs=[object(),object(),object()]
    monkeypatch.setattr(mx,'default_device',lambda:mx.gpu);monkeypatch.setattr(mx.metal,'is_available',lambda:True)
    def kernel(**kw):seen.update(kw);return outputs
    monkeypatch.setattr(s,'_kernel',lambda:kernel)
    assert s.stage(*args)is outputs
    assert all(a is b for a,b in zip(seen['inputs'],args))
    assert seen['output_dtypes']==[mx.float16,mx.float16,mx.int8]


@pytest.mark.parametrize('mode,length,result',[('reference',1,'reference'),('reference',99,'reference'),
    ('fused_decode',1,'fused_decode'),('fused_decode',99,'reference')])
def test_optin_explicit_PP_reference_fallback(mode,length,result):assert d.resolved_prepare_mode(mode,length)==result


@pytest.mark.parametrize('mode',[None,True,'fused',''])
def test_bad_mode_refuses(mode):
    with pytest.raises(ValueError):d.resolved_prepare_mode(mode,1)


def test_prepared_buffer_byte_gate_preserves_signedzero_and_padding(helper):
    a=np.zeros((1,32),np.float16);b=a.copy();b[0,3]=-0.
    assert not helper.prepared_bytes(np,a,b)['exact']
    assert helper.prepared_bytes(np,a,a)['exact']
    assert not helper.prepared_bytes(np,a,a.astype(np.float32))['exact']
    assert helper.preparation_mode({})=='reference'
    assert helper.preparation_mode({'dwarfstar_options':{'prepare_mode':'fused_decode'}})=='fused_decode'
    with pytest.raises(ValueError):helper.preparation_mode({'dwarfstar_options':{'prepare_mode':True}})
    source=Path(helper.__file__).read_text()
    assert source.index("section['required_prepared_buffers_exact']=True")<source.index('for name,values in synthetic_stability(np):')
    assert "'preparation_reference_attention_raw'"in source
    assert 'Preparationmode changedoriginalDwarfStarattention output'in source


@pytest.mark.parametrize('local,pool',[(1,1),(101,50),(101,101),(128,512)])
def test_offline_cast_copy_kernel_compile(local,pool,tmp_path):
    if not shutil.which('xcrun'):pytest.skip('OfflineMetal missing')
    path=tmp_path/'stage.metal';path.write_text(s.compilation_source(local,pool))
    result=subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal3.2','-ffast-math','-c',str(path),'-o',str(tmp_path/'stage.air')],capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stdout+result.stderr
