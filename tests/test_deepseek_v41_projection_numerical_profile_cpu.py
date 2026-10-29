"""Different numerical-law timing and bounded FP64 oracle; CPU only."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest


@pytest.fixture
def helper(monkeypatch):
    path=Path(__file__).resolve().parents[1]/'scripts/research'
    monkeypatch.syspath_prepend(str(path))
    spec=importlib.util.spec_from_file_location('projection_profile_cpu',path/'ds41_attention_projection_components.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_historical_profile_defaults_stay_exact_only(helper):
    assert helper.projection_options({})==('reference_exact_only',3)
    assert helper.projection_options({'projection_options':{'numerical_profile':'native_bf16_pp_v1'}})==('native_bf16_pp_v1',7)


@pytest.mark.parametrize('options',[{'numerical_profile':'auto'},{'repeats':True},{'repeats':8},{'bogus':1},[]])
def test_bad_options_fail_closed(helper,options):
    with pytest.raises(ValueError):helper.projection_options({'projection_options':options})


def test_different_law_is_actually_timed_without_cross_reference_exact_gate(helper):
    mx=SimpleNamespace(eval=lambda *a:None,synchronize=lambda:None)
    functions={'reference':lambda:np.array([1],np.float32),'native_bf16':lambda:np.array([1.0078125],np.float32)}
    result=helper.profile_timings(mx,np,functions,repeats=3)
    assert all(len(row['samples'])==3 for row in result.values())
    assert result['native_bf16']['median_total_seconds']>=0


@pytest.mark.parametrize('bad',['nonfinite','unstable'])
def test_profile_timer_refuses_nonfinite_or_self_unstable(helper,bad):
    mx=SimpleNamespace(eval=lambda *a:None,synchronize=lambda:None);counter=[0]
    def candidate():
        counter[0]+=1
        return np.array([np.nan if bad=='nonfinite' else counter[0]],np.float32)
    with pytest.raises(ValueError,match='finite repeat-stable'):
        helper.profile_timings(mx,np,{'reference':lambda:np.zeros(1,np.float32),'candidate':candidate},repeats=3)


def test_original_fp8_specials_subnormal_sign_scale_decode(helper):
    codes=np.tile(np.array([0,128,1,0x38,0xb8,0x7e,0xfe,0x40],np.uint8),4).reshape(1,32)
    actual=helper.decode_effective_f32(np,codes,np.array([[128]],np.uint8))
    expected=np.tile(np.array([0,-0.,2**-8,2,-2,896,-896,4],np.float32),4).reshape(1,32)
    assert actual.tobytes()==expected.tobytes()


@pytest.mark.parametrize('code,scale',[(0x7f,127),(0xff,127),(0x38,255)])
def test_nonfinite_codes_scales_refused(helper,code,scale):
    with pytest.raises(ValueError,match='Nonfinite'):
        helper.decode_effective_f32(np,np.full((1,32),code,np.uint8),np.array([[scale]],np.uint8))


def test_blocked_actual_numpy_oracle_and_effective_weight_roundtrip(helper):
    codes=np.full((5,32),0x38,np.uint8);codes[1]=0xb8;codes[2]=0x40
    scales=np.array([[127],[127],[126],[128],[125]],np.uint8)
    x=np.arange(64,dtype=np.float32).reshape(1,2,32)/32
    truth,audit=helper.fp64_reference(np,x,codes.view(np.uint32),scales,block_rows=2)
    weights=np.array([1,-1,1,2,.25],np.float64)
    expected=x.astype(np.float64).sum(axis=-1)[...,None]*weights
    assert np.array_equal(truth,expected)
    assert audit['effective_F32_coefficient_BF16_roundtrip_exact'] and audit['total_words']==160
    assert audit['maximum_decoded_weight_rows']==2 and audit['BLAS_threads']==1
    assert helper.fp64_error(np,truth,truth.astype(np.float32))['rmse']==0


def test_oracle_block_and_shape_guards(helper):
    with pytest.raises(ValueError):helper.fp64_reference(np,np.ones((1,32),np.float32),np.zeros((1,8),np.uint32),np.ones((1,1),np.uint8),block_rows=129)
    with pytest.raises(ValueError):helper.decode_effective_f32(np,np.ones((1,31),np.uint8),np.ones((1,1),np.uint8))


def test_controller_contract_and_no_model_wiring(helper):
    source=Path(helper.__file__).read_text()
    assert "section['complete']=section['harness_completed']=True" in source
    assert "required_reference_controls_exact" in source and "section['verdicts']" in source
    assert "distinct and case['phase']=='pp99'" in source
    assert 'SourceV41Weights' not in source
