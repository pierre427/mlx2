"""Head inventory/refusal fixtures never read actual coefficients or use GPU."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def module():
    path=Path(__file__).resolve().parents[1]/'scripts/research/ds41_plan_head_components.py'
    spec=importlib.util.spec_from_file_location('_head_components_cpu',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def artifact(m):
    return {'identity':{'fingerprint':m.FINGERPRINT},'root':m.MODEL_PATH,
        'aliases':{'output.weight':'head.weight'},
        'tensors':{'head.weight':{'dtype':'BF16','shape':m.SHAPE,'size':1323827200,'file':'head.safetensors'}},
        'seals':{'head.safetensors':{'size':1323827216}}}


def test_full_head_geometry_and_diagnostic_companion_are_explicit():
    p=module().plan()
    assert p['weight_shape']==(129280,5120)
    assert p['reference_f32_bytes']==2647654400 and p['direct_bf16_bytes']==1323827200
    assert p['diagnostic_retained_coefficient_bytes']==3971481600 < 6<<30
    assert p['production_companion_bytes']==0 and p['diagnostic_companion_bytes']==1323827200
    assert p['case_count']==8 and p['synthetic_activations'] and not p['model_loaded']


@pytest.mark.parametrize('kind',['fingerprint','shape','dtype','size','alias','path'])
def test_wrong_actual_artifact_metadata_refuses_before_payload_read(kind):
    m=module();a=artifact(m)
    if kind=='fingerprint':a['identity']['fingerprint']='0'*64
    elif kind=='shape':a['tensors']['head.weight']['shape']=(129279,5120)
    elif kind=='dtype':a['tensors']['head.weight']['dtype']='F32'
    elif kind=='size':a['tensors']['head.weight']['size']-=2
    elif kind=='alias':a['aliases']['output.weight']='other'
    else:a['root']=Path('/different')
    with pytest.raises(ValueError):m.validate_artifact(a)


def test_correct_original_binding_and_extent_admit_metadata_only():
    m=module();name,row=m.validate_artifact(artifact(m))
    assert name=='head.weight' and row['shape']==m.SHAPE


@pytest.mark.parametrize('checker',[None,False,1])
def test_missing_owner_refuses_before_import_load_or_compile(checker):
    m=module();backend=SimpleNamespace(gpu='gpu',default_device=lambda:'gpu')
    with pytest.raises(ValueError,match='callback'):m.run(backend,checker)


def test_CPU_backend_and_failed_live_guard_refuse_before_load():
    m=module()
    with pytest.raises(ValueError,match='GPU backend'):
        m.run(SimpleNamespace(gpu='gpu',default_device=lambda:'cpu'),lambda:True)
    with pytest.raises(RuntimeError,match='return True'):
        m.run(SimpleNamespace(gpu='gpu',default_device=lambda:'gpu'),lambda:None)


@pytest.mark.parametrize('cell',('normal','alternating_cancellation','signed_zero_tiny','sparse_spikes'))
def test_synthetic_activations_are_deterministic_finite_and_full_width(cell):
    m=module();a=m._activation(cell,4);b=m._activation(cell,4)
    assert a.shape==(1,1,5120) and a.dtype==np.float32
    assert np.all(np.isfinite(a)) and a.tobytes()==b.tobytes()


def test_output_classification_and_F64_sample_are_numerical_diagnostics():
    m=module();x=np.array([[[1.,2.]]],np.float32);weights=np.array([[1.,0.],[0.,2.]],np.float32)
    a=np.array([[[1.,4.]]],np.float32);b=a.copy()
    result=m._compare(a,b,x,weights,(0,1))
    assert result['classification']=='raw_exact_sample'
    assert result['sampled_f64']['errors']['candidate']['maximum_absolute_error']==0
    b[0,0,1]=np.nextafter(np.float32(4),np.float32(5))
    assert m._compare(a,b,x,weights,(0,1))['classification']=='distinct_numerical_profile'
