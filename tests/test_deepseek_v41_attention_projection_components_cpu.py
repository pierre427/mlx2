"""Bounded actual-code loading/controls; no GPU/model matrix execution."""
import importlib.util
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest


@pytest.fixture
def helper():
    p=Path(__file__).resolve().parents[1]/'scripts/research/ds41_attention_projection_components.py'
    s=importlib.util.spec_from_file_location('projection_components_cpu',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def artifact():
    return dict(aliases={'blk.2.attn_q_b.weight':'layers.2.attn.wq_b.weight'},tensors={
        'layers.2.attn.wq_b.weight':dict(dtype='F8_E4M3',shape=(64,32)),
        'layers.2.attn.wq_b.scale':dict(dtype='F8_E8M0',shape=(2,1))})


def test_only_requested_weight_and_scale_extents_are_read_and_exactly_expanded(helper):
    a=artifact();calls=[]
    codes=np.arange(64*32,dtype=np.uint8).reshape(64,32);scale=np.array([[127],[128]],np.uint8)
    def read(actual,names):
        calls.append(names);assert actual is a
        return{names[0]:codes,names[1]:scale}
    q,s,r=helper.coefficient_payload(np,a,'blk.2.attn_q_b.weight',read)
    assert calls==[('layers.2.attn.wq_b.weight','layers.2.attn.wq_b.scale')]
    assert q.dtype==np.uint32 and q.shape==(64,8)and q.tobytes()==codes.tobytes()
    assert s.shape==(64,1)and np.all(s[:32]==127)and np.all(s[32:]==128)
    assert r['coefficients']['qweight']['sha256']==hashlib.sha256(codes.tobytes()).hexdigest()
    assert r['source_read_bytes']==codes.nbytes+scale.nbytes
    helper.verify_coefficients(r,{'coefficients':r['coefficients']})
    with pytest.raises(ValueError):helper.verify_coefficients(r,{'coefficients':{}})


@pytest.mark.parametrize('bad',['weightdtype','scaledtype','scalegeometry','width'])
def test_other_weight_formats_or_malformed_geometry_refuse_before_read(helper,bad):
    a=artifact();w,s=a['tensors'].values()
    if bad=='weightdtype':w['dtype']='BF16'
    elif bad=='scaledtype':s['dtype']='F32'
    elif bad=='scalegeometry':s['shape']=(1,1)
    else:w['shape']=(64,31)
    def forbidden(*a):raise AssertionError('Malformed source triggered a read')
    with pytest.raises(ValueError):helper.coefficient_payload(np,a,'blk.2.attn_q_b.weight',forbidden)


@pytest.mark.parametrize('fence_fails',[False,True])
def test_upload_failure_keeps_host_until_fence_and_quarantines_only_if_needed(helper,fence_fails):
    q=np.zeros((2,8),np.uint32);s=np.zeros((2,1),np.uint8);events=[]
    def evaluate(*values):events.append(('eval',values));raise RuntimeError('upload failed')
    def fence():
        events.append(('fence',q,s))
        if fence_fails:raise RuntimeError('fence failed')
    mx=SimpleNamespace(array=lambda x,dtype:x,uint32='u4',uint8='u1',eval=evaluate,synchronize=fence)
    with pytest.raises(RuntimeError,match='upload failed'):helper.upload_coefficients(mx,q,s)
    assert events[-1][0]=='fence'
    assert bool(helper._UPLOAD_QUARANTINE)==fence_fails
    if fence_fails:assert helper._UPLOAD_QUARANTINE[0][0]is q and helper._UPLOAD_QUARANTINE[0][1]is s


def test_component_cpu_gate_precedes_capture_or_artifact_loading(helper,tmp_path):
    mx=SimpleNamespace(default_device=lambda:'cpu',gpu='gpu')
    with pytest.raises(RuntimeError,match='caller-owned'):
        helper.run_components(mx,np,tmp_path,tmp_path,{},None,None)


def test_original_weight_dtype_is_preserved_in_component_source(helper):
    source=Path(helper.__file__).read_text()
    assert 'SourceV41Weights'not in source and 'load_dense'not in source
    assert "mode='mxfp8'"in source and 'mxfp4'not in source
    assert 'notexposed/notclaimed'in source
    assert "compare(mx,np,functions,repeats=repeats)"in source
