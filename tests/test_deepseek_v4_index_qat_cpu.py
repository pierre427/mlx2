"""Independent rounding boundaries, identity separation and offline compilation."""
import shutil
import subprocess
import numpy as np
import pytest

from mlx2.runtime.models.deepseek_v4_index_qat import reference_rotated, compilation_source


def test_all_midpoints_even_codes_neighbors_sign_and_scale():
    mid=np.array([.25,.75,1.25,1.75,2.5,3.5,5],dtype=np.float32)
    lower=np.array([0,.5,1,1.5,2,3,4],dtype=np.float32)
    upper=np.array([.5,1,1.5,2,3,4,6],dtype=np.float32)
    even=np.array([0,1,1,2,2,4,4],dtype=np.float32)
    for sign in [1,-1]:
        for exponent in [-80,-10,0,10,80]:
            scale=np.float32(2.**exponent)
            for values,expected in [(np.nextafter(mid,-np.inf),lower),(mid,even),(np.nextafter(mid,np.inf),upper)]:
                row=np.zeros((4,32),dtype=np.float32);row[:,0]=6*scale;row[:,1:8]=values*scale*sign
                actual=reference_rotated(row.reshape(1,128)).reshape(4,32)[:,1:8]
                assert np.array_equal(actual.view(np.uint32),np.broadcast_to((expected*scale*sign).view(np.uint32),(4,7)))


def test_policy_identity_dispatch_and_geometry(monkeypatch):
    import mlx.core as mx
    from mlx2.runtime.models import deepseek_v4_index_qat as candidate
    from mlx2.runtime.models.deepseek_v4 import simulate_index_fp4
    from mlx2.runtime.models.deepseek_v4_identity import native_target_identity
    from mlx2.adapters.deepseek_v4_gguf import split_execution_policy
    with mx.stream(mx.cpu):
        x=mx.ones((1,128),mx.float32);seen=[]
        monkeypatch.setattr(candidate,'index_fp4',lambda a:seen.append(a) or a)
        assert simulate_index_fp4(x,policy='fused_fp32') is x and seen==[x]
        with pytest.raises(ValueError,match='group32'):simulate_index_fp4(x,64,policy='fused_fp32')
        with pytest.raises(ValueError):simulate_index_fp4(x,0)
        with pytest.raises(ValueError):simulate_index_fp4(x,policy='auto')
    reference,_=split_execution_policy({});optimized,_=split_execution_policy({'deepseek_index_qat':'fused_fp32'})
    assert reference['deepseek_index_qat']=='reference'
    assert native_target_identity('checkpoint',reference)!=native_target_identity('checkpoint',optimized)


def test_explicit_mlx_profile_and_metal_source(tmp_path):
    from mlx2.adapters.deepseek_v4_mlx import optimized_policy, DESCRIPTOR
    from mlx2.adapters.deepseek_v4_gguf import split_execution_policy
    p=optimized_policy(tmp_path,external=False)
    assert len(p)==10 and split_execution_policy(p)[0]==p
    assert DESCRIPTOR.model_type=='deepseek4_mlx'
    assert 'draft_model' not in p
    if shutil.which('xcrun') is None:pytest.skip('offline Metal compiler unavailable')
    src=tmp_path/'index.metal';src.write_text(compilation_source())
    r=subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal3.2','-c',str(src),'-o',str(tmp_path/'index.air')],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
