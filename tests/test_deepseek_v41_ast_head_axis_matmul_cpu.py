"""Explicit CPU operand contracts; no inference, shader compile or GPU work."""
import numpy as np
import pytest
import mlx.core as mx
from mlx2.runtime.gguf import source_ast_head_axis_matmul as candidate


@pytest.mark.parametrize('padded',[True,False])
@pytest.mark.parametrize('transpose_b',[True,False])
def test_actual_cpu_views_preserve_explicit_M1_geometry_and_values(padded,transpose_b,monkeypatch):
    previous=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda*a,**kw:pytest.fail('GPU compile forbidden'))
    try:
        left=np.arange(3*64*11,dtype=np.float32).reshape(3,64,11)/2048
        right=np.arange(3*11*7,dtype=np.float32).reshape(3,11,7)/2048
        calls=[];original=mx.as_strided
        def record(array,shape,strides):
            calls.append((shape,strides));return original(array,shape=shape,strides=strides)
        monkeypatch.setattr(mx,'as_strided',record)
        a,b=candidate.prepare_operands(mx,mx.array(left),mx.array(right),transpose_b=transpose_b,padded=padded)
        assert a.shape==(3,64,1,11)and b.shape==(3,64,11,7)
        mx.eval(a,b)
        np.testing.assert_array_equal(np.asarray(a),left[:,:,None])
        np.testing.assert_array_equal(np.asarray(b),np.broadcast_to(right[:,None],(3,64,11,7)))
        assert calls==([((3,64,1,11),(768,12,11,1))]if padded else[])
        # Raw CPU buffer strides confirm a zero head stride; compact NT vs NN
        # is observable without looking up any GPU memory address.
        assert np.asarray(b).strides[1]==0
        if transpose_b:assert np.asarray(b).strides[-2:]==(4,44)
        else:assert np.asarray(b).strides[-2:]==(28,4)
        assert np.asarray(a).strides[1]==(48 if padded else 44)
    finally:mx.set_default_device(previous)


def test_cpu_refusal_and_shape_contract_fail_closed(monkeypatch):
    previous=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda*a,**kw:pytest.fail('GPU compile forbidden'))
    try:
        with pytest.raises(ValueError,match='parent-owned GPU'):
            candidate.private_matmul(None,None,transpose_b=True,check_live=lambda:True)
        with pytest.raises(ValueError,match='live ownership'):
            candidate.private_matmul(None,None,transpose_b=True,check_live=None)
        for shape in[(17,64,512,11),(3,32,512,11),(3,64,0,11),(3,64,512,513)]:
            with pytest.raises(ValueError):candidate.plan(*shape,transpose_b=True)
    finally:mx.set_default_device(previous)
