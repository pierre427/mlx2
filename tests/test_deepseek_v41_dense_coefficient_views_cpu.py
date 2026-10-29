"""Synthetic CPU buffers and mocked dispatch; no GPU/Metal or model evidence."""
from dataclasses import replace
import gc
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from mlx2.runtime.gguf import source_dense as dense


class CPU:
    bfloat16=np.uint16  # Byte-preserving CPU storage fixture, not BF16 arithmetic.
    def __init__(self):self.evaluations=[]
    def eval(self,*values):self.evaluations.append(tuple(values))


def weight():return np.zeros((8192,4096),np.uint16)


def test_actual_full_woa_shape_first_row_alias_retains_owner_once():
    full=weight();owner=weakref.ref(full);mx=CPU()
    cap=dense.prepare_coefficient_view(full,'wo_a',_mx=mx)
    assert cap.full is full and cap.view.shape==(1,4096)and cap.view.nbytes==8192
    assert full.nbytes==67108864 and np.shares_memory(cap.view,full)
    assert len(mx.evaluations)==1
    assert cap.operand(full,'wo_a')is cap.view
    assert cap.operand(full,'wo_a')is cap.view
    assert len(mx.evaluations)==1
    mx.evaluations.clear();del full;gc.collect()
    assert owner()is not None
    del cap;gc.collect()
    assert owner()is None


@pytest.mark.parametrize('shape,dtype,kind', [((8191,4096),np.uint16,'wo_a'),
    ((8192,4095),np.uint16,'wo_a'),((8192,4096),np.uint8,'wo_a'),
    ((8192,4096),np.uint16,'head'),((8192,4096),np.uint16,'unknown')])
def test_bad_full_geometry_refuses_before_eval(shape,dtype,kind):
    mx=CPU();full=np.broadcast_to(np.zeros(1,dtype),shape)
    with pytest.raises(ValueError):dense.prepare_coefficient_view(full,kind,_mx=mx)
    assert not mx.evaluations


def test_copied_first_row_refuses_even_with_matching_shape():
    class CopyFirst(np.ndarray):
        def __getitem__(self,key):return super().__getitem__(key).copy()
    full=weight().view(CopyFirst);mx=CPU()
    with pytest.raises(ValueError,match='share the full allocation base'):
        dense.prepare_coefficient_view(full,'wo_a',_mx=mx)
    assert len(mx.evaluations)==1


def test_noncontiguous_full_owner_refuses():
    scalar=np.zeros(1,np.uint16)
    full=np.lib.stride_tricks.as_strided(scalar,shape=(8192,4096),strides=(16384,4))
    with pytest.raises(ValueError,match='C contiguous'):
        dense.prepare_coefficient_view(full,'wo_a',_mx=CPU())


def test_live_ownership_checked_before_and_after_prepare():
    mx=CPU();full=weight()
    with pytest.raises(ValueError,match='not live'):
        dense.prepare_coefficient_view(full,'wo_a',ownercheck=lambda:False,_mx=mx)
    assert not mx.evaluations
    checks=iter([True,False])
    with pytest.raises(ValueError,match='expired'):
        dense.prepare_coefficient_view(full,'wo_a',ownercheck=lambda:next(checks),_mx=mx)
    assert len(mx.evaluations)==1


def test_capability_is_owner_kind_and_issuer_bound():
    full=weight();cap=dense.prepare_coefficient_view(full,'wo_a',_mx=CPU())
    with pytest.raises(ValueError):cap.operand(full.copy(),'wo_a')
    with pytest.raises(ValueError):cap.operand(full,'head')
    with pytest.raises(ValueError):replace(cap,_issuer=object()).operand(full,'wo_a')
    with pytest.raises(ValueError):replace(cap,view=cap.view[:,:8]).operand(full,'wo_a')
    with pytest.raises(ValueError):replace(cap,view=cap.view.copy()).operand(full,'wo_a')
    with pytest.raises(Exception):cap.kind='head'


def test_dispatch_geometry_uses_full_owner_before_alias_and_control_remains_full(monkeypatch):
    full=weight();mx=CPU();cap=dense.prepare_coefficient_view(full,'wo_a',_mx=mx)
    seen=[]
    def geometry(x,w,kind,algorithm,tile):
        assert w is full and w.shape==(8192,4096)
        seen.append('full geometry')
        return dict(rows=1,algorithm='gemv',grid=(8192,1,8),threadgroup=(128,1,1),output_shape=(1,1,8,1024))
    def launch(**kwargs):seen.append(kwargs);return ['output']
    monkeypatch.setattr(dense,'_geometry',geometry)
    monkeypatch.setattr(dense,'device_supported',lambda:True)
    monkeypatch.setattr(dense,'_kernel',lambda *args:launch)
    x=object()
    assert dense.project(x,full,'wo_a')=='output'
    assert seen[-1]['inputs'][1]is full
    assert dense.project(x,full,'wo_a',operand_policy='coefficient_views_v1',coefficient_view=cap)=='output'
    assert seen[-2]=='full geometry'and seen[-1]['inputs'][1]is cap.view
    assert seen[-1]['output_shapes']==[(1,1,8,1024)]
    assert len(mx.evaluations)==1  # Hot consumer did not prepare/evaluate again.
    with pytest.raises(ValueError,match='idle prepared'):
        dense.project(x,full,'wo_a',operand_policy='coefficient_views_v1')
    with pytest.raises(ValueError,match='unknown coefficient'):
        dense.project(x,full,'wo_a',operand_policy='implicit')
