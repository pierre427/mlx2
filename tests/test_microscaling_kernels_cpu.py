"""SC GPU wrapper contracts exercised through independent CPU dispatch mirrors.

No checkpoint or Metal execution: synthetic inputs at current source only.
Actual compiled kernel parity needs the separately owned M5 component probe.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.microscaling import kernels as k
from mlx2.runtime.microscaling.format import PackedSCWeight, decode_numpy, pack_numpy


@pytest.fixture(autouse=True)
def cpu_device():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(old)


def _decode(codes, bc):
    shape = (*codes.shape[:-1], codes.shape[-1] * 8)
    packed = PackedSCWeight(np.array(codes), np.array(bc), np.ones(shape[:-1], np.float32), shape)
    return decode_numpy(packed)


@pytest.fixture
def mirror(monkeypatch):
    seen = []

    def dispatch(name, inputs, constants, shapes, dtypes, grid, threadgroup, dtype_name=""):
        seen.append((name, constants, shapes, dtypes))
        if name == 'quantize':
            a = np.array(inputs[0].astype(mx.float32))
            scale = np.max(np.abs(a), axis=1).astype(np.float32) / np.float32(127)
            inv = np.divide(np.float32(1), scale, out=np.zeros_like(scale), where=scale > 0)
            codes = np.rint(a * inv[:, None]).clip(-127, 127).astype(np.int8)
            padded = shapes[0][0] - len(a)
            return [mx.array(np.pad(codes, ((0, padded), (0, 0)))), mx.array(np.pad(scale, (0, padded)))]
        if name == 'unpack':
            return [mx.array(_decode(*inputs))]
        if name in ('int8_mm', 'scaled_mm'):
            a, w = map(np.array, inputs[:2])
            acc = a.astype(np.int64) @ w.astype(np.int64).T
            assert np.abs(acc).max() < 2**31
            if name == 'int8_mm':
                return [mx.array(acc.astype(np.int32))]
            xs, ws = map(np.array, inputs[2:])
            return [mx.array(acc.astype(np.float32) * xs[:, None] * ws[None, :]).astype(dtypes[0])]
        if name == 'segments':
            idx = np.array(inputs[0])
            return [mx.array(np.bincount(idx, minlength=constants['E']).astype(np.int32))]
        if name == 'grouped_mm':
            a = np.array(inputs[0]); w = _decode(inputs[1], inputs[2])
            ws, xs, off = map(np.array, inputs[3:6])
            acc = np.zeros((constants['R'], constants['N']), np.int32)
            expected = np.zeros(acc.shape, np.float32)
            for e in range(constants['E']):
                lo, hi = int(off[e]), int(off[e + 1])
                acc[lo:hi] = a[lo:hi].astype(np.int64) @ w[e].astype(np.int64).T
                if constants['SCALED']:
                    expected[lo:hi] = acc[lo:hi].astype(np.float32) * xs[lo:hi, None] * ws[e][None]
            return [mx.array(expected).astype(dtypes[0]) if constants['SCALED'] else mx.array(acc)]
        raise AssertionError(name)

    monkeypatch.setattr(k, '_dispatch', dispatch)
    return seen


def _weight(shape):
    rng = np.random.default_rng(16)
    packed = pack_numpy(rng.standard_normal(shape).astype(np.float32) * .02, allow_requantization=True)
    return packed, [mx.array(packed.codes), mx.array(packed.bcodes), mx.array(packed.row_scales)]


def test_import_and_geometry_never_load_mlx_or_compile():
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    code = '''import sys
from mlx2.runtime.microscaling import kernels
assert 'mlx.core' not in sys.modules
assert kernels.validate_dense_geometry((2,3,128),(64,16),(64,1),(64,)) == (6,64,128)
'''
    proc = subprocess.run([sys.executable, '-c', code], env=env, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr


def test_geometry_broadcast_and_offset_guards():
    assert k.validate_grouped_geometry((2,3,1,1,128),(2,3,4),(8,64,16),(8,64,1),(8,64)) == ((2,3,4),8,24,64,128)
    assert k.validate_grouped_geometry((24,1,128),(24,),(8,64,16),(8,64,1),(8,64)) == ((24,),8,24,64,128)
    for args in [((1,128),(63,16),(63,1),(63,)), ((1,128),(64,17),(64,1),(64,)),
                 ((12200,176128),(64,22016),(64,1376),(64,)), ((1,128),(64,16),(64,2),(64,))]:
        with pytest.raises(ValueError): k.validate_dense_geometry(*args)
    with pytest.raises(ValueError, match='broadcast'):
        k.validate_grouped_geometry((2,3,1,128),(4,),(8,64,16),(8,64,1),(8,64))
    with pytest.raises(ValueError, match='at most'):
        k.validate_grouped_geometry((2,1,128),(2,),(2049,64,16),(2049,64,1),(2049,64))
    with pytest.raises(ValueError, match='shape'):
        k.validate_grouped_geometry((2,2,128),(2,),(8,64,16),(8,64,1),(8,64))


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_dense_padding_rounding_and_no_persistent_weight_copy(mirror,dtype):
    packed, arrays = _weight((64,128))
    rng = np.random.default_rng(8)
    x = mx.array(rng.standard_normal((2,3,128)).astype(np.float32)).astype(dtype)
    got = k.dense(x,*arrays)
    assert got.shape == (2,3,64)
    aq, xs = k.quantize_activations(x.reshape(6,128))
    q = decode_numpy(packed)
    acc = np.array(aq).astype(np.int64) @ q.astype(np.int64).T
    ref = mx.array(acc.astype(np.float32) * np.array(xs)[:,None] * packed.row_scales[None]).astype(dtype)[:6].reshape(2,3,64)
    assert np.array_equal(np.array(got.astype(mx.float32)), np.array(ref.astype(mx.float32)))
    assert [name for name,*_ in mirror[:3]] == ['quantize','unpack','scaled_mm']
    assert not np.any(np.array(aq)[6:])


@pytest.mark.parametrize('sorted_indices', [False,True])
def test_grouped_broadcast_per_assignment_and_empty_experts(mirror,sorted_indices):
    packed, arrays = _weight((4,64,128))
    rng = np.random.default_rng(9)
    if sorted_indices:
        ids = np.array([0,0,2,2,2],np.int32)
        x = mx.array(rng.standard_normal((5,1,128)).astype(np.float32)).astype(mx.bfloat16)
    else:
        ids = np.array([[[2,0],[0,2],[2,2]]],np.int32)
        x = mx.array(rng.standard_normal((1,3,1,1,128)).astype(np.float32)).astype(mx.bfloat16)
    indices = mx.array(ids)
    got = k.grouped(x,indices,*arrays,sorted_indices=sorted_indices)
    batch = np.broadcast_shapes(x.shape[:-2],ids.shape)
    xb = mx.broadcast_to(x,(*batch,1,128)).reshape(-1,128)
    aq,xs=k.quantize_activations(xb,padded_rows=xb.shape[0])
    weights=decode_numpy(packed); expected=[]
    for row,expert in enumerate(np.broadcast_to(ids,batch).reshape(-1)):
        acc=np.array(aq[row]).astype(np.int64)@weights[expert].astype(np.int64).T
        expected.append(acc.astype(np.float32)*np.array(xs)[row]*packed.row_scales[expert])
    ref=mx.array(np.array(expected)).astype(mx.bfloat16).reshape(*batch,1,64)
    assert np.array_equal(np.array(got.astype(mx.float32)),np.array(ref.astype(mx.float32)))
    assert not any(name=='unpack' for name,*_ in mirror)
    # Down projection receives one activation row for every assignment.
    got2=k.grouped(mx.broadcast_to(x,(*batch,1,128)),indices,*arrays,sorted_indices=sorted_indices)
    assert np.array_equal(np.array(got.astype(mx.float32)),np.array(got2.astype(mx.float32)))


def test_invalid_expert_assignment_cannot_read_foreign_memory(mirror):
    _, arrays = _weight((4,64,128))
    x=mx.ones((3,1,128),mx.bfloat16)
    got=k.grouped(x,mx.array([-1,0,4],mx.int32),*arrays)
    result=np.array(got.astype(mx.float32))
    assert np.isnan(result[[0,2]]).all()
    assert np.isfinite(result[1]).all()


def test_wrong_runtime_types_fail_before_dispatch(mirror):
    _, arrays=_weight((64,128))
    with pytest.raises(ValueError,match='BF16'):
        k.dense(mx.ones((2,128),mx.float32),*arrays)
    with pytest.raises(ValueError,match='uint32'):
        k.dense(mx.ones((2,128),mx.bfloat16),arrays[0].astype(mx.int32),*arrays[1:])
    with pytest.raises(ValueError,match='padded'):
        k.int8_mm(mx.zeros((1,128),mx.int8),mx.zeros((64,128),mx.int8))
    assert mirror == []
