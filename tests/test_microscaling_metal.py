"""Owned synthetic SC-MXFP4 Metal arithmetic oracle, not model qualification.

Run only through the bounded CPG/paired-lock component runner.
"""
import os
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('MLX2_RUN_GPU_TESTS') != '1', reason='owned GPU opt-in required')


def _setup():
    import mlx.core as mx
    mx.set_default_device(mx.gpu)
    return mx


@pytest.mark.parametrize('shape', [(200,128,640),(128,192,2560),(257,128,17408)])
@pytest.mark.parametrize('dtype_name', ['bfloat16','float16'])
def test_dense_integer_and_scaled_reference(shape,dtype_name):
    from mlx2.runtime.microscaling import kernels as K
    from mlx2.runtime.microscaling.format import pack_numpy,decode_numpy,to_mxfp4_scales_numpy
    mx=_setup(); rng=np.random.default_rng(sum(shape)); m,n,k=shape
    packed=pack_numpy(rng.standard_normal((n,k))*.02,allow_requantization=True)
    codes,bc,ws=map(mx.array,(packed.codes,packed.bcodes,packed.row_scales))
    # Transposed input exercises the public contiguous-input contract.
    x=mx.array(rng.standard_normal((k,m)).astype(np.float32)).astype(getattr(mx,dtype_name)).T
    aq,asc=K.quantize_activations(x)
    # Independent per-row law (amax/127, rint, clamp); padding rows are zero.
    # Metal fast-math division may differ from NumPy by one code at a tie.
    xv=np.array(x.astype(mx.float32)); amax=np.abs(xv).max(axis=1)
    np.testing.assert_allclose(np.array(asc)[:m],amax/np.float32(127),rtol=1e-6)
    ref_q=np.clip(np.rint(xv/(amax/np.float32(127))[:,None]),-127,127)
    assert np.abs(np.array(aq)[:m].astype(np.int32)-ref_q).max()<=1
    assert not np.array(aq)[m:].any() and not np.array(asc)[m:].any()
    qw=K.unpack_i8(codes,bc)
    assert np.array_equal(np.array(qw),decode_numpy(packed))
    acc=K.int8_mm(aq,qw)
    reference=np.array(aq).astype(np.float64)@decode_numpy(packed).astype(np.float64).T
    assert np.array_equal(np.array(acc),reference)
    expected=(acc.astype(mx.float32)*asc[:,None]*ws[None,:]).astype(x.dtype)[:m]
    actual=K.dense(x,codes,bc,ws)
    assert int(mx.sum(actual!=expected).item())==0
    restored=mx.dequantize(codes,mx.array(to_mxfp4_scales_numpy(packed)),group_size=32,bits=4,mode='mxfp4',dtype=mx.float32)
    assert np.array_equal(np.array(restored),decode_numpy(packed)*packed.row_scales[:,None])


@pytest.mark.parametrize('sorted_layout', [False,True])
@pytest.mark.parametrize('k', [640,2560])
def test_grouped_broadcast_integer_reference(k,sorted_layout):
    from mlx2.runtime.microscaling import kernels as K
    from mlx2.runtime.microscaling.format import pack_numpy,decode_numpy
    mx=_setup(); rng=np.random.default_rng(k); e,n,t,top=8,128,29,3
    packed=pack_numpy(rng.standard_normal((e,n,k))*.02,allow_requantization=True)
    codes,bc,ws=map(mx.array,(packed.codes,packed.bcodes,packed.row_scales))
    # Experts 3 and 7 remain empty. This covers broadcast gate/up and sorted down layouts.
    ids=rng.choice([0,1,2,4,5,6],(t,top)).astype(np.int32)
    if sorted_layout:
        ids=np.sort(ids.reshape(-1))
        x=mx.array(rng.standard_normal((len(ids),1,k)).astype(np.float32)).astype(mx.bfloat16)
    else:
        x=mx.array(rng.standard_normal((t,1,1,k)).astype(np.float32)).astype(mx.bfloat16)
    actual=K.grouped(x,mx.array(ids),codes,bc,ws,sorted_indices=sorted_layout)
    # Quantize each logical input once, then broadcast to expert assignments.
    source_batch=x.shape[:-2]; aq,asc=K.quantize_activations(x.reshape(-1,k),padded_rows=int(np.prod(source_batch)))
    shape=np.broadcast_shapes(source_batch,ids.shape)
    q=np.broadcast_to(np.array(aq).reshape(*source_batch,k),(*shape,k)).reshape(-1,k)
    scale=np.broadcast_to(np.array(asc).reshape(source_batch),shape).reshape(-1)
    wq=decode_numpy(packed); ref=np.stack([q[i].astype(np.float64)@wq[v].astype(np.float64).T for i,v in enumerate(ids.reshape(-1))])
    rs=packed.row_scales[ids.reshape(-1)]
    expected=(mx.array(ref.astype(np.int32)).astype(mx.float32)*mx.array(scale)[:,None]*mx.array(rs)).astype(mx.bfloat16)
    assert int(mx.sum(actual.reshape(-1,n)!=expected).item())==0
