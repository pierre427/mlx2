"""HC synthetic CPU equation validation; kernels separately GPU-gated."""
import numpy as np
import pytest
from mlx2.runtime.models.deepseek_v4_hc import reference_split, compilation_source


@pytest.mark.parametrize('magnitude',[.1,10.,1000.])
def test_independent_hc_coefficients_match_graph(magnitude):
    import mlx.core as mx
    from mlx2.runtime.models.deepseek_v4 import sinkhorn_mix
    old=mx.default_device(); mx.set_default_device(mx.cpu)
    try:
        rng=np.random.default_rng(3)
        mixes=rng.normal(size=(1,9,24)).astype(np.float32)*magnitude
        scale=np.array([.7,.3,.9],np.float32);base=rng.normal(size=24).astype(np.float32)
        expected=reference_split(mixes,scale,base)
        actual=sinkhorn_mix(mx.array(mixes),mx.array(scale),mx.array(base),4,20,1e-6)
        for a,b in zip(actual,expected):
            np.testing.assert_allclose(np.array(a),b,atol=3e-6,rtol=3e-5)
        for row in range(9):
            serial=reference_split(mixes[:,row:row+1],scale,base)
            for a,b in zip(serial,expected):
                np.testing.assert_array_equal(a,b[:,row:row+1])
    finally: mx.set_default_device(old)


def test_epsilon_is_present_in_every_sinkhorn_denominator():
    pre,post,comb=reference_split(np.zeros((1,24)),np.ones(3),np.zeros(24),eps=.1)
    # Uniform rows remain symmetric, with epsilon reducing their sum below1.
    assert np.all(pre == .6) and np.all(post == 1)
    np.testing.assert_allclose(comb,.225,atol=1e-15)
    assert np.all(comb.sum(-1)<1)


def test_cpu_execution_is_refused_and_compilation_has_all_primitives():
    import mlx.core as mx
    from mlx2.runtime.models.deepseek_v4_hc import split_sinkhorn
    old=mx.default_device();mx.set_default_device(mx.cpu)
    try:
        with pytest.raises(ValueError,match='GPU'):
            split_sinkhorn(mx.zeros((1,24)),mx.ones(3),mx.zeros(24))
    finally: mx.set_default_device(old)
    source=compilation_source()
    assert all('kernel void check_'+name in source for name in ('split','sum','expand','head'))
