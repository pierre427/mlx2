"""Source-only kernel checks and independent CPU lane oracles; GPU pending."""
import hashlib
import numpy as np
import pytest

from mlx2.runtime.gguf import source_q1_few_rows as few
from mlx2.runtime.gguf.source_q1 import SOURCE


@pytest.mark.parametrize('rows',range(2,8))
@pytest.mark.parametrize('mapped',[False,True])
def test_source_keeps_K_and_affine_order_but_hoists_coefficient_loads(rows,mapped):
    spec=few.kernel_spec(rows,mapped=mapped)
    text=spec['source']
    assert f'float accumulators[{rows}][4]' in text
    assert 'block_start=lane*VALUES_PER_THREAD' in text
    assert 'block_start+=BLOCK_SIZE' in text
    assert 'constexpr uint VALUES_PER_THREAD=16' in text
    assert 'constexpr uint BLOCK_SIZE=VALUES_PER_THREAD*32' in text
    weight_load=text.index('packed_words[row]=ushort(weight[')
    input_rows=text.index(f'for(uint v=0;v<{rows};++v)')
    assert weight_load<input_rows
    assert text.count('ushort(weight[')==1
    assert text.index('accumulators[v][row]+=selected_sum*scale_values[row]') < text.index('accumulators[v][row]+=total_sum*bias_values[row]')
    assert 'simd_sum(accumulators[v][row])' in text
    assert 'if(output_row<output_dims)' in text
    assert 'if(source_row>=x_shape[0]) continue;' in text
    assert ('rows[v]' in text)==mapped
    assert spec['requires_raw_gpu_parity'] is True
    assert hashlib.sha256(SOURCE.encode()).hexdigest()==few.PINNED_DOT_SHA256


@pytest.mark.parametrize('rows',range(2,8))
@pytest.mark.parametrize('n,k',[(1,64),(7,576),(9,128)])
def test_reuse_schedule_exact_at_every_lane_with_ragged_output_columns(rows,n,k):
    rng=np.random.default_rng(625+rows+n+k)
    x=rng.normal(size=(rows+2,k)).astype(np.float32)
    # Tiny normal values and signed zero exercise the direct-load law.
    x[0,:4]=np.array([2**-126,-2**-126,0.,-0.],np.float32)
    q=rng.integers(0,2**32,size=(n,k//32),dtype=np.uint32)
    scales=rng.normal(size=(n,k//64)).astype(np.float16)
    biases=rng.normal(size=scales.shape).astype(np.float16)
    selected=np.array(list(range(rows-1,-1,-1)),np.uint32)
    ref=few.lane_oracle(x,q,scales,biases,rows=selected,reuse=False)
    actual=few.lane_oracle(x,q,scales,biases,rows=selected,reuse=True)
    np.testing.assert_array_equal(ref.view(np.uint32),actual.view(np.uint32))
    compact=few.lane_oracle(x[selected],q,scales,biases,reuse=True)
    np.testing.assert_array_equal(ref.view(np.uint32),compact.view(np.uint32))
    assert np.all(np.isfinite(ref))


def test_known_affine_bits_have_independent_hand_oracle():
    x=np.arange(1,65,dtype=np.float32)[None,:]
    q=np.array([[0xffffffff,0]],np.uint32)
    s=np.array([[2]],np.float16);b=np.array([[-1]],np.float16)
    actual=few.lane_oracle(x,q,s,b,reuse=True)
    # Lanes0/1 own chosen bits, lanes2/3 only the affine bias.
    assert actual[0,0,:4].tolist()==[sum(range(1,17)),sum(range(17,33)),
                                           -sum(range(33,49)),-sum(range(49,65))]
    assert np.all(actual[0,0,4:]==0)


@pytest.mark.parametrize('projection',('gate','up','down'))
@pytest.mark.parametrize('rows',range(2,8))
def test_strict_group64_production_geometry_and_separate_M1(projection,rows):
    p=few.plan(projection,rows)
    k,n=few.SHAPES[projection]
    assert p['weight_shape']==(n,k//32) and p['parameter_shape']==(n,k//64)
    assert p['grid'][1]==1 and p['output_shape']==(rows,n)
    assert p['coefficient_load_row_reuse']==rows
    assert p['qualification']=='unqualified'
    assert p['coefficient_repack_bytes']==0


@pytest.mark.parametrize('rows',[0,1,8,True,2.5])
def test_M1_and_unsupported_width_refused(rows):
    with pytest.raises(ValueError):few.plan('gate',rows)
    with pytest.raises(ValueError):few.kernel_spec(rows)


@pytest.mark.parametrize('projection',[None,[],{},'shared','router'])
def test_bad_projection_refused(projection):
    with pytest.raises(ValueError):few.plan(projection,2)


def test_changed_pinned_dot_refuses_source_generation(monkeypatch):
    monkeypatch.setattr(few,'SOURCE',SOURCE+'\n// changed')
    with pytest.raises(RuntimeError,match='requires review'):few.kernel_spec(2)


def test_device_probe_refuses_CPU_before_kernel_creation(monkeypatch):
    import mlx.core as mx
    assert mx.default_device()==mx.cpu
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**k:pytest.fail('GPU compilation forbidden'))
    # A legal actual geometry remains small enough for an ordinary CPU fixture.
    x=mx.zeros((2,2304),mx.float32);q=mx.zeros((5120,72),mx.uint32)
    scales=mx.zeros((5120,36),mx.float16);bias=mx.zeros_like(scales)
    with pytest.raises(ValueError,match='GPU approval'):
        few.project(x,q,scales,bias,projection='down')


@pytest.mark.parametrize('mapping',[np.array([0,2],np.uint32),np.array([0,1],np.int32),
                                  np.array([0],np.uint32)])
def test_invalid_cpu_map_refuses_without_readback_or_compile(mapping,monkeypatch):
    import mlx.core as mx
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**k:pytest.fail('GPU forbidden'))
    x=mx.zeros((2,2304),mx.float32);q=mx.zeros((5120,72),mx.uint32)
    s=mx.zeros((5120,36),mx.float16)
    with pytest.raises(ValueError,match='bounded CPU'):
        few.project(x,q,s,s,projection='down',rows=mapping)
