"""CPU descriptor/source/ownership checks; no Metal or model-quality claim."""
from collections import Counter
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.gguf import source_down_fusion as down


def array(shape, dtype):
    # Structural stand-in deliberately cannot be evaluated or copied by MLX.
    return SimpleNamespace(shape=shape,dtype=dtype)


def expert(kind):
    if kind == 'q1':
        return {'down.qweight':array((5120,72),mx.uint32),
                'down.scales':array((5120,36),mx.float16),
                'down.biases':array((5120,36),mx.float16)}
    return {'down.qweight':array((5120,288),mx.uint32),
            'down.scales':array((5120,72),mx.uint8)}


@pytest.mark.parametrize('count',range(1,7))
@pytest.mark.parametrize('tile_rows',(4,8))
def test_virtual_tiles_own_each_stock_output_once(count,tile_rows):
    ids=tuple(reversed(range(count)));kinds=tuple('q1' if i%2 else 'original' for i in ids)
    p=down.plan(ids,kinds,tile_rows=tile_rows)
    assert p['ids']==tuple(range(count))
    assert p['threads']==32*count*(tile_rows//4)
    owned=Counter()
    for tile in range(5120//tile_rows):
        for physical_sg in range(count*(tile_rows//4)):
            slot=physical_sg//(tile_rows//4)
            virtual_tile=tile//2 if tile_rows==4 else tile
            virtual_sg=tile%2 if tile_rows==4 else physical_sg%2
            # Stock fp_qmv_impl and Q1 both own four consecutive outputs
            # per SIMD group. Scheduling must preserve these coordinates.
            for offset in range(4):
                column=virtual_tile*8+virtual_sg*4+offset
                assert tile*tile_rows<=column<(tile+1)*tile_rows
                owned[slot,column]+=1
    assert len(owned)==count*5120 and set(owned.values())=={1}
    assert p['scratch_bytes']==count*tile_rows*4
    assert p['logical_input_bindings']<=24


def test_prepare_orders_inputs_stably_without_copy_or_evaluation():
    ids=[np.uint32(7),np.uint32(2),np.uint32(7)]
    middles=[array((1,2304),mx.float32) for _ in ids]
    experts=[expert('original'),expert('q1'),expert('q1')]
    p,args=down.prepare_inputs(middles,experts,ids)
    assert p['ids']==(2,7,7) and p['order']==(1,0,2)
    assert p['kinds']==('q1','original','q1')
    expected=[]
    for i in (1,0,2):
        expected.extend([middles[i],experts[i]['down.qweight'],experts[i]['down.scales']])
        if 'down.biases' in experts[i]:expected.append(experts[i]['down.biases'])
    assert len(args)==len(expected) and all(a is b for a,b in zip(args,expected))


@pytest.mark.parametrize('ids,kinds,tile_rows',[
    ([],[],4),([0]*7,['q1']*7,4),([0],[],4),([384],['q1'],4),
    ([-1],['original'],4),([True],['q1'],4),([1.0],['q1'],4),
    ([0],['unknown'],4),([0],['original'],16),([0],['original'],True),
])
def test_invalid_plan_refuses(ids,kinds,tile_rows):
    with pytest.raises(ValueError):down.plan(ids,kinds,tile_rows=tile_rows)


@pytest.mark.parametrize('mutation',[
    'rows','width','input_dtype','weight_dtype','weight_shape','scale_dtype','scale_shape','bias_dtype','missing',
])
def test_invalid_projection_refuses_before_kernel(mutation,monkeypatch):
    def forbidden(*a,**k):raise AssertionError('Metal construction reached')
    monkeypatch.setattr(down,'_kernel',forbidden)
    x=array((1,2304),mx.float32);e=expert('q1')
    if mutation=='rows':x.shape=(2,2304)
    if mutation=='width':x.shape=(1,5120)
    if mutation=='input_dtype':x.dtype=mx.bfloat16
    if mutation=='weight_dtype':e['down.qweight'].dtype=mx.int32
    if mutation=='weight_shape':e['down.qweight'].shape=(5120,71)
    if mutation=='scale_dtype':e['down.scales'].dtype=mx.float32
    if mutation=='scale_shape':e['down.scales'].shape=(5120,35)
    if mutation=='bias_dtype':e['down.biases'].dtype=mx.bfloat16
    if mutation=='missing':del e['down.scales']
    with pytest.raises(ValueError):down.down_combine([x],[e],[0])


def test_missing_inputs_and_invalid_diagnostic_flag_refuse():
    x=array((1,2304),mx.float32);e=expert('original')
    with pytest.raises(ValueError,match='count'):down.prepare_inputs([], [e], [1])
    with pytest.raises(ValueError,match='boolean'):down.down_combine([x],[e],[1],return_contributions=1)


def test_source_has_probe_only_partials_and_fixed_ordered_rounding():
    plain=down.kernel_spec(('q1','original'),return_contributions=False)
    probe=down.kernel_spec(('q1','original'),return_contributions=True)
    assert plain['output_names']==['output']
    assert probe['output_names']==['output','contributions']
    assert 'contributions[' not in plain['source']
    assert 'volatile float total' in plain['source']
    assert 'total =total+float(bfloat(raw));' in plain['source']
    assert 'threadgroup_barrier' in plain['source']
    assert len(plain['input_names'])==7


def test_original_helper_extraction_keeps_namespace_balanced():
    source=down._original_header()
    assert source.count('{')==source.count('}')
    assert source.count('METAL_FUNC void fp_qmv_to_shared_impl(')==1
    assert '    threadgroup T* y,' in source
    assert '    y += tid.x * out_vec_size' not in source


def test_changed_installed_header_refuses_cpu_source_generation(monkeypatch,tmp_path):
    from mlx2.runtime.gguf import source_mxfp4_pair as installed
    path=tmp_path/'include/mlx/backend/metal/kernels';path.mkdir(parents=True)
    (path/'fp_quantized.h').write_text('changed source')
    down._original_header.cache_clear()
    monkeypatch.setattr(installed,'_verify_library',lambda:None)
    monkeypatch.setattr(installed,'installed_root',lambda:tmp_path)
    try:
        with pytest.raises(RuntimeError,match='helper changed'):
            down.kernel_spec(('original',))
    finally:
        down._original_header.cache_clear()
