"""AST DOWN source/geometry only; no actual model/GPU qualification."""
from collections import Counter
from types import SimpleNamespace
import mlx.core as mx
import pytest
from mlx2.runtime.gguf import source_ast_down_rows as down
from mlx2.runtime.gguf import source_down_fusion as single


def array(shape,dtype):return SimpleNamespace(shape=shape,ndim=len(shape),dtype=dtype)


def expert(kind):
    if kind=='q1':return {'down.qweight':array((5120,72),mx.uint32),'down.scales':array((5120,36),mx.float16),'down.biases':array((5120,36),mx.float16)}
    return {'down.qweight':array((5120,288),mx.uint32),'down.scales':array((5120,72),mx.uint8)}


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**k:(_ for _ in ()).throw(AssertionError('GPU construction forbidden')))
    yield
    mx.set_default_device(old)


@pytest.mark.parametrize('kind',['original','q1'])
@pytest.mark.parametrize('rows',[1,2,7,12,13,16])
def test_every_occurrence_owns_all_outputs_once_and_reuses_exact_b1_virtual_coordinates(kind,rows):
    p=down.plan(rows,kind)
    assert p['grid']==(1280*32,rows,1)and p['threadgroup']==(32,1,1)
    counts=Counter()
    for occurrence in range(rows):
        for tile in range(1280):
            virtual_tile,virtual_sg=tile//2,tile%2
            for lane in range(4):
                column=virtual_tile*8+virtual_sg*4+lane
                assert column==tile*4+lane
                counts[occurrence,column]+=1
    assert len(counts)==rows*5120 and set(counts.values())=={1}
    x=array((rows,2304),mx.float32);e=expert(kind)
    desc,args=down.prepare_inputs(x,e)
    assert desc==p and args[0]is x and args[1]is e['down.qweight']


@pytest.mark.parametrize('kind',['original','q1'])
def test_only_pointers_and_output_rounding_change_in_pinned_b1_source(kind):
    spec=down.kernel_spec(kind);base=single.kernel_spec((kind,),tile_rows=4)
    assert spec['header']==base['header']
    assert spec['input_names']==base['input_names']
    assert spec['source'].count('occurrence*2304')==1
    assert 'output[occurrence*5120+tile*4+lane]=float(bfloat(partials[lane]));'in spec['source']
    assert 'threadgroup_barrier(mem_flags::mem_threadgroup);'in spec['source']
    assert 'const uint virtual_tile =tile/2;'in spec['source']
    assert 'const uint virtual_sg =tile%2;'in spec['source']
    assert 'total =total+'not in spec['source']
    assert spec['source'].count('{')==spec['source'].count('}')
    assert spec['output_names']==['output']


@pytest.mark.parametrize('rows,kind',[(0,'q1'),(17,'original'),(True,'q1'),(1.0,'original'),(1,'unknown')])
def test_invalid_plan_refused(rows,kind):
    with pytest.raises(ValueError):down.plan(rows,kind)


@pytest.mark.parametrize('mutation',['width','rank','input_dtype','weight_shape','weight_dtype','scale_shape','scale_dtype','bias_dtype','missing'])
def test_bad_projection_refused_before_kernel(mutation):
    x=array((16,2304),mx.float32);e=expert('q1')
    if mutation=='width':x.shape=(16,5120)
    if mutation=='rank':x.ndim=3;x.shape=(1,16,2304)
    if mutation=='input_dtype':x.dtype=mx.bfloat16
    if mutation=='weight_shape':e['down.qweight'].shape=(5120,71)
    if mutation=='weight_dtype':e['down.qweight'].dtype=mx.int32
    if mutation=='scale_shape':e['down.scales'].shape=(5120,35)
    if mutation=='scale_dtype':e['down.scales'].dtype=mx.float32
    if mutation=='bias_dtype':e['down.biases'].dtype=mx.bfloat16
    if mutation=='missing':del e['down.scales']
    with pytest.raises(ValueError):down.down(x,e)


def test_valid_cpu_execution_refuses_without_constructing_metal():
    with pytest.raises(ValueError,match='owned M5 Max'):
        down.down(array((16,2304),mx.float32),expert('original'))
