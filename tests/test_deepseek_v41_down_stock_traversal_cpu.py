"""Stock-axis worklist traversal is an address-only bounded ablation."""
from itertools import product
from types import SimpleNamespace as NS
import pytest

from mlx2.runtime.gguf import source_down_worklist as down
from mlx2.runtime.gguf import source_mxfp4_pair as pair
from test_deepseek_v41_source_down_worklist_cpu import fake_mx,packed,pinned_installed_source


def virtual_jobs(d):
    gx,gy,gz=d['grid'];gx//=64
    if d['traversal']=='stock':
        return {(z,x,y)for x,y,z in product(range(gx),range(gy),range(gz))}
    return {(y//d['vector_groups'],y%d['vector_groups'],x)
            for x,y,z in product(range(gx),range(gy),range(gz))}


@pytest.mark.parametrize('rows',range(1,13))
@pytest.mark.parametrize('experts',(1,3,15))
def test_stock_grid_is_bijection_to_identical_expert_vector_tile_work(rows,experts):
    a=down.plan('original',(rows,)*experts)
    b=down.plan('original',(rows,)*experts,traversal='stock')
    assert a['traversal']=='flat'and b['traversal']=='stock'
    assert virtual_jobs(a)==virtual_jobs(b)
    assert len(virtual_jobs(b))==experts*b['vector_groups']*(5120//b['tile'])
    seen=[]
    for slot in range(experts):
        for group in range(b['vector_groups']):
            for v in range(b['vecs']):
                if group*b['vecs']+v<rows:seen.append(slot*rows+group*b['vecs']+v)
    assert sorted(seen)==list(range(rows*experts))
    assert a['threadgroup']==b['threadgroup']==(64,1,1)
    assert a['output_bytes']==b['output_bytes']and a['scratch_bytes']==b['scratch_bytes']==0


@pytest.mark.parametrize('rows',(1,2,6,12))
@pytest.mark.parametrize('raw',(False,True))
def test_only_virtual_address_prefix_changes_not_math_or_binding_contract(pinned_installed_source,rows,raw):
    a=down.kernel_spec('original',rows,15,return_raw=raw)
    b=down.kernel_spec('original',rows,15,return_raw=raw,traversal='stock')
    assert a['header']==b['header']
    assert a['source'][a['source'].index('const device uchar* backing'):]==b['source'][b['source'].index('const device uchar* backing'):]
    assert a['input_names']==b['input_names']and a['output_names']==b['output_names']
    assert 'tile=threadgroup_position_in_grid.y'in b['source']
    assert 'slot=threadgroup_position_in_grid.z'in b['source']
    assert 'vector_group=threadgroup_position_in_grid.x'in b['source']


@pytest.mark.parametrize('kind,traversal',(('q1','stock'),('original','unknown'),('original',True),('original',None)))
def test_invalid_traversal_refused_before_any_shader_or_metadata_upload(kind,traversal):
    with pytest.raises(ValueError,match='traversal'):down.plan(kind,(1,),traversal=traversal)


def test_runtime_stock_key_and_dispatch_borrow_same_inputs(fake_mx,monkeypatch):
    experts=tuple(packed('original',fake_mx)for _ in range(3))
    x=NS(shape=(18,2304),dtype=fake_mx.float32);keys=[];calls=[]
    def factory(*args):
        keys.append(args)
        def call(**kw):calls.append(kw);return [NS(shape=(18,5120))]
        return call
    monkeypatch.setattr(down,'_kernel',factory);monkeypatch.setattr(pair,'_device_supported',lambda:True)
    down.pp_down(x,experts,(6,6,6),traversal='stock')
    assert keys==[('original',6,3,False,'canonical','stock')]
    assert calls[0]['grid']==(128,1280,3)
    assert calls[0]['inputs'][0]is x
    assert calls[0]['inputs'][1:]==[e.backing for e in experts]
    assert fake_mx.normalized==[x]and not fake_mx.uploaded


def test_default_factory_normalizes_flat_into_explicit_cache_key(monkeypatch):
    keys=[];monkeypatch.setattr(down,'_kernel_for_traversal',lambda *args:keys.append(args))
    down._kernel('original',6,3,False,'canonical')
    down._kernel('original',6,3,False,'canonical','stock')
    assert [key[-1]for key in keys]==['flat','stock']
