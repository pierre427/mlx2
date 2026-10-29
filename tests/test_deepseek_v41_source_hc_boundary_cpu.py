"""Source HC boundary geometry/frontier contracts; no GPU parity claim."""
from pathlib import Path
from types import SimpleNamespace
import ast
import numpy as np
import mlx.core as mx
import pytest
from mlx2.runtime.models import deepseek_v41_source_hc_boundary as hc
from mlx2.runtime.models import deepseek_v41_quant as quant


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('GPU forbidden')))
    yield
    mx.set_default_device(old)


def inputs(rows=1):
    return (mx.ones((1,rows,5120),mx.float32),mx.ones((1,rows,4,5120),mx.float32),
            mx.ones((1,rows,4),mx.float32),mx.broadcast_to(mx.eye(4,dtype=mx.float32),(1,rows,4,4)),
            mx.full((1,rows,4),.25,mx.float32),mx.ones((5120,),mx.float32))


@pytest.mark.parametrize('rows',[1,99,128])
@pytest.mark.parametrize('policy',hc.COLLAPSE_POLICIES)
def test_bounded_geometry_and_frontier_payloads(rows,policy):
    p=hc.plan(*inputs(rows),collapse_policy=policy)
    assert p['rows']==rows and p['mixed_residual_bytes']==rows*4*5120*4
    assert p['stock_sum_frontier']is(policy=='stock_frontier')
    assert p['normalized_router_bytes']==p['prepared_fp8_represented_bytes']==rows*5120*4
    assert sum(v for k,v in p.items()if k.endswith('_bytes'))<64<<20


@pytest.mark.parametrize('kind',['rows0','rows129','batch2','width','rank','residual','post','pre','comb','norm','dtype','normdtype','notarray'])
def test_invalid_geometry_refused_before_device(monkeypatch,kind):
    a=list(inputs())
    if kind=='rows0':a=list(inputs(0))
    elif kind=='rows129':a=list(inputs(129))
    elif kind=='batch2':a[0]=mx.ones((2,1,5120))
    elif kind=='width':a[0]=a[0][...,:-1]
    elif kind=='rank':a[0]=a[0][0]
    elif kind=='residual':a[1]=a[1][:,:,:3]
    elif kind=='post':a[2]=a[2][...,:3]
    elif kind=='pre':a[4]=a[4][...,:3]
    elif kind=='comb':a[3]=a[3][...,:3]
    elif kind=='norm':a[5]=a[5][:-1]
    elif kind=='dtype':a[0]=a[0].astype(mx.bfloat16)
    elif kind=='normdtype':a[5]=a[5].astype(mx.float16)
    else:a[0]=np.ones((1,1,5120),np.float32)
    monkeypatch.setattr(hc,'_device',lambda:(_ for _ in ()).throw(AssertionError('invalid shape reached device')))
    with pytest.raises(ValueError):hc.boundary(*a)


@pytest.mark.parametrize('eps',[0,-1,float('nan'),float('inf'),True,'1e-6'])
def test_invalid_epsilon(eps):
    with pytest.raises(ValueError,match='epsilon'):hc.plan(*inputs(),eps=eps)


@pytest.mark.parametrize('flag',[None,1,'true'])
def test_diagnostic_flag_is_boolean(flag):
    with pytest.raises(ValueError):hc.plan(*inputs(),return_frontiers=flag)


def test_source_body_preserves_quantization_and_no_contraction():
    source=hc.compilation_source('norm_qat')
    assert source['source']==quant.FUSED_SOURCE.replace('const float v=value[i];','const float v=bf16_value(value[i]);\nrouter[i]=v;')
    assert source['outputs']==['router','out']
    a=hc.compilation_source('residual')
    assert '#pragma clang fp contract(off)'in a['header']
    assert 'const float contribution=post[row*4+h]*y;'in a['source']
    assert 'const float raw=contribution+mixed[offset];'in a['source']
    assert 'weighted[h]=stored*pre[row*4+h];'in a['source']
    assert a['outputs']==['updated_h','products']and'collapsed[i]'not in a['source']
    assert 'expanded_raw'in hc.compilation_source('residual',return_frontiers=True)['outputs']


@pytest.mark.parametrize('policy',hc.COLLAPSE_POLICIES)
def test_bindings_and_optional_inline_are_explicit(policy):
    for diagnostics in (False,True):
        a=hc.compilation_source('residual',collapse_policy=policy,return_frontiers=diagnostics)
        assert a['inputs']==['branch','mixed','post','pre']
        assert len(a['inputs'])+len(a['outputs'])<=8
        assert ('collapsed'in a['outputs'])is(policy!='stock_frontier')
        assert ('products'in a['outputs'])is(policy=='stock_frontier'or diagnostics)
    with pytest.raises(ValueError):hc.compilation_source('residual',collapse_policy='auto_relax')


def test_four_term_reassociation_is_not_implicitly_exact():
    # These are collapse products, not residuals: F32 pre-mixes can produce
    # cancellation even when each retained residual is BF16-representable.
    x=np.array([2**25,1,-2**25,1],np.float32)
    sequential=np.float32(0)
    for v in x:sequential=np.float32(sequential+v)
    pairwise=np.float32(np.float32(x[0]+x[1])+np.float32(x[2]+x[3]))
    assert sequential==1 and pairwise==0
    assert hc.plan(*inputs())['collapse_policy']=='stock_frontier'


def test_default_keeps_stock_sum_rms_and_bf16_frontiers(monkeypatch):
    calls=[];source_inputs=inputs();original_sum=mx.sum;original_norm=mx.fast.rms_norm
    h=mx.full((1,1,4,5120),2,mx.float32)
    products=mx.full(h.shape,.5,mx.float32)
    raw=mx.full(h.shape,2.0003,mx.float32)
    tagged_router=mx.full((1,1,5120),3,mx.float32);tagged_qat=mx.full(tagged_router.shape,4,mx.float32)
    def kernel(stage,policy,diagnostics):
        assert policy=='stock_frontier'
        def invoke(**kw):
            calls.append((stage,kw))
            if stage=='residual':return [h,products,raw]if diagnostics else[h,products]
            return [tagged_router,tagged_qat]
        return invoke
    sums=[];norms=[]
    def summed(value,axis):sums.append((value,axis));return original_sum(value,axis)
    def norm(value,weight,eps):norms.append((value,weight,eps));return original_norm(value,weight,eps)
    monkeypatch.setattr(hc,'_device',lambda:mx);monkeypatch.setattr(hc,'_kernel',kernel)
    monkeypatch.setattr(mx,'sum',summed);monkeypatch.setattr(mx.fast,'rms_norm',norm)
    outputs,frontiers=hc.boundary(*source_inputs,return_frontiers=True)
    assert outputs==(h,tagged_router,tagged_qat)
    assert sums==[(products,-2)]and norms[0][0]is not products
    mx.eval(frontiers['collapsed_raw'],frontiers['collapsed_bf16'],frontiers['norm_raw'])
    assert frontiers['expanded_raw']is raw and frontiers['collapse_products']is products
    np.testing.assert_array_equal(np.array(frontiers['collapsed_bf16']),np.full((1,1,5120),2,np.float32))
    assert calls[0][1]['inputs'][0]is source_inputs[0]
    assert calls[0][1]['inputs'][2]is source_inputs[2]and calls[0][1]['inputs'][3]is source_inputs[4]
    assert calls[1][1]['inputs'][0]is frontiers['norm_raw']
    assert calls[1][1]['template']==[('ELEMENTS',5120),('BLOCK',32),('KIND',0)]


def test_reference_source_keeps_matrix_orientation_and_all_rounds():
    tree=ast.parse(Path(hc.__file__).read_text());ref=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='reference')
    text=ast.unparse(ref)
    assert 'mx.matmul(mx.swapaxes(comb, -1, -2), residual.astype(mx.float32))'in text
    assert text.count('.astype(mx.bfloat16).astype(mx.float32)')==3
    assert "fused_quantize(router.astype(mx.float32), 'fp8')"in text
    assert 'coefficient_fn'not in text  # Prediction remains outside this partial component.
