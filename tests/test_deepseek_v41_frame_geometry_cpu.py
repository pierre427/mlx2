"""CPU fake-backend expression and forward ownership checks; no MLX import."""
import ast
from contextlib import nullcontext
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
PATH = ROOT / "src/mlx2/runtime/models/deepseek_v41_frame_geometry.py"
SPEC = importlib.util.spec_from_file_location("frame_geometry_cpu", PATH)
G = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(G)


class Backend:
    float32 = np.float32
    uint32 = np.uint32
    device = "cpu"
    calls = None

    def __getattr__(self, name):
        fn = getattr(np, name)
        def call(*a, **k):
            if self.calls is not None:self.calls.append(name)
            return fn(*a, **k)
        return call

    def default_device(self):return self.device


def cache():return G.frame_geometry(NS(source_frame_reuse_policy="prefill_v1"), 7)
def rope():return NS(dims=8, inv=np.arange(4, dtype=np.float32) * .125)


@pytest.mark.parametrize("length", [1, 0])
def test_decode_no_geometry(length):
    assert G.frame_geometry(NS(source_frame_reuse_policy="prefill_v1"), length) is None


@pytest.mark.parametrize("policy", ["reference", "unknown", None])
def test_explicit_optin_only(policy):
    assert G.frame_geometry(NS(source_frame_reuse_policy=policy), 99) is None


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("positions", [None, (0, 2, 4, 6, 8, 10, 12)])
def test_original_phase_and_rotary_expression_bits(inverse, positions):
    xp=Backend();r=rope();x=np.random.default_rng(41).normal(size=(1, 2, 7, 12)).astype(np.float32)
    phase=G.phase_operands(cache(), xp, r, (8,10000,"null"), x, 13, positions=positions, inverse=inverse)
    pos=np.arange(13,20) if positions is None else np.array(positions)
    angles=pos.astype(np.float32)[:,None]*r.inv[None]
    if inverse:angles=-angles
    expected=(np.cos(angles).astype(x.dtype),np.sin(angles).astype(x.dtype))
    for a,b in zip(phase,expected):np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))
    prefix,rotary=x[...,:-8],x[...,-8:]
    a,b=rotary[...,::2],rotary[...,1::2];c,s=expected
    reference=np.concatenate((prefix,np.stack((a*c-b*s,a*s+b*c),axis=-1).reshape(rotary.shape)),axis=-1)
    np.testing.assert_array_equal(G.apply_rotary(xp,x,r,phase).view(np.uint32),reference.view(np.uint32))


def test_share_across_compatible_owners_not_other_geometry():
    xp=Backend();xp.calls=[];c=cache();x=np.zeros((1,2,7,12),np.float32);key=(8,10000,"null")
    first=G.phase_operands(c,xp,rope(),key,x,11)
    second=G.phase_operands(c,xp,rope(),key,x,11)
    assert second is first and xp.calls.count('cos')==1 and xp.calls.count('sin')==1
    for kw in ({'inverse':True},{'offset':12},{'positions':(0,2,4,6,8,10,12)}):
        offset=kw.pop('offset',11)
        assert G.phase_operands(c,xp,rope(),key,x,offset,**kw) is not first
    assert G.phase_operands(c,xp,rope(),(8,160000,"null"),x,11)is not first
    xp.device='gpu'
    assert G.phase_operands(c,xp,rope(),key,x,11)is not first
    assert G.phase_operands(c,xp,rope(),key,x.astype(np.float16),11)is not first
    assert c['counts']['phase_hit']==1


def test_array_positions_fall_back_without_readback_or_retention():
    c=cache();xp=Backend();xp.calls=[];x=np.zeros((1,2,7,12),np.float32)
    assert G.phase_operands(c,xp,rope(),(8,10000,"null"),x,positions=np.arange(7))is None
    assert not c['phases'] and not xp.calls


@pytest.mark.parametrize("start,length,band,window",[(0,99,99,128),(99,7,106,128),(200,99,226,128),(0,7,7,1)])
def test_mask_original_expression_and_same_object(start,length,band,window):
    xp=Backend();c=cache();a=G.local_mask(c,xp,start,length,band,window)
    query=np.arange(start,start+length)[:,None];keys=np.arange(start+length-band,start+length)
    np.testing.assert_array_equal(a,((query>=keys)&(query<keys+window))[None,None])
    assert G.local_mask(c,xp,start,length,band,window)is a
    assert G.local_mask(c,xp,start+1,length,band,window)is not a
    assert G.local_mask(c,xp,start,length,band,window+1)is not a


def test_public_counters_and_identity():
    counters={};c=G.frame_geometry(NS(source_frame_reuse_policy='prefill_v1'),7,counters);xp=Backend();x=np.zeros((1,1,7,12),np.float32)
    for _ in range(2):
        G.phase_operands(c,xp,rope(),(8,10000,'null'),x)
        G.local_mask(c,xp,0,7,7,128)
    assert counters=={'source_frame_rope_build':1,'source_frame_rope_hit':1,'source_frame_mask_build':1,'source_frame_mask_hit':1}
    assert len(G.implementation_identity())==64


def _forward_method():
    tree=ast.parse((ROOT/'src/mlx2/runtime/models/deepseek_v41.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='DeepSeekV41Model')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='_forward_impl')
    # Isolate the actual owning method with a CPU backend; replace only local
    # helper import, leaving control flow and frame ownership unchanged.
    fn.body=[n for n in fn.body if not(isinstance(n,ast.ImportFrom)and n.module=='deepseek_v41_frame_geometry')]
    module=ast.Module(body=[fn],type_ignores=[]);ast.fix_missing_locations(module)
    scope={'mx':Backend(),'np':np,'frame_geometry':G.frame_geometry}
    exec(compile(module,str(PATH),'exec'),scope)
    return scope['_forward_impl']


def test_real_forward_owns_fresh_dictionary_across_layers_and_errors():
    seen=[];masks=[];fail=[False];xp=Backend()
    class Block:
        def __call__(self,h,pre,state,states,shared):
            geometry=shared['_source_prefill_geometry'];seen.append(geometry)
            masks.append(G.local_mask(geometry,xp,0,5,5,128))
            if fail[0]:raise RuntimeError('injected')
            return h,pre
    states=[NS(_in_forward=False,state=(),validate_state=lambda:None)for _ in range(2)]
    weights=NS(source_frame_reuse_policy='prefill_v1',embedding=lambda ids:np.zeros((*ids.shape,4),np.float32))
    model=NS(args=NS(engram_layer_ids=(),hc_mult=4,hidden_size=4),weights=weights,hasher=None,
        validate_forward=lambda ids,cache,keep,ced_deferred_admitted=False:(states,ids),_weight_scope=nullcontext,round=lambda x:x,
        _weight_eval=lambda *a:None,layers=[Block(),Block()],scheduled_kv_calls={},eval_interval=0,
        collapse=lambda h,p:h[:,:,0],norm=lambda x,w:x,norm_weight=None)
    forward=_forward_method();ids=np.zeros((1,5),np.int32)
    forward(model,ids,logits_to_keep=0,body_only=True)
    forward(model,ids,logits_to_keep=0,body_only=True)
    assert seen[0]is seen[1]and masks[0]is masks[1]
    assert seen[2]is seen[3]and seen[2]is not seen[0]
    assert not hasattr(model,'_source_prefill_geometry')
    fail[0]=True
    with pytest.raises(RuntimeError):forward(model,ids,logits_to_keep=0,body_only=True)
    failed=seen[-1];fail[0]=False
    forward(model,ids,logits_to_keep=0,body_only=True)
    assert seen[-1]is not failed


def test_fused_phase_api_preserves_fallback_and_validates_geometry():
    path=ROOT/'src/mlx2/runtime/models/deepseek_v41_fused.py'
    tree=ast.parse(path.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_checked_phase')
    scope={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),scope)
    x=np.zeros((1,2,7,12),np.float32);r=rope();p=(np.zeros((7,4),np.float32),np.zeros((7,4),np.float32))
    assert scope['_checked_phase'](x,r,p)is p
    for invalid in ((p[0],p[1][:1]),(p[0].astype(np.float16),p[1]),list(p)):
        with pytest.raises(ValueError):scope['_checked_phase'](x,r,invalid)
    for name in ('rotate','rotate_qat'):
        func=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
        assert 'phase'in [n.arg for n in func.args.kwonlyargs]
        assert any(isinstance(n,ast.If)and ast.unparse(n.test)=='phase is None'for n in func.body)


def test_geometry_is_opaque_to_actual_completion_root_traversal():
    path=ROOT/'src/mlx2/adapters/deepseek_v41_source_completion.py'
    spec=importlib.util.spec_from_file_location('geometry_completion_cpu',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class Backend:
        array=np.ndarray
        def async_eval(self,*arrays):self.submitted=arrays
    xp=Backend();owner=module.BoundedSourceCompletion(NS(),backend=xp)
    owner.active=True;owner._scope_keys=set()
    c=cache();c.phases['example']=(np.ones(32),np.ones(32));c.masks['example']=np.ones(99,bool)
    output=np.zeros(7)
    owner.evaluate(output,{'_source_prefill_geometry':c})
    assert len(xp.submitted)==1 and xp.submitted[0]is output
    assert len(owner._values)==1 and owner._values[0]is output
    assert not isinstance(c,(dict,list,tuple))


@pytest.mark.parametrize('name',['rotate','rotate_qat'])
def test_actual_fused_wrappers_borrow_phases_without_recomputing(name):
    path=ROOT/'src/mlx2/runtime/models/deepseek_v41_fused.py'
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ('_checked_phase',name)]
    class StripImports(ast.NodeTransformer):
        def visit_Import(self,node):return None
        def visit_ImportFrom(self,node):return None
    nodes=[StripImports().visit(n)for n in nodes]
    module=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(module)
    xp=Backend();xp.calls=[];captured=[]
    def launch(**kw):captured.append(kw);return [np.zeros((1,2,7,12),np.float32)]
    scope={'mx':xp,'kernel':lambda kind:launch,'rotate_qat_kernel':lambda:launch,'_FUSED_KINDS':{'fp8':0}}
    exec(compile(module,str(path),'exec'),scope)
    x=np.zeros((1,2,7,12),np.float32);r=rope()
    p=G.phase_operands(cache(),xp,r,(8,10000,'null'),x,9,inverse=True)
    xp.calls.clear()
    args=(x,r)if name=='rotate'else(x,r,'fp8')
    scope[name](*args,offset=9,inverse=True,phase=p)
    inputs=captured[-1]['inputs']
    assert inputs[-2]is p[0]and inputs[-1]is p[1]
    assert not any(k in xp.calls for k in ('arange','cos','sin'))
    scope[name](*args,offset=9,inverse=True)
    assert xp.calls.count('cos')==1 and xp.calls.count('sin')==1
    for a,b in zip(captured[-1]['inputs'][-2:],p):np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))
