"""No GPU: actual operand-capture plumbing and early ownership refusal."""
import importlib.util
from pathlib import Path
import mlx.core as mx
import pytest


def module():
    p=Path(__file__).resolve().parents[1]/'scripts/research/ds41_ast_attention_matmul_components.py'
    s=importlib.util.spec_from_file_location('ast_matmul_components',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_refusals_precede_model_access_and_private_imports():
    m=module();mx.set_default_device(mx.cpu)
    with pytest.raises(ValueError,match='owned live guard'):m.run(None,None,None)
    with pytest.raises(ValueError,match='parent-owned GPU'):m.run(None,None,lambda:True)
    with pytest.raises(ValueError,match='owned live guard'):m.run_captured([],None)
    with pytest.raises(ValueError,match='parent-owned GPU'):m.run_captured([],lambda:True)
    with pytest.raises(ValueError,match='owned live guard'):m.run_replay(None,None,None,None,None)
    with pytest.raises(ValueError,match='parent-owned GPU'):m.run_replay(None,None,lambda:True,None,None)


def test_cpu_tiny_capture_retains_both_actual_operands_and_restores_private_hook(monkeypatch):
    from test_deepseek_v41_ast_target_cpu import fixture,PATHS
    from mlx2.adapters import deepseek_v41_ast_batched_dense_exact as dense
    from mlx2.adapters import deepseek_v41_ast_batched_dense as native
    from mlx2.adapters import deepseek_v41_ast_batched_attention_exact as attention
    from mlx2.adapters.deepseek_v41_ast_batched_target import execute_batched_tree
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**kw: (_ for _ in ()).throw(AssertionError('GPU forbidden')))
    monkeypatch.setattr(dense,'grouped_tile',native.grouped_tile)
    model,w,cache=fixture();w._ready=lambda:None;w._ensure_dense=lambda:None
    original=attention._private_matmul;m=module()
    with m.capture(model)as cases:
        result=execute_batched_tree(model,cache,3,PATHS,(2,3,4),cpu_reference=True,
                                    expert_law='ordinary_m16_v1',forward_law='fully_batched_m1_v1')
    assert attention._private_matmul is original
    assert cases and {v['role']for v in cases}=={'QK_local','QK_selected','PV_local','PV_selected'}
    for case in cases:
        assert case['left'].shape[0]==case['right'].shape[0]==3
        assert case['left'].shape[-1]==case['right'].shape[-2]
        assert case['nodes']==(1,6,11)and case['transpose_b']==case['role'].startswith('QK')
    result.transaction.abort()

@pytest.mark.parametrize('role,equation',(
    ('QK_selected','bhld,blkd->bhlk'),('PV_selected','bhlk,blkd->bhld')))
def test_actual_selected_control_uses_owning_einsum_layout(role,equation,monkeypatch):
    import numpy as np
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**kw: (_ for _ in ()).throw(AssertionError('GPU forbidden')))
    m=module();left=mx.arange(2*4,dtype=mx.float32).reshape(1,2,4)/8
    right=mx.arange(4*3,dtype=mx.float32).reshape(1,4,3)/12
    calls=[];original=mx.einsum
    def einsum(eq,a,b):
        calls.append((eq,a.shape,b.shape));return original(eq,a,b)
    monkeypatch.setattr(mx,'einsum',einsum)
    result=m.role_control(left,right,role)
    assert calls==[(equation,(1,2,1,4),(1,1,3,4)if role=='QK_selected'else(1,1,4,3))]
    np.testing.assert_allclose(np.asarray(result),np.asarray(left)@np.asarray(right),atol=1e-6)
    with pytest.raises(ValueError,match='one owning B1'):
        m.role_control(mx.broadcast_to(left,(2,2,4)),right,role)


@pytest.mark.parametrize('transpose',(False,True))
def test_actual_local_control_reshapes_both_singleton_strides(monkeypatch,transpose):
    import numpy as np
    mx.set_default_device(mx.cpu)
    m=module();left=mx.arange(8,dtype=mx.float32).reshape(1,2,4)/8
    right=mx.arange(12,dtype=mx.float32).reshape(1,4,3)/12
    calls=[];original=mx.matmul
    def matmul(a,b):
        calls.append((a.shape,b.shape,tuple(s//4 for s in np.asarray(a).strides),
                      tuple(s//4 for s in np.asarray(b).strides)))
        return original(a,b)
    monkeypatch.setattr(mx,'matmul',matmul)
    result=m.local_control(left,right,transpose_b=transpose)
    assert calls==[((1,2,1,4),(1,1,4,3),(8,4,4,1),(12,12,1,4)if transpose else(12,12,3,1))]
    np.testing.assert_allclose(np.asarray(result),np.asarray(left)@np.asarray(right),atol=1e-6)
