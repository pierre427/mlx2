"""Row-length/ancestry proof for a candidate; actual GPU bit proof is separate."""
import mlx.core as mx
import numpy as np
import pytest

from test_deepseek_v41_cpu import config,Weights
from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
from mlx2.runtime.models.deepseek_v4 import attend
from mlx2.adapters import deepseek_v41_ast_batched_attention_exact as exact
from mlx2.adapters import deepseek_v41_ast_batched_dense_exact as dense
from mlx2.adapters import deepseek_v41_ast_batched_dense as native_dense


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**kw: (_ for _ in ()).throw(AssertionError('GPU forbidden')))


@pytest.mark.parametrize('prefix',(0,5))
@pytest.mark.parametrize('finite',(False,True))
@pytest.mark.parametrize('topk',(2,32))
def test_tiny_composed_half_same_ancestry_ratio_index_and_owned_state(prefix,finite,topk,monkeypatch):
    # Only tiny WoA is a CPU test double. The production exact WoA explicitly
    # requires8x4096 and already has its independent actual-coefficient gate.
    import test_deepseek_v41_ast_batched_attention_cpu as oracle
    monkeypatch.setattr(Weights,'_ready',lambda self:None,raising=False)
    monkeypatch.setattr(Weights,'_ensure_dense',lambda self:None,raising=False)
    monkeypatch.setattr(dense,'grouped_tile',native_dense.grouped_tile)
    def candidate(block,*args,**kwargs):
        result=exact.attention_half_tile(block,*args,**kwargs)
        # Adapt only the reused oracle's historical counter spelling; the
        # production candidate retains its own explicit scheduling identity.
        counts=block.model.scheduled_kv_calls
        counts['source_ast_grouped_attention_tiles']=counts['source_ast_row_exact_attention_candidate_tiles']
        return result
    monkeypatch.setattr(oracle,'attention_half_tile',candidate)
    oracle.test_batched_half_all_ratios_index_refresh_partialgroups_and_private_caches(prefix,finite,topk,monkeypatch)


@pytest.mark.parametrize('prefix',(0,130))
def test_core_matches_ordinary_local_row_lengths_without_extra_masked_slots(prefix,monkeypatch):
    c=config();model=DeepSeekV41Model(c,Weights(c),identity='synthetic-exact-attention@785')
    base=model.make_cache()[0]
    rng=np.random.default_rng(785)
    base.offset=prefix
    if prefix:base.keys=mx.array(rng.normal(0,.2,(1,1,min(prefix,c.sliding_window),c.head_dim)).astype(np.float32))
    kv=mx.array(rng.normal(0,.2,(1,1,16,c.head_dim)).astype(np.float32))
    descriptor=exact.LayerAncestrySegments(base,kv)
    q=mx.array(rng.normal(0,.2,(1,c.num_attention_heads,16,c.head_dim)).astype(np.float32))
    sinks=mx.array([-.8,.3],mx.float32)
    calls=[];original=mx.matmul
    def matmul(a,b):calls.append((a,b));return original(a,b)
    monkeypatch.setattr(mx,'matmul',matmul)
    actual=exact.attention_core(q,descriptor,sinks)
    groups=exact.count_classes(descriptor)
    assert len(calls)==2*len(groups)
    assert sorted(node for _,_,nodes in groups for node in nodes)==list(range(16))
    assert len(groups)==(1 if prefix>=c.sliding_window else 6)
    for left,right in calls:
        assert left.ndim==right.ndim==3 and left.shape[1]==c.num_attention_heads
        assert np.asarray(left).strides[0]==(left.shape[1]*left.shape[2]+1)*4
    for node in range(16):
        local=mx.take(descriptor.local_table,mx.array(descriptor.local_rows[node],mx.int32),axis=2)
        reference=attend(q[:,:,node:node+1],local,mx.ones((1,1,1,local.shape[2]),mx.bool_),sinks=sinks)
        np.testing.assert_allclose(np.asarray(actual[:,:,node:node+1]),np.asarray(reference),atol=3e-7,rtol=3e-5)


def test_no_extra_sibling_slots_can_poison_other_branch_reduction():
    c=config();model=DeepSeekV41Model(c,Weights(c),identity='synthetic-exact-poison@1');base=model.make_cache()[0]
    values=mx.ones((1,1,16,c.head_dim));poison=mx.ones(values.shape);poison[:,:,6:]=mx.nan
    q=mx.zeros((1,c.num_attention_heads,16,c.head_dim));sinks=mx.zeros((c.num_attention_heads,))
    expected=exact.attention_core(q,exact.LayerAncestrySegments(base,values),sinks)
    actual=exact.attention_core(q,exact.LayerAncestrySegments(base,poison),sinks)
    np.testing.assert_array_equal(np.asarray(actual[:,:,:6]),np.asarray(expected[:,:,:6]))


@pytest.mark.parametrize('ratio',(0,1,2))
def test_count_plan_longprefix_one16row_batch_without_extra_local_or_selected_slots(ratio):
    p=exact.plan(1024,ratio=ratio)
    assert p['physical_query_rows']==16 and len(p['count_classes'])==1
    assert p['count_classes'][0]['local_rows']==128
    assert p['count_classes'][0]['selected_rows']==(512 if ratio else 0)
    assert p['local_gather_bound_bytes']==4_194_304 and p['selected_gather_bound_bytes']==16_777_216
    assert p['full_prefix_KV_copies']==0 and not p['exact_M1_claim']and not p['qualified']


def test_owned_splitk_scope_checks_actual_callback_and_restores_after_failure():
    with pytest.raises(ValueError,match='caller-owned'):
        with exact.owned_stock_splitk(None):pass
    with pytest.raises(RuntimeError,match='ownership lost'):
        with exact.owned_stock_splitk(lambda:False):pass
    assert exact._SPLITK_OWNERSHIP.get()is None
    with pytest.raises(RuntimeError,match='injected'):
        with exact.owned_stock_splitk(lambda:None):
            with pytest.raises(ValueError,match='already active'):
                with exact.owned_stock_splitk(lambda:True):pass
            raise RuntimeError('injected')
    assert exact._SPLITK_OWNERSHIP.get()is None


@pytest.mark.parametrize('transpose',(False,True))
def test_gpu_dispatch_predicate_only_requires_owner_and_preserves_explicit_role(monkeypatch,transpose):
    # Arrays/backend remain CPU. Simulate only the Python dispatch predicate;
    # replace the shader consumer completely, so no GPU is used or compiled.
    from mlx2.runtime.gguf import source_ast_batched_splitk as selected
    left=mx.zeros((3,64,128),mx.float32);right=mx.zeros((3,128,11),mx.float32)
    calls=[]
    def shader(a,b,*,transpose_b,check_live):
        assert check_live()is True
        calls.append((a.shape,b.shape,transpose_b))
        return mx.zeros((3,64,11),mx.float32)
    monkeypatch.setattr(selected,'private_matmul',shader)
    monkeypatch.setattr(mx,'default_device',lambda:mx.gpu)
    with pytest.raises(ValueError,match='owned_stock_splitk'):
        exact._private_matmul(left,right,transpose_b=transpose)
    with exact.owned_stock_splitk(lambda:True):
        out=exact._private_matmul(left,right,transpose_b=transpose)
    assert calls==[((3,64,128),(3,128,11),transpose)]and out.shape==(3,64,11)


@pytest.mark.parametrize('transpose',(False,True))
def test_gpu_singleton_dispatch_uses_literal_ordinary_head_broadcast(monkeypatch,transpose):
    # Predicate-only GPU test: actual arrays and evaluation remain CPU.
    from mlx2.runtime.gguf import source_ast_head_axis_matmul as head_axis
    left=mx.arange(64*128,dtype=mx.float32).reshape(1,64,128)/8192
    right=mx.arange(128*11,dtype=mx.float32).reshape(1,128,11)/1408
    calls=[];original=mx.matmul
    def forbidden(*a,**kw):raise AssertionError('singleton must preserve ordinary layout')
    def matmul(a,b):
        calls.append((a.shape,b.shape))
        return original(a,b)
    monkeypatch.setattr(head_axis,'private_matmul',forbidden)
    monkeypatch.setattr(mx,'matmul',matmul)
    monkeypatch.setattr(mx,'default_device',lambda:mx.gpu)
    checked=[]
    with exact.owned_stock_splitk(lambda:checked.append(True)):
        result=exact._private_matmul(left,right,transpose_b=transpose)
    assert calls==[((1,64,1,128),(1,1,128,11))]
    assert len(checked)==2 and result.shape==(1,64,11)
    coefficient=(mx.swapaxes(mx.contiguous(mx.swapaxes(right,-1,-2)),-1,-2)
                 if transpose else mx.contiguous(right))
    reference=original(mx.contiguous(left).reshape(1,64,1,128),coefficient[:,None]).squeeze(2)
    np.testing.assert_array_equal(np.asarray(result),np.asarray(reference))

@pytest.mark.parametrize('transpose',(False,True))
@pytest.mark.parametrize('batches',(1,3))
def test_selected_role_preserves_actual_einsum_singleton_or_pinned_group(monkeypatch,transpose,batches):
    from mlx2.runtime.gguf import source_ast_head_axis_matmul as head_axis
    from mlx2.runtime.gguf import source_ast_batched_splitk as selected
    left=mx.zeros((batches,64,128),mx.float32);right=mx.zeros((batches,128,11),mx.float32)
    calls=[];original_einsum=mx.einsum
    def forbidden(*a,**kw):raise AssertionError('selected role must not use local head-axis law')
    def pinned(a,b,*,transpose_b,check_live):
        assert check_live()is True
        calls.append(('pinned',a.shape,b.shape,transpose_b))
        return mx.zeros((batches,64,11),mx.float32)
    def einsum(eq,a,b):
        calls.append((eq,a.shape,b.shape));return original_einsum(eq,a,b)
    monkeypatch.setattr(head_axis,'private_matmul',forbidden)
    monkeypatch.setattr(selected,'private_matmul',pinned)
    monkeypatch.setattr(mx,'einsum',einsum)
    monkeypatch.setattr(mx,'default_device',lambda:mx.gpu)
    with exact.owned_stock_splitk(lambda:True):
        result=exact._private_matmul(left,right,transpose_b=transpose,role='selected')
    if batches==1:
        assert calls==[('bhld,blkd->bhlk'if transpose else'bhlk,blkd->bhld',
                        (1,64,1,128),(1,1,11,128)if transpose else(1,1,128,11))]
    else:
        assert calls==[('pinned',(3,64,128),(3,128,11),transpose)]
    assert result.shape==(batches,64,11)
