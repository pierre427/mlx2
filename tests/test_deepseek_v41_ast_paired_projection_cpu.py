"""Joined original coefficient rows and research ownership; no GPU proof."""
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters import deepseek_v41_ast_batched_dense_exact as exact
from mlx2.adapters import deepseek_v41_ast_paired_projection as joined


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('GPU forbidden')))


def model():
    quant={}
    for layer in range(40):
        for slot,name in enumerate(('attn_q_a','attn_kv','ffn_gate_shexp','ffn_up_shexp')):
            q=mx.broadcast_to((mx.arange(32,dtype=mx.uint32)+slot*40)[:,None],(32,16))
            quant[f'blk.{layer}.{name}.weight']=(q,mx.full((32,2),slot+1,mx.uint8))
    weights=SimpleNamespace(_quant=quant,source_dense_qat_policy='fused_fp32',scheduled_dense_calls={},prepared_calls=0)
    weights._ready=lambda:None;weights._ensure_dense=lambda:None
    def prepare(x):weights.prepared_calls+=1;return x
    weights._dense_input=prepare
    return SimpleNamespace(args=SimpleNamespace(num_hidden_layers=40,hidden_size=64,q_lora_rank=32,
        head_dim=32,moe_intermediate_size=32),identity='synthetic-paired-output@cpu',weights=weights,
        round=lambda value:value.astype(mx.bfloat16).astype(mx.float32),_weight_executions=0,_mlx2_ds41_ast_active=None)


def fake_qmm(monkeypatch):
    calls=[]
    def qmm(x,q,scales):
        calls.append((x.shape,q.shape,scales.shape))
        return mx.sum(x,axis=-1,keepdims=True)+q[:,0].astype(mx.float32).reshape(1,1,-1)
    monkeypatch.setattr(exact,'_quant_batch_matmul',qmm)
    return calls


def test_prepare80_pairs_preserves_original_rows_and_reports_duplication():
    m=model();original=dict(m.weights._quant);p=joined.prepare(m);r=p.receipt()
    assert r['pairs']==80 and r['original_coefficient_pairs']==160
    assert r['logical_preparation_bytes']==80*64*(16*4+2)
    assert r['duplicate_coefficient_bytes']==r['logical_preparation_bytes']
    assert len(p.evaluation_roots())==80*6
    for names,pair in p.pairs.items():
        for name,owner in zip(names,pair.owners):assert m.weights._quant[name]is original[name]is owner
        np.testing.assert_array_equal(np.asarray(pair.q[:32]),np.asarray(original[names[0]][0]))
        np.testing.assert_array_equal(np.asarray(pair.q[32:]),np.asarray(original[names[1]][0]))
        np.testing.assert_array_equal(np.asarray(pair.scales[32:]),np.asarray(original[names[1]][1]))
    assert not r['qualified']and not r['selected']and not r['physical_peak_memory_claim']


@pytest.mark.parametrize('rounded',(False,True))
def test_one_joined_qmm_matches_two_outputs_with_one_dense_input_and_record(monkeypatch,rounded):
    m=model();p=joined.prepare(m);calls=fake_qmm(monkeypatch)
    x=mx.arange(16*64,dtype=mx.float32).reshape(1,16,64)/64
    names=('blk.0.attn_q_a.weight','blk.0.attn_kv.weight')
    expected=exact.paired_linear_tile(m,x,*names,rounded=rounded)
    assert len(calls)==2 and m.weights.prepared_calls==1
    calls.clear();m.weights.prepared_calls=0;before=exact.paired_linear_tile
    with joined.installed(p)as metrics:
        # A real target is active inside the already installed research window.
        m._weight_executions=1;m._mlx2_ds41_ast_active=object()
        actual=exact.paired_linear_tile(m,x,*names,rounded=rounded)
        assert metrics['joined_calls']==1 and metrics['baseline_calls']==0
        assert calls==[((1,16,64),(64,16),(64,2))]
        assert m.weights.prepared_calls==1
    assert exact.paired_linear_tile is before
    for a,b in zip(actual,expected):np.testing.assert_array_equal(np.asarray(a),np.asarray(b))


def test_ineligible_pairs_keep_original_and_restoration_after_exception(monkeypatch):
    m=model();p=joined.prepare(m);original=exact.paired_linear_tile;sentinel=object()
    def baseline(*args,**kwargs):return sentinel
    monkeypatch.setattr(exact,'paired_linear_tile',baseline)
    with pytest.raises(RuntimeError,match='intentional'):
        with joined.installed(p)as metrics:
            assert exact.paired_linear_tile(m,None,'other.weight','different.weight')is sentinel
            assert metrics['baseline_calls']==1 and metrics['joined_calls']==0
            with pytest.raises(ValueError,match='already installed'):
                with joined.installed(p):pass
            raise RuntimeError('intentional')
    assert exact.paired_linear_tile is baseline
    monkeypatch.setattr(exact,'paired_linear_tile',original)


@pytest.mark.parametrize('change',('identity','quant_owner','quant_dict','policy'))
def test_stale_eligible_binding_refuses_before_qmm(monkeypatch,change):
    m=model();p=joined.prepare(m);calls=fake_qmm(monkeypatch);names=('blk.0.attn_q_a.weight','blk.0.attn_kv.weight')
    with joined.installed(p):
        if change=='identity':m.identity='foreign'
        elif change=='quant_owner':m.weights._quant[names[0]]=tuple(list(m.weights._quant[names[0]]))
        elif change=='quant_dict':m.weights._quant=dict(m.weights._quant)
        else:m.weights.source_dense_qat_policy='reference'
        with pytest.raises(ValueError,match='changed'):
            exact.paired_linear_tile(m,mx.zeros((1,16,64)),*names)
    assert not calls


def test_idle_foreign_model_and_original_geometry_refusals(monkeypatch):
    m=model();p=joined.prepare(m);m._weight_executions=1
    with pytest.raises(ValueError,match='idle'):
        with joined.installed(p):pass
    with pytest.raises(ValueError,match='idle'):joined.prepare(m)
    m._weight_executions=0
    with joined.installed(p):
        with pytest.raises(ValueError,match='foreign model'):
            exact.paired_linear_tile(model(),None,'other.weight','foreign.weight')
    for mutation in('scales','policy','layers'):
        m=model()
        if mutation=='scales':
            q,_=m.weights._quant['blk.0.attn_q_a.weight'];m.weights._quant['blk.0.attn_q_a.weight']=(q,mx.ones((32,2),mx.float32))
        elif mutation=='policy':m.weights.source_dense_qat_policy='reference'
        else:m.args.num_hidden_layers=39
        with pytest.raises(ValueError):joined.prepare(m)
