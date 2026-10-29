"""CPU preparation/refusal checks; no Source model/GPU qualification."""
import importlib.util
from pathlib import Path
import mlx.core as mx
import numpy as np
import pytest


def module():
    path=Path(__file__).resolve().parents[1]/'scripts/research/ds41_ast_exact_stage_probe.py'
    spec=importlib.util.spec_from_file_location('ast_exact_stage_probe',path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def test_guards_refuse_before_private_helper_import_or_adapter_access():
    value=module();mx.set_default_device(mx.cpu)
    with pytest.raises(ValueError,match='owned live guard'):value.run(None,None,None)
    with pytest.raises(ValueError,match='parent-owned GPU'):value.run(None,None,lambda:True)
    empty=np.zeros((1,0,4),np.float32)
    assert value._comparison(empty,empty)['equal']
    assert value._comparison(empty,empty)['maximum_absolute_error']==0


def test_tiny_lazy_operator_instrumentation_restores_hooks_and_retains_half_state(monkeypatch):
    from test_deepseek_v41_ast_target_cpu import fixture,PATHS
    from mlx2.adapters import deepseek_v41_ast_batched_dense_exact as dense
    from mlx2.adapters import deepseek_v41_ast_batched_dense as native
    from mlx2.adapters.deepseek_v41_ast_batched_target import execute_batched_tree
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**kw: (_ for _ in ()).throw(AssertionError('GPU forbidden')))
    model,w,cache=fixture();w._ready=lambda:None;w._ensure_dense=lambda:None
    monkeypatch.setattr(dense,'grouped_tile',native.grouped_tile)
    original=dense.linear_tile;value=module()
    with value.instrument(model,lambda:True)as rows:
        result=execute_batched_tree(model,cache,3,PATHS,(2,3,4),cpu_reference=True,
                                    expert_law='ordinary_m16_v1',forward_law='fully_batched_m1_v1')
        mx.eval([(row['candidate'],row['oracle'])for row in rows])
    assert dense.linear_tile is original
    assert rows and {row['layer']for row in rows}==set(range(5))
    assert any(row['kind']=='HC_raw_projection'for row in rows)
    assert any(row['kind']=='attention_core'for row in rows)
    assert any(row['kind']=='attention_QK_local'for row in rows)
    assert any(row['kind']=='attention_PV_local'for row in rows)
    assert any(row['kind']=='attention_logsumexp'for row in rows)
    assert any(row['kind']=='attention_exp'for row in rows)
    assert any(row['kind']=='complete_attention_cache_buf_kv'for row in rows)
    assert all(value._comparison(row['candidate'],row['oracle'])['finite']for row in rows)
    result.transaction.abort()
