"""Actual host retirement and runner-finally regressions; no MLX/native imports."""
import ast
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
import pytest
from mlx2.runtime import qwen3_paged_native_backend as backend_module
from mlx2.runtime import qwen35_paged_graph_factory as resources_module


def test_two_failed_owners_retire_after_one_way_arena_close(monkeypatch):
    arena = SimpleNamespace(_closed=False, _arena=object())
    writer = SimpleNamespace(backend=arena, poisoned=True, pending_epochs=(),
        ledger=SimpleNamespace(pending_count=0, completed_epoch=3),
        pool=SimpleNamespace(allocated_count=0))
    closes=[]
    def close():
        if not arena._closed:
            closes.append(True);arena._closed=True;arena._arena=None
    writer.backend.close_after_terminal=close
    writer.teardown_failed_arena=close
    polls=[]
    def poll(native_arena):
        assert native_arena._arena is not None, 'old code polls destroyed capsule'
        polls.append(True);return ()
    monkeypatch.setattr(backend_module,'poll_native_paged_read_events',poll)
    backend=object.__new__(backend_module.NativeQwen3PagedBackend)
    backend.writer=writer;backend._orphaned_reads={}
    owners=tuple(SimpleNamespace(fully_retired=False) for _ in range(2))
    for owner in owners:
        owner.reap_failed_after_teardown=lambda owner=owner:setattr(owner,'fully_retired',True)
    resources=resources_module.HybridServingResources(1,0,1)
    resources.owners=list(owners)
    resources.candidate=SimpleNamespace(backend=backend,_failure_roots=[],_bootstrap_failure_roots=[])
    before=resources_module._CHARGED
    assert resources.reap() is True
    assert resources.reap() is True
    assert len(closes)==1 and len(polls)==1
    assert all(owner.fully_retired for owner in owners)
    assert resources_module._CHARGED==before-1


def test_closed_arena_cannot_release_unproven_read_lease():
    backend=object.__new__(backend_module.NativeQwen3PagedBackend)
    backend.writer=SimpleNamespace(backend=SimpleNamespace(_closed=True,_arena=None))
    orphan=object();backend._orphaned_reads={7:orphan}
    with pytest.raises(RuntimeError,match='orphaned read leases'):
        backend.drain_failed_read_events()
    assert backend._orphaned_reads=={7:orphan}


def test_actual_runner_finally_preserves_q1_error_and_reports_cleanup_error():
    path=Path(__file__).parents[1]/'scripts/research/varlen_hybrid_packed_prefill_long_geometry_gate.py'
    tree=ast.parse(path.read_text())
    execute=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='execute')
    final=next(node.finalbody for node in execute.body if isinstance(node,ast.Try) and node.finalbody)
    original=ValueError('long fused selector rejected Q1')
    def close():raise TypeError('poll_read_completions(None)')
    result={'phase':'actual_b2_q1_handoff'};candidate=object();roots=[]
    namespace={'sys':sys,'traceback':traceback,'result':result,'prepared':[],
        'branches':[],'candidate':candidate,'owners':(SimpleNamespace(close=close),),
        'adapter':None,'FAILURE_ROOTS':roots,'original':original}
    wrapper=ast.Module(body=[ast.Try(body=[ast.Raise(exc=ast.Name(id='original',ctx=ast.Load()))],
        handlers=[],orelse=[],finalbody=final)],type_ignores=[])
    with pytest.raises(ValueError) as caught:exec(compile(ast.fix_missing_locations(wrapper),str(path),'exec'),namespace)
    assert caught.value is original
    assert result['error_phase']=='actual_b2_q1_handoff'
    assert 'long fused selector' in result['primary_traceback']
    assert 'poll_read_completions(None)' in result['cleanup_traceback']
    assert len(roots)==1 and roots[0][1] is candidate
