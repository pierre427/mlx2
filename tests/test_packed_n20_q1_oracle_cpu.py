"""Import-safe source and plan checks for the frozen N20 Q1 device harness."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/research/varlen_packed_n20_q1_device_oracle.py'
spec = importlib.util.spec_from_file_location('n20_q1_oracle', SCRIPT)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def test_q1_cases_cover_survivor_ragged_three_and_twenty():
    assert [len(case) for case in oracle.CASES] == [1, 2, 3, 20]
    assert oracle.case_plan(oracle.CASES[-1])['scratch_bytes'] <= 64*1024*1024
    assert len(set(oracle.CASES[-1])) == 20
    for bad in ((), (1024,), tuple([1025]*21), (8193,), [1025], (True,)):
        with pytest.raises(ValueError): oracle.case_plan(bad)
    assert len(oracle.LARGE_WRITE_COUNTS) == 20
    assert 140000 <= sum(oracle.LARGE_WRITE_COUNTS) < 163840


def test_q1_source_preserves_failure_telemetry_and_exact_proof():
    source = SCRIPT.read_text()
    assert 'result[\'cases\'].append(case)' in source
    assert "case['status']='failed'" in source
    assert "case['error']=" in source
    assert 'failure_roots_retained' in source
    assert 'raw_mismatch_count=' in source
    assert 'if not finite or not exact:' in source
    assert 'branch.prove_staged_layer_read' in source
    assert 'state.publish()' in source
    assert 'reap_native_request_owner' in source
    assert '_gpuq_owner()' in source
    assert "'read_tested':False" in source
    assert "case['status']='failed'" in source
    assert 'native.q1_scalar_dispatch_count(arena._arena)' in source
    assert 'expected=(1,n,1,1,0,0,0) if n>1 else (1,1,1,1,0,1,1)' in source


def test_b1_stock_long_selector_is_explicit_and_physically_counted():
    source = SCRIPT.read_text()
    arena = (ROOT / 'native/paged_kv/arena.cpp').read_text()
    binding = (ROOT / 'native/paged_kv/binding.cpp').read_text()
    assert "'MLX2_PAGED_Q1_STOCK_LONG_N20_SINGLETON':'1'" in source
    assert 'q1_stock_long_n20_singleton_requested' in arena
    assert 'rows >= 2 || stock_n20_singleton_requested' in arena
    assert 'record_q1_stock_long_n20_singleton_partial_dispatch()' in arena
    assert 'record_q1_stock_long_n20_singleton_reduce_dispatch()' in arena
    assert 'q1_stock_long_n20_singleton_partial_dispatch_count' in binding
    assert 'q1_stock_long_n20_singleton_reduce_dispatch_count' in binding


def test_b1_scalar_counter_is_post_dispatch_only():
    arena = (ROOT / 'native/paged_kv/arena.cpp').read_text()
    dispatch = arena.index('encoder.dispatch_threads(\n        MTL::Size(q1_simd_tile_')
    counter = arena.index('arena_->record_q1_scalar_dispatch();', dispatch)
    assert dispatch < counter
    assert 'if (!q1_simd_tile_ && outputs[0].shape(0) == 1)' in arena[dispatch:counter]
    binding = (ROOT / 'native/paged_kv/binding.cpp').read_text()
    assert 'module.def("q1_scalar_dispatch_count"' in binding
    assert 'b1_survivor_scalar_dispatches' in binding
