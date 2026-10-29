"""CPU fixture-map integrity and owned-GPU refusals; no model/math proof."""
from pathlib import Path
import importlib.util
import numpy as np
import mlx.core as mx
import pytest

spec=importlib.util.spec_from_file_location('mapped_probe_cpu',Path(__file__).parents[1]/'scripts/research/ds41_ast_mapped_kv_components.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def test_byte_dedup_preserves_negative_zero_and_chronological_duplicates():
    right=np.array([[[0.,1.],[-0.,1.],[0.,1.]],[[2.,3.],[0.,1.],[-0.,1.]]],np.float32)
    table,mapping=probe.deduplicate(right,False)
    assert table.shape==(3,2)and mapping.tolist()==[[0,1,0],[2,0,1]]
    assert table[mapping].tobytes()==right.tobytes()


def test_nt_rows_are_kv_vectors_not_dot_features():
    kv=np.array([[[1.,2.,3.],[4.,5.,6.]],[[1.,2.,3.],[7.,8.,9.]]],np.float32)
    right=np.swapaxes(kv,-1,-2)
    table,mapping=probe.deduplicate(right,True)
    assert table.shape==(3,3)and mapping.tolist()==[[0,1],[0,2]]
    assert table[mapping].tobytes()==np.ascontiguousarray(kv).tobytes()


def test_fixture_dedup_and_runtime_guard_refuse_bad_inputs(tmp_path):
    with pytest.raises(ValueError):probe.deduplicate(np.zeros((1,2,3),np.float64),False)
    mx.set_default_device(mx.cpu)
    with pytest.raises(ValueError,match='callback'):probe.run_replay(object(),tmp_path,None,'missing','missing')
    with pytest.raises(ValueError,match='parent-owned GPU'):probe.run_replay(object(),tmp_path,lambda:True,'missing','missing')
