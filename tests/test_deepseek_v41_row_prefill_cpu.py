"""Direct expert bindings must preserve every assignment and16-row tile order."""

import numpy as np
import pytest

from mlx2.runtime.gguf.reader import GGUFError
from mlx2.runtime.gguf.v41_row_prefill import GROUP_EXPERTS, routing_plan


@pytest.mark.parametrize("experts,rows", [(2,1), (13,17), (25,512), (384,1024)])
def test_grouped_plan_preserves_slot_order_and_covers_each_assignment_once(experts, rows):
    rng = np.random.default_rng(41024)
    ids = rng.integers(0, experts, (rows, min(6,experts)), np.uint32)
    inverse, groups = routing_plan(ids, experts)
    order = np.concatenate([g[2] for g in groups])
    np.testing.assert_array_equal(order[inverse], np.arange(ids.size))
    np.testing.assert_array_equal(ids.reshape(-1)[order], np.sort(ids.reshape(-1), kind="stable"))
    for begin, end, assignments, tiles in groups:
        assert 0 < end-begin <= GROUP_EXPERTS
        coverage = []
        for row, expert, valid, padding in tiles:
            assert 1 <= valid <= 16 and padding == 0
            selected = assignments[int(row):int(row+valid)]
            np.testing.assert_array_equal(ids.reshape(-1)[selected], np.full(valid, begin+expert))
            coverage.extend(range(int(row), int(row+valid)))
        assert coverage == list(range(len(assignments)))


def test_sparse_groups_keep_inverse_order_and_skip_empty_groups():
    ids = np.full((17, 1), 383, np.uint32)
    inverse, groups = routing_plan(ids, 384)
    assert len(groups) == 1 and groups[0][:2] == (372,384)
    np.testing.assert_array_equal(inverse, np.arange(17))
    np.testing.assert_array_equal(groups[0][3], [[0,11,16,0],[16,11,1,0]])


@pytest.mark.parametrize("ids,experts", [(np.zeros((0,6),np.uint32),6),
                                        (np.zeros((1,7),np.uint32),7),
                                        (np.array([[384]],np.uint32),384),
                                        (np.array([[-1]],np.int32),6)])
def test_invalid_route_never_publishes_a_partial_plan(ids, experts):
    with pytest.raises(GGUFError):
        routing_plan(ids,experts)


def test_profile_byte_oracle_distinguishes_signed_zero_dtype_and_shape(monkeypatch):
    import importlib
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts/research"))
    equal = importlib.import_module("profile_deepseek_v41").arrays_equal
    assert not equal(np.array([0.0],np.float32),np.array([-0.0],np.float32))
    assert not equal(np.zeros((1,2),np.float32),np.zeros((2,1),np.float32))
    assert not equal(np.zeros(1,np.float32),np.zeros(1,np.uint32))
    assert equal((None,np.array([1],np.float32)),(None,np.array([1],np.float32)))
