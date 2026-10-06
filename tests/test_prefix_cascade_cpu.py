import pytest

from mlx2.adapters.north_mini_code import NorthMiniCodeAdapter
from mlx2.runtime.prefix_cascade import ExactPrefixCascade, exact_prefix_geometry


def test_longest_first_prunes_impossible_siblings_and_reuses_frontier():
    paths = (
        (1, 2, 3, 4, 5, 6, 7),
        (0, 2, 3, 4, 9, 6),
        (1, 2, 0, 4, 9, 6),
        (1, 2, 3, 4, 9, 6),
    )
    cascade = ExactPrefixCascade(paths, maximum_depth=7)

    first = cascade.next_stage()
    assert first.path_index == 0
    assert first.suffix == paths[0] and first.reused_prefix_tokens == 0
    update = cascade.observe(first, (1, 2, 3, 4, 9), accepted=4)
    assert update.frontier == (1, 2, 3, 4, 9)
    assert update.pruned_siblings == 2
    assert update.remaining_paths == 1 and not update.finished

    second = cascade.next_stage()
    assert second.path_index == 3
    assert second.reused_prefix_tokens == 5
    assert second.suffix == (6,)
    update = cascade.observe(second, (6, 7), accepted=1)
    assert update.finished and update.frontier == (1, 2, 3, 4, 9, 6, 7)
    assert cascade.pruned_siblings == 2
    assert cascade.next_stage() is None


def test_correction_token_prunes_every_wrong_prefix_without_another_launch():
    cascade = ExactPrefixCascade(((1, 2, 3), (1, 2, 4), (0, 2, 9)))
    stage = cascade.next_stage()
    update = cascade.observe(stage, (1, 8), accepted=1)
    assert update.finished
    assert update.frontier == (1, 8)
    assert update.pruned_siblings == 2
    assert cascade.next_stage() is None


def test_cascade_rejects_non_target_authoritative_observations():
    cascade = ExactPrefixCascade(((1, 2, 3),))
    stage = cascade.next_stage()
    with pytest.raises(ValueError, match="target observation"):
        cascade.observe(stage, (9,), accepted=1)


class KVCache:
    def __init__(self, offset):
        self.offset = offset


class RotatingKVCache:
    def __init__(self, offset, max_size):
        self.offset = offset
        self.max_size = max_size


def test_exact_geometry_requires_one_revision_bound_offset_and_window():
    cache = [KVCache(17), RotatingKVCache(17, 4096)]
    compatible = exact_prefix_geometry(
        cache,
        expected_layers=2,
        expected_rotating_layers=1,
        sliding_window=4096,
    )
    assert compatible == {
        "compatible": True,
        "offset": 17,
        "layers": 2,
        "rotating_layers": 1,
        "sliding_window": 4096,
        "attention_math_attested": False,
    }
    assert exact_prefix_geometry(
        [KVCache(17), RotatingKVCache(18, 4096)],
        expected_layers=2,
        expected_rotating_layers=1,
        sliding_window=4096,
    ) == {"compatible": False, "reason": "layer_offset_mismatch"}
    assert exact_prefix_geometry(
        [KVCache(17), RotatingKVCache(17, 2048)],
        expected_layers=2,
        expected_rotating_layers=1,
        sliding_window=4096,
    ) == {"compatible": False, "reason": "sliding_window"}
    assert exact_prefix_geometry(
        [KVCache(17), KVCache(17)],
        expected_layers=2,
        expected_rotating_layers=1,
        sliding_window=4096,
    ) == {"compatible": False, "reason": "rotating_layer_count"}


def test_north_owns_prefill_and_keeps_cascade_unselected():
    adapter = object.__new__(NorthMiniCodeAdapter)
    assert adapter.prefill_step_default() == 2048
    policy = adapter.prefix_cascade_policy()
    assert policy["planner_implemented"] is True
    assert policy["serving_implemented"] is False
    assert policy["execution_enabled"] is False
    assert policy["qualified"] is policy["selected"] is policy["observed_used"] is False
    assert policy["blockers"] == (
        "ordinary_b1_full_attention_mask_parity",
        "rotating_cache_commit_equivalence",
        "ordinary_decode_continuation_parity",
    )
