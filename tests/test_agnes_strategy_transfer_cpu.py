"""CPU/static gates for the Agnes 3 Flash exact-prefix strategy transfer."""

import pytest

from mlx2.runtime.exact_prefix_cascade import (
    longest_first_paths,
    next_cascade_stage,
)
from mlx2.runtime.prefill_plan import prompt_length_prefill_step


def test_longest_first_is_stable_and_deduplicates_complete_paths():
    paths = [(1,), (2, 3, 4), (5, 6), (2, 3, 4), (7, 8)]
    assert longest_first_paths(paths) == (
        (2, 3, 4),
        (5, 6),
        (7, 8),
        (1,),
    )


def test_accepted_prefix_prunes_siblings_and_returns_only_uncomputed_suffix():
    paths = [(1, 2, 3, 4), (1, 2, 9), (1, 7, 8, 9), (6, 7)]
    first = next_cascade_stage(paths)
    assert first.path == (1, 2, 3, 4)
    assert first.suffix == first.path

    second = next_cascade_stage(paths, (1, 2), attempted=(0,))
    assert second.path == (1, 2, 9)
    assert second.suffix == (9,)
    assert second.viable_indices == (2,)
    assert set(second.pruned_indices) == {1, 3}


def test_agnes_text_owns_prefill_while_vision_declines_to_generic_autoscale():
    from mlx2.adapters.agnes_3_flash import Agnes3FlashAdapter
    from mlx2.adapters.agnes_vision import AgnesVisionCandidateAdapter

    assert object.__new__(Agnes3FlashAdapter).prefill_step_default() == 2048
    assert object.__new__(AgnesVisionCandidateAdapter).prefill_step_default() is None
    assert prompt_length_prefill_step(32768) == 512
    assert prompt_length_prefill_step(32769) == 2048
    assert prompt_length_prefill_step(65537) == 8192


def test_agnes_prefix_reuse_requires_exact_text_hybrid_geometry():
    from mlx2.adapters.agnes_3_flash import CACHE_LAYOUT, Agnes3FlashAdapter

    adapter = object.__new__(Agnes3FlashAdapter)
    contract = adapter.exact_prefix_cascade_contract()
    assert contract["verification_order"] == "longest_first"
    assert contract["invalid_sibling_pruning"] is True
    assert contract["shared_prefix_reuse"] == (
        "suffix_only_without_common_token_replay"
    )
    assert contract["conditional_generation_prefill"] is False
    assert contract["vision_prefill"] is False
    assert contract["qualified"] is contract["selected"] is False

    with pytest.raises(ValueError, match="exact text hybrid cache geometry"):
        adapter.plan_exact_prefix_cascade(
            [(1, 2, 3), (1, 2, 9)], (1, 2), attempted=(0,)
        )
    with pytest.raises(ValueError, match="text-decoder only"):
        adapter.plan_exact_prefix_cascade(
            [(1, 2, 3)], state_scope="conditional_generation_vision"
        )

    stage = adapter.plan_exact_prefix_cascade(
        [(1, 2, 3), (1, 7), (1, 2, 9)],
        (1, 2),
        attempted=(0,),
        cache_layout=CACHE_LAYOUT,
        exact_state_geometry=True,
    )
    assert stage.path == (1, 2, 9)
    assert stage.suffix == (9,)


@pytest.mark.parametrize("paths", [[], [()], [(1, -1)], [(True, 2)]])
def test_invalid_cascade_paths_fail_closed(paths):
    with pytest.raises(ValueError):
        longest_first_paths(paths)
