"""CPU/static gates for the Qwen3.8 -> Qwen4 strategy transfer."""

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


def test_accepted_prefix_prunes_invalid_siblings_and_reuses_only_the_suffix():
    paths = [(1, 2, 3, 4), (1, 2, 9), (1, 7, 8, 9), (6, 7)]
    first = next_cascade_stage(paths)
    assert first.path == (1, 2, 3, 4)
    assert first.suffix == first.path

    second = next_cascade_stage(paths, (1, 2), attempted=(0,))
    assert second.path == (1, 2, 9)
    assert second.suffix == (9,)
    assert second.viable_indices == (2,)
    assert set(second.pruned_indices) == {1, 3}


def test_no_sibling_survives_a_nonexistent_prefix():
    assert next_cascade_stage([(1, 2), (1, 3)], (9,)) is None


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        (1, 512),
        (32768, 512),
        (32769, 2048),
        (65536, 2048),
        (65537, 8192),
        (262144, 8192),
    ],
)
def test_generic_prompt_autoscale_uses_the_staged_schedule(prompt, expected):
    assert prompt_length_prefill_step(prompt) == expected


def test_qwen4_adapter_owns_prefill_and_refuses_unproved_multirow_state_reuse():
    from mlx2.adapters.flash_next import FlashNextAdapter
    from mlx2.adapters.flash_next_policy import FlashNextPolicy

    adapter = object.__new__(FlashNextAdapter)
    adapter.policy = FlashNextPolicy()
    assert adapter.prefill_step_default() == 8192
    contract = adapter.exact_prefix_cascade_contract()
    assert contract["verification_order"] == "longest_first"
    assert contract["invalid_sibling_pruning"] is True
    assert contract["accepted_prefix_state"] == "canonical_ordinary_replay"
    assert contract["transactional_multirow_state_reuse"] is False
    assert contract["qualified"] is contract["selected"] is False
    stage = adapter.plan_exact_prefix_cascade(
        [(1, 2, 3), (1, 7), (1, 2, 9)], (1, 2), attempted=(0,)
    )
    assert stage.path == (1, 2, 9) and stage.suffix == (9,)


def test_non_qwen4_flash_next_subclasses_do_not_inherit_the_cascade_contract():
    from mlx2.adapters.flash_next import FlashNextAdapter

    adapter = object.__new__(FlashNextAdapter)
    assert adapter.exact_prefix_cascade_contract() is None
    with pytest.raises(ValueError, match="not declared"):
        adapter.plan_exact_prefix_cascade([(1, 2)])


@pytest.mark.parametrize("paths", [[], [()], [(1, -1)], [(True, 2)]])
def test_invalid_cascade_paths_fail_closed(paths):
    with pytest.raises(ValueError):
        longest_first_paths(paths)
