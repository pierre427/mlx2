"""Exact target sampling and seed coupling across deterministic draft paths."""

import numpy as np
import pytest

from mlx2.runtime.speculative_sampling import (
    RequestRNG,
    verify_point_mass_proposals,
    verify_proposals,
)


def _law(history):
    # Conditional laws depend on committed history, including repeated IDs.
    values = np.array([1., 3., 2., 4.])
    values[sum(history) % len(values)] += 4
    return values / values.sum()


@pytest.mark.parametrize("depth", [0, 1, 2, 5])
@pytest.mark.parametrize("draft_token", [0, 1, 3])
def test_target_stream_is_independent_of_draft_path_and_round_partition(depth, draft_token):
    for seed in range(20):
        reference_rng = RequestRNG(seed)
        reference = []
        for _ in range(30):
            reference.append(reference_rng.sample(_law(reference)))
        rng = RequestRNG(seed)
        emitted = []
        while len(emitted) < len(reference):
            count = min(depth, len(reference) - len(emitted) - 1)
            drafts = [draft_token] * count
            laws = [_law(emitted + drafts[:position]) for position in range(count + 1)]
            block = verify_point_mass_proposals(drafts, laws, rng)
            assert block.accepted == len(block.emitted) - 1
            assert block.rejected == (block.accepted < count)
            emitted.extend(block.emitted)
        assert emitted == reference
        assert rng.draws == len(reference)
        assert rng.snapshot() == reference_rng.snapshot()


def test_rejection_schedule_changes_seeded_output_even_with_identical_target_laws():
    p = np.array([.2, .5, .3])
    def legacy(proposal):
        rng = RequestRNG(123)
        q = np.eye(3)[proposal]
        block = verify_proposals([proposal], [q], [p, p], rng)
        return block.emitted, rng.draws
    assert legacy(0) != legacy(1)
    left, right = RequestRNG(123), RequestRNG(123)
    first = verify_point_mass_proposals([0], [p, p], left)
    second = verify_point_mass_proposals([1], [p, p], right)
    assert first.emitted[0] == second.emitted[0]
    assert left.draws == len(first.emitted) and right.draws == len(second.emitted)


@pytest.mark.parametrize("tokens,targets", [
    ([0], [[.5, .5]]),
    ([], []),
    ([2], [[.5, .5], [.5, .5]]),
    ([0], [[.5, .5], [.5, .3, .2]]),
    ([0], [[.5, .5], [np.nan, 1]]),
    ([0], [[.5, .5], [0., 0.]]),
])
def test_all_laws_validate_before_any_random_draw(tokens, targets):
    rng = RequestRNG(1)
    before = rng.snapshot()
    with pytest.raises(ValueError):
        verify_point_mass_proposals(tokens, targets, rng)
    assert rng.snapshot() == before
