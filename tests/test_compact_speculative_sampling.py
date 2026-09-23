"""Dense-oracle checks for compact pairwise proposal verification (CPU only)."""
import numpy as np
import pytest

from mlx2.runtime.speculative_sampling import (
    FLyVerificationPolicy,
    RequestRNG,
    verify_compact_proposals,
    verify_proposals,
)


@pytest.mark.parametrize("vocab,candidates,depth", [(7, 2, 1), (32, 4, 3), (257, 16, 5), (65536, 32, 3)])
@pytest.mark.parametrize("greedy", [False, True])
@pytest.mark.parametrize("fly", [False, True])
def test_compact_matches_dense_result_and_rng(vocab, candidates, depth, greedy, fly):
    policy = FLyVerificationPolicy(enabled=True, entropy_threshold=0, window=1) if fly else None
    for seed in range(20):
        source = np.random.default_rng(seed)
        ids = [source.choice(vocab, candidates, replace=False) for _ in range(depth)]
        q = []
        tokens = []
        dense = []
        for row_ids in ids:
            row = source.random(candidates).astype(np.float32)
            row = (row / row.sum()).astype(np.float32)
            if greedy:
                row[:] = 0
                row[int(source.integers(candidates))] = 1
            proposed = int(row_ids[int(source.choice(candidates, p=row / row.sum()))])
            tokens.append(proposed)
            q.append(row)
            law = np.zeros(vocab, dtype=np.float64)
            law[row_ids] = row
            dense.append(law)
        targets = [source.random(vocab) for _ in range(depth + 1)]
        left, right = RequestRNG(seed + 1000), RequestRNG(seed + 1000)
        oracle = verify_proposals(tokens, dense, targets, left, fly_verification=policy)
        compact = verify_compact_proposals(tokens, ids, q, targets, right, fly_verification=policy)
        assert (compact.accepted, compact.emitted, compact.rejected, compact.relaxed_accepts) == (
            oracle.accepted, oracle.emitted, oracle.rejected, oracle.relaxed_accepts
        )
        assert right.snapshot() == left.snapshot()
        assert len(compact.target_probabilities) == len(oracle.target_probabilities)
        for actual, expected in zip(compact.target_probabilities, oracle.target_probabilities):
            np.testing.assert_array_equal(actual, expected)


def test_compact_validates_before_rng_draws():
    rng = RequestRNG(9)
    with pytest.raises(ValueError, match="Invalid probability distribution"):
        verify_compact_proposals([1], [[1, 2]], [[1, np.nan]], [[1, 1, 1]] * 2, rng)
    assert rng.draws == 0
    with pytest.raises(ValueError, match="Proposed token has zero proposal probability"):
        verify_compact_proposals([0], [[1, 2]], [[0.5, 0.5]], [[1, 1, 1]] * 2, rng)
    assert rng.draws == 0
    with pytest.raises(ValueError, match="Proposed token has zero proposal probability"):
        verify_proposals([0], [[0, 0.5, 0.5]], [[1, 1, 1]] * 2, rng)
    assert rng.draws == 0
    with pytest.raises(ValueError, match="Vocabulary mismatch"):
        verify_compact_proposals([1], [[1, 1]], [[0.5, 0.5]], [[1, 1, 1]] * 2, rng)
    assert rng.draws == 0


@pytest.mark.parametrize("acceptance_uniform", [np.nextafter(0.25, 0), 0.25])
def test_compact_matches_dense_at_acceptance_boundary(acceptance_uniform):
    class FixedRNG(RequestRNG):
        def __init__(self):
            super().__init__(0)
            self.values = iter((acceptance_uniform, 0.1))

        def uniform(self):
            self.draws += 1
            return next(self.values)

    left, right = FixedRNG(), FixedRNG()
    target = [np.array([0.25, 0.75]), np.array([0.3, 0.7])]
    oracle = verify_proposals([0], [np.array([1.0, 0.0])], target, left)
    compact = verify_compact_proposals([0], [[0]], [[1.0]], target, right)
    assert (compact.accepted, compact.emitted, compact.rejected) == (
        oracle.accepted, oracle.emitted, oracle.rejected
    )
    assert left.draws == right.draws == 2
