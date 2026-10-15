"""Production external executor over tiny CPU tensors and a known target law.

This exercises propose/verify/commit and lane removal across several rounds;
it is stronger than a verifier-only oracle but is not artifact/GPU validation.
The actual tiny DFlash candidate selector and target/draft cache transactions
run. Only the target's final logits are replaced with a known Markov law so its
completed distribution has an independent analytic Markov reference.
"""

import types

import mlx.core as mx
import numpy as np
import pytest
from test_dflash_pair_select import generator, tiny

from mlx2.runtime.sample_utils import LaneRNG
from mlx2.runtime.speculative_sampling import RequestRNG

TARGET = np.array([[0.05, 0.15, 0.30, 0.50], [0.60, 0.10, 0.20, 0.10],
                   [0.10, 0.60, 0.10, 0.20], [0.40, 0.10, 0.45, 0.05]], dtype=np.float64)


def _known_target(monkeypatch, *, mode="block", width=32, negative_control=False):
    model, draft = tiny(vocab=4, top_k=4, block_size=4)
    forward = model.forward_with_taps

    def fixed_logits(*args, **kwargs):
        logits, features = forward(*args, **kwargs)
        if logits is not None:
            inputs = args[0]
            if logits.shape[1] == 1:
                inputs = inputs[:, -1:]
            logits = mx.array(np.log(TARGET), dtype=mx.float32)[inputs]
        return logits, features

    monkeypatch.setattr(model, "forward_with_taps", fixed_logits)
    # A deliberately different, stochastic draft law makes repeated rejected
    # output positions common. The actual candidate selector and its inverse
    # CDF still choose and publish every proposal.
    monkeypatch.setattr(draft, "_logits", lambda features: mx.zeros((*features.shape[:-1], 4)))
    draft.candidate_selector.hidden_projection.weight = mx.zeros_like(
        draft.candidate_selector.hidden_projection.weight
    )
    batch = generator(model, draft, exact_verification=mode,
                      pairwise_selection="host", completion_batch_size=width)
    if negative_control:
        propose, commit = batch._propose, batch._commit
        saved = {}

        def remember_stream(self, cohort, **kwargs):
            for lane in cohort:
                saved.setdefault(lane.uid, lane.rng.snapshot())
            return propose(cohort, **kwargs)

        def rewind_after_commit(self, cohort, *args, **kwargs):
            result = commit(cohort, *args, **kwargs)
            # Deliberately broken: an accepted transaction rewinds the stream
            # as though it had aborted. Subsequent rounds replay proposal and
            # verifier draws instead of receiving fresh randomness.
            for lane in cohort:
                lane.rng = RequestRNG(state=saved[lane.uid])
            return result

        batch._propose = types.MethodType(remember_stream, batch)
        batch._commit = types.MethodType(rewind_after_commit, batch)
    return batch


def _sample(batch, seeds, *, outputs=8):
    result, rounds = {}, []
    for start in range(0, len(seeds), batch.capacity):
        current = seeds[start:start + batch.capacity]
        uids = batch.insert([[1, 2]] * len(current), max_tokens=[outputs] * len(current),
                            lane_rngs=[LaneRNG(seed) for seed in current],
                            sampling_configs=[{"sampling_temp": 1.0, "emit_logprobs": False}] * len(current))
        for _ in range(40):
            _, responses = batch.next()
            for response in responses:
                result.setdefault(response.uid, []).append(response.token)
                if response.finish_reason:
                    rounds.append(response.speculative_receipt["external_rounds"])
            if not batch.lanes:
                break
        assert not batch.lanes
        assert all(len(result[uid]) == outputs for uid in uids)
    assert min(rounds) >= 2  # the observed production route really repeated
    assert batch.scheduler_stats["segmented_rollbacks"] > 0
    return np.asarray([result[uid] for uid in sorted(result)])


def _marginal_error(tokens):
    expected = _position_laws(tokens.shape[1])
    return max(float(np.max(np.abs(np.bincount(tokens[:, position], minlength=4) / len(tokens) - expected[position])))
               for position in range(tokens.shape[1]))


def _position_laws(count):
    law = np.array([0, 0, 1, 0], dtype=np.float64)  # prompt ends with token 2
    laws = []
    for _ in range(count):
        law = law @ TARGET
        laws.append(law)
    return np.asarray(laws)


def _joint_error(tokens):
    pairs = np.bincount((tokens[:, :-1] * 4 + tokens[:, 1:]).reshape(-1), minlength=16)
    joint = pairs.reshape(4, 4) / pairs.sum()
    expected = (_position_laws(tokens.shape[1])[:-1, :, None] * TARGET[None]).mean(axis=0)
    return float(np.max(np.abs(joint - expected)))


@pytest.mark.parametrize("mode", ["token", "block"])
def test_multiround_executor_preserves_complete_position_marginals(monkeypatch, mode):
    batch = _known_target(monkeypatch, mode=mode)
    try:
        tokens = _sample(batch, list(range(768)))
        assert _marginal_error(tokens) < 0.08
        # Inspect adjacent joint mass, not just marginals.
        assert _joint_error(tokens) < 0.04
    finally:
        batch.close()


def test_multiround_oracle_rejects_successful_commit_rng_rewind(monkeypatch):
    batch = _known_target(monkeypatch, negative_control=True)
    try:
        tokens = _sample(batch, list(range(768)))
        errors = (_marginal_error(tokens), _joint_error(tokens))
        assert errors[0] > 0.08 or errors[1] > 0.04, errors
    finally:
        batch.close()


def test_request_stream_is_unchanged_by_batch_order_or_neighbour_completion(monkeypatch):
    seeds = list(range(16))
    outputs = []
    for order, width in ((seeds, 4), (list(reversed(seeds)), 7)):
        batch = _known_target(monkeypatch, width=width)
        try:
            tokens = _sample(batch, order)
            outputs.append({seed: tuple(row) for seed, row in zip(order, tokens, strict=True)})
        finally:
            batch.close()
    assert outputs[0] == outputs[1]
