import mlx.core as mx
import pytest

from mlx2.runtime.models.cache import KVCache, RotatingKVCache
from mlx2.runtime.pld import PromptLookupBatchGenerator

WINDOW = 8
PROMPT = [1, 2, 3] * 5  # 15 tokens: the 8-slot ring wraps during prefill


class _MixedCacheModel:
    """KVCache + stock RotatingKVCache; every 7th position breaks the cycle."""

    def __init__(self):
        self.forwards = 0

    def __call__(self, tokens, *, cache):
        self.forwards += 1
        start = int(cache[0].offset)
        values = tokens.astype(mx.float32)[:, None, :, None]
        cache[0].update_and_fetch(values, values)
        cache[1].update_and_fetch(values + 100.0, values + 200.0)
        positions = mx.arange(start, start + tokens.shape[1])[None]
        predicted = mx.where(positions % 7 == 0, 5, tokens % 3 + 1)
        return mx.where(
            mx.arange(8)[None, None, :] == predicted[..., None], 20.0, -20.0
        )


def _generator(model, **policy):
    generator = PromptLookupBatchGenerator(
        model,
        prefill_step_size=4,
        prompt_lookup={
            "num_draft": 4,
            "ngram_min": 2,
            "ngram_max": 3,
            "adaptive": False,
            # These tests compare the two per-lane rollback drivers.
            "batched_verify": False,
            **policy,
        },
    )
    generator.insert(
        [PROMPT],
        max_tokens=[40],
        caches=[[KVCache(), RotatingKVCache(max_size=WINDOW)]],
    )
    return generator


def _cache_state(generator):
    (lane,) = generator.lanes.values()
    plain, ring = lane.cache
    return {
        "plain_offset": int(plain.offset),
        "plain_keys": plain.keys[..., : plain.offset, :].tolist(),
        "ring_offset": int(ring.offset),
        "ring_idx": int(ring._idx),
        "ring_keys": ring.keys.tolist(),
        "ring_values": ring.values.tolist(),
    }


def _lockstep(reference, candidate):
    tokens = ([], [])
    finals = [None, None]
    while finals[0] is None:
        results = [generator.next() for generator in (reference, candidate)]
        for index, (_prompts, responses) in enumerate(results):
            tokens[index].extend(response.token for response in responses)
            for response in responses:
                if response.finish_reason:
                    finals[index] = response
        assert tokens[0] == tokens[1]
        if finals[0] is None:
            left, right = _cache_state(reference), _cache_state(candidate)
            assert left == right
            assert left["plain_offset"] == left["ring_offset"]
    assert finals[1] is not None
    return tokens[0], finals


def _assert_same_final(finals):
    left, right = (final.prompt_cache for final in finals)
    assert finals[0].all_tokens == finals[1].all_tokens
    for final in finals:
        plain, ring = final.prompt_cache
        assert plain.offset == ring.offset == len(final.all_tokens)
        assert not ring.speculating
    assert left[1]._idx == right[1]._idx
    assert left[1].keys.tolist() == right[1].keys.tolist()
    assert left[1].values.tolist() == right[1].values.tolist()
    assert (
        left[0].keys[..., : left[0].offset, :].tolist()
        == right[0].keys[..., : right[0].offset, :].tolist()
    )


def test_rotating_replay_is_default_off_and_validated():
    generator = _generator(_MixedCacheModel())
    assert generator.rotating_replay is False
    assert not generator.rotating_replay_policy.enabled
    with pytest.raises(ValueError, match="rotating_replay must be a boolean"):
        PromptLookupBatchGenerator.validate_policy({"rotating_replay": 1})
    with pytest.raises(ValueError, match="max_proposal_tokens"):
        PromptLookupBatchGenerator.validate_policy({"max_proposal_tokens": 0})


def test_rotating_replay_matches_copy_snapshot_on_wrapped_ring():
    reference_model, candidate_model = _MixedCacheModel(), _MixedCacheModel()
    reference = _generator(reference_model)
    candidate = _generator(candidate_model, rotating_replay=True)
    tokens, finals = _lockstep(reference, candidate)
    _assert_same_final(finals)

    assert 5 in tokens  # the cycle was broken, so some proposals were cut short
    off, on = reference.scheduler_stats, candidate.scheduler_stats
    assert off["pld_rotating_replay_rounds"] == 0
    assert off["pld_rotating_replay_replayed_tokens"] == 0
    assert on["pld_rollbacks"] == off["pld_rollbacks"] > 0
    assert on["pld_accepted"] == off["pld_accepted"] > 0
    assert on["pld_rotating_replay_rounds"] == on["pld_retrieval_cycles"] > 0
    assert on["pld_rotating_replay_replayed_tokens"] > 0
    assert on["pld_rotating_replay_refusals"] == 0
    # Exactly one replay forward per rolled-back round, shared by both caches.
    assert candidate_model.forwards == reference_model.forwards
    prefill = on["pld_prefill_rounds"]
    assert candidate_model.forwards == prefill + on["pld_cycles"] + on["pld_rollbacks"]


def test_rotating_replay_refusal_falls_back_to_copy_snapshot():
    reference = _generator(_MixedCacheModel())
    # A one-token bound refuses every multi-token proposal.
    candidate = _generator(
        _MixedCacheModel(), rotating_replay=True, max_proposal_tokens=1
    )
    tokens, finals = _lockstep(reference, candidate)
    _assert_same_final(finals)
    stats = candidate.scheduler_stats
    assert stats["pld_rotating_replay_refusals"] > 0
    assert stats["pld_rollbacks"] > 0
    assert (
        stats["pld_rotating_replay_refusals"] + stats["pld_rotating_replay_rounds"]
        == stats["pld_retrieval_cycles"]
    )
    assert len(tokens) == 40


def test_commit_time_replay_error_rebuilds_the_lane_instead_of_escaping(monkeypatch):
    from mlx2.runtime import pld, rotating_replay

    model = _MixedCacheModel()
    model.make_cache = lambda: [KVCache(), RotatingKVCache(max_size=WINDOW)]
    reference = _generator(_MixedCacheModel())
    candidate = _generator(model, rotating_replay=True)
    original = rotating_replay.RotatingReplayTransaction.commit
    failures = []

    def flaky(self, accepted, replay):
        if not failures:
            failures.append(True)
            self._restore()
            self.closed = True
            raise rotating_replay.RotatingReplayError("injected publication mismatch")
        return original(self, accepted, replay)

    monkeypatch.setattr(pld.RotatingReplayTransaction, "commit", flaky)
    tokens = ([], [])
    done = [False, False]
    while not all(done):
        for index, generator in enumerate((reference, candidate)):
            if done[index]:
                continue
            _prompts, responses = generator.next()  # must never raise
            tokens[index].extend(response.token for response in responses)
            done[index] = any(response.finish_reason for response in responses)
    assert failures and tokens[0] == tokens[1] and len(tokens[0]) == 40
    assert candidate.scheduler_stats["pld_rotating_replay_rebuilds"] == 1
    assert candidate.scheduler_stats["pld_recovery_checkpoint_restores"] == 1
    assert candidate.scheduler_stats["pld_recovery_full_rebuilds"] == 0
