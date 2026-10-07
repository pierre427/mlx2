"""SPEC-10: prompt lookup on hybrid (GDN ``ArraysCache``) targets batches.

Qwen3.8 caches are recurrent ``ArraysCache`` planes beside KV planes.  PLD
used to accept only KV planes for its batched verify, so B4 ran as four
serial B1 rounds, latched lanes stayed in a speculation epoch (verify kernel
per token) and every round captured a recovery snapshot.
"""
import mlx.core as mx
import pytest
from test_segmented_mtp import _tiny_hybrid_mtp_model

from mlx2.runtime.models.cache import ArraysCache, KVCache
from mlx2.runtime.pld import PromptLookupBatchGenerator

PROMPTS = [
    [5, 6, 7, 8, 9, 10] * 6 + [5, 6, 7],
    [20, 21, 22, 23] * 8 + [20, 21],
    [40, 41, 42, 43, 44] * 6 + [40],
    [60, 61, 62] * 10 + [60],
]


def test_hybrid_lane_is_batchable():
    cache = [ArraysCache(size=2) if i % 4 != 3 else KVCache() for i in range(8)]
    assert PromptLookupBatchGenerator._batchable(cache)


def _greedy(model, prompt, count):
    cache = model.make_cache()
    logits = model(mx.array([prompt]), cache=cache)
    tokens = []
    for _ in range(count):
        token = int(mx.argmax(logits[0, -1]).item())
        tokens.append(token)
        logits = model(mx.array([[token]]), cache=cache)
    return tokens


class _Forwards:
    """Width and recurrent-epoch state of every decode-phase forward."""

    def __init__(self, model):
        self.model = model
        self.log = []

    def __call__(self, tokens, cache=None, **kwargs):
        recurrent = [c for c in cache if isinstance(c, ArraysCache)]
        self.log.append(
            (tokens.shape[0], tokens.shape[1], any(c.speculating for c in recurrent))
        )
        return self.model(tokens, cache=cache, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)


def _run(prompts, max_tokens, **policy):
    model = _tiny_hybrid_mtp_model()
    forwards = _Forwards(model)
    generator = PromptLookupBatchGenerator(
        forwards, completion_batch_size=4, prefill_step_size=64,
        prompt_lookup={"num_draft": 3, "ngram_min": 2, "ngram_max": 3, **policy},
    )
    generator.insert(prompts, max_tokens=[max_tokens] * len(prompts),
                     caches=[model.make_cache() for _ in prompts])
    tokens = {}
    try:
        for _ in range(500):
            _prompts, responses = generator.next()
            for response in responses:
                tokens.setdefault(response.uid, []).append(response.token)
            if not generator.lanes:
                break
    finally:
        generator.close()
    return model, [tokens[uid] for uid in sorted(tokens)], generator.scheduler_stats, forwards.log


@pytest.mark.parametrize("lanes", [1, 4])
def test_hybrid_pld_batches_and_matches_greedy_decode(lanes):
    model, outputs, stats, _log = _run(PROMPTS[:lanes], 16)
    for prompt, output in zip(PROMPTS, outputs):
        assert output == _greedy(model, prompt, 16)
    assert stats["pld_batched_rounds"] > 0
    assert stats["pld_batched_max_width"] == lanes
    assert stats["pld_proposed"] > 0 and stats["pld_rollbacks"] > 0


def test_proposal_free_rounds_are_plain_decode_steps_without_snapshots():
    _model, _outputs, stats, log = _run(PROMPTS, 12)
    decode = [entry for entry in log if entry[1] <= 4]  # anchor + num_draft
    single = [entry for entry in decode if entry[1] == 1]
    verify = [entry for entry in decode if entry[1] > 1]
    assert single and verify
    # One-token rounds run outside any rollback epoch; verify rounds in one.
    assert not any(speculating for _rows, _width, speculating in single)
    assert all(speculating for _rows, _width, speculating in verify)
    # A recovery snapshot at prefill end, then only before proposing rounds.
    assert stats["pld_recovery_checkpoint_captures"] <= 4 + stats["pld_retrieval_cycles"]


def test_latched_lanes_leave_the_verify_epoch():
    """Latched (ordinary) lanes decode in one plain B-wide step per round."""
    _model, _outputs, stats, log = _run(
        PROMPTS, 8, deferred_admission=True, admission_window=1000
    )
    decode = [entry for entry in log if entry[1] <= 4]
    assert stats["pld_proposed"] == 0
    # Lanes join as their prefills finish, then all four share each step.
    assert all(width == 1 and not speculating for _rows, width, speculating in decode)
    assert sum(rows == 4 for rows, _width, _speculating in decode) >= 4
    assert stats["pld_recovery_checkpoint_captures"] == 4
