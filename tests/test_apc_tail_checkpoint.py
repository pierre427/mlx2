"""Opt-in trimmed-tail APCv2 checkpoint (``apc_tail_checkpoint``).

An agent turn that rewrites the tail of the previous turn's last message
diverges just below the previous prompt's replay boundary.  A hybrid cache
cannot cut back, so without a state near there the turn re-prefills from the
last rolling checkpoint.  Design references: incoai/splash#784,
jundot/omlx#5251.  CPU tiny-checkpoint tests only; no GPU A/B has run.
"""

import pytest

from mlx2.runtime.interior_placement import cold_prefill_cuts, trimmed_tail_position
from mlx2.serving import apc_tail_checkpoint_policy

from test_apc_hits_hybrid_gdn_self_mtp import host  # noqa: F401 - fixture
from test_apc_interior_placement import (
    CASES,
    _ChatTemplateTokenizer,
    _chat,
    _chat_engine,
    _words,
)


def test_policy_is_off_by_default_and_validated():
    assert apc_tail_checkpoint_policy(None) == {"tokens": 0, "min_gain_tokens": 512}
    assert apc_tail_checkpoint_policy({"tokens": 256}) == {"tokens": 256, "min_gain_tokens": 512}
    for bad in ({"tokens": 8}, {"tokens": True}, {"tokens": -1}, {"min_gain_tokens": -1},
                {"tokens": 256, "other": 1}, "on"):
        with pytest.raises(ValueError):
            apc_tail_checkpoint_policy(bad)


def test_tail_position_geometry():
    assert trimmed_tail_position(10_000, tokens_before=256) == 9_743
    # Never at or below the cached prefix / media end.
    assert trimmed_tail_position(10_000, tokens_before=256, floor_tokens=9_743) is None
    assert trimmed_tail_position(200, tokens_before=256) is None
    # A shallower planned state within min_gain makes the capture redundant;
    # one above the tail does not count.
    assert trimmed_tail_position(
        10_000, tokens_before=256, shallower=(9_500, 9_990), min_gain_tokens=512
    ) is None
    assert trimmed_tail_position(
        10_000, tokens_before=256, shallower=(8_192, 9_990), min_gain_tokens=512
    ) == 9_743


def test_tail_is_a_cold_cut_whatever_the_gain():
    tokens = list(range(1, 3_001))
    policy = {"count": 0, "min_stride": 1}
    assert cold_prefill_cuts(tokens, policy=policy) == ()
    assert cold_prefill_cuts(tokens, policy=policy, tail_tokens=256) == (2_743,)


@pytest.mark.parametrize(("factory", "mtp"), CASES)
def test_trimmed_tail_turn_resumes_at_the_tail_checkpoint(host, factory, mtp):
    model, vocab = factory()
    tail_tokens = 24
    system = {"role": "system", "content": _words(vocab, 1, 60)}
    user = _words(vocab, 2, 20).split()
    first = [system, {"role": "user", "content": " ".join(user)}]
    template = _ChatTemplateTokenizer(vocab, keeps_think_block=True)
    prompt1 = template.apply_chat_template(first, add_generation_prompt=True)
    results = {}
    for name, policy in (
        ("off", None),
        ("on", {"apc_tail_checkpoint": {"tokens": tail_tokens, "min_gain_tokens": 0}}),
    ):
        engine = _chat_engine(model, vocab, keeps_think_block=True, mtp=mtp, policy=policy)
        try:
            out, _receipt, _job = _chat(engine, first)
            # The agent trims the last 11 words of its previous message, then
            # sends the reply and a new turn.
            second = [
                system,
                {"role": "user", "content": " ".join(user[:-11])},
                {"role": "assistant", "content": " ".join(map(str, out))},
                {"role": "user", "content": _words(vocab, 3, 20)},
            ]
            prompt2 = template.apply_chat_template(second, add_generation_prompt=True)
            out2, receipt, job = _chat(engine, second)
            results[name] = (int(job.cached_tokens or 0), out2, receipt, dict(engine.counts))
        finally:
            engine.close()
    divergence = next(i for i, (a, b) in enumerate(zip(prompt1, prompt2)) if a != b)
    tail = len(prompt1) - 1 - tail_tokens
    assert tail <= divergence
    off_cached, off_out, _, off_counts = results["off"]
    on_cached, on_out, on_receipt, on_counts = results["on"]
    assert off_cached < tail
    assert "apc_interior_positions_planned_trimmed_tail" not in off_counts
    assert on_counts["apc_interior_positions_planned_trimmed_tail"] >= 1
    assert on_cached == tail
    assert on_receipt.get("cache_checkpoint_role") == "interior_checkpoint"
    # Greedy output from the tail restore equals the reference engine's.
    assert on_out == off_out
