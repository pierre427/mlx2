"""Inkling 2 DSpark serves several lanes (tiny model, CPU).

The 2026-11-21 B1-only refusal predated batched ring verify
(HybridVerifyRows ``batched_rings``); lifted 2026-11-27 (Pierre).  Greedy
lanes verified together equal ordinary decode (wrapped and unwrapped rings,
ragged prompts, lanes joining mid-flight), and sampled lanes are exact with
themselves across widths: B1 and B4 outputs of the same seeds are
token-identical on the host round and on the device round (token-wise and
block verification)."""
import mlx.core as mx
import pytest

from test_inkling2_dspark_cpu import _cpu, pair  # noqa: F401
from test_inkling2_dspark_predraft_cpu import agreeing  # noqa: F401


def _ordinary(target, prompt, count):
    cache, tokens, out = target.make_cache(), list(prompt), []
    for step in range(count):
        logits = target(mx.array([tokens if step == 0 else [tokens[-1]]]), cache=cache)
        tokens.append(int(mx.argmax(logits[0, -1]).item()))
        out.append(tokens[-1])
    return out


class _Key:
    def __init__(self, seed):
        self.key = [seed, 7]


def _serve(target, draft, prompts, *, stagger=False, temp=0.0, seeds=None, count=14, **options):
    from mlx2.runtime.drafters.inkling2_dspark_external import Inkling2DSparkExternal
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    batch = ExternalDraftBatchGenerator(
        target, draft_model=Inkling2DSparkExternal(draft, num_draft=7), binding="t",
        num_draft=7, prefill_step_size=16, completion_batch_size=4, **options)
    seeds = seeds or list(range(len(prompts)))

    def insert(ps, ss):
        extra = {"lane_rngs": [_Key(s) for s in ss]} if temp else {}
        return list(batch.insert(ps, max_tokens=[count] * len(ps),
                                 sampling_configs=[{"sampling_temp": temp}] * len(ps), **extra))

    try:
        uids = insert(prompts[:1], seeds[:1]) if stagger else insert(prompts, seeds)
        got = {}
        for step in range(2000):
            if stagger and step == 3:
                uids += insert(prompts[1:], seeds[1:])
            _, responses = batch.next()
            for r in responses:
                got.setdefault(r.uid, []).append(r.token)
            if len(uids) == len(prompts) and not batch.lanes:
                break
        return [got[u] for u in uids], dict(batch.scheduler_stats)
    finally:
        batch.close()


def _prompts(seed, lengths):
    mx.random.seed(seed)
    return [[int(t) for t in mx.random.randint(1, 400, (n,)).tolist()] for n in lengths]


@pytest.mark.parametrize("lengths,stagger", [((24, 9), False), ((30, 12, 20, 41), False),
                                             ((24, 9, 17), True)])
@pytest.mark.parametrize("options", [
    {}, {"prefill_allocator_reclaim": True, "ready_drain": "all"},
    {"prefill_allocator_reclaim": True, "ready_drain": "all", "predraft": True}],
    ids=["engine", "adapter_defaults", "predraft"])
def test_greedy_lanes_verified_together_equal_ordinary(agreeing, lengths, stagger, options):
    # ``agreeing`` accepts drafts (``pair``'s random weights never do), so lanes
    # accept different lengths, trim their rings and drain several tokens.
    target, draft = agreeing
    prompts = _prompts(7, lengths)
    got, stats = _serve(target, draft, prompts, stagger=stagger, **options)
    assert got == [_ordinary(target, p, 14) for p in prompts]
    assert stats.get("external_cohort_target_batched_standard_verify_rounds", 0) > 0
    assert stats["accepted_proposals"] > 0
    if options.get("predraft"):
        assert stats["external_predraft_launches"] > 0
        if len(lengths) == 2 and not stagger:
            # Wider cohorts run the shortest lane's depth, which discards
            # pre-drafts sized from a longer lane's budget (correct, slower).
            assert stats["external_predraft_adoptions"] > 0


SAMPLED = {"lazy_draft": True, "decide": True, "sampled_drafts": True}


@pytest.mark.parametrize("options", [
    {}, {"device_round": SAMPLED, "exact_verification": "token"}, {"device_round": SAMPLED}],
    ids=["host", "device_token", "device_block_default"])
@pytest.mark.parametrize("drain", [None, "all"])
def test_sampled_b1_equals_b4(agreeing, options, drain):
    target, draft = agreeing
    if drain:
        options = {**options, "ready_drain": drain}
    prompts, seeds = _prompts(9, (22, 31, 15, 27)), [1, 2, 3, 4]
    b4, stats = _serve(target, draft, prompts, temp=1.0, seeds=seeds, count=16, **options)
    b1 = [_serve(target, draft, [p], temp=1.0, seeds=[s], count=16, **options)[0][0]
          for p, s in zip(prompts, seeds)]
    assert b1 == b4
    assert stats["accepted_proposals"] > 0
    if drain == "all":
        # Without it a lane that accepted several tokens drains one per poll
        # and sits out the next verifies, so lanes stop verifying together
        # (the adapter applies "all" above one lane for this reason).
        assert stats.get("external_cohort_target_batched_standard_verify_rounds", 0) > 0
    if options.get("device_round"):
        lanes = "block" if "exact_verification" not in options else "ratio"
        assert stats[f"external_device_round_decide_{lanes}_lanes"] > 0
