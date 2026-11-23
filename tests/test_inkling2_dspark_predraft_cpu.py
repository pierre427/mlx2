"""DSpark pre-drafting (``ExternalDraftBatchGenerator(predraft=True)``), CPU only.

The next chain block is queued after the verify forward and before the host
reads acceptance, on a copy of the lane's draft plane, assuming the whole
block commits; the next round adopts it only at that exact boundary.  These
tests pin: greedy output equal to pre-drafting off and to ordinary decode
with the rings wrapped, every round's proposals identical to the off run,
adoption and discard both observed, sampled lanes untouched, and the APCv2
interior-checkpoint resume.
"""

import mlx.core as mx
import pytest

from test_inkling2_dspark_cpu import TEXT, _config, _cpu, _hidden, _random, pair  # noqa: F401

from mlx2.runtime.drafters.inkling2_dspark import Inkling2DSparkConfig, Inkling2DSparkDraftModel
from mlx2.runtime.drafters.inkling2_dspark_external import Inkling2DSparkExternal
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator, _block_row
from mlx2.runtime.models.inkling2 import Model, TextConfig

COUNTERS = ("launches", "adoptions", "discards", "sampled_skips")


@pytest.fixture(scope="module")
def agreeing():
    """The random tiny pair never accepts a draft, so a pre-draft is never
    adopted.  Keeping one LM-head row (token 17) makes every target and
    draft row choose between 17 and 0 (softcapped, ties to the lowest id),
    so blocks are often fully accepted and sometimes rejected, while the
    target and draft features stay those of the random weights."""
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    target = Model(TextConfig.from_dict(TEXT))
    _random(target, 0)
    keep = mx.zeros((target.lm_head.weight.shape[0], 1))
    keep[17] = 1
    target.lm_head.weight = target.lm_head.weight * keep
    mx.eval(target.parameters())
    draft = Inkling2DSparkDraftModel(Inkling2DSparkConfig.from_hf(_config()))
    _random(draft, 1)
    draft.bind(target)
    mx.set_default_device(old)
    return target, draft


def _ordinary(target, prompt, steps):
    cache = target.make_cache()
    tokens, expect = list(prompt), []
    for step in range(steps):
        logits = target(mx.array([tokens if step == 0 else [tokens[-1]]]), cache=cache)
        tokens.append(int(mx.argmax(logits[0, -1]).item()))
        expect.append(tokens[-1])
    return expect


def _engine(target, draft, num_draft, predraft):
    return ExternalDraftBatchGenerator(
        target, draft_model=Inkling2DSparkExternal(draft, num_draft=num_draft),
        binding="test", num_draft=num_draft, prefill_step_size=16, predraft=predraft,
    )


def _record_proposals(batch):
    """Every verify round's proposal rows, in order."""
    seen = []
    verify = batch._verify

    def wrapped(cohort, blocks, logits):
        seen.append([None if b is None else list(_block_row(b, None)[0]) for b in blocks])
        return verify(cohort, blocks, logits)

    batch._verify = wrapped
    return seen


def _run(batch, uid):
    got = []
    for _ in range(600):
        _, responses = batch.next()
        got += [r.token for r in responses if r.uid == uid]
        if not batch.lanes:
            break
    return got


def _decode(target, draft, prompt, num_draft, predraft, steps, sampling=None):
    batch = _engine(target, draft, num_draft, predraft)
    try:
        proposals = _record_proposals(batch)
        uid = batch.insert([prompt], max_tokens=[steps],
                           sampling_configs=[sampling or {"sampling_temp": 0}])[0]
        got = _run(batch, uid)
        stats = dict(batch.scheduler_stats)
        return got, proposals, stats, dict(batch.draft.stats)
    finally:
        batch.close()


def _prompt(seed, length=40):
    mx.random.seed(seed)
    return [int(t) for t in mx.random.randint(1, 400, (length,)).tolist()]


def test_backend_predraft_on_a_copy_equals_draft_distributions(pair):
    """The drafter half: a pre-draft on a descriptor copy with a lazy anchor
    returns the tokens ``draft_distributions`` returns on the live plane,
    and leaves the live plane where it was."""
    from mlx2.runtime.cow_cache import snapshot_prompt_cache_descriptors

    _, draft, _ = pair
    backend = Inkling2DSparkExternal(draft, num_draft=7)
    lane = backend.make_cache()
    h = _hidden(1, 20, seed=31)
    backend.append_context(h[:, :9], backend.batch_caches([lane]))  # local window 4 wrapped
    for start, stop, anchor in ((9, 13, 17), (13, 14, 5), (14, 20, 250)):
        levels = [c.offset for c in lane]
        copy, _, _ = snapshot_prompt_cache_descriptors(lane, None, measure=False)
        lazy = backend.predraft(mx.array([anchor]), h[:, start:stop], backend.batch_caches([copy]), 7)
        assert [c.offset for c in lane] == levels
        want, _ = backend.draft_distributions([anchor], h[:, start:stop],
                                              backend.batch_caches([lane]), 7, None, [0])
        assert lazy.tolist() == want
        assert [c.offset for c in copy] == [c.offset for c in lane]
        tokens, laws = backend.adopt_predraft(lazy, stop - start)
        assert tokens == want[0] and [law.token for law in laws] == tokens


@pytest.mark.parametrize("num_draft", [1, 3, 7])
@pytest.mark.parametrize("weights", ["random", "agreeing"])
def test_predraft_greedy_equals_off_and_ordinary_with_rings_wrapped(
    pair, agreeing, weights, num_draft,
):
    """40-token prompt + 32 tokens: past the drafter's local window (4) and
    the target ring (8). Output equals pre-drafting off and ordinary decode;
    every round proposes exactly the tokens the off run proposed, adopted
    blocks included.  On the agreeing pair both adoption and discard occur
    at every depth; on the random pair every pre-draft is discarded."""
    target, draft = pair[:2] if weights == "random" else agreeing
    prompt = _prompt(101)
    steps = 32
    expect = _ordinary(target, prompt, steps)
    off, off_props, off_stats, off_draft = _decode(target, draft, prompt, num_draft, False, steps)
    on, on_props, on_stats, on_draft = _decode(target, draft, prompt, num_draft, True, steps)
    assert off == expect and on == expect
    assert on_props == off_props
    for key in ("accepted_proposals", "proposed_tokens", "external_rounds"):
        assert on_stats[key] == off_stats[key]
    assert on_draft["block_drafts"] == off_draft["block_drafts"]
    assert on_draft["context_rows"] == off_draft["context_rows"]
    assert not any(f"external_predraft_{k}" in off_stats for k in COUNTERS)
    launches = on_stats["external_predraft_launches"]
    adopted = on_stats["external_predraft_adoptions"]
    discarded = on_stats["external_predraft_discards"]
    assert launches > 0 and on_stats["external_predraft_sampled_skips"] == 0
    assert launches == adopted + discarded
    if weights == "agreeing":
        assert adopted > 0 and discarded > 0
    else:
        assert adopted == 0 and on_stats["accepted_proposals"] == 0


def test_predraft_leaves_sampled_lanes_untouched(agreeing):
    """Temperature 0.7, fixed seed (each fresh engine seeds lane 0's RNG
    with its uid): the sampled bonus token comes from the
    lane RNG after the host read, so the lane is skipped (counted) and its
    output and RNG schedule are those of pre-drafting off."""
    target, draft = agreeing
    prompt = _prompt(101)
    sampling = {"sampling_temp": 0.7}
    off, off_props, _, _ = _decode(target, draft, prompt, 3, False, 16, sampling)
    on, on_props, stats, _ = _decode(target, draft, prompt, 3, True, 16, sampling)
    assert on == off and on_props == off_props and len(on) == 16
    assert stats["external_predraft_launches"] == 0
    assert stats["external_predraft_sampled_skips"] > 0


def test_adapter_policy_option_is_boolean_and_default_off():
    from mlx2.adapters.inkling2 import PREDRAFT_OPTION, validate_policy

    assert validate_policy({})[0]["dspark_predraft"] is False
    assert validate_policy({PREDRAFT_OPTION: True})[0]["dspark_predraft"] is True
    with pytest.raises(TypeError):
        validate_policy({PREDRAFT_OPTION: 1})


def test_predraft_requires_a_capable_drafter(pair):
    target, draft, _ = pair

    class Plain(Inkling2DSparkExternal):
        predraft = None

    with pytest.raises(ValueError, match="predraft"):
        ExternalDraftBatchGenerator(target, draft_model=Plain(draft, num_draft=3),
                                    binding="test", num_draft=3, predraft=True)
    with pytest.raises(ValueError, match="boolean"):
        _engine(target, draft, 3, 1)


@pytest.mark.parametrize("boundary", [16, 32])
def test_predraft_interior_checkpoint_resumes_like_the_cold_prompt(agreeing, boundary):
    """APCv2 interior checkpoint on the DSpark route with pre-drafting on:
    the cold run and the resume from the mid-prefill checkpoint (target
    rings, conv state, DSpark draft sidecar) both equal ordinary decode."""
    from mlx2.runtime.external_speculative import ExternalDraftState

    target, draft = agreeing
    prompt = _prompt(101)
    expect = _ordinary(target, prompt, 12)
    cold, resumed = _engine(target, draft, 3, True), _engine(target, draft, 3, True)
    try:
        uid = cold.insert([prompt], max_tokens=[12], sampling_configs=[{"sampling_temp": 0}],
                          apc_interior_positions=[(boundary,)])[0]
        lane = cold.lanes[uid]
        while lane.anchor is None:
            cold._prefill(lane)
        (checkpoint,) = cold.pop_interior_checkpoints(uid)
        sidecar = checkpoint["cache_sidecar"]
        assert isinstance(sidecar, ExternalDraftState)
        sidecar.validate("test", boundary)
        assert _run(cold, uid) == expect
        assert cold.scheduler_stats["external_predraft_adoptions"] > 0

        rid = resumed.insert([prompt[boundary:]], max_tokens=[12],
                             sampling_configs=[{"sampling_temp": 0}],
                             caches=[checkpoint["target_cache"]], cache_states=[sidecar],
                             all_tokens=[checkpoint["tokens"]])[0]
        assert resumed.scheduler_stats["paired_cache_resumes"] == 1
        assert _run(resumed, rid) == expect
        assert resumed.scheduler_stats["external_predraft_adoptions"] > 0
    finally:
        resumed.close()
        cold.close()


def test_verify_failure_releases_uncommitted_predraft_and_retry_matches(agreeing, monkeypatch):
    target, draft = agreeing
    prompt = _prompt(101)
    batch = _engine(target, draft, 3, True)
    uid = batch.insert([prompt], max_tokens=[16], sampling_configs=[{"sampling_temp": 0}])[0]
    original = batch._verify

    def fail_after_launch(*args):
        assert uid in batch._predrafts
        raise RuntimeError("synthetic verify failure")

    got = []
    try:
        monkeypatch.setattr(batch, "_verify", fail_after_launch)
        with pytest.raises(RuntimeError, match="synthetic verify failure"):
            while True:
                _, responses = batch.next()
                got.extend(r.token for r in responses if r.uid == uid)
        assert uid not in batch._predrafts
        monkeypatch.setattr(batch, "_verify", original)
        got.extend(_run(batch, uid))
        assert got == _ordinary(target, prompt, 16)
        assert batch.scheduler_stats["external_predraft_launches"] == (
            batch.scheduler_stats["external_predraft_adoptions"]
            + batch.scheduler_stats["external_predraft_discards"]
        )
    finally:
        batch.close()
