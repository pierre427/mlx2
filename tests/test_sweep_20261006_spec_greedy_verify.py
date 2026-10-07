"""SPEC-06: greedy external chain verification by token compare (CPU)."""
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.drafters.draft_block import DraftBlock
from mlx2.runtime.external_speculative import (
    CompactDraftRow,
    ExternalDraftBatchGenerator,
    HostDraftRow,
)
from mlx2.runtime.speculative_sampling import (
    FLyVerificationPolicy,
    RequestRNG,
    verify_greedy_proposals,
    verify_proposals,
)


def _one_hot(index, vocab):
    law = np.zeros(vocab)
    law[index] = 1
    return law


@pytest.mark.parametrize("seed", range(40))
def test_greedy_verifier_matches_dense_verifier_and_rng_schedule(seed):
    rng = np.random.default_rng(seed)
    vocab, count = 6, int(rng.integers(0, 5))
    selected = rng.integers(0, vocab, count + 1).tolist()
    drafts = [s if rng.random() < 0.6 else int(rng.integers(0, vocab)) for s in selected[:count]]
    laws = []
    for token in drafts:
        q = rng.dirichlet(np.ones(vocab)) if rng.random() < 0.5 else np.zeros(vocab)
        q[token] += 0.5
        laws.append(q / q.sum())
    dense_rng, fast_rng = RequestRNG(seed), RequestRNG(seed)
    dense = verify_proposals(drafts, laws, [_one_hot(s, vocab) for s in selected], dense_rng)
    fast = verify_greedy_proposals(drafts, selected, fast_rng)
    assert (fast.accepted, fast.emitted, fast.rejected) == (
        dense.accepted, dense.emitted, dense.rejected
    )
    assert fast_rng.draws == dense_rng.draws
    assert fast_rng.uniform() == dense_rng.uniform()


def _generator():
    gen = ExternalDraftBatchGenerator.__new__(ExternalDraftBatchGenerator)
    gen.mx = mx
    gen.stops = set()
    gen.fly_verification = FLyVerificationPolicy()
    return gen


def _lane(temp, seed=3):
    return SimpleNamespace(
        uid=0, anchor=1, history=[0], processors=[], rng=RequestRNG(seed),
        sampling={"sampling_temp": temp, "emit_logprobs": True},
    )


def _blocks(drafts, vocab):
    ids = [[token, (token + 1) % vocab, (token + 2) % vocab] for token in drafts]
    probs = [[0.7, 0.2, 0.1] for _ in drafts]
    draft = DraftBlock(
        tokens=mx.array([drafts], dtype=mx.int32),
        cand_ids=mx.array([ids], dtype=mx.int32),
        cand_q=mx.array([probs], dtype=mx.float32),
        lengths=(len(drafts),),
    )
    compact = CompactDraftRow(
        list(drafts), np.array(ids, dtype=np.int64), np.array(probs), 3,
    )
    host = HostDraftRow(list(drafts), draft.dense_laws(vocab)[0])
    return {"draft_block": draft, "compact": compact, "host": host}


@pytest.mark.parametrize("kind", ["draft_block", "compact", "host"])
def test_greedy_verify_round_matches_dense_reference(kind):
    vocab = 16
    logits = mx.array(np.random.default_rng(5).normal(size=(1, 4, vocab)), dtype=mx.float32)
    argmax = np.asarray(mx.argmax(logits[0], axis=-1)).tolist()
    drafts = [argmax[0], argmax[1], (argmax[2] + 1) % vocab]
    block = _blocks(drafts, vocab)[kind]
    gen, lane = _generator(), _lane(0.0)
    (decision,) = gen._verify([lane], [block], logits)
    reference_rng = RequestRNG(3)
    laws = _blocks(drafts, vocab)["host"].laws
    reference = verify_proposals(
        drafts, laws, [_one_hot(s, vocab) for s in argmax], reference_rng
    )
    assert decision.accepted == reference.accepted == 2
    assert decision.emitted == list(reference.emitted)
    assert lane.rng.draws == reference_rng.draws
    # Published logprobs are the processed log-softmax rows, as before.
    assert decision.target_laws is None
    expected = np.asarray(logits[0, 0] - mx.logsumexp(logits[0, 0]))
    np.testing.assert_allclose(np.asarray(decision.response_logprobs[0]), expected, rtol=1e-6)


def test_greedy_verify_refuses_a_draft_its_law_could_not_draw():
    vocab = 16
    logits = mx.zeros((1, 2, vocab))
    block = _blocks([4], vocab)["compact"]
    block.candidate_probs = np.array([[0.0, 0.6, 0.4]])
    with pytest.raises(ValueError, match="zero proposal probability"):
        _generator()._verify([_lane(0.0)], [block], logits)


# ------------------------------------------- full q validation (codex P2)
def _malformed(kind, case, vocab):
    """``(block, dense_laws)`` with one malformed proposal row, or None."""
    drafts = [4, 5]
    blocks = _blocks(drafts, vocab)
    if kind == "host":
        laws = [np.array(law, dtype=np.float64) for law in blocks["host"].laws]
        if case == "negative":
            laws[1][0] = -0.25
        elif case == "nonfinite":
            laws[1][0] = np.nan
        elif case == "zero_sum":
            laws[1][:] = 0.0
        elif case == "shape":
            laws[1] = laws[1][:-1]
        elif case == "count":
            laws = laws[:1]
        else:
            return None
        return HostDraftRow(drafts, laws), laws
    ids = blocks["compact"].candidate_ids.copy()
    probs = blocks["compact"].candidate_probs.astype(np.float64).copy()
    if case == "negative":
        probs[1, 2] = -0.1
    elif case == "nonfinite":
        probs[1, 2] = np.inf
    elif case == "zero_sum":
        probs[1, :] = 0.0
    elif case == "duplicate":
        ids[1, 2] = ids[1, 1]
    elif case == "out_of_range":
        ids[1, 2] = vocab
    elif case == "negative_id":
        ids[1, 2] = -1
    elif case == "count":
        ids, probs = ids[:1], probs[:1]
    else:
        return None
    if kind == "compact":
        return CompactDraftRow(drafts, ids, probs, 3), None
    return DraftBlock(
        tokens=mx.array([drafts], dtype=mx.int32),
        cand_ids=mx.array(ids[None].astype(np.int32)),
        cand_q=mx.array(probs[None].astype(np.float32)),
        lengths=(len(drafts),),
    ), None


CASES = ["negative", "nonfinite", "zero_sum", "shape", "count",
         "duplicate", "out_of_range", "negative_id"]


@pytest.mark.parametrize("kind", ["host", "compact", "draft_block"])
@pytest.mark.parametrize("case", CASES)
def test_greedy_verify_refuses_every_law_the_dense_verifier_refuses(kind, case):
    vocab = 16
    made = _malformed(kind, case, vocab)
    if made is None or (kind == "draft_block" and case == "count"):
        pytest.skip("not representable for this row kind")
    block, _laws = made
    logits = mx.zeros((1, 3, vocab))
    lane = _lane(0.0)
    with pytest.raises(ValueError):
        _generator()._verify([lane], [block], logits)
    assert lane.rng.draws == 0
    # The sampled (non-greedy) verifier refuses the same row.
    if kind != "draft_block":
        with pytest.raises(ValueError):
            _generator()._verify([_lane(1.0)], [block], logits)
