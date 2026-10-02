"""Committed pool critic evidence survives an ordinary terminal token."""

import mlx.core as mx
import pytest
from test_external_continuation_pool_cpu import batch

from mlx2.adapters.proposal_path_sources import SourceContinuation


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    from mlx2.runtime import proposal_pool

    monkeypatch.setattr(
        proposal_pool,
        "_SHARED_RANKING_REGISTRY",
        proposal_pool.ProposalRankingRegistry(),
    )
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def full_pool_then_ordinary(monkeypatch):
    model, _base, draft, generator = batch()
    prompt = [1, 2, 3]
    cache = model.make_cache()
    mx.eval(model(mx.array([prompt[:-1]]), cache=cache))
    logits = model(mx.array([[prompt[-1]]]), cache=cache)[0, -1]
    first = int(mx.argmax(logits).item())
    vocab = model.args.vocab_size
    paths = [(first, token) for token in range(vocab)]
    paths += [((first + 1) % vocab, token) for token in range(15 - vocab)]
    assert len(paths) == 15
    monkeypatch.setitem(
        draft.providers,
        "external",
        lambda _context, limit: [
            SourceContinuation(
                path, (None, None), float(15 - index), "synthetic-full-prefix-coverage"
            )
            for index, path in enumerate(paths[:limit])
        ],
    )
    uid = generator.insert([prompt], max_tokens=[4])[0]
    lane = generator.lanes[uid]
    while lane.anchor is None:
        generator._prefill(lane)
    generator._round([lane])
    assert lane.generated == 3 and lane.external_rounds == 1
    before = draft.proposal_pool.receipt(lane.session_scope_hash)
    assert all(
        before["ranking_registry"][key]
        for key in ("global_counts", "model_counts", "session_counts")
    )
    return draft, generator, lane, before


def test_terminal_ordinary_receipt_retains_actual_committed_three_tier_counts(
    monkeypatch,
):
    draft, generator, lane, before = full_pool_then_ordinary(monkeypatch)
    try:
        generator._round([lane])
        final = lane.ready[-1]
        assert final.finish_reason == "length"
        assert final.speculative_receipt["current_execution"] == "ordinary_target"
        assert final.speculative_receipt["continuation_pool"]["observed_used"]
        ranking = final.speculative_receipt["continuation_pool"]["ranking"]
        assert ranking == before == draft.proposal_pool.receipt(lane.session_scope_hash)
        assert ranking["feedback_revision"] == 1
        assert all(
            ranking["ranking_registry"][key]
            for key in ("global_counts", "model_counts", "session_counts")
        )
        assert not draft.proposal_pool._pending
    finally:
        generator.close()


def test_receipt_query_is_read_only_and_diagnostic_failure_does_not_abort_terminal(
    monkeypatch,
):
    draft, generator, lane, before = full_pool_then_ordinary(monkeypatch)
    try:
        original = draft.proposal_pool.receipt
        for _ in range(3):
            assert (
                generator._continuation_receipt(lane)["continuation_pool"]["ranking"]
                == before
            )
        assert original(lane.session_scope_hash) == before

        def broken(_scope):
            raise RuntimeError("diagnostic receipt unavailable")

        monkeypatch.setattr(draft.proposal_pool, "receipt", broken)
        generator._round([lane])
        final = lane.ready[-1]
        assert final.finish_reason == "length"
        assert (
            final.speculative_receipt["continuation_pool"]["ranking_error"]
            == "diagnostic receipt unavailable"
        )
        assert draft.proposal_pool.feedback_revision == 1
        assert original(lane.session_scope_hash) == before
    finally:
        generator.close()
