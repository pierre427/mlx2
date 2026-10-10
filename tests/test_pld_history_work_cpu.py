"""Recent-segment receipts must not copy the whole response each round."""

import pytest
from test_external_history_work_cpu import CountedHistory
from test_pld_live import _PatternModel

from mlx2.runtime.models.cache import KVCache
from mlx2.runtime.pld import PromptLookupBatchGenerator
from mlx2.runtime.prompt_lookup import RecentCommittedSegmentStore


@pytest.mark.parametrize("enabled", [False, True])
def test_recent_response_receipts_only_read_the_configured_tail(enabled):
    limit = 6
    batch = PromptLookupBatchGenerator(
        _PatternModel(), prefill_step_size=16,
        prompt_lookup={
            "num_draft": 2, "ngram_min": 2, "ngram_max": 2,
            "adaptive": False, "recent_committed_segments": enabled,
            "recent_committed_response_tokens": limit,
        },
        recent_source_store=RecentCommittedSegmentStore() if enabled else None,
    )
    try:
        uid = batch.insert(
            [[1, 2, 1, 2]], max_tokens=[48], caches=[[KVCache()]],
            pld_source_scopes=[("model", "tenant")],
        )[0]
        _, responses = batch.next()
        assert not responses
        lane = batch.lanes[uid]
        history = lane.lookup_history = CountedHistory(lane.lookup_history)
        emitted = []
        rounds = 0
        while uid in batch.lanes:
            _, responses = batch.next()
            rounds += 1
            for response in responses:
                emitted.append(response.token)
                if enabled:
                    assert response.pld_committed_response_tokens == tuple(emitted[-limit:])
                else:
                    assert not hasattr(response, "pld_committed_response_tokens")
        assert len(emitted) == 48
        assert history.copied <= (rounds * limit if enabled else 0)
    finally:
        batch.close()
