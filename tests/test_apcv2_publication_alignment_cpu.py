"""APCv2 refuses, at save time, a checkpoint whose tokens and state disagree.

ml-explore/mlx-lm PR #2104 (read only): after speculative decoding the last
generated token was never fed through the target, so a prompt cache was
saved one entry short of its key and reuse trimmed one token too many.  In
APCv2 a hit resumes after every stored token, so the same shape would skip a
token.  ``publication_alignment_refusal`` binds the saved token count to every
positional plane and refuses an
open speculation/rollback transaction (rejected draft rows may still be in
state).  A refusal costs the checkpoint, never the request.  CPU only.
"""

from collections import Counter

import mlx.core as mx

from mlx2.runtime.apc_v2 import (
    APCKey,
    APCv2,
    MTPAPCSidecar,
    publication_alignment_refusal,
)
from mlx2.runtime.models.cache import ArraysCache, KVCache
from mlx2.serving import ServingEngine

TOKENS = [17, 4, 91, 33, 8, 62, 5, 70]


def _kv(length, seed=0):
    cache = KVCache()
    values = mx.arange(seed, seed + length, dtype=mx.float32).reshape(1, 1, length, 1)
    cache.update_and_fetch(values, values)
    mx.eval(cache.state)
    return cache


def _recurrent():
    cache = ArraysCache(1)
    cache[0] = mx.ones((1, 4), dtype=mx.float32)
    mx.eval(cache.state)
    return cache


def _sidecar(covered):
    draft = [_kv(max(covered - 1, 1), seed=50)]
    return MTPAPCSidecar((draft, mx.ones((1, 1, 4), dtype=mx.float32)), covered)


def _apc():
    return APCv2(max_size=8, layout_name="test-alignment-v1")


def test_consistent_kv_and_hybrid_caches_publish():
    apc = _apc()
    key = APCKey("aligned")
    assert publication_alignment_refusal([_kv(5)], 5) is None
    assert apc.store(key, TOKENS[:5], [_kv(5)]).stored
    assert apc.store(key, TOKENS[:6], [_recurrent(), _kv(6), _kv(6)]).stored
    assert apc.alignment_refusals == {} and apc.last_alignment_refusal is None
    hit = apc.lookup(key, TOKENS)
    assert hit.hit and hit.cached_tokens == 6
    hit.cache.close()
    apc.clear()


def test_kv_one_token_short_of_its_key_is_refused_with_a_receipt():
    """The #2104 shape: 5 tokens saved, the target holds 4 (bonus not fed)."""
    apc = _apc()
    key = APCKey("short")
    result = apc.store(key, TOKENS[:5], [_kv(4)])
    assert result.stored is False
    assert result.reason == "publication_alignment:position_mismatch"
    assert apc.boundary_rejections == 1
    assert apc.alignment_refusals == {"position_mismatch": 1}
    receipt = apc.last_alignment_refusal
    assert receipt["event"] == "apcv2_publication_alignment_refused"
    assert (receipt["layer"], receipt["tokens"], receipt["position"]) == (0, 5, 4)
    assert apc._storage_stats["alignment_refusals"] == {"position_mismatch": 1}
    assert not apc.lookup(key, TOKENS).hit
    apc.clear()


def test_hybrid_with_one_short_attention_layer_is_refused():
    """Every positional layer must agree, not only the first or the deepest."""
    apc = _apc()
    result = apc.store(APCKey("hybrid"), TOKENS[:6], [_recurrent(), _kv(6), _kv(5)])
    assert result.stored is False
    assert apc.last_alignment_refusal["layer"] == 2
    assert apc.last_alignment_refusal["position"] == 5
    apc.clear()


def test_open_speculation_in_recurrent_state_is_refused_not_raised():
    """Rejected draft rows may still be in a speculating recurrent plane."""
    apc = _apc()
    recurrent = _recurrent()
    recurrent.start_speculation()
    result = apc.store(APCKey("speculating"), TOKENS[:4], [recurrent, _kv(4)])
    assert result.stored is False
    assert result.reason == "publication_alignment:uncommitted_transaction"
    assert "speculation transaction is active" in apc.last_alignment_refusal["detail"]
    recurrent.stop_speculation()
    assert apc.store(APCKey("speculating"), TOKENS[:4], [recurrent, _kv(4)]).stored
    apc.clear()


def test_purely_recurrent_state_is_bound_only_by_its_transaction_state():
    """No recurrent plane records a token count (documented gap): a committed
    one publishes, an open speculation does not."""
    recurrent = _recurrent()
    assert publication_alignment_refusal([recurrent], 3) is None
    recurrent.start_speculation()
    assert publication_alignment_refusal([recurrent], 3).reason == (
        "uncommitted_transaction"
    )


def test_a_non_integer_position_is_indeterminate():
    cache = _kv(3)
    cache.offset = mx.array([3])
    refusal = publication_alignment_refusal([cache], 3)
    assert refusal.reason == "position_indeterminate" and refusal.layer == 0


def test_misaligned_draft_sidecar_is_dropped_and_the_target_still_publishes():
    apc = _apc()
    key = APCKey("sidecar")
    assert apc.store(key, TOKENS[:4], [_kv(4)], sidecar=_sidecar(3)).stored
    assert apc.alignment_refusals == {"sidecar_coverage_mismatch": 1}
    assert apc.last_alignment_refusal["position"] == 3
    hit = apc.lookup(key, TOKENS[:6])
    assert hit.hit and hit.sidecar is None and hit.cached_tokens == 4
    hit.cache.close()
    apc.clear()


def test_serving_counts_the_refusal_and_does_not_raise():
    engine = ServingEngine.__new__(ServingEngine)
    engine.apc_reuse_disabled_reason = None
    engine.counts = Counter()
    apc = _apc()
    assert not engine._publish_checkpoint(apc, APCKey("served"), TOKENS[:5], [_kv(4)])
    assert engine.counts["apcv2_store_refused_alignment"] == 1
    assert engine.counts["apcv2_store_refused_position_mismatch"] == 1
    assert engine.counts["apcv2_store_failures"] == 0
    assert engine._publish_checkpoint(apc, APCKey("served"), TOKENS[:4], [_kv(4)])
    assert engine.counts["apcv2_store_refused_alignment"] == 1
    apc.clear()


def test_store_honors_cache_list_subclass_transaction_before_publishing():
    from mlx2.runtime.models.cache import CacheList

    class PendingCommit(CacheList):
        def checkpoint_publication_refusal(self):
            return "uncommitted wrapper state"

    apc = _apc()
    key = APCKey("wrapper")
    result = apc.store(key, TOKENS[:4], [PendingCommit(_kv(4))])
    assert result.stored is False
    assert result.reason == "publication_alignment:uncommitted_transaction"
    assert "uncommitted wrapper state" in apc.last_alignment_refusal["detail"]
    assert not apc.lookup(key, TOKENS).hit
    apc.clear()
