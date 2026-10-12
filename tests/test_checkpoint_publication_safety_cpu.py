import mlx.core as mx
import pytest

from mlx2.runtime.checkpoint_publication import (
    CheckpointPublicationUnsafe,
    checkpoint_publication_refusal,
    require_checkpoint_publishable,
)
from mlx2.runtime.cow_cache import (
    COWCheckpointPublicationUnsafe,
    restore_recovery_descriptors,
    snapshot_recovery_descriptors,
    snapshot_prompt_cache_descriptors,
)
from mlx2.runtime.hybrid_speculative import capture_self_mtp_checkpoint
from mlx2.runtime.models.cache import ArraysCache, CacheList, KVCache, RotatingKVCache
from mlx2.runtime.models.qwen4_exp import QSAKVCache, Qwen4ArraysCache


class CompactReplayCache:
    def __init__(self, pending=0):
        self.pending = pending

    def checkpoint_publication_refusal(self):
        if self.pending:
            return f"{self.pending} temporal ring updates are not flushed"
        return None


def _kv(tokens):
    cache = KVCache()
    values = mx.ones((1, 1, tokens, 2), dtype=mx.float32)
    cache.update_and_fetch(values, values)
    return cache


def test_declared_compact_replay_ring_blocks_publication_until_flushed():
    cache = CompactReplayCache(pending=3)
    assert checkpoint_publication_refusal([cache]) == (
        "CompactReplayCache: 3 temporal ring updates are not flushed"
    )
    with pytest.raises(CheckpointPublicationUnsafe, match="3 temporal ring updates"):
        require_checkpoint_publishable([cache])
    cache.pending = 0
    require_checkpoint_publishable([cache])


def test_descriptor_capture_uses_the_general_publication_contract():
    cache = CompactReplayCache(pending=1)
    with pytest.raises(COWCheckpointPublicationUnsafe, match="temporal ring"):
        snapshot_prompt_cache_descriptors([cache])


def test_mtp_boundary_never_deepcopies_an_open_transaction(monkeypatch):
    target = RotatingKVCache(max_size=16)
    values = mx.ones((1, 1, 4, 2), dtype=mx.float32)
    target.update_and_fetch(values, values)
    draft = _kv(3)
    target.start_speculation()
    seed = mx.ones((1, 1, 2), dtype=mx.float32)

    called = False

    def deepcopy(value):
        nonlocal called
        called = True
        return value

    monkeypatch.setattr("mlx2.runtime.hybrid_speculative.copy.deepcopy", deepcopy)
    with pytest.raises(CheckpointPublicationUnsafe, match="speculation transaction"):
        capture_self_mtp_checkpoint([target], ([draft], seed))
    assert called is False


@pytest.mark.parametrize(
    ("cache", "mutate", "reason"),
    [
        (
            ArraysCache(1),
            lambda cache: cache.start_speculation(),
            "speculation transaction is active",
        ),
        (
            Qwen4ArraysCache(4),
            lambda cache: setattr(cache, "_ple_rollback", object()),
            "PLE temporal update is staged",
        ),
        (
            QSAKVCache(),
            lambda cache: setattr(cache, "_mtp_share_topk", True),
            "QSA/MTP transaction is active",
        ),
    ],
)
def test_current_cache_families_declare_checkpoint_publication_guards(
    cache, mutate, reason
):
    mutate(cache)
    assert reason in checkpoint_publication_refusal([cache])


def test_private_field_lookalike_is_not_used_as_a_publication_contract():
    class UndeclaredLookalike:
        speculating = True
        _rollbacks = [object()]
        _ple_rollback = object()
        _mtp_share_topk = True

    assert checkpoint_publication_refusal([UndeclaredLookalike()]) == (
        "UndeclaredLookalike: cache family has no checkpoint-publication "
        "operations declaration"
    )


def test_wrapper_owned_refusal_is_checked_before_declared_children():
    class TransactionalWrapper:
        def __init__(self):
            self.caches = [KVCache()]

        def checkpoint_publication_refusal(self):
            return "wrapper transaction is active"

    assert checkpoint_publication_refusal([TransactionalWrapper()]) == (
        "TransactionalWrapper: wrapper transaction is active"
    )


def test_exact_cache_list_is_the_only_implicit_transparent_wrapper():
    require_checkpoint_publishable([CacheList(KVCache())])

    class UnknownCacheList(CacheList):
        pass

    with pytest.raises(CheckpointPublicationUnsafe, match="no checkpoint-publication"):
        require_checkpoint_publishable([UnknownCacheList(KVCache())])


def test_declared_safe_wrapper_still_checks_its_children():
    class DeclaredWrapper:
        def __init__(self, child):
            self.caches = [child]

        def checkpoint_publication_refusal(self):
            return None

    child = ArraysCache(1)
    child.start_speculation()
    with pytest.raises(CheckpointPublicationUnsafe, match="speculation transaction"):
        require_checkpoint_publishable([DeclaredWrapper(child)])


def test_private_recovery_round_trips_an_open_transaction_without_publishing_it():
    cache = ArraysCache(1)
    cache.cache[0] = mx.ones((1, 2, 3), dtype=mx.float32)
    cache.start_speculation()

    snapshot, sidecar, borrowed = snapshot_recovery_descriptors([cache])
    restored, _ = restore_recovery_descriptors(snapshot, sidecar, borrowed)

    assert restored[0].speculating is True
    with pytest.raises(CheckpointPublicationUnsafe, match="speculation transaction"):
        require_checkpoint_publishable(restored)


def test_mtp_boundary_still_captures_a_committed_pair():
    target = _kv(4)
    draft = _kv(3)
    seed = mx.ones((1, 1, 2), dtype=mx.float32)
    captured = capture_self_mtp_checkpoint([target], ([draft], seed))
    assert captured is not None
    assert captured["covered_tokens"] == 4
    assert captured["committed_only"] is True
