import mlx.core as mx
import pytest

from mlx2.runtime import hybrid_speculative as hs
from mlx2.runtime.checkpoint_publication import (
    CheckpointPublicationUnsafe,
    checkpoint_publication_refusal,
    require_checkpoint_publishable,
)
from mlx2.runtime.cow_cache import (
    COWCheckpointPublicationUnsafe,
    restore_recovery_descriptors,
    snapshot_prompt_cache_descriptors,
    snapshot_recovery_descriptors,
)
from mlx2.runtime.generate import PromptChunkCorruption
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


@pytest.mark.parametrize("poison", [float("nan"), float("inf"), -float("inf")])
def test_mtp_checkpoint_refuses_poisoned_seed_before_snapshot(monkeypatch, poison):
    def snapshot(*args, **kwargs):
        pytest.fail("poisoned boundary reached descriptor/deep-copy snapshot")

    monkeypatch.setattr(hs, "snapshot_prompt_cache_descriptors", snapshot)
    monkeypatch.setattr(hs.copy, "deepcopy", snapshot)
    target, draft = [_kv(4)], [_kv(3)]
    seed = mx.array([[[1.0, poison]]])
    with pytest.raises(CheckpointPublicationUnsafe, match="hidden evidence"):
        capture_self_mtp_checkpoint(target, (draft, seed))
    with pytest.raises(PromptChunkCorruption) as exc:
        capture_self_mtp_checkpoint(target, (draft, seed), uid=17)
    assert exc.value.uids == (17,)


@pytest.mark.parametrize("poison_row", [0, 2])
def test_mtp_prefill_refuses_any_teacher_force_poison_before_draft_mutation(monkeypatch, poison_row):
    from test_batched_mtp import _tiny_qwen4_model

    model = _tiny_qwen4_model()
    real_backbone = hs._mtp_backbone

    def backbone(model, tokens, cache):
        logits, hidden = real_backbone(model, tokens, cache)
        hidden[:, poison_row, 0] = float("nan")
        return logits, hidden

    def teacher_force(*args, **kwargs):
        pytest.fail("poisoned hidden rows reached draft teacher forcing")

    monkeypatch.setattr(hs, "_mtp_backbone", backbone)
    monkeypatch.setattr(model, "mtp_step", teacher_force)
    with pytest.raises(PromptChunkCorruption) as exc:
        hs.advance_self_mtp_prefill(mx.array([1, 2, 3, 4]), model,
            prompt_cache=None, mtp_state=None, max_tokens=3, uid=23)
    assert exc.value.uids == (23,)


def test_mtp_restore_refuses_poisoned_seed_before_mutating_cache(monkeypatch):
    from test_batched_mtp import _tiny_qwen4_model

    def backbone(*args, **kwargs):
        pytest.fail("poisoned restored seed reached target execution")

    monkeypatch.setattr(hs, "_mtp_backbone", backbone)
    with pytest.raises(PromptChunkCorruption) as exc:
        hs.advance_self_mtp_prefill(mx.array([5, 6]), _tiny_qwen4_model(),
            prompt_cache=[_kv(4)], mtp_state=([_kv(3)], mx.array([[[float("nan"), 1]]])),
            max_tokens=1, uid=29)
    assert exc.value.uids == (29,)


@pytest.mark.parametrize("bad_index", [0, 1])
def test_mtp_lane_preparation_poison_does_not_orphan_selected_healthy_peer(monkeypatch, bad_index):
    from test_batched_mtp import _tiny_qwen4_model

    from mlx2.runtime.generate import BatchGenerator

    real = hs._mtp_backbone
    marker = 11 if bad_index == 0 else 21

    def backbone(model, tokens, cache):
        logits, hidden = real(model, tokens, cache)
        if int(tokens[0, 0].item()) == marker:
            hidden[:, 0, 0] = float("nan")
        return logits, hidden

    monkeypatch.setattr(hs, "_mtp_backbone", backbone)
    batch = BatchGenerator(_tiny_qwen4_model(), completion_batch_size=2,
                           self_mtp={"persistent": True, "num_draft": 2})
    uids = batch.insert([[11, 12, 13], [21, 22, 23]], max_tokens=[4, 4])
    healthy, bad = uids[1 - bad_index], uids[bad_index]
    delivered, failures = [], []
    try:
        for _ in range(15):
            _, responses = batch.next()
            delivered.extend(r for r in responses if r.uid == healthy)
            failures.extend(batch.take_lane_failures())
            if delivered and delivered[-1].finish_reason:
                break
        assert len(delivered) == 4 and delivered[-1].finish_reason == "length"
        assert [item["uid"] for item in failures] == [bad]
        assert not batch.take_atomic_cohort_failures()
    finally:
        batch.close()


@pytest.mark.parametrize("bad_index", [0, 1])
def test_atomic_mtp_preparation_poison_retires_whole_selected_cohort(monkeypatch, bad_index):
    from test_batched_mtp import _tiny_qwen4_model

    from mlx2.runtime import generate

    real = hs._mtp_backbone
    marker = 11 if bad_index == 0 else 21
    closed = []
    close = generate._close_segmented_detached

    def backbone(model, tokens, cache):
        logits, hidden = real(model, tokens, cache)
        if int(tokens[0, 0].item()) == marker:
            hidden[:, 0, 0] = float("nan")
        return logits, hidden

    def record_close(detached, **kwargs):
        closed.append(detached.lane.uid)
        return close(detached, **kwargs)

    monkeypatch.setattr(hs, "_mtp_backbone", backbone)
    monkeypatch.setattr(generate, "_close_segmented_detached", record_close)
    batch = generate.BatchGenerator(_tiny_qwen4_model(), completion_batch_size=2,
                                    self_mtp={"persistent": True, "num_draft": 2})
    cohort = {"tenant_id": "cpu", "id": "poison-cohort", "size": 2}
    uids = batch.insert([[11, 12, 13], [21, 22, 23]], max_tokens=[4, 4],
                        self_mtp_configs=[{"batch_cohort": cohort, "num_draft": 2}] * 2)
    try:
        assert batch._admit_mtp_joining(2) == 2
        result, progress = batch._make_mtp_batch(2)
        assert result is None and progress == []
        assert not batch._unprocessed_sequences
        assert not batch._generation_batch.state.lanes
        assert closed == uids[:bad_index]
        failures = batch.take_atomic_cohort_failures()
        assert len(failures) == 1 and failures[0]["uids"] == tuple(uids)
        assert "hidden evidence" in failures[0]["reason"]
        assert not batch.take_lane_failures()
    finally:
        batch.close()


def test_atomic_mtp_poison_cleanup_finishes_after_one_sibling_close_raises(monkeypatch):
    from test_batched_mtp import _tiny_qwen4_model

    from mlx2.runtime import generate

    real = hs._mtp_backbone
    closed = []
    close = generate._close_segmented_detached

    def backbone(model, tokens, cache):
        logits, hidden = real(model, tokens, cache)
        if int(tokens[0, 0].item()) == 31:
            hidden[:, 0, 0] = float("nan")
        return logits, hidden

    def record_close(detached, **kwargs):
        closed.append(detached.lane.uid)
        result = close(detached, **kwargs)
        if len(closed) == 1:
            raise RuntimeError("injected sibling close failure")
        return result

    monkeypatch.setattr(hs, "_mtp_backbone", backbone)
    monkeypatch.setattr(generate, "_close_segmented_detached", record_close)
    batch = generate.BatchGenerator(_tiny_qwen4_model(), completion_batch_size=3,
                                    self_mtp={"persistent": True, "num_draft": 2})
    cohort = {"tenant_id": "cpu", "id": "poison-close-cohort", "size": 3}
    uids = batch.insert([[11, 12, 13], [21, 22, 23], [31, 32, 33]], max_tokens=[4] * 3,
                        self_mtp_configs=[{"batch_cohort": cohort, "num_draft": 2}] * 3)
    try:
        with pytest.raises(RuntimeError, match="sibling close failure"):
            batch._make_mtp_batch(3)
        assert closed == uids[:2]
        assert not batch._unprocessed_sequences
        failures = batch.take_atomic_cohort_failures()
        assert len(failures) == 1 and failures[0]["uids"] == tuple(uids)
    finally:
        batch.close()
