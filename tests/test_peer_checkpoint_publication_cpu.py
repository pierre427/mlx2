"""Replay publication consumes executor evidence, including removed terminal lanes."""

from collections import Counter
from types import SimpleNamespace as NS

from mlx2.serving import ServingEngine, checkpoint_schedule_changed


def test_generated_state_after_geometry_change_is_not_published():
    owner = NS(uid=3, prompt_tokens=4)
    batch = NS(_generation_batch=NS(numerical_schedule_for_uid=lambda uid: {
        "numerical_schedule_changed": uid == 3}))
    assert not checkpoint_schedule_changed(batch, owner, [0] * 4)
    assert checkpoint_schedule_changed(batch, owner, [0] * 5)
    assert not checkpoint_schedule_changed(batch, owner, [0] * 5,
                                           checkpoint={"numerical_schedule_changed": False})
    terminal = NS(mtp_receipt={"numerical_schedule_changed": True})
    assert checkpoint_schedule_changed(NS(), owner, [0] * 5, response=terminal)
    assert checkpoint_schedule_changed(NS(), owner, None, response=terminal)
    engine = ServingEngine.__new__(ServingEngine)
    engine.apc_reuse_disabled_reason = None
    engine.counts = Counter()
    stored = []
    assert not engine._publish_checkpoint(
        NS(store=lambda *args, **kwargs: stored.append(args)), "key", [0] * 5, [],
        numerical_schedule_changed=True,
    )
    assert stored == []
    assert engine.counts["apcv2_store_skipped_numerical_schedule"] == 1


def test_prompt_only_and_stable_generated_checkpoints_still_publish():
    engine = ServingEngine.__new__(ServingEngine)
    engine.apc_reuse_disabled_reason = None
    engine.counts = Counter()
    stored = []
    apc = NS(store=lambda *args, **kwargs: stored.append(args))
    assert engine._publish_checkpoint(apc, "key", [1], [], numerical_schedule_changed=False)
    assert len(stored) == 1
