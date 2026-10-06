import inspect

import pytest

from mlx2.runtime import hybrid_speculative
from mlx2.runtime.committed_recovery import (
    CommittedRecoverySlot,
    RecoveryCheckpointMismatch,
)
from scripts.bench_upstream_lifecycle_viability import (
    Page,
    RequestOwnedStatePool,
    committed_prefix,
    evict_under_pressure,
    overcommitted_prefix,
    predraft_open_allowed,
    run_simulations,
    swa_endpoint_priority,
)


def test_terminal_predraft_guard_covers_budget_and_pending_eos():
    assert predraft_open_allowed(7, 8, False)
    assert not predraft_open_allowed(8, 8, False)
    assert not predraft_open_allowed(9, 8, False)
    assert not predraft_open_allowed(7, 8, True)


def test_commit_only_emitted_excludes_the_invisible_verify_suffix():
    prefix, outputs = (1, 2, 3), (4, 5, 6, 7)
    assert committed_prefix(prefix, outputs, accepted=3, emitted=2) == (1, 2, 3, 4, 5)
    assert overcommitted_prefix(prefix, outputs, accepted=3) == (1, 2, 3, 4, 5, 6, 7)


def test_mlx2_commit_path_already_settles_the_emitted_window():
    source = inspect.getsource(hybrid_speculative.commit_batched_self_mtp)
    assert "Commit exactly the delivered prefix" in source
    assert "_settle_verify_windows(proposal, emitted)" in source
    assert "count != len(outputs)" in source


def test_suspended_recurrent_state_follows_request_not_reused_slot():
    pool = RequestOwnedStatePool(2)
    pool.attach("a", 0)
    pool.write("a", {"conv": (1,), "recurrent": (2,), "accepted": 1})
    pool.suspend("a")
    pool.attach("b", 0, context=True)
    pool.write("b", {"conv": (9,), "recurrent": (8,), "accepted": 3})
    pool.record_late_acceptance("a", 2)
    pool.attach("a", 1)
    assert pool.read("a") == {"conv": (1,), "recurrent": (2,), "accepted": 2}
    assert pool.read("b") == {"conv": (9,), "recurrent": (8,), "accepted": 3}


def test_mlx2_recovery_slot_rejects_cross_request_identity():
    slot = CommittedRecoverySlot()
    slot.capture(
        route="self_mtp",
        revision="artifact-a",
        boundary=32,
        value={"state": "a"},
        snapshot=dict,
        restore=dict,
    )
    assert slot.restore(route="self_mtp", revision="artifact-a", boundary=32) == {"state": "a"}
    with pytest.raises(RecoveryCheckpointMismatch):
        slot.restore(route="self_mtp", revision="artifact-b", boundary=32)


def test_swa_endpoint_priority_changes_real_pressure_victim_order():
    pages = (
        Page("old", "swa", 0, 64, 9),
        Page("endpoint", "swa", 960, 1024, 1),
        Page("full", "full", 512, 576, 2),
    )
    options = {"reusable_prompt_tokens": 1024, "window_tokens": 128, "rewind_tokens": 64}
    priority = lambda page: swa_endpoint_priority(page, **options)
    assert {page.name: priority(page) for page in pages} == {"old": 0, "endpoint": 70, "full": 35}
    lru_evicted, _ = evict_under_pressure(pages, keep=2)
    policy_evicted, policy_retained = evict_under_pressure(pages, keep=2, priority=priority)
    assert lru_evicted == ("endpoint",)
    assert policy_evicted == ("old",)
    assert "endpoint" in policy_retained


def test_lifecycle_benchmark_receipt_is_explicitly_nonselecting():
    report = run_simulations(iterations=5)
    assert report["gpu_used"] is False
    assert report["production_behavior_changed"] is False
    assert report["request_owned_recurrent_state"]["exact"] is True
    assert report["swa_endpoint_priority"]["endpoint_retained_under_pressure"] is True
