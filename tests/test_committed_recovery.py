import copy

import pytest

from mlx2.runtime.committed_recovery import (
    CommittedRecoverySlot,
    RecoveryCheckpointMismatch,
)


def test_committed_recovery_slot_restores_fresh_route_owned_state():
    slot = CommittedRecoverySlot()
    source = {"tokens": [1, 2], "sidecar": {"offset": 1}}
    slot.capture(
        route="self_mtp",
        revision="model-a",
        boundary=2,
        value=source,
        snapshot=copy.deepcopy,
        restore=copy.deepcopy,
    )
    source["tokens"].append(3)
    first = slot.restore(route="self_mtp", revision="model-a", boundary=2)
    second = slot.restore(route="self_mtp", revision="model-a", boundary=2)
    first["tokens"].append(99)
    assert second == {"tokens": [1, 2], "sidecar": {"offset": 1}}
    assert slot.status()["captures"] == 1
    assert slot.status()["restores"] == 2


@pytest.mark.parametrize(
    "route,revision,boundary",
    [("prompt_lookup", "model-a", 2), ("self_mtp", "model-b", 2), ("self_mtp", "model-a", 3)],
)
def test_committed_recovery_slot_rejects_cross_route_revision_or_boundary(
    route, revision, boundary
):
    slot = CommittedRecoverySlot()
    slot.capture(
        route="self_mtp",
        revision="model-a",
        boundary=2,
        value={"state": "committed"},
        snapshot=copy.deepcopy,
        restore=copy.deepcopy,
    )
    with pytest.raises(RecoveryCheckpointMismatch):
        slot.restore(route=route, revision=revision, boundary=boundary)


def test_committed_recovery_slot_miss_and_invalidation_are_explicit():
    slot = CommittedRecoverySlot()
    assert slot.restore(route="pld", revision="r", boundary=0) is None
    slot.capture(
        route="pld",
        revision="r",
        boundary=0,
        value=(),
        snapshot=tuple,
        restore=tuple,
    )
    slot.invalidate()
    assert slot.restore(route="pld", revision="r", boundary=0) is None
    assert slot.status()["misses"] == 2
    assert slot.status()["invalidations"] == 1
