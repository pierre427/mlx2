from dataclasses import dataclass, replace

import pytest

from mlx2.runtime.approximate_state import (
    ApproximateKVController,
    ApproximateKVPolicy,
    ApproximateStateError,
)


@dataclass(frozen=True)
class State:
    revision: str
    payload: tuple[int, ...]


class Adapter:
    revision = "r1"

    def apply(self, state):
        return replace(state, revision="r1:k8v4", payload=state.payload[::2])


def test_approximate_kv_is_default_off_and_does_not_touch_state():
    state = State("r1", (1, 2, 3))
    updated, receipt = ApproximateKVController().apply(
        request_id="req", state_revision="r1", state=state, adapters={}
    )
    assert updated is state
    assert receipt["selected"] is False
    assert receipt["reason"] == "disabled"
    assert receipt["fidelity"] == "approximate"


def test_unqualified_or_unimplemented_approximate_kv_fails_closed():
    state = State("r1", (1, 2, 3))
    controller = ApproximateKVController(
        ApproximateKVPolicy("k8v4", enabled=True)
    )
    with pytest.raises(ApproximateStateError, match="not qualified"):
        controller.apply(
            request_id="req", state_revision="r1", state=state, adapters={}
        )
    qualified = ApproximateKVController(
        ApproximateKVPolicy(
            "k8v4", enabled=True, qualified=True, evidence=("receipt.json",)
        )
    )
    with pytest.raises(ApproximateStateError, match="does not implement"):
        qualified.apply(
            request_id="req", state_revision="r1", state=state, adapters={}
        )


def test_qualified_adapter_operation_advances_revision_and_emits_receipt():
    state = State("r1", (1, 2, 3, 4))
    controller = ApproximateKVController(
        ApproximateKVPolicy(
            "k8v4", enabled=True, qualified=True, evidence=("receipt.json",)
        )
    )
    updated, receipt = controller.apply(
        request_id="req",
        state_revision="r1",
        state=state,
        adapters={"k8v4": Adapter()},
    )
    assert updated == State("r1:k8v4", (1, 3))
    assert receipt["selected"] is True
    assert receipt["target_revision"] == "r1:k8v4"
    assert receipt["evidence"] == ["receipt.json"]


def test_adapter_revision_and_publication_are_revision_bound():
    state = State("r2", (1, 2))
    controller = ApproximateKVController(
        ApproximateKVPolicy(
            "k8v4", enabled=True, qualified=True, evidence=("receipt.json",)
        )
    )
    with pytest.raises(ApproximateStateError, match="revision mismatch"):
        controller.apply(
            request_id="req",
            state_revision="r2",
            state=state,
            adapters={"k8v4": Adapter()},
        )


def test_invalid_in_place_adapter_cannot_mutate_live_state():
    @dataclass
    class MutableState:
        revision: str
        payload: list[int]

    class MutatingInvalidAdapter:
        revision = "r1"

        def apply(self, state):
            state.payload.clear()
            return state

    state = MutableState("r1", [1, 2, 3])
    controller = ApproximateKVController(
        ApproximateKVPolicy(
            "k8v4", enabled=True, qualified=True, evidence=("receipt.json",)
        )
    )
    with pytest.raises(ApproximateStateError, match="did not advance"):
        controller.apply(
            request_id="req",
            state_revision="r1",
            state=state,
            adapters={"k8v4": MutatingInvalidAdapter()},
        )
    assert state == MutableState("r1", [1, 2, 3])
