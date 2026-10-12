import inspect
from types import SimpleNamespace

import pytest

from mlx2.runtime.models.qwen4_state_plane import QWEN4_STATE_PLANE_OPERATIONS
from mlx2.runtime.state_plane_operations import (
    StatePlaneContract,
    StatePlaneContractError,
    require_state_plane_contracts,
)


class _Provider:
    provider_revision = "test-provider-v1"

    def __init__(self, contract):
        self.value = contract

    def contract(self, _cache):
        return self.value


def _contract(*, fidelity="exact", operations=None):
    return StatePlaneContract(
        revision="test-state-v1",
        layout="test-layout-v1",
        fidelity=fidelity,
        operations=frozenset(
            operations or {"checkpoint", "private_successor", "rollback", "promotion"}
        ),
    )


def test_contract_validation_accepts_a_complete_exact_model_provider():
    provider = _Provider(_contract())
    result = require_state_plane_contracts(
        provider,
        [object(), object()],
        operations=("checkpoint", "private_successor", "rollback", "promotion"),
    )
    assert result == (_contract(), _contract())


@pytest.mark.parametrize(
    ("provider", "message"),
    [
        (_Provider(None), "no declared state-plane contract"),
        (_Provider(_contract(fidelity="approximate")), "exact state is required"),
        (
            _Provider(_contract(operations={"checkpoint", "rollback"})),
            "lacks state-plane operations",
        ),
    ],
)
def test_contract_validation_fails_closed(provider, message):
    with pytest.raises(StatePlaneContractError, match=message):
        require_state_plane_contracts(
            provider,
            [object()],
            operations=("checkpoint", "private_successor", "rollback", "promotion"),
        )


def test_qwen4_provider_refuses_unknown_state_types():
    assert QWEN4_STATE_PLANE_OPERATIONS.contract(SimpleNamespace()) is None


def test_shared_promotion_lifecycle_has_no_qwen4_cache_type_branches():
    import mlx2.runtime.segmented_physical_promotion as promotion

    source = inspect.getsource(promotion)
    assert ".models.qwen4" not in source
    assert "QSAKVCache" not in source
    assert "SegmentedBatchArraysCache" not in source
