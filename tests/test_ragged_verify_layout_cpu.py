"""Host-only contract tests for the authoritative ragged verify layout."""

import pytest

from mlx2.runtime.ragged_verify_layout import (
    RaggedBackendMode,
    RaggedVerifyCapabilities,
    RaggedVerifyError,
    RaggedVerifyLayout,
    capabilities_from_mapping,
    padded_capabilities,
)
from mlx2.runtime.ragged_verify_observation import (
    RaggedVerifyObserver,
    activate,
    observed_stage,
)


def test_mixed_depth_layout_has_one_authoritative_geometry():
    layout = RaggedVerifyLayout.from_draft_depths((17, 23, 41), (0, 2, 5))
    assert layout.query_lengths == (1, 3, 6)
    assert layout.cu_query_lengths == (0, 1, 4, 10)
    assert layout.token_to_lane == (0, 1, 1, 1, 2, 2, 2, 2, 2, 2)
    assert layout.right_padding == (5, 3, 0)
    assert layout.logical_rows == 10
    assert layout.padded_rows == 18
    assert layout.is_ragged
    assert layout.lane_slice(1) == slice(1, 4)
    assert layout.validate_acceptance((0, 1, 5)) == (0, 1, 5)


def test_backend_selection_is_explicit_and_fail_closed():
    layout = RaggedVerifyLayout.from_draft_depths((1, 2), (1, 3))
    padded = layout.select_backend(padded_capabilities(max_query_len=4))
    assert padded.mode is RaggedBackendMode.PADDED
    assert (padded.logical_rows, padded.physical_rows, padded.padding_rows) == (6, 8, 2)
    native = layout.select_backend(
        RaggedVerifyCapabilities(native_ragged=True, padded=True)
    )
    assert native.mode is RaggedBackendMode.NATIVE_RAGGED
    assert native.physical_rows == native.logical_rows == 6
    with pytest.raises(RaggedVerifyError, match="query length"):
        layout.select_backend(padded_capabilities(max_query_len=3))
    with pytest.raises(RaggedVerifyError, match="accepted prefix"):
        layout.validate_acceptance((2, 4))


@pytest.mark.parametrize(
    "uids,depths,message",
    [
        ((1, 1), (0, 1), "unique"),
        ((1,), (0, 1), "equal length"),
        ((1,), (-1,), ">= 0"),
        ((True,), (0,), "lane_uids"),
    ],
)
def test_invalid_layout_refuses_before_execution(uids, depths, message):
    with pytest.raises(RaggedVerifyError, match=message):
        RaggedVerifyLayout.from_draft_depths(uids, depths)


def test_capability_mapping_rejects_unknown_or_implicit_modes():
    assert capabilities_from_mapping({"flattened": True}).flattened
    with pytest.raises(RaggedVerifyError, match="unknown"):
        capabilities_from_mapping({"padding": True})
    with pytest.raises(RaggedVerifyError, match="safe mode"):
        capabilities_from_mapping({"padded": False})


def test_observer_is_request_local_and_labels_diagnostic_fences():
    evaluated = []
    observer = RaggedVerifyObserver(evaluator=lambda *values: evaluated.extend(values))
    with activate(observer):
        with observed_stage("target.forward", rows=7) as materialize:
            materialize("logits", "hidden")
    receipt = observer.receipt()
    assert evaluated == ["logits", "hidden"]
    assert receipt["diagnostic_only"]
    assert receipt["evaluation_fences_inserted"]
    assert receipt["stages"]["target.forward"]["calls"] == 1
    assert receipt["stages"]["target.forward"]["rows"] == 7
