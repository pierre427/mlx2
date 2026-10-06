"""CPU-only adapter contract for exact native self-MTP verify/rollback rows."""

import pytest

from mlx2.adapters.self_mtp_rows import (
    ExactSelfMTPRows,
    constrain_self_mtp_proposers,
    declared_exact_self_mtp_rows,
)
from mlx2.runtime.copy_draft import CopyDraftPolicy


def _adapter(verify=None, rollback=None):
    values = {}
    if verify is not None:
        values["max_exact_self_mtp_verification_rows"] = verify
    if rollback is not None:
        values["max_exact_self_mtp_rollback_rows"] = rollback
    return type("DeclaredAdapter", (), values)()


@pytest.mark.parametrize(
    "verify,rollback,expected_depth",
    [(16, 8, 7), (8, 16, 7), (17, 17, 16)],
)
def test_proposer_depth_uses_the_smaller_exact_row_cap(
    verify, rollback, expected_depth
):
    declared = declared_exact_self_mtp_rows(_adapter(verify, rollback))
    assert declared.effective_max_self_mtp_proposer_depth == expected_depth

    requested = CopyDraftPolicy(
        enabled=True,
        max_span=15,
        probe_span=9,
        batched_max_span=12,
        min_match=8,
        strong_match=16,
        strong_max_span=15,
        initial_span=14,
    )
    effective_num_draft, effective, receipt = constrain_self_mtp_proposers(
        declared,
        self_mtp_num_draft=12,
        self_mtp_copy_draft_policy=requested,
    )
    assert effective.span_ceiling == min(15, expected_depth)
    assert effective.probe_span <= effective.max_span
    assert effective.initial_span <= effective.span_ceiling
    assert effective.batched_max_span <= expected_depth

    assert receipt["effective_max_self_mtp_rows"] == min(verify, rollback)
    assert (
        effective_num_draft
        == receipt["effective_self_mtp_num_draft"]
        == min(12, expected_depth)
    )
    assert receipt["clamped"] is (expected_depth < 15)


def test_safe_policy_is_identity_preserving():
    policy = CopyDraftPolicy(enabled=True, max_span=7, probe_span=2)
    assert policy.clamped_to_self_mtp_proposer_depth(16) is policy


@pytest.mark.parametrize(
    "adapter,error",
    [
        (_adapter(verify=16), "declare both"),
        (_adapter(rollback=8), "declare both"),
        (_adapter(True, 8), "integer >= 2"),
        (_adapter(1, 8), "integer >= 2"),
    ],
)
def test_partial_or_invalid_explicit_capabilities_fail_closed(adapter, error):
    with pytest.raises(ValueError, match=error):
        declared_exact_self_mtp_rows(adapter)


def test_undeclared_adapter_preserves_historical_behavior():
    assert declared_exact_self_mtp_rows(type("LegacyAdapter", (), {})()) is None


def test_declared_adapters_own_paired_limits_without_family_inheritance():
    from mlx2.adapters.flash_next import FlashNextAdapter
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    flash = object.__new__(FlashNextAdapter)
    qwen38 = object.__new__(Qwen3827BAdapter)
    assert declared_exact_self_mtp_rows(flash) == ExactSelfMTPRows(17, 17)
    assert declared_exact_self_mtp_rows(qwen38) == ExactSelfMTPRows(9, 9)

    class Sibling(Qwen3827BAdapter):
        pass

    assert declared_exact_self_mtp_rows(object.__new__(Sibling)) is None


def test_current_adapter_defaults_are_not_clamped():
    from mlx2.adapters.flash_next import FlashNextAdapter
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    cases = (
        (FlashNextAdapter, 3),
        (Qwen3827BAdapter, 3),
    )
    for adapter_type, num_draft in cases:
        adapter = object.__new__(adapter_type)
        selected = adapter_type.default_route_execution_policy["native_mtp"]
        policy = CopyDraftPolicy.from_value(selected["self_mtp_copy_draft"])
        effective_num_draft, effective_policy, receipt = (
            constrain_self_mtp_proposers(
                declared_exact_self_mtp_rows(adapter),
                self_mtp_num_draft=num_draft,
                self_mtp_copy_draft_policy=policy,
            )
        )
        assert effective_num_draft == num_draft
        assert effective_policy is policy
        assert receipt["clamped"] is False
