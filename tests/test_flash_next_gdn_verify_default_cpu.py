"""Select batched row-exact GDN only for the profile that needs it."""

import pytest

from mlx2.adapters.flash_next_policy import FlashNextPolicy


@pytest.mark.parametrize("settings,expected", [
    ({}, "off"),
    ({"row_exact_verify": True}, "row_exact"),
    ({"row_exact_verify": True, "fused_gdn_batch_verify": "off"}, "off"),
    ({"fused_gdn_batch_verify": "row_exact"}, "row_exact"),
])
def test_resolved_selection_round_trips_into_environment(settings, expected):
    policy = FlashNextPolicy.from_mapping(settings)
    assert policy.fused_gdn_batch_verify == expected
    assert FlashNextPolicy.from_mapping(policy.as_dict()) == policy
    assert policy.environment().get("MLX_QWEN4_FUSED_GDN_BATCH_VERIFY", "off") == expected
    if expected == "row_exact":
        assert policy.as_dict()["fused_gdn_batch_verify"] == expected
