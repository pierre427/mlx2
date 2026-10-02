"""FlashNextPolicy refuses MoE combinations that can never engage (sweep
2026-10-02 M1): ``moe_topk_fold="fold"`` without a routed down that hosts the
fold (and no row window to take it), and the router kernel or ``two_launch``
beside the batch-decode/verify MoE windows, which then always decline."""

import pytest

from mlx2.adapters.flash_next_policy import FlashNextPolicy


@pytest.mark.parametrize("routed", ["off", "gate_up", "two_launch"])
def test_fold_without_a_routed_down_is_refused(routed):
    with pytest.raises(ValueError, match="moe_topk_fold"):
        FlashNextPolicy.from_mapping({"moe_topk_fold": "fold", "moe_routed_decode": routed})


@pytest.mark.parametrize("routed", ["gate_up_down", "gate_up_down_shared"])
def test_fold_with_a_routed_down_is_accepted(routed):
    policy = FlashNextPolicy.from_mapping({"moe_topk_fold": "fold", "moe_routed_decode": routed})
    assert policy.as_dict()["moe_topk_fold"] == "fold"


def test_fold_is_accepted_when_a_row_window_takes_it():
    FlashNextPolicy.from_mapping(
        {"moe_topk_fold": "fold", "moe_routed_decode": "gate_up", "moe_window_verify": True}
    )


@pytest.mark.parametrize("window", ["moe_window_batch_decode", "moe_window_verify"])
@pytest.mark.parametrize(
    "selection",
    [
        {"moe_router_kernel": True, "moe_topk_fold": "off"},
        {"moe_routed_decode": "two_launch", "moe_topk_fold": "off"},
    ],
)
def test_router_kernel_or_two_launch_with_a_window_is_refused(window, selection):
    with pytest.raises(ValueError, match="every window would decline"):
        FlashNextPolicy.from_mapping({**selection, window: True})


def test_router_kernel_alone_and_defaults_still_validate():
    FlashNextPolicy.from_mapping({"moe_router_kernel": True, "moe_topk_fold": "off"})
    FlashNextPolicy.from_mapping({"moe_routed_decode": "two_launch"})
    FlashNextPolicy()
