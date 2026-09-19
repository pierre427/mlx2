from dataclasses import asdict

import pytest

from mlx2.adapters.qwen38_memory import Qwen38CacheBudget
from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController


def config():
    return {
        "num_hidden_layers": 64,
        "full_attention_interval": 4,
        "mtp_num_hidden_layers": 1,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "linear_num_value_heads": 48,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    }


def test_real_geometry_bounds_262k_without_full_attention_envelope():
    ordinary = Qwen38CacheBudget.from_config(config(), mtp=False)
    mtp = Qwen38CacheBudget.from_config(config(), mtp=True)
    assert (ordinary.attention_layers, ordinary.recurrent_layers) == (16, 48)
    assert (ordinary.mtp_layers, mtp.mtp_layers) == (0, 1)
    assert 32 < ordinary.project(262144) / 2**30 < 33
    assert 34 < mtp.project(262144) / 2**30 < 35
    assert ordinary.project(263168) > ordinary.project(262144)
    assert ordinary.as_dict()["schema"] == "qwen38-cache-geometry-v1"


def test_controller_keeps_reserve_and_charges_dense_workspace():
    for mtp, depth in ((False, 0), (True, 2)):
        geometry = Qwen38CacheBudget.from_config(config(), mtp=mtp)
        controller = SelfMTPLaneAdmissionController(
            cache_estimator=geometry.project,
            transient_gib_per_lane=geometry.transient_gib_per_lane,
        )
        required = controller.lane_gib(262144, depth)
        assert controller.hard_reserve_gib == 20
        assert required <= 103 - controller.hard_reserve_gib
        if mtp:
            assert controller.decide([262144], 103, max_draft=depth).modes == (
                "self_mtp",
            )
            constrained = controller.decide(
                [262144], controller.hard_reserve_gib + required - 0.01,
                max_draft=depth,
            )
            assert constrained.draft_depths != (depth,)
        else:
            assert required > (
                controller.hard_reserve_gib + required - 0.01
                - controller.hard_reserve_gib
            )


def test_warm_copy_and_growth_are_both_charged():
    geometry = Qwen38CacheBudget.from_config(config(), mtp=True)
    controller = SelfMTPLaneAdmissionController(
        cache_estimator=geometry.project,
        transient_gib_per_lane=geometry.transient_gib_per_lane,
    )
    cold = controller.lane_gib(131072, 2)
    warm = controller.lane_gib(131072, 2, cache_gib=8.5)
    resident = controller.lane_gib(
        131072, 2, cache_gib=8.5, resident_cache=True
    )
    assert warm == resident + 8.5
    assert cold > 0 and resident > geometry.transient_gib_per_lane


@pytest.mark.parametrize(
    "update",
    [
        {"num_hidden_layers": 63},
        {"full_attention_interval": 0},
        {"layer_types": ["full_attention"] * 64},
        {"num_key_value_heads": 0},
    ],
)
def test_invalid_or_unknown_geometry_fails_closed(update):
    candidate = {**config(), **update}
    with pytest.raises(ValueError):
        Qwen38CacheBudget.from_config(candidate, mtp=True)


def test_projection_fields_are_nonnegative_and_serializable():
    budget = Qwen38CacheBudget.from_config(config(), mtp=True)
    assert all(value >= 0 for value in asdict(budget).values())
    assert budget.project(0) >= budget.fixed_bytes
