import json
from pathlib import Path

import pytest

from mlx2.adapters.nemotron35_media import target_grid
from mlx2.adapters.nemotron35_super import (
    Nemotron35SuperAdapter,
    _split_super_execution_policy,
    native_draft_depth,
)
from mlx2.runtime.models.switch_layers import _switch_sort_decision


def adapter():
    model = Nemotron35SuperAdapter.__new__(Nemotron35SuperAdapter)
    model.draft_model = None
    model._num_draft = 3
    return model


def test_super_expert_sort_policy_is_adapter_owned_and_nonmutating():
    value = {"expert_gather_sort": "unsorted_decode", "num_draft": 3}
    assert _split_super_execution_policy(value) == (
        "unsorted_decode",
        "auto",
        "auto",
        "auto",
        "auto",
        "auto",
        "auto",
        {"num_draft": 3},
    )
    assert value["expert_gather_sort"] == "unsorted_decode"


@pytest.mark.parametrize("value", [True, False, "always", "", 1, {}])
def test_super_rejects_unknown_expert_sort_policy(value):
    with pytest.raises(ValueError, match="expert_gather_sort"):
        _split_super_execution_policy({"expert_gather_sort": value})


def test_super_mamba_conv_policy_is_adapter_owned_and_default_off():
    assert _split_super_execution_policy({"mamba_conv": "fused_b1"}) == (
        "auto",
        "fused_b1",
        "auto",
        "auto",
        "auto",
        "auto",
        "auto",
        {},
    )
    assert _split_super_execution_policy({"mamba_conv": "fused_b1_compact"}) == (
        "auto",
        "fused_b1_compact",
        "auto",
        "auto",
        "auto",
        "auto",
        "auto",
        {},
    )
    for value in (True, False, "always", "", 1, {}):
        with pytest.raises(ValueError, match="mamba_conv"):
            _split_super_execution_policy({"mamba_conv": value})


def test_super_mamba_norm_policy_is_adapter_owned_and_default_off():
    assert _split_super_execution_policy({"mamba_norm": "fused_b1"}) == (
        "auto",
        "auto",
        "fused_b1",
        "auto",
        "auto",
        "auto",
        "auto",
        {},
    )
    assert _split_super_execution_policy({"mamba_norm": "fused_short"}) == (
        "auto",
        "auto",
        "fused_short",
        "auto",
        "auto",
        "auto",
        "auto",
        {},
    )
    for value in (True, False, "always", "", 1, {}):
        with pytest.raises(ValueError, match="mamba_norm"):
            _split_super_execution_policy({"mamba_norm": value})


def test_super_relu2_down_policy_is_adapter_owned_and_default_off():
    assert _split_super_execution_policy({"moe_relu2_down": "fused_b1"}) == (
        "auto",
        "auto",
        "auto",
        "auto",
        "auto",
        "auto",
        "fused_b1",
        {},
    )
    for value in (True, False, "always", "", 1, {}):
        with pytest.raises(ValueError, match="moe_relu2_down"):
            _split_super_execution_policy({"moe_relu2_down": value})


def test_super_mamba_ssm_policy_is_adapter_owned_and_default_off():
    assert _split_super_execution_policy({"mamba_ssm": "fused_short"}) == (
        "auto",
        "auto",
        "auto",
        "fused_short",
        "auto",
        "auto",
        "auto",
        {},
    )
    assert _split_super_execution_policy({"mamba_ssm": "fused_short_gate"}) == (
        "auto",
        "auto",
        "auto",
        "fused_short_gate",
        "auto",
        "auto",
        "auto",
        {},
    )
    for value in (True, False, "always", "", 1, {}):
        with pytest.raises(ValueError, match="mamba_ssm"):
            _split_super_execution_policy({"mamba_ssm": value})


def test_super_mamba_short_conv_policy_is_adapter_owned_and_default_off():
    assert _split_super_execution_policy({"mamba_conv": "fused_short"}) == (
        "auto",
        "fused_short",
        "auto",
        "auto",
        "auto",
        "auto",
        "auto",
        {},
    )


def test_super_mamba_projection_policy_is_adapter_owned_and_default_off():
    assert _split_super_execution_policy({"mamba_projection": "compact_padded"}) == (
        "auto",
        "auto",
        "auto",
        "auto",
        "compact_padded",
        "auto",
        "auto",
        {},
    )
    assert _split_super_execution_policy({"mamba_projection": "compact_row_exact"}) == (
        "auto",
        "auto",
        "auto",
        "auto",
        "compact_row_exact",
        "auto",
        "auto",
        {},
    )
    assert _split_super_execution_policy(
        {"mamba_projection": "adaptive_row_exact"}
    ) == (
        "auto",
        "auto",
        "auto",
        "auto",
        "adaptive_row_exact",
        "auto",
        "auto",
        {},
    )
    assert _split_super_execution_policy({"mamba_projection": "short_row_exact"}) == (
        "auto",
        "auto",
        "auto",
        "auto",
        "short_row_exact",
        "auto",
        "auto",
        {},
    )
    assert _split_super_execution_policy(
        {"mamba_projection": "native_qmm_residual_b1"}
    ) == (
        "auto",
        "auto",
        "auto",
        "auto",
        "native_qmm_residual_b1",
        "auto",
        "auto",
        {},
    )
    for value in (True, False, "always", "", 1, {}):
        with pytest.raises(ValueError, match="mamba_projection"):
            _split_super_execution_policy({"mamba_projection": value})


def test_super_mamba_segment_policy_is_adapter_owned_and_default_off():
    assert _split_super_execution_policy(
        {"mamba_segment": "layerwise_short_exact"}
    ) == (
        "auto",
        "auto",
        "auto",
        "auto",
        "auto",
        "layerwise_short_exact",
        "auto",
        {},
    )
    for value in (True, False, "always", "", 1, {}):
        with pytest.raises(ValueError, match="mamba_segment"):
            _split_super_execution_policy({"mamba_segment": value})

    assert _split_super_execution_policy(
        {"mamba_segment": "layerwise_short_exact_routed_batch"}
    )[5] == "layerwise_short_exact_routed_batch"


def test_super_top22_candidate_changes_decode_only():
    assert _switch_sort_decision(
        "auto", assignments=22, token_length=1, invariant=False
    ) == (True, "auto_sorted")
    assert _switch_sort_decision(
        "unsorted_decode", assignments=22, token_length=1, invariant=False
    ) == (False, "unsorted_decode")
    assert _switch_sort_decision(
        "unsorted_decode", assignments=44, token_length=2, invariant=False
    ) == (True, "auto_sorted")
    assert _switch_sort_decision(
        "unsorted_decode", assignments=22, token_length=1, invariant=True
    ) == (True, "invariant_sorted")


def test_switch_mlp_top22_candidate_preserves_b1_output_and_counts_routes():
    import mlx.core as mx

    from mlx2.runtime.models.switch_layers import SwitchMLP

    switch = SwitchMLP(4, 3, 22)
    hidden = mx.arange(4, dtype=mx.float32).reshape(1, 1, 4)
    experts = mx.array(
        [
            [
                [
                    21,
                    0,
                    20,
                    1,
                    19,
                    2,
                    18,
                    3,
                    17,
                    4,
                    16,
                    5,
                    15,
                    6,
                    14,
                    7,
                    13,
                    8,
                    12,
                    9,
                    11,
                    10,
                ]
            ]
        ]
    )
    sorted_output = switch(hidden, experts)
    assert switch.set_sort_policy("unsorted_decode") == "auto"
    unsorted_output = switch(hidden, experts)
    mx.eval(sorted_output, unsorted_output)

    assert mx.array_equal(sorted_output, unsorted_output).item()
    assert switch.sort_status() == {
        "policy": "unsorted_decode",
        "counts": {
            "auto_sorted": 1,
            "auto_unsorted": 0,
            "invariant_sorted": 0,
            "unsorted_decode": 1,
        },
    }


@pytest.mark.parametrize("draft_depth", [3, 4, 7, 8])
def test_one_session_limit_does_not_cap_its_verification_rows(draft_depth):
    model = adapter()
    model._num_draft = native_draft_depth(draft_depth)
    result = model.execution_config(max_lanes=1, prefill_step=512)
    assert result["num_draft"] == draft_depth
    assert result["segment_aware_cohort_size"] == 1
    for width in (0, 2, 4, 5, True):
        with pytest.raises(ValueError, match="one session|B=2/B=4"):
            model.execution_config(max_lanes=width, prefill_step=512)
    with pytest.raises(ValueError, match="B=2/B=4"):
        model.create_external_batch(completion_batch_size=4)


def test_b2_b4_require_exact_progressive_external_verification():
    model = adapter()
    model.external_policy = {
        "draft_model": "head",
        "num_draft": 7,
        "progressive_verification_tile": 4,
        "progressive_verification_fidelity": "exact",
    }
    model.draft_model = object()
    result = model.execution_config(max_lanes=2, prefill_step=512)
    assert result["segment_aware_cohort_size"] == 2
    assert result["progressive_verification_tile"] == 4
    assert model._validate_session_width(2) == 2
    result = model.execution_config(max_lanes=4, prefill_step=512)
    assert result["segment_aware_cohort_size"] == 4
    assert model._validate_session_width(4) == 4

    for policy in (
        {"draft_model": "head", "num_draft": 7},
        {
            "draft_model": "head",
            "num_draft": 7,
            "progressive_verification_tile": 4,
            "progressive_verification_fidelity": "approximate",
        },
    ):
        model.external_policy = policy
        for width in (2, 4):
            with pytest.raises(ValueError, match="exact progressive"):
                model.execution_config(max_lanes=width, prefill_step=512)

    model.external_policy = {
        "draft_model": "head",
        "num_draft": 7,
        "progressive_verification_tile": 4,
    }
    for width in (0, 3, 5, True):
        with pytest.raises(ValueError, match="one session|B=2/B=4"):
            model.execution_config(max_lanes=width, prefill_step=512)


@pytest.mark.parametrize("trained_block_size", [8, 9])
def test_external_head_can_verify_its_full_trained_depth_in_one_session(
    trained_block_size,
):
    from types import SimpleNamespace

    model = adapter()
    model.external_policy = {"draft_model": "head", "num_draft": trained_block_size - 1}
    model._check_num_draft({"args": SimpleNamespace(block_size=trained_block_size)})
    model.draft_model = object()
    result = model.execution_config(max_lanes=1, prefill_step=512)
    assert result["num_draft"] == trained_block_size - 1
    assert result["segment_aware_cohort_size"] == 1


def test_image_prefix_cannot_resume_inside_media_span():
    model = adapter()
    request = {"_mlx2_prompt_tokens": [5, 18, 18, 6]}
    payload = {"image_pixels": [object()]}
    assert model.validate_prefill_inputs(request, [5, 18, 18, 6], payload) is payload
    assert model.validate_prefill_inputs(request, [6], payload) == {}
    with pytest.raises(ValueError, match="middle"):
        model.validate_prefill_inputs(request, [18, 6], payload)


@pytest.mark.parametrize(
    "width,height", [(512, 512), (1024, 768), (1920, 1080), (3, 3), (512, 3000)]
)
def test_image_grid_matches_pinned_resize_arithmetic(width, height):
    h, w = target_grid(width, height)
    assert h % 2 == w % 2 == 0
    assert h * w <= 13312
    assert h * w >= 1024


def test_source_explicit_time_step_bound_and_fp32_router():
    from mlx2.adapters.nemotron35_super_config import language_config

    config = json.loads(
        (Path(__file__).parent / "fixtures/nemotron35_super_config.json").read_text()
    )
    text = language_config(config)
    assert text["time_step_limit"][0] == 0.0
    assert text["router_compute_dtype"] == "float32"


@pytest.mark.parametrize("route", ["native_mtp", "external_draft", "prompt_lookup"])
def test_serving_refuses_unsupported_media_route_before_preprocessing(route):
    from types import SimpleNamespace

    from mlx2.api_resources import CapabilityUnavailable
    from mlx2.serving import ServingEngine

    engine = ServingEngine.__new__(ServingEngine)
    engine.route_capabilities = frozenset()
    engine.snapshot = {"settings": {"route": route}}
    calls = []
    engine.adapter = SimpleNamespace(
        multimodal_routes=frozenset({"ordinary"}),
        prepare_multimodal_request=lambda *a, **kw: calls.append(1),
    )
    request = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA=="},
                    }
                ],
            }
        ]
    }
    with pytest.raises(
        CapabilityUnavailable, match="no implemented multimodal prefill"
    ):
        engine._prepare_job(request, tenant_id="test")
    assert calls == []
