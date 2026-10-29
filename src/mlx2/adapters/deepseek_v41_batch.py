# SPDX-License-Identifier: MIT
"""Explicit ordinary Source multi-request FFN/head profile, B1 default intact."""

from dataclasses import replace


def validate_physical_batch_policy(policy, *, width):
    """Reject unsupported combinations before any Source tensor materializes."""
    if type(width) is not int or width not in (2, 4):
        raise ValueError("physical batch width must be2 or4")
    if (
        policy.get("draft_model") not in (None, "none")
        or policy.get("ast_prompt_lookup", False)
        or policy.get("deepseek_vision", "reference") != "reference"
    ):
        raise ValueError("ordinary physical batching excludes learned draft and images")
    if any(
        key.startswith("deepseek_ced") and value != "reference"
        for key, value in policy.items()
    ):
        raise ValueError("batch route requires full-depth ordinary prefill")
    source_defaults = {
        'deepseek_source_prefill_partition': 'adaptive_v1',
        'deepseek_source_dspark_ast': 'off',
        'deepseek_source_dspark_prepare': 'off',
    }
    for key, value in policy.items():
        if (
            key.startswith("deepseek_source_")
            and key != "deepseek_source_experts"
            and value != source_defaults.get(key, "reference")
        ):
            raise ValueError(
                "batch route requires reference Source provider geometry: " + key
            )
    return width


def configure_physical_batch(adapter, *, width):
    from ..contracts import Capability
    from ..runtime.models.deepseek_v41_batch import REVISION, DeepSeekV41BatchedModel

    validate_physical_batch_policy(adapter.kernel_policy, width=width)
    if (
        adapter.draft_model is not None
        or getattr(adapter, "vision_weights", None) is not None
    ):
        raise ValueError("ordinary physical batching excludes learned draft and images")
    adapter.model = DeepSeekV41BatchedModel(adapter.model, width=width)
    adapter.layout += ":" + REVISION + ":width=" + str(width)
    adapter.profile += ":joint-ffn-head-b" + str(width)
    adapter.descriptor = replace(
        adapter.descriptor,
        capabilities=adapter.descriptor.capabilities | {Capability.CONTINUOUS_BATCH},
        cache_layout=adapter.layout,
        metadata={
            **adapter.descriptor.metadata,
            "qualification": "unqualified",
            "physical_batch": adapter.model.batch_execution_profile,
        },
    )
    adapter.physical_batch_width = width
    return adapter.model.batch_execution_profile
