# SPDX-License-Identifier: MIT
"""Compact native IQ1_S experts with native affine dense V4.1 serving."""

from dataclasses import replace

from .deepseek_v41_iq1_artifact import (
    FORMAT,
    MODEL_TYPE,
    NUMERIC_PROFILE,
    inspect_artifact,
)
from .deepseek_v41_mlx import DESCRIPTOR as AFFINE_DESCRIPTOR
from .deepseek_v41_mlx import DeepSeekV41MLXAdapter, admission_plan
from .deepseek_v41_mlx import execution_policy as affine_policy
from .deepseek_v41_mlx import optimized_policy as affine_optimized_policy

NUMERICS_REVISION = "mlx2-v41-iq1s-native-f32-dense-v1"
DESCRIPTOR = replace(
    AFFINE_DESCRIPTOR,
    model_type=MODEL_TYPE,
    variant="flash-1017-maya-xs-native-iq1s-text",
    metadata={
        **AFFINE_DESCRIPTOR.metadata,
        "execution": "mlx2.adapters.deepseek_v41_iq1.DeepSeekV41IQ1Adapter",
        "numeric_profile": NUMERIC_PROFILE,
        "scope": "standalone native IQ1_S experts and affine dense B1 text; full-depth V4.1; exact APCv2 within compact profile",
    },
)


def execution_policy(value=None):
    if value is not None and not isinstance(value, dict):
        raise ValueError("compact V4.1 execution policy must be a dictionary")
    value = dict(value or {})
    prefill = value.pop("deepseek_expert_prefill", "reference")
    layout = value.pop("deepseek_expert_layout", "row_major")
    completion = value.pop("deepseek_arena_completion", "sync")
    decode = value.pop("deepseek_tiled_decode", "row_exact")
    if type(prefill) is not str or prefill not in ("reference", "segmented_nax_fp16"):
        raise ValueError("unsupported compact V4.1 expert prefill policy")
    if type(layout) is not str or layout not in ("row_major", "tiled64"):
        raise ValueError("unsupported compact V4.1 expert layout")
    if (
        type(completion) is not str
        or completion not in ("sync", "async")
        or (completion == "async" and layout != "tiled64")
    ):
        raise ValueError("asynchronous arena completion requires tiled64 experts")
    if (
        type(decode) is not str
        or decode not in ("row_exact", "column_f32")
        or (decode == "column_f32" and layout != "tiled64")
    ):
        raise ValueError("column_f32 expert arithmetic requires tiled64 experts")
    return {
        **affine_policy(value),
        "deepseek_expert_prefill": prefill,
        "deepseek_expert_layout": layout,
        "deepseek_arena_completion": completion,
        "deepseek_tiled_decode": decode,
    }


def optimized_policy(*, copy_chain=False):
    return {
        **affine_optimized_policy(copy_chain=copy_chain),
        "deepseek_expert_prefill": "segmented_nax_fp16",
    }


class DeepSeekV41IQ1Adapter(DeepSeekV41MLXAdapter):
    descriptor = DESCRIPTOR
    artifact_inspector = staticmethod(inspect_artifact)
    profile = "deepseek-v41-1017-maya-xs-native-iq1s-ordinary-b1"
    weight_format = FORMAT
    numeric_profile = NUMERIC_PROFILE
    numerics_revision = NUMERICS_REVISION
    policy_parser = staticmethod(execution_policy)
    weight_streaming_modes = frozenset({"moe_experts"})

    @staticmethod
    def weight_streaming_external_draft(policy):
        if 'draft_models' in policy or policy.get('draft_model') != 'prompt_lookup':
            return False
        execution_policy(policy)
        return True

    adapter_owned_default_policy_keys = (
        DeepSeekV41MLXAdapter.adapter_owned_default_policy_keys
        | {
            "deepseek_expert_prefill",
            "deepseek_expert_layout",
            "deepseek_arena_completion",
            "deepseek_tiled_decode",
        }
    )

    def __init__(self, model_path, *, execution_policy=None, weight_streaming=None):
        selected = self.policy_parser(execution_policy)
        if selected["deepseek_expert_layout"] == "tiled64":
            from .deepseek_v41_iq1_tiled import numerical_profile

            self.numeric_profile, self.numerics_revision = numerical_profile(
                selected["deepseek_tiled_decode"]
            )
            self.profile = (
                "deepseek-v41-native-iq1s-tiled-ordinary-b1:"
                + selected["deepseek_tiled_decode"]
            )
        super().__init__(
            model_path,
            execution_policy=execution_policy,
            weight_streaming=weight_streaming,
        )
        self.descriptor = replace(
            self.descriptor,
            metadata={
                **self.descriptor.metadata,
                "numeric_profile": self.numeric_profile,
                "expert_layout": selected["deepseek_expert_layout"],
                "arena_completion": selected["deepseek_arena_completion"],
                "tiled_decode": selected["deepseek_tiled_decode"],
            },
        )

    def _weight_admission_plan(self, artifact, request):
        if self.kernel_policy["deepseek_expert_layout"] == "tiled64":
            from .deepseek_v41_iq1_arena import tiled_weight_plan

            return admission_plan(
                artifact,
                request,
                planner=lambda item, budget: tiled_weight_plan(
                    item, budget, read_workers=request.read_workers
                ),
            )
        from .deepseek_v41_iq1_weights import compact_weight_plan

        return admission_plan(artifact, request, planner=compact_weight_plan)

    def _construct_weights(self, artifact, request):
        from .deepseek_v41_iq1_weights import CompactV41Weights

        provider = CompactV41Weights
        options = {}
        if self.kernel_policy["deepseek_expert_layout"] == "tiled64":
            from .deepseek_v41_iq1_tiled import TiledV41Weights

            provider = TiledV41Weights
            options["arena_completion"] = self.kernel_policy[
                "deepseek_arena_completion"
            ]
            options["tiled_decode"] = self.kernel_policy["deepseek_tiled_decode"]
        return provider(
            artifact,
            expert_cache_bytes=self.native_admission["expert_cache_bytes"],
            expert_read_workers=request.read_workers,
            expert_prefill_policy=self.kernel_policy["deepseek_expert_prefill"],
            **options,
        )

    def diagnostics(self):
        result = super().diagnostics()
        result.update(
            source_affine_required=False,
            expert_storage="native IQ1_S bytes",
            dense_arithmetic="native F32 affine8 QMM",
            expert_prefill_policy=self.kernel_policy["deepseek_expert_prefill"],
            expert_layout=self.kernel_policy["deepseek_expert_layout"],
            arena_completion=self.kernel_policy["deepseek_arena_completion"],
            tiled_decode=self.kernel_policy["deepseek_tiled_decode"],
        )
        if self.kernel_policy["deepseek_expert_layout"] == "tiled64":
            result.update(
                expert_storage="immutable tiled IQ1_S banks from compact source",
                block_transactions=dict(self.weights.block_transactions),
                tiled_routing="GPU slot map with block-end miss repair",
                asynchronous_epochs=self.kernel_policy["deepseek_arena_completion"]
                == "async",
                expert_arithmetic=(
                    "column F32 group accumulation; distinct numerical profile"
                    if self.kernel_policy["deepseek_tiled_decode"] == "column_f32"
                    else "row-exact F32 SIMD accumulation"
                ),
            )
        return result
