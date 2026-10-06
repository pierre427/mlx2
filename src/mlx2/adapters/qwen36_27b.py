"""Revision-bound adapter for the dense Qwen3.6 27B artifacts.

The public checkpoint metadata uses the same ``qwen3_5`` topology as the
later Qwen3.8 27B artifact.  Topology alone therefore cannot select family
defaults.  This adapter is admitted only for the two locally attested config
revisions below; unknown revisions continue through the generic dense-Qwen
resolver and receive no Qwen3.6-specific strategy defaults.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import ClassVar

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..process_env import PROCESS_NUMERICS, require_process_numerics
from .qwen38_27b import Qwen3827BAdapter
from .qwen38_27b import inspect_artifact as inspect_dense_artifact

CACHE_LAYOUT = "qwen36-27b-hybrid-layer-segments-v1"
DEFAULT_PREFILL_STEP = 2048
KNOWN_CONFIG_REVISIONS = frozenset(
    {
        # Qwen3.6-27B-Abliterated-Heretic-Uncensored-MLX-4bit
        "02895cd1ac8eaff1ba9727548f61db21581298b2c68333b12574b01bef49a8b0",
        # Qwen3.6-27B-MLX-8bit
        "93f5fa0d6c683b2278a5505337d0d756bbf4e633f9432dd15633aac50a87e55e",
    }
)


def config_revision(model_path: str | Path) -> str:
    return hashlib.sha256(
        (Path(model_path).expanduser().resolve() / "config.json").read_bytes()
    ).hexdigest()


def is_attested_revision(model_path: str | Path) -> bool:
    return config_revision(model_path) in KNOWN_CONFIG_REVISIONS


def descriptor_for(*, has_mtp: bool = False) -> ModelDescriptor:
    if has_mtp:
        raise ValueError("the attested dense Qwen3.6 artifacts have no MTP tensors")
    return ModelDescriptor(
        model_type="qwen3_5",
        family="qwen3.6-27b",
        variant="dense-ordinary",
        state_planes=frozenset(
            {
                StatePlane.ATTENTION_KV,
                StatePlane.RECURRENT,
                StatePlane.RNG,
                StatePlane.TRANSCRIPT,
            }
        ),
        capabilities=frozenset(
            {
                Capability.TEXT,
                Capability.STREAMING,
                Capability.TOOLS,
                Capability.REASONING,
                Capability.CONTINUOUS_BATCH,
                Capability.PREFIX_REUSE,
                Capability.APC_V2,
                Capability.LAYERED_CACHE,
                Capability.PROMPT_LOOKUP,
                Capability.GRAMMAR,
            }
        ),
        cache_layout=CACHE_LAYOUT,
        metadata={
            "execution": "mlx2.adapters.qwen36_27b.Qwen3627BAdapter",
            "qualification": "continuation-strategy-unqualified",
            "scope": "text-only",
            "mtp": "not-implemented-for-attested-artifacts",
            "continuation_strategy": "implemented-unqualified-unselected",
        },
    )


QWEN36_27B = descriptor_for()


def inspect_artifact(model_path: str | Path) -> dict:
    revision = config_revision(model_path)
    if revision not in KNOWN_CONFIG_REVISIONS:
        raise ValueError(
            "dense Qwen3.6 27B requires an attested artifact config revision"
        )
    artifact = inspect_dense_artifact(model_path)
    if artifact["has_mtp"] or artifact["mtp_tensor_count"]:
        raise ValueError("attested dense Qwen3.6 27B artifacts must be ordinary-only")
    return {**artifact, "family_revision": revision}


def configure_environment() -> dict[str, str]:
    """Pin the ordinary dense-Qwen profile without Qwen3.8 MTP defaults."""
    require_process_numerics("the dense Qwen3.6 27B profile")
    profile = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        **PROCESS_NUMERICS,
        "MLX_GDN_PACKED": "1",
        "MLX_GDN_CORE": "0",
        "MLX_LM_COMPILED_DECODE": "0",
        "MLX_LM_SEGMENTED_SELF_MTP": "0",
        "MLX_LM_TRUE_BATCHED_SEGMENTED_MTP": "0",
        "MLX_LM_SHARED_QSA_SUFFIX": "0",
        "MLX_LM_MTP_BOUNDARY_COW": "0",
    }
    for name in tuple(os.environ):
        if name.startswith(("MLX_QWEN", "MLX_LM_", "MLXUAG_", "MLX_GDN_")):
            del os.environ[name]
    os.environ.update(profile)
    return profile


class Qwen3627BAdapter(Qwen3827BAdapter):
    """Dense Qwen3.6 adapter with family-owned serving decisions."""

    default_route = "ordinary"
    default_route_execution_policy: ClassVar[dict] = {}
    descriptor = QWEN36_27B
    artifact_inspector = staticmethod(inspect_artifact)
    descriptor_builder = staticmethod(descriptor_for)
    environment_configurator = staticmethod(configure_environment)
    fused_gdn_architecture = "qwen38"
    from .qwen import QWEN36_27B_SAMPLING as sampling_defaults

    def prefill_step_default(self):
        """Retain the artifact family's exercised 2,048-row prefill geometry."""
        return DEFAULT_PREFILL_STEP

    def continuation_verification_strategy(self):
        """Exact request-private candidate strategy; default serving stays ordinary."""
        from ..runtime.continuation_strategy import ContinuationStrategy

        return ContinuationStrategy(
            algorithm="longest_first_exact_prefix_v1",
            prune_incompatible_siblings=True,
            shared_prefix_reuse=True,
            state_scope="request_private_exact_recovery_descriptors",
            qualified=False,
        )

    def create_external_batch(self, **kwargs):
        if hasattr(
            getattr(self, "draft_model", None), "last_continuation_selections"
        ):
            kwargs.setdefault(
                "continuation_verification_strategy",
                self.continuation_verification_strategy(),
            )
        return super().create_external_batch(**kwargs)

    def profile_name(self, mtp):
        if mtp:
            raise ValueError("dense Qwen3.6 27B attested artifacts are ordinary-only")
        return "qwen36-27b-apcv2-ordinary"
