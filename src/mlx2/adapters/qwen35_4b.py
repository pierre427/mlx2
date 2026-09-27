"""CPU-safe Qwen3.5 4B artifact gate and ordinary text adapter."""

from __future__ import annotations

from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from .qwen35_9b import Qwen359BAdapter, configure_environment
from .qwen35_9b import inspect_artifact as inspect_dense_artifact


CACHE_LAYOUT = "qwen35-4b-hybrid-layer-segments-v1"
_TOPOLOGY = {
    "num_hidden_layers": 32,
    "hidden_size": 2560,
    "intermediate_size": 9216,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "full_attention_interval": 4,
    "vocab_size": 248320,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
}


def inspect_artifact(model_path: str | Path) -> dict:
    return inspect_dense_artifact(
        model_path, expected_topology=_TOPOLOGY, family="4B"
    )


def descriptor_for(*, has_mtp: bool = False) -> ModelDescriptor:
    if has_mtp:
        raise ValueError("Qwen3.5 4B MTP is not implemented")
    return ModelDescriptor(
        model_type="qwen3_5",
        family="qwen3.5-4b",
        variant="4b-ordinary",
        state_planes=frozenset(
            {StatePlane.ATTENTION_KV, StatePlane.RECURRENT, StatePlane.RNG, StatePlane.TRANSCRIPT}
        ),
        capabilities=frozenset(
            {
                Capability.TEXT, Capability.STREAMING, Capability.TOOLS,
                Capability.REASONING, Capability.CONTINUOUS_BATCH,
                Capability.PREFIX_REUSE, Capability.APC_V2,
                Capability.LAYERED_CACHE, Capability.PROMPT_LOOKUP,
                Capability.GRAMMAR,
            }
        ),
        cache_layout=CACHE_LAYOUT,
        metadata={
            "execution": "mlx2.adapters.qwen35_4b.Qwen354BAdapter",
            "qualification": "pending",
            "scope": "text-only ordinary decode",
            "mtp": "not-implemented",
            "vision": "not-implemented",
        },
    )


QWEN35_4B = descriptor_for()


class Qwen354BAdapter(Qwen359BAdapter):
    """Use the parameterized Qwen3.5 hybrid runtime at the 4B geometry."""

    default_route = "ordinary"
    default_route_execution_policy = {}
    descriptor = QWEN35_4B
    artifact_inspector = staticmethod(inspect_artifact)
    descriptor_builder = staticmethod(descriptor_for)
    environment_configurator = staticmethod(configure_environment)
    # Do not inherit the 9B model card's vendor sampling profile.
    sampling_defaults = None

    def profile_name(self, mtp):
        if mtp:
            raise ValueError("Qwen3.5 4B MTP is not implemented")
        return "qwen35-4b-apcv2-ordinary"

    def approximate_kv_operations(self):
        return {}
