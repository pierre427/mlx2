"""Declared Qwen capabilities; selectable profiles require separate evidence."""

from ..contracts import Capability, ModelDescriptor, StatePlane


QWEN4_FLASH_NEXT = ModelDescriptor(
    model_type="qwen4_exp",
    family="qwen4-flash-next",
    variant="flash-next",
    state_planes=frozenset(
        {
            StatePlane.ATTENTION_KV,
            StatePlane.RECURRENT,
            StatePlane.SPARSE_INDEX,
            StatePlane.DRAFT,
            StatePlane.RNG,
            StatePlane.TRANSCRIPT,
            StatePlane.GRAMMAR,
        }
    ),
    capabilities=frozenset(
        {
            Capability.TEXT,
            Capability.CONTINUOUS_BATCH,
            Capability.PREFIX_REUSE,
            Capability.APC_V2,
            Capability.MTP,
            Capability.PROMPT_LOOKUP,
            Capability.STREAMING,
            Capability.TOOLS,
            Capability.REASONING,
            Capability.LAYERED_CACHE,
            Capability.GRAMMAR,
            Capability.SEGMENTED_MTP,
            Capability.COMPACTION,
        }
    ),
    cache_layout="qwen4-exp-layer-segments-v1",
    metadata={"execution": "mlx2.adapters.flash_next.FlashNextAdapter"},
)


QWEN36 = ModelDescriptor(
    model_type="qwen3_5_moe",
    family="qwen3.6",
    variant="base",
    state_planes=frozenset(
        {StatePlane.ATTENTION_KV, StatePlane.RECURRENT, StatePlane.RNG}
    ),
    capabilities=frozenset(
        {Capability.TEXT, Capability.CONTINUOUS_BATCH, Capability.PREFIX_REUSE}
    ),
    metadata={"execution": "pending", "apc_v2_bridge": "pending"},
)
