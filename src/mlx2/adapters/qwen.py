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


# ---------------------------------------------------------------------------
# Vendor sampling defaults (see ``mlx2.sampling_defaults``).  Values are the
# vendor's; mlx2 applies them only to fields a request leaves unset.
#
# generation_config.json of every served Qwen3.6/3.8 artifact
# (~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP,
# Qwen3.8-27B-oQ4e-mtp, Qwen3.6-35B-A3B-*): do_sample true, temperature 1.0,
# top_p 0.95, top_k 20 -- the thinking-mode (vendor default mode) profile.
# The remaining fields and the non-thinking profile come from the Qwen model
# cards ("Best Practices" / sampling parameters), read 2026-09-18.
# ---------------------------------------------------------------------------
from ..sampling_defaults import GENERATION_CONFIG, SamplingDefaults, VendorSampling

_FROM_GENERATION_CONFIG = {
    name: GENERATION_CONFIG for name in ("temperature", "top_p", "top_k")
}


def _qwen_thinking(card, presence_penalty):
    return SamplingDefaults(
        temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
        presence_penalty=presence_penalty, repetition_penalty=1.0,
        source=f"model card ({card}): thinking mode",
        field_sources=_FROM_GENERATION_CONFIG,
    )


def _qwen_instruct(card):
    return SamplingDefaults(
        temperature=0.7, top_p=0.8, top_k=20, min_p=0.0,
        presence_penalty=1.5, repetition_penalty=1.0,
        source=f"model card ({card}): instruct (non-thinking) mode",
    )


def qwen38_vendor_sampling(card: str) -> VendorSampling:
    """Qwen3.8 cards (27B, Flash-Next): thinking and instruct profiles.

    Thinking: temperature 1.0, top_p 0.95, top_k 20, min_p 0,
    presence_penalty 0, repetition_penalty 1.0.  Instruct: temperature 0.7,
    top_p 0.8, top_k 20, min_p 0, presence_penalty 1.5,
    repetition_penalty 1.0.  Thinking is the vendor's default mode, so it is
    also the general profile (used for raw completions, whose mode mlx2
    cannot know).
    """
    return VendorSampling(
        {
            "thinking": _qwen_thinking(card, 0.0),
            "instruct": _qwen_instruct(card),
        },
        general="thinking",
        thinking="thinking",
        non_thinking="instruct",
        model=card,
    )


# Qwen/Qwen3.8-Flash-Next model card.
QWEN38_FLASH_NEXT_SAMPLING = qwen38_vendor_sampling("Qwen/Qwen3.8-Flash-Next")
# Qwen/Qwen3.8-27B model card.
QWEN38_27B_SAMPLING = qwen38_vendor_sampling("Qwen/Qwen3.8-27B")

# Qwen/Qwen3.5-9B model card, Best Practices. Unlike Qwen3.8, the
# general-thinking profile uses presence_penalty=1.5 and also declares the
# precise-coding profile used by the wider Qwen3.5 family.
QWEN35_9B_SAMPLING = VendorSampling(
    {
        "thinking": _qwen_thinking("Qwen/Qwen3.5-9B", 1.5),
        "coding": SamplingDefaults(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            source="model card (Qwen/Qwen3.5-9B): thinking mode, precise coding tasks",
        ),
        "instruct": _qwen_instruct("Qwen/Qwen3.5-9B"),
    },
    general="thinking",
    thinking="thinking",
    non_thinking="instruct",
    model="Qwen/Qwen3.5-9B",
)

# Qwen/Qwen3.6-35B-A3B model card: thinking mode for general tasks uses
# presence_penalty 1.5 (unlike Qwen3.8), and a separate thinking-mode profile
# for precise coding tasks (e.g. WebDev): temperature 0.6, top_p 0.95,
# top_k 20, min_p 0, presence_penalty 0, repetition_penalty 1.0.  The coding
# profile is selected per request with ``sampling_profile: "coding"``.
QWEN36_35B_SAMPLING = VendorSampling(
    {
        "thinking": _qwen_thinking("Qwen/Qwen3.6-35B-A3B", 1.5),
        "coding": SamplingDefaults(
            temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
            presence_penalty=0.0, repetition_penalty=1.0,
            source="model card (Qwen/Qwen3.6-35B-A3B): thinking mode, precise coding tasks",
        ),
        "instruct": _qwen_instruct("Qwen/Qwen3.6-35B-A3B"),
    },
    general="thinking",
    thinking="thinking",
    non_thinking="instruct",
    model="Qwen/Qwen3.6-35B-A3B",
)
