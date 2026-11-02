"""Pinned GLM external heads; preserve each trained block-attention contract."""

from ..runtime.drafters.dflash2_config import DFlash2Config
from .dflash2 import inspect_drafter, load_drafter


class Glm53DFlashConfig(DFlash2Config):
    # Opt-in backend: DFlashAttention._proposal_mask supplies the causal and
    # sliding visibility law. Other DFlash2 families retain their refusal.
    supports_causal_blocks = True


def inspect_glm53_drafter(path, target):
    record = inspect_drafter(path, target, config_type=Glm53DFlashConfig)
    args = record["args"]
    if (
        args.num_target_layers != 45
        or args.hidden_size != 4096
        or args.vocab_size != 154880
        or args.block_size != 8
        or args.target_layer_ids not in ([5, 14, 24, 33, 42], [23, 27, 31, 35, 39, 43])
    ):
        raise ValueError(
            "GLM-5.3 drafter topology differs from inspected released heads"
        )
    if args.is_causal != (args.target_layer_ids == [23, 27, 31, 35, 39, 43]):
        raise ValueError("GLM-5.3 drafter trained mask contract changed")
    return record


def load_glm53_drafter(record, target, *, activation="target"):
    # Retain BF16 weights; use stock mask-aware attention and full target head.
    if activation not in ("target", "bfloat16"):
        raise ValueError("unsupported GLM external draft activation dtype")
    model = load_drafter(record, target)
    if activation == "bfloat16":
        import mlx.core as mx

        object.__setattr__(model, "target_activation_dtype", mx.bfloat16)
    return model
