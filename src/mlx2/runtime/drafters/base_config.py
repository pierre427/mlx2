# SPDX-License-Identifier: MIT
# Mined closure; provenance/muse-dflash2.json.
import inspect
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional


def check_dflash_window_contract(flat: Mapping, dflash: Mapping) -> None:
    """Refuse a sliding window that ``layer_types`` does not carry.

    The drafters window exactly the layers typed ``sliding_attention``.  Some
    DFlash2 checkpoints declare the window only as ``dflash_config.use_swa``
    with every layer ``full_attention`` (TensorRT-LLM #21367 windows every
    layer for them); serving one here would silently drop the window, so it
    is refused, as is a ``use_swa`` or window size the layer types contradict.
    """
    sliding = "sliding_attention" in (flat.get("layer_types") or ())
    use_swa = dflash.get("use_swa")
    if use_swa is not None and bool(use_swa) != sliding:
        raise ValueError(
            f"DFlash dflash_config.use_swa={use_swa!r} contradicts layer_types "
            f"({'some' if sliding else 'no'} sliding_attention layers); refusing to guess the window."
        )
    window = dflash.get("swa_window_size")
    if window is not None and sliding and window != flat.get("sliding_window"):
        raise ValueError(
            f"DFlash dflash_config.swa_window_size={window!r} differs from sliding_window="
            f"{flat.get('sliding_window')!r}."
        )


@dataclass
class DFlashConfig:
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 5
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    rope_theta: float = 10000000.0
    rope_scaling: Optional[dict[str, Any]] = None
    attention_bias: bool = False
    tie_word_embeddings: bool = True
    block_size: int = 16
    mask_token_id: int = 248070
    target_layer_ids: List[int] = field(default_factory=lambda: [1, 8, 15, 22, 29])
    num_target_layers: int = 32
    layer_types: List[str] = field(default_factory=list)
    sliding_window: Optional[int] = None
    final_logit_softcapping: Optional[float] = None
    runtime_block_size: int | None = None
    draft_window_size: int | None = None

    @classmethod
    def from_dict(cls, params: dict) -> "DFlashConfig":
        flat = dict(params)
        dflash_cfg = flat.pop("dflash_config", None) or {}
        check_dflash_window_contract(flat, dflash_cfg)
        # This parser applies no logit transform; the DFlash2 parser does.
        for key, identity in (("output_multiplier", 1.0), ("final_logit_softcapping", None)):
            if dflash_cfg.get(key, identity) != identity:
                raise ValueError(f"DFlash dflash_config.{key}={dflash_cfg[key]!r} is not applied by this drafter.")
        if "mask_token_id" in dflash_cfg:
            flat["mask_token_id"] = dflash_cfg["mask_token_id"]
        if "target_layer_ids" in dflash_cfg:
            flat["target_layer_ids"] = list(dflash_cfg["target_layer_ids"])
        if "runtime_block_size" in dflash_cfg:
            flat["runtime_block_size"] = dflash_cfg["runtime_block_size"]
        if "draft_window_size" in dflash_cfg:
            flat["draft_window_size"] = dflash_cfg["draft_window_size"]
        sig = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in flat.items() if k in sig})

    from_hf_dict = from_dict
