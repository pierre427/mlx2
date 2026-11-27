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


def resolve_num_target_layers(flat: Mapping, dflash: Mapping, target_layers=None, *, label="DFlash"):
    """The target layer count a DFlash drafter config declares, or the target's.

    ``num_target_layers`` may sit at the top level or in ``dflash_config``;
    both spellings must agree.  An explicit value is authoritative (a target
    whose depth differs is refused at binding).  When neither is present the
    count comes from the target (``target_layers``); without one the config
    is refused rather than assuming the dataclass default of 32, which would
    mis-range every tap of a deeper target (mlx-vlm #2841).
    """
    top = flat.get("num_target_layers")
    nested = dflash.get("num_target_layers")
    if top is not None and nested is not None and top != nested:
        raise ValueError(
            f"{label} num_target_layers conflict: top level {top!r} != dflash_config {nested!r}."
        )
    declared = nested if nested is not None else top
    if declared is not None:
        return declared
    if target_layers is None:
        raise ValueError(
            f"{label} config omits num_target_layers and no target layer count was "
            "supplied; refusing to assume one."
        )
    if type(target_layers) is not int or target_layers <= 0:
        raise ValueError(f"{label} target layer count must be a positive integer.")
    return target_layers


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
    # Residual streams per tapped target layer.  A hyper-connection target
    # (Raon-1-mini's 4-stream mHC) feeds each tap's full n-stream residual,
    # so one tap is ``n * hidden_size`` wide (``dflash_config.
    # target_streams_per_layer``).  1 is every single-residual target.
    target_streams_per_layer: int = 1

    @property
    def feature_width(self) -> int:
        """Width of one concatenated target feature row (the ``fc`` input)."""
        return len(self.target_layer_ids) * self.hidden_size * int(self.target_streams_per_layer)

    @classmethod
    def from_dict(cls, params: dict, *, num_target_layers: int | None = None) -> "DFlashConfig":
        """``num_target_layers``: the target's layer count, used only when the
        config declares none (see ``resolve_num_target_layers``)."""
        flat = dict(params)
        dflash_cfg = flat.pop("dflash_config", None) or {}
        check_dflash_window_contract(flat, dflash_cfg)
        flat["num_target_layers"] = resolve_num_target_layers(
            flat, dflash_cfg, num_target_layers, label="DFlash"
        )
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
        if "target_streams_per_layer" in dflash_cfg:
            flat["target_streams_per_layer"] = dflash_cfg["target_streams_per_layer"]
        sig = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in flat.items() if k in sig})

    from_hf_dict = from_dict
