# SPDX-License-Identifier: MIT
# Mined closure; provenance/muse-dflash2.json.
import inspect
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar

from .base_config import DFlashConfig, check_dflash_window_contract, resolve_num_target_layers


def _is_strictly_increasing_ints(values: list[int]) -> bool:
    return all(
        isinstance(value, int) and (index == 0 or values[index - 1] < value)
        for index, value in enumerate(values)
    )


def _refuse_other_draft_contracts(flat: Mapping, dflash: Mapping) -> None:
    """Refuse checkpoint metadata declaring a draft contract mlx2 does not run.

    The DFlash2 drafter drafts the anchor plus ``block - 1`` mask rows and
    drops the anchor row, with full-head RoPE.  A DSpark-style
    ``sample_from_anchor`` block (oMLX #4525) or partial RoPE (oMLX #4508)
    changes every proposal; ignoring either key would serve the checkpoint
    with a layout it was not trained with.
    """
    if flat.get("sample_from_anchor") or dflash.get("sample_from_anchor"):
        raise ValueError("DFlash2 does not implement sample_from_anchor draft blocks.")
    factors = [flat.get("partial_rotary_factor")]
    for key in ("rope_parameters", "rope_scaling"):
        nested = flat.get(key)
        if isinstance(nested, Mapping):
            factors.append(nested.get("partial_rotary_factor"))
    if any(factor is not None and float(factor) != 1.0 for factor in factors):
        raise ValueError("DFlash2 supports full-head RoPE only (partial_rotary_factor).")


# Known selector softcap placements; anything else is refused.
SELECTOR_SOFTCAP_MODES = ("before_correction", "after_correction")


@dataclass
class DFlash2Config(DFlashConfig):
    # Existing sixteen-position anchor/mask lattice.  Keep block_size as the
    # artifact's training metadata; the length-generic runtime supports this
    # separately bounded proposal domain without adding a new tensor law.
    maximum_proposal_length: ClassVar[int] = 15
    supports_causal_blocks: ClassVar[bool] = False
    model_type: str = "dflash2"
    backbone_model_type: str = "qwen3"
    architectures: list[str] = field(default_factory=list)
    hidden_act: str = "silu"
    rope_is_neox_style: bool = True
    input_embedding_scale: float = 1.0
    output_multiplier: float = 1.0
    conv_kernel_size: int = 0
    conv_group_size: int = 0
    selector_rank: int = 0
    selector_top_k: int = 0
    is_causal: bool = False
    # Draft layers without positional encoding (Ripple's DFlash2 global
    # layer; SGLang's DFlash path needed the same patch). Unknown config keys
    # are dropped by ``from_dict``, so this must be a declared field: a silent
    # drop would rotate a NoPE layer and cut acceptance without any error.
    no_rope_layers: list[int] = field(default_factory=list)
    # Where ``final_logit_softcapping`` applies (mlx2, Inkling 2 DFlash2
    # release).  "before_correction" (also the absent key) caps the drafter
    # logits, so the selector's unary term is already capped: today's law.
    # "after_correction" leaves the drafter logits uncapped and caps the
    # selector's path score (unary plus the pairwise correction) instead.
    # A declared field: ``from_dict`` drops unknown keys, and a silent drop
    # would serve the wrong selector law with no error.
    selector_softcap_mode: str = "before_correction"

    def __post_init__(self) -> None:
        if not self.layer_types:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        self.validate()

    def validate(self) -> None:
        positive_fields = (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "max_position_embeddings",
            "block_size",
            "num_target_layers",
            "conv_kernel_size",
            "conv_group_size",
            "selector_rank",
            "selector_top_k",
            "target_streams_per_layer",
        )
        for key in positive_fields:
            value = getattr(self, key)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"DFlash2 requires a positive integer {key}.")
        if self.model_type != "dflash2":
            raise ValueError(
                f"DFlash2 requires normalized model_type='dflash2', got "
                f"{self.model_type!r}."
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("DFlash2 attention heads must be divisible by KV heads.")
        if self.hidden_size % self.conv_group_size:
            raise ValueError(
                "DFlash2 hidden_size must be divisible by conv_group_size."
            )
        if self.hidden_act != "silu":
            raise ValueError("DFlash2 currently supports only hidden_act='silu'.")
        if not 0 < self.selector_top_k <= self.vocab_size:
            raise ValueError("DFlash2 selector_top_k must fit inside the vocabulary.")
        if not 0 <= self.mask_token_id < self.vocab_size:
            raise ValueError("DFlash2 mask_token_id must be inside the vocabulary.")
        if len(self.layer_types) != self.num_hidden_layers or any(
            layer_type not in {"full_attention", "sliding_attention"}
            for layer_type in self.layer_types
        ):
            raise ValueError(
                "DFlash2 requires one supported attention type per draft layer."
            )
        if "sliding_attention" in self.layer_types and self.sliding_window is None:
            raise ValueError(
                "DFlash2 requires sliding_window for sliding_attention layers."
            )
        if (
            not self.target_layer_ids
            or not _is_strictly_increasing_ints(self.target_layer_ids)
            or any(
                layer_id < 0 or layer_id > self.num_target_layers
                for layer_id in self.target_layer_ids
            )
        ):
            raise ValueError(
                "DFlash2 target_layer_ids must be unique, increasing indices "
                "inside the target layer range."
            )
        # An id equal to ``num_target_layers`` is the target's final-normed
        # hidden (the Inkling 2 release taps visit 48 of 48).  The config
        # admits it; binding admits it only for a target that declares the
        # final-norm tap (``adapters/dflash2.py``), so every other target
        # still refuses it.
        if (
            not isinstance(self.no_rope_layers, (list, tuple))
            or any(type(i) is not int or not 0 <= i < self.num_hidden_layers for i in self.no_rope_layers)
            or len(set(self.no_rope_layers)) != len(self.no_rope_layers)
        ):
            raise ValueError("DFlash2 no_rope_layers must be unique draft layer indices.")
        if self.runtime_block_size is not None and (
            not isinstance(self.runtime_block_size, int)
            or not 2 <= self.runtime_block_size <= self.block_size
        ):
            raise ValueError(
                "DFlash2 runtime_block_size must be between 2 and block_size."
            )
        if self.selector_softcap_mode not in SELECTOR_SOFTCAP_MODES:
            raise ValueError(
                "DFlash2 selector_softcap_mode must be one of "
                f"{list(SELECTOR_SOFTCAP_MODES)}, got {self.selector_softcap_mode!r}."
            )
        cap = self.final_logit_softcapping
        if self.selector_softcap_mode == "after_correction" and (
            type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0
        ):
            raise ValueError(
                "DFlash2 selector_softcap_mode 'after_correction' needs a "
                "positive final_logit_softcapping."
            )
        if self.is_causal and not self.supports_causal_blocks:
            raise ValueError("DFlash2 causal draft blocks are not currently supported.")

    @property
    def taps_final_norm(self) -> bool:
        """Whether the last tap is the target's final-normed hidden."""
        return self.target_layer_ids[-1] == self.num_target_layers

    @classmethod
    def from_dict(cls, params: dict, *, num_target_layers: int | None = None) -> "DFlash2Config":
        """``num_target_layers``: the target's tap count, used only when the
        config declares none (mlx-vlm #2841; ``resolve_num_target_layers``)."""
        flat = dict(params)
        dflash = flat.pop("dflash_config", None)
        if not isinstance(dflash, Mapping):
            raise TypeError("DFlash2 requires a dflash_config object.")
        flat["num_target_layers"] = resolve_num_target_layers(
            flat, dflash, num_target_layers, label="DFlash2"
        )

        _refuse_other_draft_contracts(flat, dflash)
        check_dflash_window_contract(flat, dflash)
        # Every spelling of block causality the DFlash family uses selects
        # ``is_causal``, which ``validate`` refuses: the drafter implements
        # only the bidirectional block, and serving a causally trained
        # checkpoint bidirectionally (or the reverse, vLLM #61537) would
        # silently cut acceptance.
        if any(
            bool(value)
            for value in (
                dflash.get("causal"),
                flat.get("dflash_query_causal"),
                flat.get("is_causal"),
            )
        ):
            flat["is_causal"] = True

        flat["backbone_model_type"] = str(flat.pop("model_type", "qwen3"))
        flat["model_type"] = "dflash2"
        for key in (
            "block_size",
            "mask_token_id",
            "target_layer_ids",
            "runtime_block_size",
            "draft_window_size",
            "final_logit_softcapping",
            "input_embedding_scale",
            "output_multiplier",
            "conv_kernel_size",
            "conv_group_size",
            "selector_rank",
            "selector_top_k",
            "target_streams_per_layer",
            "selector_softcap_mode",
        ):
            if key in dflash:
                flat[key] = dflash[key]

        rope_parameters = flat.pop("rope_parameters", None)
        if isinstance(rope_parameters, Mapping):
            rope_parameters = dict(rope_parameters)
            if "rope_theta" in rope_parameters:
                flat["rope_theta"] = rope_parameters.pop("rope_theta")
            flat["rope_scaling"] = rope_parameters

        if "runtime_block_size" not in flat:
            flat["runtime_block_size"] = min(5, int(flat["block_size"]))

        signature = inspect.signature(cls).parameters
        return cls(**{key: value for key, value in flat.items() if key in signature})

    from_hf_dict = from_dict


def check_target_taps(args, *, layer_count, tap_count=None, final_norm_tap=False):
    """Refuse a drafter whose tap space differs from the target's.

    ``tap_count`` is a target-supplied count of tappable positions in
    execution order (a looped target's layer visits); None means the
    physical layer count, today's rule.  ``final_norm_tap`` says the target
    serves tap index ``tap_count`` as its final-normed hidden; without it a
    drafter that taps that index is refused.
    """
    expected = layer_count if tap_count is None else tap_count
    if type(expected) is not int or expected <= 0:
        raise ValueError("DFlash2 target tap count must be a positive integer")
    if expected != args.num_target_layers:
        raise ValueError(
            "DFlash2 target layer count mismatch"
            if tap_count is None
            else f"DFlash2 target visit count mismatch ({args.num_target_layers} != {expected})"
        )
    if getattr(args, "taps_final_norm", False) and not final_norm_tap:
        raise ValueError("DFlash2 drafter taps the final-norm state; the target declares no such tap")


__all__ = ["DFlash2Config", "SELECTOR_SOFTCAP_MODES", "check_target_taps"]
