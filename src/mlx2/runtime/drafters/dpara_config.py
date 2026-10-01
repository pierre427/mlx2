# SPDX-License-Identifier: Apache-2.0
"""Explicit M-DFlash compatibility; ordinary DFlash weights are insufficient."""

import math
from dataclasses import dataclass

from .base_config import DFlashConfig


@dataclass
class DParaConfig(DFlashConfig):
    markov_rank: int = 128
    backbone_revision: str = "synthetic"
    target_revision: str = "synthetic-target"
    training_regime: str = "multiple_anchor_featureless"

    def __post_init__(self):
        dimensions = (
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.vocab_size,
            self.markov_rank,
            self.block_size,
        )
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError("DPara dimensions must be positive integers")
        if self.block_size < 2 or self.head_dim % 2:
            raise ValueError(
                "DPara requires a positive draft length and even RoPE dimension"
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("DPara query heads must be divisible by KV heads")
        if not 0 <= self.mask_token_id < self.vocab_size:
            raise ValueError("DPara mask token is outside vocabulary")
        if (
            not self.target_layer_ids
            or any(
                type(index) is not int or not 0 <= index < self.num_target_layers
                for index in self.target_layer_ids
            )
            or len(set(self.target_layer_ids)) != len(self.target_layer_ids)
        ):
            raise ValueError("DPara needs target feature taps")
        if self.layer_types and (
            len(self.layer_types) != self.num_hidden_layers
            or any(kind != "full_attention" for kind in self.layer_types)
        ):
            raise ValueError("DPara candidate supports full attention only")
        if self.draft_window_size or self.sliding_window:
            raise ValueError("DPara candidate does not implement context windows")
        if (
            self.rope_scaling
            and self.rope_scaling.get("rope_type", "default") != "default"
        ):
            raise ValueError("DPara candidate supports default RoPE only")
        if self.training_regime != "multiple_anchor_featureless":
            raise ValueError(
                "DPara requires a trained multiple-anchor featureless backbone"
            )
        if not isinstance(self.backbone_revision, str) or not self.backbone_revision:
            raise ValueError("DPara requires a backbone revision")
        if not isinstance(self.target_revision, str) or not self.target_revision:
            raise ValueError("DPara requires a target revision")
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
            for value in (self.rope_theta, self.rms_norm_eps)
        ):
            raise ValueError(
                "DPara RoPE base and RMS epsilon must be finite positive numbers"
            )
        if self.final_logit_softcapping is not None and (
            not isinstance(self.final_logit_softcapping, (int, float))
            or isinstance(self.final_logit_softcapping, bool)
            or not math.isfinite(self.final_logit_softcapping)
            or self.final_logit_softcapping <= 0
        ):
            raise ValueError("DPara logit softcap must be finite and positive")

    @classmethod
    def from_dict(cls, params):
        """Refuse a DSpark/DFlash checkpoint before any payload allocation.

        The marker is a compatibility declaration, never qualification evidence.
        Direct construction is intentionally available for synthetic tests.
        """
        marker = params.get("dpara_config")
        if (
            params.get("model_type") != "dpara"
            or not isinstance(marker, dict)
            or marker.get("training_regime") != "multiple_anchor_featureless"
            or not marker.get("backbone_revision")
            or marker.get("backbone_revision") == "synthetic"
            or not marker.get("target_revision")
            or marker.get("markov_head") != "low_rank_previous_token"
        ):
            raise ValueError("Missing compatible trained M-DFlash backbone declaration")
        values = dict(params)
        values.update(
            {
                name: marker[name]
                for name in (
                    "training_regime",
                    "backbone_revision",
                    "target_revision",
                    "markov_rank",
                )
                if name in marker
            }
        )
        return super().from_dict(values)

    from_hf_dict = from_dict
