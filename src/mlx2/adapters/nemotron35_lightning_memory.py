"""Conservative cache geometry for the 52-layer Nemotron 3.5 Lightning target."""
from __future__ import annotations

from dataclasses import dataclass

from .nemotron3_super_memory import NemotronCacheBudget


@dataclass(frozen=True)
class LightningCacheBudget(NemotronCacheBudget):
    # Source inspection cannot establish the target's actual peak forward
    # memory. Retain a conservative per-running-lane allowance until a
    # source-bound device qualification measures it.
    transient_gib_per_lane: float = 4.0

    @classmethod
    def from_config(cls, config, *, mtp=False):
        pattern = config.hybrid_override_pattern
        if (not isinstance(pattern, list) or len(pattern) != 52
                or pattern.count("M") != 23 or pattern.count("*") != 6
                or pattern.count("E") != 23):
            raise ValueError("unknown Nemotron 3.5 Lightning cache topology")
        return cls(
            attention_layers=6,
            recurrent_layers=23,
            mtp_attention_layers=1 if mtp else 0,
            kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            mamba_heads=config.mamba_num_heads,
            mamba_head_dim=config.mamba_head_dim,
            ssm_state_size=config.ssm_state_size,
            groups=config.n_groups,
            conv_kernel=config.conv_kernel,
        )

    def as_dict(self):
        return {
            **super().as_dict(),
            "schema": "nemotron35-lightning-cache-geometry-v1",
            "transient_bound": "conservative-unmeasured-allowance",
        }
