"""Conservative ordinary and native-MTP cache projection for Nemotron 3 Super."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class NemotronCacheBudget:
    attention_layers: int
    recurrent_layers: int
    mtp_attention_layers: int
    kv_heads: int
    head_dim: int
    mamba_heads: int
    mamba_head_dim: int
    ssm_state_size: int
    groups: int
    conv_kernel: int
    item_bytes: int = 4
    allocation_step: int = 256
    transcript_bytes_per_token: int = 16
    # A conservative interim allowance: the first full-model B1 MTP3 direct
    # run peaked 2.40 GiB above loaded active memory, including allocator and
    # first-graph effects. The shared memory controller scales this 1.76 GiB
    # K2 value by 1.25 at depth 3 and retains 20 GiB of host/driver reserve.
    # Recalibrate with simultaneous B2-B4 device measurements before raising
    # the concurrency cap further.
    transient_gib_per_lane: float = 1.76

    @classmethod
    def from_config(cls, config, *, mtp=False):
        pattern = config.hybrid_override_pattern
        if len(pattern) != 88 or pattern.count("M") != 40 or pattern.count("*") != 8:
            raise ValueError("unknown Nemotron 3 Super cache topology")
        return cls(
            attention_layers=8, recurrent_layers=40,
            mtp_attention_layers=1 if mtp else 0,
            kv_heads=config.num_key_value_heads, head_dim=config.head_dim,
            mamba_heads=config.mamba_num_heads, mamba_head_dim=config.mamba_head_dim,
            ssm_state_size=config.ssm_state_size, groups=config.n_groups,
            conv_kernel=config.conv_kernel,
        )

    @property
    def recurrent_live_bytes(self):
        conv_dim = self.mamba_heads * self.mamba_head_dim + 2 * self.groups * self.ssm_state_size
        state = self.mamba_heads * self.mamba_head_dim * self.ssm_state_size
        return self.recurrent_layers * (state + (self.conv_kernel - 1) * conv_dim) * self.item_bytes

    @property
    def speculative_scratch_bytes(self):
        # Each Mamba layer retains its pre-verify state for exact rollback.
        return self.recurrent_live_bytes if self.mtp_attention_layers else 0

    @property
    def ledger_bytes(self):
        return (self.attention_layers + self.recurrent_layers + self.mtp_attention_layers) * 4096

    @property
    def resident_fixed_bytes(self):
        return self.recurrent_live_bytes + self.ledger_bytes

    @property
    def fixed_bytes(self):
        return self.resident_fixed_bytes + self.speculative_scratch_bytes

    def _variable_bytes(self, context_tokens):
        capacity = (math.ceil(context_tokens / self.allocation_step) + 1) * self.allocation_step
        return capacity * ((self.attention_layers + self.mtp_attention_layers) * 2 * self.kv_heads * self.head_dim * self.item_bytes + self.transcript_bytes_per_token)

    @staticmethod
    def _validate_context(value):
        if type(value) is not int or value < 0:
            raise ValueError("context_tokens must be a nonnegative integer")

    def project_resident(self, context_tokens):
        self._validate_context(context_tokens)
        return self.resident_fixed_bytes + self._variable_bytes(context_tokens)

    def project(self, context_tokens):
        return self.project_resident(context_tokens) + self.speculative_scratch_bytes

    def project_pool(self, context_tokens, *, resident_lanes, running_lanes):
        self._validate_context(context_tokens)
        if type(resident_lanes) is not int or type(running_lanes) is not int or not 0 <= running_lanes <= resident_lanes:
            raise ValueError("invalid resident/running lane counts")
        return (resident_lanes * self.project_resident(context_tokens)
                + running_lanes * self.speculative_scratch_bytes)

    def as_dict(self):
        return {"schema": "nemotron3-super-cache-geometry-v1", **asdict(self),
                "fixed_bytes": self.fixed_bytes,
                "resident_fixed_bytes": self.resident_fixed_bytes,
                "speculative_scratch_bytes": self.speculative_scratch_bytes,
                "scratch_scope": "running-mtp-lanes-only",
                "bound": "fp32-recurrent-and-kv-capacity-plus-mtp-rollback-and-forward-reserve"}
