"""Configuration-derived cache bound for dense Qwen3.8 hybrid models.

Full-attention target and embedded-MTP K/V capacity is charged at fp32.
GDN convolution and recurrent state is fixed-size and charged twice so a live
state and its speculative rollback snapshot fit together.  The shared
controller separately retains its 20 GiB host/driver reserve and charges the
dense-family forward workspace configured here.
"""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Qwen38CacheBudget:
    attention_layers: int
    recurrent_layers: int
    mtp_layers: int
    kv_heads: int
    head_dim: int
    recurrent_heads: int
    recurrent_key_heads: int
    recurrent_key_dim: int
    recurrent_value_dim: int
    conv_kernel: int
    item_bytes: int = 4
    allocation_step: int = 256
    transcript_bytes_per_token: int = 16
    transient_gib_per_lane: float = 3.1
    # Chunked-prefill transient, measured on a 36 GiB M3 Pro (2026-09-25,
    # Qwen3.8-27B-CRACK-MLX-4bit, 16K prompt, 2048-token chunks, MLX memory
    # limit 24.08 GiB): wired memory peaked 2.0-2.8 GB above each chunk's
    # trough, the peaks rising with context (26.7 -> 29.4 GB over 16K), and a
    # 32K prefill without the limit reached 5.5 GB.  Two-second sampling
    # misses the true spikes, so the charge is set above every observation:
    # a fixed 2.0 GiB per full chunk (scaled by the chunk's rows) plus one
    # extra bf16 copy of the attention K/V at the prompt's length, which a
    # growing cache reallocates and copies while the old buffer is live.
    prefill_chunk_transient_gib: float = 2.0
    prefill_chunk_rows: int = 2048

    @classmethod
    def from_config(cls, config, *, mtp):
        layer_count = int(config["num_hidden_layers"])
        interval = int(config["full_attention_interval"])
        if layer_count < 1 or interval < 1 or layer_count % interval:
            raise ValueError("unknown Qwen3.8 hybrid cache topology")
        layer_types = config.get("layer_types")
        expected = [
            "full_attention" if (index + 1) % interval == 0 else "linear_attention"
            for index in range(layer_count)
        ]
        if layer_types is not None and layer_types != expected:
            raise ValueError("unknown Qwen3.8 hybrid cache topology")
        attention_layers = layer_count // interval
        mtp_layers = int(config.get("mtp_num_hidden_layers", 0)) if mtp else 0
        result = cls(
            attention_layers=attention_layers,
            recurrent_layers=layer_count - attention_layers,
            mtp_layers=mtp_layers,
            kv_heads=int(config["num_key_value_heads"]),
            head_dim=int(config["head_dim"]),
            recurrent_heads=int(config["linear_num_value_heads"]),
            recurrent_key_heads=int(config["linear_num_key_heads"]),
            recurrent_key_dim=int(config["linear_key_head_dim"]),
            recurrent_value_dim=int(config["linear_value_head_dim"]),
            conv_kernel=int(config["linear_conv_kernel_dim"]),
        )
        numeric = asdict(result)
        if any(
            (not math.isfinite(value)) or value < 0
            for value in numeric.values()
            if isinstance(value, (int, float))
        ) or not all(
            (
                result.attention_layers,
                result.recurrent_layers,
                result.kv_heads,
                result.head_dim,
                result.recurrent_heads,
                result.recurrent_key_heads,
                result.recurrent_key_dim,
                result.recurrent_value_dim,
                result.conv_kernel,
                result.item_bytes,
                result.allocation_step,
            )
        ):
            raise ValueError("invalid Qwen3.8 cache dimensions")
        return result

    @property
    def recurrent_live_bytes(self):
        recurrent = (
            self.recurrent_heads * self.recurrent_value_dim * self.recurrent_key_dim
        )
        conv_dim = (
            2 * self.recurrent_key_heads * self.recurrent_key_dim
            + self.recurrent_heads * self.recurrent_value_dim
        )
        per_layer = recurrent + max(0, self.conv_kernel - 1) * conv_dim
        return self.recurrent_layers * per_layer * self.item_bytes

    @property
    def speculative_scratch_bytes(self):
        """Rollback state required only while a lane is running speculation."""
        return self.recurrent_live_bytes

    @property
    def ledger_bytes(self):
        return (self.attention_layers + self.recurrent_layers + self.mtp_layers) * 4096

    @property
    def resident_fixed_bytes(self):
        """Fixed bytes retained by a resident, non-running cache entry."""
        return self.recurrent_live_bytes + self.ledger_bytes

    @property
    def fixed_bytes(self):
        """Fixed bytes for one actively running speculative lane."""
        return self.resident_fixed_bytes + self.speculative_scratch_bytes

    def _variable_bytes(self, context_tokens):
        capacity = (
            math.ceil(context_tokens / self.allocation_step) * self.allocation_step
            + self.allocation_step
        )
        kv = (
            (self.attention_layers + self.mtp_layers)
            * capacity
            * 2
            * self.kv_heads
            * self.head_dim
            * self.item_bytes
        )
        return kv + capacity * self.transcript_bytes_per_token

    @staticmethod
    def _validate_context(context_tokens):
        if type(context_tokens) is not int or context_tokens < 0:
            raise ValueError("context_tokens must be nonnegative integer")

    def prefill_transient_bytes(self, context_tokens, chunk_rows):
        """Transient bytes one prefill chunk of ``chunk_rows`` needs at ``context_tokens``."""
        self._validate_context(context_tokens)
        rows = max(0, int(chunk_rows))
        fixed = (
            self.prefill_chunk_transient_gib
            * min(rows, self.prefill_chunk_rows)
            / self.prefill_chunk_rows
            * (1 << 30)
        )
        kv_copy = (
            (self.attention_layers + self.mtp_layers)
            * int(context_tokens)
            * 2
            * self.kv_heads
            * self.head_dim
            * 2
        )
        return int(fixed + kv_copy)

    def project_resident(self, context_tokens):
        """Bytes for one retained cache without speculative scratch."""
        self._validate_context(context_tokens)
        return self.resident_fixed_bytes + self._variable_bytes(context_tokens)

    def project(self, context_tokens):
        """Bytes for one running lane, including its rollback scratch."""
        self._validate_context(context_tokens)
        return self.project_resident(context_tokens) + self.speculative_scratch_bytes

    def project_pool(self, context_tokens, *, resident_lanes, running_lanes):
        """Project a cache pool without multiplying scratch by cache slots.

        Every resident lane owns live recurrent/KV state.  Only lanes admitted
        to the current speculative forward own rollback scratch.
        """
        self._validate_context(context_tokens)
        for name, value in (
            ("resident_lanes", resident_lanes),
            ("running_lanes", running_lanes),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if running_lanes > resident_lanes:
            raise ValueError("running_lanes cannot exceed resident_lanes")
        return (
            resident_lanes * self.project_resident(context_tokens)
            + running_lanes * self.speculative_scratch_bytes
        )

    def as_dict(self):
        return {
            "schema": "qwen38-cache-geometry-v1",
            **asdict(self),
            "fixed_bytes": self.fixed_bytes,
            "resident_fixed_bytes": self.resident_fixed_bytes,
            "speculative_scratch_bytes": self.speculative_scratch_bytes,
            "scratch_scope": "running-lanes-only",
            "bound": "fp32-capacity-plus-per-running-lane-recurrent-rollback",
            "warm_copy": "observed-complete-cache-plus-projected-growth",
            "workspace": "dense-forward-transient-separate-from-cache",
        }
