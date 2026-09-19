"""Configuration-derived cache bound for dense Qwen3.8 hybrid models.

Full-attention target and embedded-MTP K/V capacity is charged at fp32.
GDN convolution and recurrent state is fixed-size and charged twice so a live
state and its speculative rollback snapshot fit together.  The shared
controller separately retains its 20 GiB host/driver reserve and charges the
dense-family forward workspace configured here.
"""

from dataclasses import asdict, dataclass
import math


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
    def fixed_bytes(self):
        recurrent = (
            self.recurrent_heads
            * self.recurrent_value_dim
            * self.recurrent_key_dim
        )
        conv_dim = (
            2 * self.recurrent_key_heads * self.recurrent_key_dim
            + self.recurrent_heads * self.recurrent_value_dim
        )
        per_layer = recurrent + max(0, self.conv_kernel - 1) * conv_dim
        # Live state plus rollback state, and a conservative per-layer ledger.
        return (
            2 * self.recurrent_layers * per_layer * self.item_bytes
            + (self.attention_layers + self.recurrent_layers + self.mtp_layers)
            * 4096
        )

    def project(self, context_tokens):
        if type(context_tokens) is not int or context_tokens < 0:
            raise ValueError("context_tokens must be nonnegative integer")
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
        return self.fixed_bytes + kv + capacity * self.transcript_bytes_per_token

    def as_dict(self):
        return {
            "schema": "qwen38-cache-geometry-v1",
            **asdict(self),
            "fixed_bytes": self.fixed_bytes,
            "bound": "fp32-capacity-plus-recurrent-rollback",
            "warm_copy": "observed-complete-cache-plus-projected-growth",
            "workspace": "dense-forward-transient-separate-from-cache",
        }
