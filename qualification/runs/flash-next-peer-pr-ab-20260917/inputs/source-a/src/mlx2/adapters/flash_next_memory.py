"""Configuration-derived upper bound for the Flash-Next hybrid cache ABI.

QSA retains K/V, raw index keys and one pooled index key per block; GDN retains
fixed convolution/recurrent state. This is not the full-attention envelope.
All arrays are charged at fp32, including normally bf16 K/V, and recurrent
state is charged twice for rollback. Kernel/prefill temporaries and the 20 GiB
host/driver reserve remain separate in the common admission controller.
"""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class FlashNextCacheBudget:
    qsa_layers: int
    recurrent_layers: int
    kv_heads: int
    head_dim: int
    index_dim: int
    pool_ratio: int
    recurrent_heads: int
    recurrent_key_dim: int
    recurrent_value_dim: int
    recurrent_key_heads: int
    conv_kernel: int
    ple_layers: int
    ple_dim: int
    ple_kernel: int
    item_bytes: int = 4
    allocation_step: int = 256

    @classmethod
    def from_config(cls, config, *, mtp):
        layers = config["layer_types"]
        if len(layers) != int(config["num_hidden_layers"]) or set(layers) - {"full_attention", "linear_attention"}:
            raise ValueError("unknown Flash-Next cache topology")
        result = cls(
            qsa_layers=layers.count("full_attention") + (int(config["mtp_num_hidden_layers"]) if mtp else 0),
            recurrent_layers=layers.count("linear_attention"),
            kv_heads=int(config["num_key_value_heads"]), head_dim=int(config["head_dim"]),
            index_dim=int(config["indexer_head_dim"]), pool_ratio=int(config["indexer_compress_ratio"]),
            recurrent_heads=int(config["linear_num_value_heads"]),
            recurrent_key_dim=int(config["linear_key_head_dim"]),
            recurrent_value_dim=int(config["linear_value_head_dim"]),
            recurrent_key_heads=int(config["linear_num_key_heads"]),
            conv_kernel=int(config["linear_conv_kernel_dim"]),
            ple_layers=len(config.get("ple_layer_ids", [])),
            ple_dim=int(config["ple_embed_dim"]), ple_kernel=int(config["ple_conv_kernel_size"]),
        )
        if any(value < 0 for value in asdict(result).values()) or not result.pool_ratio:
            raise ValueError("invalid cache dimensions")
        return result

    @property
    def fixed_bytes(self):
        recurrent = self.recurrent_heads * self.recurrent_key_dim * self.recurrent_value_dim
        conv_dim = 2 * self.recurrent_key_heads * self.recurrent_key_dim + self.recurrent_heads * self.recurrent_value_dim
        per_recurrent = recurrent + max(0, self.conv_kernel - 1) * conv_dim
        # A live recurrent state and rollback snapshot, plus PLE convolution
        # and a generous scalar/token bookkeeping allowance per layer.
        return self.item_bytes * (2 * self.recurrent_layers * per_recurrent
            + 2 * self.ple_layers * self.ple_dim * self.ple_kernel) + (self.qsa_layers + self.recurrent_layers) * 4096

    def project(self, context_tokens):
        if type(context_tokens) is not int or context_tokens < 0:
            raise ValueError("context_tokens must be nonnegative integer")
        # Multi-chunk append may retain one extra allocation quantum.
        capacity = math.ceil(context_tokens / self.allocation_step) * self.allocation_step + self.allocation_step
        qsa = self.qsa_layers * self.item_bytes * (
            capacity * (2 * self.kv_heads * self.head_dim + self.index_dim)
            + math.ceil(capacity / self.pool_ratio) * self.index_dim)
        # Request transcript, MTP token history and seed bookkeeping.
        return self.fixed_bytes + qsa + capacity * 16

    def as_dict(self):
        return {"schema": "flash-next-cache-geometry-v1", **asdict(self),
                "fixed_bytes": self.fixed_bytes, "bound": "fp32-capacity-plus-recurrent-rollback"}
