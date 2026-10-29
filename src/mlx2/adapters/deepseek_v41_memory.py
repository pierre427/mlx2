"""Physical shared-source V4.1 cache and bounded Engram/prefill charges."""

from dataclasses import dataclass

from .deepseek_v4_memory import DeepSeekV4CacheBudget, _integer


@dataclass(frozen=True)
class DeepSeekV41CacheBudget(DeepSeekV4CacheBudget):
    hidden_size: int = 5120
    q_lora_rank: int = 1280
    routed_experts: int = 384
    expert_hidden: int = 2304
    index_heads: int = 32
    kv_source_layer_ids: tuple[int, ...] = (2, 8, 14, 20)
    engram_layers: int = 2
    engram_cache_bytes: int = 16 << 20

    def __post_init__(self):
        if not self.compress_ratios or any(
            type(r) is not int or r not in (0, 1, 2) for r in self.compress_ratios
        ):
            raise ValueError("V4.1 cache budget requires ratio0/1/2 geometry")
        if self.item_bytes != 4 or any(
            not 0 <= i < self.layers or not self.compress_ratios[i]
            for i in self.kv_source_layer_ids
        ):
            raise ValueError("invalid V4.1 source cache budget")

    @classmethod
    def from_config(cls, c):
        mapping = {
            "attention_heads": "num_attention_heads",
            "index_heads": "index_n_heads",
            "experts_per_token": "num_experts_per_tok",
            "routed_experts": "n_routed_experts",
            "expert_hidden": "moe_intermediate_size",
            "shared_experts": "n_shared_experts",
        }
        names = (
            "compress_ratios",
            "hidden_size",
            "head_dim",
            "index_head_dim",
            "sliding_window",
            "hc_mult",
            "q_lora_rank",
            "o_groups",
            "o_lora_rank",
            "kv_source_layer_ids",
        )
        return cls(
            **{k: getattr(c, k) for k in names},
            **{k: getattr(c, v) for k, v in mapping.items()},
            engram_layers=len(c.engram_layer_ids),
            engram_cache_bytes=(8 << 20) * len(c.engram_layer_ids),
        )

    @property
    def compression_fixed_bytes(self):
        return (
            sum(
                2 * (self.compress_ratios[i] - 1) * self.head_dim * self.item_bytes
                for i in self.kv_source_layer_ids
            )
            + 12
        )

    @property
    def resident_fixed_bytes(self):
        return (
            self.local_live_bytes
            + self.compression_fixed_bytes
            + self.layers * self.ledger_bytes_per_layer
            + self.engram_cache_bytes
        )

    def pooled_bytes(self, context_tokens):
        _integer("context_tokens", context_tokens)
        return sum(
            self._pool_capacity(context_tokens, self.compress_ratios[i])
            * (self.head_dim + self.index_head_dim)
            * self.item_bytes
            for i in self.kv_source_layer_ids
        )

    def prefill_transient_bytes(self, context_tokens, chunk_rows):
        base = super().prefill_transient_bytes(context_tokens, chunk_rows)
        # Two bounded selected-row float buffers plus packed reads and scatter;
        # candidate mask, selected-index masks and exact gather scratch.
        engram = chunk_rows * self.engram_layers * 24 * 256 * 4 * 2
        candidates = chunk_rows * context_tokens * 3
        return base + engram + candidates + (32 << 20)

    def as_dict(self):
        return {
            **super().as_dict(),
            "schema": "deepseek-v41-shared-source-cache-v1",
            "activation_simulation": "BF16/whole-vector-FP8/local; E2M1/E4M3-pools; E2M1/E8M0-index",
            "bound": "owner-only KV+index; local per layer; partial groups; exact ngram tail; host row LRU; running full snapshot",
            "engram_table_resident_bytes": 0,
            "source_layers": list(self.kv_source_layer_ids),
        }
