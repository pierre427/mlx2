"""Physical mixed FP8 cache budget with explicit temporary FP32 unpack cost."""

from dataclasses import dataclass

from .deepseek_v4_memory import DeepSeekV4CacheBudget, _integer


@dataclass(frozen=True)
class DeepSeekV4PackedCacheBudget(DeepSeekV4CacheBudget):
    def __post_init__(self):
        super().__post_init__()
        if self.head_dim != 512:
            raise ValueError("mixed FP8 budget requires width512")

    @property
    def packed_row_bytes(self):
        return 448 + 7 * 4 + 64 * 4

    @property
    def local_live_bytes(self):
        return self.layers * self.sliding_window * self.packed_row_bytes

    def pooled_bytes(self, context_tokens):
        _integer("context_tokens", context_tokens)
        return sum(
            self._pool_capacity(context_tokens, r)
            * (self.packed_row_bytes + (self.index_head_dim * 4 if r == 4 else 0))
            for r in self.compress_ratios
            if r
        )

    def unpack_workspace_bytes(self, context_tokens):
        """Conservative graph-live FP32 reconstructed local/pool rows, all layers.

        Until attention directly reads packed storage, this is a running-lane
        charge even for one-token decode; it cannot hide in a fixed allowance.
        """
        _integer("context_tokens", context_tokens)
        return (
            (
                self.layers * self.sliding_window
                + sum(
                    self._pool_capacity(context_tokens, r)
                    for r in self.compress_ratios
                    if r
                )
            )
            * self.head_dim
            * 4
        )

    def project(self, context_tokens):
        return super().project(context_tokens) + self.unpack_workspace_bytes(
            context_tokens
        )

    def project_pool(self, context_tokens, *, resident_lanes, running_lanes):
        return super().project_pool(
            context_tokens, resident_lanes=resident_lanes, running_lanes=running_lanes
        ) + running_lanes * self.unpack_workspace_bytes(context_tokens)

    def as_dict(self):
        result = super().as_dict()
        result.update(
            schema="deepseek-v4-fp8-mixed-cache-geometry-v1",
            physical_storage="uint8-E4M3FN448+FP32-scales7+FP32-RoPE64; indexer/carry FP32",
            packed_attention_row_bytes=self.packed_row_bytes,
            activation_simulation="same V4 QAT; completed attention/local physically packed",
            workspace="running lanes additionally charged all-layer FP32 unpack; prefill bounds unchanged",
            bound="packed-local+FP32-pending/previous/indexer+packed-pools+snapshot+temporary-FP32-unpack",
        )
        return result


@dataclass(frozen=True)
class DeepSeekV4DirectPackedCacheBudget(DeepSeekV4PackedCacheBudget):
    """Only the selected direct-pool consumer removes historical FP32 pools.

    Local windows still reconstruct FP32 bands, conservatively charged across
    all layers. Prefill query/output/indexer bounds and the1.76GiB forward
    allowance remain unchanged; raw pool/snapshot storage is unchanged.
    """

    def unpack_workspace_bytes(self, context_tokens):
        _integer("context_tokens", context_tokens)
        return self.layers * self.sliding_window * self.head_dim * 4

    def as_dict(self):
        result = super().as_dict()
        result.update(
            schema="deepseek-v4-direct-packed-visible-cache-geometry-v1",
            pool_fp32_materialization=False,
            workspace="all-layer FP32 local windows retained; compressed pools decoded only inside selected visible-attention kernel; prefill/forward reserves unchanged",
            bound="packed-local+FP32-carry/indexer+packed-pools+snapshot+FP32-local-band",
        )
        return result
