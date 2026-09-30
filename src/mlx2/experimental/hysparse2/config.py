"""Allocation-free configuration and capacity accounting."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    model_type: str = "mlx2_hysparse2_research"
    vocab_size: int = 32768
    hidden_size: int = 1024
    num_heads: int = 16
    head_dim: int = 256
    self_layers: int = 25
    self_full_layer: int = 12
    cross_blocks: int = 4
    sparse_per_block: int = 5
    num_experts: int = 32
    experts_per_token: int = 2
    expert_dim: int = 256
    shared_experts: int = 1
    residual_streams: int = 4
    local_window: int = 128
    global_tokens: int = 1024
    rope_dims: int = 64
    rope_base: float = 10000.0
    max_context: int = 2097152
    query_tile: int = 32
    key_tile: int = 1024
    prefill_chunk: int = 256
    norm_eps: float = 1e-6
    mtp: bool = True

    def __post_init__(self):
        for name in (
            "vocab_size",
            "hidden_size",
            "num_heads",
            "head_dim",
            "self_layers",
            "cross_blocks",
            "num_experts",
            "experts_per_token",
            "expert_dim",
            "residual_streams",
            "local_window",
            "global_tokens",
            "max_context",
            "query_tile",
            "key_tile",
            "prefill_chunk",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "self_full_layer",
            "sparse_per_block",
            "shared_experts",
            "rope_dims",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.self_full_layer >= self.self_layers:
            raise ValueError("self_full_layer must lie in the self decoder")
        if self.experts_per_token > self.num_experts:
            raise ValueError("experts_per_token exceeds expert count")
        if self.rope_dims > self.head_dim or self.rope_dims % 2:
            raise ValueError("RoPE dimensions must be even and fit head_dim")
        if (
            not all(math.isfinite(v) and v > 0 for v in (self.rope_base, self.norm_eps))
            or type(self.mtp) is not bool
        ):
            raise ValueError("invalid normalization, RoPE or MTP setting")

    @property
    def layers(self):
        return self.self_layers + self.cross_blocks * (1 + self.sparse_per_block)

    def as_dict(self):
        return asdict(self)

    def capacity(self, *, batch=1, context=None, bytes_per_element=2):
        """Conservative tensor sizes; no claim of measured peak memory."""
        context = self.max_context if context is None else context
        if (
            type(batch) is not int
            or batch < 1
            or type(context) is not int
            or not 1 <= context <= self.max_context
        ):
            raise ValueError("invalid batch or context")
        if type(bytes_per_element) is not int or bytes_per_element < 1:
            raise ValueError("bytes_per_element must be positive")
        d, h, e, r = (
            self.hidden_size,
            self.num_heads * self.head_dim,
            self.expert_dim,
            self.residual_streams,
        )
        # Each layer has two identity-residual hyperconnections and two RMS norms.
        hyper = 2 * ((r * d) * (2 * r) + 2 * r + 2)
        moe = (
            self.num_experts + self.shared_experts
        ) * 3 * d * e + d * self.num_experts
        base = hyper + moe + 2 * d

        def attention(kind):
            count = 2 * d * h
            if kind != "sparse":
                count += 2 * d * self.head_dim
            if kind != "full":
                count += d * self.num_heads + self.num_heads
            if kind == "cross":
                # Cross FA has an independent normalization for its bridged source;
                # it does not have a gate/sink (handled as full below).
                return 2 * d * h + 2 * d * self.head_dim + d
            return count

        parameters = self.vocab_size * d + d + self.layers * base
        parameters += attention("full") + (self.self_layers - 1) * attention("swa")
        parameters += self.cross_blocks * (
            attention("cross") + self.sparse_per_block * attention("sparse")
        )
        if self.mtp:
            parameters += 2 * d * d + 3 * d + attention("swa") + 3 * d * (4 * d)
        active = (
            parameters
            - self.layers * (self.num_experts - self.experts_per_token) * 3 * d * e
        )
        kv = (
            batch
            * (1 + self.cross_blocks)
            * context
            * 2
            * self.head_dim
            * bytes_per_element
        )
        rolling = (
            batch
            * (self.self_layers - 1)
            * min(context, self.local_window)
            * 2
            * self.head_dim
            * bytes_per_element
        )
        return {
            "parameters": parameters,
            "active_parameters_including_embeddings_and_mtp": active,
            "weight_bytes": parameters * bytes_per_element,
            "full_kv_bytes": kv,
            "rolling_kv_bytes": rolling,
            "layers": self.layers,
            "context": context,
            "full_attention_caches": 1 + self.cross_blocks,
            "score_tile_elements": batch
            * self.num_heads
            * self.query_tile
            * self.key_tile,
            "selected_kv_elements_per_query_tile": batch
            * self.query_tile
            * (self.global_tokens + self.local_window)
            * 2
            * self.head_dim,
            "training_note": "Optimizer, gradients, saved activations and allocator overhead are additional; max_context is not validated retrieval quality.",
        }

    @classmethod
    def smoke(cls):
        return cls(
            vocab_size=48,
            hidden_size=16,
            num_heads=2,
            head_dim=8,
            self_layers=3,
            self_full_layer=1,
            cross_blocks=2,
            sparse_per_block=1,
            num_experts=4,
            experts_per_token=2,
            expert_dim=12,
            residual_streams=2,
            local_window=3,
            global_tokens=2,
            rope_dims=4,
            query_tile=3,
            key_tile=4,
            prefill_chunk=4,
        )
