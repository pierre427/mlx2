"""Architecture-derived cache bound for Maxwell-V1-Flash.

Every trunk layer caches GQA K and V (4 KV heads x 128) plus one indexer-key
plane (one head x 128 fp32 keys plus a 1-wide value of a plain ``KVCache``); the
native MTP layer adds one dense GQA layer when that route is selected.  mHC
streams live only inside a forward pass, so they add workspace, not cache.
Arrays are charged at fp32 like the other adapters.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MaxwellCacheBudget:
    trunk_layers: int
    mtp_layers: int
    kv_heads: int
    head_dim: int
    indexer_dim: int
    item_bytes: int = 4
    allocation_step: int = 256
    transcript_bytes_per_token: int = 16
    # UNMEASURED placeholder.  The small-active MoE figure (0.35 GiB) plus the
    # sparse route's long-context scratch: the fp32 index-score chunk is
    # bounded to ~256 MB (maxwell_moe._route) and union gathers are small.
    transient_gib_per_lane: float = 0.6

    @classmethod
    def from_config(cls, config: dict, *, mtp: bool):
        mtp_layers = int(config.get("mtp_num_layers", 0)) if mtp else 0
        if mtp and mtp_layers != 1:
            raise ValueError("Maxwell native MTP needs its single MTP layer")
        return cls(
            trunk_layers=int(config["num_hidden_layers"]),
            mtp_layers=mtp_layers,
            kv_heads=int(config["num_key_value_heads"]),
            head_dim=int(config["attention_head_size"]),
            indexer_dim=int(config["block_sparse_gqa_indexer_head_dim"]),
        )

    def project(self, context_tokens: int) -> int:
        if type(context_tokens) is not int or context_tokens < 0:
            raise ValueError("context_tokens must be nonnegative integer")
        capacity = (math.ceil(context_tokens / self.allocation_step) * self.allocation_step
                    + self.allocation_step)
        kv = 2 * self.kv_heads * self.head_dim
        trunk = self.trunk_layers * capacity * (kv + self.indexer_dim + 1)
        mtp = self.mtp_layers * capacity * kv
        planes = 2 * self.trunk_layers + self.mtp_layers
        return ((trunk + mtp) * self.item_bytes + capacity * self.transcript_bytes_per_token
                + planes * 4096)

    def as_dict(self) -> dict:
        return {
            "schema": "maxwell-gqa-indexer-cache-geometry-v1",
            **asdict(self),
            "bound": "fp32-gqa-kv-plus-indexer-key-planes",
            "workspace": "0.6-GiB-per-lane unmeasured placeholder",
        }
