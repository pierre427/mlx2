"""Architecture-derived cache bound for Xing4.0 MLA attention.

Every trunk layer caches the normed MLA latent (``kv_lora_rank``) as its keys
and the roped key (``qk_rope_head_dim``) as its values, one head wide; the
embedded MTP layer adds one more such layer when the native MTP route is
selected.  mHC streams live only inside a forward pass, so they add
workspace, not cache.  Arrays are charged at fp32 like the other adapters.
The shared controller keeps its host/driver reserve separately.
"""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class XingCacheBudget:
    attention_layers: int
    mtp_layers: int
    latent_dim: int
    rope_dim: int
    item_bytes: int = 4
    allocation_step: int = 256
    transcript_bytes_per_token: int = 16
    # UNMEASURED.  3.1 is the dense Qwen3.8-27B workspace, carried here as a
    # placeholder.  Two small-active MoE models measured on 2026-09-19
    # (Qwen3.6-35B-A3B, North-Mini-Code-1.0) came in at 0.015-0.071 GiB/lane
    # at k=2 -- 40-200x below this -- and were corrected to
    # ``MOE_TRANSIENT_GIB_PER_LANE``.  Xing4.0 is left at 3.1 deliberately:
    # its mHC multi-stream tensors live only inside the forward and are
    # exactly the workspace term this number stands for, so the MoE figure
    # cannot be transferred to it without measuring it, and the Xing4.0
    # artifact was not on the measurement host.  Until it is measured this
    # value will consume the entire lane budget of any host under ~64 GiB and
    # pin the self-MTP route to its depth floor there.
    transient_gib_per_lane: float = 3.1

    @classmethod
    def from_config(cls, config, *, mtp):
        mtp_layers = int(config.get("num_nextn_predict_layers", 0)) if mtp else 0
        if mtp and mtp_layers != 1:
            raise ValueError("Xing4.0 native MTP needs its single embedded MTP layer")
        result = cls(
            attention_layers=int(config["num_hidden_layers"]),
            mtp_layers=mtp_layers,
            latent_dim=int(config["kv_lora_rank"]),
            rope_dim=int(config["qk_rope_head_dim"]),
        )
        numeric = asdict(result)
        if any(
            (not math.isfinite(value)) or value < 0
            for value in numeric.values()
            if isinstance(value, (int, float))
        ) or not all(
            (
                result.attention_layers,
                result.latent_dim,
                result.rope_dim,
                result.item_bytes,
                result.allocation_step,
            )
        ):
            raise ValueError("invalid Xing4.0 cache dimensions")
        return result

    def project(self, context_tokens):
        if type(context_tokens) is not int or context_tokens < 0:
            raise ValueError("context_tokens must be nonnegative integer")
        capacity = (
            math.ceil(context_tokens / self.allocation_step) * self.allocation_step
            + self.allocation_step
        )
        per_layer = capacity * (self.latent_dim + self.rope_dim) * self.item_bytes
        layers = self.attention_layers + self.mtp_layers
        return (
            layers * per_layer
            + capacity * self.transcript_bytes_per_token
            + layers * 4096
        )

    def as_dict(self):
        return {
            "schema": "xing4-0-mla-cache-geometry-v1",
            **asdict(self),
            "bound": "fp32-mla-latent-plus-rope-key",
            "workspace": "3.1-GiB-per-lane conservative static assumption; live qualification pending",
        }
