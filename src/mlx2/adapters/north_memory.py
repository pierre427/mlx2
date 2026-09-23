"""Architecture-derived cache bound for North Mini Code ordinary decode.

The bound charges all K/V arrays at fp32.  Global layers grow with the
context; sliding-RoPE layers are capped at their 4K window and additionally
charge the four exact restore snapshots retained by the shared cache runtime.
Forward workspace and the common 20 GiB service/driver reserve remain separate.
"""

from dataclasses import asdict, dataclass
import math

from ..runtime.memory_policy import SelfMTPLaneAdmissionController


@dataclass(frozen=True)
class NorthCacheBudget:
    global_layers: int
    sliding_layers: int
    kv_heads: int
    head_dim: int
    sliding_window: int
    item_bytes: int = 4
    allocation_step: int = 256
    checkpoint_copies: int = 4
    transcript_bytes_per_token: int = 16
    # Measured on this model, 2026-09-19, M3 Pro: the k=2 verify transient is
    # 0.044-0.071 GiB/lane across 1K/4K/16K x 1/2/4 lanes, and 0.144 GiB/lane
    # at worst including the cold-allocator first forward.  The 3.1 that stood
    # here was the dense Qwen3.8-27B figure carried over as a placeholder, not
    # a North measurement; it exceeded the entire lane budget of a 36 GiB
    # host.  ``MOE_TRANSIENT_GIB_PER_LANE`` is 0.35 -- 2.4x the worst North
    # observation.  See provenance/lane-transient-moe.json.
    transient_gib_per_lane: float = (
        SelfMTPLaneAdmissionController.MOE_TRANSIENT_GIB_PER_LANE
    )

    @classmethod
    def from_config(cls, config, *, mtp):
        if mtp:
            raise ValueError("North Mini Code has no qualified native MTP route")
        layers = config.get("layer_types")
        count = int(config["num_hidden_layers"])
        if (
            not isinstance(layers, list)
            or len(layers) != count
            or set(layers) - {"full_attention", "sliding_attention"}
        ):
            raise ValueError("unknown North cache topology")
        result = cls(
            global_layers=layers.count("full_attention"),
            sliding_layers=layers.count("sliding_attention"),
            kv_heads=int(config["num_key_value_heads"]),
            head_dim=int(config["head_dim"]),
            sliding_window=int(config["sliding_window"]),
        )
        numeric = asdict(result)
        if any(
            (not math.isfinite(value)) or value < 0
            for value in numeric.values()
            if isinstance(value, (int, float))
        ) or not all(
            (
                result.global_layers,
                result.sliding_layers,
                result.kv_heads,
                result.head_dim,
                result.sliding_window,
                result.item_bytes,
                result.allocation_step,
            )
        ):
            raise ValueError("invalid North cache dimensions")
        return result

    def project(self, context_tokens):
        if type(context_tokens) is not int or context_tokens < 0:
            raise ValueError("context_tokens must be nonnegative integer")
        capacity = (
            math.ceil(context_tokens / self.allocation_step) * self.allocation_step
            + self.allocation_step
        )
        elements_per_token = 2 * self.kv_heads * self.head_dim
        global_bytes = (
            self.global_layers * capacity * elements_per_token * self.item_bytes
        )
        sliding_capacity = min(capacity, self.sliding_window)
        sliding_bytes = (
            self.sliding_layers
            * sliding_capacity
            * elements_per_token
            * self.item_bytes
            * (1 + self.checkpoint_copies)
        )
        ledger = (self.global_layers + self.sliding_layers) * 4096
        return (
            global_bytes
            + sliding_bytes
            + capacity * self.transcript_bytes_per_token
            + ledger
        )

    def as_dict(self):
        return {
            "schema": "north-mini-code-cache-geometry-v1",
            **asdict(self),
            "bound": "fp32-global-plus-window-and-restore-snapshots",
            "workspace": "3.1-GiB-per-lane conservative static assumption; live qualification pending",
        }
