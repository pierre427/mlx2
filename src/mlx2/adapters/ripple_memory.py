"""Ripple cache geometry; forward workspace remains provisional, unmeasured."""

from dataclasses import asdict, dataclass

from .laguna_memory import LagunaCacheBudget


@dataclass(frozen=True)
class RippleCacheBudget(LagunaCacheBudget):
    # BF16 projection and norm dtypes are checked by the artifact inspector.
    item_bytes: int = 2
    intermediate_size: int = 40960
    query_width: int = 10240

    @classmethod
    def from_config(cls, config, *, mtp):
        from .ripple import validate_config

        validate_config(config)
        if mtp:
            raise ValueError("Ripple has no implemented native MTP cache topology")
        base = LagunaCacheBudget.from_config(config, mtp=False)
        values = asdict(base)
        values["item_bytes"] = 2
        return cls(**values, intermediate_size=config["intermediate_size"],
                   query_width=config["num_attention_heads"] * config["head_dim"])

    def prefill_transient_bytes(self, context_tokens, chunk_rows):
        # W4A16 QMM uses FP32 input/output; include MLP intermediates and
        # explicit BF16-rounded RoPE temporaries beyond the shared KV bound.
        return (super().prefill_transient_bytes(context_tokens, chunk_rows)
                + chunk_rows * (6 * self.intermediate_size + 8 * self.query_width) * 4)

    def as_dict(self):
        return {**asdict(self), "schema": "ripple-bf16-full-sliding-cache-v1",
                "bound": "bf16-KV-with-exact-sliding-restore-snapshots",
                "workspace": "provisional-unmeasured; prefill copies and FP32 MLP scratch charged separately"}
