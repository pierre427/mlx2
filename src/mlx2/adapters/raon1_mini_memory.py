"""Raon-1-mini session cost: NoPE latent MLA plus fp32 RDA state.

Same geometry law as Kimi Linear's compressed MLA/KDA budget (one 512+64
latent row per MLA layer per token, an fp32 [H, D, D] recurrent matrix and a
3-row conv history per RDA layer).  Live free memory stays the admission
authority; no fixed session count.
"""
from dataclasses import asdict, dataclass
import os

from .kimi_linear_memory import KimiLinearCacheBudget


@dataclass(frozen=True)
class RaonCacheBudget(KimiLinearCacheBudget):
    # The MLA expanded-K/V switch (policy mla_expanded_gib); Kimi's budget
    # charges its fixed 4 GiB bound.
    expanded_bytes: int = 4 << 30

    @classmethod
    def from_config(cls, config, *, mtp=False, mla="auto", expanded_bytes=4 << 30):
        if mtp:
            raise ValueError("Raon-1-mini has no MTP memory contract")
        from .raon1_mini import validate_config

        validate_config(config)
        rda = config["rda_config"]
        copies = int(os.environ.get("MLX_LM_STATE_CHECKPOINT_MAX", "4"))
        if copies < 0:
            raise ValueError("negative recurrent checkpoint budget")
        return cls(len(rda["mla_layers"]), len(rda["rda_layers"]), config["kv_lora_rank"],
                   config["qk_rope_head_dim"], rda["num_heads"], rda["head_dim"],
                   config["qk_nope_head_dim"], config["v_head_dim"], config["hidden_size"],
                   config["vocab_size"], rda["conv_kernel_size"] - 1, copies, mla,
                   expanded_bytes=int(expanded_bytes))

    def prefill_transient_bytes(self, context_tokens, chunk_rows):
        base = super().prefill_transient_bytes(context_tokens, chunk_rows)
        if self.mla != "auto" or self.expanded_bytes <= 4 << 30:
            return base
        # A larger bound keeps fused D256 (padded K/V, 512 channels a head)
        # in force at wider prefixes than Kimi's fixed 4 GiB charge.
        expanded = self._tokens(context_tokens) * self.heads * 512 * self.item_bytes
        return base + max(0, min(2 * expanded, 2 * self.expanded_bytes) - min(2 * expanded, 2 * (4 << 30)))

    def as_dict(self):
        return dict(schema="raon1-mini-latent-mla-rda-v1", **asdict(self),
                    kv_bytes_per_token=self.kv_bytes_per_token,
                    recurrent_live_bytes=self.recurrent_live_bytes,
                    bound="BF16 latent/shared cache and convolution, FP32 recurrent matrix, restore copies",
                    session_policy="live free memory after model residency and reserves; never a fixed count")
