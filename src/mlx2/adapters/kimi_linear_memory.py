"""Kimi's compressed NoPE MLA plus fp32 recurrent session cost.

Geometry only: measured resident weights and live free/advisory memory remain
the serving admission controller's authority. No hard-coded session count.
"""
import os
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class KimiLinearCacheBudget:
    global_layers: int
    recurrent_layers: int
    latent_dim: int
    shared_dim: int
    heads: int
    head_dim: int
    nope_dim: int
    value_dim: int
    hidden_size: int
    vocab_size: int
    conv_history: int
    checkpoint_copies: int
    mla: str = "auto"
    # The artifact gate checks BF16 projections/norms; MLA and conv retain
    # that activation dtype. The recurrent matrix is always charged at F32.
    item_bytes: int = 2
    allocation_step: int = 256
    transcript_bytes_per_token: int = 16
    transient_gib_per_lane: float = 1.76

    @classmethod
    def from_config(cls, config, *, mtp=False, mla="auto"):
        if mtp:
            raise ValueError("Kimi Linear has no staged MTP memory contract")
        from .kimi_linear import validate_config
        validate_config(config)
        linear = config["linear_attn_config"]
        copies = int(os.environ.get("MLX_LM_STATE_CHECKPOINT_MAX", "4"))
        if copies < 0:
            raise ValueError("negative recurrent checkpoint budget")
        return cls(len(linear["full_attn_layers"]), len(linear["kda_layers"]),
                   config["kv_lora_rank"], config["qk_rope_head_dim"],
                   linear["num_heads"], linear["head_dim"],
                   config["qk_nope_head_dim"], config["v_head_dim"],
                   config["hidden_size"], config["vocab_size"],
                   linear["short_conv_kernel_size"]-1, copies, mla)

    @staticmethod
    def _tokens(n):
        if type(n) is not int or n < 0:
            raise ValueError("cache dimensions must be nonnegative integers")
        return n

    @property
    def recurrent_live_bytes(self):
        return self.recurrent_layers * (self.heads * self.head_dim**2 * 4
               + 3 * self.conv_history * self.heads * self.head_dim * self.item_bytes)

    @property
    def kv_bytes_per_token(self):
        return self.global_layers * (self.latent_dim+self.shared_dim) * self.item_bytes

    def project(self, context_tokens):
        n = self._tokens(context_tokens)
        capacity = ((n+self.allocation_step-1)//self.allocation_step+1)*self.allocation_step
        return (capacity * (self.kv_bytes_per_token+self.transcript_bytes_per_token)
                + self.recurrent_live_bytes*(1+self.checkpoint_copies)
                + (self.global_layers+self.recurrent_layers)*4096)

    def prefill_transient_bytes(self, context_tokens, chunk_rows):
        n, rows = self._tokens(context_tokens), self._tokens(chunk_rows)
        # Cache growth may hold old and new buffers. Prompt chunks project one
        # checkpoint row through the vocabulary head, and unpadded chunks use
        # the causal offset mode: every MLA law builds causal visibility per
        # block it reads (latent_single's whole block is bounded by its score
        # block), never a [rows, context] array. Only a left-padded batch
        # cache hands MLA an explicit bool mask, and on the serving routes
        # that is decode/verify rows (<= 128): plan-exact prefill runs one
        # unpadded prompt per forward.
        copies = n*self.kv_bytes_per_token
        activations = rows*(16*self.heads*self.head_dim+8*self.hidden_size)*4 + self.vocab_size*4
        mask = n*min(rows, 128)
        if self.mla == "latent_single":
            # Its one-pass mask, inside the 256 MiB score bound: [rows,
            # context] <= 256 MiB / (4*heads), above the 128-row charge only
            # at short contexts (n < 16K at 32 heads). auto sends > 128 rows
            # to the latent law only past its 4 GiB expanded bound, which no
            # shape inside the one-pass score bound reaches.
            mask = max(mask, min(rows*n, (256 << 20)//(4*self.heads)))
        if self.mla == "auto":
            # Expanded prefix K/V is bounded by the module's 4 GiB switch;
            # beyond it the bounded latent score block applies.
            # Fused D256 pads keys and values to 256 channels each.
            expanded = n*self.heads*512*self.item_bytes
            scratch = max(min(2*expanded, 2*(4 << 30)), 3*min(rows*n*self.heads*4, 256 << 20))
        elif self.mla == "latent_single":
            # One score block, bounded by the module; larger shapes tile.
            scratch = 3*min(rows*n*self.heads*4, 256 << 20)
        elif self.mla in {"latent_tiled", "latent_fused"}:
            scratch = 3*min(rows, 64)*min(n, 2048)*self.heads*4
        else:
            scratch = (n*self.heads*(self.nope_dim+self.value_dim+self.shared_dim)
                       + 3*rows*n*self.heads)*4
        return copies+activations+mask+scratch

    def full_context_sessions(self, free_bytes, *, context_tokens=1048576,
                              chunk_rows=512, reserve_bytes=0):
        """Conservative full-context session ceiling AFTER model residency.

        Live serving admits by actual current contexts and measured free
        memory. This estimate additionally reserves a complete context for
        every session, plus one in-flight forward workspace per session.
        """
        free_bytes, reserve_bytes = self._tokens(free_bytes), self._tokens(reserve_bytes)
        per_session = self.project(context_tokens)+self.prefill_transient_bytes(context_tokens, chunk_rows)
        return max(0, free_bytes-reserve_bytes)//max(1, per_session)

    def as_dict(self):
        return dict(schema="kimi-linear-compressed-mla-kda-v1", **asdict(self),
                    kv_bytes_per_token=self.kv_bytes_per_token,
                    recurrent_live_bytes=self.recurrent_live_bytes,
                    bound="BF16 latent/shared cache and convolution, FP32 recurrent matrix, restore copies",
                    session_policy="live free memory after model residency and reserves; never a fixed count")
