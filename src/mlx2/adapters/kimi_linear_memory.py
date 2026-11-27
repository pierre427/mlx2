"""Kimi's compressed NoPE MLA plus fp32 recurrent session cost.

Geometry only: measured resident weights and live free/advisory memory remain
the serving admission controller's authority. No hard-coded session count.
"""
import os
from dataclasses import asdict, dataclass

from .kimi_linear import DEFAULT_CONTEXT


@dataclass(frozen=True)
class DFlashCacheGeometry:
    layers: int
    kv_heads: int
    head_dim: int
    query_heads: int
    block_rows: int
    capture_layers: int
    item_bytes: int

    @classmethod
    def from_drafter(cls, draft):
        config = draft.config
        return cls(config.num_hidden_layers, config.num_key_value_heads, config.head_dim,
                   config.num_attention_heads, config.block_size, len(config.target_layer_ids),
                   draft.fc.weight.dtype.size)

    @property
    def layer_kv_bytes_per_token(self):
        return 2 * self.kv_heads * self.head_dim * self.item_bytes

    @property
    def kv_bytes_per_token(self):
        return self.layers * self.layer_kv_bytes_per_token


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
    draft: DFlashCacheGeometry | None = None
    tree_branch_copies: int = 0

    @classmethod
    def from_config(cls, config, *, mtp=False, mla="auto", draft=None, tree_branch_copies=0):
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
                   linear["short_conv_kernel_size"]-1, copies, mla,
                   draft=draft, tree_branch_copies=tree_branch_copies)

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
        draft_kv = 0 if self.draft is None else self.draft.kv_bytes_per_token
        return (capacity * (self.kv_bytes_per_token+draft_kv+self.transcript_bytes_per_token)
                + self.recurrent_live_bytes*(1+self.checkpoint_copies)
                + (self.global_layers+self.recurrent_layers)*4096)

    def decode_transient_bytes(self, context_tokens):
        """Context-sized copies during speculative verify and draft attention."""
        n = self._tokens(context_tokens)
        draft = self.draft
        target = self.kv_bytes_per_token // self.global_layers * self.tree_branch_copies
        if draft is None:
            return n * target
        # Reserve a draft cache copy for COW, plus one layer's context+block
        # concatenate and its FP32 attention score workspace.
        scores = draft.query_heads * draft.block_rows * 4
        return (n * (target + draft.kv_bytes_per_token)
                + (n + draft.block_rows) * (draft.layer_kv_bytes_per_token + scores))

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
        if self.draft is not None:
            # Layer taps and their concatenation coexist until the draft
            # projection consumes them. Resident draft KV COW is reserved
            # by decode_transient_bytes during prefill admission as well.
            activations += 2 * rows * self.hidden_size * self.draft.capture_layers * self.item_bytes
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

    def full_context_sessions(self, free_bytes, *, context_tokens=DEFAULT_CONTEXT,
                              chunk_rows=512, reserve_bytes=0):
        """Conservative full-context session ceiling AFTER model residency.

        Live serving admits by actual current contexts and measured free
        memory. This estimate additionally reserves a complete context for
        every session, plus one in-flight forward workspace per session.
        """
        free_bytes, reserve_bytes = self._tokens(free_bytes), self._tokens(reserve_bytes)
        per_session = (self.project(context_tokens) + self.prefill_transient_bytes(context_tokens, chunk_rows)
                       + self.decode_transient_bytes(context_tokens))
        return max(0, free_bytes-reserve_bytes)//max(1, per_session)

    def as_dict(self):
        fields = asdict(self)
        if self.draft is None:
            fields.pop("draft")
        else:
            fields["draft_kv_bytes_per_token"] = self.draft.kv_bytes_per_token
        if not self.tree_branch_copies:
            fields.pop("tree_branch_copies")
        return dict(schema="kimi-linear-compressed-mla-kda-v1", **fields,
                    kv_bytes_per_token=self.kv_bytes_per_token,
                    recurrent_live_bytes=self.recurrent_live_bytes,
                    bound="BF16 latent/shared cache and convolution, FP32 recurrent matrix, restore copies",
                    session_policy="live free memory after model residency and reserves; never a fixed count")
