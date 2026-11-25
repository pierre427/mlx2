"""Ordinary Ling cache geometry: compressed MLA plus bounded KDA state.

Derived from bailing_hybrid.Model.make_cache and its two attention modules.
MLA stores one latent and one RoPE vector, not expanded per-head K/V.
KDA keeps three short-convolution histories and a per-head recurrent matrix.
"""
from dataclasses import asdict, dataclass
import os


@dataclass(frozen=True)
class LingCacheBudget:
    global_layers: int
    recurrent_layers: int
    kv_lora_rank: int
    rope_dim: int
    heads: int
    head_dim: int
    nope_dim: int
    value_dim: int
    conv_history: int
    hidden_size: int
    vocab_size: int
    checkpoint_copies: int
    item_bytes: int = 4
    allocation_step: int = 256
    transcript_bytes_per_token: int = 16
    transient_gib_per_lane: float = 1.76
    # Prefill transient law. The defaults keep the original bound (three fp32
    # score buffers, fp32 copies). Ling-3.2-air passes its measured law
    # (ling32_air.PREFILL_LAW): bf16 activations, two score buffers (pe_scores
    # and its masked copy) and a score budget that caps prefill rows by
    # context. 0 means "use item_bytes" / "no row recommendation".
    prefill_activation_bytes: int = 0
    score_buffers: int = 3
    prefill_score_budget_bytes: int = 0
    min_prefill_rows: int = 64

    @classmethod
    def from_config(cls, config, *, mtp):
        nextn = config.get('num_nextn_predict_layers') or 0
        if mtp and (type(nextn) is not int or nextn < 1):
            raise ValueError('Ling ordinary cache budget does not cover speculative state')
        names = ('num_hidden_layers', 'layer_group_size', 'kv_lora_rank',
                 'qk_rope_head_dim', 'num_attention_heads', 'head_dim',
                 'qk_nope_head_dim', 'v_head_dim', 'short_conv_kernel_size',
                 'hidden_size', 'vocab_size')
        if any(type(config.get(k)) is not int or config[k] <= 0 for k in names):
            raise ValueError('invalid Ling cache geometry')
        count, group = config['num_hidden_layers'], config['layer_group_size']
        # Match DecoderLayer exactly, including a partial final MLA group.
        global_layers = sum((i + 1) % group == 0 or i >= (count // group) * group
                            for i in range(count))
        # Native MTP: each NEXTN layer is one more compressed-MLA cache.
        global_layers += nextn if mtp else 0
        try:
            checkpoints = max(0, int(os.environ.get('MLX_LM_STATE_CHECKPOINT_MAX', '4')))
        except ValueError:
            checkpoints = 4
        return cls(global_layers, count - global_layers, config['kv_lora_rank'],
                   config['qk_rope_head_dim'], config['num_attention_heads'],
                   config['head_dim'], config['qk_nope_head_dim'],
                   config['v_head_dim'], config['short_conv_kernel_size'] - 1,
                   config['hidden_size'], config['vocab_size'], checkpoints)

    @staticmethod
    def _tokens(value):
        if type(value) is not int or value < 0:
            raise ValueError('cache dimensions must be nonnegative integers')
        return value

    @property
    def recurrent_live_bytes(self):
        matrix = self.heads * self.head_dim * self.head_dim
        conv = 3 * self.conv_history * self.heads * self.head_dim
        return self.recurrent_layers * (matrix + conv) * self.item_bytes

    def project(self, context_tokens):
        n = self._tokens(context_tokens)
        capacity = ((n + self.allocation_step - 1) // self.allocation_step + 1) * self.allocation_step
        return (self.global_layers * capacity * (self.kv_lora_rank + self.rope_dim) * self.item_bytes
                + self.recurrent_live_bytes * (1 + self.checkpoint_copies)
                + capacity * self.transcript_bytes_per_token
                + (self.global_layers + self.recurrent_layers) * 4096)

    def prefill_transient_bytes(self, context_tokens, chunk_rows):
        context, rows = self._tokens(context_tokens), self._tokens(chunk_rows)
        # Global caches may retain old buffers during concatenation. One MLA
        # layer expands latent K/V for prefill and materializes its RoPE score
        # mask (pe_scores [heads, rows, context], then its mx.where copy).
        # Layers run in sequence, while their persistent compressed caches are
        # in project(). Activations stay at item_bytes (fp32 logits upcast).
        act = self.prefill_activation_bytes or self.item_bytes
        cache_copy = self.global_layers * context * (self.kv_lora_rank + self.rope_dim)
        expanded = context * self.heads * (self.nope_dim + self.value_dim)
        scores = self.score_buffers * rows * context * self.heads
        activations = rows * (16 * self.heads * self.head_dim + 8 * self.hidden_size + self.vocab_size)
        return (cache_copy + expanded + scores) * act + activations * self.item_bytes

    def recommended_prefill_rows(self, maximum, *, context_tokens=0):
        """Largest power-of-two prefill chunk whose score buffers fit the budget.

        The MLA RoPE score mask grows with rows x context, so a fixed chunk that
        is cheap at 4K is not at 128K. Never widens ``maximum``; without a
        budget the caller's chunk stands.
        """
        if type(maximum) is not int or maximum < 1:
            raise ValueError('prefill maximum must be a positive integer')
        if not self.prefill_score_budget_bytes:
            return maximum
        context = max(1, self._tokens(int(context_tokens)))
        act = self.prefill_activation_bytes or self.item_bytes
        fit = self.prefill_score_budget_bytes // (self.score_buffers * act * context * self.heads)
        rows = 1 << max(0, int(fit).bit_length() - 1) if fit >= 1 else 1
        return min(maximum, max(self.min_prefill_rows, rows))

    def as_dict(self):
        return dict(schema='ling-mla-kda-cache-v1', **asdict(self),
                    bound='fp32-compressed-MLA-plus-KDA-convolution-and-matrix-restore-state',
                    workspace='Existing generic forward allowance retained; MLA prefill copies, expansion and scores charged separately')
