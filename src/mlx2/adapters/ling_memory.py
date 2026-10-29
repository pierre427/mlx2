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

    @classmethod
    def from_config(cls, config, *, mtp):
        if mtp:
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
        # mask. Charge three fp32 score buffers plus expansion; layers run in
        # sequence, while their persistent compressed caches are in project().
        cache_copy = self.global_layers * context * (self.kv_lora_rank + self.rope_dim)
        expanded = context * self.heads * (self.nope_dim + self.value_dim)
        scores = 3 * rows * context * self.heads
        activations = rows * (16 * self.heads * self.head_dim + 8 * self.hidden_size + self.vocab_size)
        return (cache_copy + expanded + scores + activations) * self.item_bytes

    def as_dict(self):
        return dict(schema='ling-mla-kda-cache-v1', **asdict(self),
                    bound='fp32-compressed-MLA-plus-KDA-convolution-and-matrix-restore-state',
                    workspace='Existing generic forward allowance retained; MLA prefill copies, expansion and scores charged separately')
