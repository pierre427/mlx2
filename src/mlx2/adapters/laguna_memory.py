"""Laguna's full-attention and bounded-window cache admission geometry.

Derived from runtime.models.laguna.Model.make_cache and the shared KV caches.
The fp32 bound includes allocation padding and exact sliding-window restore
snapshots. It does not change tensor arithmetic or the host/driver reserves.
"""
from dataclasses import asdict, dataclass
import os


@dataclass(frozen=True)
class LagunaCacheBudget:
    global_layers: int
    sliding_layers: int
    kv_heads: int
    head_dim: int
    sliding_window: int
    hidden_size: int
    vocab_size: int
    checkpoint_copies: int
    item_bytes: int = 4
    allocation_step: int = 256
    transcript_bytes_per_token: int = 16
    # Preserve the previous generic forward-workspace allowance; this repair
    # replaces only the incorrectly generic persistent-cache estimate.
    transient_gib_per_lane: float = 1.76

    @classmethod
    def from_config(cls, config, *, mtp):
        if mtp:
            raise ValueError('Laguna has no native MTP cache topology')
        names = ('num_hidden_layers', 'num_key_value_heads', 'head_dim',
                 'sliding_window', 'hidden_size', 'vocab_size')
        if any(type(config.get(k)) is not int or config[k] <= 0 for k in names):
            raise ValueError('invalid Laguna cache dimensions')
        layers = config.get('layer_types')
        if (not isinstance(layers, list) or len(layers) != config['num_hidden_layers']
                or set(layers) - {'full_attention', 'sliding_attention'}):
            raise ValueError('unknown Laguna cache topology')
        try:
            checkpoints = max(0, int(os.environ.get('MLX_LM_STATE_CHECKPOINT_MAX', '4')))
        except ValueError:
            checkpoints = 4
        return cls(layers.count('full_attention'), layers.count('sliding_attention'),
                   config['num_key_value_heads'], config['head_dim'],
                   config['sliding_window'], config['hidden_size'],
                   config['vocab_size'], checkpoints)

    @staticmethod
    def _tokens(value):
        if type(value) is not int or value < 0:
            raise ValueError('cache dimensions must be nonnegative integers')
        return value

    @property
    def kv_bytes_per_token(self):
        return 2 * self.kv_heads * self.head_dim * self.item_bytes

    def project(self, context_tokens):
        n = self._tokens(context_tokens)
        capacity = ((n + self.allocation_step - 1) // self.allocation_step + 1) * self.allocation_step
        return (self.kv_bytes_per_token * (
            self.global_layers * capacity
            + self.sliding_layers * min(capacity, self.sliding_window) * (1 + self.checkpoint_copies))
            + capacity * self.transcript_bytes_per_token
            + (self.global_layers + self.sliding_layers) * 4096)

    def prefill_transient_bytes(self, context_tokens, chunk_rows):
        context = self._tokens(context_tokens)
        rows = self._tokens(chunk_rows)
        # KV concatenation can retain the previous global buffers and append
        # a whole chunk to rotating windows before compaction. Forward
        # activations/logits are separate from those persistent arrays.
        return (self.kv_bytes_per_token * (
            self.global_layers * context
            + (self.global_layers + self.sliding_layers) * rows)
            + rows * (8 * self.hidden_size + self.vocab_size) * self.item_bytes)

    def as_dict(self):
        return dict(schema='laguna-full-and-sliding-cache-v1', **asdict(self),
                    bound='fp32-global-KV-plus-window-and-exact-restore-snapshots',
                    workspace='Existing generic forward allowance retained; prefill copies charged separately')
