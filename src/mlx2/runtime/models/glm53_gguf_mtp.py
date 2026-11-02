"""Original exact-packed NextN head; ggml source graph recorded in provenance.

Seed is the trunk's post-head-norm HC mean, not its raw final layer tap.
The head has ordinary residuals and its own latent/raw-indexer state.
"""

import mlx.core as mx
from mlx import nn

from .glm53_gguf import rms


class Glm53MTPHead(nn.Module):
    def __init__(self, target):
        super().__init__()
        # This head owns no second packed bank. Avoid a module reference cycle.
        object.__setattr__(self, "_target", target)
        self.layers = [None]

    def __call__(self, hidden, tokens, cache):
        m = self._target
        p = "blk.45"
        w = m.weights
        eps = m.args.rms_norm_eps
        if (
            tokens.ndim != 2
            or tokens.shape[0] != 1
            or hidden.shape != (*tokens.shape, 4096)
            or len(cache) != 1
        ):
            raise ValueError(
                "GLM-5.3 MTP requires paired B1 seed/token rows and one head cache"
            )
        e = rms(
            w.embedding(tokens.astype(mx.uint32)),
            w.array(p + ".nextn.enorm.weight"),
            eps,
        )
        h = rms(hidden, w.array(p + ".nextn.hnorm.weight"), eps)
        x = w.linear(p + ".nextn.eh_proj.weight")(mx.concatenate((e, h), axis=-1))
        x = x + m._mla(rms(x, w.array(p + ".attn_norm.weight"), eps), cache[0], p)
        x = x + m._mlp(rms(x, w.array(p + ".ffn_norm.weight"), eps), p, 45)
        seed = rms(x, w.array(p + ".nextn.shared_head_norm.weight"), eps)
        return w.linear("output.weight")(seed), seed
