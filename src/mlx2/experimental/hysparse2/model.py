"""Original trainable MLX implementation of the published HySparse2 mechanisms.

This is a research reference, not a reconstruction of undisclosed Xiaomi weights
or training details and not registered as a qualified mlx2 serving adapter.
"""

import math
from dataclasses import dataclass, field

import mlx.core as mx
from mlx import nn
from mlx.nn.utils import checkpoint

from .attention import attention, sparse_attention
from .config import Config


class IdentityHyperConnection(nn.Module):
    """Dynamic pre/post mixing, with the residual-stream matrix exactly I.

    Four streams by default. Dynamic parameterization and initialization are
    explicit research choices; the HySparse2 paper specifies identity residual
    mixing but does not disclose the complete simplified-mHC implementation.
    """

    def __init__(self, c):
        super().__init__()
        self.streams = c.residual_streams
        self.eps = c.norm_eps
        self.mix = nn.Linear(
            c.residual_streams * c.hidden_size, 2 * c.residual_streams, bias=False
        )
        self.scales = mx.full((2,), 0.01)
        pre = -math.log(max(c.residual_streams - 1, 1))
        self.bias = mx.concatenate(
            (mx.full((c.residual_streams,), pre), mx.zeros((c.residual_streams,)))
        )

    def read(self, x):
        flat = x.reshape(*x.shape[:-2], -1)
        flat = flat * mx.rsqrt(
            mx.mean(mx.square(flat.astype(mx.float32)), axis=-1, keepdims=True)
            + self.eps
        ).astype(x.dtype)
        dynamic = self.mix(flat).reshape(*flat.shape[:-1], 2, self.streams)
        mixing = dynamic * self.scales[:, None] + self.bias.reshape(2, self.streams)
        pre, post = mx.sigmoid(mixing[..., 0, :]), 2 * mx.sigmoid(mixing[..., 1, :])
        return mx.sum(x * pre[..., None], axis=-2), post

    def write(self, x, update, post):
        return x + post[..., None] * update[..., None, :]


class MoE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.router = nn.Linear(c.hidden_size, c.num_experts, bias=False)
        # Gathered products execute selected experts, not all E dense outputs.
        scale = c.hidden_size**-0.5
        self.gate = (
            mx.random.normal((c.num_experts, c.expert_dim, c.hidden_size)) * scale
        )
        self.up = mx.random.normal((c.num_experts, c.expert_dim, c.hidden_size)) * scale
        self.down = (
            mx.random.normal((c.num_experts, c.hidden_size, c.expert_dim))
            * c.expert_dim**-0.5
        )
        self.shared = [
            MLP(c.hidden_size, c.expert_dim) for _ in range(c.shared_experts)
        ]

    def __call__(self, x):
        p = mx.softmax(self.router(x).astype(mx.float32), axis=-1)
        idx = mx.stop_gradient(
            mx.argpartition(
                p, kth=self.c.num_experts - self.c.experts_per_token, axis=-1
            )[..., -self.c.experts_per_token :]
        )
        weights = mx.take_along_axis(p, idx, axis=-1)
        weights = weights / mx.sum(weights, axis=-1, keepdims=True)
        expanded = x[..., None, None, :]
        gate = mx.gather_mm(expanded, self.gate.swapaxes(-1, -2), rhs_indices=idx)
        up = mx.gather_mm(expanded, self.up.swapaxes(-1, -2), rhs_indices=idx)
        hidden = nn.silu(gate) * up
        values = mx.gather_mm(
            hidden, self.down.swapaxes(-1, -2), rhs_indices=idx
        ).squeeze(-2)
        out = mx.sum(values * weights[..., None].astype(x.dtype), axis=-2)
        for expert in self.shared:
            out = out + expert(x)
        # Conventional differentiable load-balance auxiliary objective; explicit
        # choice, not a claimed reproduction of the undisclosed router recipe.
        frequency = mx.mean(
            (idx[..., None] == mx.arange(self.c.num_experts)).astype(mx.float32),
            axis=(0, 1, 2),
        )
        auxiliary = self.c.num_experts * mx.sum(
            mx.stop_gradient(frequency) * mx.mean(p, axis=(0, 1))
        )
        return out, auxiliary


class MLP(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.gate = nn.Linear(d, m, bias=False)
        self.up = nn.Linear(d, m, bias=False)
        self.down = nn.Linear(m, d, bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class PermanentKnowledgePLE(nn.Module):
    """Trainable n-gram knowledge sidecar kept with the model checkpoint.

    Semantic capsules remain the authoritative, revisioned memory. This PLE
    table learns a bounded neural prior for recurring token sequences during
    ordinary training and never mutates from an inference request.
    """

    def __init__(self, c):
        super().__init__()
        self.rows = c.semantic_ple_rows
        self.order = c.semantic_ngram
        self.embedding = nn.Embedding(c.semantic_ple_rows, c.semantic_ple_dim)
        self.key = nn.Linear(c.semantic_ple_dim, c.hidden_size, bias=False)
        self.value = nn.Linear(c.semantic_ple_dim, c.hidden_size, bias=False)
        self.gate_bias = mx.zeros((c.hidden_size,))

    def indices(self, tokens, history=None):
        values = tokens.astype(mx.int64)
        if history is not None:
            values = mx.concatenate((history.astype(mx.int64), values), axis=1)
        padding = mx.zeros((values.shape[0], self.order - 1), dtype=mx.int64)
        padded = mx.concatenate((padding, values), axis=1)
        start = self.order - 1
        hashed = padded[:, start : start + values.shape[1]] % self.rows
        for shift in range(1, self.order):
            previous = padded[:, start - shift : start - shift + values.shape[1]]
            hashed = (hashed * 1000003 + previous * (97 + shift * 2)) % self.rows
        return hashed[:, -tokens.shape[1] :]

    def __call__(self, tokens, query, history=None):
        row = self.embedding(self.indices(tokens, history))
        key = self.key(row)
        value = self.value(row)
        query = query * mx.rsqrt(
            mx.mean(mx.square(query.astype(mx.float32)), axis=-1, keepdims=True) + 1e-6
        ).astype(query.dtype)
        gate = mx.sigmoid(
            (query * key).astype(mx.float32) * query.shape[-1] ** -0.5 + self.gate_bias
        ).astype(value.dtype)
        return gate * value


class DiffusionLayer(nn.Module):
    """Bidirectional denoising layer; all positions refine in parallel."""

    def __init__(self, c):
        super().__init__()
        d = c.hidden_size
        self.heads = c.num_heads
        if d % self.heads:
            raise ValueError("diffusion hidden size must divide by query heads")
        self.head_dim = d // self.heads
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.norm1 = nn.RMSNorm(d, eps=c.norm_eps)
        self.norm2 = nn.RMSNorm(d, eps=c.norm_eps)
        self.mlp = MLP(d, c.diffusion_width_multiplier * d)
        self.time_scale = mx.zeros((d,))

    def __call__(self, x, noise_level):
        h = self.norm1(x) + noise_level * self.time_scale
        shape = (*h.shape[:-1], self.heads, self.head_dim)
        q = self.q(h).reshape(shape).transpose(0, 2, 1, 3)
        k = self.k(h).reshape(shape).transpose(0, 2, 1, 3)
        v = self.v(h).reshape(shape).transpose(0, 2, 1, 3)
        # The student is bidirectional, so the fused SDPA kernel needs no mask.
        # Keeping the probability matrix implicit avoids materializing the
        # B,H,T,T tensor during long-sequence diffusion training.
        attended = (
            mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5)
            .transpose(0, 2, 1, 3)
            .reshape(x.shape)
        )
        x = x + self.out(attended)
        return x + self.mlp(self.norm2(x))


class DiffusionStudent(nn.Module):
    """Masked-diffusion auxiliary student with a 2x-standard (8d) MLP."""

    def __init__(self, c):
        super().__init__()
        self.layers = [DiffusionLayer(c) for _ in range(c.diffusion_layers)]
        self.condition = nn.Linear(c.hidden_size, c.hidden_size, bias=False)
        self.norm = nn.RMSNorm(c.hidden_size, eps=c.norm_eps)

        self.trunk_gradient_scale = c.diffusion_trunk_gradient_scale

    def __call__(self, tokens, embedding, teacher):
        positions = mx.arange(tokens.shape[1], dtype=mx.int32)[None]
        seed = tokens[:, :1].astype(mx.int32) % 16
        phase = tokens[:, :1].astype(mx.int32) % 4
        threshold = mx.take(mx.array([2, 4, 8, 12]), phase)
        score = (positions * 13 + seed * 7) % 16
        mask = (score < threshold) | (positions == 0)
        masked = mx.where(mask, mx.ones_like(tokens), tokens)
        # Preserve the same conditioning values while controlling how much of
        # the denoising gradient trains the causal teacher. Zero preserves
        # historical detached checkpoints; one couples the full objective.
        scale = self.trunk_gradient_scale
        conditioned = teacher if scale == 1 else mx.stop_gradient(teacher)
        if 0 < scale < 1:
            conditioned = conditioned + scale * (teacher - mx.stop_gradient(teacher))
        x = embedding(masked) + self.condition(conditioned)
        noise_level = (threshold.astype(x.dtype) / 16)[..., None]
        for layer in self.layers:
            x = layer(x, noise_level)
        return embedding.as_linear(self.norm(x)), mask


class Attention(nn.Module):
    def __init__(self, c, kind):
        super().__init__()
        self.c, self.kind = c, kind
        self.q = nn.Linear(c.hidden_size, c.num_heads * c.head_dim, bias=False)
        self.out = nn.Linear(c.num_heads * c.head_dim, c.hidden_size, bias=False)
        if kind != "sparse":
            self.k = nn.Linear(c.hidden_size, c.head_dim, bias=False)
            self.v = nn.Linear(c.hidden_size, c.head_dim, bias=False)
        if kind == "cross":
            self.bridge_norm = nn.RMSNorm(c.hidden_size, eps=c.norm_eps)
        if kind in ("swa", "sparse"):
            self.gate = nn.Linear(c.hidden_size, c.num_heads, bias=False)
            self.sinks = mx.zeros((c.num_heads,))

    def project_kv(self, x, offset):
        if self.kind == "cross":
            x = self.bridge_norm(x)
        k, v = self.k(x)[:, None], self.v(x)[:, None]
        if self.kind == "swa" and self.c.rope_dims:
            k = mx.fast.rope(
                k,
                self.c.rope_dims,
                traditional=False,
                base=self.c.rope_base,
                scale=1.0,
                offset=offset,
            )
        return k, v, offset

    def __call__(self, x, blocks, offset, selected=None):
        c = self.c
        q = (
            self.q(x)
            .reshape(*x.shape[:-1], c.num_heads, c.head_dim)
            .transpose(0, 2, 1, 3)
        )
        if self.kind == "swa" and c.rope_dims:
            q = mx.fast.rope(
                q,
                c.rope_dims,
                traditional=False,
                base=c.rope_base,
                scale=1.0,
                offset=offset,
            )
        if self.kind == "sparse":
            out = sparse_attention(q, selected, offset=offset, sinks=self.sinks)
        else:
            out, selected = attention(
                q,
                blocks,
                offset=offset,
                query_tile=c.query_tile,
                key_tile=c.key_tile,
                window=c.local_window if self.kind == "swa" else None,
                sinks=self.sinks if self.kind == "swa" else None,
                select=(c.local_window, c.global_tokens)
                if self.kind == "cross"
                else None,
                block_select=(c.candidate_block_size, c.candidate_blocks)
                if self.kind == "cross" and c.candidate_blocks
                else None,
            )
        out = out.transpose(0, 2, 1, 3)
        if self.kind in ("swa", "sparse"):
            out = out * mx.sigmoid(self.gate(x))[..., None]
        return self.out(out.reshape(*x.shape[:-1], -1)), selected


class Layer(nn.Module):
    def __init__(self, c, kind):
        super().__init__()
        self.kind = kind
        self.attention = Attention(c, kind)
        self.attention_norm = nn.RMSNorm(c.hidden_size, eps=c.norm_eps)
        self.ffn_norm = nn.RMSNorm(c.hidden_size, eps=c.norm_eps)
        self.attention_hc = IdentityHyperConnection(c)
        self.ffn_hc = IdentityHyperConnection(c)
        self.moe = MoE(c)

    def __call__(self, x, blocks, offset, selected=None):
        source, post = self.attention_hc.read(x)
        normalized = self.attention_norm(source)
        if blocks is None and self.kind in ("swa", "full"):
            blocks = [self.attention.project_kv(normalized, offset)]
        out, selected = self.attention(normalized, blocks, offset, selected)
        x = self.attention_hc.write(x, out, post)
        y, post = self.ffn_hc.read(x)
        y, aux = self.moe(self.ffn_norm(y))
        return self.ffn_hc.write(x, y, post), selected, source, aux


class MTP(nn.Module):
    """One boundary-conditioned next-next-token training head with tied output."""

    def __init__(self, c):
        super().__init__()
        self.hidden_norm = nn.RMSNorm(c.hidden_size, eps=c.norm_eps)
        self.token_norm = nn.RMSNorm(c.hidden_size, eps=c.norm_eps)
        self.projection = nn.Linear(2 * c.hidden_size, c.hidden_size, bias=False)
        self.attention = Attention(c, "swa")
        self.norm = nn.RMSNorm(c.hidden_size, eps=c.norm_eps)
        self.mlp = MLP(c.hidden_size, 4 * c.hidden_size)

    def __call__(self, boundary, next_embedding):
        x = self.projection(
            mx.concatenate(
                (self.hidden_norm(boundary), self.token_norm(next_embedding)), axis=-1
            )
        )
        x = x + self.attention(x, [self.attention.project_kv(x, 0)], 0)[0]
        return x + self.mlp(self.norm(x))


@dataclass
class Cache:
    """Request-private inference state; never used in differentiable training."""

    owner: object
    batch: int
    length: int = 0
    self_kv: dict = field(default_factory=dict)
    cross_kv: dict = field(default_factory=dict)
    boundary: object = None
    self_layer_calls: int = 0
    cross_layer_calls: int = 0
    ple_history: object = None
    apcv2_identity: object = None

    def arrays(self):
        return [
            a
            for group in (self.self_kv, self.cross_kv)
            for blocks in group.values()
            for block in blocks
            for a in block[:2]
        ]

    def resident_bytes(self):
        """KV tensor bytes only; excludes boundary activations and allocator overhead."""
        return sum(a.nbytes for a in self.arrays())


class Model(nn.Module):
    def __init__(self, c=None):
        super().__init__()
        c = Config() if c is None else c
        self.config = c
        self.embedding = nn.Embedding(c.vocab_size, c.hidden_size)
        if c.semantic_ple_rows:
            self.semantic_ple = PermanentKnowledgePLE(c)
        self.self_decoder = [
            Layer(c, "full" if i == c.self_full_layer else "swa")
            for i in range(c.self_layers)
        ]
        self.cross_decoder = [
            Layer(c, "cross" if i % (1 + c.sparse_per_block) == 0 else "sparse")
            for i in range(c.cross_blocks * (1 + c.sparse_per_block))
        ]
        self.norm = nn.RMSNorm(c.hidden_size, eps=c.norm_eps)
        if c.mtp:
            self.mtp_head = MTP(c)
        if c.diffusion_layers:
            self.diffusion_student = DiffusionStudent(c)
        self.checkpoint_layers = False
        # Set by exact checkpoint loading. An unsaved research initialization
        # remains in an explicit, non-colliding APCv2 namespace.
        self.ple_sidecar_digest = None
        self._cache_owner = object()
        self.adapter_revision = None

    def _embed(self, tokens, ple_history=None):
        if (
            tokens.ndim != 2
            or tokens.shape[0] < 1
            or not mx.issubdtype(tokens.dtype, mx.integer)
            or not 0 < tokens.shape[1] <= self.config.max_context
        ):
            raise ValueError("tokens must be a nonempty B,T array within max_context")
        x = self.embedding(tokens)
        if self.config.semantic_ple_rows:
            x = x + self.semantic_ple(tokens, x, ple_history)
        return mx.broadcast_to(
            x[..., None, :], (*x.shape[:-1], self.config.residual_streams, x.shape[-1])
        )

    def _call(self, layer, *args):
        return (
            checkpoint(layer)(*args)
            if self.checkpoint_layers and self.training
            else layer(*args)
        )

    def __call__(self, tokens, *, next_tokens=None):
        """Full teacher-forced training: all layers, no prefill early exit."""
        x = self._embed(tokens)
        aux = mx.array(0.0)
        source = None
        for i, layer in enumerate(self.self_decoder):
            x, _, raw, loss = self._call(layer, x, None, 0, None)
            aux = aux + loss
            if i == self.config.self_full_layer:
                source = raw
        boundary = mx.mean(x, axis=-2)
        selected = None
        for layer in self.cross_decoder:
            blocks = (
                [layer.attention.project_kv(source, 0)]
                if layer.kind == "cross"
                else None
            )
            x, selected, _, loss = self._call(layer, x, blocks, 0, selected)
            aux = aux + loss
        logits = self.embedding.as_linear(self.norm(mx.mean(x, axis=-2)))
        mtp_logits = None
        if next_tokens is not None:
            if not self.config.mtp or next_tokens.shape != tokens.shape:
                raise ValueError("MTP requires enabled head and aligned next_tokens")
            mtp_logits = self.embedding.as_linear(
                self.norm(self.mtp_head(boundary, self.embedding(next_tokens)))
            )
        diffusion = None
        if self.config.diffusion_layers and self.training:
            diffusion = self.diffusion_student(
                tokens, self.embedding, self.norm(mx.mean(x, axis=-2))
            )
        return logits, aux / self.config.layers, mtp_logits, diffusion

    def new_cache(self, batch=1, *, semantic_capsule_digest=None):
        if type(batch) is not int or batch < 1:
            raise ValueError("cache batch must be positive")
        return Cache(
            self._cache_owner,
            batch,
            apcv2_identity={
                **self.config.apcv2_identity(semantic_capsule_digest, self.ple_sidecar_digest),
                **({"adapter_revision": self.adapter_revision} if self.adapter_revision else {}),
            },
        )

    def _append(self, tokens, cache):
        if self.training:
            raise ValueError("cached inference requires model.eval()")
        if cache.owner is not self._cache_owner or cache.batch != tokens.shape[0]:
            raise ValueError("cache belongs to another model or batch")
        expected = self.config.apcv2_identity(
            ple_sidecar_digest=self.ple_sidecar_digest
        )
        actual = cache.apcv2_identity
        if (
            not isinstance(actual, dict)
            or actual.get("adapter_revision") != self.adapter_revision
            or actual.get("cache_layout_fingerprint")
            != expected["cache_layout_fingerprint"]
            or actual.get("semantic_fingerprint", ())[2:]
            != expected["semantic_fingerprint"][2:]
        ):
            raise ValueError("cache geometry, attention math or PLE revision differs")
        offset = cache.length
        if offset + tokens.shape[1] > self.config.max_context:
            raise ValueError("context limit exceeded")
        x = self._embed(tokens, cache.ple_history)
        if self.config.semantic_ple_rows:
            history = (
                tokens
                if cache.ple_history is None
                else mx.concatenate((cache.ple_history, tokens), axis=1)
            )
            cache.ple_history = (
                None
                if self.config.semantic_ngram == 1
                else mx.contiguous(history[:, -(self.config.semantic_ngram - 1) :])
            )
        source = None
        fresh = []
        for i, layer in enumerate(self.self_decoder):
            raw, _ = layer.attention_hc.read(x)
            current = layer.attention.project_kv(layer.attention_norm(raw), offset)
            blocks = cache.self_kv.get(i, []) + [current]
            x, _, raw, _ = layer(x, blocks, offset)
            cache.self_layer_calls += 1
            if layer.kind == "swa":
                k = mx.concatenate([b[0] for b in blocks], axis=2)[
                    :, :, -self.config.local_window :
                ]
                v = mx.concatenate([b[1] for b in blocks], axis=2)[
                    :, :, -self.config.local_window :
                ]
                cache.self_kv[i] = [
                    (
                        mx.contiguous(k),
                        mx.contiguous(v),
                        offset + tokens.shape[1] - k.shape[2],
                    )
                ]
            else:
                cache.self_kv[i] = blocks
                source = raw
            fresh.extend(cache.self_kv[i][-1][:2])
        for i, layer in enumerate(self.cross_decoder):
            if layer.kind == "cross":
                block = layer.attention.project_kv(source, offset)
                cache.cross_kv.setdefault(i, []).append(block)
                fresh.extend(block[:2])
        cache.length += tokens.shape[1]
        cache.boundary = mx.contiguous(x[:, -1:])
        mx.eval(fresh, x)
        return x, offset

    def _cross(self, x, cache, offset):
        selected = None
        for i, layer in enumerate(self.cross_decoder):
            x, selected, _, _ = layer(x, cache.cross_kv.get(i), offset, selected)
            cache.cross_layer_calls += 1
        return self.embedding.as_linear(self.norm(mx.mean(x, axis=-2)))

    def prefill(self, tokens, cache=None, *, return_logits=True):
        """Build caches through self decoder; run cross only for the last logit."""
        if tokens.ndim != 2 or tokens.shape[1] < 1:
            raise ValueError("nonempty B,T tokens required")
        cache = self.new_cache(tokens.shape[0]) if cache is None else cache
        if cache.length + tokens.shape[1] > self.config.max_context:
            raise ValueError("context limit exceeded")
        for i in range(0, tokens.shape[1], self.config.prefill_chunk):
            self._append(tokens[:, i : i + self.config.prefill_chunk], cache)
        logits = (
            self._cross(cache.boundary, cache, cache.length - 1)
            if return_logits
            else None
        )
        if logits is not None:
            mx.eval(logits)
        return logits, cache

    def decode(self, tokens, cache):
        if tokens.ndim != 2 or tokens.shape[1] != 1:
            raise ValueError("decode consumes one new token per row")
        x, offset = self._append(tokens, cache)
        logits = self._cross(x, cache, offset)
        mx.eval(logits)
        return logits
