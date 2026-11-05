# SPDX-License-Identifier: MIT
"""Kimi Linear NoPE MLA/KDA; provenance/kimi-linear-reap.json.

Reuses Ling's hybrid traversal, convolution, caches and weight stacking.
The checkpoint's normalization, unbounded gate and NoPE math stay explicit.
Performance candidates are per-model policies, never scheduler model branches.
"""
from collections import Counter
from dataclasses import dataclass, replace

import mlx.core as mx
from mlx import nn

from . import bailing_hybrid as ling
from . import moe_nax_gather
from .base import scaled_dot_product_attention
from .gated_delta import gated_delta_kernel, gated_delta_ops


@dataclass
class ModelArgs(ling.ModelArgs):
    linear_attn_config: dict | None = None
    mla_use_nope: bool = True
    moe_layer_freq: int = 1

    @classmethod
    def from_dict(cls, config):
        linear = config["linear_attn_config"]
        return super().from_dict({
            **config,
            "num_experts_per_tok": config["num_experts_per_token"],
            "n_group": config["num_expert_group"],
            "layer_group_size": 4,
            "max_position_embeddings": config["model_max_length"],
            "norm_topk_prob": config["moe_renormalize"],
            "score_function": config["moe_router_activation_func"],
            "short_conv_kernel_size": linear["short_conv_kernel_size"],
        })


def l2_qk(q, k):
    # FLA normalizes in fp32, epsilon on SUM, with no bf16 round before scan.
    q, k = q.astype(mx.float32), k.astype(mx.float32)
    return (q * mx.rsqrt(mx.sum(q * q, -1, keepdims=True) + 1e-6) * q.shape[-1]**-0.5,
            k * mx.rsqrt(mx.sum(k * k, -1, keepdims=True) + 1e-6))


class RMSNorm(nn.RMSNorm):
    """``law="reference"`` is KimiRMSNorm's two roundings: FP32 normalize,
    cast to the activation dtype, then the weight product in that dtype.
    ``"fused"`` (MLX convention) rounds once after the weight product."""

    law = "fused"

    def __call__(self, x):
        if self.law == "reference":
            return self.weight * mx.fast.rms_norm(x, None, self.eps)
        return super().__call__(x)


@mx.compile
def kda_gates(a, beta, a_log, dt_bias):
    shifted = a.astype(mx.float32) + dt_bias.astype(mx.float32)
    # Stable softplus; checkpoint FLA gate returns log-decay in fp32.
    softplus = mx.maximum(shifted, 0) + mx.log1p(mx.exp(-mx.abs(shifted)))
    # Log decay stays finite where exp() of it underflows to zero.
    return (-mx.exp(a_log.astype(mx.float32)) * softplus,
            mx.sigmoid(beta.astype(mx.float32)))


class KimiDeltaAttention(ling.KimiDeltaAttention):
    def __init__(self, args):
        linear = args.linear_attn_config
        super().__init__(replace(args, num_attention_heads=linear["num_heads"],
                                 head_dim=linear["head_dim"]))
        self.calls = Counter()
        self.reference = False
        self.fused_projections = False
        self.fused_norm = False
        self.chunk32 = False
        self.chunk_wy = False
        self.norm_rounding = "fla_fp32"

    def prepare_projections(self):
        # SGLang's qkv/beta/f_a/g_a GEMM and batched f_b/g_b mechanism.
        # Views are installed only after strict loading; never replace weights.
        names = ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj")
        object.__setattr__(self, "_joined", mx.concatenate([self[n].weight for n in names]))
        object.__setattr__(self, "_second", mx.stack([self.f_b_proj.weight, self.g_b_proj.weight]))
        object.__setattr__(self, "_conv", mx.concatenate([
            self[n].conv.weight for n in ("q_conv1d", "k_conv1d", "v_conv1d")]))

    def __call__(self, x, mask=None, cache=None, *, tree=None):
        if tree is not None:
            raise ValueError("Kimi Linear has no tree execution contract")
        b, t, _ = x.shape
        shape_kind = "multirow" if t > 1 else "single_row"
        self.calls[shape_kind+"_calls"] += 1
        self.calls[shape_kind+"_tokens"] += b*t
        h, d = self.num_heads, self.head_dim
        p = h * d
        if self.fused_projections:
            q, k, v, beta, fa, ga = mx.split(x @ self._joined.T,
                                            [p, 2*p, 3*p, 3*p+h, 3*p+h+d], -1)
            fg = mx.stack([fa, ga], axis=0) @ self._second[:, None].swapaxes(-1, -2)
            a, gate = fg[0], fg[1]
            self.calls["fused_projections"] += 1
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            beta = self.b_proj(x)
            a = self.f_b_proj(self.f_a_proj(x))
            gate = self.g_b_proj(self.g_a_proj(x))
        states = [None] * 4 if cache is None else list(cache)
        lengths = None if cache is None else cache.lengths
        if self.fused_projections and mask is None and lengths is None:
            # GLM/vLLM's merged depthwise convolution: each channel remains
            # independent, while the three launches become one.
            history = (mx.zeros((b,self.conv_kernel-1,3*p),dtype=x.dtype) if states[0] is None
                       else mx.concatenate(states[:3],axis=-1))
            conv_input = mx.concatenate([history,mx.concatenate([q,k,v],axis=-1)],axis=1)
            q,k,v = mx.split(nn.silu(mx.conv1d(conv_input,self._conv,groups=3*p)),3,axis=-1)
            states[:3] = mx.split(mx.contiguous(conv_input[:,-(self.conv_kernel-1):]),3,axis=-1)
            self.calls["merged_conv"] += 1
        else:
            q, states[0] = self.q_conv1d(q, states[0], mask, lengths)
            k, states[1] = self.k_conv1d(k, states[1], mask, lengths)
            v, states[2] = self.v_conv1d(v, states[2], mask, lengths)
        q, k, v = [z.reshape(b, t, h, d) for z in (q, k, v)]
        q, k = l2_qk(q, k)
        log_decay, beta = kda_gates(a.reshape(b, t, h, d), beta,
                                    self.A_log.reshape(h, 1), self.dt_bias.reshape(h, d))
        decay = mx.exp(log_decay)
        state = states[3]
        if state is None:
            state = mx.zeros((b, h, d, d), dtype=mx.float32)
        # Widen v: the existing vector Metal kernel shares one operand dtype.
        v = v.astype(mx.float32)
        chunked = t >= 17 and mask is None and not getattr(cache, "speculating", False)
        if self.chunk_wy and chunked:
            from .kda_chunk_wy import chunk_wy
            out, state = chunk_wy(q, k, v, log_decay, beta, state)
            self.calls["chunk_wy"] += 1
        elif self.chunk32 and chunked:
            from .gdn_chunk32 import chunk32
            out, state = chunk32(q, k, v, decay, beta, state)
            self.calls["chunk32"] += 1
        elif not self.reference and mx.default_device() == mx.gpu:
            out, state = gated_delta_kernel(q, k, v, decay, beta, state, mask)
            self.calls["vector_kernel"] += 1
        else:
            out, state = gated_delta_ops(q, k, v, decay, beta, state, mask)
            self.calls["reference"] += 1
        if cache is not None:
            for i in range(3):
                cache[i] = states[i]
            cache[3] = state
            cache.advance(t)
        # FLA writes recurrence output in the value dtype before gated norm.
        out = out.astype(x.dtype)
        gate = gate.reshape(b, t, h, d)
        if self.fused_norm and mx.default_device() == mx.gpu and out.dtype == mx.bfloat16:
            if self.norm_rounding == "qwen_bf16":
                from .qwen4_fused_gdn_prefill import qwen4_gdn_prefill_norm_gate
                out = qwen4_gdn_prefill_norm_gate(out,gate,self.o_norm.weight,self.o_norm.eps,
                                                 num_value_heads=h,value_head_dim=d)
            else:
                from .kimi_linear_fused import norm_gate
                out = norm_gate(out, gate, self.o_norm.weight, self.o_norm.eps)
            self.calls["fused_norm"] += 1
        else:
            # FLA RMSNormGated widens both norm and sigmoid before final cast.
            if self.norm_rounding == "qwen_bf16":
                normed = mx.fast.rms_norm(out,self.o_norm.weight,self.o_norm.eps).astype(mx.float32)
            else:
                normed = mx.fast.rms_norm(out.astype(mx.float32),self.o_norm.weight.astype(mx.float32),
                                         self.o_norm.eps)
            out = (normed * mx.sigmoid(gate.astype(mx.float32))).astype(x.dtype)
        self.calls["norm_"+self.norm_rounding] += 1
        return self.o_proj(out.reshape(b, t, p))


class MultiLatentAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        h, n, r, v, c = (args.num_attention_heads, args.qk_nope_head_dim,
                          args.qk_rope_head_dim, args.v_head_dim, args.kv_lora_rank)
        self.q_proj = nn.Linear(args.hidden_size, h * (n+r), bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(args.hidden_size, c+r, bias=False)
        # Released KimiMLAAttention constructs this norm at its default 1e-6.
        self.kv_a_layernorm = RMSNorm(c, eps=1e-6)
        self.kv_b_proj = nn.Linear(c, h * (n+v), bias=False)
        self.o_proj = nn.Linear(h * v, args.hidden_size, bias=False)
        self.scale = (n+r)**-0.5
        self.policy = "expanded"
        self.calls = Counter()
        self.key_tile = 2048
        # latent_single bound on the FP32 score block; larger shapes tile.
        self.single_score_bytes = 256 << 20
        # auto: prompt chunks wider than auto_rows use expanded SDPA while
        # its transient expanded prefix stays under expanded_bytes.
        self.auto_rows = 16
        self.expanded_bytes = 4 << 30

    def __call__(self, x, mask=None, cache=None, *, tree=None):
        if tree is not None:
            raise ValueError("Kimi Linear MLA has no tree contract")
        a = self.args
        b, t, _ = x.shape
        h, n, r, v, c = (a.num_attention_heads, a.qk_nope_head_dim,
                          a.qk_rope_head_dim, a.v_head_dim, a.kv_lora_rank)
        shape_kind = "multirow" if t > 1 else "single_row"
        self.calls[shape_kind+"_calls"] += 1
        self.calls[shape_kind+"_tokens"] += b*t
        q = self.q_proj(x).reshape(b, t, h, n+r).transpose(0, 2, 1, 3)
        latent, shared = mx.split(self.kv_a_proj_with_mqa(x), [c], -1)
        latent = self.kv_a_layernorm(latent)[:, None]
        shared = shared[:, None]
        # "rope" names in the config describe a shared key slice. No rotation.
        if cache is not None:
            latent, shared = cache.update_and_fetch(latent, shared)
        policy = self.policy
        if policy == "auto":
            # DeepSeek/Ling phase split: expanded K/V for prompt chunks while
            # the expanded prefix fits its bound, one latent per token for
            # decode/verify rows and for longer prefixes.
            expanded = t > self.auto_rows and latent.shape[2]*h*(n+r+v)*x.itemsize <= self.expanded_bytes
            policy = "expanded" if expanded else "latent_single"
            self.calls["auto_expanded" if expanded else "auto_latent"] += 1
        if policy in {"latent_tiled", "latent_fused", "latent_single"}:
            out = self._latent(q, latent, shared, mask, policy)
            if policy != "latent_single":
                self.calls["latent_tiled"] += 1
        else:
            kv = self.kv_b_proj(latent[:, 0]).reshape(b, -1, h, n+v).transpose(0, 2, 1, 3)
            keys, values = mx.split(kv, [n], -1)
            keys = mx.concatenate([keys, mx.broadcast_to(shared, (b, h, shared.shape[2], r))], -1)
            out = scaled_dot_product_attention(q, keys, values, cache=None, scale=self.scale, mask=mask)
            self.calls["expanded_sdpa"] += 1
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(b, t, h*v))

    def _latent(self, q, latent, shared, mask, policy):
        # GLM/DS phase split: absorb K projection, attend on one latent per
        # token, then unabsorb V. Bounded online softmax avoids O(context*rows)
        # score storage and full-prefix expanded K/V, including at 1M.
        a = self.args
        h, n, v = a.num_attention_heads, a.qk_nope_head_dim, a.v_head_dim
        w = self.kv_b_proj.weight.reshape(h, n+v, a.kv_lora_rank)
        qn, qs = mx.split(q, [n], -1)
        qn = qn @ w[:, :n]
        if policy == "latent_fused":
            from .kimi_linear_mla_fused import admits, attend
            if admits(qn,qs,latent,shared,mask):
                self.calls["fused_mla_decode"] += 1
                result = attend(qn,qs,latent,shared).astype(q.dtype)
                return result @ w[:,n:].swapaxes(-1,-2)
            self.calls["fused_mla_declined"] += 1
        if policy == "latent_single" and q.size // q.shape[-1] * latent.shape[2] * 4 <= self.single_score_bytes:
            # One pass over the latent cache: the same FP32 score/softmax law
            # as one key tile, without O(context/key_tile) launches per row.
            score = ((qn @ latent.swapaxes(-1, -2)).astype(mx.float32)
                     + (qs @ shared.swapaxes(-1, -2)).astype(mx.float32)) * self.scale
            if mask is not None:
                if isinstance(mask, str):
                    raise ValueError("latent MLA requires an explicit visibility mask")
                score = mx.where(mask, score, -float("inf"))
            # A fully masked (padding) row yields zeros, as in the tiled law.
            top = score.max(-1, keepdims=True)
            prob = mx.exp(score - mx.where(mx.isfinite(top), top, 0))
            total = mx.maximum(prob.sum(-1, keepdims=True), 1e-30)
            self.calls["latent_single"] += 1
            return ((prob @ latent.astype(mx.float32)) / total).astype(q.dtype) @ w[:, n:].swapaxes(-1, -2)
        if policy == "latent_single":
            # Over the score bound: same law in key tiles, with its own receipt.
            self.calls["latent_single_tiled"] += 1
        pieces = []
        for start in range(0, q.shape[2], 64):
            end = min(start+64, q.shape[2])
            qn_tile, qs_tile = qn[:, :, start:end], qs[:, :, start:end]
            maximum = mx.full((*qn_tile.shape[:-1], 1), -float("inf"), dtype=mx.float32)
            total = mx.zeros_like(maximum)
            accum = mx.zeros((*qn_tile.shape[:-1], a.kv_lora_rank), dtype=mx.float32)
            for k0 in range(0, latent.shape[2], self.key_tile):
                k1 = min(k0+self.key_tile, latent.shape[2])
                score = ((qn_tile @ latent[:, :, k0:k1].swapaxes(-1, -2)).astype(mx.float32)
                         + (qs_tile @ shared[:, :, k0:k1].swapaxes(-1, -2)).astype(mx.float32)) * self.scale
                if mask is not None:
                    if isinstance(mask, str):
                        raise ValueError("latent tiled MLA requires an explicit visibility mask")
                    score = mx.where(mask[..., start:end, k0:k1], score, -float("inf"))
                new_max = mx.maximum(maximum, score.max(-1, keepdims=True))
                safe_max = mx.where(mx.isfinite(new_max), new_max, 0)
                alpha = mx.exp(maximum-safe_max)
                prob = mx.exp(score-safe_max)
                accum = accum * alpha + prob @ latent[:, :, k0:k1].astype(mx.float32)
                total = total * alpha + prob.sum(-1, keepdims=True)
                maximum = new_max
            result = (accum / mx.maximum(total, 1e-30)).astype(q.dtype) @ w[:, n:].swapaxes(-1, -2)
            pieces.append(result)
        return mx.concatenate(pieces, axis=2)


class Gate(ling.Gate):
    def __init__(self, args):
        super().__init__(args)
        self.combine = "checkpoint"

    def __call__(self, x):
        # Source F.linear widens operands BEFORE GEMM, not just logits.
        logits = x.astype(mx.float32) @ self.gate_proj.weight.astype(mx.float32).T
        scores = mx.sigmoid(logits)
        choice = scores + self.expert_bias
        if self.n_group != 1:
            raise ValueError("staged Kimi router supports the checkpoint's single expert group")
        ids = mx.argpartition(-choice, kth=self.top_k-1, axis=-1)[..., :self.top_k]
        # modeling_kimi.py aliases scores.view() then += correction bias;
        # its gather therefore sees corrected scores. SGLang/vLLM use the
        # conventional uncorrected combine scores. Keep the discrepancy named.
        source = choice if self.combine == "checkpoint" else scores
        weights = mx.take_along_axis(source, ids, -1)
        if self.norm_topk_prob:
            weights = weights / (weights.sum(-1, keepdims=True) + 1e-20)
        return ids, weights * self.routed_scaling_factor


class SparseMoeBlock(ling.SparseMoeBlock):
    def __init__(self, args):
        super().__init__(args)
        self.gate = Gate(args)

    def __call__(self, x):
        ids, weights = self.gate(x)
        expert = self.switch_mlp(x, ids)
        out = (expert.astype(mx.float32) * weights[..., None]).sum(-2).astype(x.dtype)
        return out + self.shared_experts(x) if self.shared_experts is not None else out


class DecoderLayer(ling.DecoderLayer):
    def __init__(self, args, index):
        nn.Module.__init__(self)
        self.is_global = index+1 in args.linear_attn_config["full_attn_layers"]
        self.attention = MultiLatentAttention(args) if self.is_global else KimiDeltaAttention(args)
        self.mlp = (SparseMoeBlock(args) if index >= args.first_k_dense_replace
                    and index % args.moe_layer_freq == 0 else ling.MLP(args))
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)


class LanguageModel(ling.LanguageModel):
    def __init__(self, args):
        nn.Module.__init__(self)
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self._attention_idx = next(i for i, l in enumerate(self.layers) if l.is_global)
        self._kda_idx = next(i for i, l in enumerate(self.layers) if not l.is_global)


class Model(ling.Model):
    def __init__(self, args):
        nn.Module.__init__(self)
        self.args = args
        self.model_type = args.model_type
        self.model = LanguageModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        with moe_nax_gather.forward_scope(inputs, cache):
            return super().__call__(inputs, cache)

    def sanitize(self, weights):
        renamed = {}
        for key, value in weights.items():
            key = key.replace("model.embed_tokens.", "model.word_embeddings.")
            key = key.replace(".self_attn.", ".attention.").replace(".block_sparse_moe.", ".mlp.")
            key = key.replace(".gate.e_score_correction_bias", ".gate.expert_bias")
            for old, new in (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")):
                if ".experts." in key:
                    key = key.replace(f".{old}.", f".{new}.")
            if key.endswith(".A_log"):
                value = value.reshape(-1)
            renamed[key] = value
        # Ling's MLA transforms are intentionally bypassed: Kimi retains the
        # original kv_b projection for its ordinary-reference arithmetic.
        for i, layer in enumerate(self.layers):
            prefix = f"model.layers.{i}"
            if isinstance(layer.mlp, SparseMoeBlock):
                for name in ("gate_proj", "up_proj", "down_proj"):
                    key = f"{prefix}.mlp.experts.0.{name}.weight"
                    if key in renamed:
                        renamed[f"{prefix}.mlp.switch_mlp.{name}.weight"] = mx.stack([
                            renamed.pop(f"{prefix}.mlp.experts.{e}.{name}.weight")
                            for e in range(self.args.num_experts)])
                key = f"{prefix}.mlp.gate.weight"
                if key in renamed:
                    renamed[f"{prefix}.mlp.gate.gate_proj.weight"] = renamed.pop(key)
            if not layer.is_global:
                for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                    key = f"{prefix}.attention.{name}.weight"
                    if key in renamed:
                        renamed[f"{prefix}.attention.{name}.conv.weight"] = renamed.pop(key).moveaxis(2, 1)
        return renamed

    def configure(self, policy):
        for module in self.modules():
            if isinstance(module, RMSNorm):
                module.law = policy["rms_law"]
        for layer in self.layers:
            attn = layer.attention
            if layer.is_global:
                attn.policy = policy["mla"]
            else:
                attn.reference = policy["kda"] == "reference"
                attn.chunk32 = policy["kda"] == "chunk32"
                attn.chunk_wy = policy["kda"] == "chunk_wy"
                attn.fused_projections = policy["fused_projections"]
                attn.fused_norm = policy["fused_norm"]
                attn.norm_rounding = policy["norm_rounding"]
                if attn.fused_projections:
                    attn.prepare_projections()
                    mx.eval(attn._joined, attn._second, attn._conv)
            if isinstance(layer.mlp, SparseMoeBlock):
                layer.mlp.gate.combine = policy["router_combine"]

    def route_status(self):
        counters = Counter()
        for layer in self.layers:
            counters.update(layer.attention.calls)
        return dict(counters)

    def mechanism_status(self):
        kda,mla = Counter(),Counter()
        for layer in self.layers:
            (mla if layer.is_global else kda).update(layer.attention.calls)
        return {"kda":dict(kda),"mla":dict(mla)}
